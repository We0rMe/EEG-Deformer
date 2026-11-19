"""PyTorch 实现 EEG-Deformer，结构与运算顺序以作者公开实现为准。"""

import torch
from torch import nn

import config


class MaxNormConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    @torch.no_grad()
    def project_weight(self):
        self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)

    def forward(self, x):
        # 保留本次前向的权重值，后续前向的约束操作不影响这次反向。
        return self._conv_forward(x, self.weight.clone(), self.bias)


class KernelAttention(nn.Module):
    def __init__(self, width, heads, head_dim, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = heads * head_dim
        self.qkv = nn.Linear(width, 3 * inner_dim, bias=False)
        if heads == 1 and head_dim == width:
            self.output = nn.Identity()
        else:
            self.output = nn.Sequential(nn.Linear(inner_dim, width), nn.Dropout(dropout))

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        # 每个卷积核对应一个 token，池化后的时间轴是 token 的特征维。
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = (query @ key.transpose(-2, -1)) * self.scale
        attended = scores.softmax(dim=-1) @ value
        return self.output(attended.transpose(1, 2).flatten(start_dim=2))


class HCTBlock(nn.Module):
    def __init__(self, width, num_filters, temporal_kernel, heads, head_dim, ff_dim, dropout):
        super().__init__()
        self.coarse_pool = nn.MaxPool1d(2, stride=2)
        self.attention = KernelAttention(width, heads, head_dim, dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, width),
            nn.Dropout(dropout),
        )
        self.fine_branch = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(num_filters, num_filters, temporal_kernel, padding=temporal_kernel // 2),
            nn.BatchNorm1d(num_filters),
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),
        )

    def forward(self, x):
        coarse = self.coarse_pool(x)
        coarse = self.attention(coarse) + coarse
        fine = self.fine_branch(x)
        return self.feed_forward(coarse) + fine, fine


def log_mean_power(features):
    # 在半精度下用 float32 计算功率，避免平方时上溢或下溢。
    if features.dtype in (torch.float16, torch.bfloat16):
        features = features.float()
    power = features.square().mean(dim=-1)
    return power.clamp_min(torch.finfo(power.dtype).tiny).log()


class EEGDeformer(nn.Module):
    """输入 [批大小, 通道, 采样点]，输出 [批大小, 类别数] 的 logits。"""

    def __init__(
        self,
        num_samples,
        *,
        num_channels=len(config.CHANNELS),
        num_classes=config.NUM_CLASSES,
        depth=config.HCT_DEPTH,
        temporal_kernel=config.TEMPORAL_KERNEL,
        num_filters=config.NUM_FILTERS,
        heads=config.ATTENTION_HEADS,
        head_dim=config.ATTENTION_HEAD_DIM,
        ff_dim=config.FEEDFORWARD_DIM,
        dropout=config.MODEL_DROPOUT,
        max_norm=config.ENCODER_MAX_NORM,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("HCT 块数至少为 1")
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("时间卷积核长度须为正奇数")

        self.input_shape = (num_channels, num_samples)
        encoder_width = num_samples // 2
        widths = []
        width = encoder_width
        for _ in range(depth):
            width //= 2
            if width < 1:
                raise ValueError(f"{num_samples} 点输入无法经过浅层编码器和 {depth} 个 HCT 块的池化")
            widths.append(width)

        self.encoder = nn.Sequential(
            MaxNormConv2d(
                1, num_filters, (1, temporal_kernel),
                padding=(0, temporal_kernel // 2), max_norm=max_norm,
            ),
            MaxNormConv2d(num_filters, num_filters, (num_channels, 1), max_norm=max_norm),
            nn.BatchNorm2d(num_filters),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
        )
        self.position = nn.Parameter(torch.randn(1, num_filters, encoder_width))
        self.hct_blocks = nn.ModuleList(
            HCTBlock(width, num_filters, temporal_kernel, heads, head_dim, ff_dim, dropout)
            for width in widths
        )
        self.classifier = nn.Linear(num_filters * (widths[-1] + depth), num_classes)

    def project_encoder_weights(self):
        self.encoder[0].project_weight()
        self.encoder[1].project_weight()

    def forward(self, eeg, *, project_weights=True):
        if eeg.ndim != 3 or eeg.shape[1:] != self.input_shape:
            raise ValueError(f"预期输入为 [批大小, {self.input_shape[0]}, {self.input_shape[1]}]，收到 {tuple(eeg.shape)}")

        if project_weights:
            self.project_encoder_weights()
        encoded = self.encoder(eeg.unsqueeze(1)).squeeze(2)
        features = (encoded + self.position).to(dtype=encoded.dtype)
        powers = []
        for block in self.hct_blocks:
            features, fine = block(features)
            powers.append(log_mean_power(fine))

        # DIP 拼接末层主干特征和各层细粒度分支的对数平均功率。
        embedding = torch.cat([features.flatten(start_dim=1), *powers], dim=-1)
        return self.classifier(embedding)


class EEGDeformerDataParallel(nn.DataParallel):
    def forward(self, eeg):
        # 先约束主模型参数，避免在副本中原地修改 Broadcast 的视图。
        self.module.project_encoder_weights()
        return super().forward(eeg, project_weights=False)
