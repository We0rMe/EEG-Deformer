from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.functional import cross_entropy

import config
from model import EEGDeformer, EEGDeformerDataParallel, log_mean_power


OUTPUT_DIR = config.PROJECT_DIR / "smoke_test" / "output"
BATCH_SIZE = 4


def check_batch(eeg, labels):
    num_samples = eeg.shape[-1]
    model = EEGDeformer(num_samples)
    branch_widths = []

    def record_widths(_module, _inputs, outputs):
        fused, fine = outputs
        branch_widths.append((fused.shape[-1], fine.shape[-1]))

    hooks = [block.register_forward_hook(record_widths) for block in model.hct_blocks]
    logits = model(eeg)
    for hook in hooks:
        hook.remove()

    widths = [num_samples // 2 ** (index + 2) for index in range(config.HCT_DEPTH)]
    assert branch_widths == [(width, width) for width in widths]
    assert model.position.shape == (1, config.NUM_FILTERS, num_samples // 2)
    assert model.classifier.in_features == config.NUM_FILTERS * (widths[-1] + config.HCT_DEPTH)
    assert logits.shape == (BATCH_SIZE, config.NUM_CLASSES) and torch.isfinite(logits).all()

    loss = cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    for name, weight in model.named_parameters():
        assert weight.grad is not None and torch.isfinite(weight.grad).all(), name

    model.eval()
    with torch.inference_mode():
        prediction = model(eeg[:1])
    assert prediction.shape == (1, config.NUM_CLASSES) and torch.isfinite(prediction).all()

    length_trace = " → ".join(map(str, [num_samples, num_samples // 2, *widths]))
    return f"{num_samples} 点输入，时间维 {length_trace}，logits {tuple(logits.shape)}"


def check_log_power():
    zeros = torch.zeros(2, 3, 7, requires_grad=True)
    output = log_mean_power(zeros)
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(zeros.grad).all()

    small = torch.full((2, 3, 7), 1e-4, dtype=torch.float16)
    expected = small.float().square().mean(dim=-1).log()
    torch.testing.assert_close(log_mean_power(small), expected)


def check_parallel_projection(eeg, labels):
    model = EEGDeformer(eeg.shape[-1])
    with torch.no_grad():
        for conv in model.encoder[:2]:
            conv.weight.mul_(20)
            assert (conv.weight.flatten(1).norm(dim=1) > conv.max_norm).any()

    def replica_forward(parallel, inputs, **kwargs):
        replica = deepcopy(parallel.module)
        for source, target in zip(parallel.module.encoder[:2], replica.encoder[:2], strict=True):
            norms = source.weight.detach().flatten(1).norm(dim=1)
            assert (norms <= source.max_norm + 1e-6).all()
            # unbind 返回多个视图，模拟 DataParallel 副本权重的原地修改限制。
            view, _ = torch.stack((source.weight, source.weight)).unbind()
            del target.weight
            target.weight = view
        return replica(inputs, **kwargs)

    with patch.object(torch.nn.DataParallel, "forward", replica_forward):
        logits = EEGDeformerDataParallel(model)(eeg)
        cross_entropy(logits, labels).backward()

    for conv in model.encoder[:2]:
        assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()


def main():
    archive_file = OUTPUT_DIR / "benchmark.npz"
    report_file = OUTPUT_DIR / "model_report.txt"
    report_file.unlink(missing_ok=True)
    if not archive_file.exists():
        raise FileNotFoundError("请先运行 python -m smoke_test.check_preprocess 生成真实样本")

    torch.set_num_threads(1)
    torch.manual_seed(config.SEED)
    report = [f"EEG-Deformer CPU 冒烟测试通过（PyTorch {torch.__version__}）"]
    with np.load(archive_file, allow_pickle=False) as archive:
        labels = torch.from_numpy(archive["labels"][:BATCH_SIZE])
        for count in archive["window_samples"]:
            eeg = torch.from_numpy(archive[f"eeg_{count}"][0, :BATCH_SIZE])
            report.append(check_batch(eeg, labels))
            check_parallel_projection(eeg, labels)
    check_log_power()
    report.append("真实样本前向、反向及单样本推理通过，DIP 零功率和半精度平方检查通过。")
    report.append("CPU 模拟并行副本的权重视图，确认最大范数约束先于复制，梯度可回传到主模型。")

    summary = "\n".join(report) + "\n"
    report_file.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
