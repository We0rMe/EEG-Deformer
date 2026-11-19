import csv
import hashlib
import io
import json
import math
import os
from time import perf_counter

import numpy as np
import torch
from filelock import FileLock
from matplotlib.figure import Figure
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import config
from model import EEGDeformer
from preprocess import get_window_samples


EPOCH_COLUMNS = (
    "epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "train_seconds", "epoch_seconds",
)


def check_archive(archive, sample_counts):
    start = round((config.CUE_SECONDS + config.VISUAL_LATENCY_SECONDS) * config.SAMPLING_RATE_HZ)
    expected = {
        "subject_ids": list(range(1, config.NUM_SUBJECTS + 1)),
        "channels": list(config.CHANNELS),
        "sampling_rate_hz": config.SAMPLING_RATE_HZ,
        "window_start_sample": start,
        "bandpass_hz": list(config.BANDPASS_HZ),
        "butterworth_order": config.BUTTERWORTH_ORDER,
    }
    actual = {name: archive[name].tolist() for name in expected}
    actual["subject_ids"].sort()
    mismatched = [name for name in expected if actual[name] != expected[name]]
    missing = {f"eeg_{count}" for count in sample_counts}.difference(archive.files)
    if mismatched or missing:
        fields = ", ".join(mismatched + sorted(missing))
        raise ValueError(f"预处理文件缺少字段或与配置不符（{fields}），请核对后运行 preprocess.py")


def loso_splits(subject_ids, num_folds, validation_fraction, seed):
    subjects = sorted(map(int, subject_ids))
    if not 1 <= num_folds <= len(subjects):
        raise ValueError("LOSO 折数须在 1 至被试总数之间")
    if len(subjects) < 3 or not 0 < validation_fraction < 1:
        raise ValueError("训练至少需要 3 名被试，验证比例须在 0 和 1 之间")

    num_remaining = len(subjects) - 1
    num_validation = min(num_remaining - 1, max(1, round(num_remaining * validation_fraction)))
    rng = np.random.default_rng(seed)
    splits = []
    for test_id in subjects[:num_folds]:
        remaining = [subject_id for subject_id in subjects if subject_id != test_id]
        shuffled = rng.permutation(remaining).tolist()
        splits.append({
            "test": test_id,
            "train": sorted(shuffled[num_validation:]),
            "validation": sorted(shuffled[:num_validation]),
        })
    return splits


def subject_loader(eeg, labels, subject_ids, selected_ids, batch_size, *, pin_memory, generator=None):
    indices = {int(subject_id): index for index, subject_id in enumerate(subject_ids)}
    selected = eeg[[indices[subject_id] for subject_id in selected_ids]]
    inputs = torch.from_numpy(selected.reshape(-1, *eeg.shape[2:]))
    targets = torch.from_numpy(np.tile(labels, len(selected_ids)))
    return DataLoader(
        TensorDataset(inputs, targets), batch_size=batch_size, shuffle=generator is not None,
        pin_memory=pin_memory, generator=generator,
    )


def configure_runtime():
    if config.DEVICE not in ("cpu", "cuda"):
        raise ValueError('DEVICE 须为 "cpu" 或 "cuda"')
    device = torch.device(config.DEVICE)
    cuda_ids = list(config.CUDA_DEVICE_IDS) if config.DEVICE == "cuda" else []
    if device.type == "cuda":
        if config.DETERMINISTIC:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        count = torch.cuda.device_count()
        if (len(cuda_ids) not in (1, 2) or len(set(cuda_ids)) != len(cuda_ids)
                or any(not 0 <= index < count for index in cuda_ids)):
            raise ValueError(f"当前有 {count} 张可见 CUDA 卡，CUDA_DEVICE_IDS 须指定一张或两张可用卡")
        device = torch.device("cuda", cuda_ids[0])
        torch.cuda.set_device(device)
        precision = "tf32" if config.ALLOW_TF32 else "ieee"
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.conv.fp32_precision = precision

    torch.use_deterministic_algorithms(config.DETERMINISTIC)
    torch.backends.cudnn.deterministic = config.DETERMINISTIC
    torch.backends.cudnn.benchmark = bool(cuda_ids) and not config.DETERMINISTIC
    return device, cuda_ids


def runtime_details(device, cuda_ids):
    return {
        "torch_version": str(torch.__version__),
        "device": str(device),
        "cuda_ids": cuda_ids,
        "gpu_names": [torch.cuda.get_device_name(index) for index in cuda_ids],
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_ids else None,
        "cudnn_enabled": torch.backends.cudnn.enabled if cuda_ids else None,
        "precision": "float32",
        "cuda_matmul_precision": torch.backends.cuda.matmul.fp32_precision if cuda_ids else None,
        "cudnn_conv_precision": torch.backends.cudnn.conv.fp32_precision if cuda_ids else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace": os.getenv("CUBLAS_WORKSPACE_CONFIG") if cuda_ids else None,
    }


def make_model(num_samples, device, cuda_ids):
    model = EEGDeformer(
        num_samples, num_channels=len(config.CHANNELS), num_classes=config.NUM_CLASSES,
        depth=config.HCT_DEPTH, temporal_kernel=config.TEMPORAL_KERNEL, num_filters=config.NUM_FILTERS,
        heads=config.ATTENTION_HEADS, head_dim=config.ATTENTION_HEAD_DIM, ff_dim=config.FEEDFORWARD_DIM,
        dropout=config.MODEL_DROPOUT, max_norm=config.ENCODER_MAX_NORM,
    ).to(device)
    network = nn.DataParallel(model, device_ids=cuda_ids) if len(cuda_ids) == 2 else model
    return model, network


def batch_step(network, batch, device, optimizer=None):
    inputs, targets = (tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    logits = network(inputs)
    loss = cross_entropy(logits, targets)
    if optimizer is not None:
        loss.backward()
        optimizer.step()
    return loss.detach(), (logits.detach().argmax(dim=1) == targets).sum(), targets.numel()


def run_epoch(network, loader, device, optimizer=None):
    training = optimizer is not None
    network.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    num_examples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            loss, matches, size = batch_step(network, batch, device, optimizer)
            loss_sum += loss * size
            correct += matches
            num_examples += size
    return loss_sum.item() / num_examples, correct.item() / num_examples


def atomic_write(path, write):
    # 同一实验目录由文件锁保护，临时文件和目标文件位于同一个目录。
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as file:
            write(file)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_text(path, text):
    atomic_write(path, lambda file: file.write(text.encode("utf-8")))


def save_manifest(path, settings):
    text = json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    if not path.exists():
        save_text(path, text)
    elif path.read_text(encoding="utf-8") != text:
        raise ValueError("实验目录已有不同的数据、环境或配置，请恢复原设置或修改 RUN_NAME")


def save_history(path, history):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EPOCH_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(history)
    save_text(path, buffer.getvalue())


def load_history(path, committed_epochs):
    with path.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))[:committed_epochs]
    if [int(row["epoch"]) for row in records] != list(range(1, committed_epochs + 1)):
        raise ValueError(f"{path} 缺少 checkpoint 对应的 epoch 记录")
    return [
        {key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
        for row in records
    ]


def save_curve(path, history):
    figure = Figure(figsize=(9, 3.4), dpi=125, layout="constrained")
    loss_axis, acc_axis = figure.subplots(1, 2)
    epochs = [record["epoch"] for record in history]
    marker = "." if len(epochs) == 1 else None
    for split, label in (("train", "Train"), ("val", "Validation")):
        loss_axis.plot(epochs, [row[f"{split}_loss"] for row in history], label=label, marker=marker)
        acc_axis.plot(epochs, [100 * row[f"{split}_acc"] for row in history], label=label, marker=marker)
    loss_axis.set(xlabel="Epoch", ylabel="Loss")
    acc_axis.set(xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
    for axis in (loss_axis, acc_axis):
        axis.grid(alpha=0.2)
        axis.legend()
    atomic_write(path, lambda file: figure.savefig(file, format="png"))
    figure.clear()


def save_checkpoint(path, state):
    atomic_write(path, lambda file: torch.save(state, file))


def write_epoch(fold_dir, history, state):
    # 日志先落盘，以 checkpoint 中的 epoch 为准截掉中断时多写的记录。
    save_history(fold_dir / "artifacts" / "training.log", history)
    save_checkpoint(fold_dir / "artifacts" / "checkpoint.pt", state)
    row = history[-1]
    print(
        f"Epoch {row['epoch']:03d}/{config.EPOCHS}  "
        f"train loss {row['train_loss']:.4f}  ACC {row['train_acc']:.2%}  "
        f"val loss {row['val_loss']:.4f}  ACC {row['val_acc']:.2%}  "
        f"LR {row['lr']:.3g}  训练 {row['train_seconds']:.2f}s  本轮 {row['epoch_seconds']:.2f}s",
        flush=True,
    )
    save_curve(fold_dir / "train.png", history)


def train_fold(eeg, labels, subject_ids, split, fold_dir, device, cuda_ids):
    artifacts = fold_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint_file = artifacts / "checkpoint.pt"
    log_file = artifacts / "training.log"

    torch.manual_seed(config.SEED)
    shuffle = torch.Generator().manual_seed(config.SEED)
    model, network = make_model(eeg.shape[-1], device, cuda_ids)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    history, best_epoch, best_model = [], 0, None
    if checkpoint_file.exists():
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        history = load_history(log_file, state["epoch"])
        save_history(log_file, history)
        save_curve(fold_dir / "train.png", history)
        best_epoch, best_model = state["best_epoch"], state["best_model"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        shuffle.set_state(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        for index, rng in zip(cuda_ids, state["cuda_rng"], strict=True):
            torch.cuda.set_rng_state(rng, device=index)
        print(f"已恢复 epoch {len(history)}/{config.EPOCHS}", flush=True)

    if len(history) < config.EPOCHS:
        train_loader = subject_loader(
            eeg, labels, subject_ids, split["train"], config.BATCH_SIZE,
            pin_memory=bool(cuda_ids), generator=shuffle,
        )
        val_loader = subject_loader(
            eeg, labels, subject_ids, split["validation"], config.BATCH_SIZE, pin_memory=bool(cuda_ids),
        )

    for epoch in range(len(history) + 1, config.EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        started = perf_counter()
        train_loss, train_acc = run_epoch(network, train_loader, device, optimizer)
        train_seconds = perf_counter() - started
        val_loss, val_acc = run_epoch(network, val_loader, device)
        epoch_seconds = perf_counter() - started
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise FloatingPointError(f"Epoch {epoch} 出现非有限损失，已保留上一个可恢复断点")
        scheduler.step()

        if best_epoch == 0 or val_acc > history[best_epoch - 1]["val_acc"]:
            best_epoch = epoch
            best_model = {key: value.detach().to(device="cpu", copy=True) for key, value in model.state_dict().items()}
        history.append({
            "epoch": epoch, "lr": lr, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "train_seconds": train_seconds, "epoch_seconds": epoch_seconds,
        })
        state = {
            "epoch": epoch, "model": model.state_dict(), "best_epoch": best_epoch, "best_model": best_model,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "shuffle_rng": shuffle.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": [torch.cuda.get_rng_state(index) for index in cuda_ids],
        }
        write_epoch(fold_dir, history, state)

    return history[best_epoch - 1]["val_acc"], sum(row["train_seconds"] for row in history)


def train_experiment(data_file, run_dir):
    if config.EPOCHS < 1:
        raise ValueError("EPOCHS 至少为 1")
    device, cuda_ids = configure_runtime()
    sample_counts = get_window_samples(config.WINDOW_SECONDS)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with FileLock(artifacts / "run.lock", timeout=0), data_file.open("rb") as source:
        data_hash = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        with np.load(source, allow_pickle=False) as archive:
            check_archive(archive, sample_counts)
            subject_ids, labels = archive["subject_ids"], archive["labels"]
            splits = loso_splits(subject_ids, config.LOSO_FOLDS, config.VALIDATION_FRACTION, config.SEED)
            options = (
                "SEED", "LOSO_FOLDS", "VALIDATION_FRACTION", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY",
                "CHANNELS", "NUM_CLASSES", "HCT_DEPTH", "TEMPORAL_KERNEL", "NUM_FILTERS", "ATTENTION_HEADS",
                "ATTENTION_HEAD_DIM", "FEEDFORWARD_DIM", "MODEL_DROPOUT", "ENCODER_MAX_NORM",
            )
            save_manifest(artifacts / "run.json", {
                "data_sha256": data_hash, "sampling_rate_hz": config.SAMPLING_RATE_HZ,
                "window_samples": sample_counts, "runtime": runtime_details(device, cuda_ids),
                "settings": {name: getattr(config, name) for name in options}, "splits": splits,
            })

            table = ["# 训练汇总", "", "| 窗长 | 完成折数 | 最佳验证 ACC（均值 ± 标准差） | 累计训练时间（s） |",
                     "| --- | ---: | ---: | ---: |"]
            for count in sample_counts:
                eeg = archive[f"eeg_{count}"]
                expected = (len(subject_ids), len(labels), len(config.CHANNELS), count)
                if eeg.shape != expected:
                    raise ValueError(f"eeg_{count} 尺寸为 {eeg.shape}，预期为 {expected}")
                scores, total_seconds = [], 0.0
                for split in splits:
                    fold_name = f"{count}pt_s{split['test']:02d}"
                    print(f"\n{fold_name}  训练 {len(split['train'])} 名  验证 {len(split['validation'])} 名", flush=True)
                    score, seconds = train_fold(eeg, labels, subject_ids, split, run_dir / fold_name, device, cuda_ids)
                    scores.append(score)
                    total_seconds += seconds
                mean, sd = 100 * np.mean(scores), 100 * np.std(scores)
                table.append(f"| {count / config.SAMPLING_RATE_HZ:g} s | {len(scores)} | {mean:.2f}% ± {sd:.2f} pp | {total_seconds:.2f} |")

            save_text(run_dir / "report.md", "\n".join(table) + "\n\n统计采用折间总体标准差，单折标准差为 0。累计时间为所有已完成 epoch 的训练用时之和。\n")
    print(f"训练汇总已保存到 {run_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    train_experiment(config.PROCESSED_FILE, config.RESULTS_DIR / f"{config.RUN_NAME}_seed{config.SEED}")
