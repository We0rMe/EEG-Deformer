import json
import statistics
from time import perf_counter

import numpy as np
import torch
from torch.optim import Adam
from torch.profiler import ProfilerActivity, profile, supported_activities

import config
from features import input_settings, prepare_inputs
from preprocess import get_window_samples
from train import (
    batch_step, check_archive, configure_runtime, loso_splits, make_model, runtime_details, subject_loader,
)


def probe_window(eeg, labels, subject_ids, train_ids, device, cuda_ids):
    torch.manual_seed(config.SEED)
    shuffle = torch.Generator().manual_seed(config.SEED)
    loader = subject_loader(
        eeg, labels, subject_ids, train_ids, config.BATCH_SIZE, pin_memory=bool(cuda_ids), generator=shuffle,
    )
    batches = iter(loader)
    if len(loader.dataset) // config.BATCH_SIZE < config.GPU_WARMUP_BATCHES + config.GPU_PROBE_BATCHES + 1:
        raise ValueError("训练样本不足以提供所需的完整预热和计时 batch")
    model, network = make_model(eeg.shape[-2:], device, cuda_ids)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    def synchronize():
        for index in cuda_ids:
            torch.cuda.synchronize(index)

    for _ in range(config.GPU_WARMUP_BATCHES):
        batch_step(network, next(batches), device, optimizer)
    synchronize()
    for index in cuda_ids:
        torch.cuda.reset_peak_memory_stats(index)

    durations = []
    for _ in range(config.GPU_PROBE_BATCHES):
        batch = next(batches)
        started = perf_counter()
        loss, _, _ = batch_step(network, batch, device, optimizer)
        synchronize()
        durations.append(1000 * (perf_counter() - started))
        if not torch.isfinite(loss):
            raise FloatingPointError("预热检查出现非有限损失")
    memory = {f"cuda:{index}": round(torch.cuda.max_memory_allocated(index) / 1024 ** 2) for index in cuda_ids}

    observed = {}

    def record_output(_module, _inputs, output):
        observed[str(output.device)] = {"shape": list(output.shape), "dtype": str(output.dtype)}

    hook = model.classifier.register_forward_hook(record_output)
    activities = [ProfilerActivity.CPU]
    if cuda_ids and ProfilerActivity.CUDA in supported_activities():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, acc_events=True) as trace:
        loss, _, _ = batch_step(network, next(batches), device, optimizer)
        synchronize()
    hook.remove()
    if not torch.isfinite(loss) or any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in model.parameters()
    ):
        raise FloatingPointError("预热检查的损失或参数梯度无效")

    operators = sorted({
        event.key for event in trace.key_averages()
        if any(name in event.key.lower() for name in ("convolution", "cudnn", "aten::bmm", "aten::mm", "adam"))
    })
    kernels = [event for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    kernels.sort(key=lambda event: event.device_time_total, reverse=True)
    return {
        "input_shape": [config.BATCH_SIZE, eeg.shape[-2], eeg.shape[-1]],
        "logits_per_device": observed,
        "loss_and_gradients_finite": True,
        "median_train_batch_ms": round(statistics.median(durations), 2),
        "peak_allocated_memory_mib": memory,
        "observed_operators": operators,
        "cuda_activity_enabled": ProfilerActivity.CUDA in activities,
        "cuda_kernel_examples": list(dict.fromkeys(event.name[:180] for event in kernels))[:5],
    }


def main():
    if config.GPU_WARMUP_BATCHES < 0 or config.GPU_PROBE_BATCHES < 1:
        raise ValueError("预热 batch 数须非负，计时 batch 数至少为 1")
    device, cuda_ids = configure_runtime()
    if not cuda_ids:
        raise ValueError('请将 DEVICE 设为 "cuda"，并指定参与检查的 GPU')
    sample_counts = get_window_samples(config.WINDOW_SECONDS)
    report = runtime_details(device, cuda_ids)
    report["input"] = input_settings()
    report.update({"warmup_batches": config.GPU_WARMUP_BATCHES, "timed_batches": config.GPU_PROBE_BATCHES, "windows": {}})
    with np.load(config.PROCESSED_FILE, allow_pickle=False) as archive:
        check_archive(archive, sample_counts)
        subject_ids, labels = archive["subject_ids"], archive["labels"]
        split = loso_splits(subject_ids, 1, config.VALIDATION_FRACTION, config.SEED)[0]
        for count in sample_counts:
            eeg = prepare_inputs(archive[f"eeg_{count}"], config.SAMPLING_RATE_HZ, report["input"])
            print(f"正在检查 {count} 点分类窗，模型输入形状 {eeg.shape[-2:]}", flush=True)
            report["windows"][str(count)] = probe_window(
                eeg, labels, subject_ids, split["train"], device, cuda_ids,
            )
    filename = "gpu_report.json" if config.INPUT_TYPE == "time" else f"gpu_report_{config.INPUT_TYPE}.json"
    output = config.PROJECT_DIR / "smoke_test" / "output" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"GPU 检查结果已保存到 {output}", flush=True)


if __name__ == "__main__":
    main()
