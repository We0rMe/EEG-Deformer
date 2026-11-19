import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

import config


def get_channel_indices(location_file):
    """按配置顺序取电极下标，兼容通道名大小写差异。"""
    channel_indices = {}
    for line in location_file.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if fields:
            channel_indices[fields[-1].casefold()] = int(fields[0]) - 1
    return [channel_indices[name.casefold()] for name in config.CHANNELS]


def get_window_samples(window_seconds):
    sample_counts = []
    max_seconds = config.STIMULUS_SECONDS - config.VISUAL_LATENCY_SECONDS
    for seconds in window_seconds:
        if not 0 < seconds <= max_seconds:
            raise ValueError(f"窗长须大于 0 且不超过 {max_seconds:g} s，收到 {seconds} s")
        samples = seconds * config.SAMPLING_RATE_HZ
        count = round(samples)
        if not math.isclose(samples, count, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"窗长 {seconds:g} s 在当前采样率下须对应整数采样点")
        sample_counts.append(count)
    if not sample_counts or len(set(sample_counts)) != len(sample_counts):
        raise ValueError("请至少配置一个窗长，且各窗长不能重复")
    return sample_counts


def filter_trials(raw_data, channel_indices, start, sample_count, sos):
    """返回形状为 [试次, 通道, 采样点] 的时域数据。"""
    window = raw_data[channel_indices, start : start + sample_count, :, :]
    # 先截窗，仅沿时间轴滤波，每个通道和试次独立处理。
    filtered = sosfiltfilt(sos, window, axis=1)
    # 从 [通道, 时间, 类别, 区块] 转为区块优先的试次顺序。
    trials = filtered.transpose(3, 2, 0, 1).reshape(-1, len(channel_indices), sample_count)
    return np.ascontiguousarray(trials, dtype=np.float32)


def save_archive(output_path, arrays):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 完整写入后再替换输出文件，避免中断时留下半份数据。
    temp_path = None
    try:
        with NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as file:
            temp_path = Path(file.name)
            np.savez(file, **arrays)
        temp_path.replace(output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def prepare_benchmark(data_dir, output_path, subject_ids, window_seconds):
    """逐个读取被试，将选定窗长写入同一个 NPZ 文件。"""
    sample_counts = get_window_samples(window_seconds)
    start = round((config.CUE_SECONDS + config.VISUAL_LATENCY_SECONDS) * config.SAMPLING_RATE_HZ)
    channels = get_channel_indices(data_dir / "64-channels.loc")
    targets = loadmat(data_dir / "Freq_Phase.mat", variable_names=("freqs", "phases"))
    frequencies = targets["freqs"].ravel()
    phases = targets["phases"].ravel()
    if frequencies.size != config.NUM_CLASSES or phases.size != config.NUM_CLASSES:
        raise ValueError("Freq_Phase.mat 中的频率、相位数量应与目标数一致")

    sos = butter(
        config.BUTTERWORTH_ORDER, config.BANDPASS_HZ, btype="bandpass",
        fs=config.SAMPLING_RATE_HZ, output="sos",
    )
    raw_shape = (config.NUM_CHANNELS, config.TRIAL_SAMPLES, config.NUM_CLASSES, config.NUM_BLOCKS)
    num_trials = config.NUM_CLASSES * config.NUM_BLOCKS
    eeg = {
        f"eeg_{count}": np.empty((len(subject_ids), num_trials, len(channels), count), dtype=np.float32)
        for count in sample_counts
    }
    for index, subject_id in enumerate(subject_ids):
        source_file = data_dir / f"S{subject_id}.mat"
        raw = loadmat(source_file, variable_names=("data",))["data"]
        if raw.shape != raw_shape:
            raise ValueError(f"{source_file.name} 的形状是 {raw.shape}，预期为 {raw_shape}")
        for count in sample_counts:
            eeg[f"eeg_{count}"][index] = filter_trials(raw, channels, start, count, sos)
        del raw
        print(f"[{index + 1}/{len(subject_ids)}] 已处理 S{subject_id}", flush=True)

    arrays = {
        "subject_ids": np.asarray(subject_ids, dtype=np.int64),
        "labels": np.tile(np.arange(config.NUM_CLASSES, dtype=np.int64), config.NUM_BLOCKS),
        "block_ids": np.repeat(np.arange(1, config.NUM_BLOCKS + 1), config.NUM_CLASSES),
        "frequencies_hz": frequencies,
        "phases_rad": phases,
        "channels": np.asarray(config.CHANNELS),
        "sampling_rate_hz": config.SAMPLING_RATE_HZ,
        "window_start_sample": start,
        "window_samples": np.asarray(sample_counts),
        "bandpass_hz": np.asarray(config.BANDPASS_HZ),
        "butterworth_order": config.BUTTERWORTH_ORDER,
        "scipy_version": scipy.__version__,
    }
    save_archive(output_path, {**arrays, **eeg})
    shapes = ", ".join(f"{key} {value.shape}" for key, value in eeg.items())
    print(f"已保存 {output_path}（{shapes}）", flush=True)


if __name__ == "__main__":
    prepare_benchmark(
        config.BENCHMARK_DIR,
        config.PROCESSED_FILE,
        range(1, config.NUM_SUBJECTS + 1),
        config.WINDOW_SECONDS,
    )
