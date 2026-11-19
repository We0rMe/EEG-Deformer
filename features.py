"""将已截窗、滤波的 EEG 转为模型输入。"""

import math

import numpy as np

import config


def input_settings():
    settings = {"input_type": config.INPUT_TYPE}
    if config.INPUT_TYPE != "time":
        settings.update(fft_resolution=config.FFT_RESOLUTION, spectrum_hz=config.SPECTRUM_HZ,
                        include_upper_bin=True, fft_scale=1.0)
    return settings


def prepare_inputs(eeg, sample_rate, settings):
    """保留被试和试次顺序，返回 [被试, 试次, 输入通道, 输入长度]。"""
    layout = settings["input_type"]
    if layout == "time":
        return eeg
    if layout not in ("complex_channels", "complex"):
        raise ValueError("INPUT_TYPE 可选 time、complex_channels 或 complex")

    spacing = settings["fft_resolution"]
    if spacing <= 0:
        raise ValueError("FFT 频点间距须为正数")
    nfft = round(sample_rate / spacing)
    low, high = settings["spectrum_hz"]
    first, last = round(low / spacing), round(high / spacing)
    if (nfft < eeg.shape[-1] or not 0 <= low < high <= sample_rate / 2
            or not math.isclose(sample_rate / nfft, spacing)
            or not math.isclose(first * spacing, low) or not math.isclose(last * spacing, high)):
        raise ValueError("FFT 长度须覆盖数据窗，频段端点须落在有效频点上")

    stop = last + int(settings["include_upper_bin"])
    channels, bins = eeg.shape[-2], stop - first
    shape = (2 * channels, bins) if layout == "complex_channels" else (channels, 2 * bins)
    axis = -2 if layout == "complex_channels" else -1
    features = np.empty((*eeg.shape[:2], *shape), dtype=np.float32)
    # 逐被试计算以限制临时内存，各试次仅沿自己的时间轴做 FFT。
    for index, trials in enumerate(eeg):
        spectrum = np.fft.rfft(trials, n=nfft, axis=-1)[..., first:stop]
        spectrum *= settings["fft_scale"] / eeg.shape[-1]
        features[index] = np.concatenate((spectrum.real, spectrum.imag), axis=axis)
    return features
