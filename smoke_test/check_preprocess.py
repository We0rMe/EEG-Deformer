import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

import config
from preprocess import filter_trials, prepare_benchmark


OUTPUT_DIR = config.PROJECT_DIR / "smoke_test" / "output"
SUBJECT_IDS = (1, 2)
WINDOW_SECONDS = (0.5, 1.0)
WINDOW_SAMPLES = (125, 250)
EXPECTED_CHANNELS = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "Pz", "PO5", "PO6")
CHANNEL_INDICES = [60, 61, 62, 54, 55, 56, 47, 53, 57]
START = 160


def check_archive(output_file, sos):
    raw = loadmat(config.BENCHMARK_DIR / "S1.mat", variable_names=("data",))["data"]
    targets = loadmat(config.BENCHMARK_DIR / "Freq_Phase.mat", variable_names=("freqs", "phases"))
    shapes = []
    with np.load(output_file, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["subject_ids"], SUBJECT_IDS)
        np.testing.assert_array_equal(archive["channels"], EXPECTED_CHANNELS)
        np.testing.assert_array_equal(archive["window_samples"], WINDOW_SAMPLES)
        assert archive["window_start_sample"].item() == START
        np.testing.assert_array_equal(archive["block_ids"], np.repeat(np.arange(1, 7), 40))
        np.testing.assert_array_equal(archive["labels"], list(range(40)) * 6)
        np.testing.assert_array_equal(archive["frequencies_hz"], targets["freqs"].ravel())
        np.testing.assert_array_equal(archive["phases_rad"], targets["phases"].ravel())

        for count in WINDOW_SAMPLES:
            eeg = archive[f"eeg_{count}"]
            assert eeg.shape == (2, 240, 9, count)
            assert eeg.dtype == np.float32 and np.isfinite(eeg).all()
            assert not np.array_equal(eeg[0], eeg[1])

            # 对照 S1 第 3 区块的第 8 个目标，直接对 9 条波形做滤波。
            window = raw[CHANNEL_INDICES, START : START + count, 7, 2]
            expected = sosfiltfilt(sos, window, axis=-1).astype(np.float32)
            np.testing.assert_array_equal(eeg[0, 87], expected)
            shapes.append(f"eeg_{count} {eeg.shape}")
    return "，".join(shapes)


def check_filter_isolation(sos):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(64, 1500, 2, 2))
    for count in WINDOW_SAMPLES:
        end = START + count
        baseline = filter_trials(raw, CHANNEL_INDICES, START, count, sos)

        changed = raw.copy()
        changed[:, :START, :, :] *= 100
        changed[:, end:, :, :] *= 100
        actual = filter_trials(changed, CHANNEL_INDICES, START, count, sos)
        np.testing.assert_array_equal(actual, baseline)

        changed[:, START:end, 1, 0] *= 2
        actual = filter_trials(changed, CHANNEL_INDICES, START, count, sos)
        np.testing.assert_array_equal(actual[[0, 2, 3]], baseline[[0, 2, 3]])
        assert not np.array_equal(actual[1], baseline[1])


def main():
    output_file = OUTPUT_DIR / "benchmark.npz"
    report_file = OUTPUT_DIR / "report.txt"
    report_file.unlink(missing_ok=True)
    prepare_benchmark(config.BENCHMARK_DIR, output_file, SUBJECT_IDS, WINDOW_SECONDS)
    sos = butter(4, (8, 64), btype="bandpass", fs=250, output="sos")
    shapes = check_archive(output_file, sos)
    check_filter_isolation(sos)

    report = (
        "Benchmark 预处理冒烟测试通过\n"
        f"使用 S1、S2 的真实数据，输出维度为 {shapes}\n"
        "已核对通道顺序、原始目标标签、区块、频率与相位、数值有限性及单试次波形。\n"
        "改变窗外采样或其他试次的采样，目标窗输出保持一致。\n"
    )
    report_file.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
