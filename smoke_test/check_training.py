import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

import config
import train
from preprocess import prepare_benchmark


OUTPUT_DIR = config.PROJECT_DIR / "smoke_test" / "output"


def make_small_archive(path):
    prepare_benchmark(config.BENCHMARK_DIR, path, (1, 2, 3), (0.5, 1.0))
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    for key in ("eeg_125", "eeg_250"):
        data[key] = data[key][:, :8].copy()
    for key in ("labels", "block_ids"):
        data[key] = data[key][:8]
    np.savez(path, **data)


def check_resume(path, reference, resumed):
    train.train_experiment(path, reference)
    save_checkpoint = train.save_checkpoint

    def interrupt_after_log(checkpoint_path, state):
        if state["epoch"] == 2:
            raise KeyboardInterrupt
        save_checkpoint(checkpoint_path, state)

    with patch.object(train, "save_checkpoint", interrupt_after_log):
        try:
            train.train_experiment(path, resumed)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("没有触发模拟中断")

    partial = resumed / "125pt_s01" / "artifacts"
    state = torch.load(partial / "checkpoint.pt", weights_only=True)
    assert state["epoch"] == 1
    assert len(train.load_history(partial / "training.log", 2)) == 2
    train.train_experiment(path, resumed)
    with patch.object(train, "batch_step", side_effect=AssertionError("完成的折不应重新训练")):
        train.train_experiment(path, resumed)

    manifest = json.loads((resumed / "artifacts" / "run.json").read_text(encoding="utf-8"))
    assert [split["test"] for split in manifest["splits"]] == [1, 2]
    for split in manifest["splits"]:
        ids = [*split["train"], *split["validation"], split["test"]]
        assert sorted(ids) == [1, 2, 3]
    for count in (125, 250):
        for subject_id in (1, 2):
            fold = f"{count}pt_s{subject_id:02d}"
            actual = resumed / fold / "artifacts"
            expected = reference / fold / "artifacts"
            actual_state = torch.load(actual / "checkpoint.pt", weights_only=True)
            expected_state = torch.load(expected / "checkpoint.pt", weights_only=True)
            assert actual_state["epoch"] == 2 and actual_state["best_epoch"] == expected_state["best_epoch"]
            for key in ("model", "best_model", "optimizer", "scheduler", "torch_rng", "shuffle_rng"):
                torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
            assert len((actual / "training.log").read_text(encoding="utf-8").splitlines()) == 3
            actual_history = train.load_history(actual / "training.log", 2)
            expected_history = train.load_history(expected / "training.log", 2)
            for row, other in zip(actual_history, expected_history):
                for key in ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"):
                    assert row[key] == other[key]
                assert 0 < row["train_seconds"] <= row["epoch_seconds"]
            assert (resumed / fold / "train.png").read_bytes().startswith(b"\x89PNG")
    assert (resumed / "report.md").is_file()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_DIR / "training_report.txt"
    report_file.unlink(missing_ok=True)
    torch.set_num_threads(1)
    with TemporaryDirectory(dir=OUTPUT_DIR) as directory:
        root = Path(directory)
        data_file = root / "benchmark.npz"
        make_small_archive(data_file)
        with patch.multiple(config, DEVICE="cpu", NUM_SUBJECTS=3, LOSO_FOLDS=2, EPOCHS=2, BATCH_SIZE=4):
            check_resume(data_file, root / "reference", root / "resumed")

    report = (
        "EEG-Deformer CPU 训练冒烟测试通过\n"
        "使用真实 Benchmark S1–S3 的短窗样本，每窗两折，每折两个 epoch。\n"
        "已核对被试隔离、训练日志、耗时、曲线、汇总及跳过已完成折。\n"
        "模拟在日志写入后、checkpoint 提交前中断。恢复后的指标、模型、优化器、调度器和随机状态与连续训练完全一致。\n"
    )
    report_file.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
