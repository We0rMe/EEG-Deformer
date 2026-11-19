# EEG-Deformer for SSVEP

本项目将 EEG-Deformer 用于清华大学 Benchmark 数据集的 40 类跨被试 SSVEP 解码。当前阶段提供数据预处理、模型实现、LOSO 训练与验证。 [EEG-Deformer 原代码](https://github.com/yi-ding-cs/EEG-Deformer) · [Benchmark 数据集](https://bci.med.tsinghua.edu.cn/download.html)

## 数据预处理

默认从项目上一级的 `data/benchmark/` 读取 `S1.mat` 至 `S35.mat`、`64-channels.loc` 和 `Freq_Phase.mat`。换电脑后可在 `config.py` 修改 `BENCHMARK_DIR`、输出路径和窗长。

1. 按 `O1、Oz、O2、PO3、POz、PO4、Pz、PO5、PO6` 的顺序选取九个通道，保留数据发布时的 250 Hz 采样率。
2. 以提示出现为试次起点，闪烁从 0.50 s 开始，加上 0.14 s 视觉延迟，从 0.64 s 开始截取左闭右开窗口 `[0.64, 0.64+d)`。默认 `d` 为 0.5 s 和 1.0 s，切片分别为 `160:285` 和 `160:410`。
3. 对每个窗长独立处理。先截窗，再对各通道、各试次沿时间轴做 8–64 Hz 巴特沃斯双向零相位滤波。SciPy 的 `butter(N=4, ..., output="sos")` 中 `N=4` 是低通原型阶数，`sosfiltfilt` 使用窗内采样做边界延拓。输出保留时域波形并转为 `float32`。 [时域基线的预处理口径](https://arxiv.org/pdf/2210.04172#page=8) · [SciPy 双向滤波](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)

每个试次在每种窗长下生成一个样本。输出文件 `data_processed/benchmark.npz` 可用 `numpy.load` 读取。

| 数据键 | 内容 |
| --- | --- |
| `eeg_125`、`eeg_250` | `[被试, 试次, 通道, 采样点]`，默认形状为 `(35, 240, 9, 125)` 和 `(35, 240, 9, 250)` |
| `subject_ids` | EEG 第一轴对应的原始被试编号 |
| `labels`、`block_ids` | 每名被试共用的试次索引，区块优先。每个区块内按原始目标轴排列，类别为 0 至 39，区块为 1 至 6 |
| `frequencies_hz`、`phases_rad` | 原文件中的频率和相位，直接用类别索引查询 |
| `channels`、`window_samples` 等 | 通道顺序、窗长采样点数、起点、采样率及滤波设置 |

预处理只使用固定滤波参数。后续每个窗长独立训练，各折由 `subject_ids` 划分。若加入需要拟合的标准化，统计量只从当前折的训练被试估计。`config.py` 已集中预留 LOSO 折数、种子与训练设备，并记录 ITR 口径 `T = d + 0.5 s`。 [SSVEPformer 的 LOSO 与 ITR 口径](https://arxiv.org/pdf/2210.04172#page=17)

## EEG-Deformer 模型

### 选择时域或复频谱输入

在 `config.py` 设置 `INPUT_TYPE`。本次默认 `"complex_channels"`。训练和 GPU 预热均直接读取已有的 `data_processed/benchmark.npz`，每个窗长在内存中转换一次，所有 LOSO 折复用。频谱变换逐试次进行。

| `INPUT_TYPE` | 单试次输入 | 最后一轴 |
| --- | --- | --- |
| `time` | `[9, 125]` 或 `[9, 250]` | 时间采样点 |
| `complex_channels` | `[18, 281]` | 频点。前九行是各通道实部，后九行是对应虚部 |
| `complex` | `[9, 562]` | 每行依次排列 281 个实部和 281 个虚部 |

两种复频谱输入均对滤波后的原窗沿时间轴做实数 FFT。在 250 Hz 采样率下补零到 1250 点，保留 8–64 Hz 的频点（含两端），实部和虚部均除以原窗的 125 或 250 个采样点。两个窗长仍独立训练。使用 `complex_channels` 时，卷积和池化沿频率轴进行，默认三级 HCT 的长度依次为 `281 → 140 → 70 → 35 → 17`。

### 时域基线结构

使用 `model.EEGDeformer(num_samples=125)` 或 `model.EEGDeformer(num_samples=250)` 为对应窗长创建模型。输入为 `[批大小, 9, 采样点]`，输出为 `[批大小, 40]` 的 logits。模型参数在 `config.py` 中设置，`HCT_DEPTH` 默认是 3，设为 4 可使用官方示例的深度。250 Hz 下，浅层编码器和 HCT 的细粒度分支均使用 25 点时间卷积核。池化按官方实现向下取整，125 点和 250 点的时间维分别为 `125 → 62 → 31 → 15 → 7` 和 `250 → 125 → 62 → 31 → 15`。 [论文 §III、§IV-D](https://arxiv.org/html/2405.00719v1) · [官方模型](https://github.com/yi-ding-cs/EEG-Deformer/blob/main/models/EEGDeformer.py)

浅层编码器依次提取时间和空间特征，并对卷积核应用最大范数约束。HCT 的粗粒度分支沿卷积核对应的 token 轴做注意力，细粒度分支沿时间轴做卷积，两路相加。DIP 将各层细粒度分支的对数平均功率与末层主干特征拼接。功率在半精度输入时用 `float32` 计算，并在取对数前设置数值下限。模型依据论文和官方实现编写。 [上游 CBCR 许可证与版权声明](https://github.com/yi-ding-cs/EEG-Deformer/blob/main/CBCR%20License)

用 PyTorch 2.11.0 CPU 版对照了官方提交 `47df1c4`。在相同初始权重、输入和 Dropout 随机种子下，125 点和 250 点输入在深度 2、3、4 的训练和推理 logits、输入及参数梯度、卷积核最大范数约束均通过数值核对。logits 与梯度使用 `rtol=1e-5`、`atol=1e-6`。 [对照用的官方模型版本](https://github.com/yi-ding-cs/EEG-Deformer/blob/47df1c4d8ce375ae21117f84e81e969041d907f6/models/EEGDeformer.py)

## 本地 CPU 冒烟测试

在项目根目录运行。若已具备预处理依赖，可直接使用相应的 Python 环境。新建环境可用 Python 3.12 或更高版本，在 Windows PowerShell 中按以下命令安装 NumPy 和 SciPy 并运行测试。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.4.6 scipy==1.18.0
.\.venv\Scripts\python.exe -m smoke_test.check_preprocess
```

冒烟测试使用 S1 和 S2，检查真实波形、样本形状、通道与标签顺序，并验证窗外采样和其他试次无法影响当前分类窗。测试数据和简短报告位于 `smoke_test/output/`。

生成测试数据后，可安装同版本的 CPU 版 PyTorch 并运行模型冒烟测试。报告保存在 `smoke_test/output/model_report.txt`。

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m smoke_test.check_model
```

## 服务器预处理

在服务器配置好 `BENCHMARK_DIR` 后，从项目根目录运行。`requirements.txt` 为后续训练固定了 CUDA 12.8 版 PyTorch，预处理本身在 CPU 上执行。下面以 Linux SSH 为例，Windows 服务器可使用 `.venv\Scripts\python.exe`。 [PyTorch CUDA 12.8 安装包](https://download.pytorch.org/whl/cu128/torch/)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python preprocess.py
```

脚本逐个读取被试，处理完成后统一写出数据。重复执行会用完整的新文件替换旧输出。原始数据、预处理产物、冒烟测试结果和本地虚拟环境均已列入 Git 忽略规则。

## LOSO 训练与验证

在 `config.py` 中设置窗长、种子、折数与设备。每个窗长从 S1 开始依次留出 `LOSO_FOLDS` 名被试。每折从剩余被试中按固定种子分出约 20% 用于验证，其余用于训练。两个窗长共用同一份被试划分。每折训练 200 个 epoch，使用交叉熵、Adam 和余弦退火，保留验证 ACC 最高的权重。同分时选最早的 epoch。留出被试只供后续测试阶段使用。 [EEG-Deformer 论文 §IV-C、§IV-D](https://arxiv.org/html/2405.00719v1#S4)

`DEVICE = "cuda"` 时，`CUDA_DEVICE_IDS = (0,)` 用单卡，`(0, 1)` 用 `DataParallel` 双卡。`BATCH_SIZE` 是两卡合计的 batch 大小。默认使用 FP32、关闭 TF32 并启用确定性算法。`DEVICE = "cpu"` 可在 CPU 运行。

在服务器的训练环境中先运行 GPU 预热。它从第一折训练被试读取真实数据，每个窗长执行 3 个预热 batch、5 个计时 batch 和 1 个记录算子的训练 batch。本次结果写入 `smoke_test/output/gpu_report_complex_channels.json`，其中包含输入设置、张量形状、梯度检查、耗时、显存峰值和实际后端。时域检查仍写入 `gpu_report.json`。

```bash
python -m smoke_test.check_gpu
```

核对预热结果后，在同一环境运行 `python train.py`。结果目录默认按输入类型命名，时域基线沿用 `eeg_deformer_seed42`。同一实验目录重复执行会从最近已提交的 epoch 恢复。每个 epoch 结束即保存日志、checkpoint 和更新曲线。若在同一种输入下更改频谱参数、设备或训练参数，请另设 `RUN_NAME`。

```text
results/eeg_deformer_complex_channels_seed42/
├── report.md                  # 所有折完成后的验证 ACC 汇总和累计训练时间
├── artifacts/run.json         # 配置、设备、数据指纹和每折的被试划分
└── 125pt_s01/                 # 125 点窗，留出 S1
    ├── train.png              # 训练与验证的 Loss、ACC 曲线
    └── artifacts/
        ├── training.log       # CSV 格式，保留每轮 Loss、ACC、LR 和耗时
        └── checkpoint.pt     # 最新训练状态和最佳验证权重
```

日志的 `train_seconds` 是本轮训练用时，`epoch_seconds` 包含训练和验证。两列都在保存日志和画图前计时。累计训练时间按最终提交的 epoch 求和。CPU 冒烟测试可运行 `python -m smoke_test.check_training`，需额外安装 `matplotlib` 和 `filelock`，结果写入 `smoke_test/output/training_report.txt`。
