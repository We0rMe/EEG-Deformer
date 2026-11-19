from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

# Benchmark 数据属性
SAMPLING_RATE_HZ = 250           # 发布数据的采样率
NUM_SUBJECTS = 35                 # 被试编号为 1 到 35
NUM_CHANNELS = 64                 # 原始电极数
TRIAL_SAMPLES = 1500              # 每个原始试次的采样点数
NUM_CLASSES = 40                  # 每个区块的目标数
NUM_BLOCKS = 6                    # 每名被试的区块数
CUE_SECONDS = 0.50               # 提示出现到闪烁开始的时长
STIMULUS_SECONDS = 5.0            # 闪烁持续时间

# 预处理设置
BENCHMARK_DIR = PROJECT_DIR.parent / "data" / "benchmark"  # 原始数据目录
PROCESSED_FILE = PROJECT_DIR / "data_processed" / "benchmark.npz"  # 正式数据文件
CHANNELS = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "Pz", "PO5", "PO6")  # 输入通道及顺序
VISUAL_LATENCY_SECONDS = 0.14     # 从闪烁开始到分类窗起点的延迟
WINDOW_SECONDS = (0.5, 1.0)       # 分类窗长，可同时指定多个
BANDPASS_HZ = (8.0, 64.0)         # 巴特沃斯带通的临界频率
BUTTERWORTH_ORDER = 4             # 巴特沃斯低通原型阶数

# 模型输入，直接使用已滤波的分类窗
INPUT_TYPE = "complex"   # time 为时域，complex_channels 沿通道拼接实虚部，complex 沿频率拼接
FFT_RESOLUTION = 0.2              # FFT 频点间距（Hz）
SPECTRUM_HZ = (8.0, 64.0)        # 复频谱的起止频率（Hz），包含两端

# EEG-Deformer 模型设置
HCT_DEPTH = 3                    # 级联 HCT 块数
TEMPORAL_KERNEL = 25             # 沿最后一轴的卷积核长度，单位为采样点或频点，须为奇数
NUM_FILTERS = 64                 # 浅层编码器的卷积核数，也是注意力的 token 数
ATTENTION_HEADS = 16             # 多头注意力的头数
ATTENTION_HEAD_DIM = 16          # 每个注意力头的投影维度
FEEDFORWARD_DIM = 16             # HCT 前馈网络的隐藏维度
MODEL_DROPOUT = 0.5              # HCT 的 Dropout 概率
ENCODER_MAX_NORM = 2.0           # 浅层编码器各卷积核的最大 L2 范数

# 训练与评估设置
RESULTS_DIR = PROJECT_DIR / "results"  # 训练产物目录
RUN_NAME = "eeg_deformer" if INPUT_TYPE == "time" else f"eeg_deformer_{INPUT_TYPE}"  # 同一输入下改参数可另命名
LOSO_FOLDS = NUM_SUBJECTS         # 从被试 1 起依次留出的折数
SEED = 42                        # 每次实验使用一个随机种子
VALIDATION_FRACTION = 0.20        # 非测试被试中用于验证的比例
EPOCHS = 200                      # 每个窗长、每折的训练轮数
BATCH_SIZE = 64                   # 总 batch 大小，双卡时由两张卡分担
LEARNING_RATE = 1e-3              # Adam 初始学习率，随后做余弦退火
WEIGHT_DECAY = 1e-5               # Adam 权重衰减系数
DEVICE = "cuda"                   # 训练设备，可设为 "cpu" 或 "cuda"
CUDA_DEVICE_IDS = (0,)           # 单卡用 (0,)，双卡用 (0, 1)
DETERMINISTIC = True              # 启用确定性算法，并关闭 cuDNN 自动寻优
ALLOW_TF32 = False                # CUDA 下允许 FP32 矩阵乘和卷积使用 TF32
ITR_GAZE_SHIFT_SECONDS = 0.5      # ITR 中 T = 窗长 + 视线转移时间

# GPU 预热检查设置
GPU_WARMUP_BATCHES = 3            # 正式计时前先运行的训练 batch 数
GPU_PROBE_BATCHES = 5             # 预热后计时的训练 batch 数
