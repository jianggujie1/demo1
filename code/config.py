import os

# 路径配置
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")  # 原始数据路径
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")  # 预处理数据路径
RESULTS_MODEL_DIR = os.path.join(ROOT_DIR, "results", "models")  # 模型保存路径
RESULTS_LOG_DIR = os.path.join(ROOT_DIR, "results", "logs")  # 日志路径
RESULTS_FIG_DIR = os.path.join(ROOT_DIR, "results", "figures")  # 可视化路径
# ESC-50数据集路径
ESC50_RAW_DIR = os.path.join(DATA_RAW_DIR, "ESC-50")  # 指向ESC-50根目录
ESC50_PROCESSED_DIR = os.path.join(
    DATA_PROCESSED_DIR, "esc50_logmel"
)  # 预处理结果保存路径
os.makedirs(ESC50_PROCESSED_DIR, exist_ok=True)
# ESC-50参数（样本时长5秒，时间帧稍长）
ESC50_MAX_TIME_FRAMES = 80  # 适配5秒音频的时间帧长度（比UrbanSound8K的64稍大）
ESC50_NUM_CLASSES = 50  # ESC-50共50类（若想只跑ESC-10子集，可改为10）
# 创建必要目录
for dir_path in [
    DATA_PROCESSED_DIR,
    RESULTS_MODEL_DIR,
    RESULTS_LOG_DIR,
    RESULTS_FIG_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)

# 音频处理参数
SAMPLE_RATE = 16000  # 采样率
N_FFT = 512  # FFT点数
HOP_LENGTH = N_FFT // 2  # 帧移（保证时间分辨率）
MEL_BINS = 128  # Mel频带数量
MAX_TIME_FRAMES = 64  # 时间帧最大长度（适配UrbanSound8K≤4s样本）
R = 0.35  # 分频比（论文最优值）

# 模型训练参数
BATCH_SIZE = 32  # 批次大小（3080Ti可跑32）
EPOCHS = 500  # 训练轮数（论文指定）
INIT_LR = 0.0001  # 初始学习率
NUM_CLASSES = 50  # 类别数（UrbanSound8K/ESC-10均为10类）
VAL_SPLIT = 0.2  # 验证集比例
