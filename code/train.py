import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau,
)
from sklearn.model_selection import train_test_split
from config import *  # 导入全局配置
from dataset_loader import load_esc50  # 仅使用ESC-50数据集
from model import build_rf_drsn_ema
import random


# --------------------------
# 新增：数据增强函数（音频频谱增强）
# --------------------------
def audio_augmentation(spec):
    """
    对单条梅尔频谱进行数据增强
    :param spec: 输入频谱 (time, freq, 1)
    :return: 增强后的频谱
    """
    # 1. 随机添加高斯噪声（强度可控）
    if random.random() < 0.4:
        noise = np.random.normal(0, 0.005, spec.shape)  # 噪声强度可调整
        spec = spec + noise

    # 2. 随机缩放（0.9-1.1倍）
    if random.random() < 0.4:
        scale = random.uniform(0.9, 1.1)
        spec = spec * scale

    # 3. 随机时间平移（左右不超过8个帧）
    if random.random() < 0.3 and spec.shape[0] > 8:
        shift = random.randint(-8, 8)
        spec = np.roll(spec, shift, axis=0)
        # 平移后边缘填充0（避免信息泄露）
        if shift > 0:
            spec[:shift, :, :] = 0
        else:
            spec[shift:, :, :] = 0

    # 4. 随机频率平移（上下不超过4个频带）
    if random.random() < 0.3 and spec.shape[1] > 4:
        shift = random.randint(-4, 4)
        spec = np.roll(spec, shift, axis=1)
        # 平移后边缘填充0
        if shift > 0:
            spec[:, :shift, :] = 0
        else:
            spec[:, shift:, :] = 0

    return spec


# --------------------------
# 新增：带数据增强的训练生成器
# --------------------------
def train_generator(X, y, batch_size):
    """
    训练数据生成器，实时进行数据增强
    :param X: 训练数据
    :param y: 训练标签
    :param batch_size: 批次大小
    :return: 增强后的批次数据 (X_batch, y_batch)
    """
    num_samples = len(X)
    while True:
        # 打乱数据索引
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            # 取当前批次索引
            batch_indices = indices[i : i + batch_size]
            X_batch = X[batch_indices].copy()
            y_batch = y[batch_indices].copy()

            # 对批次内每个样本进行增强
            for j in range(len(X_batch)):
                X_batch[j] = audio_augmentation(X_batch[j])

            yield X_batch, y_batch


# --------------------------
# 1. 配置数据集与训练参数
# --------------------------
# 选择是否使用ESC-10子集（True=10类，False=50类）
USE_ESC10_SUBSET = False

# 加载数据（自动适配ESC-10/ESC-50）
X, y = load_esc50(use_processed=True, use_esc10_subset=USE_ESC10_SUBSET)

# 调整类别数（ESC-10为10类，ESC-50为50类）
NUM_CLASSES = 10 if USE_ESC10_SUBSET else 50

# 调整批次大小（ESC-10样本少，用16更稳定）
BATCH_SIZE = 16 if USE_ESC10_SUBSET else 32

# 划分训练集/验证集（按8:2拆分，保持类别平衡）
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VAL_SPLIT,
    random_state=42,
    stratify=y,  # 确保验证集类别分布与训练集一致
)
print(
    f"数据集加载完成：训练集{X_train.shape}，验证集{X_val.shape}，类别数{NUM_CLASSES}"
)

# --------------------------
# 2. 构建模型
# --------------------------
# 适配输入形状和类别数
model = build_rf_drsn_ema()
# 打印模型结构（确认输出层类别数正确）
model.summary()

# --------------------------
# 3. 配置训练回调（新增学习率衰减）
# --------------------------
# 结果保存路径（区分ESC-10和ESC-50）
prefix = "esc10" if USE_ESC10_SUBSET else "esc50"
model_save_path = os.path.join(RESULTS_MODEL_DIR, f"{prefix}_best_model.h5")
log_save_path = os.path.join(RESULTS_LOG_DIR, f"{prefix}_train.log")
fig_save_path = os.path.join(RESULTS_FIG_DIR, f"{prefix}_accuracy_curve.png")

# 回调函数：保存最优模型+早停+日志记录+学习率衰减
callbacks = [
    ModelCheckpoint(
        model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=20 if USE_ESC10_SUBSET else 30,  # ESC-50早停从50轮改为30轮
        mode="max",
        restore_best_weights=True,  # 新增：恢复最佳权重
        verbose=1,
    ),
    CSVLogger(log_save_path),  # 保存训练日志（loss、accuracy等）
    # 新增：学习率衰减（验证损失10轮不下降则减半）
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        mode="min",
        min_lr=1e-6,  # 最小学习率
        verbose=1,
    ),
]

# --------------------------
# 4. 启动训练（使用增强生成器）
# --------------------------
print(f"开始训练{prefix}模型，共{EPOCHS}轮，批次大小{BATCH_SIZE}...")
history = model.fit(
    # 训练时使用数据增强生成器
    train_generator(X_train, y_train, BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,  # 每轮步数=训练样本数//批次大小
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=False,  # 生成器已内部打乱，此处关闭
)

# --------------------------
# 5. 训练结果可视化与输出
# --------------------------
# 设置中文字体（兼容多平台）
plt.rcParams["font.family"] = [
    "SimHei",
    "Heiti TC",
    "Microsoft YaHei",
]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 绘制准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="训练准确率", color="blue")
plt.plot(history.history["val_accuracy"], label="验证准确率", color="orange")
plt.title(f"{prefix}模型训练曲线（数据增强+学习率衰减）")
plt.xlabel("轮次（Epoch）")
plt.ylabel("准确率（Accuracy）")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
plt.close()

# 输出最佳验证准确率
best_val_acc = max(history.history["val_accuracy"])
print(f"\n训练完成！最佳验证准确率：{best_val_acc:.4f}")
print(f"最优模型已保存至：{model_save_path}")
print(f"训练日志已保存至：{log_save_path}")
print(f"准确率曲线已保存至：{fig_save_path}")
