import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from config import *  # 导入全局配置
from dataset_loader import load_esc50  # 仅使用ESC-50数据集
from model import build_rf_drsn_ema

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
# 若模型输入形状与数据不匹配（如时间帧长度），可在此处调整
# 例如：若ESC-50时间帧为80，确保build_rf_drsn_ema()的input_shape为(80, 128, 1)

# 打印模型结构（确认输出层类别数正确）
model.summary()

# --------------------------
# 3. 配置训练回调
# --------------------------
# 结果保存路径（区分ESC-10和ESC-50）
prefix = "esc10" if USE_ESC10_SUBSET else "esc50"
model_save_path = os.path.join(RESULTS_MODEL_DIR, f"{prefix}_best_model.h5")
log_save_path = os.path.join(RESULTS_LOG_DIR, f"{prefix}_train.log")
fig_save_path = os.path.join(RESULTS_FIG_DIR, f"{prefix}_accuracy_curve.png")

# 回调函数：保存最优模型+早停+日志记录
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
        patience=20 if USE_ESC10_SUBSET else 50,  # ESC-10样本少，早停更激进
        mode="max",
        verbose=1,
    ),
    CSVLogger(log_save_path),  # 保存训练日志（loss、accuracy等）
]

# --------------------------
# 4. 启动训练
# --------------------------
print(f"开始训练{prefix}模型，共{EPOCHS}轮，批次大小{BATCH_SIZE}...")
history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=True,  # 每轮训练前打乱数据
)

# --------------------------
# 5. 训练结果可视化与输出
# --------------------------
# 绘制准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="训练准确率", color="blue")
plt.plot(history.history["val_accuracy"], label="验证准确率", color="orange")
plt.title(f"{prefix}模型训练曲线")
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
