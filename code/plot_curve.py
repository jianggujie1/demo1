import matplotlib.pyplot as plt
import pandas as pd

# --- 1. 设置中文字体（解决中文显示为方框的问题）---
plt.rcParams["font.family"] = [
    "SimHei",
    "Microsoft YaHei",
]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方框的问题

# --- 2. 定义文件路径 ---
# 你的训练日志路径
log_path = "results/logs/esc50_train.log"
# 图片保存路径
fig_save_path = "results/figures/esc50_accuracy_curve.png"

# --- 3. 使用pandas加载CSV日志文件 ---
# pandas可以自动识别CSV的表头（epoch, accuracy, ...）
df = pd.read_csv(log_path)
print("训练日志数据预览：")
print(df.head())  # 打印前5行数据预览

# --- 4. 绘制准确率曲线 ---
plt.figure(figsize=(10, 6))  # 设置图片大小

# 绘制训练准确率曲线
plt.plot(df["epoch"], df["accuracy"], label="训练准确率")

# 绘制验证准确率曲线
plt.plot(df["epoch"], df["val_accuracy"], label="验证准确率")

# --- 5. 美化图表 ---
plt.title("训练与验证准确率曲线")
plt.xlabel("训练轮次 (Epoch)")
plt.ylabel("准确率 (Accuracy)")
plt.legend()  # 显示图例
# plt.grid(True, linestyle=":", alpha=0.6)  # 添加网格线，增强可读性
plt.grid(alpha=0.3)

# --- 6. 保存图片 ---
plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
print(f"准确率曲线图片已成功保存到: {fig_save_path}")
plt.close()

# 如果想在运行后直接显示图片，可以取消下面这行的注释
# plt.show()
