import os
import numpy as np
from tensorflow.keras.models import load_model
from config import *
from reverberation_reduction import reduce_high_freq_reverb

# 类别映射（UrbanSound8K）
ESC50_CLASS_NAMES = ["dog", "rooster", "pig", "cow", "frog", ...]  # 共50类


def predict(audio_path, model_path=None):
    """预测新音频的类别"""
    # 加载模型（默认使用最佳模型）
    if not model_path:
        model_path = os.path.join(RESULTS_MODEL_DIR, "esc50_best_model.h5")
    model = load_model(model_path)

    # 预处理音频
    feat = reduce_high_freq_reverb(audio_path)
    feat = np.expand_dims(feat, axis=0)  # 增加批次维度

    # 预测
    pred_probs = model.predict(feat)[0]
    pred_class = np.argmax(pred_probs)
    return {
        "class": ESC50_CLASS_NAMES[pred_class],
        "probability": float(pred_probs[pred_class]),
    }


# 示例：预测单个音频
if __name__ == "__main__":
    test_audio = os.path.join(DATA_RAW_DIR, "ESC-50", "audio", "1-137-A-32.wav")
    result = predict(test_audio)
    print(f"Prediction: {result['class']} (Probability: {result['probability']:.2f})")
