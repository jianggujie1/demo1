import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from config import *
from reverberation_reduction import reduce_high_freq_reverb

def load_esc50(use_processed=True, use_esc10_subset=False):
    """
    加载ESC-50数据集
    use_esc10_subset: 若为True，只加载ESC-10子集（10类）；否则加载全量50类
    """
    from config import ESC50_RAW_DIR, ESC50_PROCESSED_DIR, ESC50_MAX_TIME_FRAMES, ESC50_NUM_CLASSES
    
    # 标签文件路径
    meta_path = os.path.join(ESC50_RAW_DIR, "meta", "esc50.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到ESC-50标签文件：{meta_path}")
    meta_df = pd.read_csv(meta_path)
    
    # 筛选ESC-10子集（若需要）
    if use_esc10_subset:
        meta_df = meta_df[meta_df["esc10"] == 1].reset_index(drop=True)
        print(f"加载ESC-10子集（10类），共{len(meta_df)}个样本")
    else:
        print(f"加载ESC-50全量（50类），共{len(meta_df)}个样本")
    
    features, labels = [], []
    processed_dir = ESC50_PROCESSED_DIR
    
    # 优先使用预处理结果
    if use_processed and os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
        print("加载预处理后的ESC-50特征...")
        for _, row in meta_df.iterrows():
            # 预处理文件名（用原音频文件名替换.wav为.npy）
            save_name = row["filename"].replace(".wav", ".npy")
            feat_path = os.path.join(processed_dir, save_name)
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"预处理文件缺失：{feat_path}")
            feat = np.load(feat_path)
            features.append(feat)
            # 标签编码（one-hot）
            label = to_categorical(row["target"], num_classes=ESC50_NUM_CLASSES if not use_esc10_subset else 10)
            labels.append(label)
        return np.array(features), np.array(labels)
    
    # 无预处理结果则重新生成
    print("处理ESC-50原始音频...")
    for _, row in meta_df.iterrows():
        audio_path = os.path.join(ESC50_RAW_DIR, "audio", row["filename"])
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件缺失：{audio_path}")
        
        # 调用混响抑制模块处理音频（注意使用ESC-50的时间帧参数）
        # 临时修改全局的MAX_TIME_FRAMES，适配ESC-50的5秒音频
        from config import MAX_TIME_FRAMES
        temp_max_t = MAX_TIME_FRAMES
        from config import MAX_TIME_FRAMES
        MAX_TIME_FRAMES = ESC50_MAX_TIME_FRAMES
        
        save_name = row["filename"].replace(".wav", ".npy")
        feat = reduce_high_freq_reverb(
            audio_path,
            save_processed=True,
            save_name=save_name  # 保存到ESC50_PROCESSED_DIR
        )
        features.append(feat)
        # 恢复全局参数
        MAX_TIME_FRAMES = temp_max_t
        
        # 标签编码
        label = to_categorical(row["target"], num_classes=ESC50_NUM_CLASSES if not use_esc10_subset else 10)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def load_urbansound8k(use_processed=True):
    """加载UrbanSound8K数据集（优先使用预处理结果）"""
    dataset_dir = os.path.join(DATA_RAW_DIR, "UrbanSound8K")
    meta_path = os.path.join(dataset_dir, "metadata", "UrbanSound8K.csv")
    meta_df = pd.read_csv(meta_path)
    features, labels = [], []

    # 检查是否有预处理结果
    processed_dir = os.path.join(DATA_PROCESSED_DIR, "urbansound8k_logmel")
    if use_processed and os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
        print("Loading processed UrbanSound8K features...")
        for _, row in meta_df.iterrows():
            save_name = f"{row['fold']}_{row['slice_file_name'].replace('.wav', '.npy')}"
            feat = np.load(os.path.join(processed_dir, save_name))
            features.append(feat)
            labels.append(to_categorical(row["classID"], num_classes=NUM_CLASSES))
        return np.array(features), np.array(labels)

    # 无预处理结果则重新生成
    print("Processing UrbanSound8K raw audio...")
    for _, row in meta_df.iterrows():
        audio_path = os.path.join(dataset_dir, "audio", f"fold{row['fold']}", row["slice_file_name"])
        # 处理音频并保存
        save_name = f"{row['fold']}_{row['slice_file_name'].replace('.wav', '.npy')}"
        feat = reduce_high_freq_reverb(audio_path, save_processed=True, save_name=save_name)
        features.append(feat)
        labels.append(to_categorical(row["classID"], num_classes=NUM_CLASSES))
    return np.array(features), np.array(labels)

# ESC-10/DCASE2020加载函数类似，仅需修改路径和标签处理逻辑
def load_esc10(use_processed=True):
    dataset_dir = os.path.join(DATA_RAW_DIR, "ESC-10")
    # 实现逻辑参考load_urbansound8k，筛选ESC-50中esc10=1的样本
    pass