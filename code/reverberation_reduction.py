import numpy as np
import librosa
import os
from scipy.signal import butter, filtfilt
from config import *


def calculate_butterworth_params(f_h, r=R, B_t=1000, delta1=1, delta2=40):
    """计算巴特沃斯滤波器参数（论文公式2-5）"""
    f_l = 0
    B_i = f_h - f_l
    B_o = B_i * r
    f_p = f_h - B_o
    f_s = f_p - B_t
    f_c = f_p / (99**0.5)  # 截止频率
    # 计算滤波器阶数
    N = np.log10((10 ** (delta2 / 10) - 1) / (10 ** (delta1 / 10) - 1)) / (
        2 * np.log10(f_s / f_c)
    )
    return f_p, f_c, int(np.ceil(N))


def butterworth_highpass(spectrogram, f_c, N, fs=SAMPLE_RATE):
    """巴特沃斯高通滤波（分离高频段）"""
    nyq = 0.5 * fs
    f_c_norm = f_c / nyq
    b, a = butter(N, f_c_norm, btype="high", analog=False)
    return filtfilt(b, a, spectrogram, axis=1)  # 对频率维度滤波


def wpe_dereverberation(high_freq_spect, D=5):
    """WPE去混响（仅处理高频段）"""
    t_frames, f_bins = high_freq_spect.shape
    dereverb_spect = np.copy(high_freq_spect)
    g = 0.8 * np.ones(f_bins)  # 预测权重（简化实现）
    for t in range(D, t_frames):
        dereverb_spect[t] = high_freq_spect[t] - g * high_freq_spect[t - D]
    return dereverb_spect


def reduce_high_freq_reverb(audio_path, save_processed=False, save_name=None):
    """
    完整高频混响抑制流程：返回处理后的Log-Mel谱
    save_processed: 是否保存预处理结果
    save_name: 保存文件名（如"fold1_xxx.npy"）
    """
    # 1. 加载音频并转频域
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = np.abs(stft).T  # [时间帧, 频率点]

    # 2. 计算滤波器参数并分离高频段
    f_bins = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    f_h = f_bins[-1]
    _, f_c, N = calculate_butterworth_params(f_h)
    high_freq_spect = butterworth_highpass(spectrogram, f_c, N)

    # 3. 高频段去混响并合成频谱
    high_freq_dereverb = wpe_dereverberation(high_freq_spect)
    low_mid_freq_spect = spectrogram - high_freq_spect
    final_spect = low_mid_freq_spect + high_freq_dereverb

    # 4. 转换为Log-Mel谱
    mel_filter = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=MEL_BINS)
    mel_spect = np.dot(mel_filter, final_spect.T)  # [mel_bins, 时间帧]
    log_mel = librosa.power_to_db(mel_spect, ref=np.max).T  # [时间帧, mel_bins]

    # 5. 特征对齐（补0/截断）
    if log_mel.shape[0] < MAX_TIME_FRAMES:
        pad = np.zeros((MAX_TIME_FRAMES - log_mel.shape[0], MEL_BINS))
        log_mel = np.vstack([log_mel, pad])
    else:
        log_mel = log_mel[:MAX_TIME_FRAMES]

    # 6. 保存预处理结果（可选）
    if save_processed and save_name:
        save_path = os.path.join(DATA_PROCESSED_DIR, ESC50_PROCESSED_DIR)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, save_name), log_mel)

    return log_mel[..., np.newaxis]  # 增加通道维度：[T, F, 1]
