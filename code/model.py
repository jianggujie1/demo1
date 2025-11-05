import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    GlobalAveragePooling2D,
    Dense,
    Multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from config import *


def rf_self_calibration_block(input_tensor, nf=32, l2_reg=1e-4):
    """RF自校正模块（论文图3b）"""
    # 短距离残差
    short_res = Conv2D(nf // 2, (3, 3), padding="same", kernel_regularizer=l2(l2_reg))(
        input_tensor
    )
    short_res = BatchNormalization()(short_res)
    short_res = Activation("relu")(short_res)

    # FSC频域自校正
    # 只对频率维度（axis=2）做全局池化，保留时间维度（axis=1）
    fsc = tf.reduce_mean(short_res, axis=2, keepdims=True)  # 输出：(None, T, 1, nf//2)
    # fsc = GlobalAveragePooling2D()(short_res)  # 压缩频率维度
    fsc = Conv2D(nf // 2, (1, 1), padding="same", kernel_regularizer=l2(l2_reg))(fsc)
    fsc = Activation("relu")(fsc)
    fsc = Conv2D(short_res.shape[-1], (1, 1), padding="same", activation="sigmoid")(fsc)
    short_res = Multiply()([short_res, fsc])
    short_res = Conv2D(nf, (3, 3), padding="same", kernel_regularizer=l2(l2_reg))(
        short_res
    )

    # 长距离残差
    long_res = Conv2D(nf, (3, 3), padding="same", kernel_regularizer=l2(l2_reg))(
        input_tensor
    )
    long_res = BatchNormalization()(long_res)
    long_res = Activation("relu")(long_res)

    return Activation("relu")(BatchNormalization()(Add()([short_res, long_res])))


def ema_attention_block(input_tensor, l2_reg=1e-4):
    """EMA多尺度注意力模块（论文图4）"""
    C = input_tensor.shape[-1]
    # 1×1支路（时域+频域池化）
    branch1 = Conv2D(C // 2, (1, 1), padding="same", kernel_regularizer=l2(l2_reg))(
        input_tensor
    )
    # time_pool = GlobalAveragePooling2D(axis=1)(branch1)[:, tf.newaxis, :]  # 补频率维度
    # freq_pool = GlobalAveragePooling2D(axis=2)(branch1)[tf.newaxis, :, :]  # 补时间维度
    # 替换GlobalAveragePooling2D(axis=1)：对时间维度（axis=1）池化
    time_pool = tf.reduce_mean(
        branch1, axis=1, keepdims=True
    )  # 输出：(None, 1, F, C//2)
    # 替换GlobalAveragePooling2D(axis=2)：对频率维度（axis=2）池化
    freq_pool = tf.reduce_mean(
        branch1, axis=2, keepdims=True
    )  # 输出：(None, T, 1, C//2)

    branch1 = Add()([time_pool, freq_pool])
    branch1 = Conv2D(C, (1, 1), padding="same", kernel_regularizer=l2(l2_reg))(branch1)

    # 3×3支路
    branch2 = Conv2D(C // 2, (3, 3), padding="same", kernel_regularizer=l2(l2_reg))(
        input_tensor
    )
    branch2 = Conv2D(C, (3, 3), padding="same", kernel_regularizer=l2(l2_reg))(branch2)

    # 注意力融合
    ema_out = Activation("sigmoid")(Add()([branch1, branch2]))
    return Multiply()([input_tensor, ema_out])


def residual_shrinkage_unit(input_tensor, nf=32):
    # 获取输入张量的通道数（关键：用输入通道数作为输出通道数）
    input_channels = input_tensor.shape[-1]
    """残差收缩单元（RF+软阈值+EMA）"""
    x = rf_self_calibration_block(input_tensor, nf)
    # 软阈值化（简化实现）
    threshold = 0.1
    x = Activation(
        lambda t: tf.where(
            t > threshold, t - threshold, tf.where(t < -threshold, t + threshold, 0)
        )
    )(x)
    x = ema_attention_block(x)
    # 确保x的通道数与input_tensor一致（关键修复）
    if x.shape[-1] != input_channels:
        x = Conv2D(input_channels, (1, 1), padding="same")(x)

    return Add()([x, input_tensor])


def build_rf_drsn_ema():
    """构建完整RF-DRSN-EMA模型"""
    inputs = Input(shape=(MAX_TIME_FRAMES, MEL_BINS, 1))  # [T, F, 1]

    # 初始卷积
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # 堆叠4个残差收缩单元
    for _ in range(4):
        x = residual_shrinkage_unit(x, nf=32)

    # 分类头
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
