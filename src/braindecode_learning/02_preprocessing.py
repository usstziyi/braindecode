"""
Braindecode 学习教程 - 第02章: 数据预处理
======================================================

本教程学习如何使用 braindecode 预处理 EEG 数据。

知识点:
1. 数据滤波 (带通滤波、陷波滤波)
2. 重采样
3. 数据标准化 (指数移动平均)
4. 滑窗分割
5. 应用多个预处理步骤

参考: https://braindecode.org/stable/index.html#braindecodepreprocessing
"""

import numpy as np
import mne
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
)


# ============================================================
# 1. 数据滤波
# ============================================================

def tutorial_filtering():
    """
    滤波: 去除噪声和伪迹
    
    常用滤波器:
    - 带通滤波 (bandpass): 保留特定频段 (如 0.5-40 Hz)
    - 陷波滤波 (notch): 去除工频干扰 (如 50 Hz / 60 Hz)
    - 低通滤波 (lowpass): 去除高频噪声
    - 高通滤波 (highpass): 去除基线漂移
    
    Braindecode 预处理基于 MNE-Python:
    - raw.filter(l_freq, h_freq) 进行滤波
    - raw.notch_filter(freqs) 去除工频
    """
    print("=" * 60)
    print("教程 2.1: 数据滤波")
    print("=" * 60)
    
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    
    # 获取原始数据
    raw = dataset.datasets[0].raw.copy()
    
    print(f"\n滤波前:")
    print(f"  - 数据范围: {raw.get_data().min():.4f} ~ {raw.get_data().max():.4f}")
    print(f"  - 数据形状: {raw.get_data().shape}")
    
    # 方法1: 手动滤波
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    
    print(f"\n带通滤波后 (0.5-40 Hz):")
    print(f"  - 数据范围: {raw_filtered.get_data().min():.4f} ~ {raw_filtered.get_data().max():.4f}")
    
    # 方法2: 使用 braindecode Preprocessor
    # Preprocessor 可以链式应用多个预处理步骤
    from braindecode.preprocessing import Preprocessor
    
    raw_preprocessed = raw.copy()
    
    # 定义预处理步骤
    preprocessors = [
        Preprocessor("filter", l_freq=0.5, h_freq=40.0),
    ]
    
    # 应用预处理
    for prep in preprocessors:
        raw_preprocessed = prep.apply(raw_preprocessed)
    
    print(f"\n使用 Preprocessor 滤波后:")
    print(f"  - 数据范围: {raw_preprocessed.get_data().min():.4f} ~ {raw_preprocessed.get_data().max():.4f}")
    
    # 陷波滤波 (去除工频)
    raw_notch = raw.copy()
    raw_notch.notch_filter(freqs=[50], verbose=False)
    
    print(f"\n陷波滤波后 (50 Hz):")
    print(f"  - 数据范围: {raw_notch.get_data().min():.4f} ~ {raw_notch.get_data().max():.4f}")


# ============================================================
# 2. 重采样
# ============================================================

def tutorial_resampling():
    """
    重采样: 改变数据的采样率
    
    - 降采样 (downsample): 降低采样率, 减少数据量
    - 升采样 (resample): 提高采样率 (通常不需要)
    
    注意:
    - 重采样需要抗混叠滤波
    - 会影响数据的时间分辨率
    - 对深度学习来说, 256 Hz 通常足够
    """
    print("\n" + "=" * 60)
    print("教程 2.2: 重采样")
    print("=" * 60)
    
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    raw = dataset.datasets[0].raw.copy()
    
    print(f"\n原始采样率: {raw.info['sfreq']} Hz")
    print(f"原始数据点数: {raw.n_times}")
    
    # 降采样到 128 Hz
    raw_resampled = raw.copy()
    raw_resampled.resample(128, verbose=False)
    
    print(f"\n降采样后 (128 Hz):")
    print(f"  - 采样率: {raw_resampled.info['sfreq']} Hz")
    print(f"  - 数据点数: {raw_resampled.n_times}")
    
    # 降采样到 64 Hz
    raw_64hz = raw.copy()
    raw_64hz.resample(64, verbose=False)
    
    print(f"\n降采样后 (64 Hz):")
    print(f"  - 采样率: {raw_64hz.info['sfreq']} Hz")
    print(f"  - 数据点数: {raw_64hz.n_times}")


# ============================================================
# 3. 数据标准化
# ============================================================

def tutorial_standardization():
    """
    数据标准化: 将数据变换到标准尺度
    
    方法:
    1. 经典标准化: (x - mean) / std
    2. 指数移动平均 (EMA) 标准化: exponential_moving_standardize
       - 适合在线学习场景
       - 对非平稳 EEG 信号更鲁棒
       - 公式: x_norm = (x - ema_mean) / ema_std
    """
    print("\n" + "=" * 60)
    print("教程 2.3: 数据标准化")
    print("=" * 60)
    
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    raw = dataset.datasets[0].raw.copy()
    
    # 先进行滤波和重采样
    raw_prep = raw.copy()
    raw_prep.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    raw_prep.resample(128, verbose=False)
    
    data = raw_prep.get_data()
    
    print(f"\n标准化前:")
    print(f"  - 均值: {data.mean():.6f}")
    print(f"  - 标准差: {data.std():.6f}")
    print(f"  - 最小值: {data.min():.6f}")
    print(f"  - 最大值: {data.max():.6f}")
    
    # 方法1: 经典 z-score 标准化
    mean = data.mean()
    std = data.std()
    data_zscore = (data - mean) / std
    
    print(f"\nZ-Score 标准化后:")
    print(f"  - 均值: {data_zscore.mean():.6f}")
    print(f"  - 标准差: {data_zscore.std():.6f}")
    print(f"  - 最小值: {data_zscore.min():.6f}")
    print(f"  - 最大值: {data_zscore.max():.6f}")
    
    # 方法2: 指数移动平均标准化
    # exponential_moving_standardize 应用到 numpy 数组
    data_ema = exponential_moving_standardize(
        data, factor_new=0.001, init_block_size=None
    )
    
    print(f"\nEMA 标准化后:")
    print(f"  - 均值: {data_ema.mean():.6f}")
    print(f"  - 标准差: {data_ema.std():.6f}")


# ============================================================
# 4. 完整预处理流水线
# ============================================================

def tutorial_preprocess_pipeline():
    """
    完整预处理流水线: 从原始数据到训练就绪数据
    
    典型流程:
    1. 加载数据
    2. 滤波 (带通 + 陷波)
    3. 重采样
    4. 窗口化
    5. 标准化 (可选, 通常在训练时做)
    
    使用 preprocess() 函数批量处理整个数据集
    """
    print("\n" + "=" * 60)
    print("教程 2.4: 完整预处理流水线")
    print("=" * 60)
    
    # 加载数据
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    
    print(f"\n原始数据集大小: {len(dataset.datasets)} 个记录 (run)")
    
    # 定义预处理步骤
    # preprocess() 会对所有记录 (run) 应用相同的预处理
    preprocessors = [
        # 1. 带通滤波: 0.5-40 Hz
        Preprocessor("filter", l_freq=0.5, h_freq=40.0),
        # 2. 陷波滤波: 去除 50 Hz 工频
        Preprocessor("notch_filter", freqs=[50]),
        # 3. 重采样到 128 Hz
        Preprocessor("resample", sfreq=128),
        # 4. EMA 标准化 (使用 numpy 数组上的指数移动平均)
        Preprocessor(exponential_moving_standardize),
    ]
    
    # 应用预处理
    print("\n应用预处理步骤...")
    dataset_preprocessed = preprocess(dataset, preprocessors)
    
    print(f"预处理完成!")
    print(f"  - 数据集大小: {len(dataset_preprocessed.datasets)} 个记录 (run)")
    
    # 窗口化
    windows_dataset = create_windows_from_events(
        dataset_preprocessed,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=512,      # 4 秒 @ 128 Hz
        window_stride_samples=128,     # 1 秒步长 @ 128 Hz
        preload=True,
    )
    
    print(f"\n窗口化完成!")
    print(f"  - 窗口数量: {len(windows_dataset)}")
    
    # 查看样本
    sample = windows_dataset[0]
    if isinstance(sample, (list, tuple)):
        X, y = sample[0], sample[1]
        print(f"  - 输入形状: {X.shape}  [channels, time]")
        print(f"  - 标签: {y}")
    
    return dataset_preprocessed, windows_dataset


# ============================================================
# 5. 预处理选项详解
# ============================================================

def tutorial_preprocessor_options():
    """
    Preprocessor 选项详解
    
    常用预处理操作:
    - "filter": 带通/低通/高通滤波
      参数: l_freq (低截止), h_freq (高截止)
    - "notch_filter": 陷波滤波
      参数: freqs (陷波频率列表)
    - "resample": 重采样
      参数: sfreq (目标采样率)
    - "exponential_moving_standardize": EMA 标准化 (通过 Preprocessor 调用)
      参数: 无
    - "set_channel_types": 设置通道类型
      参数: mapping (通道类型映射)
    - "pick": 选择通道
      参数: picks (通道名列表或选择规则)
    - "drop_channels": 丢弃通道
      参数: ch_names (要丢弃的通道名)
    """
    print("\n" + "=" * 60)
    print("教程 2.5: Preprocessor 选项详解")
    print("=" * 60)
    
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    
    raw = dataset.datasets[0].raw.copy()
    print(f"\n原始通道数: {len(raw.ch_names)}")
    print(f"原始通道: {raw.ch_names}")
    
    # 选择特定通道
    preprocessors = [
        # 只保留 EEG 通道
        Preprocessor("pick", picks="eeg"),
    ]
    
    for prep in preprocessors:
        raw = prep.apply(raw)
    
    print(f"\n只保留 EEG 通道后:")
    print(f"  - 通道数: {len(raw.ch_names)}")
    print(f"  - 通道: {raw.ch_names}")
    
    # 丢弃特定通道 (通道名 "STI" 在不同数据集/版本中可能不同, 动态检测)
    raw_2 = dataset.datasets[0].raw.copy()
    stim_channels = mne.utils._get_stim_channel(None, raw_2.info, raise_error=False)
    if stim_channels:
        preprocessors_2 = [
            Preprocessor("drop_channels", ch_names=[stim_channels[0]]),
        ]
        for prep in preprocessors_2:
            raw_2 = prep.apply(raw_2)
        print(f"\n丢弃刺激通道 ({stim_channels[0]}) 后:")
    else:
        print(f"\n无刺激通道可丢弃")
    print(f"  - 通道数: {len(raw_2.ch_names)}")


if __name__ == "__main__":
    tutorial_filtering()
    tutorial_resampling()
    tutorial_standardization()
    tutorial_preprocess_pipeline()
    tutorial_preprocessor_options()
    
    print("\n" + "=" * 60)
    print("🎉 第02章完成! 你已经学会了:")
    print("  ✅ 数据滤波 (带通、陷波)")
    print("  ✅ 重采样")
    print("  ✅ 数据标准化 (Z-Score, EMA)")
    print("  ✅ 完整预处理流水线")
    print("  ✅ Preprocessor 选项")
    print("\n进入 03_models_basics.py 学习模型!")
    print("=" * 60)
