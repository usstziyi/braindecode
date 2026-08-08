import numpy as np
import torch
from torch.utils.data import DataLoader
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    preprocess,
    Filter,
    Resample,
    PickTypes,
    Rescale,
    Preprocessor,
    exponential_moving_standardize,
    create_windows_from_events,
)
from braindecode.models import EEGNet, ShallowFBCSPNet
from braindecode import EEGClassifier, EEGRegressor
from skorch.callbacks import EpochScoring, Checkpoint
from skorch.helper import predefined_split
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import (
    accuracy_score,           # 准确率: 正确预测数 / 总样本数
    balanced_accuracy_score,  # 平衡准确率: 各类别准确率的平均值, 处理类别不平衡
    precision_score,          # 精确率: 预测为正的样本中实际为正的比例
    recall_score,             # 召回率: 实际为正的样本中被正确预测为正的比例
    f1_score,                 # F1分数: 精确率和召回率的调和平均, 综合评估指标
    confusion_matrix,         # 混淆矩阵: 展示各类别预测与真实标签的对应关系
    classification_report,    # 分类报告: 生成包含精确率/召回率/F1的详细文本报告
)

# 准备数据集
print("*"*50)
print("Loading dataset...")
print("*"*50)
dataset = MOABBDataset(dataset_name="BNCI2014_001")
print(f"Total runs: {len(dataset.datasets)}")
# 根据session类别划分train、test
splits = dataset.split(by="session")
train_dataset = splits["0train"]
test_dataset = splits["1test"]

print(f"Train dataset runs: {len(train_dataset.datasets)}") # 54个run
print(f"Test dataset runs: {len(test_dataset.datasets)}") # 54个run


# 预处理
print("*"*50)
print("Preprocessing dataset...")
print("*"*50)
preprocessors = [
    PickTypes(eeg=True, verbose=False),
    Filter(l_freq=4, h_freq=40.0, verbose=False),
    Rescale(scalings=1e6, verbose=False),
    Resample(sfreq=128, verbose=False),
    Preprocessor(exponential_moving_standardize),
]
train_dataset = preprocess(train_dataset, preprocessors)
test_dataset = preprocess(test_dataset, preprocessors)
# 创建窗口
print("*"*50)
print("Creating windows...")
print("*"*50)
train_windows = create_windows_from_events(
    train_dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=512,
    window_stride_samples=512,
    preload=True,
)
test_windows = create_windows_from_events(
    test_dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=512,
    window_stride_samples=512,
    preload=True,
)

"""
train_dataset:BaseConcatDataset[RawDataset]:每个 run 一个 RawDataset
train_windows:BaseConcatDataset[EEGWindowsDataset]:每个 run 一个 EEGWindowsDataset
"""

print(f"Train windows runs: {len(train_windows.datasets)}") # 54个run
print(f"Test windows runs: {len(test_windows.datasets)}") # 54个run


"""
train_dataset.datasets[i]:第 i 个 run 的 数据 (以时间点为单位)
train_windows.datasets[i]:第 i 个 run 的 数据 (以窗口为单位)
"""

train_dataset_run0 = train_dataset.datasets[0] # RawDataset：第一个run
train_windows_run0 = train_windows.datasets[0] # EEGWindowsDataset：第一个run

print(type(train_dataset_run0)) # RawDataset
print(type(train_windows_run0)) # EEGWindowsDataset

print(len(train_dataset_run0)) # 128,每个run有49528个时间点，按照128hz采样率计算，总时长为49528/128=386秒
print(len(train_windows_run0)) # 48,每个run划分出48个窗口，每个窗口有512个时间点

train_dataset_run0_time0 = train_dataset_run0[0] # 第一个最小单位(时间点)
train_windows_run0_window0 = train_windows_run0[0] # 第一个最小单位(窗口)

print(type(train_dataset_run0_time0)) # tuple
print(type(train_windows_run0_window0)) # tuple

print(len(train_dataset_run0_time0)) # 2,(X, y)
print(len(train_windows_run0_window0)) # 3,(X, y, crop_inds)

# tuple第一个元素:X
print(train_dataset_run0_time0[0].shape) # (22, 1):run0,time0,X:一个时间点的 EEG 数据
print(train_windows_run0_window0[0].shape) # (22, 512):run0,window0,X:一个窗口的 EEG 数据

# tuple第二个元素:y
print(train_dataset_run0_time0[1]) # None,run0,time0,y:None
print(train_windows_run0_window0[1]) # 3,run0,window0,y:类标签

# tuple第三个元素:crop_inds[i_window_in_trial,i_start_in_trial,i_stop_in_trial]
print(train_windows_run0_window0[2]) # [0, 384, 896],run0,window0,crop_inds:窗口在trial中的索引范围 

# 获取raw
train_dataset_run0_raw = train_dataset.datasets[0].raw
train_windows_run0_raw = train_windows.datasets[0].raw

print(type(train_dataset_run0_raw)) # mne.io.array._array.RawArray
print(type(train_windows_run0_raw)) # mne.io.array._array.RawArray

# 因为 train_dataset.datasets[0].raw 和 train_windows.datasets[0].raw 指向的是同一个 mne.raw 对象 。
# EGWindowsDataset 只存窗口元数据（起始/结束位置），真正取数据时再从 raw 按 slice 读取
print(id(train_dataset_run0_raw)) # 4753455632
print(id(train_windows_run0_raw)) # 4753455632

print(train_dataset_run0_raw.get_data().shape) # (22, 49528)
print(train_windows_run0_raw.get_data().shape) # (22, 49528)

print(train_dataset_run0_raw.n_times) # 49528
print(train_windows_run0_raw.n_times) # 49528

print(train_dataset_run0_raw.ch_names) # ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
print(train_windows_run0_raw.ch_names) # ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']


"""
train_dataset[i]:第 i 个 时间点 的 数据,tuple:(X, y)
train_windows[i]:第 i 个 窗口 的 数据,tuple:(X, y, crop_inds)
"""

print(type(train_dataset[0])) # tuple
print(type(train_windows[0])) # tuple

print(len(train_dataset[0])) # 2,run0的第一个最小单位(时间点)的tuple:(X, y)
print(len(train_windows[0])) # 3,run0的第一个最小单位(窗口)的tuple:(X, y, crop_inds)


"""
train_dataset:所有run的时间点总数=49528*54=2674512
train_windows:所有run的窗口总数=48*512*54=2592
"""
print(len(train_dataset)) # 2674512=54*49528:所有run的时间点总数
print(len(train_windows)) # 2592=54*48*512:所有run的窗口总数
