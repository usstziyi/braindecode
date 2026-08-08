import numpy as np
import torch
import mne

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

train_dataset_run0_raw = train_dataset.datasets[0].raw

print(type(train_dataset_run0_raw)) # mne.io.array._array.RawArray
print(len(train_dataset_run0_raw)) # 96735个采样点=250hz * 386.94

# 此时events的sample_offset是原始数据的stim时刻的采样点索引
events = mne.find_events(train_dataset_run0_raw)
print(events.shape) # (48, 3)

import pandas as pd
print(pd.DataFrame(events, columns=["sample_offset", "duration", "event_id"]))


# 此时annotations的onset是原始数据的stim时刻+2s=3s
annotations = train_dataset_run0_raw.annotations
df = annotations.to_data_frame()
print(df)



