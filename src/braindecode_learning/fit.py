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
print("Loading dataset...")
dataset = MOABBDataset(dataset_name="BNCI2014_001")
# 根据session类别划分train、test
splits = dataset.split(by="session")
train_dataset = splits["0train"]
test_dataset = splits["1test"]
# 预处理
print("Preprocessing dataset...")
preprocessors = [
    PickTypes(eeg=True, verbose=False),
    Filter(l_freq=4, h_freq=40.0, verbose=False),
    Rescale(scalings=1e6, verbose=False),
    Resample(sfreq=128, verbose=False),
    Preprocessor(exponential_moving_standardize),
]
train_dataset = preprocess(train_dataset, preprocessors)
test_dataset = preprocess(test_dataset, preprocessors)
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

# 构建模型
print("Building model...")
model = EEGNet(
    n_chans=22,
    n_outputs=4,
    n_times=512,               # ← 128Hz × 4s = 512（原值256仅对应2s）
    sfreq=128,                 # ✅ 无需修改
    F1=8,                      # ← 显式指定
    D=2,                       # ← 显式指定
    F2=16,                     # ← F1×D = 8×2
    kernel_length=64,          # ← 0.5s @ 128Hz
    depthwise_kernel_length=16,# ← 深度卷积核长度
    pool1_kernel_size=4,       # ← 平均池化
    pool2_kernel_size=8,       # ← 平均池化
    drop_prob=0.5,             # ← 关键！不是默认的0.25
    norm_rate=0.25,            # ← MaxNorm约束
    conv_spatial_max_norm=1,   # ← 空间卷积MaxNorm
    final_conv_length='auto',  # ← 自动调整输出长度
)
# 构建classifier
print("Building classifier...")
callbacks = [
    ("checkpoint", Checkpoint(
        monitor="valid_acc_best",         # 监控最佳验证集准确率
        f_params="EEGNet/best_model.pt",  # 保存到 models 目录
        f_optimizer="EEGNet/optimizer.pt",  # 优化器状态
        f_criterion="EEGNet/criterion.pt",  # 损失函数状态
        f_history="EEGNet/history.json",  # 训练历史
    )),
]
classifier = EEGClassifier(
    model,
    optimizer=torch.optim.Adam,
    optimizer__lr=0.001,           # ← 显式指定优化器及学习率
    optimizer__betas=(0.9, 0.999), # ← 原论文Adam参数，optimizer__betas 是传递给 Adam 优化器的 两个动量衰减系数，即 (β₁, β₂)。它们控制 Adam 对历史梯度信息的"记忆长度"。
    criterion=torch.nn.CrossEntropyLoss,
    batch_size=64,                 # ✅ 与原论文一致
    max_epochs=20,                 # 原论文500轮
    callbacks=callbacks,           # 添加 checkpoint 回调
    device="mps",
)
# 提取 X, y
# WindowsDataset to numpy array
def dataset_to_xy(dataset):
    X, y = [], []
    for x_i, y_i, _ in dataset:
        X.append(x_i)
        y.append(y_i)
    return np.array(X), np.array(y)

# 训练模型
print("Training model...")
X_train, y_train = dataset_to_xy(train_windows)
classifier.fit(X_train, y_train)

# 评估模型
print("Evaluating model...")
X_test, y_test = dataset_to_xy(test_windows)
y_pred = classifier.predict(X_test)

# 历史记录
print("History:")
# history 只记录 fit() 训练过程 中的指标。 predict() 是纯推理操作，不产生任何历史记录。
# skorch history 用 [epoch, "key"] 或 [:, "key"] 索引
epochs = classifier.history[:, "epoch"]  # 从1开始
train_loss = classifier.history[:, "train_loss"]
val_loss = classifier.history[:, "valid_loss"]
val_acc = classifier.history[:, "valid_acc"]
print("epochs:", epochs)
# print("train_loss:", train_loss)
# print("val_loss:", val_loss)
# print("val_acc:", val_acc)

# 分类报告
print("Classification Report:")
report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1", "Class 2", "Class 3"])
print(report)


# 混淆矩阵
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)