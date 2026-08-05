import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, resample, filterbank
from braindecode.preprocessing import create_windows_from_events
from braindecode.classifier import EEGClassifier
from braindecode.models import ShallowFBCSPNet

# ---------------------- 参数配置 ----------------------
DATASET_NAME = "BNCI2014001"
SFREQ = 100
FREQ_BANDS = [[4, 30]]
TRIAL_START = 0.0
TRIAL_STOP = 4.0
N_CHANS = 22
N_CLASSES = 4
MAX_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- 加载原始连续数据集 ----------------------
raw_dataset = MOABBDataset(dataset_name=DATASET_NAME)
# group = 受试者ID，用于跨被试GroupKFold
groups = np.array([d.description["subject"].iloc[0] for d in raw_dataset.datasets])

# ---------------------- 定义单次fold训练函数 ----------------------
def train_single_fold(train_raw, test_raw):
    # 1. 预处理流水线（仅作用于训练集原始信号）
    preprocessors = [
        Preprocessor(resample, sfreq=SFREQ),
        Preprocessor(filterbank, frequency_ranges=FREQ_BANDS)
    ]
    train_raw = preprocess(train_raw, preprocessors)
    test_raw = preprocess(test_raw, preprocessors)

    # 2. fold内部在线切窗（关键：train/test独立切窗，无泄露）
    train_windows = create_windows_from_events(
        train_raw,
        trial_start_offset_seconds=TRIAL_START,
        trial_stop_offset_seconds=TRIAL_STOP
    )
    test_windows = create_windows_from_events(
        test_raw,
        trial_start_offset_seconds=TRIAL_START,
        trial_stop_offset_seconds=TRIAL_STOP
    )

    # 3. 构建模型 + EEGClassifier
    model = ShallowFBCSPNet(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=int((TRIAL_STOP - TRIAL_START) * SFREQ),
        final_conv_length="auto"
    )
    clf = EEGClassifier(
        module=model,
        optimizer=torch.optim.Adam,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        verbose=False,
        train_split=None,
        device=DEVICE
    )
    clf.fit(train_windows)

    y_pred = clf.predict(test_windows)
    y_true = test_windows.targets
    acc = accuracy_score(y_true, y_pred)
    return acc

# ---------------------- GroupKFold 跨被试交叉验证 ----------------------
gkf = GroupKFold(n_splits=4)
all_acc = []
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(raw_dataset, groups=groups)):
    print(f"\n==== Fold {fold_idx+1} ====")
    train_raw = raw_dataset.select(train_idx)
    test_raw = raw_dataset.select(test_idx)
    fold_acc = train_single_fold(train_raw, test_raw)
    all_acc.append(fold_acc)
    print(f"Fold Acc: {fold_acc:.4f}")

print(f"\n==== 最终结果 ====")
print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")