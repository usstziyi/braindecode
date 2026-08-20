"""小 demo：create_windows_from_events 返回的数据集，其 __getitem__ 到底返回什么。

实测（braindecode 1.7.0）：
  - create_windows_from_events 返回 BaseConcatDataset（内部持有若干 EEGWindowsDataset；
    仅当用到 reject/picks/flat/drop_bad_windows 等依赖 mne.Epochs 的功能时，内部才是 WindowsDataset）
  - train_windows[i] 返回 3 元组 (X, y, crop_inds)
      X         (n_chans, n_times) float32
      y         标量类别（来自 metadata 的 target 列）
      crop_inds [i_window_in_trial, i_start, i_stop] 窗口在原始 trial 中的位置
  - skorch 的 EEGClassifier 通过 ThrowAwayIndexLoader 丢掉 crop_inds，
    所以 fit 时每个 batch 才是 (X, y)，也因此可以省略 y。

用法：
    uv run python windows_dataset_demo.py
"""

from __future__ import annotations

import torch

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Filter,
    PickTypes,
    Preprocessor,
    Resample,
    Rescale,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)


def main():
    # 1) 加载单被试数据（会下载，已缓存则很快）
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

    # 2) 最简预处理：保留 EEG、V→µV、4–38 Hz 带通、重采样到 128 Hz、指数移动标准化
    preprocess(
        dataset,
        [
            PickTypes(eeg=True, stim=False, verbose=False),
            Rescale(scalings={"eeg": 1e6}),  # V→µV（mne.io.Raw.rescale 包装，可序列化）
            Filter(l_freq=4.0, h_freq=38.0, verbose=False),
            Resample(sfreq=128, verbose=False),
            Preprocessor(
                exponential_moving_standardize,
                factor_new=1e-3,
                init_block_size=1000,
            ),
        ],
    )

    # 3) 切窗 → BaseConcatDataset（默认路径内部元素是 EEGWindowsDataset）
    train_windows = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=512,
        window_stride_samples=512,
        preload=True,
    )
    print(f"窗口总数: {len(train_windows)}")
    print(f"类型: {type(train_windows).__name__}")

    # 4) 直接索引 → 3 元组 (X, y, crop_inds)
    X, y, crop_inds = train_windows[0]
    print(f"\n[索引 train_windows[0]] → (X, y, crop_inds)")
    print(f"  X         类型: {type(X).__name__}, 形状: {tuple(X.shape)}  (n_chans, n_times)")
    print(f"  y         类型: {type(y).__name__}, 值: {y}")
    print(f"  crop_inds 含义 [trial内序号, 起始, 结束]: {crop_inds}")

    # 5) 与 metadata 里的 target 对照（y 的来源）
    meta = train_windows.get_metadata()
    print(f"\n[与 metadata 对照]")
    print(f"  metadata['target'][0] = {meta['target'].iloc[0]}  → 与 y 一致: {meta['target'].iloc[0] == y}")

    # 6) 直接用 torch DataLoader 迭代：默认也产出 3 元组
    loader = torch.utils.data.DataLoader(train_windows, batch_size=16, shuffle=False)
    batch = next(iter(loader))
    X_batch, y_batch, crop_batch = batch
    print(f"\n[DataLoader 一个 batch] → (X_batch, y_batch, crop_batch)")
    print(f"  X_batch 形状: {tuple(X_batch.shape)}  → (batch, n_chans, n_times)")
    print(f"  y_batch 形状: {tuple(y_batch.shape)}, 类别: {y_batch.tolist()}")

    # 7) 手动复现 skorch 的做法：丢掉 crop_inds 得到 (X, y)
    X, y = X_batch, y_batch
    print(f"\n[丢掉 crop_inds 后] → 每 batch 就是 (X, y)，可直接喂给模型/算损失")
    print(f"  X 形状 {tuple(X.shape)}, y 形状 {tuple(y.shape)}")

    # 8) 结论：标签就在数据里（crop_inds 只是辅助信息），
    #    skorch 的 EEGClassifier 自动丢弃 crop_inds → clf.fit(train_windows, y=None)
    print(f"\n结论: 数据集自带 (X, y)，skorch 自动丢掉 crop_inds → clf.fit(train_windows, y=None) 无需再传 y")


if __name__ == "__main__":
    main()
