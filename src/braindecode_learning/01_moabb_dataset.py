"""
Braindecode 学习教程 - 第01章: MOABB 数据集加载与使用
======================================================

本教程学习如何使用 braindecode 的数据集模块加载和处理 EEG 数据。

知识点:
1. MOABB (Mother of All BCI Benchmarks) 是什么
2. 使用 MOABBDataset 加载数据集
3. 数据集的基本信息检查
4. 数据窗口化 (EEGWindowsDataset)
5. 数据加载器 (DataLoader)

参考: https://braindecode.org/stable/index.html#braindecodedatasets
"""

import numpy as np
import mne
import torch
from braindecode.datasets import MOABBDataset, BaseConcatDataset
from braindecode.preprocessing import create_windows_from_events

# ============================================================
# 1. MOABB 数据集加载
# ============================================================

def tutorial_moabb_basic():
    """
    基础: 加载 BCI Competition IV Dataset 2a
    
    这是运动想象 (Motor Imagery) 分类任务的经典数据集:
    - 22 通道 EEG
    - 4 类: 左手、右手、双脚、舌头
    - 9 名被试
    - 采样率 250 Hz
    """
    print("=" * 60)
    print("教程 1.1: 加载 MOABB 数据集")
    print("=" * 60)
    
    # 加载 BCI IV 2a 数据集
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    
    print(f"\n数据集类型: {type(dataset)}")
    # 每个记录 (run) 的 RawDataset 的长度累计和
    # 9 * 2 * 6 = 108 （run）
    # 108 * 250 * 386.94 = 10447380 (时间点数)
    print(f"数据集大小: {len(dataset.datasets)} 个记录 (run) (共 {len(dataset)} 个时间点)")
    print(f"\n数据集信息:")
    # 打印数据集元信息
    # print(dataset.description)  


def tutorial_data_inspection():
    """
    数据检查: 查看原始数据的基本属性
    
    每个 RawDataset 包含:
    - raw: MNE Raw 对象 (原始 EEG 数据)
    - description: 元信息 (被试ID、会话等)
    """
    print("\n" + "=" * 60)
    print("教程 1.2: 数据检查")
    print("=" * 60)
    
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    
    # 查看第一个记录 (run) 的数据
    run_0 = dataset.datasets[0]
    print(f"\n记录 0 类型: {type(run_0)}")
    print(f"记录 0 描述: {run_0.description}")
    
    # 获取 MNE Raw 对象
    raw = run_0.raw
    print(f"\n原始数据信息:")
    print(f"  - 通道数: {len(raw.ch_names)}")
    print(f"  - 通道名: {raw.ch_names[:5]}...")
    print(f"  - 采样率: {raw.info['sfreq']} Hz")
    print(f"  - 时长: {raw.n_times / raw.info['sfreq']:.1f} 秒")
    print(f"  - 通道类型: {raw.get_channel_types()[:5]}...")


def tutorial_windowing():
    """
    数据窗口化: 将连续 EEG 数据切分为固定长度的窗口
    
    窗口化是深度学习训练前的关键步骤:
    - 每个窗口包含固定长度的 EEG 片段
    - 窗口对应一个标签 (如运动想象类别)
    - 用于后续的 DataLoader 批量加载
    
    create_windows_from_events() 参数:
    - trial_start_offset_samples: 相对于事件开始的偏移
    - window_size_samples: 窗口大小 (采样点数)
    - window_stride_samples: 窗口步长 (采样点数)
    - drop_last_window: 是否丢弃不完整的最后一个窗口
    """
    print("\n" + "=" * 60)
    print("教程 1.3: 数据窗口化")
    print("=" * 60)
    
    # 加载数据
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    
    # 查看事件信息
    # 取第一个记录 (run) 的数据
    raw = dataset.datasets[0].raw
    """
        events 数组:
                列0    列1    列2
                [位置]  [前值]  [event_id]
        行0  →   [3,      0,      1    ]   ← event_id = 1
        行1  →   [6,      1,      2    ]   ← event_id = 2  
        行2  →   [9,      2,      1    ]   ← event_id = 1
        events[:, 2] → [1, 2, 1]   # 取出第2列的所有值
    """
    # stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
    # # 选择第一个刺激通道，如果没有刺激通道则设为 None，用于后续事件检测
    # stim_channel = stim_channels[0] if stim_channels else None
    # if stim_channel:
    #     print(f"  - 刺激通道: {stim_channel}")
    #     # events shape(事件数量, 3)
    #     events = mne.find_events(raw, stim_channel=stim_channel)
    #     print(f"\n事件数量: {len(events)}")
    #     print(f"事件类型: {np.unique(events[:, 2])}") # 选取最后一列，即事件类型
    # else:
    #     print("  - 无刺激通道，事件存储在 annotations 中")
    #     events, event_id = mne.events_from_annotations(raw)
    #     print(f"  - 事件ID映射: {event_id}")
    #     print(f"\n事件数量: {len(events)}")
    #     print(f"事件类型: {np.unique(events[:, 2])}") # 选取最后一列，即事件类型
    

    """
    时间轴:
    0s        2s              6s        8s
    ┌─────────┬───────────────┬─────────┐
    │ 提示期   │   想象期       │  休息期  │
    │  (cue)  │ (imagination) │  (rest) │
    └─────────┴───────────────┴─────────┘
        ↑
    stim 上升沿
    (事件触发点)
    """
    # 打印 annotations 信息
    ann = raw.annotations
    print(f"\n{'='*60}")
    print(f"  Annotations 信息 (共 {len(ann)} 个事件)")
    print(f"{'='*60}")
    print(f"  {'#':<4} {'Description':<20} {'Onset (s)':<12} {'Duration (s)':<12}")
    print(f"  {'─'*4} {'─'*20} {'─'*12} {'─'*12}")
    for i, a in enumerate(ann):
        print(f"  {i:<4} {a['description']:<20} {a['onset']:<12.2f} {a['duration']:<12.2f}")
    print(f"{'='*60}")
 
    # 创建窗口数据集
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,      # 这两个参数是 braindecode 自己的 ，用来 微调 从 annotations 获取的边界
        trial_stop_offset_samples=0,       # annotations 是moabb在raw上附加的切片
        window_size_samples=1000,          # 窗口大小: 1000 采样点 = 4秒
        window_stride_samples=1000,        # 步长: 1000 采样点 = 1步
        mapping=None,                      # 标签映射 (可选)
        preload=True,
    )
    
    print(f"\n窗口数据集类型: {type(windows_dataset)}")
    print(f"窗口数据集大小: {len(windows_dataset)} 个窗口")
    
    # 查看第一个窗口
    window_0 = windows_dataset[0]
    print(f"\n第一个窗口:")
    print(f"  - 数据类型: {type(window_0)}")
    if isinstance(window_0, tuple):
        print(f"  - 数据形状: {window_0[0].shape}")
        print(f"  - 标签: {window_0[1]}")


def tutorial_dataloader():
    """
    DataLoader: 使用 PyTorch DataLoader 批量加载数据
    
    DataLoader 参数:
    - batch_size: 批量大小
    - shuffle: 是否在每轮打乱数据
    - num_workers: 并行加载的工作线程数
    """
    print("\n" + "=" * 60)
    print("教程 1.4: DataLoader 使用")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    
    # 加载数据并窗口化
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0, 
        window_size_samples=1000,
        window_stride_samples=1000,
        preload=True,
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        windows_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,  # Windows/Mac 建议用 0
    )
    
    # 查看一个 batch
    batch = next(iter(train_loader))
    
    if isinstance(batch, (list, tuple)):
        X, y = batch[0], batch[1]
        print(f"\nBatch 信息:")
        print(f"  - X 形状: {X.shape}  [batch, channels, time]")
        print(f"  - y 形状: {y.shape}")
        print(f"  - y 值示例: {y[:10]}")
        print(f"  - 唯一标签: {torch.unique(y)}")
    else:
        print(f"\nBatch 类型: {type(batch)}")


def tutorial_data_split():
    """
    数据划分: 训练集 / 验证集 / 测试集
    
    方法1: 按描述列划分 (如 session / subject)
    方法2: 按索引手动划分
    方法3: 按窗口随机划分
    """
    print("\n" + "=" * 60)
    print("教程 1.5: 数据集划分")
    print("=" * 60)
    
    # 加载数据
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    
    # 方法1: 按 session 划分 (train / test)
    splits = dataset.split(by="session")
    
    print(f"\n按 session 划分:")
    for split_name, split_ds in splits.items():
        print(f"  {split_name}: {len(split_ds.datasets)} 个run, {len(split_ds)} 个时间点")
    
    # 方法2: 按 subject 划分 (跨被试评估)
    splits_by_subject = dataset.split(by="subject")
    print(f"\n按 subject 划分:")
    for split_name, split_ds in list(splits_by_subject.items())[:3]:
        print(f"  Subject {split_name}: {len(split_ds.datasets)} 个run")
    print(f"  ... 共 {len(splits_by_subject)} 个subject分组")

    # 方法3: 窗口化后按比例随机划分
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=1000,      # 决定窗口大小
        window_stride_samples=1000,    # 决定窗口步长
        preload=True,
    )
    
    # 按比例随机划分
    train_size = int(0.8 * len(windows_dataset))
    val_size = len(windows_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        windows_dataset,
        [train_size, val_size],
        generator=generator,
    )
    
    print(f"\n随机划分 (80/20):")
    print(f"  训练集: {len(train_dataset)} 个窗口")
    print(f"  验证集: {len(val_dataset)} 个窗口")


def tutorial_other_datasets():
    """
    其他数据集: 除了 MOABB 之外的数据集类型
    
    Braindecode 支持:
    - MOABBDataset: MOABB 基准数据集
    - TUHDataset: TUH EEG 语料库
    - HBNDataset: Healthy Brain Network
    - SleepPhysionet: 睡眠分期数据集
    - BIDSDataset: BIDS 格式数据
    """
    print("\n" + "=" * 60)
    print("教程 1.6: 其他数据集介绍")
    print("=" * 60)
    
    print("""
Braindecode 支持的数据集类型:

1. MOABBDataset:
   - 通过 MOABB 加载 150+ 公共数据集
   - 名称列表: "BNCI2014_001", "BNCI2014004", "PhysioNetMI", ...
   - 用法: MOABBDataset(dataset_name="BNCI2014_001")

2. TUHDataset:
   - 加载 TUH EEG 语料库数据
   - 用法: TUHDataset(dataset_type="TUHAbnormal", ...)

3. SleepPhysionet:
   - 睡眠分期数据集
   - 用法: SleepPhysionet(subject_ids=[0, 1, 2], ...)

4. HBNDataset:
   - Healthy Brain Network 数据
   - 用法: HBNDataset(subject_ids=[0, 1], ...)

5. BIDSDataset:
   - BIDS 格式数据
   - 用法: BIDSDataset(data_path="path/to/bids", ...)

6. EEGWindowsDataset:
   - 从已有数组创建窗口数据集
   - 用法: EEGWindowsDataset(X, y, ...)

7. EEGWindowsDataset:
   - 从 RawDataset 创建窗口数据集
   - 用法: create_windows_from_events(dataset, ...)
""")


if __name__ == "__main__":
    # 运行所有教程
    # tutorial_moabb_basic()
    # tutorial_data_inspection()
    # tutorial_windowing()
    # tutorial_dataloader()
    tutorial_data_split()
    # tutorial_other_datasets()
    
    # print("\n" + "=" * 60)
    # print("🎉 第01章完成! 你已经学会了:")
    # print("  ✅ MOABB 数据集加载")
    # print("  ✅ 数据检查与可视化")
    # print("  ✅ 数据窗口化")
    # print("  ✅ DataLoader 使用")
    # print("  ✅ 数据集划分")
    # print("  ✅ 其他数据集类型")
    # print("\n进入 02_preprocessing.py 学习数据预处理!")
    # print("=" * 60)
