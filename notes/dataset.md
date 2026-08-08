`train_windows` 和 `train_dataset` 是**两个独立的数据集对象**，但共享相同的底层原始 EEG 数据。它们的关系如下：

## 核心转换逻辑

[create_windows_from_events](file:///Users/usst_ziyi/Programs/trae/DeepL/braindecode/.venv/lib/python3.14/site-packages/braindecode/preprocessing/windowers.py#L198-L409) 会遍历 `train_dataset.datasets` 中的每个 `RawDataset`，为每个 run 创建一个新的 `EEGWindowsDataset`，然后封装成新的 `BaseConcatDataset` 返回。

## 结构对比

| 属性 | `train_dataset` | `train_windows` |
|------|----------------|-----------------|
| 类型 | `BaseConcatDataset[RawDataset]` | `BaseConcatDataset[WindowsDataset]` |
| 子数据集 | 每个 run 一个 `RawDataset` | 每个 run 一个 `WindowsDataset` |
| 索引含义 | `dataset[i]` → 第 i 个**时间点**的 22 通道电压值 | `windows[i]` → 第 i 个**窗口** (512 samples = 2.048s) 的 (X, y, crop_inds) |
| 长度单位 | 时间点数（如 ~386K） | 窗口数（如 ~288） |

## 数据引用关系

```
train_dataset.datasets[0].raw  ←→  train_windows.datasets[0].raw
     (mne.io.Raw 原始连续数据)         (同一个 mne.io.Raw 对象)
```

**共享同一份 Raw 数据**——`EEGWindowsDataset` 在读取窗口时（[windowers.py#L823-L827](file:///Users/usst_ziyi/Programs/trae/DeepL/braindecode/.venv/lib/python3.14/site-packages/braindecode/preprocessing/windowers.py#L823-L827)）直接引用原始 `raw` 对象，通过 `metadata` 中的 `i_start_in_trial` / `i_stop_in_trial` 切片获取对应片段。

## 窗口元数据

每个 `WindowsDataset` 包含一个 `metadata` DataFrame，记录每个窗口的信息：

```
   i_window_in_trial  i_start_in_trial  i_stop_in_trial  target
0                  0                 0              512       0
1                  1               512             1024       0
2                  2              1024             1536       0
...
```

| 列 | 含义 |
|---|---|
| `i_window_in_trial` | 该 trial 内的第几个窗口 |
| `i_start_in_trial` | 窗口起始采样点（相对于 raw 起点） |
| `i_stop_in_trial` | 窗口结束采样点 |
| `target` | 标签（由事件类型映射而来，如 0=左手, 1=右手, 2=脚, 3=舌头） |

## 一句话总结

`train_windows` 是 `train_dataset` 从**时间点索引**转换为**窗口索引**的产物——相同的原始数据，不同的组织方式，外加窗口级的标签和位置信息，用于后续模型训练。