## 两种方式，取决于你想用 MOABB 还是 braindecode

### 方式一：直接用 braindecode（推荐，更简单）

如果是自己采集的数据，**不需要通过 MOABB**，直接用 braindecode 的 `RawDataset` + `create_windows_from_events`：

```python
import mne
from braindecode.datasets import RawDataset
from braindecode.preprocessing import create_windows_from_events

# 1. 加载你的数据
raw = mne.io.read_raw_edf("your_data.edf")  # 或 .set, .bdf 等

# 2. 手动创建 annotations（这就是你"告诉"的方式）
#    假设：cue=2s, 想象期=4s, 总共一个 trial=6s
sfreq = raw.info["sfreq"]
cue_duration = 2.0      # 你的 cue 时长
trial_duration = 4.0    # 你的想象期时长

# 用 mne.find_events 找到 stim 触发点
events = mne.find_events(raw, stim_channel='STI')

# 3. 手动构建 annotations，把 cue 期排除掉
annotations = mne.Annotations(
    onset=events[:, 0] / sfreq + cue_duration,  # 从 cue 结束开始
    duration=[trial_duration] * len(events),    # 想象期长度
    description=['left_hand'] * len(events)     # 事件名称
)
raw.set_annotations(annotations)

# 4. 创建窗口数据集
dataset = RawDataset(raw, description={'target': 'subject_01'})
windows = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=1000,
    window_stride_samples=1000,
    preload=True,
)
```

### 方式二：创建自定义 MOABB 数据集类

如果你想复用 MOABB 的评测框架，需要继承 `BaseDataset`：

```python
from moabb.datasets import BaseDataset

class MyCustomDataset(BaseDataset):
    def __init__(self):
        super().__init__(
            subjects=[1],          # 被试列表
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},  # 事件映射
            code="MyCustom",       # 数据集代号
            interval=[2, 6],       # ← 关键：[cue时长, cue+想象时长]
            paradigm="imagery",
            doi=None,
        )
    
    def _get_single_subject_data(self, subject):
        # 返回数据字典
        return {"session_0": {"run_0": self.raw_data}}
```

其中 `interval=[cue时长, cue+想象时长]` 就是告诉 MOABB cue 有多长：

```
interval = [2, 6]
            ↑  ↑
            │  └─ cue + 想象期 = 2 + 4 = 6s
            └──── cue 时长 = 2s
```

MOABB 会自动计算：
- offset = `interval[0] * sfreq` = 2s 对应的采样点
- duration = `interval[1] - interval[0]` = 4s（想象期）

### 对比

| 方式 | 适用场景 | 复杂度 |
|------|---------|--------|
| **braindecode 直接用** | 自己采集数据，想快速训练 | ⭐ 简单 |
| **MOABB 自定义类** | 想在 MOABB 框架内做对比评测 | ⭐⭐⭐ 较复杂 |

### 核心原理

无论哪种方式，本质都是**手动构建 annotations**，把 cue 期"跳过"：

```
原始 stim:
  0s      2s            6s
  ┌───────┬─────────────┐
  │ cue期 │   想象期     │
  └───────┴─────────────┘
          ↑
          annotations 从这里开始
          onset = stim_onset + cue_duration
          duration = trial_duration
```