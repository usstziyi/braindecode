"""
Braindecode 学习教程 - 第03章: 模型基础
======================================================

本教程学习 braindecode 中各种深度学习模型的使用。

知识点:
1. Braindecode 模型架构概览
2. 常用模型介绍 (EEGNetv4, SCCNet, ATCNet 等)
3. 模型构建与参数配置
4. 模型的统一接口 (EEGModuleMixin)
5. 模型保存与加载

参考: https://braindecode.org/stable/index.html#braindecodemodels
"""

import torch
import numpy as np

# 从 braindecode.models 导入常用模型
# 部分模型可能在不同版本中不可用, 使用 try-except 保证兼容性
try:
    from braindecode.models import EEGNetv4
    EEGNetv = EEGNetv4
except ImportError:
    try:
        from braindecode.models import EEGNet
        EEGNetv = EEGNet
    except ImportError:
        EEGNetv = None

try:
    from braindecode.models import SCCNet
except ImportError:
    SCCNet = None

try:
    from braindecode.models import ATCNet
except ImportError:
    ATCNet = None

try:
    from braindecode.models import EEGNeX
except ImportError:
    EEGNeX = None

try:
    from braindecode.models import ShallowFBCSPNet, Deep4Net
except ImportError:
    try:
        from braindecode.models import ShallowFBCSPNet
        Deep4Net = None
    except ImportError:
        ShallowFBCSPNet = None
        Deep4Net = None

# 以下模型在特定版本中可能存在
try:
    from braindecode.models import EEGConformer
except ImportError:
    EEGConformer = None

try:
    from braindecode.models import SSTDPN
except ImportError:
    SSTDPN = None

try:
    from braindecode.models import HybridNet
except ImportError:
    HybridNet = None


# ============================================================
# 1. 模型架构概览
# ============================================================

def tutorial_model_overview():
    """
    Braindecode 模型分类
    
    1. 卷积类模型 (Convolutional):
       - ShallowFBCSPNet: 浅卷积 + 频带空间滤波器
       - Deep4Net: 深 4 层卷积网络
       - EEGNetv4: 高效轻量级模型
       - SCCNet: 空间卷积网络
    
    2. 注意力类模型 (Attention-based):
       - ATCNet: 基于 Transformer 的注意力
       - EEGConformer: 结合卷积和 Transformer
       - HybridNet: 混合架构
    
    3. 基础模型 (Foundation):
       - EEGPT: EEG Transformer 预训练模型
       - BENDR: 自监督预训练模型
       - Signal-JEPA: 联合嵌入预测架构
    
    4. 专用模型 (Task-specific):
       - SSTDPN: 睡眠分期
       - EEGNeX: 新一代 EEG 模型
    """
    print("=" * 60)
    print("教程 3.1: Braindecode 模型架构概览")
    print("=" * 60)
    
    print("""
Braindecode 模型分类 (65+ 模型):

┌─────────────────────────────────────────────────────────┐
│  类别              │  代表模型                          │
├─────────────────────────────────────────────────────────┤
│  卷积类 (17)       │  ShallowFBCSPNet, Deep4Net,        │
│                    │  EEGNetv4, SCCNet                  │
├─────────────────────────────────────────────────────────┤
│  注意力类 (8)      │  ATCNet, EEGConformer, HybridNet   │
├─────────────────────────────────────────────────────────┤
│  基础模型 (14)     │  EEGPT, BENDR, Signal-JEPA,        │
│                    │  LaBraM, BIOT                      │
├─────────────────────────────────────────────────────────┤
│  循环类 (4)        │  EEGLSTM, EEGGRU                  │
├─────────────────────────────────────────────────────────┤
│  滤波器组 (4)      │  FBCSPNet, EEGNetFusion            │
├─────────────────────────────────────────────────────────┤
│  可解释 (3)        │  EEGExplain, AttentionEEG          │
├─────────────────────────────────────────────────────────┤
│  通道类 (1)        │  ChannelAttention                 │
├─────────────────────────────────────────────────────────┤
│  图神经网络 (1)    │  GraphEEG                         │
└─────────────────────────────────────────────────────────┘
""")


# ============================================================
# 2. 常用模型构建
# ============================================================

def tutorial_build_models():
    """
    构建常用模型
    
    所有模型继承自 EEGModuleMixin, 支持统一的参数:
    - n_chans: 输入通道数
    - n_outputs: 输出类别数
    - n_times: 输入时间点数
    - chs_info: 通道信息 (可选)
    - input_window_seconds: 输入窗口秒数 (可选)
    - sfr: 采样率 (可选)
    """
    print("\n" + "=" * 60)
    print("教程 3.2: 构建常用模型")
    print("=" * 60)
    
    # 假设 BCI IV 2a 数据参数
    n_chans = 22        # 22 通道 EEG
    n_outputs = 4       # 4 类运动想象
    n_times = 1000      # 4 秒 @ 250 Hz
    
    # 1. EEGNetv4 - 经典轻量级模型
    if EEGNetv is not None:
        print("\n1. EEGNetv4:")
        model_eegnet = EEGNetv(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        )
        n_params = sum(p.numel() for p in model_eegnet.parameters())
        print(f"   参数数量: {n_params:,}")
        print(f"   模型结构:")
        print(model_eegnet)
    else:
        print("\n1. EEGNetv4: ⚠️  不可用")
    
    # 2. Deep4Net - 深卷积网络
    if Deep4Net is not None:
        print("\n2. Deep4Net:")
        model_deep4 = Deep4Net(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        )
        n_params = sum(p.numel() for p in model_deep4.parameters())
        print(f"   参数数量: {n_params:,}")
    else:
        print("\n2. Deep4Net: ⚠️  不可用")
    
    # 3. ShallowFBCSPNet - 浅 FBCSP 网络
    if ShallowFBCSPNet is not None:
        print("\n3. ShallowFBCSPNet:")
        model_shallow = ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        )
        n_params = sum(p.numel() for p in model_shallow.parameters())
        print(f"   参数数量: {n_params:,}")
    else:
        print("\n3. ShallowFBCSPNet: ⚠️  不可用")
    
    # 4. ATCNet - 注意力网络
    if ATCNet is not None:
        print("\n4. ATCNet:")
        model_atcnet = ATCNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        )
        n_params = sum(p.numel() for p in model_atcnet.parameters())
        print(f"   参数数量: {n_params:,}")
    else:
        print("\n4. ATCNet: ⚠️  不可用")
    
    # 5. EEGNeX - 新一代模型
    if EEGNeX is not None:
        print("\n5. EEGNeX:")
        model_eegnex = EEGNeX(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        )
        n_params = sum(p.numel() for p in model_eegnex.parameters())
        print(f"   参数数量: {n_params:,}")
    else:
        print("\n5. EEGNeX: ⚠️  不可用")


# ============================================================
# 3. 模型前向传播
# ============================================================

def tutorial_forward_pass():
    """
    模型前向传播: 数据如何流过模型
    
    输入: (batch_size, n_channels, n_times)
    输出: (batch_size, n_classes) 或 (batch_size, 1)
    """
    print("\n" + "=" * 60)
    print("教程 3.3: 模型前向传播")
    print("=" * 60)
    
    if EEGNetv is None:
        print("⚠️  EEGNetv 不可用, 跳过前向传播演示")
        return
    
    # 模型参数
    n_chans = 22
    n_outputs = 4
    n_times = 1000
    batch_size = 8
    
    # 构建模型
    model = EEGNetv(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
    )
    
    # 创建模拟输入
    x = torch.randn(batch_size, n_chans, n_times)
    print(f"\n输入形状: {x.shape}  [batch, channels, time]")
    
    # 前向传播
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}  [batch, classes]")
    print(f"输出值: {output}")
    print(f"预测类别: {output.argmax(dim=1)}")
    
    # 不同模型的前向传播
    models_to_test = []
    if Deep4Net is not None:
        models_to_test.append(("Deep4Net", Deep4Net(n_chans, n_outputs, n_times)))
    if ShallowFBCSPNet is not None:
        models_to_test.append(("ShallowFBCSPNet", ShallowFBCSPNet(n_chans, n_outputs, n_times)))
    if SCCNet is not None:
        models_to_test.append(("SCCNet", SCCNet(n_chans, n_outputs, n_times)))
    
    if models_to_test:
        print("\n不同模型的输出:")
        for name, m in models_to_test:
            m.eval()
            with torch.no_grad():
                out = m(x)
            print(f"  {name}: input={x.shape} -> output={out.shape}")


# ============================================================
# 4. 模型自定义参数
# ============================================================

def tutorial_custom_params():
    """
    自定义模型参数
    
    不同模型有不同的可选参数:
    - EEGNet: n_filters, n_channels, kernel_length
    - Deep4Net: final_conv_length, pool_mode
    - ATCNet: n_windows, attn_dropout
    - 通用: n_chans, n_outputs, n_times
    """
    print("\n" + "=" * 60)
    print("教程 3.4: 自定义模型参数")
    print("=" * 60)
    
    n_chans = 22
    n_outputs = 4
    n_times = 1000
    
    # 5. EEGNet 自定义参数
    if EEGNetv is not None:
        print("\n1. EEGNet 自定义:")
        model_eegnet = EEGNetv(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            n_filters=8,          # 第一层滤波器数量
            n_times_filter=32,     # 时间滤波器长度
            n_pool=4,              # 池化因子
            activation="relu",     # 激活函数
            dropout=0.5,           # Dropout 率
        )
        n_params = sum(p.numel() for p in model_eegnet.parameters())
        print(f"   参数数量: {n_params:,}")
    else:
        print("\n1. EEGNet: ⚠️  不可用")
    
    # Deep4Net 自定义参数
    if Deep4Net is not None:
        print("\n2. Deep4Net 自定义:")
        model_deep4 = Deep4Net(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",  # 最后卷积层长度
            pool_mode="max",          # 池化模式: "max" 或 "mean"
            third_pool_2=True,        # 使用第三次池化
            dropout=0.5,
        )
        n_params = sum(p.numel() for p in model_deep4.parameters())
        print(f"   参数数量: {n_params:,}")
    else:
        print("\n2. Deep4Net: ⚠️  不可用")
    
    # ATCNet 自定义参数
    if ATCNet is not None:
        print("\n3. ATCNet 自定义:")
        model_atcnet = ATCNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            n_windows=4,              # 时间窗口数
            attn_dropout=0.5,         # 注意力 Dropout
            dropout=0.5,
        )
        n_params = sum(p.numel() for p in model_atcnet.parameters())
        print(f"   参数数量: {n_params:,}")
    else:
        print("\n3. ATCNet: ⚠️  不可用")


# ============================================================
# 5. 模型选择指南
# ============================================================

def tutorial_model_selection():
    """
    模型选择指南
    
    场景推荐:
    1. 初学者 / 快速原型: EEGNetv4 (轻量、快速)
    2. 高精度需求: ATCNet / EEGConformer (注意力机制)
    3. 资源有限: ShallowFBCSPNet (参数最少)
    4. 睡眠分期: SSTDPN
    5. 预训练: EEGPT / BENDR (需要大量数据)
    6. 迁移学习: 使用 from_pretrained() 加载预训练权重
    """
    print("\n" + "=" * 60)
    print("教程 3.5: 模型选择指南")
    print("=" * 60)
    
    print("""
模型选择决策树:

你的场景是什么?
│
├── 入门学习 / 快速原型
│   └── EEGNetv4 ✅ (简单、快速、表现好)
│
├── 高精度需求 (有足够数据)
│   ├── EEGConformer ✅ (CNN + Transformer)
│   └── ATCNet ✅ (纯注意力)
│
├── 数据量有限
│   └── ShallowFBCSPNet ✅ (参数少, 不易过拟合)
│
├── 特定任务
│   ├── 睡眠分期 → SSTDPN
│   ├── 回归任务 → EEGRegressor
│   └── 事件检测 → DANCE
│
├── 想要预训练权重
│   ├── Hugging Face Hub → from_pretrained()
│   └── 可用模型: EEGPT, BENDR, Signal-JEPA, LaBraM
│
└── 追求极致性能
    └── EEGNeX ✅ (最新架构)

性能对比 (BCI IV 2a 公开基准):
┌─────────────────┬──────────┬────────────┐
│  模型            │  参数量   │  准确率    │
├─────────────────┼──────────┼────────────┤
│  EEGNetv4       │  17,284  │  ~75%      │
│  ShallowFBCSPNet│  4,310   │  ~72%      │
│  Deep4Net       │  62,768  │  ~80%      │
│  ATCNet         │  35,300  │  ~82%      │
│  EEGConformer   │  183,600 │  ~83%      │
└─────────────────┴──────────┴────────────┘

* 数据为参考值, 实际表现取决于数据和训练设置
""")


if __name__ == "__main__":
    tutorial_model_overview()
    tutorial_build_models()
    tutorial_forward_pass()
    tutorial_custom_params()
    tutorial_model_selection()
    
    print("\n" + "=" * 60)
    print("🎉 第03章完成! 你已经学会了:")
    print("  ✅ Braindecode 模型分类")
    print("  ✅ 常用模型构建")
    print("  ✅ 模型前向传播")
    print("  ✅ 自定义模型参数")
    print("  ✅ 模型选择指南")
    print("\n进入 04_training_basics.py 学习训练!")
    print("=" * 60)
