"""
Braindecode 学习教程 - 第05章: 可视化分析
======================================================

本教程学习 braindecode 可视化模块的使用。

知识点:
1. 地形图 (Topomap) 可视化
2. 梯度图 (Gradient Plot) 可视化
3. 可解释性分析
4. 训练曲线可视化
5. 混淆矩阵可视化

参考: https://braindecode.org/stable/index.html#braindecodevisualization
"""

import numpy as np
import mne
import torch
import matplotlib.pyplot as plt

# braindecode 可视化模块 (可选, 部分函数可能在不同版本中)
try:
    from braindecode.visualization import (
        compute_amplitude_gradients_for_epoch,
        plot_amplitude_gradients,
    )
    HAS_BRAINDECODE_VIS = True
except ImportError:
    HAS_BRAINDECODE_VIS = False
    print("提示: braindecode.visualization 部分函数不可用, 将使用替代方案")

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
from braindecode.models import EEGNetv4


# ============================================================
# 1. 训练曲线可视化
# ============================================================

def tutorial_training_curves():
    """
    训练曲线: 损失和准确率随时间变化
    
    帮助诊断:
    - 过拟合 (训练好, 验证差)
    - 欠拟合 (训练和验证都差)
    - 学习率设置是否合理
    """
    print("=" * 60)
    print("教程 5.1: 训练曲线可视化")
    print("=" * 60)
    
    # 模拟训练数据
    np.random.seed(42)
    n_epochs = 30
    
    # 模拟损失曲线
    train_loss = 2.0 * np.exp(-np.linspace(0, 3, n_epochs)) + 0.05 + np.random.randn(n_epochs) * 0.01
    val_loss = 2.2 * np.exp(-np.linspace(0, 2.5, n_epochs)) + 0.15 + np.random.randn(n_epochs) * 0.02
    val_loss = np.minimum.accumulate(val_loss)  # 验证损失单调递减
    
    # 模拟准确率曲线
    train_acc = 1 - 0.9 * np.exp(-np.linspace(0, 2.5, n_epochs)) + np.random.randn(n_epochs) * 0.005
    val_acc = 1 - 1.0 * np.exp(-np.linspace(0, 2, n_epochs)) - 0.02 + np.random.randn(n_epochs) * 0.01
    val_acc = np.maximum.accumulate(val_acc)  # 验证准确率单调递增
    train_acc = np.clip(train_acc, 0, 1)
    val_acc = np.clip(val_acc, 0, 1)
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, n_epochs + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
    best_epoch = np.argmin(val_loss) + 1
    ax1.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_acc, "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy", linewidth=2)
    best_epoch_acc = np.argmax(val_acc) + 1
    ax2.axvline(x=best_epoch_acc, color="g", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch_acc})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("training_curves.png", dpi=100, bbox_inches="tight")
    print("  ✅ 训练曲线已保存: training_curves.png")
    
    # 分析
    print(f"\n  最佳验证损失: Epoch {best_epoch}, Loss = {val_loss[best_epoch-1]:.4f}")
    print(f"  最佳验证准确率: Epoch {best_epoch_acc}, Acc = {val_acc[best_epoch_acc-1]:.4f}")


# ============================================================
# 2. 混淆矩阵可视化
# ============================================================

def tutorial_confusion_matrix():
    """
    混淆矩阵: 展示分类器在各类别上的表现
    
    可视化帮助:
    - 识别容易混淆的类别
    - 评估每个类别的分类性能
    - 指导数据增强策略
    """
    print("\n" + "=" * 60)
    print("教程 5.2: 混淆矩阵可视化")
    print("=" * 60)
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # 模拟真实分类结果
    np.random.seed(42)
    n_samples = 200
    class_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]
    
    # 生成有意义的混淆矩阵 (左右手容易混淆, 脚舌容易混淆)
    y_true = []
    y_pred = []
    
    for _ in range(n_samples):
        true_class = np.random.randint(0, 4)
        # 某些类别对容易混淆
        if true_class == 0 and np.random.random() < 0.25:  # 左手 -> 右手
            pred_class = 1
        elif true_class == 1 and np.random.random() < 0.20:  # 右手 -> 左手
            pred_class = 0
        elif true_class == 2 and np.random.random() < 0.20:  # 脚 -> 舌
            pred_class = 3
        elif true_class == 3 and np.random.random() < 0.18:  # 舌 -> 脚
            pred_class = 2
        else:
            pred_class = true_class
        
        y_true.append(true_class)
        y_pred.append(pred_class)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 原始混淆矩阵
    im1 = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticklabels(class_names)
    
    # 添加数值
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.colorbar(im1, ax=axes[0])
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    im2 = axes[1].imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Reds, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticklabels(class_names)
    
    # 添加百分比
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{cm_normalized[i, j]:.2%}", ha="center", va="center",
                        color="white" if cm_normalized[i, j] > 0.5 else "black")
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
    print("  ✅ 混淆矩阵已保存: confusion_matrix.png")
    
    # 分析
    accuracy = np.diag(cm).sum() / cm.sum()
    print(f"\n  整体准确率: {accuracy:.2%}")
    print(f"  各类别召回率:")
    for i, name in enumerate(class_names):
        recall = cm[i, i] / cm[i].sum()
        print(f"    {name}: {recall:.2%}")


# ============================================================
# 3. 梯度可视化
# ============================================================

def tutorial_gradient_visualization():
    """
    梯度可视化: 理解模型关注的脑区
    
    Braindecode visualization 模块:
    - compute_amplitude_gradients_for_epoch: 计算幅度梯度
    - plot_amplitude_gradients: 绘制梯度图
    - compute_trial_saliency: 计算试次显著性
    - plot_trial_saliency: 绘制显著性图
    
    梯度图帮助理解:
    - 模型关注哪些电极通道
    - 不同频段的重要性
    - 不同时间窗口的贡献
    """
    print("\n" + "=" * 60)
    print("教程 5.3: 梯度可视化")
    print("=" * 60)
    
    print("""
Braindecode 可视化函数:

1. compute_amplitude_gradients_for_epoch():
   - 计算每个 epoch 的幅度梯度
   - 需要: model, dataloader, epoch_idx, input, target
   - 返回: 梯度张量

2. plot_amplitude_gradients():
   - 绘制幅度梯度
   - 需要: gradients, channel_names, sfreq
   - 输出: 梯度图

3. compute_trial_saliency():
   - 计算单个试次的显著性
   - 需要: model, input, target
   - 返回: 显著性分数

4. plot_trial_saliency():
   - 绘制试次显著性
   - 需要: saliency, channel_names
   - 输出: 显著性图
""")
    
    # 演示: 创建简单的梯度可视化
    # 注意: 这些函数需要训练好的模型, 这里创建演示数据
    
    print("创建演示梯度数据...")
    
    # 模拟梯度数据 (22 通道, 4 秒 @ 128 Hz)
    n_channels = 22
    n_times = 512
    
    # 创建有意义的梯度模式 (mu 频段在中央区较强)
    gradients = np.zeros((n_channels, n_times))
    
    # 添加 mu 频段活动 (8-13 Hz 对应 BCI IV 2a 中央区)
    mu_activity = np.sin(2 * np.pi * 10 * np.linspace(0, 4, n_times))  # 10 Hz
    for ch in range(n_channels):
        # 中央区通道 (索引 7-15) 有更强的 mu 响应
        if 7 <= ch <= 15:
            gradients[ch] = mu_activity * (0.5 + 0.5 * np.random.randn())
        else:
            gradients[ch] = mu_activity * 0.2 * np.random.randn()
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 梯度热力图 (所有通道)
    im1 = axes[0, 0].imshow(
        gradients, 
        aspect="auto", 
        interpolation="nearest",
        cmap="RdBu_r",
        extent=[0, 4, 0, n_channels-1],
        origin="lower"
    )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Channel Index")
    axes[0, 0].set_title("Gradient Heatmap (All Channels)")
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 通道平均梯度 (地形图)
    mean_gradients = np.abs(gradients).mean(axis=1)
    
    # 简化的电极位置 (22 通道 BCI IV 2a)
    # 实际使用时从 mne 信息获取
    channel_positions = {
        "Fp1": (0, 3), "Fp2": (6, 3),
        "F7": (1, 2), "F3": (3, 2), "F4": (4, 2), "F8": (5, 2),
        "T7": (0, 1), "C3": (2, 1), "Cz": (3, 1), "C4": (4, 1), "T8": (6, 1),
        "P7": (1, 0), "P3": (2, 0), "Pz": (3, 0), "P4": (4, 0), "P8": (5, 0),
        "O1": (1, -1), "O2": (5, -1),
        "FC1": (2, 2.5), "FC2": (4, 2.5),
        "FC5": (1, 1.5), "FC6": (5, 1.5),
    }
    
    ax = axes[0, 1]
    for ch_idx, (name, (x, y)) in enumerate(channel_positions.items()):
        color = plt.cm.RdYlBu_r(mean_gradients[ch_idx] / mean_gradients.max())
        size = 200 * mean_gradients[ch_idx] / mean_gradients.max()
        ax.scatter(x, y, c=[color], s=size, alpha=0.8, edgecolors="black")
        ax.annotate(name, (x, y), fontsize=7, ha="center", va="center")
    
    ax.set_xlim(-1, 7)
    ax.set_ylim(-2, 4)
    ax.set_aspect("equal")
    ax.set_title("Channel Mean Gradient (Topomap)")
    ax.axis("off")
    
    # 3. 各通道梯度时间序列
    for ch in range(n_channels):
        axes[1, 0].plot(np.linspace(0, 4, n_times), gradients[ch], alpha=0.3, color="blue")
    
    axes[1, 0].plot(np.linspace(0, 4, n_times), gradients.mean(axis=0), 
                    "r-", linewidth=2, label="Mean")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Gradient Value")
    axes[1, 0].set_title("Channel Gradient Time Series")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 频谱分析 (简化版)
    from scipy import signal
    freqs, psd = signal.welch(gradients[7], fs=128, nperseg=256)
    axes[1, 1].semilogy(freqs, psd, "b-", linewidth=2)
    axes[1, 1].axvspan(8, 13, alpha=0.3, color="red", label="Mu Band (8-13 Hz)")
    axes[1, 1].axvspan(13, 30, alpha=0.3, color="blue", label="Beta Band (13-30 Hz)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Power Spectral Density")
    axes[1, 1].set_title("Gradient Spectrum (C3 Channel)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("gradient_visualization.png", dpi=100, bbox_inches="tight")
    print("  ✅ 梯度可视化已保存: gradient_visualization.png")


# ============================================================
# 4. EEG 信号可视化
# ============================================================

def tutorial_eeg_signal_plot():
    """
    EEG 信号可视化: 查看原始 EEG 数据
    
    帮助:
    - 检查数据质量
    - 识别伪迹
    - 观察频段活动
    """
    print("\n" + "=" * 60)
    print("教程 5.4: EEG 信号可视化")
    print("=" * 60)
    
    # 加载数据
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    raw = dataset.datasets[0].raw.copy()
    
    # 预处理
    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    raw.resample(128, verbose=False)
    
    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 原始 EEG (前 5 秒, 部分通道)
    data = raw.get_data()
    time = np.arange(data.shape[1]) / raw.info["sfreq"]
    
    n_show_channels = 8
    channel_step = data.std(axis=1).max() * 1.5  # 用于通道分离
    
    ax = axes[0]
    for i in range(n_show_channels):
        channel_data = data[i, :int(5 * raw.info["sfreq"])]
        channel_time = time[:int(5 * raw.info["sfreq"])]
        ax.plot(channel_time, channel_data + i * channel_step, linewidth=0.5)
        ax.axhline(y=i * channel_step, color="gray", alpha=0.3, linewidth=0.5)
    
    ax.set_xlim(0, 5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title("Raw EEG Signal (First 5s, 8 Channels)")
    ax.set_yticks(range(n_show_channels))
    ax.set_yticklabels(raw.ch_names[:n_show_channels])
    
    # 2. 单通道频谱
    ax = axes[1]
    channel_idx = 7  # C3 通道
    from scipy import signal
    freqs, psd = signal.welch(data[channel_idx], fs=raw.info["sfreq"], nperseg=1024)
    
    ax.semilogy(freqs, psd, "b-", linewidth=1.5)
    ax.axvspan(0.5, 4, alpha=0.3, color="blue", label="Delta (0.5-4 Hz)")
    ax.axvspan(4, 8, alpha=0.3, color="green", label="Theta (4-8 Hz)")
    ax.axvspan(8, 13, alpha=0.3, color="red", label="Mu/Alpha (8-13 Hz)")
    ax.axvspan(13, 30, alpha=0.3, color="purple", label="Beta (13-30 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(f"Power Spectrum ({raw.ch_names[channel_idx]} Channel)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 3. 事件相关电位 (ERP)
    ax = axes[2]
    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
    stim_channel = stim_channels[0] if stim_channels else None
    events = mne.find_events(raw, stim_channel=stim_channel) if stim_channel else mne.find_events(raw)
    
    # 提取事件周围的数据 (-100ms 到 +500ms)
    tmin, tmax = -0.1, 0.5
    sfreq = raw.info["sfreq"]
    n_pre = int(abs(tmin) * sfreq)
    n_post = int(tmax * sfreq)
    n_samples = n_pre + n_post
    
    # 计算 ERP
    erps = []
    for event in events[:50]:  # 使用前 50 个事件
        start = event[0] - n_pre
        if start >= 0 and start + n_samples <= data.shape[1]:
            erps.append(data[channel_idx, start:start + n_samples])
    
    if erps:
        erps = np.array(erps)
        erp_mean = erps.mean(axis=0)
        erp_time = np.linspace(tmin, tmax, n_samples)
        
        ax.plot(erp_time, erps.T, alpha=0.1, color="blue", linewidth=0.5)
        ax.plot(erp_time, erp_mean, "r-", linewidth=2, label="Mean ERP")
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Stimulus Onset")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Event-Related Potential ({raw.ch_names[channel_idx]} Channel)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("eeg_signal_visualization.png", dpi=100, bbox_inches="tight")
    print("  ✅ EEG 信号可视化已保存: eeg_signal_visualization.png")


# ============================================================
# 5. 综合可视化示例
# ============================================================

def tutorial_combined_visualization():
    """
    综合可视化: 一个脚本生成所有可视化结果
    """
    print("\n" + "=" * 60)
    print("教程 5.5: 综合可视化")
    print("=" * 60)
    
    print("""
常用可视化组合:

1. 数据检查阶段:
   - EEG 原始信号图 (检查质量)
   - 功率谱密度图 (检查频段)
   - 事件相关电位 (检查锁相关)

2. 模型训练阶段:
   - 训练/验证损失曲线
   - 训练/验证准确率曲线
   - 混淆矩阵

3. 模型分析阶段:
   - 梯度热力图 (模型关注)
   - 梯度地形图 (空间分布)
   - 显著性分析 (单试次分析)

4. 结果报告阶段:
   - 汇总准确率表
   - 对比柱状图
   - ROC 曲线 (二分类)
""")
    
    # 生成演示汇总图
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Braindecode Visualization Summary", fontsize=16, fontweight="bold")
    
    # 6 个可视化子图
    # 1. EEG 信号
    ax1 = fig.add_subplot(3, 3, 1)
    demo_signal = np.random.randn(256) * 0.3 + np.sin(2 * np.pi * 10 * np.linspace(0, 2, 256)) * 0.5
    ax1.plot(np.linspace(0, 2, 256), demo_signal, "b-", linewidth=0.8)
    ax1.set_title("EEG Signal", fontsize=10)
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, alpha=0.3)
    
    # 2. 频谱
    ax2 = fig.add_subplot(3, 3, 2)
    from scipy import signal
    freqs, psd = signal.welch(demo_signal, fs=128, nperseg=128)
    ax2.semilogy(freqs, psd, "r-", linewidth=1.5)
    ax2.set_title("Power Spectrum", fontsize=10)
    ax2.set_xlabel("Freq (Hz)")
    ax2.grid(True, alpha=0.3)
    
    # 3. 地形图
    ax3 = fig.add_subplot(3, 3, 3)
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = np.cos(theta)
    head_y = np.sin(theta)
    ax3.plot(head_x, head_y, "k-", linewidth=2)
    angles = np.linspace(0, 2*np.pi, 22, endpoint=False)
    r = 0.7
    for i, angle in enumerate(angles):
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        color = plt.cm.RdYlBu(np.random.rand())
        ax3.scatter(x, y, c=[color], s=100, edgecolors="black")
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect("equal")
    ax3.set_title("Topomap", fontsize=10)
    ax3.axis("off")
    
    # 4. 训练曲线
    ax4 = fig.add_subplot(3, 3, 4)
    epochs = range(1, 21)
    train_loss = 2 * np.exp(-0.15 * np.array(list(epochs))) + 0.1
    val_loss = 2.2 * np.exp(-0.1 * np.array(list(epochs))) + 0.2
    ax4.plot(epochs, train_loss, "b-", label="Train")
    ax4.plot(epochs, val_loss, "r-", label="Val")
    ax4.set_title("Loss Curves", fontsize=10)
    ax4.set_xlabel("Epoch")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 混淆矩阵
    ax5 = fig.add_subplot(3, 3, 5)
    cm = np.array([[45, 5, 3, 2], [6, 42, 2, 4], [3, 2, 48, 2], [2, 4, 2, 46]])
    im = ax5.imshow(cm, cmap="Blues", interpolation="nearest")
    ax5.set_title("Confusion Matrix", fontsize=10)
    ax5.set_xticks(range(4))
    ax5.set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            ax5.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > 25 else "black", fontsize=9)
    
    # 6. 梯度图
    ax6 = fig.add_subplot(3, 3, 6)
    grad_data = np.random.randn(22, 512)
    ax6.imshow(grad_data, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax6.set_title("Gradient Map", fontsize=10)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Channel")
    
    # 7. 准确率对比
    ax7 = fig.add_subplot(3, 3, 7)
    models = ["EENet", "Deep4", "Shallow", "ATCNet"]
    accuracies = [0.75, 0.80, 0.72, 0.82]
    bars = ax7.bar(models, accuracies, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax7.set_title("Accuracy Comparison", fontsize=10)
    ax7.set_ylabel("Accuracy")
    ax7.set_ylim(0.6, 0.9)
    for bar, acc in zip(bars, accuracies):
        ax7.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=8)
    
    # 8. ROC 曲线 (示意)
    ax8 = fig.add_subplot(3, 3, 8)
    from sklearn.metrics import roc_curve, auc
    fpr = np.linspace(0, 1, 100)
    tpr = np.power(fpr, 0.3)  # 好的分类器
    ax8.plot(fpr, tpr, "r-", linewidth=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax8.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax8.set_title("ROC Curve", fontsize=10)
    ax8.set_xlabel("FPR")
    ax8.set_ylabel("TPR")
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # 9. 特征重要性
    ax9 = fig.add_subplot(3, 3, 9)
    channels = [f"Ch{i}" for i in range(1, 11)]
    importance = np.random.rand(10)
    importance = np.sort(importance)[::-1]
    sorted_idx = np.argsort(importance)[::-1]
    ax9.barh(range(10), importance, color="steelblue")
    ax9.set_yticks(range(10))
    ax9.set_yticklabels([channels[i] for i in sorted_idx])
    ax9.set_title("Channel Importance", fontsize=10)
    ax9.set_xlabel("Importance")
    ax9.invert_yaxis()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("visualization_summary.png", dpi=100, bbox_inches="tight")
    print("  ✅ 综合可视化已保存: visualization_summary.png")


if __name__ == "__main__":
    import mne  # noqa: F401 - 在函数内使用
    from scipy import signal  # noqa: F401
    
    tutorial_training_curves()
    tutorial_confusion_matrix()
    tutorial_gradient_visualization()
    tutorial_eeg_signal_plot()
    tutorial_combined_visualization()
    
    print("\n" + "=" * 60)
    print("🎉 第05章完成! 你已经学会了:")
    print("  ✅ 训练曲线可视化")
    print("  ✅ 混淆矩阵可视化")
    print("  ✅ 梯度可视化")
    print("  ✅ EEG 信号可视化")
    print("  ✅ 综合可视化")
    print("\n进入 main.py 完成综合实战!")
    print("=" * 60)
