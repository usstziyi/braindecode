"""
Braindecode 学习教程 - 第05章: 可视化分析
======================================================

知识点:
1. 训练曲线可视化
2. 归因/梯度可视化 (saliency, integrated_gradients, amplitude_gradients)
3. 混淆矩阵可视化 (braindecode.visualization.plot_confusion_matrix)
4. EEG 信号可视化
5. 综合可视化

参考: https://braindecode.org/stable/index.html#braindecodevisualization
"""

import numpy as np
import mne
import torch
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from sklearn.metrics import confusion_matrix, auc

from braindecode.visualization import (
    saliency,
    integrated_gradients,
    amplitude_gradients,
    plot_confusion_matrix as bd_plot_confusion_matrix,
)
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGNet


# ============================================================
# 1. 训练曲线可视化
# ============================================================

def tutorial_training_curves():
    """训练曲线: 损失和准确率随时间变化"""
    print("=" * 60)
    print("教程 5.1: 训练曲线可视化")
    print("=" * 60)

    np.random.seed(42)
    n_epochs = 30
    train_loss = 2.0 * np.exp(-np.linspace(0, 3, n_epochs)) + 0.05 + np.random.randn(n_epochs) * 0.01
    val_loss = 2.2 * np.exp(-np.linspace(0, 2.5, n_epochs)) + 0.15 + np.random.randn(n_epochs) * 0.02
    val_loss = np.minimum.accumulate(val_loss)
    train_acc = np.clip(1 - 0.9 * np.exp(-np.linspace(0, 2.5, n_epochs)) + np.random.randn(n_epochs) * 0.005, 0, 1)
    val_acc = np.clip(1 - 1.0 * np.exp(-np.linspace(0, 2, n_epochs)) - 0.02 + np.random.randn(n_epochs) * 0.01, 0, 1)
    val_acc = np.maximum.accumulate(val_acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, n_epochs + 1)

    ax1.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
    best_epoch = np.argmin(val_loss) + 1
    ax1.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
    plt.savefig("training_curves.png", dpi=100, bbox_inches="tight")
    print(f"  训练曲线已保存: training_curves.png")
    print(f"  最佳验证损失: Epoch {best_epoch}, Loss = {val_loss[best_epoch-1]:.4f}")
    print(f"  最佳验证准确率: Epoch {best_epoch_acc}, Acc = {val_acc[best_epoch_acc-1]:.4f}")
    plt.close(fig)


# ============================================================
# 2. 归因/梯度可视化
# ============================================================

def tutorial_gradient_visualization():
    """使用 braindecode.visualization API 进行梯度/归因可视化"""
    print("\n" + "=" * 60)
    print("教程 5.2: 梯度/归因可视化")
    print("=" * 60)

    n_channels, n_times, n_classes = 22, 256, 4
    model = EEGNet(n_chans=n_channels, n_outputs=n_classes, n_times=n_times)
    model.eval()

    x_demo = torch.randn(1, n_channels, n_times)
    target = torch.tensor([0])

    print("\n  1. 显著性图 (saliency)...")
    sal_map = saliency(model, x_demo, target)
    sal_np = sal_map.squeeze().detach().numpy()

    print("  2. 积分梯度 (integrated_gradients)...")
    ig_map = integrated_gradients(model, x_demo, target)
    ig_np = ig_map.squeeze().detach().numpy()

    print("  3. 频域幅度梯度 (amplitude_gradients)...")
    amp_grads = amplitude_gradients(model, x_demo.detach().numpy())
    amp_grad_np = amp_grads[0, 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im1 = axes[0, 0].imshow(sal_np, aspect="auto", cmap="hot", interpolation="nearest")
    axes[0, 0].set_xlabel("Time (samples)")
    axes[0, 0].set_ylabel("Channel")
    axes[0, 0].set_title("Saliency Map")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(ig_np, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0, 1].set_xlabel("Time (samples)")
    axes[0, 1].set_ylabel("Channel")
    axes[0, 1].set_title("Integrated Gradients")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(amp_grad_np, aspect="auto", cmap="viridis",
                             interpolation="nearest", extent=[0, 128, 0, n_channels - 1], origin="lower")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Channel")
    axes[1, 0].set_title("Amplitude Gradients (Frequency Domain)")
    plt.colorbar(im3, ax=axes[1, 0])

    channel_importance = np.abs(sal_np).mean(axis=1)
    axes[1, 1].barh(range(n_channels), channel_importance, color="steelblue")
    axes[1, 1].set_xlabel("Mean |Saliency|")
    axes[1, 1].set_ylabel("Channel Index")
    axes[1, 1].set_title("Channel Importance (Saliency)")
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gradient_visualization.png", dpi=100, bbox_inches="tight")
    print("  梯度可视化已保存: gradient_visualization.png")
    plt.close(fig)


# ============================================================
# 3. 混淆矩阵可视化
# ============================================================

def tutorial_confusion_matrix():
    """使用 braindecode.visualization.plot_confusion_matrix 绘制混淆矩阵"""
    print("\n" + "=" * 60)
    print("教程 5.3: 混淆矩阵可视化")
    print("=" * 60)

    class_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]
    np.random.seed(42)
    y_true, y_pred = [], []
    for _ in range(200):
        tc = np.random.randint(0, 4)
        if tc == 0 and np.random.random() < 0.25:
            pc = 1
        elif tc == 1 and np.random.random() < 0.20:
            pc = 0
        elif tc == 2 and np.random.random() < 0.20:
            pc = 3
        elif tc == 3 and np.random.random() < 0.18:
            pc = 2
        else:
            pc = tc
        y_true.append(tc)
        y_pred.append(pc)

    cm = confusion_matrix(np.array(y_true), np.array(y_pred))
    fig = bd_plot_confusion_matrix(cm, class_names=class_names)
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
    print("  混淆矩阵已保存: confusion_matrix.png")
    accuracy = np.diag(cm).sum() / cm.sum()
    print(f"\n  整体准确率: {accuracy:.2%}")
    for i, name in enumerate(class_names):
        print(f"    {name}: {cm[i, i] / cm[i].sum():.2%}")
    plt.close(fig)


# ============================================================
# 4. EEG 信号可视化
# ============================================================

def tutorial_eeg_signal_plot():
    """EEG 信号可视化: 查看原始 EEG 数据, 频谱, ERP"""
    print("\n" + "=" * 60)
    print("教程 5.4: EEG 信号可视化")
    print("=" * 60)

    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    raw = dataset.datasets[0].raw.copy()
    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    raw.resample(128, verbose=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    data = raw.get_data()
    time = np.arange(data.shape[1]) / raw.info["sfreq"]
    n_show = 8
    step = data.std(axis=1).max() * 1.5

    # 1. 原始 EEG
    ax = axes[0]
    for i in range(n_show):
        n5 = int(5 * raw.info["sfreq"])
        ax.plot(time[:n5], data[i, :n5] + i * step, linewidth=0.5)
        ax.axhline(y=i * step, color="gray", alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, 5)
    ax.set_xlabel("Time (s)")
    ax.set_title("Raw EEG Signal (First 5s, 8 Channels)")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(raw.ch_names[:n_show])

    # 2. 频谱
    ax = axes[1]
    freqs, psd = sp_signal.welch(data[7], fs=raw.info["sfreq"], nperseg=1024)
    ax.semilogy(freqs, psd, "b-", linewidth=1.5)
    ax.axvspan(8, 13, alpha=0.3, color="red", label="Mu/Alpha (8-13 Hz)")
    ax.axvspan(13, 30, alpha=0.3, color="purple", label="Beta (13-30 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"Power Spectrum ({raw.ch_names[7]})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. ERP
    ax = axes[2]
    try:
        events = mne.find_events(raw)
    except ValueError:
        events, _ = mne.events_from_annotations(raw)

    sfreq = raw.info["sfreq"]
    n_pre, n_post = int(0.1 * sfreq), int(0.5 * sfreq)
    n_samples = n_pre + n_post
    erps = []
    for ev in events[:50]:
        s = ev[0] - n_pre
        if 0 <= s and s + n_samples <= data.shape[1]:
            erps.append(data[7, s:s + n_samples])

    if erps:
        erps = np.array(erps)
        erp_t = np.linspace(-0.1, 0.5, n_samples)
        ax.plot(erp_t, erps.T, alpha=0.1, color="blue", linewidth=0.5)
        ax.plot(erp_t, erps.mean(axis=0), "r-", linewidth=2, label="Mean ERP")
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Stimulus")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"ERP ({raw.ch_names[7]})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eeg_signal_visualization.png", dpi=100, bbox_inches="tight")
    print("  EEG 信号可视化已保存: eeg_signal_visualization.png")
    plt.close(fig)


# ============================================================
# 5. 综合可视化示例
# ============================================================

def tutorial_combined_visualization():
    """综合可视化: 一个脚本生成所有可视化结果"""
    print("\n" + "=" * 60)
    print("教程 5.5: 综合可视化")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Braindecode Visualization Summary", fontsize=16, fontweight="bold")

    # 1. EEG 信号
    ax = fig.add_subplot(3, 3, 1)
    sig = np.random.randn(256) * 0.3 + np.sin(2 * np.pi * 10 * np.linspace(0, 2, 256)) * 0.5
    ax.plot(np.linspace(0, 2, 256), sig, "b-", linewidth=0.8)
    ax.set_title("EEG Signal", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    # 2. 频谱
    ax = fig.add_subplot(3, 3, 2)
    freqs, psd = sp_signal.welch(sig, fs=128, nperseg=128)
    ax.semilogy(freqs, psd, "r-", linewidth=1.5)
    ax.set_title("Power Spectrum", fontsize=10)
    ax.set_xlabel("Freq (Hz)")
    ax.grid(True, alpha=0.3)

    # 3. 地形图示意
    ax = fig.add_subplot(3, 3, 3)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    for a in np.linspace(0, 2 * np.pi, 22, endpoint=False):
        ax.scatter(0.7 * np.cos(a), 0.7 * np.sin(a), c=[plt.cm.RdYlBu(np.random.rand())], s=100, edgecolors="black")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title("Topomap", fontsize=10)
    ax.axis("off")

    # 4. 训练曲线
    ax = fig.add_subplot(3, 3, 4)
    ep = range(1, 21)
    ax.plot(ep, 2 * np.exp(-0.15 * np.array(list(ep))) + 0.1, "b-", label="Train")
    ax.plot(ep, 2.2 * np.exp(-0.1 * np.array(list(ep))) + 0.2, "r-", label="Val")
    ax.set_title("Loss Curves", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 混淆矩阵
    ax = fig.add_subplot(3, 3, 5)
    cm = np.array([[45, 5, 3, 2], [6, 42, 2, 4], [3, 2, 48, 2], [2, 4, 2, 46]])
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title("Confusion Matrix", fontsize=10)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > 25 else "black", fontsize=9)

    # 6. 梯度图
    ax = fig.add_subplot(3, 3, 6)
    ax.imshow(np.random.randn(22, 256), aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title("Gradient Map", fontsize=10)

    # 7. 准确率对比
    ax = fig.add_subplot(3, 3, 7)
    bars = ax.bar(["EEGNet", "Deep4", "Shallow", "ATCNet"], [0.75, 0.80, 0.72, 0.82],
                   color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax.set_title("Accuracy Comparison", fontsize=10)
    ax.set_ylim(0.6, 0.9)
    for bar, acc in zip(bars, [0.75, 0.80, 0.72, 0.82]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=8)

    # 8. ROC 曲线
    ax = fig.add_subplot(3, 3, 8)
    fpr = np.linspace(0, 1, 100)
    tpr = np.power(fpr, 0.3)
    ax.plot(fpr, tpr, "r-", linewidth=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_title("ROC Curve", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 9. 特征重要性
    ax = fig.add_subplot(3, 3, 9)
    imp = np.sort(np.random.rand(10))[::-1]
    ax.barh(range(10), imp, color="steelblue")
    ax.set_yticklabels([f"Ch{i}" for i in range(1, 11)])
    ax.set_title("Channel Importance", fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("visualization_summary.png", dpi=100, bbox_inches="tight")
    print("  综合可视化已保存: visualization_summary.png")
    plt.close(fig)


if __name__ == "__main__":
    tutorial_training_curves()
    tutorial_gradient_visualization()
    tutorial_confusion_matrix()
    tutorial_eeg_signal_plot()
    tutorial_combined_visualization()

    print("\n" + "=" * 60)
    print("第05章完成! 你已经学会了:")
    print("  训练曲线可视化")
    print("  梯度/归因可视化 (saliency, integrated_gradients, amplitude_gradients)")
    print("  混淆矩阵可视化 (braindecode.visualization.plot_confusion_matrix)")
    print("  EEG 信号可视化")
    print("  综合可视化")
    print("=" * 60)