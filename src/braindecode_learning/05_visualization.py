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

# 导入 braindecode 可视化模块中的归因/梯度可视化函数
from braindecode.visualization import (
    saliency,              # 显著性图: 计算输入对输出的显著性梯度
    integrated_gradients,  # 积分梯度: 通过积分路径计算更稳定的归因梯度
    amplitude_gradients,   # 频域幅度梯度: 在频域中计算幅度对应的梯度
    plot_confusion_matrix as bd_plot_confusion_matrix,  # 混淆矩阵可视化绘图函数 (使用别名 bd_plot_confusion_matrix)
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

    import json
    from skorch.history import History
    with open("models/history.json", "r") as f:
        history = History(json.load(f))

    epochs = history[:, "epoch"]
    train_loss = history[:, "train_loss"]
    val_loss = history[:, "valid_loss"]
    val_acc = history[:, "valid_acc"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=1)
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=1)
    best_epoch = [i + 1 for i, h in enumerate(history) if h.get("valid_loss_best", False)][-1]
    ax1.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy", linewidth=1)
    best_epoch_acc = [i + 1 for i, h in enumerate(history) if h.get("valid_acc_best", False)][-1]
    ax2.axvline(x=best_epoch_acc, color="g", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch_acc})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "viz_pic/training_curves.png",  # 保存路径和文件名: 将图像保存到 viz_pic 目录下的 training_curves.png 文件
        dpi=150,                         # 分辨率: 设置图像的分辨率为每英寸150像素, 数值越高图像越清晰
        bbox_inches="tight"              # 紧凑裁剪: 自动裁剪图像四周的空白边距, 使保存的图像更加紧凑
    )
    print(f"  训练曲线已保存: viz_pic/training_curves.png")
    print(f"  最佳验证损失: Epoch {best_epoch}, Loss = {val_loss[best_epoch-1]:.4f}")
    print(f"  最佳验证准确率: Epoch {best_epoch_acc}, Acc = {val_acc[best_epoch_acc-1]:.4f}")
    plt.close(fig)


# ============================================================
# 2. 归因/梯度可视化
# ============================================================

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

def tutorial_gradient_visualization():
    """使用 braindecode.visualization API 进行梯度/归因可视化"""
    print("\n" + "=" * 60)
    print("教程 5.2: 梯度/归因可视化")
    print("=" * 60)

    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    # 根据session类别划分train、test
    splits = dataset.split(by="session")
    train_dataset = splits["0train"]
    test_dataset = splits["1test"]
    # 预处理
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

    # 定义通道数和时间点数
    n_channels = 22
    n_times = 512

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
    model.eval()

    # 使用 MPS 加速 (Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # 使用一条真实记录 (x_demo: np.ndarray, target: int)
    x_demo, target, _ = train_windows[0]
    x_demo = torch.tensor(x_demo).unsqueeze(0).to(device)  # numpy -> tensor, 添加batch维度, 迁移到MPS
    target = torch.tensor([target]).to(device)  # int -> tensor, 迁移到MPS

    print("\n  1. 显著性图 (saliency)...")
    # 每次调用 saliency 都会执行 恰好一次前向传播 + 一次反向传播 。
    # 它不接受外部已计算好的模型输出，无法跳过前向传播。
    # 归因分析需要反向传播来计算 输出对输入的梯度dy/dx ，而不是损失对权重的梯度dL/dw。
    # 分析输入哪些部分重要，即哪些通道对模型的输出有显著的影响。
    # 通过计算输入的梯度，我们可以了解模型在不同时间点对不同通道的敏感度。
    # 例如，如果一个通道的梯度值较大，说明该通道对模型的输出有显著的影响。
    # 可以根据梯度值来可视化输入的通道重要性，从而帮助我们理解模型的决策过程。
    sal_map = saliency(model, x_demo, target)  # shape: (1, 22, 512)
    sal_np = sal_map.squeeze().cpu().numpy()  # shape: (22, 512)

    print("  2. 积分梯度 (integrated_gradients)...")
    ig_map = integrated_gradients(model, x_demo, target)
    ig_np = ig_map.squeeze().cpu().numpy()

    print("  3. 频域幅度梯度 (amplitude_gradients)...")
    amp_grads = amplitude_gradients(model, x_demo.cpu().numpy()) # shape: (4, 1, 22, 257)
    print(amp_grads.shape)
    amp_grad_np = amp_grads[0, 0] # shape: (22, 257)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im1 = axes[0, 0].imshow(sal_np, aspect="auto", cmap="hot", interpolation="nearest")
    axes[0, 0].set_xlabel("Time (samples)")
    axes[0, 0].set_ylabel("Channel")
    axes[0, 0].set_title("Saliency Map")
    axes[0, 0].set_yticks(range(n_channels))
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(ig_np, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0, 1].set_xlabel("Time (samples)")
    axes[0, 1].set_ylabel("Channel")
    axes[0, 1].set_title("Integrated Gradients")
    axes[0, 1].set_yticks(range(n_channels))
    plt.colorbar(im2, ax=axes[0, 1])

    # extent=[0, 128, 0, n_channels - 1] — 坐标轴范围
    # - x 轴：0 → 128 Hz（频率）
    # - y 轴：0 → 21（通道索引）
    # origin="lower" — 原点在左下角，y 轴向上增长
    im3 = axes[1, 0].imshow(amp_grad_np, aspect="auto", cmap="viridis",
                             interpolation="nearest", extent=[0, amp_grad_np.shape[1], 0, amp_grad_np.shape[0] - 1], origin="lower")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Channel")
    axes[1, 0].set_title("Amplitude Gradients (Frequency Domain)")
    axes[1, 0].set_yticks(range(n_channels))
    plt.colorbar(im3, ax=axes[1, 0])

    channel_importance = np.abs(sal_np).mean(axis=1)
    axes[1, 1].barh(range(n_channels), channel_importance, color="steelblue")
    axes[1, 1].set_xlabel("Mean |Saliency|")
    axes[1, 1].set_ylabel("Channel Index")
    axes[1, 1].set_title("Channel Importance (Saliency)")
    axes[1, 1].invert_yaxis() # 让0在顶部
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
    # tutorial_training_curves()        # 训练曲线可视化
    tutorial_gradient_visualization()  # 梯度/归因可视化 (saliency, integrated_gradients, amplitude_gradients)
    # tutorial_confusion_matrix()        # 混淆矩阵可视化 (braindecode.visualization.plot_confusion_matrix)
    # tutorial_eeg_signal_plot()         # EEG 信号可视化
    # tutorial_combined_visualization()  # 综合可视化