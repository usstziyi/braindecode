"""
Braindecode 学习教程 - 第07章: 可视化循序渐进
==================================================

本章聚焦 braindecode.visualization 模块, 从入门到精通.

前置知识:
  - 已完成 01~04 章 (数据集 / 预处理 / 模型 / 训练)
  - 理解 EEG 数据形状: (batch, n_channels, n_times)

本章结构:
  Tutorial 1: 环境搭建 —— 真实数据 + 真实模型
  Tutorial 2: 基础归因 —— Saliency Map (最基础的可解释性)
  Tutorial 3: 主流归因 —— Integrated Gradients (Integrated Gradients)
  Tutorial 4: 高级归因 —— Guided Backprop / DeepLIFT / IxG / LRP
  Tutorial 5: 频域 & 反卷积 —— frequency / deconvolution / amplitude_gradients
  Tutorial 6: 地形图 & 健全性检查 —— project_to_topomap + sanity check + 指标

依赖:
  braindecode>=1.5, captum, scipy, scikit-learn

运行:
  uv run python src/braindecode_learning/07_viz_step_by_step.py
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

# 输出目录: 生成的图片保存在这里
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "viz_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 公共组件: 真实数据 + 真实模型
# ============================================================

def build_real_context(
    n_channels: int = 22,
    n_times: int = 500,
    n_classes: int = 4,
    pretrained: bool = False,
):
    """
    构建"真实"的上下文 (不下载 MOABB, 使用模拟数据 + 真实 EEGNet).

    返回:
        model      :  EEGNet (支持 chs_info 属性)
        x          :  随机输入 (16, n_channels, n_times)
        y          :  随机标签 (16,)
        ch_names   :  通道名列表
        chs_info   :  braindecode 要求的通道信息 (用于 project_to_topomap)
    """
    from braindecode.models import EEGNet

    rng = np.random.default_rng(42)

    x = rng.normal(size=(16, n_channels, n_times)).astype(np.float32)
    y = rng.integers(0, n_classes, size=16)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    model = EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        sfreq=250,
    )

    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "O2", "FC1", "FC2", "FC5", "FC6",
    ][:n_channels]

    # 标准 10-20 简化坐标 (半径 0.08 m 的圆, 随机散布)
    chs_info = []
    for i, name in enumerate(ch_names):
        angle = (i / len(ch_names)) * 2 * np.pi
        r = 0.05 + 0.03 * np.sin(angle * 2 + i)
        loc = np.zeros(12)
        loc[0] = r * np.cos(angle)
        loc[1] = r * np.sin(angle)
        chs_info.append({"ch_name": name, "loc": loc, "kind": 1})

    # 给 model 附加 chs_info (braindecode 内部会读取)
    model.__dict__["chs_info"] = chs_info

    model.eval()
    with torch.no_grad():
        _ = model(x_tensor)

    print(f"  [context] x: {x_tensor.shape}, y: {y_tensor.shape}, channels: {len(ch_names)}")
    print(f"  [context] model params: {sum(p.numel() for p in model.parameters()):,}")
    return model, x_tensor, y_tensor, ch_names, chs_info


def savefig(name: str):
    path = OUT_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  ✔ saved → {path.name}")


# ============================================================
# Tutorial 1: Saliency Map —— 最基础的归因
# ============================================================

def tutorial_saliency():
    """
    学习目标:
      - 理解 Saliency Map 原理: |∂f_c / ∂x|
      - 调用 braindecode.visualization.saliency
      - 可视化单样本 & 多样本平均的显著性图

    关键 API:
        saliency(model, x, target)  →  Tensor, 形状同 x
    """
    print("\n" + "=" * 60)
    print("Tutorial 1: Saliency Map")
    print("=" * 60)

    from braindecode.visualization import saliency

    model, x, y, ch_names, chs_info = build_real_context()

    # --- 1.1 单样本 ---
    x_single = x[0:1].clone().requires_grad_(True)
    target = y[0:1]

    sal_map = saliency(model, x_single, target)
    print(f"  sal_map shape: {sal_map.shape}")  # (1, 22, 500)

    sal_np = sal_map.squeeze().detach().numpy()  # (22, 500)

    # --- 1.2 可视化 ---
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle("Saliency Map (单样本)", fontweight="bold")

    # (1) 通道 × 时间 热力图
    im = axes[0, 0].imshow(
        sal_np, aspect="auto", cmap="hot",
        extent=[0, sal_np.shape[1] - 1, len(ch_names) - 1, 0],
    )
    axes[0, 0].set_xlabel("Time sample")
    axes[0, 0].set_ylabel("Channel")
    axes[0, 0].set_yticks(range(0, len(ch_names), 2))
    axes[0, 0].set_yticklabels(ch_names[::2], fontsize=7)
    plt.colorbar(im, ax=axes[0, 0])

    # (2) 通道平均重要性
    ch_imp = sal_np.mean(axis=1)
    axes[0, 1].barh(range(len(ch_names)), ch_imp, color="steelblue")
    axes[0, 1].set_yticks(range(len(ch_names)))
    axes[0, 1].set_yticklabels(ch_names, fontsize=7)
    axes[0, 1].set_xlabel("mean |gradient|")
    axes[0, 1].set_title("per-channel importance")
    axes[0, 1].invert_yaxis()

    # (3) 时间维度重要性
    t_imp = sal_np.mean(axis=0)
    axes[1, 0].plot(t_imp, "r-", lw=1.2)
    axes[1, 0].fill_between(range(len(t_imp)), t_imp, alpha=0.25, color="red")
    axes[1, 0].set_xlabel("Time sample")
    axes[1, 0].set_ylabel("mean |gradient|")
    axes[1, 0].grid(True, alpha=0.3)

    # (4) Top-5 通道
    top_k = 5
    top_idx = np.argsort(ch_imp)[::-1][:top_k]
    axes[1, 1].barh(
        [ch_names[i] for i in top_idx],
        ch_imp[top_idx],
        color=["#E53935", "#FB8C00", "#FDD835", "#43A047", "#1E88E5"],
    )
    axes[1, 1].set_xlabel("mean |gradient|")
    axes[1, 1].set_title(f"Top-{top_k} channels")
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    savefig("tutorial_1_saliency.png")

    # --- 1.3 多样本平均 (群体层面) ---
    print("\n  averaging over all samples ...")
    avg_sal = []
    for i in range(x.shape[0]):
        xi = x[i : i + 1].clone().requires_grad_(True)
        ti = y[i : i + 1]
        avg_sal.append(saliency(model, xi, ti).squeeze().detach().numpy())
    avg_sal = np.mean(avg_sal, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(avg_sal, aspect="auto", cmap="hot")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    ax.set_yticks(range(0, len(ch_names), 2))
    ax.set_yticklabels(ch_names[::2], fontsize=7)
    ax.set_title("Mean Saliency over all samples")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    savefig("tutorial_1_saliency_mean.png")

    print("  💡 Saliency 要点: 梯度越大 ≠ 越重要, 但能快速定位区域")


# ============================================================
# Tutorial 2: Integrated Gradients —— 主流归因
# ============================================================

def tutorial_integrated_gradients():
    """
    学习目标:
      - 理解 IG 原理: 沿 baseline → x 的线性插值积分
      - 对比 Saliency 与 IG 的差异
      - 验证"完整性公理"

    关键 API:
        integrated_gradients(model, x, target, baseline=None, steps=50)
    """
    print("\n" + "=" * 60)
    print("Tutorial 2: Integrated Gradients")
    print("=" * 60)

    from braindecode.visualization import (
        integrated_gradients,
        saliency,
    )

    model, x, y, ch_names, chs_info = build_real_context()

    # --- 2.1 计算 IG ---
    sample_idx = 0
    xi = x[sample_idx : sample_idx + 1].clone().requires_grad_(True)
    target = y[sample_idx : sample_idx + 1]

    ig_map = integrated_gradients(model, xi, target, baseline=None, steps=50)
    ig_np = ig_map.squeeze().detach().numpy()  # (22, 500)

    sal_map = saliency(model, xi, target).squeeze().detach().numpy()

    print(f"  IG shape: {ig_np.shape}")
    print(f"  IG range: [{ig_np.min():.4f}, {ig_np.max():.4f}]")

    # --- 2.2 可视化对比 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Integrated Gradients vs Saliency", fontweight="bold")

    # IG 热力图
    im = axes[0, 0].imshow(ig_np, aspect="auto", cmap="RdBu_r")
    axes[0, 0].set_title("Integrated Gradients")
    axes[0, 0].set_xlabel("Time sample")
    axes[0, 0].set_ylabel("Channel")
    plt.colorbar(im, ax=axes[0, 0])

    # Saliency 热力图
    im = axes[0, 1].imshow(sal_map, aspect="auto", cmap="hot")
    axes[0, 1].set_title("Saliency Map (for comparison)")
    axes[0, 1].set_xlabel("Time sample")
    axes[0, 1].set_ylabel("Channel")
    plt.colorbar(im, ax=axes[0, 1])

    # 通道归因对比
    ch_ig = ig_np.mean(axis=1)
    ch_sal = sal_map.mean(axis=1)
    axes[1, 0].plot(range(len(ch_names)), ch_ig, "o-", label="IG", alpha=0.8)
    axes[1, 0].plot(range(len(ch_names)), ch_sal, "s-", label="Saliency", alpha=0.8)
    axes[1, 0].set_xticks(range(len(ch_names)))
    axes[1, 0].set_xticklabels(ch_names, rotation=90, fontsize=7)
    axes[1, 0].set_ylabel("mean attribution")
    axes[1, 0].set_title("per-channel attribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color="gray", ls="--", alpha=0.5)

    # 散点对比 (归一化)
    ig_norm = (ig_np - ig_np.min()) / (ig_np.max() - ig_np.min() + 1e-8)
    sal_norm = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
    axes[1, 1].scatter(ig_norm.flatten()[::20], sal_norm.flatten()[::20], alpha=0.4, s=8)
    lims = [0, 1]
    axes[1, 1].plot(lims, lims, "r--", alpha=0.5, label="y = x")
    axes[1, 1].set_xlabel("IG (normalized)")
    axes[1, 1].set_ylabel("Saliency (normalized)")
    axes[1, 1].set_title("IG vs Saliency scatter")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("tutorial_2_ig_vs_saliency.png")

    # --- 2.3 完整性公理验证 ---
    print("\n  verifying completeness axiom ...")
    with torch.no_grad():
        f_x = model(xi)[:, target.item()].item()
        baseline = torch.zeros_like(xi)
        f_b = model(baseline)[:, target.item()].item()
    ig_sum = ig_map.sum().item()
    print(f"  f(x) - f(baseline) = {f_x - f_b:.4f}")
    print(f"  sum(IG)            = {ig_sum:.4f}")
    print(f"  ratio              = {ig_sum / (f_x - f_b + 1e-8):.4f}")

    # --- 2.4 类别特定归因 ---
    print("\n  class-specific IG ...")
    fig, ax = plt.subplots(figsize=(10, 4))
    class_ig_means = []
    for c in range(4):
        fake_target = torch.tensor([c])
        ig_c = integrated_gradients(model, xi, fake_target, steps=50)
        class_ig_means.append(ig_c.squeeze().detach().numpy().mean())
    bars = ax.bar(
        ["Class 0", "Class 1", "Class 2", "Class 3"],
        class_ig_means,
        color=["#EF5350", "#66BB6A", "#42A5F5", "#FFA726"],
    )
    for bar, val in zip(bars, class_ig_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("mean IG value")
    ax.set_title("class-specific attribution (same input)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savefig("tutorial_2_class_specific_ig.png")


# ============================================================
# Tutorial 3: 高级归因 —— GB / DeepLIFT / IxG / LRP
# ============================================================

def tutorial_advanced_attribution():
    """
    学习目标:
      - 一次性对比 4 种高级归因方法
      - 理解每种方法的优缺点与适用场景

    关键 API:
        guided_backprop(model, x, target)
        deep_lift(model, x, target, baseline=None)
        input_x_gradient(model, x, target)
        lrp(model, x, target)
    """
    print("\n" + "=" * 60)
    print("Tutorial 3: Advanced Attribution Methods")
    print("=" * 60)

    from braindecode.visualization import (
        guided_backprop,
        deep_lift,
        input_x_gradient,
        lrp,
    )

    model, x, y, ch_names, chs_info = build_real_context()

    xi = x[0:1].clone().requires_grad_(True)
    target = y[0:1]

    methods = {}

    # Guided Backprop
    print("  → guided_backprop ...")
    methods["Guided\nBackprop"] = guided_backprop(model, xi, target).squeeze().detach().numpy()

    # Input × Gradient
    print("  → input_x_gradient ...")
    methods["Input ×\nGradient"] = input_x_gradient(model, xi, target).squeeze().detach().numpy()

    # DeepLIFT
    print("  → deep_lift ...")
    try:
        methods["DeepLIFT"] = deep_lift(model, xi, target).squeeze().detach().numpy()
    except Exception as e:
        print(f"    ⚠ deep_lift failed: {e}, fallback to zeros")
        methods["DeepLIFT"] = np.zeros_like(methods["Guided\nBackprop"])

    # LRP — 某些模型不支持, 捕获异常
    print("  → lrp (may fail for some models) ...")
    try:
        methods["LRP"] = lrp(model, xi, target).squeeze().detach().numpy()
    except Exception as e:
        print(f"    ⚠ lrp failed: {e}")
        methods["LRP"] = np.zeros_like(methods["Guided\nBackprop"])

    # --- 3.1 2×2 对比图 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("4 Advanced Attribution Methods (same sample)", fontweight="bold")
    cmaps = ["hot", "plasma", "viridis", "RdBu_r"]

    for ax, (name, arr), cmap in zip(axes.flat, methods.items(), cmaps):
        im = ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(name.replace("\n", " "))
        ax.set_xlabel("Time sample", fontsize=8)
        ax.set_ylabel("Channel", fontsize=8)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    savefig("tutorial_3_advanced_attribution.png")

    # --- 3.2 各方法的通道重要性柱状图 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Per-channel importance by method", fontweight="bold")
    for ax, (name, arr) in zip(axes.flat, methods.items()):
        ch_imp = np.abs(arr).mean(axis=1)
        ax.bar(range(len(ch_names)), ch_imp, color="steelblue")
        ax.set_xticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax.set_ylabel("mean |attribution|")
        ax.set_title(name.replace("\n", " "))
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savefig("tutorial_3_advanced_channel_imp.png")

    # --- 3.3 总结表格 ---
    print("\n  Per-channel importance (top-5 per method):")
    print(f"  {'Channel':<8}", end="")
    for name in methods:
        short = name.replace("\n", " ")
        print(f"  {short:>18}", end="")
    print()
    print("  " + "-" * (8 + 18 * len(methods)))
    for i, ch in enumerate(ch_names):
        print(f"  {ch:<8}", end="")
        for arr in methods.values():
            val = np.abs(arr[i]).mean()
            print(f"  {val:>18.4f}", end="")
        print()

    print("\n  📘 选择建议:")
    print("    - 追求稳定 / 完整性公理 → Integrated Gradients")
    print("    - 追求视觉锐度       → Guided Backprop")
    print("    - 简单快速           → Input × Gradient")
    print("    - 理论完美 (慢)      → LRP (若可用)")


# ============================================================
# Tutorial 4: 频域归因 + 反卷积 + 振幅梯度
# ============================================================

def tutorial_frequency_and_gradients():
    """
    学习目标:
      - 使用 frequency 模块进行频域归因
      - 使用 deconvolution 理解特征
      - 使用 amplitude_gradients / amplitude_gradients_per_trial
        计算"某频段 / 某时期"的梯度

    关键 API:
        braindecode.visualization.frequency 模块
        deconvolution(model, x, target)
        amplitude_gradients(model, x)
        amplitude_gradients_per_trial(model, dataset, batch_size)
    """
    print("\n" + "=" * 60)
    print("Tutorial 4: Frequency Attribution & Amplitude Gradients")
    print("=" * 60)

    import braindecode.visualization.frequency as freq_module
    from braindecode.visualization import deconvolution, amplitude_gradients

    model, x, y, ch_names, chs_info = build_real_context()

    xi = x[0:1].clone().requires_grad_(True)
    target = y[0:1]

    # --- 4.1 频域模块结构探索 ---
    print("\n  frequency 模块内容:")
    for name in dir(freq_module):
        if not name.startswith("_"):
            obj = getattr(freq_module, name)
            print(f"    {name}: {type(obj).__name__}")

    # --- 4.2 deconvolution ---
    print("\n  → deconvolution ...")
    try:
        deconv_map = deconvolution(model, xi, target)
        print(f"    deconv shape: {deconv_map.shape}")
        deconv_np = deconv_map.squeeze().detach().numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(deconv_np, aspect="auto", cmap="RdBu_r")
        ax.set_title("Deconvolution (解释输入贡献的分解)")
        ax.set_xlabel("Time sample")
        ax.set_ylabel("Channel")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        savefig("tutorial_4_deconvolution.png")
    except Exception as e:
        print(f"    ⚠ deconvolution failed: {e}")

    # --- 4.3 amplitude_gradients ---
    # amplitude_gradients 返回 numpy ndarray, 形状 (n_freq_bands, 1, n_channels, n_times)
    print("\n  → amplitude_gradients ...")
    try:
        amp_grad = amplitude_gradients(model, xi.detach().clone())
        print(f"    amp_grad shape: {amp_grad.shape}  (freq_bands, 1, channels, times)")

        # 取平均频率的梯度, 形状变为 (channels, times)
        amp_grad_np = amp_grad.mean(axis=(0, 1))

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(amp_grad_np, aspect="auto", cmap="hot")
        ax.set_title("Amplitude Gradients (频谱梯度, 多频段平均)")
        ax.set_xlabel("Time sample")
        ax.set_ylabel("Channel")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        savefig("tutorial_4_amplitude_gradients.png")

        # 按频段展示
        n_freq = amp_grad.shape[0]
        fig, axes = plt.subplots(1, n_freq, figsize=(4 * n_freq, 4))
        if n_freq == 1:
            axes = [axes]
        freq_bands = ["Delta", "Theta", "Alpha", "Beta"][:n_freq]
        for fi, (ax_f, name) in enumerate(zip(axes, freq_bands)):
            band = amp_grad[fi, 0]
            im = ax_f.imshow(band, aspect="auto", cmap="hot")
            ax_f.set_title(f"Band {fi}: {name}")
            ax_f.set_xlabel("Time")
            ax_f.set_ylabel("Channel")
            plt.colorbar(im, ax=ax_f)
        plt.suptitle("Amplitude Gradients per Frequency Band", fontweight="bold")
        plt.tight_layout()
        savefig("tutorial_4_amplitude_per_band.png")
    except Exception as e:
        print(f"    ⚠ amplitude_gradients failed: {e}")
        import traceback
        traceback.print_exc()

    # --- 4.4 amplitude_gradients_per_trial (需要 dataset) ---
    print("\n  → amplitude_gradients_per_trial (需要 WindowedDataset)")
    print("    示例代码 (真实项目中使用):")
    print("    from braindecode.visualization import amplitude_gradients_per_trial")
    print("    grads = amplitude_gradients_per_trial(model, windowed_dataset, batch_size=16)")
    print("    # grads: list, 每个元素是单试次的梯度张量")


# ============================================================
# Tutorial 5: 地形图投影 & 混淆矩阵
# ============================================================

def tutorial_topomap_and_confusion():
    """
    学习目标:
      - 使用 project_to_topomap 将 1D 通道值投射到 2D 头皮地形图
      - 使用 plot_confusion_matrix 绘制增强版混淆矩阵

    关键 API:
        project_to_topomap(data, chs_info, res=64)
        plot_confusion_matrix(confusion_mat, ..., with_f1_score=True)
    """
    print("\n" + "=" * 60)
    print("Tutorial 5: Topomap Projection & Enhanced Confusion Matrix")
    print("=" * 60)

    from braindecode.visualization import (
        project_to_topomap,
        saliency,
        plot_confusion_matrix,
    )

    model, x, y, ch_names, chs_info = build_real_context()

    # --- 5.1 从 Saliency 提取通道归因 ---
    xi = x[0:1].clone().requires_grad_(True)
    target = y[0:1]
    sal_map = saliency(model, xi, target).squeeze().detach().numpy()
    channel_attr = np.abs(sal_map).mean(axis=1)  # (n_channels,)
    print(f"  channel_attr shape: {channel_attr.shape}")

    # --- 5.2 投影到 2D 头皮 ---
    print("  → project_to_topomap ...")
    # data 提供 数值 ， chs_info 提供 空间坐标 ， res 决定 输出精度
    # data (通道值)  +  chs_info (3D坐标)  +  res (网格分辨率)
    #     ↓                ↓                    ↓
    # MNE 球体拟合 → 3D坐标投影到2D平面 (pos2d)
    #     ↓
    # Clough-Tocher 三角剖分插值
    #     ↓
    # 输出 (res, res) 的 2D 头皮热力图
    # （头皮外区域填充 NaN）
    topomap = project_to_topomap(channel_attr, chs_info, res=64)
    print(f"  topomap shape: {topomap.shape}") # (64, 64)
    # 统计 topomap 中有多少个有效（非 NaN）网格单元
    print(f"  valid cells: {np.sum(~np.isnan(topomap))}")

    # --- 5.3 可视化 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Topomap Projection", fontweight="bold")

    # (1) 1D 通道值
    axes[0].barh(range(len(ch_names)), channel_attr, color="steelblue")
    axes[0].set_yticks(range(len(ch_names)))
    axes[0].set_yticklabels(ch_names, fontsize=7)
    axes[0].set_xlabel("mean |saliency|")
    axes[0].set_title("1D channel attribution")
    axes[0].invert_yaxis()

    # (2) 2D topomap
    masked = np.ma.masked_invalid(topomap)
    im = axes[1].imshow(masked, cmap="RdBu_r", aspect="equal", origin="lower", interpolation="bicubic")
    # 电极位置
    for ch in chs_info:
        xp = (ch["loc"][0] + 0.1) / 0.2 * 64
        yp = (ch["loc"][1] + 0.1) / 0.2 * 64
        axes[1].plot(xp, yp, "k.", ms=4)
    axes[1].set_title("2D topomap (64×64)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    # (3) Saliency 原始热力图 (对比)
    im = axes[2].imshow(sal_map, aspect="auto", cmap="hot")
    axes[2].set_title("original saliency heatmap")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Channel")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    savefig("tutorial_5_topomap.png")

    # --- 5.4 增强版混淆矩阵 ---
    print("\n  → plot_confusion_matrix ...")
    class_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]
    confusion_mat = np.array([
        [44, 4, 3, 1],
        [5, 42, 2, 3],
        [3, 2, 45, 2],
        [1, 3, 2, 46],
    ])

    fig_cm = plot_confusion_matrix(
        confusion_mat,
        class_names=class_names,
        with_f1_score=True,
        colormap=plt.cm.Blues,
    )
    fig_cm.savefig(OUT_DIR / "tutorial_5_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  ✔ saved → tutorial_5_confusion_matrix.png")


# ============================================================
# Tutorial 6: 健全性检查 + 指标评估
# ============================================================

def tutorial_sanity_and_metrics():
    """
    学习目标:
      - 理解健全性检查 (Adebayo et al., 2018)
      - 使用 random_target 生成对照标签
      - 使用 cascading_layer_reset 重置模型
      - 使用 compute_metrics / compute_ssim_metrics 量化解释质量

    关键 API:
        braindecode.visualization.sanity.random_target
        braindecode.visualization.sanity.cascading_layer_reset
        braindecode.visualization.metrics.compute_metrics
        braindecode.visualization.metrics.compute_ssim_metrics
    """
    print("\n" + "=" * 60)
    print("Tutorial 6: Sanity Checks & Attribution Metrics")
    print("=" * 60)

    from braindecode.visualization import (
        saliency,
        integrated_gradients,
    )
    from braindecode.visualization.sanity import (
        random_target,
        cascading_layer_reset,
    )
    from braindecode.visualization.metrics import (
        compute_metrics,
        compute_ssim_metrics,
    )

    model, x, y, ch_names, chs_info = build_real_context()

    xi = x[0:1].clone().requires_grad_(True)
    target = y[0:1]

    # --- 6.1 random_target 用法 ---
    print("\n  → random_target ...")
    fake_target = random_target(target, n_classes=4)
    print(f"    真实标签: {target.item()},  随机标签: {fake_target.item()}")

    # --- 6.2 cascading_layer_reset ---
    # 注意: cascading_layer_reset 是一个 generator, 每次 yield (layer_name, model_with_reset)
    print("\n  → cascading_layer_reset ...")
    reset_steps = list(cascading_layer_reset(copy.deepcopy(model), deepcopy_first=True))
    print(f"    共 {len(reset_steps)} 个重置步骤")
    for step_name, step_model in reset_steps:
        print(f"      - {step_name}")

    # --- 6.3 健全性检查: 逐层随机化后归因是否显著变化 ---
    print("\n  → Sanity Check: layer-wise randomization ...")
    layer_attrs = []
    layer_names = ["original"]

    # 原始模型
    attr_orig = integrated_gradients(model, xi, target, steps=30).squeeze().detach().numpy()
    layer_attrs.append(attr_orig)

    # 使用 cascading_layer_reset 生成的逐层模型
    for step_name, step_model in reset_steps[:5]:
        try:
            attr_l = integrated_gradients(step_model, xi, target, steps=30).squeeze().detach().numpy()
            layer_attrs.append(attr_l)
            layer_names.append(step_name)
            print(f"    {step_name}: mean |Δ| = {np.abs(attr_orig - attr_l).mean():.4f}")
        except Exception as e:
            print(f"    {step_name} failed: {e}")

    # 可视化
    n_panels = len(layer_attrs)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Sanity Check: Cascading Layer Reset", fontweight="bold")
    axes = axes.flat
    for i, (name, arr) in enumerate(zip(layer_names[:6], layer_attrs[:6])):
        im = axes[i].imshow(arr, aspect="auto", cmap="RdBu_r")
        axes[i].set_title(name)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i])
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    savefig("tutorial_6_sanity_check.png")

    # --- 6.4 量化指标 ---
    # compute_metrics / compute_ssim_metrics 接受 numpy ndarray, 形状 (n_samples, n_chans, n_times)
    # 注意: 返回 (metrics_array, n_skipped) 元组
    print("\n  → compute_metrics ...")
    explanations_arr = np.stack(layer_attrs[1:], axis=0)  # (n_reset, channels, times)
    reference_arr = np.stack([layer_attrs[0]] * len(layer_attrs[1:]), axis=0)
    try:
        metrics_result, n_skipped = compute_metrics(
            explanations_arr, reference_arr, chs_info=chs_info
        )
        print(f"    metrics shape: {metrics_result.shape}, 跳过样本: {n_skipped}")
        print(f"    可用指标数: {metrics_result.shape[1]}")
    except Exception as e:
        print(f"    compute_metrics failed: {e}")
        import traceback; traceback.print_exc()

    print("\n  → compute_ssim_metrics ...")
    try:
        ssim_result, n_skipped_ssim = compute_ssim_metrics(
            explanations_arr, reference_arr, chs_info=chs_info, win_size=5
        )
        print(f"    SSIM shape: {ssim_result.shape}, 跳过样本: {n_skipped_ssim}")
    except Exception as e:
        print(f"    compute_ssim_metrics failed: {e}")
        import traceback; traceback.print_exc()

    # --- 6.5 class-conditional 健全性 ---
    print("\n  → class-conditional sanity (same input, different targets) ...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Attribution for different target classes (same input)", fontweight="bold")
    for c in range(4):
        fake_t = torch.tensor([c])
        attr_c = integrated_gradients(model, xi, fake_t, steps=30).squeeze().detach().numpy()
        im = axes[c].imshow(attr_c, aspect="auto", cmap="RdBu_r")
        axes[c].set_title(f"Class {c}")
        axes[c].axis("off")
        plt.colorbar(im, ax=axes[c])
    plt.tight_layout()
    savefig("tutorial_6_class_conditional.png")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # print("╔══════════════════════════════════════════════════════════╗")
    # print("║  Braindecode Visualization — 循序渐进教程              ║")
    # print("║  输出目录: " + str(OUT_DIR))
    # print("╚══════════════════════════════════════════════════════════╝")

    # print("\n  建议学习顺序:")
    # print("    Tutorial 1 → Saliency Map              (15 min)")
    # print("    Tutorial 2 → Integrated Gradients     (20 min)")
    # print("    Tutorial 3 → 高级归因方法对比          (20 min)")
    # print("    Tutorial 4 → 频域 / 反卷积 / 振幅梯度  (15 min)")
    # print("    Tutorial 5 → 地形图投影 & 混淆矩阵     (15 min)")
    # print("    Tutorial 6 → 健全性检查 & 指标         (20 min)")

    # tutorial_saliency()
    # tutorial_integrated_gradients()
    # tutorial_advanced_attribution()
    # tutorial_frequency_and_gradients()
    tutorial_topomap_and_confusion()
    # tutorial_sanity_and_metrics()

    # print("\n" + "=" * 60)
    # print("🎉 全部教程执行完成!")
    # print(f"  所有图片已保存至: {OUT_DIR}")
    # print("\n  下一步建议:")
    # print("    1. 用真实 MOABB 数据替换模拟数据 (见 01_moabb_dataset.py)")
    # print("    2. 用训练好的 EEGClassifier 替换未训练的 EEGNet")
    # print("    3. 阅读 braindecode 官方示例:")
    # print("       https://braindecode.org/stable/auto_examples/")
    # print("=" * 60)
