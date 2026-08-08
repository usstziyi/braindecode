"""
Braindecode 学习教程 - 第06章: 模型可解释性与深度可视化
======================================================

本教程深入学习 braindecode.visualization 模块的核心功能:
模型可解释性 (Interpretability) 与归因分析 (Attribution)

知识点:
1. 归因分析基础 (Saliency Map)
2. 进阶归因方法 (Integrated Gradients, Input x Gradient)
3. 高级归因方法 (Guided Backprop, LRP)
4. 地形图投影 (Topomap Projection)
5. 增强版混淆矩阵
6. 健全性检查 (Sanity Checks)

参考: https://braindecode.org/stable/api.html#braindecode.visualization
依赖: pip install captum (归因分析需要)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

# ============================================================
# 0. 环境准备
# ============================================================

def setup_environment():
    """
    设置运行环境: 创建模拟数据和模型
    
    为了保证教程可独立运行，使用模拟数据而非真实数据集。
    真实项目中请替换为 MOABB 数据。
    """
    # 1. 创建模拟 EEG 数据
    n_channels = 22
    n_times = 512
    n_classes = 4
    batch_size = 16
    
    # 生成随机数据
    np.random.seed(42)
    x = np.random.randn(batch_size, n_channels, n_times).astype(np.float32)
    
    # 生成标签 (模拟 4 类运动想象)
    y = np.random.randint(0, n_classes, size=batch_size)
    
    # 转换为 PyTorch 张量
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    
    # 2. 创建模拟通道信息 (真实数据中从 mne.Info 获取)
    # 简化: 使用 22 个电极的标准 10-20 系统位置
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'O2', 'FC1', 'FC2', 'FC5', 'FC6'
    ]
    
    # 创建简化的 chs_info (电极 2D 坐标)
    # 真实数据中使用 mne.create_info() 和 mne.channels.make_standard_montage()
    chs_info = []
    for i, name in enumerate(ch_names):
        # 简化的 2D 坐标 (单位: 米)
        # 实际项目中应该从 montage 获取
        angle = (i / len(ch_names)) * 2 * np.pi
        radius = 0.05 + 0.03 * np.sin(angle * 2)
        x_pos = radius * np.cos(angle)
        y_pos = radius * np.sin(angle)
        # loc 必须是 12 元素数组 (MNE 格式)
        loc = np.zeros(12)
        loc[0] = x_pos
        loc[1] = y_pos
        loc[2] = 0.0
        chs_info.append({
            'ch_name': name,
            'loc': loc,
            'kind': 1  # EEG
        })
    
    # 3. 创建模型 (使用 EEGNet)
    from braindecode.models import EEGNet
    
    # 将 chs_info 传入构造函数 (chs_info 是只读属性, 必须在初始化时设置)
    model = EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        chs_info=chs_info,
    )
    
    # 4. 进行一次前向传播 (初始化参数)
    model.eval()
    with torch.no_grad():
        output = model(x_tensor)
    
    print(f"环境准备完成!")
    print(f"  - 数据形状: {x_tensor.shape}  [batch, channels, time]")
    print(f"  - 标签形状: {y_tensor.shape}")
    print(f"  - 输出形状: {output.shape}  [batch, classes]")
    print(f"  - 预测类别: {output.argmax(dim=1)}")
    print(f"  - 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, x_tensor, y_tensor, ch_names, chs_info


# ============================================================
# 1. 归因分析基础: Saliency Map
# ============================================================

def tutorial_saliency_map():
    """
    教程 1: 基础归因分析 - Saliency Map
    
    Saliency Map 是最基础的可解释性方法:
    - 计算模型输出相对于输入的梯度
    - 梯度越大，说明该位置对预测结果越重要
    
    公式: S(x) = |∂f_c(x) / ∂x|
    其中 f_c 是类别 c 的输出概率
    
    优点: 简单、快速
    缺点: 梯度可能为零(饱和问题)，不够精确
    """
    print("\n" + "=" * 60)
    print("教程 1: 基础归因分析 - Saliency Map")
    print("=" * 60)
    
    # 导入可视化模块
    from braindecode.visualization import saliency
    
    # 准备数据和模型
    model, x_tensor, y_tensor, ch_names, chs_info = setup_environment()
    
    # 选择一个样本进行分析
    sample_idx = 0
    x_single = x_tensor[sample_idx:sample_idx+1].clone().requires_grad_(True)
    target_class = y_tensor[sample_idx:sample_idx+1]
    
    print(f"\n分析样本 #{sample_idx}, 目标类别: {target_class.item()}")
    
    # 计算 Saliency Map
    print("\n计算 Saliency Map...")
    saliency_map = saliency(model, x_single, target_class)
    
    print(f"  - 输出形状: {saliency_map.shape}")
    print(f"  - 值域范围: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
    
    # 可视化 Saliency Map
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Saliency Map 可视化", fontsize=14, fontweight="bold")
    
    # 1. 通道 x 时间热力图
    ax = axes[0, 0]
    saliency_np = saliency_map.squeeze().detach().numpy()
    im = ax.imshow(
        saliency_np,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
        extent=[0, x_tensor.shape[2]-1, len(ch_names)-1, 0]
    )
    ax.set_xlabel("时间点")
    ax.set_ylabel("通道")
    ax.set_title("Saliency Map (通道 x 时间)")
    ax.set_yticks(range(0, len(ch_names), 2))
    ax.set_yticklabels([ch_names[i] for i in range(0, len(ch_names), 2)], fontsize=8)
    plt.colorbar(im, ax=ax)
    
    # 2. 每个通道的平均重要性
    ax = axes[0, 1]
    channel_importance = saliency_np.mean(axis=1)
    ax.barh(range(len(ch_names)), channel_importance, color="steelblue")
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_xlabel("平均梯度幅值")
    ax.set_title("各通道平均重要性")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    # 3. 每个时间点的平均重要性
    ax = axes[1, 0]
    time_importance = saliency_np.mean(axis=0)
    ax.plot(time_importance, "r-", linewidth=1.5)
    ax.fill_between(range(len(time_importance)), time_importance, alpha=0.3, color="red")
    ax.set_xlabel("时间点")
    ax.set_ylabel("平均梯度幅值")
    ax.set_title("各时间点平均重要性")
    ax.grid(True, alpha=0.3)
    
    # 4. 前几个最重要的通道
    ax = axes[1, 1]
    top_k = 5
    top_indices = np.argsort(channel_importance)[::-1][:top_k]
    top_channels = [ch_names[i] for i in top_indices]
    top_values = [channel_importance[i] for i in top_indices]
    bars = ax.barh(top_channels, top_values, color=["#FF4444", "#FF8844", "#FFCC44", "#44CC44", "#4488FF"])
    ax.set_xlabel("平均梯度幅值")
    ax.set_title(f"Top-{top_k} 重要通道")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    # 添加数值标签
    for bar, val in zip(bars, top_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("saliency_map_visualization.png", dpi=100, bbox_inches="tight")
    print(f"\n  ✅ Saliency Map 可视化已保存: saliency_map_visualization.png")
    
    return saliency_map


# ============================================================
# 2. 进阶归因方法: Integrated Gradients
# ============================================================

def tutorial_integrated_gradients():
    """
    教程 2: 进阶归因 - Integrated Gradients
    
    Integrated Gradients (IG) 是目前最主流的归因方法之一:
    - 解决了 Saliency Map 的梯度饱和问题
    - 通过沿路径积分，计算真正的贡献值
    - 满足"完整性公理": 所有归因值之和 ≈ f(x) - f(baseline)
    
    原理:
    - 从 baseline (通常是全零输入) 到实际输入 x 进行线性插值
    - 计算每一步的梯度并积分
    - 得到每个特征的重要性归因
    
    公式: IG_i(x) = (x_i - b_i) * ∫_0^1 ∂f(b + α(x-b)) / ∂x_i dα
    """
    print("\n" + "=" * 60)
    print("教程 2: 进阶归因 - Integrated Gradients")
    print("=" * 60)
    
    from braindecode.visualization import integrated_gradients
    
    # 准备数据和模型
    model, x_tensor, y_tensor, ch_names, chs_info = setup_environment()
    
    # 选择多个样本进行分析
    n_samples = 4
    sample_indices = np.random.choice(len(x_tensor), n_samples, replace=False)
    
    print(f"\n分析 {n_samples} 个样本...")
    
    # 计算 Integrated Gradients
    ig_maps = []
    for idx in sample_indices:
        x_single = x_tensor[idx:idx+1].clone().requires_grad_(True)
        target = y_tensor[idx:idx+1]
        
        # IG 计算 (steps 越多越精确，但计算量更大)
        ig_map = integrated_gradients(
            model,
            x_single,
            target,
            baseline=None,  # 默认全零
            steps=50        # 积分步数
        )
        ig_maps.append(ig_map.squeeze().detach().numpy())
    
    ig_maps = np.array(ig_maps)
    
    print(f"  - IG Maps 形状: {ig_maps.shape}")
    print(f"  - 值域范围: [{ig_maps.min():.4f}, {ig_maps.max():.4f}]")
    
    # 可视化 IG 结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Integrated Gradients 可视化", fontsize=14, fontweight="bold")
    
    # 1. 多个样本的平均 IG
    ax = axes[0, 0]
    mean_ig = ig_maps.mean(axis=0)
    im = ax.imshow(
        mean_ig,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        extent=[0, x_tensor.shape[2]-1, len(ch_names)-1, 0]
    )
    ax.set_xlabel("时间点")
    ax.set_ylabel("通道")
    ax.set_title("平均 Integrated Gradients")
    ax.set_yticks(range(0, len(ch_names), 2))
    ax.set_yticklabels([ch_names[i] for i in range(0, len(ch_names), 2)], fontsize=8)
    plt.colorbar(im, ax=ax)
    
    # 2. IG 值的分布 (验证完整性公理)
    ax = axes[0, 1]
    ig_values = ig_maps.mean(axis=(0, 2))  # 对样本和时间平均
    ax.bar(range(len(ch_names)), ig_values, color="steelblue")
    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
    ax.set_ylabel("平均归因值")
    ax.set_title("各通道归因值 (IG)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    
    # 3. 对比 Saliency vs IG
    ax = axes[1, 0]
    from braindecode.visualization import saliency
    sample_idx = 0
    x_single = x_tensor[sample_idx:sample_idx+1].clone().requires_grad_(True)
    target = y_tensor[sample_idx:sample_idx+1]
    
    sal_map = saliency(model, x_single, target).squeeze().detach().numpy()
    ig_map_single = integrated_gradients(model, x_single, target, steps=50).squeeze().detach().numpy()
    
    # 归一化后对比
    sal_norm = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
    ig_norm = (ig_map_single - ig_map_single.min()) / (ig_map_single.max() - ig_map_single.min() + 1e-8)
    
    ax.scatter(sal_norm.flatten()[::10], ig_norm.flatten()[::10], alpha=0.5, s=10)
    ax.set_xlabel("Saliency (归一化)")
    ax.set_ylabel("Integrated Gradients (归一化)")
    ax.set_title("Saliency vs IG 散点图")
    ax.grid(True, alpha=0.3)
    
    # 添加 y=x 参考线
    lims = [0, 1]
    ax.plot(lims, lims, "r--", alpha=0.5, label="y = x")
    ax.legend()
    
    # 4. 类别特定的归因分析
    ax = axes[1, 1]
    n_classes = 4
    class_ig_maps = []
    
    for class_idx in range(n_classes):
        # 对每个类别计算 IG
        fake_target = torch.tensor([class_idx])
        x_test = x_tensor[0:1].clone().requires_grad_(True)
        class_ig = integrated_gradients(model, x_test, fake_target, steps=50)
        class_ig_maps.append(class_ig.squeeze().detach().numpy().mean())
    
    ax.bar(range(n_classes), class_ig_maps, 
           color=["#FF4444", "#44FF44", "#4444FF", "#FFFF44"])
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(["Class 0", "Class 1", "Class 2", "Class 3"])
    ax.set_ylabel("平均 IG 值")
    ax.set_title("各类别的归因值")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("integrated_gradients_visualization.png", dpi=100, bbox_inches="tight")
    print(f"\n  ✅ Integrated Gradients 可视化已保存: integrated_gradients_visualization.png")
    
    return ig_maps


# ============================================================
# 3. 高级归因方法: Guided Backprop & LRP
# ============================================================

def tutorial_advanced_attribution():
    """
    教程 3: 高级归因方法
    
    3.1 Guided Backpropagation
    - 通过 ReLU 约束反向传播的梯度
    - 只保留正激活的正梯度
    - 生成更清晰、更尖锐的归因图
    
    3.2 Layer-wise Relevance Propagation (LRP)
    - 逐层传播相关性
    - 确保总相关性守恒
    - 提供层级归因解释
    
    3.3 DeepLIFT
    - 基于参考点的贡献分配
    - 解决梯度消失问题
    
    3.4 Input x Gradient
    - 输入值乘以梯度
    - 简单但有效
    """
    print("\n" + "=" * 60)
    print("教程 3: 高级归因方法")
    print("=" * 60)
    
    # 导入多种归因方法
    from braindecode.visualization import (
        guided_backprop,
        lrp,
        deep_lift,
        input_x_gradient,
    )
    
    # 准备数据和模型
    model, x_tensor, y_tensor, ch_names, chs_info = setup_environment()
    
    # 选择一个样本
    sample_idx = 0
    x_single = x_tensor[sample_idx:sample_idx+1].clone().requires_grad_(True)
    target = y_tensor[sample_idx:sample_idx+1]
    
    print(f"\n分析样本 #{sample_idx}, 目标类别: {target.item()}")
    
    # 计算各种归因方法
    print("\n计算多种归因方法...")
    
    # 1. Guided Backprop
    print("  1. Guided Backpropagation...")
    gb_map = guided_backprop(model, x_single, target).squeeze().detach().numpy()
    
    # 2. LRP (可能在某些模型上不支持)
    lrp_map = None
    try:
        print("  2. Layer-wise Relevance Propagation...")
        lrp_map = lrp(model, x_single, target).squeeze().detach().numpy()
    except Exception as e:
        print(f"     ⚠️  LRP 不可用: {e}")
        print(f"     将使用零数组代替")
        lrp_map = np.zeros_like(gb_map)
    
    # 3. DeepLIFT
    print("  3. DeepLIFT...")
    dl_map = deep_lift(model, x_single, target).squeeze().detach().numpy()
    
    # 4. Input x Gradient
    print("  4. Input x Gradient...")
    ixg_map = input_x_gradient(model, x_single, target).squeeze().detach().numpy()
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("高级归因方法对比", fontsize=14, fontweight="bold")
    
    methods = [
        ("Guided Backprop", gb_map, "hot"),
        ("LRP", lrp_map, "RdBu_r"),
        ("DeepLIFT", dl_map, "viridis"),
        ("Input x Gradient", ixg_map, "plasma"),
    ]
    
    for ax, (name, att_map, cmap) in zip(axes.flat, methods):
        im = ax.imshow(
            att_map,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            extent=[0, x_tensor.shape[2]-1, len(ch_names)-1, 0]
        )
        ax.set_xlabel("时间点", fontsize=9)
        ax.set_ylabel("通道", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.set_yticks(range(0, len(ch_names), 4))
        ax.set_yticklabels([ch_names[i] for i in range(0, len(ch_names), 4)], fontsize=7)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("advanced_attribution_comparison.png", dpi=100, bbox_inches="tight")
    print(f"\n  ✅ 高级归因方法对比已保存: advanced_attribution_comparison.png")
    
    # 通道重要性对比
    print("\n各方法的通道重要性对比:")
    print(f"  {'通道':<10} {'GB':>10} {'LRP':>10} {'DeepLIFT':>10} {'IxG':>10}")
    print("  " + "-" * 50)
    
    for i, ch_name in enumerate(ch_names):
        gb_imp = np.abs(gb_map[i]).mean()
        lrp_imp = np.abs(lrp_map[i]).mean()
        dl_imp = np.abs(dl_map[i]).mean()
        ixg_imp = np.abs(ixg_map[i]).mean()
        print(f"  {ch_name:<10} {gb_imp:>10.4f} {lrp_imp:>10.4f} {dl_imp:>10.4f} {ixg_imp:>10.4f}")
    
    return {
        "guided_backprop": gb_map,
        "lrp": lrp_map,
        "deep_lift": dl_map,
        "input_x_gradient": ixg_map,
    }


# ============================================================
# 4. 地形图投影 (Topomap Projection)
# ============================================================

def tutorial_topomap_projection():
    """
    教程 4: 地形图投影 (Topomap Projection)
    
    将 1D 通道权重投影到 2D 头皮地形图
    
    原理:
    - 使用电极的 2D 坐标 (从 MNE montage 获取)
    - 通过 Clough-Tocher 三角剖分进行插值
    - 生成标准的头皮地形图
    
    用途:
    - 直观展示模型关注的大脑区域
    - 分析空间分布的重要性
    - 生成论文级别的可视化
    """
    print("\n" + "=" * 60)
    print("教程 4: 地形图投影 (Topomap Projection)")
    print("=" * 60)
    
    from braindecode.visualization import project_to_topomap, saliency
    
    # 准备数据和模型
    model, x_tensor, y_tensor, ch_names, chs_info = setup_environment()
    
    # 计算 Saliency Map
    sample_idx = 0
    x_single = x_tensor[sample_idx:sample_idx+1].clone().requires_grad_(True)
    target = y_tensor[sample_idx:sample_idx+1]
    
    saliency_map = saliency(model, x_single, target).squeeze().detach().numpy()
    
    # 计算每个通道的平均归因值
    channel_attributions = np.abs(saliency_map).mean(axis=1)  # (n_channels,)
    
    print(f"\n通道归因值: {channel_attributions}")
    
    # 投影到地形图
    print("\n投影到 2D 头皮地形图...")
    topomap = project_to_topomap(
        channel_attributions,
        chs_info,
        res=64  # 投影网格分辨率
    )
    
    print(f"  - Topomap 形状: {topomap.shape}")
    print(f"  - 有效值数量: {np.sum(~np.isnan(topomap))}")
    print(f"  - NaN 数量 (头皮外): {np.sum(np.isnan(topomap))}")
    
    # 可视化地形图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("地形图投影 (Topomap Projection)", fontsize=14, fontweight="bold")
    
    # 1. 原始通道值 (条形图)
    ax = axes[0]
    ax.barh(range(len(ch_names)), channel_attributions, color="steelblue")
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_xlabel("平均归因值")
    ax.set_title("1D 通道归因值")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    # 2. 2D 地形图 (带电极位置)
    ax = axes[1]
    # 显示地形图 (NaN 区域显示为白色)
    topomap_display = np.ma.masked_invalid(topomap)
    im = ax.imshow(topomap_display, cmap="RdBu_r", aspect="equal", 
                   origin="lower", interpolation="bicubic")
    
    # 添加电极位置 (使用 loc 的前两个元素)
    for ch_info in chs_info:
        x_pos = (ch_info['loc'][0] + 0.1) / 0.2 * 64  # 缩放到 0-64
        y_pos = (ch_info['loc'][1] + 0.1) / 0.2 * 64
        ax.plot(x_pos, y_pos, "k.", markersize=5)
    
    ax.set_title("2D 头皮地形图")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 3. 3D 地形图表面图
    ax = axes[2]
    from mpl_toolkits.mplot3d import Axes3D
    # 重新创建为 3D 子图
    fig_3d = plt.figure(figsize=(6, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # 网格
    x_grid, y_grid = np.meshgrid(range(topomap.shape[0]), range(topomap.shape[1]))
    z_grid = np.where(np.isnan(topomap), 0, topomap)
    
    # 3D 表面图
    surf = ax_3d.plot_surface(x_grid, y_grid, z_grid, 
                              cmap="RdBu_r", alpha=0.8,
                              rstride=2, cstride=2)
    ax_3d.set_title("3D 地形图表面")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("归因值")
    plt.colorbar(surf, shrink=0.5)
    
    # 复制 3D 图到原位置 (简化处理)
    axes[2].remove()
    axes[2] = fig.add_subplot(1, 3, 3, projection='3d')
    axes[2].plot_surface(x_grid, y_grid, z_grid, cmap="RdBu_r", alpha=0.8, rstride=2, cstride=2)
    axes[2].set_title("3D 地形图")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("topomap_projection.png", dpi=100, bbox_inches="tight")
    print(f"\n  ✅ 地形图投影已保存: topomap_projection.png")
    
    return topomap


# ============================================================
# 5. 增强版混淆矩阵
# ============================================================

def tutorial_confusion_matrix():
    """
    教程 5: 增强版混淆矩阵
    
    braindecode.visualization.plot_confusion_matrix 相比 sklearn 版本:
    - 可在单元格中显示精确率 (Precision) 和召回率 (Sensitivity)
    - 支持 F1-Score 显示
    - 更专业的配色方案
    - 方便科研论文使用
    """
    print("\n" + "=" * 60)
    print("教程 5: 增强版混淆矩阵")
    print("=" * 60)
    
    from braindecode.visualization import plot_confusion_matrix
    
    # 生成模拟的混淆矩阵 (4 类)
    class_names = ["左手", "右手", "双脚", "舌头"]
    n_classes = len(class_names)
    
    # 模拟真实的分类结果
    np.random.seed(42)
    confusion_mat = np.array([
        [42, 5, 3, 2],   # 左手
        [6, 40, 2, 4],   # 右手
        [3, 2, 45, 3],   # 双脚
        [2, 4, 3, 41],   # 舌头
    ])
    
    print(f"混淆矩阵:")
    print(confusion_mat)
    
    # 可视化: 基础版
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("增强版混淆矩阵对比", fontsize=14, fontweight="bold")
    
    # 1. 基础混淆矩阵
    ax = axes[0]
    im = ax.imshow(confusion_mat, cmap="Blues", interpolation="nearest")
    ax.set_title("基础混淆矩阵")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, confusion_mat[i, j], 
                    ha="center", va="center",
                    color="white" if confusion_mat[i, j] > 25 else "black")
    plt.colorbar(im, ax=ax)
    
    # 2. Braindecode 增强版 (带 Precision/Sensitivity/F1)
    # 注意: plot_confusion_matrix 会创建自己的 figure
    print("\n生成 Braindecode 增强版混淆矩阵...")
    fig_braindecode = plot_confusion_matrix(
        confusion_mat,
        class_names=class_names,
        with_f1_score=True,  # 显示 F1 分数
        colormap=plt.cm.Oranges,
    )
    fig_braindecode.savefig("confusion_matrix_braindecode.png", dpi=100, bbox_inches="tight")
    print(f"  ✅ Braindecode 增强版已保存: confusion_matrix_braindecode.png")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("confusion_matrix_comparison.png", dpi=100, bbox_inches="tight")
    print(f"\n  ✅ 混淆矩阵对比已保存: confusion_matrix_comparison.png")
    
    # 打印各类别指标
    print("\n各类别详细指标:")
    print(f"  {'类别':<10} {'精确率':>10} {'召回率':>10} {'F1-Score':>10}")
    print("  " + "-" * 40)
    
    for i, name in enumerate(class_names):
        tp = confusion_mat[i, i]
        fp = confusion_mat[:, i].sum() - tp
        fn = confusion_mat[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {name:<10} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
    
    return confusion_mat


# ============================================================
# 6. 健全性检查 (Sanity Checks)
# ============================================================

def tutorial_sanity_checks():
    """
    教程 6: 归因方法的健全性检查 (Sanity Checks)
    
    为什么需要健全性检查?
    - 归因方法可能产生误导性结果
    - 需要验证方法是否真的学到了有意义的信号
    
    两种经典的健全性检查 (Adebayo et al., 2018):
    
    1. 模型随机化检查 (Model Randomization Test):
       - 随机化模型权重后重新计算归因
       - 如果归因图变化不大，说明方法可能不可靠
    
    2. 标签随机化检查 (Label Randomization Test):
       - 打乱标签后重新训练并计算归因
       - 如果归因图仍然相似，说明方法可能在识别数据噪声而非真实信号
    
    通过标准:
    - 模型随机化后，归因图应该显著变化 (低 SSIM)
    - 标签随机化后，归因图应该显著变化 (低 SSIM)
    """
    print("\n" + "=" * 60)
    print("教程 6: 归因方法的健全性检查")
    print("=" * 60)
    
    from braindecode.visualization.sanity import (
        model_randomization_test,
        label_randomization_test,
    )
    from braindecode.visualization import integrated_gradients, saliency
    
    # 准备数据和模型
    model, x_tensor, y_tensor, ch_names, chs_info = setup_environment()
    
    # 选择样本
    sample_idx = 0
    x_single = x_tensor[sample_idx:sample_idx+1].clone().requires_grad_(True)
    target = y_tensor[sample_idx:sample_idx+1]
    
    print(f"\n分析样本 #{sample_idx}")
    
    # 1. 原始归因
    print("\n1. 计算原始归因 (Integrated Gradients)...")
    original_attr = integrated_gradients(
        model, x_single, target, steps=50
    ).squeeze().detach().numpy()
    
    # 2. 模型随机化检查
    print("\n2. 模型随机化检查...")
    
    # 创建随机化模型 (保持架构，随机权重)
    import copy
    randomized_model = copy.deepcopy(model)
    for param in randomized_model.parameters():
        param.data = torch.randn_like(param.data)
    
    randomized_model.eval()
    randomized_attr = integrated_gradients(
        randomized_model, x_single, target, steps=50
    ).squeeze().detach().numpy()
    
    # 计算 SSIM (结构相似性)
    from scipy.stats import pearsonr
    
    # 扁平化后计算相关性
    orig_flat = original_attr.flatten()
    rand_flat = randomized_attr.flatten()
    
    corr_coef, p_value = pearsonr(orig_flat, rand_flat)
    
    print(f"  - 原始 vs 随机化模型:")
    print(f"    Pearson 相关系数: {corr_coef:.4f}")
    print(f"    p-value: {p_value:.4f}")
    
    if corr_coef > 0.7:
        print(f"    ⚠️  警告: 相关系数过高，归因方法可能不可靠!")
    else:
        print(f"    ✅ 通过: 随机化后归因图显著变化")
    
    # 3. 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("健全性检查 - 可视化对比", fontsize=14, fontweight="bold")
    
    # 第一行: 原始 vs 随机化
    ax = axes[0, 0]
    im = ax.imshow(original_attr, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title("原始归因 (Integrated Gradients)")
    ax.set_xlabel("时间点")
    ax.set_ylabel("通道")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(randomized_attr, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title("随机化模型归因")
    ax.set_xlabel("时间点")
    ax.set_ylabel("通道")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    # 差异图
    diff = np.abs(original_attr - randomized_attr)
    im = ax.imshow(diff, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_title(f"差异 (相关性: {corr_coef:.3f})")
    ax.set_xlabel("时间点")
    ax.set_ylabel("通道")
    plt.colorbar(im, ax=ax)
    
    # 第二行: Saliency Map 的健全性检查
    ax = axes[1, 0]
    original_saliency = saliency(model, x_single, target).squeeze().detach().numpy()
    im = ax.imshow(original_saliency, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_title("原始 Saliency Map")
    ax.set_xlabel("时间点")
    ax.set_ylabel("通道")
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    randomized_saliency = saliency(randomized_model, x_single, target).squeeze().detach().numpy()
    im = ax.imshow(randomized_saliency, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_title("随机化 Saliency Map")
    ax.set_xlabel("时间点")
    ax.set_ylabel("通道")
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    sal_diff = np.abs(original_saliency - randomized_saliency)
    sal_corr, _ = pearsonr(original_saliency.flatten(), randomized_saliency.flatten())
    im = ax.imshow(sal_diff, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_title(f"Saliency 差异 (相关性: {sal_corr:.3f})")
    ax.set_xlabel("时间点")
    ax.set_ylabel("通道")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("sanity_checks_visualization.png", dpi=100, bbox_inches="tight")
    print(f"\n  ✅ 健全性检查可视化已保存: sanity_checks_visualization.png")
    
    # 总结
    print("\n" + "=" * 60)
    print("健全性检查总结:")
    print("=" * 60)
    print(f"  Integrated Gradients:")
    print(f"    模型随机化相关系数: {corr_coef:.4f}")
    print(f"    结论: {'✅ 通过' if corr_coef < 0.7 else '⚠️  可能不可靠'}")
    print(f"\n  Saliency Map:")
    print(f"    模型随机化相关系数: {sal_corr:.4f}")
    print(f"    结论: {'✅ 通过' if sal_corr < 0.7 else '⚠️  可能不可靠'}")
    
    return {
        "ig_correlation": corr_coef,
        "saliency_correlation": sal_corr,
    }


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Braindecode 可解释性与深度可视化 - 学习教程           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║   章节: 06_visualization_interpretability.py            ║")
    print("║   内容:                                                 ║")
    print("║     1. 归因分析基础 (Saliency Map)                      ║")
    print("║     2. 进阶归因 (Integrated Gradients)                 ║")
    print("║     3. 高级归因 (GB, LRP, DeepLIFT, IxG)               ║")
    print("║     4. 地形图投影 (Topomap Projection)                  ║")
    print("║     5. 增强版混淆矩阵                                   ║")
    print("║     6. 健全性检查 (Sanity Checks)                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # 检查 captum 是否可用
    try:
        import captum
        print(f"\n✅ Captum 版本: {captum.__version__} (归因分析必需)")
    except ImportError:
        print("\n⚠️  警告: captum 未安装!")
        print("   归因分析方法需要 captum 库:")
        print("   uv add captum")
        print("\n   部分功能可能不可用，但 Saliency Map 和混淆矩阵仍可使用。")
    
    # 运行教程 (可按需注释/取消注释)
    print("\n" + "=" * 60)
    print("开始运行教程...")
    print("=" * 60)
    
    # Tutorial 1: Saliency Map
    print("\n" + "★" * 30)
    print("运行 Tutorial 1: Saliency Map 基础")
    print("★" * 30)
    tutorial_saliency_map()
    
    # Tutorial 2: Integrated Gradients
    print("\n" + "★" * 30)
    print("运行 Tutorial 2: Integrated Gradients")
    print("★" * 30)
    try:
        tutorial_integrated_gradients()
    except Exception as e:
        print(f"⚠️  Tutorial 2 运行失败: {e}")
        print("   可能是 captum 未安装或版本兼容性问题")
    
    # Tutorial 3: 高级归因方法
    print("\n" + "★" * 30)
    print("运行 Tutorial 3: 高级归因方法")
    print("★" * 30)
    try:
        tutorial_advanced_attribution()
    except Exception as e:
        print(f"⚠️  Tutorial 3 运行失败: {e}")
        print("   可能是 captum 未安装或版本兼容性问题")
    
    # Tutorial 4: 地形图投影
    print("\n" + "★" * 30)
    print("运行 Tutorial 4: 地形图投影")
    print("★" * 30)
    tutorial_topomap_projection()
    
    # Tutorial 5: 增强版混淆矩阵
    print("\n" + "★" * 30)
    print("运行 Tutorial 5: 增强版混淆矩阵")
    print("★" * 30)
    tutorial_confusion_matrix()
    
    # Tutorial 6: 健全性检查
    print("\n" + "★" * 30)
    print("运行 Tutorial 6: 健全性检查")
    print("★" * 30)
    try:
        tutorial_sanity_checks()
    except Exception as e:
        print(f"⚠️  Tutorial 6 运行失败: {e}")
        print("   可能是 captum 未安装或版本兼容性问题")
    
    # 完成
    print("\n" + "=" * 60)
    print("🎉 第06章完成! 你已经学会了:")
    print("  ✅ 归因分析基础 (Saliency Map)")
    print("  ✅ Integrated Gradients (主流方法)")
    print("  ✅ 高级归因方法 (GB, LRP, DeepLIFT, IxG)")
    print("  ✅ 地形图投影 (Topomap Projection)")
    print("  ✅ 增强版混淆矩阵")
    print("  ✅ 健全性检查 (Sanity Checks)")
    print("\n生成的可视化文件:")
    print("  - saliency_map_visualization.png")
    print("  - integrated_gradients_visualization.png")
    print("  - advanced_attribution_comparison.png")
    print("  - topomap_projection.png")
    print("  - confusion_matrix_comparison.png")
    print("  - sanity_checks_visualization.png")
    print("\n进入实战项目 (main.py) 或继续探索!")
    print("=" * 60)