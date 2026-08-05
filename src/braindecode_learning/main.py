"""
Braindecode 学习教程 - 综合实战项目
======================================================

本项目完成一个完整的运动想象分类任务流程:

1. 数据加载 (MOABB)
2. 数据预处理 (滤波、重采样)
3. 数据窗口化与划分
4. 模型构建 (EEGNet)
5. 模型训练
6. 模型评估
7. 结果可视化

数据集: BCI Competition IV Dataset 2a
任务: 4 类运动想象分类 (左手、右手、双脚、舌头)
"""

import os
import sys
import time
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================
# 配置参数
# ============================================================

class Config:
    """项目配置"""
    
    # 数据参数
    DATASET_NAME = "BNCI2014_001"  # BCI IV 2a
    N_CHANNELS = 22
    N_CLASSES = 4
    CLASS_NAMES = ["Left Hand", "Right Hand", "Feet", "Tongue"]
    
    # 预处理参数
    LOW_FREQ = 0.5       # 高通滤波截止频率
    HIGH_FREQ = 40.0     # 低通滤波截止频率
    NOTCH_FREQ = 50.0    # 陷波滤波频率 (工频)
    TARGET_SFREQ = 128   # 目标采样率
    
    # 窗口参数
    WINDOW_SECONDS = 4   # 窗口长度
    STRIDE_SECONDS = 1   # 窗口步长
    WINDOW_SAMPLES = WINDOW_SECONDS * TARGET_SFREQ  # 512
    STRIDE_SAMPLES = STRIDE_SECONDS * TARGET_SFREQ  # 128
    
    # 划分参数
    TRAIN_RATIO = 0.8    # 训练集比例
    VAL_RATIO = 0.1      # 验证集比例
    TEST_RATIO = 0.1     # 测试集比例
    RANDOM_SEED = 42
    
    # 模型参数
    MODEL_NAME = "EEGNetv4"
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    N_EPOCHS = 10
    WEIGHT_DECAY = 1e-4
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 输出目录
    OUTPUT_DIR = "output"
    
    @classmethod
    def summary(cls):
        print("\n" + "=" * 60)
        print("项目配置:")
        print("=" * 60)
        print(f"  数据集: {cls.DATASET_NAME}")
        print(f"  通道数: {cls.N_CHANNELS}")
        print(f"  类别数: {cls.N_CLASSES}")
        print(f"  采样率: {cls.TARGET_SFREQ} Hz")
        print(f"  窗口长度: {cls.WINDOW_SECONDS}s")
        print(f"  模型: {cls.MODEL_NAME}")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print(f"  批次大小: {cls.BATCH_SIZE}")
        print(f"  训练轮数: {cls.N_EPOCHS}")
        print(f"  设备: {cls.DEVICE}")
        print("=" * 60)


# ============================================================
# Step 1: 数据加载
# ============================================================

def step1_load_data():
    """Step 1: 加载 MOABB 数据集"""
    print("\n" + "=" * 60)
    print("STEP 1: 数据加载")
    print("=" * 60)
    
    from braindecode.datasets import MOABBDataset
    
    print(f"加载数据集: {Config.DATASET_NAME}")
    dataset = MOABBDataset(dataset_name=Config.DATASET_NAME)
    
    print(f"  ✅ 加载完成")
    print(f"     - 记录数量 (run): {len(dataset.datasets)}")
    print(f"     - 数据类型: {type(dataset)}")
    
    # 查看数据信息
    run_0 = dataset.datasets[0]
    raw = run_0.raw
    print(f"     - 通道数: {len(raw.ch_names)}")
    print(f"     - 采样率: {raw.info['sfreq']} Hz")
    print(f"     - 时长: {raw.n_times / raw.info['sfreq']:.1f}s")
    
    return dataset


# ============================================================
# Step 2: 数据预处理
# ============================================================

def step2_preprocess(dataset):
    """Step 2: 数据预处理"""
    print("\n" + "=" * 60)
    print("STEP 2: 数据预处理")
    print("=" * 60)
    
    from braindecode.preprocessing import preprocess, Preprocessor
    
    print("定义预处理步骤...")
    preprocessors = [
        Preprocessor("filter", l_freq=Config.LOW_FREQ, h_freq=Config.HIGH_FREQ, verbose=False),
        Preprocessor("notch_filter", freqs=[Config.NOTCH_FREQ], verbose=False),
        Preprocessor("resample", sfreq=Config.TARGET_SFREQ, verbose=False),
    ]
    
    print("应用预处理...")
    processed_dataset = preprocess(dataset, preprocessors)
    
    # 验证预处理结果
    raw = processed_dataset.datasets[0].raw
    print(f"  ✅ 预处理完成")
    print(f"     - 采样率: {raw.info['sfreq']} Hz")
    print(f"     - 通道数: {len(raw.ch_names)}")
    print(f"     - 数据范围: [{raw.get_data().min():.4f}, {raw.get_data().max():.4f}]")
    
    return processed_dataset


# ============================================================
# Step 3: 数据窗口化与划分
# ============================================================

def step3_window_and_split(dataset):
    """Step 3: 数据窗口化与划分"""
    print("\n" + "=" * 60)
    print("STEP 3: 数据窗口化与划分")
    print("=" * 60)
    
    from braindecode.preprocessing import create_windows_from_events
    
    print(f"窗口化参数:")
    print(f"  - 窗口长度: {Config.WINDOW_SECONDS}s ({Config.WINDOW_SAMPLES} samples)")
    print(f"  - 窗口步长: {Config.STRIDE_SECONDS}s ({Config.STRIDE_SAMPLES} samples)")
    
    print("\n创建窗口数据集...")
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=Config.WINDOW_SAMPLES,
        window_stride_samples=Config.STRIDE_SAMPLES,
        preload=True,
    )
    
    print(f"  ✅ 窗口化完成: {len(windows_dataset)} 个窗口")
    
    # 划分数据集
    print("\n划分数据集 (Train/Val/Test = 80/10/10)...")
    generator = torch.Generator().manual_seed(Config.RANDOM_SEED)
    
    n_total = len(windows_dataset)
    n_train = int(Config.TRAIN_RATIO * n_total)
    n_val = int(Config.VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val
    
    # 随机划分
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(windows_dataset, train_indices)
    val_dataset = Subset(windows_dataset, val_indices)
    test_dataset = Subset(windows_dataset, test_indices)
    
    print(f"  ✅ 划分完成")
    print(f"     - 训练集: {len(train_dataset)} 窗口")
    print(f"     - 验证集: {len(val_dataset)} 窗口")
    print(f"     - 测试集: {len(test_dataset)} 窗口")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# Step 4: 模型构建
# ============================================================

def step4_build_model():
    """Step 4: 构建模型"""
    print("\n" + "=" * 60)
    print("STEP 4: 模型构建")
    print("=" * 60)
    
    from braindecode.models import EEGNetv4
    
    print(f"构建模型: {Config.MODEL_NAME}")
    print(f"  - 输入通道: {Config.N_CHANNELS}")
    print(f"  - 输出类别: {Config.N_CLASSES}")
    print(f"  - 时间点数: {Config.WINDOW_SAMPLES}")
    
    model = EEGNetv4(
        n_chans=Config.N_CHANNELS,
        n_outputs=Config.N_CLASSES,
        n_times=Config.WINDOW_SAMPLES,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✅ 模型构建完成")
    print(f"     - 总参数量: {n_params:,}")
    print(f"     - 可训练参数: {n_trainable:,}")
    
    # 打印模型结构
    print(f"\n模型结构:")
    print(model)
    
    # 移到设备
    model = model.to(Config.DEVICE)
    
    return model


# ============================================================
# Step 5: 模型训练
# ============================================================

def step5_train(model, train_loader, val_loader):
    """Step 5: 模型训练"""
    print("\n" + "=" * 60)
    print("STEP 5: 模型训练")
    print("=" * 60)
    
    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True,
    )
    
    print(f"训练配置:")
    print(f"  - 优化器: Adam (lr={Config.LEARNING_RATE}, weight_decay={Config.WEIGHT_DECAY})")
    print(f"  - 损失函数: CrossEntropyLoss")
    print(f"  - 学习率调度: ReduceLROnPlateau")
    print(f"  - 训练轮数: {Config.N_EPOCHS}")
    print(f"  - 设备: {Config.DEVICE}")
    
    # 训练历史
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    
    best_val_acc = 0.0
    best_state = None
    
    print(f"\n开始训练...")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<10}")
    print("-" * 66)
    
    for epoch in range(Config.N_EPOCHS):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(Config.DEVICE)
            batch_y = batch_y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(Config.DEVICE)
                batch_y = batch_y.to(Config.DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        epoch_time = time.time() - epoch_start
        
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {current_lr:<10.6f} ({epoch_time:.1f}s)")
    
    # 加载最佳模型
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  ✅ 训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    
    return model, history


# ============================================================
# Step 6: 模型评估
# ============================================================

def step6_evaluate(model, test_loader):
    """Step 6: 模型评估"""
    print("\n" + "=" * 60)
    print("STEP 6: 模型评估")
    print("=" * 60)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(Config.DEVICE)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    
    print(f"\n测试集评估结果:")
    print(f"  - 样本数量: {len(all_labels)}")
    print(f"  - 准确率 (Accuracy): {accuracy:.4f} ({accuracy:.2%})")
    print(f"  - 平衡准确率 (Balanced Accuracy): {balanced_acc:.4f} ({balanced_acc:.2%})")
    
    print(f"\n分类报告:")
    report = classification_report(
        all_labels, all_predictions,
        target_names=Config.CLASS_NAMES,
        digits=4,
    )
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"混淆矩阵:")
    print(cm)
    
    return {
        "predictions": all_predictions,
        "labels": all_labels,
        "accuracy": accuracy,
        "balanced_acc": balanced_acc,
        "confusion_matrix": cm,
    }


# ============================================================
# Step 7: 结果可视化
# ============================================================

def step7_visualize(history, eval_results, model):
    """Step 7: 结果可视化"""
    print("\n" + "=" * 60)
    print("STEP 7: 结果可视化")
    print("=" * 60)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Braindecode 运动想象分类 - 训练结果", fontsize=14, fontweight="bold")
    
    # 1. 训练/验证损失曲线
    ax = axes[0, 0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-", linewidth=2, label="Val Loss")
    best_epoch = np.argmin(history["val_loss"]) + 1
    ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7,
               label=f"Best Epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练/验证损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 训练/验证准确率曲线
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], "b-", linewidth=2, label="Train Acc")
    ax.plot(epochs, history["val_acc"], "r-", linewidth=2, label="Val Acc")
    best_epoch_acc = np.argmax(history["val_acc"]) + 1
    ax.axvline(x=best_epoch_acc, color="g", linestyle="--", alpha=0.7,
               label=f"Best Epoch ({best_epoch_acc})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("训练/验证准确率曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 混淆矩阵
    ax = axes[1, 0]
    cm = eval_results["confusion_matrix"]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("混淆矩阵")
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_xticks(range(Config.N_CLASSES))
    ax.set_yticks(range(Config.N_CLASSES))
    ax.set_xticklabels(Config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(Config.CLASS_NAMES)
    
    # 添加数值
    for i in range(Config.N_CLASSES):
        for j in range(Config.N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    
    # 4. 各类别性能柱状图
    ax = axes[1, 1]
    per_class_accuracy = []
    for i in range(Config.N_CLASSES):
        class_correct = cm[i, i]
        class_total = cm[i].sum()
        per_class_accuracy.append(class_correct / class_total if class_total > 0 else 0)
    
    bars = ax.bar(Config.CLASS_NAMES, per_class_accuracy,
                  color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
                  edgecolor="black")
    ax.set_title("各类别准确率")
    ax.set_ylabel("准确率")
    ax.set_ylim(0, 1.0)
    
    # 添加数值标签
    for bar, acc in zip(bars, per_class_accuracy):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f"{acc:.2%}", ha="center", va="bottom", fontsize=10)
    
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存图片
    save_path = os.path.join(Config.OUTPUT_DIR, "training_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✅ 可视化已保存: {save_path}")
    
    # 保存模型
    model_path = os.path.join(Config.OUTPUT_DIR, "model_best.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "n_chans": Config.N_CHANNELS,
            "n_outputs": Config.N_CLASSES,
            "n_times": Config.WINDOW_SAMPLES,
            "model_name": Config.MODEL_NAME,
        },
        "eval_accuracy": eval_results["accuracy"],
    }, model_path)
    print(f"  ✅ 模型已保存: {model_path}")
    
    return save_path, model_path


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数: 运行完整的训练流程"""
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Braindecode 运动想象分类 - 综合实战项目          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  数据集: BCI Competition IV Dataset 2a                  ║")
    print("║  任务:   4 类运动想象分类                               ║")
    print("║  模型:   EEGNetv4                                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # 打印配置
    Config.summary()
    
    # 记录开始时间
    start_time = time.time()
    
    # Step 1: 数据加载
    dataset = step1_load_data()
    
    # Step 2: 数据预处理
    processed_dataset = step2_preprocess(dataset)
    
    # Step 3: 数据窗口化与划分
    train_loader, val_loader, test_loader = step3_window_and_split(processed_dataset)
    
    # Step 4: 模型构建
    model = step4_build_model()
    
    # Step 5: 模型训练
    model, history = step5_train(model, train_loader, val_loader)
    
    # Step 6: 模型评估
    eval_results = step6_evaluate(model, test_loader)
    
    # Step 7: 结果可视化
    viz_path, model_path = step7_visualize(history, eval_results, model)
    
    # 完成
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🎉 项目完成!")
    print("=" * 60)
    print(f"  总耗时: {total_time:.1f} 秒")
    print(f"  测试集准确率: {eval_results['accuracy']:.4f} ({eval_results['accuracy']:.2%})")
    print(f"  平衡准确率: {eval_results['balanced_acc']:.4f} ({eval_results['balanced_acc']:.2%})")
    print(f"\n输出文件:")
    print(f"  - 可视化结果: {viz_path}")
    print(f"  - 训练好的模型: {model_path}")
    print(f"\n教程文件:")
    print(f"  - 01_moabb_dataset.py  - 数据集加载")
    print(f"  - 02_preprocessing.py   - 数据预处理")
    print(f"  - 03_models_basics.py   - 模型基础")
    print(f"  - 04_training_basics.py - 训练基础")
    print(f"  - 05_visualization.py   - 可视化分析")
    print(f"  - main.py               - 综合实战")
    
    return eval_results


if __name__ == "__main__":
    main()
