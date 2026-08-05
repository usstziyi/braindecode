"""
Braindecode 学习教程 - 第04章: 训练基础
======================================================

本教程学习如何训练 braindecode 模型。

知识点:
1. EEGClassifier / EEGRegressor 使用
2. 训练循环
3. 评估指标
4. 回调函数 (callbacks)
5. 模型保存与加载

参考: https://braindecode.org/stable/index.html#braindecodetraining
"""

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


# ============================================================
# 1. EEGClassifier 基础用法
# ============================================================

def tutorial_eegclassifier_basic():
    """
    EEGClassifier: 类似 sklearn 接口的分类器
    
    优点:
    - 简洁的 fit() / predict() / score() 接口
    - 自动处理训练循环
    - 内置评估指标
    
    参数:
    - model: Braindecode 模型
    - lr: 学习率
    - batch_size: 批量大小
    - max_epochs: 训练轮数
    - device: 设备 (cpu/cuda)
    - callbacks: 回调函数列表
    """
    print("=" * 60)
    print("教程 4.1: EEGClassifier 基础用法")
    print("=" * 60)
    
    if EEGClassifier is None:
        print("⚠️  EEGClassifier 不可用, 跳过此教程")
        print("   请检查 braindecode 版本或使用手动训练循环 (教程 4.2)")
        return
    
    # 1. 准备数据
    print("\n1. 准备数据...")
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    
    # 预处理
    preprocessors = [
        PickTypes(eeg=True, verbose=False),
        Filter(l_freq=4, h_freq=40.0, verbose=False),
        Rescale(scalings=1e6, verbose=False),
        # Resample(128, verbose=False),
        Preprocessor(exponential_moving_standardize),
    ]
    dataset = preprocess(dataset, preprocessors)

    
    # 窗口化
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=1000,
        window_stride_samples=1000,
        preload=True,
    ) # 每条记录是一个窗口

    
    # 划分数据集
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(windows_dataset))
    val_size = len(windows_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        windows_dataset, [train_size, val_size], generator=generator
    )
    
    print(f"   训练集: {len(train_dataset)} 个窗口")
    print(f"   验证集: {len(val_dataset)} 个窗口")
    
    # 2. 构建模型
    print("\n2. 构建模型...")
    model = EEGNet(
        n_chans=22,
        n_outputs=4,
        n_times=1000,
        sfreq=250,
    )


    # 查看模型配置
    print("\n2.1 模型配置:")
    config = model.get_config()
    for key, value in config.items():
        print(f"    {key}: {value}")
    
    # 3. 创建 EEGClassifier
    print("\n3. 创建 EEGClassifier...")
    classifier = EEGClassifier(
        model,
        lr=0.001,
        batch_size=64,
        max_epochs=20,  # 只用 2 轮演示
        device="mps",
        train_split=None,  # 不使用内置验证拆分, 手动管理
    )
    # 4. 训练
    print("\n4. 开始训练...")
    classifier.fit(train_dataset)
    
    # 5. 评估
    print("\n5. 评估模型...")
    
    # score() 来自 sklearn.ClassifierMixin, 需要 (X, y) 参数
    # WindowsDataset 返回 (x, y, idx), 需要提取
    def dataset_to_xy(dataset):
        X, y = [], []
        for x_i, y_i, _ in dataset:
            X.append(x_i)
            y.append(y_i)
        return np.array(X), np.array(y)
    
    X_train, y_train = dataset_to_xy(train_dataset)
    X_val, y_val = dataset_to_xy(val_dataset)
    
    train_score = classifier.score(X_train, y_train)
    val_score = classifier.score(X_val, y_val)
    
    print(f"   训练集准确率: {train_score:.4f}")
    print(f"   验证集准确率: {val_score:.4f}")


# ============================================================
# 2. 手动训练循环
# ============================================================

def tutorial_manual_training():
    """
    手动训练循环: 更灵活的控制
    
    适合:
    - 自定义损失函数
    - 复杂的训练逻辑
    - 多模型训练
    """
    print("\n" + "=" * 60)
    print("教程 4.2: 手动训练循环")
    print("=" * 60)
    
    # 准备数据
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    preprocessors = [
        PickTypes(eeg=True, verbose=False),
        Filter(l_freq=4, h_freq=40.0, verbose=False),
        Rescale(scalings=1e6, verbose=False),
        Preprocessor(exponential_moving_standardize),
        Resample(sfreq=128, verbose=False),
    ]
    dataset = preprocess(dataset, preprocessors)
    
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=512,
        window_stride_samples=128,
        preload=True,
    )
    
    # 划分
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(windows_dataset))
    val_size = len(windows_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        windows_dataset, [train_size, val_size], generator=generator
    )
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 模型
    model = EEGNet(n_chans=22, n_outputs=4, n_times=512, sfreq=128)
    
    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    n_epochs = 3
    device = torch.device("cpu")
    model.to(device)
    
    print(f"\n开始训练 ({n_epochs} epochs)...")
    
    for epoch in range(n_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
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
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        print(f"   Epoch {epoch+1}/{n_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")


# ============================================================
# 3. 评估指标
# ============================================================

def tutorial_evaluation_metrics():
    """
    评估指标
    
    分类任务:
    - accuracy: 准确率
    - balanced_accuracy: 平衡准确率 (处理类别不平衡)
    - precision: 精确率
    - recall: 召回率
    - f1_score: F1 分数
    - confusion_matrix: 混淆矩阵
    
    回归任务:
    - mse: 均方误差
    - mae: 平均绝对误差
    - r2: R² 决定系数
    """
    print("\n" + "=" * 60)
    print("教程 4.3: 评估指标")
    print("=" * 60)
    
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    
    # 模拟真实场景
    np.random.seed(42)
    y_true = np.random.randint(0, 4, size=100)
    y_pred = y_true.copy()
    # 添加一些错误
    error_idx = np.random.choice(100, size=15, replace=False)
    y_pred[error_idx] = np.random.randint(0, 4, size=15)
    
    print(f"\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print(f"\n分类报告:")
    report = classification_report(y_true, y_pred, target_names=["Class 0", "Class 1", "Class 2", "Class 3"])
    print(report)
    
    print(f"\n各项指标:")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  F1-Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")


# ============================================================
# 4. 回调函数
# ============================================================

def tutorial_callbacks():
    """
    回调函数 (Callbacks)
    
    Braindecode 训练回调:
    - EarlyStopping: 早停 (防止过拟合)
    - LearningRateScheduler: 学习率调度
    - ModelCheckpoint: 保存最优模型
    - CSVLogger: 记录训练日志
    - LambdaCallback: 自定义回调
    
    注意: Braindecode 使用 skorch, 回调接口基于 skorch
    """
    print("\n" + "=" * 60)
    print("教程 4.4: 回调函数")
    print("=" * 60)
    
    print("""
常用回调函数:

1. EarlyStopping:
   - 监控验证损失
   - 当连续 N 轮没有改善时停止训练
   - 参数: patience (耐心轮数), threshold (改善阈值)

2. LearningRateScheduler:
   - 动态调整学习率
   - 可以实现余弦退火、阶梯下降等
   - 参数: policy (调度策略), factor (衰减因子)

3. ModelCheckpoint:
   - 保存验证集上表现最好的模型
   - 防止过拟合时恢复最佳状态
   - 参数: monitor (监控指标), filepath (保存路径)

4. CSVLogger:
   - 将训练日志保存为 CSV
   - 方便后续分析和可视化
   - 参数: filename (CSV 路径)

5. LambdaCallback:
   - 自定义回调逻辑
   - 参数: on_epoch_begin/end, on_train_begin/end 等
""")
    
    # 示例: 如何使用回调 (伪代码, 实际使用 EEGClassifier)
    print("""
代码示例:
```python
from braindecode.callbacks import (
    EarlyStopping,
    CSVLogger,
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        threshold=1e-4,
    ),
    CSVLogger(
        filename="training_log.csv",
    ),
]

classifier = EEGClassifier(
    model=model,
    lr=0.001,
    batch_size=64,
    max_epochs=100,
    callbacks=callbacks,
)
```
""")


# ============================================================
# 5. 模型保存与加载
# ============================================================

def tutorial_save_load():
    """
    模型保存与加载
    
    方法:
    1. 保存整个模型: torch.save(model, "model.pth")
    2. 只保存权重: torch.save(model.state_dict(), "model_weights.pth")
    3. 保存检查点: torch.save({...}, "checkpoint.pth")
    4. 使用 from_pretrained(): 从 Hugging Face 加载预训练模型
    """
    print("\n" + "=" * 60)
    print("教程 4.5: 模型保存与加载")
    print("=" * 60)
    
    import os
    import tempfile
    
    # 创建一个简单模型
    model = ShallowFBCSPNet(n_chans=22, n_outputs=4, n_times=512)
    
    # 方法1: 保存权重
    temp_dir = tempfile.mkdtemp()
    weights_path = os.path.join(temp_dir, "model_weights.pth")
    
    torch.save(model.state_dict(), weights_path)
    print(f"1. 保存权重到: {weights_path}")
    
    # 加载权重
    loaded_model = ShallowFBCSPNet(n_chans=22, n_outputs=4, n_times=512)
    loaded_model.load_state_dict(torch.load(weights_path, weights_only=True))
    print(f"   权重加载成功!")
    
    # 验证
    x = torch.randn(4, 22, 512)
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = loaded_model(x)
    assert torch.allclose(out1, out2), "模型输出不一致!"
    print(f"   模型输出验证: ✅ 一致")
    
    # 方法2: 保存检查点 (包含优化器状态)
    checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
    optimizer = torch.optim.Adam(model.parameters())
    
    checkpoint = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": 0.1234,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\n2. 保存检查点到: {checkpoint_path}")
    
    # 加载检查点
    loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
    print(f"   检查点内容: epoch={loaded_checkpoint['epoch']}, loss={loaded_checkpoint['loss']:.4f}")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n3. Hugging Face 预训练模型:")
    print("   # 从 Hugging Face Hub 加载")
    print("   model = EEGPT.from_pretrained('braindecode/EEPT')")
    print("   model = BENDR.from_pretrained('braindecode/BENDR')")


if __name__ == "__main__":
    tutorial_eegclassifier_basic()
    # tutorial_manual_training()
    # tutorial_evaluation_metrics()
    # tutorial_callbacks()
    # tutorial_save_load()
    
    # print("\n" + "=" * 60)
    # print("🎉 第04章完成! 你已经学会了:")
    # print("  ✅ EEGClassifier 使用")
    # print("  ✅ 手动训练循环")
    # print("  ✅ 评估指标")
    # print("  ✅ 回调函数")
    # print("  ✅ 模型保存与加载")
    # print("\n进入 05_visualization.py 学习可视化!")
    # print("=" * 60)
