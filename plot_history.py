import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history_path="models/history.json"):
    """读取 history.json 并绘制训练曲线"""
    
    # 1. 读取历史数据
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # 2. 提取指标
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    valid_loss = [h['valid_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    valid_acc = [h['valid_acc'] for h in history]
    kappa = [h['kappa'] for h in history]
    
    # 3. 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制 Loss 曲线
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue')
    axes[0].plot(epochs, valid_loss, label='Valid Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制 Accuracy 曲线
    axes[1].plot(epochs, train_acc, label='Train Acc', color='blue')
    axes[1].plot(epochs, valid_acc, label='Valid Acc', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # 绘制 Kappa 曲线
    # Cohen's Kappa (科恩卡帕系数) 是衡量分类模型一致性的指标，考虑了随机猜测的因素。
    axes[2].plot(epochs, kappa, label='Kappa', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Kappa')
    axes[2].set_title('Cohen\'s Kappa Score')
    axes[2].legend()
    axes[2].grid(True)
    
    # 4. 找到最佳指标
    best_kappa_epoch = epochs[np.argmax(kappa)]
    best_kappa_value = max(kappa)
    best_valid_acc_epoch = epochs[np.argmax(valid_acc)]
    best_valid_acc_value = max(valid_acc)
    
    # 标注最佳 Kappa 点
    axes[2].scatter([best_kappa_epoch], [best_kappa_value], 
                    color='red', s=100, zorder=5, label=f'Best: {best_kappa_value:.4f}')
    axes[2].annotate(f'Epoch {best_kappa_epoch}', 
                     xy=(best_kappa_epoch, best_kappa_value),
                     xytext=(10, 10), textcoords='offset points')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print(f"图表已保存到 training_history.png")
    
    # 5. 打印最佳指标
    print("\n" + "="*50)
    print("训练结果汇总")
    print("="*50)
    print(f"总 Epoch 数: {len(history)}")
    print(f"最佳验证准确率: {best_valid_acc_value:.4f} (Epoch {best_valid_acc_epoch})")
    print(f"最佳 Kappa: {best_kappa_value:.4f} (Epoch {best_kappa_epoch})")
    print(f"最终训练损失: {train_loss[-1]:.4f}")
    print(f"最终验证损失: {valid_loss[-1]:.4f}")
    print("="*50)
    
    plt.show()

if __name__ == "__main__":
    plot_training_history()
