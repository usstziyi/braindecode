import matplotlib.pyplot as plt
import numpy as np

# 准备数据
x = np.linspace(0, 10, 50)

# 方法1：直接传入多个y值序列
# 每一行是一个数据序列
y_data = np.array([
    np.sin(x),           # 正弦波
    np.cos(x),           # 余弦波
    np.sin(x) + np.cos(x),  # 正弦+余弦
    0.5 * np.sin(2*x),   # 倍频正弦
    0.8 * np.cos(0.5*x)  # 半频余弦
])

y_data_mean = y_data.mean(axis=0)
print(y_data_mean.shape)



# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

print(y_data.shape)

# ========== 方法1：一步绘制多条线（自动样式） ==========
ax1.plot(y_data.T)  # 注意转置！每列代表一条线
ax1.set_title('Method 1: Draw multiple lines in one step\n(Automatic color cycle)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True, alpha=0.3)
ax1.legend(['sin(x)', 'cos(x)', 'sin+cos', '0.5sin(2x)', '0.8cos(0.5x)'])

# ========== 方法2：一步绘制多条线（自定义样式） ==========
# 使用关键字参数一次性设置所有线的样式
lines = ax2.plot(x, y_data.T, linewidth=2)
ax2.set_title('Method 2: Draw multiple lines in one step\nUniform style for all lines)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True, alpha=0.3)

# 设置图例
ax2.legend(['sin(x)', 'cos(x)', 'sin+cos', '0.5sin(2x)', '0.8cos(0.5x)'])

plt.tight_layout()
plt.show()

