你提到的 `viz` 模块，正式名称是 **`braindecode.visualization`**，是 Braindecode 库中专门面向**脑电解码深度学习模型**的可视化与可解释性分析模块，业内常简称为 viz 模块。它的核心定位不是通用的脑电信号绘图（这部分主要依赖 MNE-Python），而是聚焦于「从深度学习模型视角」分析网络行为、解释预测依据、评估模型性能。

## 核心功能与子模块

### 1. 卷积网络结构与感受野分析（`input_windows` 子模块）
专门针对 EEG 卷积神经网络设计，用于理解网络的时空感受野和卷积核的学习模式：
- **感受野计算**：`calc_receptive_field_size` 可计算网络任意一层的单个神经元，在原始输入 EEG 信号上对应的时间/空间感受野大小，辅助设计网络结构与输入窗口长度。
- **最高激活输入提取**：`most_activating_input_windows` 筛选出让指定卷积核激活程度最高的原始 EEG 片段，直观呈现网络“关注”的信号模式。
- **反向相关分析**：`activation_reverse_correlation` 通过统计激活与输入的相关性，得到每个卷积核偏好的输入波形模式，解释卷积层的特征提取逻辑。
- 配套工具：输入输出尺寸推导、最大激活点定位等辅助函数。

### 2. 模型可解释性：扰动与归因分析（`perturbation` 子模块）
通过扰动输入信号量化不同特征对模型预测的影响，是脑电深度学习可解释性的核心工具集：
- **振幅-预测相关性**：`compute_amplitude_gradients` / `compute_amplitude_prediction_correlations` 对 EEG 信号的频谱振幅施加微小扰动，计算扰动幅度与模型预测变化的相关性，输出「不同通道、不同频率段对预测的贡献度」，是 Braindecode 最经典的可解释性方法。
- **相位扰动分析**：`phase_perturbation` 随机扰动信号相位，验证相位信息在模型解码中的作用。
- **特征图差异分析**：`correlate_feature_maps`、`mean_diff_feature_maps` 等工具，用于对比不同实验条件下网络内部激活的差异。

### 3. 专用结果可视化绘图
提供脑电解码场景下的定制化绘图能力：
- **增强版混淆矩阵**：`plot_confusion_matrix` 不仅绘制标准混淆矩阵，还自动计算并展示每一类的精确率、召回率（灵敏度），可选展示 F1 分数，适配脑电解码的分类结果评估。
- **头皮拓扑投影**：`ax_scalp` 可将通道维度的数值（如通道重要性、归因得分）直接绘制为头皮拓扑图，内置标准 10-20 电极位置坐标，直观展示模型关注的脑区分布。

## 补充说明
- **版本差异**：早期原作者版本（Robin Tibor 版）的子模块功能最完整，包含全套扰动分析与感受野工具；后续官方维护的 0.8.x / 1.x 稳定版对模块做了精简，核心保留高频使用的混淆矩阵、振幅梯度计算，部分高级功能仍可通过子模块调用。
- **与 MNE-Python 的分工**：`mne.viz` 侧重原始脑电信号、ERP、时频分析等基础神经数据可视化；`braindecode.visualization` 则完全服务于深度学习模型的分析与解释，二者互补使用。

```python
# 常用功能导入示例
from braindecode.visualization import plot_confusion_matrix
from braindecode.visualization import compute_amplitude_gradients
```

需要我补充混淆矩阵绘制或振幅梯度计算的完整代码示例吗？