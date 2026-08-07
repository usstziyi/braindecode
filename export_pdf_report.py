"""
生成 Braindecode 可视化教程 PDF 报告
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# 注册中文字体
# macOS 系统自带的华文黑体
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_NAME = "STHeiti"

try:
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
    HAS_CHINESE_FONT = True
except Exception as e:
    print(f"警告: 无法加载中文字体 {FONT_PATH}: {e}")
    print("将使用默认字体 (可能不支持中文)")
    HAS_CHINESE_FONT = False
    FONT_NAME = "Helvetica"

# 图片目录
IMG_DIR = Path(__file__).resolve().parent / "viz_output"
OUTPUT_PDF = Path(__file__).resolve().parent / "Braindecode_Visualization_Report.pdf"

def create_report():
    """
    创建 PDF 报告
    """
    # 获取样式
    styles = getSampleStyleSheet()
    
    # 自定义样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName=FONT_NAME,
        fontSize=28,
        leading=36,
        spaceAfter=30,
        textColor=colors.HexColor('#1E88E5'),
        alignment=1  # 居中
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontName=FONT_NAME,
        fontSize=22,
        leading=28,
        spaceBefore=20,
        spaceAfter=15,
        textColor=colors.HexColor('#1565C0')
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontName=FONT_NAME,
        fontSize=18,
        leading=24,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#1976D2')
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontName=FONT_NAME,
        fontSize=15,
        leading=20,
        spaceBefore=10,
        spaceAfter=8,
        textColor=colors.HexColor('#2196F3')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=11,
        leading=16,
        spaceBefore=5,
        spaceAfter=5,
        alignment=0  # 左对齐
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=9,
        leading=12,
        spaceBefore=5,
        spaceAfter=15,
        alignment=1,  # 居中
        textColor=colors.gray
    )

    # 收集内容
    story = []
    
    # 封面
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Braindecode 可视化教程", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("模型可解释性与深度可视化学习报告", 
                          ParagraphStyle('subtitle', parent=body_style, fontSize=16, alignment=1)))
    story.append(Spacer(1, 1 * inch))
    story.append(Paragraph("涵盖 6 个循序渐进的 Tutorial", 
                          ParagraphStyle('info', parent=body_style, fontSize=12, alignment=1, textColor=colors.gray)))
    story.append(Spacer(1, 0.5 * inch))
    
    info_text = """
    生成时间: 2026年8月7日<br/>
    数据源: Braindecode 模拟数据 + EEGNet 模型<br/>
    内容范围: Saliency Map, Integrated Gradients, 高级归因方法, 
    频域分析, 地形图投影, 健全性检查<br/><br/>
    本报告基于 braindecode.visualization 模块的官方 API 示例。
    """
    story.append(Paragraph(info_text, 
                          ParagraphStyle('infoBox', parent=body_style, 
                                        fontSize=11, alignment=1,
                                        leftIndent=50, rightIndent=50,
                                        textColor=colors.HexColor('#424242'))))
    
    story.append(PageBreak())
    
    # Tutorial 1: Saliency Map
    story.append(Paragraph("教程 1: Saliency Map", h1_style))
    story.append(Paragraph(
        "Saliency Map 是最基础的可解释性方法，通过计算模型输出相对于输入的梯度 (|∂f_c / ∂x|) 来识别重要区域。"
        "梯度越大，说明该位置对预测结果的影响越大。",
        body_style
    ))
    story.append(Spacer(1, 20))
    
    img1 = IMG_DIR / "tutorial_1_saliency.png"
    if img1.exists():
        story.append(Image(str(img1), width=6.5*inch, height=4*inch))
        story.append(Paragraph("图 1.1: Saliency Map 单样本可视化 (通道 × 时间热力图)", caption_style))
    
    img2 = IMG_DIR / "tutorial_1_saliency_mean.png"
    if img2.exists():
        story.append(Image(str(img2), width=6.5*inch, height=2.6*inch))
        story.append(Paragraph("图 1.2: 多样本平均 Saliency Map (群体层面的显著性)", caption_style))
    
    story.append(PageBreak())
    
    # Tutorial 2: Integrated Gradients
    story.append(Paragraph("教程 2: Integrated Gradients", h1_style))
    story.append(Paragraph(
        "Integrated Gradients (IG) 是目前最主流的归因方法之一，解决了 Saliency Map 的梯度饱和问题。"
        "通过沿 baseline → x 的线性插值积分，计算真正的贡献值，满足完整性公理。",
        body_style
    ))
    story.append(Spacer(1, 20))
    
    img3 = IMG_DIR / "tutorial_2_ig_vs_saliency.png"
    if img3.exists():
        story.append(Image(str(img3), width=6.5*inch, height=3.8*inch))
        story.append(Paragraph("图 2.1: Integrated Gradients vs Saliency Map 对比", caption_style))
    
    img4 = IMG_DIR / "tutorial_2_class_specific_ig.png"
    if img4.exists():
        story.append(Image(str(img4), width=6*inch, height=2.4*inch))
        story.append(Paragraph("图 2.2: 类别特定的归因分析 (同一输入，不同目标类别)", caption_style))
    
    story.append(PageBreak())
    
    # Tutorial 3: 高级归因方法
    story.append(Paragraph("教程 3: 高级归因方法对比", h1_style))
    story.append(Paragraph(
        "本教程对比 4 种高级归因方法: Guided Backpropagation、Input × Gradient、"
        "DeepLIFT 和 LRP (Layer-wise Relevance Propagation)。每种方法都有其独特的优势和适用场景。",
        body_style
    ))
    story.append(Spacer(1, 20))
    
    img5 = IMG_DIR / "tutorial_3_advanced_attribution.png"
    if img5.exists():
        story.append(Image(str(img5), width=6.5*inch, height=4.5*inch))
        story.append(Paragraph("图 3.1: 4 种高级归因方法可视化对比", caption_style))
    
    img6 = IMG_DIR / "tutorial_3_advanced_channel_imp.png"
    if img6.exists():
        story.append(Image(str(img6), width=6.5*inch, height=4.5*inch))
        story.append(Paragraph("图 3.2: 各方法的通道重要性分析", caption_style))
    
    story.append(PageBreak())
    
    # Tutorial 4: 频域与反卷积
    story.append(Paragraph("教程 4: 频域归因与反卷积分析", h1_style))
    story.append(Paragraph(
        "使用 frequency 模块进行频域归因，通过 deconvolution 理解特征贡献，"
        "并使用 amplitude_gradients 计算 Delta、Theta、Alpha、Beta 等不同频段的梯度。",
        body_style
    ))
    story.append(Spacer(1, 20))
    
    img7 = IMG_DIR / "tutorial_4_deconvolution.png"
    if img7.exists():
        story.append(Image(str(img7), width=6.5*inch, height=2.6*inch))
        story.append(Paragraph("图 4.1: Deconvolution 反卷积分析", caption_style))
    
    img8 = IMG_DIR / "tutorial_4_amplitude_gradients.png"
    if img8.exists():
        story.append(Image(str(img8), width=6.5*inch, height=2.6*inch))
        story.append(Paragraph("图 4.2: 多频段平均振幅梯度", caption_style))
    
    img9 = IMG_DIR / "tutorial_4_amplitude_per_band.png"
    if img9.exists():
        story.append(Image(str(img9), width=6.5*inch, height=2*inch))
        story.append(Paragraph("图 4.3: 按频段展示振幅梯度 (Delta, Theta, Alpha, Beta)", caption_style))
    
    story.append(PageBreak())
    
    # Tutorial 5: 地形图与混淆矩阵
    story.append(Paragraph("教程 5: 地形图投影与增强版混淆矩阵", h1_style))
    story.append(Paragraph(
        "使用 project_to_topomap 将 1D 通道权重投影到 2D 头皮地形图，"
        "直观展示模型关注的大脑区域。同时使用 plot_confusion_matrix 绘制带 F1-Score 的增强版混淆矩阵。",
        body_style
    ))
    story.append(Spacer(1, 20))
    
    img10 = IMG_DIR / "tutorial_5_topomap.png"
    if img10.exists():
        story.append(Image(str(img10), width=6.8*inch, height=2.6*inch))
        story.append(Paragraph("图 5.1: 1D 通道值 → 2D 头皮地形图投影", caption_style))
    
    img11 = IMG_DIR / "tutorial_5_confusion_matrix.png"
    if img11.exists():
        story.append(Image(str(img11), width=5*inch, height=4*inch))
        story.append(Paragraph("图 5.2: 增强版混淆矩阵 (带 Precision/Sensitivity/F1)", caption_style))
    
    story.append(PageBreak())
    
    # Tutorial 6: 健全性检查
    story.append(Paragraph("教程 6: 健全性检查与归因指标", h1_style))
    story.append(Paragraph(
        "健全性检查 (Sanity Checks) 用于验证归因方法的可靠性。"
        "通过 random_target 生成对照标签，cascading_layer_reset 逐层随机化模型参数，"
        "然后使用 compute_metrics 和 compute_ssim_metrics 量化解释质量。",
        body_style
    ))
    story.append(Spacer(1, 20))
    
    img12 = IMG_DIR / "tutorial_6_sanity_check.png"
    if img12.exists():
        story.append(Image(str(img12), width=6.8*inch, height=3.8*inch))
        story.append(Paragraph("图 6.1: 逐层随机化归因图对比 (Cascading Layer Reset)", caption_style))
    
    img13 = IMG_DIR / "tutorial_6_class_conditional.png"
    if img13.exists():
        story.append(Image(str(img13), width=6.8*inch, height=1.8*inch))
        story.append(Paragraph("图 6.2: 类别条件下的归因分析 (同一输入，不同目标)", caption_style))
    
    story.append(PageBreak())
    
    # 总结与建议
    story.append(Paragraph("总结与学习建议", h1_style))
    
    story.append(Paragraph("核心 API 总结", h2_style))
    summary_text = """
    <b>1. 基础归因:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• saliency(model, x, target)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• integrated_gradients(model, x, target, baseline, steps)<br/><br/>
    <b>2. 高级归因:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• guided_backprop(model, x, target)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• deep_lift(model, x, target, baseline)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• input_x_gradient(model, x, target)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• lrp(model, x, target)<br/><br/>
    <b>3. 频域分析:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• deconvolution(model, x, target)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• amplitude_gradients(model, x)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• amplitude_gradients_per_trial(model, dataset, batch_size)<br/><br/>
    <b>4. 可视化投影:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• project_to_topomap(data, chs_info, res)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• plot_confusion_matrix(confusion_mat, class_names, with_f1_score)<br/><br/>
    <b>5. 质量评估:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• sanity.random_target(target, n_classes)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• sanity.cascading_layer_reset(model)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• metrics.compute_metrics(explanations, reference)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;• metrics.compute_ssim_metrics(explanations, reference)
    """
    story.append(Paragraph(summary_text, body_style))
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("下一步学习路径", h2_style))
    next_text = """
    1. 使用真实 MOABB 数据: 替换模拟数据为 BNCI2014_001 等真实 EEG 数据集<br/>
    2. 训练 EEGClassifier: 将未训练的 EEGNet 替换为经过训练的分类器<br/>
    3. 阅读官方示例: https://braindecode.org/stable/auto_examples/<br/>
    4. 深入研究 captum 库: 支持更多归因方法 (SHAP, occlusion 等)<br/>
    5. 尝试其他模型: Deep4Net, ShallowFBCSPNet, ATCNet 等
    """
    story.append(Paragraph(next_text, body_style))
    
    story.append(Spacer(1, 50))
    story.append(Paragraph("— 报告结束 —", 
                          ParagraphStyle('end', parent=body_style, alignment=1, 
                                        fontSize=14, textColor=colors.gray)))
    
    # 创建 PDF
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # 构建 PDF
    doc.build(story)
    print(f"PDF 报告已生成: {OUTPUT_PDF}")

if __name__ == "__main__":
    if not HAS_CHINESE_FONT:
        print("警告: 中文字体加载失败，报告中的中文可能显示为乱码。")
        print(f"请检查字体路径: {FONT_PATH}")
    create_report()
