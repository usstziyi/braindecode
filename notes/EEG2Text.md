我来帮你查找关于 EEG2Text（脑电到文本解码）的最新 arXiv 论文。


        
            
toolName: web_search
            
status: success
          
query: EEG2Text arXiv 2025 EEG to text decoding
          
我找到了多篇 EEG2Text（脑电到文本解码）相关的最新 arXiv 论文，按时间整理给你。

## 最新论文（2025）

### 1. EEG-to-Text Translation: A Model for Deciphering Human Brain Activity
- arXiv: [2505.13936](https://arxiv.org/abs/2505.13936)（2025-05-20）
- 作者：Saydul Akbar Murad 等
- **核心贡献**：提出 **R1 Translator** 模型，结合**双向 LSTM 编码器 + 预训练 Transformer 解码器**，在 ROUGE、CER、WER 指标上超越 T5 和 Brain Translator。ROUGE-1 达 38.00%（比 T5 高 9%、比 Brain 高 3%）。
- 代码开源：[github.com/Mmurrad/EEG-To-text](https://github.com/Mmurrad/EEG-To-text)

### 2. Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation
- arXiv: [2505.17099](https://arxiv.org/abs/2505.17099)（2025-05）
- 提出 **GLIM（Generative Language Inspection Model）**，强调学习可解释、语义对齐的 EEG 表征，在 ZuCo 数据集上无需 teacher forcing 也能生成流畅、与 EEG 语义一致的句子。

### 3. EEG2TEXT-CN: Open-Vocabulary Chinese Text-EEG Alignment
- arXiv: [2506.00854](https://arxiv.org/abs/2506.00854)（2025-06-17）
- **面向中文**的开放词表 EEG 到文本生成框架，基于 NICE-EEG 编码器 + MiniLM 语言模型，通过掩码预训练和对比学习对齐多通道脑信号与语言表征（ChineseEEG 数据集，128 通道 / 256Hz）。

## 重要背景/经典工作（供参考）

- **Deep Representation Learning for Open Vocabulary EEG-to-Text Decoding** — arXiv: [2312.09430](https://arxiv.org/abs/2312.09430)（2023-11）。该领域代表性工作：EEG 表征学习模块 + **BART** 语言模型 + **GPT-4** 句子精炼模块，在 ZuCo 上 BLEU-1 达 42.75%。
- **DeWave**（Duan et al., 2023）：基于离散量化变分编码器 + 预训练语言模型的经典框架，也是 EEG2Text 领域被广泛引用的工作。

---

如果你需要，我可以：
1. 深入解读其中某一篇的方法细节（比如 R1 Translator 或 GLIM）
2. 帮你拉取某篇的 PDF 全文
3. 结合你当前项目，介绍如何把 EEG 特征接入语言模型

你想看哪篇？或者你的具体需求是什么？

Sources:
- [EEG-to-Text Translation: R1 Translator (2505.13936)](https://arxiv.org/abs/2505.13936)
- [GLIM: Interpretable Representations (2505.17099)](https://arxiv.org/abs/2505.17099)
- [EEG2TEXT-CN (2506.00854)](https://arxiv.org/abs/2506.00854)
- [Open Vocabulary EEG-to-Text (2312.09430)](https://arxiv.org/abs/2312.09430)