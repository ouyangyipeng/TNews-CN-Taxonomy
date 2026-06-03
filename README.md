# 📰 TNEWS 中文短新闻主题分类

<div align="center">

**基于深度神经网络的中文短新闻 15 分类任务**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🌐 **结果主页**: [tnews.nexa-lang.com](https://tnews.nexa-lang.com) | 📄 **实验报告**: [report/main.pdf](report/main.pdf)

</div>

---

## 📋 项目概述

本项目在 **TNEWS（今日头条中文新闻短文本分类）** 数据集上完成 15 分类的新闻主题预测任务。实现了以下模型架构：

| 架构 | 描述 | 参数量 |
|------|------|--------|
| **BiLSTM + Bahdanau Attention** | 双向 LSTM + 加性注意力机制 | 4.8M |
| **Transformer Encoder** | 多头自注意力 + 位置编码 | 15.2M |
| **BERT-base-chinese** | 预训练语言模型全量微调 | 110M |
| **RoBERTa-large** | 大规模预训练模型微调 | 325M |

### 🏆 最佳结果

| 指标 | 数值 | 模型 |
|------|------|------|
| **Test Accuracy** | **54.99%** | BERT Ensemble (Average) |
| **Test Macro-F1** | **54.11%** | BERT Ensemble (Average) |
| **Val Macro-F1** | **54.36%** | BERT (simplified head) |

> 📊 对比 CLUE 基准：BERT-base baseline 为 56.58%，SOTA (RoBERTa-wwm-ext-large) 为 58.61%

---

## 📁 项目结构

```
.
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 清洗后数据
│   │   ├── train_clean.json     # 训练集 (49,726 条)
│   │   ├── dev_clean.json       # 验证/测试集 (9,765 条)
│   │   └── test_clean.json      # 无标签测试集
│   └── vocab/                   # 词表文件
├── src/                         # 源代码
│   ├── models/                  # 模型定义
│   │   ├── bilstm_attention.py  # BiLSTM + Bahdanau Attention
│   │   ├── transformer.py       # Transformer Encoder
│   │   ├── bert_classifier.py   # BERT 分类器
│   │   └── qwen_classifier.py   # Qwen 分类器
│   ├── data_cleaner.py          # 数据清洗
│   ├── text_tokenizers.py       # 多粒度分词器
│   ├── dataset.py               # 数据集加载
│   ├── train.py                 # 基础训练脚本
│   ├── train_enhanced.py        # 增强训练脚本
│   ├── ensemble_predict.py      # 集成学习预测
│   └── visualize.py             # 可视化脚本
├── scripts/                     # 辅助脚本
│   ├── eda.py                   # 探索性数据分析
│   ├── build_vocab.py           # 词表构建
│   └── plot_optimization_progress.py
├── checkpoints/                 # 模型检查点
├── logs/                        # 训练日志
├── report/                      # 实验报告
│   ├── main.tex                 # LaTeX 源文件
│   ├── main.pdf                 # 编译后的 PDF
│   └── figures/                 # 图表文件
├── site/                        # GitHub Pages 网站
└── plans/                       # 项目计划文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.12.0+
- CUDA 12.0+ (GPU 训练)

### 安装依赖

```bash
# 创建虚拟环境
uv venv --python=3.10
source .venv/bin/activate

# 安装依赖
uv pip install torch jieba pandas numpy matplotlib seaborn scikit-learn transformers
```

### 使用流程

#### 1️⃣ 数据清洗

```bash
python src/data_cleaner.py
```

#### 2️⃣ 探索性数据分析

```bash
python scripts/eda.py
```

#### 3️⃣ 构建词表

```bash
python scripts/build_vocab.py
```

#### 4️⃣ 模型训练

**BiLSTM + Attention (字符级)**
```bash
python src/train.py \
  --model bilstm_attention \
  --tokenizer char \
  --epochs 30 \
  --batch_size 64 \
  --lr 0.001 \
  --max_len 50 \
  --experiment_name bilstm_char
```

**BERT 全量微调**
```bash
python src/train_enhanced.py \
  --model bert_classifier \
  --tokenizer subword \
  --epochs 20 \
  --batch_size 32 \
  --lr 2e-5 \
  --max_len 128 \
  --use_class_weights \
  --label_smoothing 0.1 \
  --experiment_name bert_finetune
```

#### 5️⃣ 可视化与分析

```bash
python src/visualize.py --experiment bilstm_char
```

#### 6️⃣ 集成学习预测

```bash
python src/ensemble_predict.py \
  --checkpoints checkpoints/bert_v1/best_model.pt checkpoints/bert_v2/best_model.pt \
  --strategy average
```

---

## 📊 实验结果

### 消融实验

| 模型 | 分词方式 | Val Acc | Val Macro-F1 | Best Epoch |
|------|----------|---------|--------------|------------|
| BiLSTM+Attention | Char | 0.4999 | 0.4833 | 6 |
| BiLSTM+Attention | Word | 0.4683 | 0.4579 | 5 |
| Transformer | Char (v2) | 0.4404 | 0.4093 | 4 |
| Transformer | Word | 0.4675 | 0.4222 | 3 |
| BERT-base (full finetune) | Subword | 0.5291 | 0.5208 | 5 |
| BERT-base (augmented) | Subword | 0.5339 | 0.5330 | 4 |
| BERT-base (simplified head) | Subword | 0.5453 | **0.5436** | 4 |
| RoBERTa-large | Subword | 0.5516 | 0.5405 | 2 |

### 集成学习结果

| 集成策略 | 模型数量 | Test Accuracy | Test Macro-F1 |
|----------|----------|---------------|---------------|
| Majority Voting | 4 | 0.5481 | 0.5410 |
| **Average Probability** | **4** | **0.5499** | **0.5411** |
| RoBERTa-large (single) | 1 | 0.5499 | 0.5359 |

### 优化策略累积效果

```
BiLSTM+Attn (47.00%) → BERT-base (52.08%) → +Augment (53.30%) → +Simplified (54.36%) → Ensemble (54.11%)
```

---

## 🔍 关键发现

1. **BiLSTM+Attention 优于 Transformer**：在 TNEWS 短文本分类任务上，BiLSTM+Attention 的表现明显优于 Transformer。这可能是因为 TNEWS 文本较短（平均 22 字符），BiLSTM 的循环结构更适合捕获短距离依赖。

2. **Char-level 优于 Word-level**：对于 BiLSTM+Attention，Char-level 分词的 Macro-F1 比 Word-level 高 2.5 个百分点。短文本中 Char-level 能保留更多细粒度信息。

3. **预训练模型显著提升**：BERT-base 全量微调将性能从 47.00% 提升到 52.08%，提升约 5 个百分点。

4. **类别不平衡问题**：news_stock（仅 45 条测试样本）的 F1 分数仅为 0.11，模型在少数类别上表现较差。

---

## 🏗️ 技术细节

### BiLSTM + Bahdanau Attention

| 组件 | 配置 |
|------|------|
| Embedding | 300 维 |
| BiLSTM | 2 层，隐藏维度 256，双向 |
| Attention | Bahdanau 加性注意力，attention_dim=128 |
| 分类头 | 1024 → 512 → 15 |
| Dropout | 0.3 |
| **参数量** | **4.8M** |

### Transformer Encoder

| 组件 | 配置 |
|------|------|
| Embedding | 512 维 |
| Positional Encoding | 正弦/余弦位置编码 |
| Encoder | 4 层，8 头注意力 |
| FFN 维度 | 2048 |
| 池化 | Global Average Pooling |
| 分类头 | 512 → 256 → 15 |
| **参数量** | **15.2M** |

### BERT-base 分类器

| 组件 | 配置 |
|------|------|
| 预训练模型 | bert-base-chinese |
| 池化策略 | Mean Pooling |
| 分类头 | LayerNorm → Linear(768→256) → GELU → Dropout → Linear(256→15) |
| **参数量** | **110M** |

---

## 📝 注意事项

1. **数据划分**：训练集和验证集从 `train_clean.json` 中按 90%/10% 划分，随机种子 42。测试集使用 `dev_clean.json`。

2. **词表构建**：严格遵守"仅使用训练集"的原则，验证集和测试集的信息不会泄露到词表中。

3. **Early Stopping**：当验证集 Macro-F1 在连续 5 个 epoch 内没有提升时，停止训练。

4. **学习率调度**：使用带 warmup 的余弦退火调度器，warmup 阶段 5 个 epoch。

---

## 📚 参考资料

- [TNEWS 数据集 (CLUE Benchmark)](https://github.com/CLUEbenchmark/CLUE)
- [Bahdanau Attention 论文](https://arxiv.org/abs/1409.0473)
- [Transformer 论文 (Attention Is All You Need)](https://arxiv.org/abs/1706.03762)
- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)

---

## 📄 许可证

本项目仅供课程作业使用，请勿用于商业用途。

---

<div align="center">

**中山大学 · 人工神经网络课程项目**

学号: 23336188 | 姓名: 欧阳易芃 | 指导教师: 潘炎

🌐 [tnews.nexa-lang.com](https://tnews.nexa-lang.com)

</div>
