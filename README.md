# TNEWS 中文短新闻主题分类

基于深度神经网络的中文短新闻主题分类项目，使用 BiLSTM+Attention 和 Transformer 两种架构在 TNEWS 数据集上进行 15 分类任务。

## 项目结构

```
.
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   ├── processed/           # 清洗后数据
│   └── vocab/               # 词表文件
├── src/                     # 源代码
│   ├── models/              # 模型定义
│   │   ├── bilstm_attention.py
│   │   └── transformer.py
│   ├── data_cleaner.py      # 数据清洗
│   ├── tokenizers.py        # 分词器
│   ├── dataset.py           # 数据集加载
│   ├── train.py             # 训练脚本
│   └── visualize.py         # 可视化脚本
├── scripts/                 # 辅助脚本
│   ├── eda.py               # 探索性数据分析
│   └── build_vocab.py       # 词表构建
├── configs/                 # 配置文件
├── checkpoints/             # 模型检查点
├── logs/                    # 训练日志
├── report/                  # 实验报告
│   ├── main.tex             # LaTeX 源文件
│   ├── main.pdf             # 编译后的 PDF
│   └── figures/             # 图表文件
└── plans/                   # 项目计划文档
```

## 环境要求

- Python 3.10+
- PyTorch 2.12.0+
- CUDA 13.0+ (GPU 训练)

## 安装依赖

```bash
# 创建虚拟环境
uv venv --python=3.10
source .venv/bin/activate

# 安装依赖
uv pip install torch jieba pandas numpy matplotlib seaborn scikit-learn transformers
```

## 使用流程

### 1. 数据准备

数据已包含在 `data/raw/` 目录中，无需额外下载。

### 2. 数据清洗

```bash
python src/data_cleaner.py
```

输出：
- `data/processed/train_clean.json` - 清洗后的训练集
- `data/processed/dev_clean.json` - 清洗后的验证集
- `data/processed/test_clean.json` - 清洗后的测试集

### 3. 探索性数据分析

```bash
python scripts/eda.py
```

输出：
- `report/figures/class_distribution.png` - 类别分布图
- `report/figures/text_length_distribution.png` - 文本长度分布图
- `report/figures/keyword_stats.png` - 关键词统计图
- `data/eda_report.md` - EDA 报告

### 4. 构建词表

```bash
python scripts/build_vocab.py
```

输出：
- `data/vocab/char_vocab.json` - 字符级词表
- `data/vocab/word_vocab.json` - 词级词表
- `data/vocab/subword_vocab.json` - 子词级词表

### 5. 模型训练

#### BiLSTM + Attention (字符级)

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

#### BiLSTM + Attention (词级)

```bash
python src/train.py \
  --model bilstm_attention \
  --tokenizer word \
  --epochs 30 \
  --batch_size 64 \
  --lr 0.001 \
  --max_len 30 \
  --experiment_name bilstm_word
```

#### Transformer (字符级)

```bash
python src/train.py \
  --model transformer \
  --tokenizer char \
  --epochs 30 \
  --batch_size 64 \
  --lr 0.001 \
  --max_len 50 \
  --experiment_name transformer_char
```

#### Transformer (词级)

```bash
python src/train.py \
  --model transformer \
  --tokenizer word \
  --epochs 30 \
  --batch_size 64 \
  --lr 0.001 \
  --max_len 30 \
  --experiment_name transformer_word
```

输出：
- `checkpoints/{experiment_name}/best_model.pt` - 最佳模型
- `checkpoints/{experiment_name}/experiment_config.json` - 实验配置
- `logs/{experiment_name}/training_log.json` - 训练日志

### 6. 可视化与分析

```bash
python src/visualize.py --experiment bilstm_char
```

输出：
- `report/figures/{experiment}/training_curves.png` - 训练曲线
- `report/figures/{experiment}/confusion_matrix.png` - 混淆矩阵
- `report/figures/{experiment}/attention_heatmap_*.png` - Attention 热力图
- `report/figures/{experiment}/tsne_visualization.png` - t-SNE 可视化
- `report/figures/{experiment}/bad_case_analysis.md` - Bad Case 分析

### 7. 编译报告

```bash
cd report
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

输出：`report/main.pdf`

## 实验结果

### 消融实验

| 模型 | 分词方式 | Val Acc | Val Macro-F1 | Best Epoch |
|------|----------|---------|--------------|------------|
| BiLSTM+Attention | Char | 0.4999 | **0.4833** | 6 |
| BiLSTM+Attention | Word | 0.4683 | 0.4579 | 5 |
| Transformer | Char (v1, lr=5e-4) | 0.1094 | 0.0131 | --- |
| Transformer | Char (v2, lr=1e-3) | 0.4404 | 0.4093 | 4 |
| Transformer | Word | 0.4675 | 0.4222 | 3 |
| Transformer | Subword | 0.1094 | 0.0131 | --- |

### 最佳模型测试结果

- **Test Accuracy**: 0.4972
- **Test Macro-F1**: 0.4700

## 关键发现

1. **BiLSTM+Attention 优于 Transformer**：在 TNEWS 短文本分类任务上，BiLSTM+Attention 的表现明显优于 Transformer。这可能是因为 TNEWS 文本较短（平均 22 字符），BiLSTM 的循环结构更适合捕获短距离依赖。

2. **Char-level 优于 Word-level**：对于 BiLSTM+Attention，Char-level 分词的 Macro-F1 比 Word-level 高 2.5 个百分点。这可能是因为短文本中 Char-level 能保留更多细粒度信息。

3. **类别不平衡问题**：news_stock（仅 45 条测试样本）的 F1 分数仅为 0.11，模型在少数类别上表现较差。

## 技术细节

### BiLSTM + Bahdanau Attention

- **Embedding**: 300 维
- **BiLSTM**: 2 层，隐藏维度 256，双向
- **Attention**: Bahdanau 加性注意力
- **分类头**: 1024 → 512 → 15
- **参数量**: 4.8M

### Transformer Encoder

- **Embedding**: 512 维
- **Positional Encoding**: 正弦/余弦位置编码
- **Encoder**: 4 层，8 头注意力，FFN 维度 2048
- **池化**: Global Average Pooling
- **分类头**: 512 → 256 → 15
- **参数量**: 15.2M

## 注意事项

1. **数据划分**：训练集和验证集从 `train_clean.json` 中按 90%/10% 划分，随机种子 42。测试集使用 `dev_clean.json`。

2. **词表构建**：严格遵守"仅使用训练集"的原则，验证集和测试集的信息不会泄露到词表中。

3. **Early Stopping**：当验证集 Macro-F1 在连续 5 个 epoch 内没有提升时，停止训练。

4. **学习率调度**：使用带 warmup 的余弦退火调度器，warmup 阶段 5 个 epoch。

## 参考资料

- [TNEWS 数据集](https://github.com/CLUEbenchmark/CLUE)
- [Bahdanau Attention 论文](https://arxiv.org/abs/1409.0473)
- [Transformer 论文](https://arxiv.org/abs/1706.03762)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)

## 许可证

本项目仅供课程作业使用，请勿用于商业用途。
