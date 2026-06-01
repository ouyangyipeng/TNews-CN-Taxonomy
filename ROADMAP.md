# 项目历程记录 (ROADMAP)

## 项目概述
TNEWS 中文短新闻主题分类，使用 BiLSTM+Attention 和 Transformer 两种架构。

## 时间线

### 2026-05-30 阶段一：数据工程
- 下载 TNEWS 数据集（53,360 条训练样本，15 个类别）
- 数据清洗：去除重复样本 3,634 条，保留 49,726 条
- EDA 分析：类别不平衡比例 23.17，字符级 95 分位数 32
- 实现三种分词器：Char-level (4,809)、Word-level (30,729)、Subword (21,128)

### 2026-05-30 阶段二：模型架构
- 手动实现 BiLSTM + Bahdanau Attention（参数量 4.8M）
- 手动实现 Transformer Encoder（参数量 14.8M）
- 关键设计：Attention Mask 处理 padding、Positional Encoding、Multi-Head Attention

### 2026-05-30 阶段三：训练与调优
- BiLSTM+Char: Macro-F1=0.4833 (Epoch 6) ← 最佳模型
- BiLSTM+Word: Macro-F1=0.4579 (Epoch 5)
- Transformer+Char v1 (lr=5e-4): 失败，未收敛
- Transformer+Char v2 (lr=1e-3): Macro-F1=0.4093 (Epoch 4)
- Transformer+Word: Macro-F1=0.4222 (Epoch 3)
- Transformer+Subword: 失败，未收敛（BERT 词表与从头训练不匹配）
- 关键发现：Transformer 需要 warmup 才能正常训练

### 2026-05-30 阶段四：可视化与分析
- 生成训练曲线、混淆矩阵、Attention 热力图、t-SNE 聚类图
- Bad Case 分析：news_military↔news_world、news_finance↔news_agriculture 等混淆对

### 2026-05-30 阶段五：报告撰写
- LaTeX 报告 8 大结构完整撰写
- 11 篇真实引用（经 tavily 验证）

### 2026-05-31 阶段六：审查与修复
- 第三方审查发现 5 个严重问题 + 5 个中等问题 + 5 个轻微问题
- 修复：统一 min_freq=1、重新运行实验确保 warmup 生效
- 修复：用真实 Bad Case 替换手动编写案例
- 修复：补充失败实验（transformer_char v1、subword）到报告
- 修复：修正心得体会中不准确的代码示例
- 修复：移除 xeCJK 包冲突、填充 reference.bib

## 关键决策记录

| 决策 | 理由 |
|------|------|
| 选择 Char-level 作为主要分词方式 | 短文本上表现更好，无 OOV 问题 |
| 使用 Bahdanau Attention 而非 Luong | 加性注意力在短文本上更稳定 |
| 使用 Global Average Pooling 而非 [CLS] | 实现更简单，效果相当 |
| 使用 warmup + cosine 学习率调度 | Transformer 训练必需 |
| Early Stopping patience=5 | 平衡过拟合防护和充分训练 |

## 踩坑记录

1. **Transformer 不学习**：初始 lr=5e-4 太小，改为 1e-3 + warmup 后恢复
2. **listings 等号消失**：`columns=fullflexible` 会吞掉 `=`，改为 `columns=fixed`
3. **Subword 分词失败**：BERT WordPiece 词表不适合从头训练
4. **维度不连续报错**：`transpose` 后必须 `contiguous()` 才能 `view()`
5. **matplotlib 中文乱码**：需设置 `font.sans-serif` 为 `Noto Sans CJK SC`
