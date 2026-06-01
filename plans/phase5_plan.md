# 阶段五：LaTeX 报告撰写

## 执行时间
2026-05-30

## 目标
撰写完整的实验报告，覆盖课程要求的八大结构。

## 报告结构

### 1. 任务描述
- 任务目标：TNEWS 15 分类
- 数据集说明：53,360 条训练样本，15 个类别
- 评价指标：Accuracy + Macro-F1

### 2. 数据分析与预处理
- 类别分布统计与不平衡分析
- 文本长度分布（字符级 + 词级）
- 数据清洗流程
- 多粒度分词策略
- 词表构建规则

### 3. 模型设计说明
- BiLSTM + Bahdanau Attention 架构
  - 数学公式推导
  - 维度变换说明
- Transformer Encoder 架构
  - Positional Encoding
  - Multi-Head Self-Attention
  - FFN + LayerNorm

### 4. 训练设置
- 数据划分：90%/10% 分层抽样
- 损失函数：Cross-Entropy
- 优化器：AdamW + warmup cosine
- Early Stopping 机制
- 硬件环境

### 5. 实验结果
- 消融实验对比表
- 最佳模型测试结果
- 类别级 F1 分数
- 训练过程分析

### 6. 结果分析
- 模型优势分析
- 失败案例分析（3 个典型 Bad Case）
- Attention 可视化解读
- t-SNE 聚类分析
- 模型仍存在的问题

### 7. 心得体会
- Transformer warmup 的重要性
- 维度匹配调试经验
- 分词策略选择的思考
- Early Stopping 的作用

### 8. 学术诚信说明
- 独立完成声明
- 参考资源列表

## 编译命令
```bash
cd report
xelatex main.tex
xelatex main.tex  # 二次编译确保目录和引用正确
```

## 交付物
- [x] `report/main.tex` - LaTeX 源文件
- [ ] `report/main.pdf` - 编译后的 PDF（需重新编译）
