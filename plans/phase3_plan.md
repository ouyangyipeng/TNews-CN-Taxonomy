# 阶段三：硬核训练与超参搜索 (Training & Tuning)

## 执行时间
2026-05-30

## 目标
在 GPU 上完成多组消融实验，对比不同模型架构 × 不同分词方式的组合表现。

## 实验矩阵

| 实验编号 | 模型 | 分词方式 | max_len | batch_size | lr | 结果 (Macro-F1) |
|---------|------|---------|---------|------------|-----|----------------|
| exp1 | BiLSTM+Attention | Char | 50 | 64 | 0.001 | 0.5027 |
| exp2 | BiLSTM+Attention | Word | 30 | 64 | 0.001 | 0.4708 |
| exp3 | Transformer | Char | 50 | 64 | 0.001 | 0.4093 |
| exp4 | Transformer | Word | 30 | 64 | 0.001 | 0.4222 |
| exp5 | Transformer | Subword | 50 | 64 | 0.001 | 失败(导入冲突) |

## 训练策略

### 优化器
- AdamW (lr=0.001, weight_decay=1e-4)
- 学习率调度：带 warmup 的余弦退火（warmup 5 epochs）

### 正则化
- BiLSTM: Dropout=0.3
- Transformer: Dropout=0.1
- 梯度裁剪: max_norm=1.0

### Early Stopping
- 监控指标: 验证集 Macro-F1
- Patience: 5 epochs
- 保存最佳 checkpoint

## 关键发现

1. **BiLSTM+Attention 优于 Transformer**：短文本上 BiLSTM 更适合
2. **Char-level 优于 Word-level**：短文本保留更多细粒度信息
3. **Transformer 需要 warmup**：否则初始阶段不学习
4. **过拟合明显**：BiLSTM 在 epoch 5 后开始过拟合

## 遇到的问题与解决

1. **ReduceLROnPlateau verbose 参数**：PyTorch 2.12 移除了该参数，改用 LambdaLR
2. **Transformer 不学习**：初始 lr 过大导致，添加 warmup 后解决
3. **Subword 导入冲突**：src/tokenizers.py 与 transformers 库的 tokenizers 模块冲突

## 交付物
- [x] `checkpoints/bilstm_char/` - 最佳模型
- [x] `checkpoints/bilstm_word/`
- [x] `checkpoints/transformer_char_v2/`
- [x] `checkpoints/transformer_word/`
- [x] `logs/*/training_log.json` - 训练日志
