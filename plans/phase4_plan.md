# 阶段四：深入解析与可视化 (Analysis & Visualization)

## 执行时间
2026-05-30

## 目标
对最佳模型（BiLSTM+Attention+Char）进行全面的可视化分析和错误归因。

## 可视化任务

### 1. 训练曲线
- Loss 曲线（Train vs Val）
- Accuracy 曲线
- Macro-F1 曲线
- 学习率调度曲线
- 输出: `report/figures/bilstm_char/training_curves.png`

### 2. 混淆矩阵
- 使用 seaborn 绘制 15×15 混淆矩阵热力图
- 识别容易混淆的类别对
- 输出: `report/figures/bilstm_char/confusion_matrix.png`

### 3. Attention Weights 热力图
- 选取 3 条典型样本
- 可视化 Bahdanau Attention 的权重分布
- 分析模型关注了哪些关键字符
- 输出: `report/figures/bilstm_char/attention_heatmap_*.png`

### 4. t-SNE 特征降维聚类图
- 提取 1000 个测试样本的 sentence embedding
- 使用 t-SNE 降维到 2D
- 按类别着色，观察聚类效果
- 输出: `report/figures/bilstm_char/tsne_visualization.png`

### 5. Bad Case 分析
- 提取 10 个预测错误的典型样本
- 分析错判原因（语义重叠、样本不足、文本歧义等）
- 输出: `report/figures/bilstm_char/bad_case_analysis.md`

## 关键发现

### 表现最好的类别
- news_sports (F1=0.64): 关键词明显（世界杯、乒乓球等）
- news_car (F1=0.61): 领域特征强（发动机、车型等）
- news_game (F1=0.56): 游戏名称易识别

### 表现最差的类别
- news_stock (F1=0.24): 样本极少（仅 45 条测试样本）
- news_finance (F1=0.43): 与 news_house 语义重叠
- news_travel (F1=0.44): 与 news_culture 语义重叠

### 常见混淆对
- news_finance ↔ news_house: 都涉及房地产、经济
- news_travel ↔ news_culture: 都涉及景点、历史文化
- news_tech ↔ news_car: 都涉及新技术、新产品

## 交付物
- [x] 训练曲线图
- [x] 混淆矩阵图
- [x] Attention 热力图 (3 张)
- [x] t-SNE 可视化图
- [x] Bad Case 分析报告
