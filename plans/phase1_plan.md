# 阶段一：数据工程与探索性分析 (Data Engineering & EDA)

## 执行时间
2026-05-30

## 目标
完成 TNEWS 数据集的完整探索性分析，构建高质量的数据处理流水线，为后续模型训练提供干净、规范的数据输入。

## 核心任务

### 1. 数据目录搭建与原始数据复制
**输入**: `docs/tnews_public/` 中的原始数据  
**输出**: `data/` 目录结构

**操作步骤**:
- 创建 `data/raw/` 存放原始数据副本
- 创建 `data/processed/` 存放清洗后的数据
- 创建 `data/vocab/` 存放词表文件
- 创建 `data/splits/` 存放划分后的训练/验证集
- 复制 `train.json`, `dev.json`, `test.json`, `labels.json` 到 `data/raw/`

**验收标准**: 目录结构清晰，原始数据完整备份

---

### 2. EDA 脚本编写
**输入**: `data/raw/train.json`  
**输出**: `report/figures/` 中的统计图表 + `data/eda_report.md`

**统计维度**:
1. **类别分布分析**
   - 统计 15 个类别的样本数量
   - 计算类别不平衡比例（最大类/最小类）
   - 绘制类别分布柱状图（`class_distribution.png`）
   - 识别长尾类别

2. **文本长度分布**
   - 按字符数统计（Char-level）
   - 按词数统计（Word-level, jieba 分词后）
   - 计算 50/90/95/99 分位数
   - 绘制长度分布直方图（`text_length_distribution.png`）
   - 确定 `max_len` 参数（建议取 95 或 99 分位数）

3. **关键词字段分析**
   - 统计 `keywords` 字段的缺失率
   - 分析关键词数量分布
   - 评估是否将关键词作为辅助特征

4. **文本质量检查**
   - 检测空文本、重复样本
   - 检测异常字符（如 HTML 实体、特殊符号）
   - 检测全角/半角混用情况

**技术栈**:
- `pandas` 数据处理
- `matplotlib` + `seaborn` 可视化
- `jieba` 分词
- `collections.Counter` 统计

**验收标准**: 
- 生成至少 3 张高质量统计图表
- 输出明确的 `max_len` 建议值
- 识别出数据质量问题并记录

---

### 3. 数据清洗模块
**输入**: `data/raw/train.json`, `data/raw/dev.json`  
**输出**: `data/processed/train_clean.json`, `data/processed/dev_clean.json`

**清洗规则**:
1. **去除无效样本**
   - 删除 `sentence` 为空的样本
   - 删除 `label` 不在 `labels.json` 中的样本
   - 删除完全重复的样本（基于 `sentence` 去重）

2. **文本规范化**
   - 统一全角字符为半角（数字、英文字母、标点）
   - 去除多余空格（连续空格合并为单个）
   - 去除不可见字符（如 `\u200b`, `\xa0`）
   - 去除 HTML 实体（如 `&`, `<`）
   - 保留中文标点，去除无意义的英文标点堆叠

3. **标签映射**
   - 将字符串标签（如 `"100"`）映射为整数索引（0-14）
   - 构建 `label2id.json` 和 `id2label.json`

**实现方式**:
- 编写 `src/data_cleaner.py` 模块
- 使用 `unicodedata` 处理 Unicode 规范化
- 使用正则表达式处理特殊字符

**验收标准**:
- 清洗前后样本数量对比记录
- 清洗日志输出到 `data/clean_log.txt`
- 清洗后数据无空文本、无非法标签

---

### 4. 多粒度分词实现
**输入**: `data/processed/train_clean.json`  
**输出**: 三种分词结果文件

**分词策略**:

#### 4.1 字符级（Char-level）
- 将每个汉字、英文字母、数字、标点视为独立 token
- 优点：无 OOV 问题，词表小（~5000）
- 缺点：序列长，语义信息稀疏
- 输出格式：`["北", "京", "天", "气", "晴"]`

#### 4.2 词级（Word-level, jieba）
- 使用 `jieba.lcut()` 进行中文分词
- 保留标点符号（可选）
- 优点：语义单元完整，序列较短
- 缺点：OOV 问题严重，需处理低频词
- 输出格式：`["北京", "天气", "晴"]`

#### 4.3 子词级（Subword, BPE/WordPiece）
- 使用 `transformers` 库的 `BertTokenizer`（仅分词器，不用模型）
- 或使用 `sentencepiece` 训练中文 BPE
- 优点：平衡 OOV 与序列长度
- 缺点：需额外训练/加载词表
- 输出格式：`["北", "##京", "天", "##气", "晴"]`

**实现方式**:
- 编写 `src/tokenizers.py` 模块
- 封装三个类：`CharTokenizer`, `WordTokenizer`, `SubwordTokenizer`
- 统一接口：`encode(text) -> List[str]`

**验收标准**:
- 三种分词器均可正常工作
- 对同一文本输出分词结果对比
- 记录各分词方式的平均序列长度

---

### 5. 词表构建与 Dataset/DataLoader
**输入**: 分词后的训练集文本  
**输出**: 词表文件 + PyTorch Dataset 类

**词表构建规则**:
1. **仅使用训练集**（严禁使用验证集/测试集）
2. **最小词频阈值**：`min_freq=2`（去除低频词）
3. **特殊 token**：
   - `<PAD>`: 填充符，索引 0
   - `<UNK>`: 未知词，索引 1
   - `<BOS>`: 句子开始，索引 2（可选）
   - `<EOS>`: 句子结束，索引 3（可选）
4. **词表大小**：
   - Char-level: ~5000-8000
   - Word-level: ~30000-50000（截断至 `max_vocab_size`）
   - Subword: 使用预训练词表（~21128 for BERT-base-chinese）

**Dataset 实现**:
```python
class TNEWSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, vocab):
        # 存储原始文本、标签、分词器、词表
    def __len__(self):
        # 返回样本数
    def __getitem__(self, idx):
        # 返回: input_ids, attention_mask, label
```

**DataLoader 配置**:
- `batch_size`: 32/64/128（待超参搜索）
- `shuffle`: True for train, False for val/test
- `collate_fn`: 动态 padding 至 batch 内最大长度

**数据划分**:
- 从 `train_clean.json` 中划分 90% 训练 / 10% 验证
- 固定随机种子 `seed=42`
- 使用 `sklearn.model_selection.train_test_split`
- 输出划分统计到 `data/splits/split_info.json`

**验收标准**:
- 词表文件保存到 `data/vocab/`
- Dataset 可正常迭代，输出 shape 正确
- 划分后训练集 ~48000 条，验证集 ~5300 条
- 验证集类别分布与训练集一致（分层抽样）

---

## 技术栈
- Python 3.8+
- PyTorch 2.0+
- pandas, numpy
- matplotlib, seaborn
- jieba
- transformers (仅 Tokenizer)
- scikit-learn

## 风险与应对
1. **数据下载失败**: 已备份在 `docs/tnews_public/`
2. **jieba 分词慢**: 使用多进程加速或预分词缓存
3. **BPE 训练失败**: 回退到 Char-level
4. **类别极度不平衡**: 考虑类别权重或过采样

## 交付物清单
- [ ] `data/` 目录结构
- [ ] `src/data_cleaner.py`
- [ ] `src/tokenizers.py`
- [ ] `src/dataset.py`
- [ ] `scripts/eda.py`
- [ ] `report/figures/class_distribution.png`
- [ ] `report/figures/text_length_distribution.png`
- [ ] `data/eda_report.md`
- [ ] `data/vocab/char_vocab.json`
- [ ] `data/vocab/word_vocab.json`
- [ ] `data/splits/split_info.json`

## 下一步
完成阶段一后，进入阶段二：模型架构设计，实现 BiLSTM+Attention 和 Transformer 分类器。
