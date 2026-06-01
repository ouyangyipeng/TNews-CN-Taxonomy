# 阶段二：模型架构设计 (Architecture Engineering)

## 执行时间
2026-05-30

## 目标
手动实现两种高质量文本分类模型：
1. **BiLSTM + Bahdanau Attention**: 双向 LSTM 编码器 + 加性注意力机制
2. **Transformer Encoder**: 多头自注意力 + 位置编码 + 前馈网络

所有核心组件必须使用 `torch.nn` 基础算子手动搭建，禁止调用现成分类模型。

## 核心任务

### 1. BiLSTM + Bahdanau Attention 模型

**架构设计**:
```
Input (batch, seq_len)
  ↓
Embedding (vocab_size, embed_dim)
  ↓
BiLSTM (embed_dim, hidden_dim, num_layers, bidirectional=True)
  ↓
Encoder Output: (batch, seq_len, hidden_dim * 2)
  ↓
Bahdanau Attention:
  - Query: 可学习参数 or 最后一个 hidden state
  - Key/Value: encoder output
  - Attention Score: V^T * tanh(W1*query + W2*key)
  - Context Vector: sum(attention_weight * value)
  ↓
Concat [context_vector, last_hidden_state]
  ↓
FC (hidden_dim * 3, num_classes)
  ↓
Output (batch, num_classes)
```

**关键实现细节**:
1. **Embedding 层**: 使用 `nn.Embedding`，支持 padding_idx
2. **BiLSTM**: 
   - `num_layers=2`，`dropout=0.3`
   - `bidirectional=True`，输出维度为 `hidden_dim * 2`
   - 返回所有时间步的 hidden states 和最后一个 hidden state
3. **Bahdanau Attention**:
   - 公式: $e_{ti} = v_a^T \tanh(W_a s_{t-1} + U_a h_i)$
   - $\alpha_{ti} = \text{softmax}(e_{ti})$
   - $c_t = \sum_i \alpha_{ti} h_i$
   - 使用 `nn.Linear` 实现 $W_a$ 和 $U_a$
   - 使用 `nn.Parameter` 实现 $v_a$
4. **分类头**: 单层 FC + Dropout

**文件**: `src/models/bilstm_attention.py`

---

### 2. Transformer Encoder 模型

**架构设计**:
```
Input (batch, seq_len)
  ↓
Embedding (vocab_size, d_model)
  ↓
Positional Encoding (sinusoidal)
  ↓
Transformer Encoder (num_layers=4):
  - Multi-Head Self-Attention (num_heads=8)
  - Add & Norm (LayerNorm)
  - Feed-Forward Network (d_ff=2048)
  - Add & Norm
  ↓
Global Average Pooling or [CLS] token
  ↓
FC (d_model, num_classes)
  ↓
Output (batch, num_classes)
```

**关键实现细节**:

#### 2.1 Positional Encoding
- 使用正弦/余弦位置编码（固定，不可学习）
- 公式:
  - $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
  - $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$
- 实现为 `nn.Module`，在 forward 中直接加到 embedding 上

#### 2.2 Multi-Head Self-Attention
- **手动实现 Scaled Dot-Product Attention**:
  - $Q, K, V = X W^Q, X W^K, X W^V$
  - $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) V$
  - 使用 `torch.bmm` 或 `torch.matmul` 实现矩阵乘法
  - 使用 `masked_fill` 实现 padding mask
- **Multi-Head**:
  - 将 $Q, K, V$ 拆分为 `num_heads` 个头
  - 每个头独立计算 attention
  - 拼接后通过线性层 $W^O$
- **维度变换**:
  - 输入: `(batch, seq_len, d_model)`
  - 拆分: `(batch, num_heads, seq_len, d_k)` where `d_k = d_model / num_heads`
  - 输出: `(batch, seq_len, d_model)`

#### 2.3 Feed-Forward Network
- 两层线性变换 + ReLU
- $FFN(x) = \max(0, x W_1 + b_1) W_2 + b_2$
- $d_{ff} = 4 \times d_{model}$（通常 2048）

#### 2.4 Layer Normalization & Residual Connection
- 使用 `nn.LayerNorm`
- 残差连接: $output = \text{LayerNorm}(x + \text{Sublayer}(x))$

#### 2.5 池化策略
- **方案 A**: Global Average Pooling（对所有时间步取平均，忽略 padding）
- **方案 B**: 使用 [CLS] token（在序列开头添加特殊 token，取其输出）
- 推荐方案 A，实现更简单且效果相当

**文件**: `src/models/transformer.py`

---

### 3. 模型配置与工厂函数

**配置文件**: `configs/model_config.yaml`
```yaml
bilstm_attention:
  embed_dim: 300
  hidden_dim: 256
  num_layers: 2
  dropout: 0.3
  attention_dim: 128

transformer:
  d_model: 512
  num_heads: 8
  num_layers: 4
  d_ff: 2048
  dropout: 0.1
  max_len: 128
```

**工厂函数**: `src/models/__init__.py`
```python
def get_model(model_type: str, vocab_size: int, num_classes: int, **kwargs):
    if model_type == 'bilstm_attention':
        return BiLSTMAttention(vocab_size, num_classes, **kwargs)
    elif model_type == 'transformer':
        return TransformerClassifier(vocab_size, num_classes, **kwargs)
```

---

### 4. 代码质量要求

1. **类型注解**: 所有函数必须有完整的 type hints
2. **文档字符串**: 每个类和函数必须有 docstring，说明输入输出和维度
3. **维度注释**: 在关键张量操作旁注释维度变化
4. **数学公式**: 在注释中写明对应的数学公式
5. **可解释性**: Attention 模型必须返回 attention weights，用于后续可视化

---

## 技术栈
- PyTorch 2.0+
- torch.nn (基础算子)
- torch.nn.functional (函数式 API)

## 风险与应对
1. **维度不匹配**: 在每个模块的 forward 中添加 shape assert
2. **梯度消失/爆炸**: 使用 Xavier/Kaiming 初始化，添加 LayerNorm
3. **过拟合**: Dropout + Early Stopping
4. **训练慢**: 使用 `pin_memory=True` 和 `non_blocking=True`

## 交付物清单
- [ ] `src/models/__init__.py`
- [ ] `src/models/bilstm_attention.py`
- [ ] `src/models/transformer.py`
- [ ] `configs/model_config.yaml`
- [ ] 单元测试脚本（验证维度变换正确性）

## 下一步
完成模型实现后，进入阶段三：训练与超参搜索。
