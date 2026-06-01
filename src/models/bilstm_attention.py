#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiLSTM + Bahdanau Attention 文本分类模型

架构：
1. Embedding 层：将 token ID 映射为稠密向量
2. BiLSTM 编码器：双向 LSTM 提取上下文特征
3. Bahdanau Attention：加性注意力机制，计算 context vector
4. 分类头：全连接层输出类别 logits

数学公式：
- BiLSTM: h_t = [h_t^forward; h_t^backward]
- Attention Score: e_ti = v_a^T * tanh(W_a * s_{t-1} + U_a * h_i)
- Attention Weight: α_ti = softmax(e_ti)
- Context Vector: c_t = Σ_i α_ti * h_i
- Output: y = W * [c_t; h_T] + b

作者：手动实现，基于 PyTorch nn 基础算子
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention 机制
    
    公式：
    e_ti = v_a^T * tanh(W_a * query + U_a * key)
    α_ti = softmax(e_ti)
    c_t = Σ_i α_ti * value
    
    输入维度：
    - query: (batch, query_dim) - 通常是 decoder 的 hidden state
    - key: (batch, seq_len, key_dim) - encoder 的所有 hidden states
    - value: (batch, seq_len, value_dim) - 通常与 key 相同
    
    输出维度：
    - context: (batch, value_dim) - 加权求和后的 context vector
    - attention_weights: (batch, seq_len) - 注意力权重分布
    """
    
    def __init__(self, query_dim: int, key_dim: int, attention_dim: int):
        """
        Args:
            query_dim: query 向量的维度（如 LSTM hidden_dim * 2）
            key_dim: key 向量的维度（如 LSTM hidden_dim * 2）
            attention_dim: 中间隐藏层维度（超参数）
        """
        super().__init__()
        
        # W_a: 将 query 投影到 attention 空间
        self.W_a = nn.Linear(query_dim, attention_dim, bias=False)
        
        # U_a: 将 key 投影到 attention 空间
        self.U_a = nn.Linear(key_dim, attention_dim, bias=False)
        
        # v_a: 可学习的权重向量，用于计算最终 score
        self.v_a = nn.Parameter(torch.randn(attention_dim))
        
        # 初始化
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.U_a.weight)
        nn.init.normal_(self.v_a, mean=0, std=0.01)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, query_dim)
            key: (batch, seq_len, key_dim)
            value: (batch, seq_len, value_dim)
            mask: (batch, seq_len) - 1 表示有效位置，0 表示 padding
        
        Returns:
            context: (batch, value_dim)
            attention_weights: (batch, seq_len)
        """
        batch_size, seq_len, _ = key.shape
        
        # Step 1: 计算 query 和 key 的投影
        # query_proj: (batch, attention_dim)
        query_proj = self.W_a(query)
        
        # key_proj: (batch, seq_len, attention_dim)
        key_proj = self.U_a(key)
        
        # Step 2: 计算 attention score
        # 将 query_proj 扩展为 (batch, 1, attention_dim) 以便广播
        query_proj = query_proj.unsqueeze(1)
        
        # energy: (batch, seq_len, attention_dim)
        # tanh(W_a * query + U_a * key)
        energy = torch.tanh(query_proj + key_proj)
        
        # score: (batch, seq_len)
        # v_a^T * energy
        score = torch.sum(self.v_a * energy, dim=-1)
        
        # Step 3: 应用 mask（将 padding 位置的 score 设为负无穷）
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        
        # Step 4: Softmax 归一化得到 attention weights
        # attention_weights: (batch, seq_len)
        attention_weights = F.softmax(score, dim=-1)
        
        # Step 5: 加权求和得到 context vector
        # context: (batch, value_dim)
        # attention_weights: (batch, seq_len, 1)
        # value: (batch, seq_len, value_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), value).squeeze(1)
        
        return context, attention_weights


class BiLSTMAttention(nn.Module):
    """
    BiLSTM + Bahdanau Attention 文本分类器
    
    架构流程：
    1. Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
    2. BiLSTM: (batch, seq_len, embed_dim) -> (batch, seq_len, hidden_dim * 2)
    3. Attention: 计算 context vector
    4. Classification: FC([context; last_hidden]) -> logits
    
    输入：
    - input_ids: (batch, seq_len) - token ID 序列
    - attention_mask: (batch, seq_len) - 1 表示有效，0 表示 padding
    
    输出：
    - logits: (batch, num_classes) - 类别 logits
    - attention_weights: (batch, seq_len) - 注意力权重（用于可视化）
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 128,
        pad_token_id: int = 0
    ):
        """
        Args:
            vocab_size: 词表大小
            num_classes: 分类类别数
            embed_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏层维度（单向）
            num_layers: LSTM 层数
            dropout: Dropout 概率
            attention_dim: Attention 中间维度
            pad_token_id: padding token 的 ID
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # 1. Embedding 层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_token_id
        )
        
        # 2. BiLSTM 编码器
        # bidirectional=True 使得输出维度为 hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3. Bahdanau Attention
        # query 使用最后一个 hidden state（拼接正向和反向）
        # key 和 value 使用所有时间步的 hidden states
        self.attention = BahdanauAttention(
            query_dim=hidden_dim * 2,  # 双向拼接
            key_dim=hidden_dim * 2,
            attention_dim=attention_dim
        )
        
        # 4. 分类头
        # 输入：[context_vector; last_hidden_state] = hidden_dim * 2 + hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
        # Dropout for embedding
        self.embed_dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # Embedding 初始化
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # padding 位置保持为 0
        if self.pad_token_id is not None:
            self.embedding.weight.data[self.pad_token_id].zero_()
        
        # LSTM 权重由 PyTorch 默认初始化（Xavier）
        
        # 分类头初始化
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len) - token ID 序列
            attention_mask: (batch, seq_len) - 1 表示有效，0 表示 padding
        
        Returns:
            logits: (batch, num_classes) - 类别 logits
            attention_weights: (batch, seq_len) - 注意力权重
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Embedding
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(input_ids)
        embedded = self.embed_dropout(embedded)
        
        # Step 2: BiLSTM 编码
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        # hidden: (num_layers * 2, batch, hidden_dim) - 最后一层所有时间步的 hidden state
        # cell: (num_layers * 2, batch, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Step 3: 提取最后一个 hidden state 作为 query
        # hidden: (num_layers * 2, batch, hidden_dim)
        # 取最后一层的正向和反向 hidden state 拼接
        # hidden[-2]: 最后一层正向的 hidden state (batch, hidden_dim)
        # hidden[-1]: 最后一层反向的 hidden state (batch, hidden_dim)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (batch, hidden_dim * 2)
        
        # Step 4: Bahdanau Attention
        # context: (batch, hidden_dim * 2)
        # attention_weights: (batch, seq_len)
        context, attention_weights = self.attention(
            query=last_hidden,
            key=lstm_out,
            value=lstm_out,
            mask=attention_mask
        )
        
        # Step 5: 拼接 context 和 last_hidden
        # combined: (batch, hidden_dim * 4)
        combined = torch.cat([context, last_hidden], dim=-1)
        
        # Step 6: 分类
        # logits: (batch, num_classes)
        logits = self.classifier(combined)
        
        return logits, attention_weights


def test_model():
    """测试模型维度变换"""
    print("="*60)
    print("BiLSTM + Attention 模型测试")
    print("="*60)
    
    # 超参数
    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    num_classes = 15
    embed_dim = 128
    hidden_dim = 64
    num_layers = 2
    
    # 创建模型
    model = BiLSTMAttention(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # 创建输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # 模拟 padding
    attention_mask[0, 15:] = 0
    attention_mask[1, 18:] = 0
    
    # 前向传播
    logits, attention_weights = model(input_ids, attention_mask)
    
    # 验证输出维度
    print(f"\n输入维度:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    print(f"\n输出维度:")
    print(f"  logits: {logits.shape}")
    print(f"  attention_weights: {attention_weights.shape}")
    
    assert logits.shape == (batch_size, num_classes), f"logits 维度错误: {logits.shape}"
    assert attention_weights.shape == (batch_size, seq_len), f"attention_weights 维度错误: {attention_weights.shape}"
    
    # 验证 attention weights 归一化
    weight_sum = attention_weights.sum(dim=-1)
    print(f"\nAttention weights 归一化检查:")
    print(f"  权重和: {weight_sum}")
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), "Attention weights 未归一化"
    
    # 验证 padding 位置的 attention weight 接近 0
    print(f"\nPadding 位置 attention weight 检查:")
    print(f"  样本 0, 位置 15-19: {attention_weights[0, 15:].tolist()}")
    
    print("\n✅ 所有维度检查通过!")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")


if __name__ == '__main__':
    test_model()
