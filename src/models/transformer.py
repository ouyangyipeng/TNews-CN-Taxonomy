#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Encoder 文本分类模型

架构：
1. Embedding 层：将 token ID 映射为稠密向量
2. Positional Encoding：添加位置信息（正弦/余弦编码）
3. Transformer Encoder：多层 Multi-Head Self-Attention + FFN
4. Pooling：对 encoder 输出进行全局平均池化
5. 分类头：全连接层输出类别 logits

数学公式：
- Positional Encoding:
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

- Scaled Dot-Product Attention:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

- Multi-Head Attention:
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
  where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

- Feed-Forward Network:
  FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

作者：手动实现，基于 PyTorch nn 基础算子
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码
    
    公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    输入：(batch, seq_len, d_model)
    输出：(batch, seq_len, d_model) - 添加了位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度（embedding 维度）
            max_len: 最大序列长度
            dropout: Dropout 概率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 扩展为 (1, max_len, d_model) 以便广播
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与梯度更新，但会随模型保存/加载）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            (batch, seq_len, d_model) - 添加了位置编码
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    输入：
    - Q: (batch, num_heads, seq_len, d_k)
    - K: (batch, num_heads, seq_len, d_k)
    - V: (batch, num_heads, seq_len, d_v)
    - mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
    
    输出：
    - output: (batch, num_heads, seq_len, d_v)
    - attention_weights: (batch, num_heads, seq_len, seq_len)
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, num_heads, seq_len, d_k)
            key: (batch, num_heads, seq_len, d_k)
            value: (batch, num_heads, seq_len, d_v)
            mask: (batch, 1, 1, seq_len) - padding mask
        
        Returns:
            output: (batch, num_heads, seq_len, d_v)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Step 1: 计算 QK^T
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Step 2: 应用 mask（将 padding 位置的 score 设为负无穷）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 3: Softmax 归一化
        # attention_weights: (batch, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Step 4: 加权求和
        # output: (batch, num_heads, seq_len, d_v)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    公式：
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    输入：
    - query: (batch, seq_len, d_model)
    - key: (batch, seq_len, d_model)
    - value: (batch, seq_len, d_model)
    - mask: (batch, 1, 1, seq_len)
    
    输出：
    - output: (batch, seq_len, d_model)
    - attention_weights: (batch, num_heads, seq_len, seq_len)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout 概率
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention 计算
        self.attention = ScaledDotProductAttention(dropout)
        
        # 初始化
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape
        
        # Step 1: 线性投影
        # Q, K, V: (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Step 2: 拆分为多个头
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 3: 计算 attention
        # attn_output: (batch, num_heads, seq_len, d_k)
        # attention_weights: (batch, num_heads, seq_len, seq_len)
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Step 4: 拼接所有头
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Step 5: 最终线性投影
        # output: (batch, seq_len, d_model)
        output = self.W_o(attn_output)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    公式：
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    
    输入：(batch, seq_len, d_model)
    输出：(batch, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: FFN 中间层维度（通常为 4 * d_model）
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # 初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder 单层
    
    架构：
    1. Multi-Head Self-Attention
    2. Add & Norm (残差连接 + LayerNorm)
    3. Feed-Forward Network
    4. Add & Norm
    
    输入：(batch, seq_len, d_model)
    输出：(batch, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: FFN 中间层维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Step 1: Self-Attention + Add & Norm
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Step 2: FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attention_weights


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder 文本分类器
    
    架构流程：
    1. Embedding: (batch, seq_len) -> (batch, seq_len, d_model)
    2. Positional Encoding: 添加位置信息
    3. Transformer Encoder: 多层 encoder layer
    4. Global Average Pooling: 对所有时间步取平均
    5. Classification: FC -> logits
    
    输入：
    - input_ids: (batch, seq_len) - token ID 序列
    - attention_mask: (batch, seq_len) - 1 表示有效，0 表示 padding
    
    输出：
    - logits: (batch, num_classes) - 类别 logits
    - attention_weights: (batch, num_layers, num_heads, seq_len, seq_len) - 所有层的注意力权重
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 2048,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """
        Args:
            vocab_size: 词表大小
            num_classes: 分类类别数
            d_model: 模型维度（embedding 维度）
            num_heads: 注意力头数
            num_layers: Encoder 层数
            d_ff: FFN 中间层维度
            max_len: 最大序列长度
            dropout: Dropout 概率
            pad_token_id: padding token 的 ID
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # 1. Embedding 层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id
        )
        
        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 3. Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # Embedding 初始化
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # padding 位置保持为 0
        if self.pad_token_id is not None:
            self.embedding.weight.data[self.pad_token_id].zero_()
        
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
            all_attention_weights: (batch, num_layers, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Embedding + Positional Encoding
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Step 2: 构建 padding mask
        # mask: (batch, 1, 1, seq_len) - 用于 attention 计算
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        # Step 3: Transformer Encoder
        all_attention_weights = []
        for encoder_layer in self.encoder_layers:
            x, attention_weights = encoder_layer(x, mask)
            all_attention_weights.append(attention_weights)
        
        # Step 4: Global Average Pooling（忽略 padding 位置）
        if attention_mask is not None:
            # (batch, seq_len, d_model) * (batch, seq_len, 1)
            x = x * attention_mask.unsqueeze(-1)
            # 求和并除以有效长度
            x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        # Step 5: 分类
        # logits: (batch, num_classes)
        logits = self.classifier(x)
        
        # 堆叠所有层的 attention weights
        # all_attention_weights: (batch, num_layers, num_heads, seq_len, seq_len)
        all_attention_weights = torch.stack(all_attention_weights, dim=1)
        
        return logits, all_attention_weights


def test_model():
    """测试模型维度变换"""
    print("="*60)
    print("Transformer Encoder 模型测试")
    print("="*60)
    
    # 超参数
    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    num_classes = 15
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 256
    
    # 创建模型
    model = TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # 创建输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # 模拟 padding
    attention_mask[0, 15:] = 0
    attention_mask[1, 18:] = 0
    
    # 切换到 eval 模式（关闭 dropout，确保 attention weights 归一化）
    model.eval()
    
    # 前向传播
    logits, all_attention_weights = model(input_ids, attention_mask)
    
    # 验证输出维度
    print(f"\n输入维度:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    print(f"\n输出维度:")
    print(f"  logits: {logits.shape}")
    print(f"  all_attention_weights: {all_attention_weights.shape}")
    
    assert logits.shape == (batch_size, num_classes), f"logits 维度错误: {logits.shape}"
    assert all_attention_weights.shape == (batch_size, num_layers, num_heads, seq_len, seq_len), \
        f"attention_weights 维度错误: {all_attention_weights.shape}"
    
    # 验证 attention weights 归一化
    weight_sum = all_attention_weights.sum(dim=-1)
    print(f"\nAttention weights 归一化检查:")
    print(f"  权重和 (应接近 1): {weight_sum[0, 0, 0, :5].tolist()}")
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), "Attention weights 未归一化"
    
    print("\n✅ 所有维度检查通过!")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")


if __name__ == '__main__':
    test_model()
