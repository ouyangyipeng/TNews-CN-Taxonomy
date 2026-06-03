#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Feature Extractor + Custom Classifier

架构：
1. BERT Base (frozen): 提取上下文嵌入
2. Pooling: 对BERT输出进行池化
3. Custom Classifier: 自定义分类头

注意：这不是使用 BertForSequenceClassification，
而是仅使用 BERT 的基础模型作为特征提取器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BERTClassifier(nn.Module):
    """
    BERT Feature Extractor + Custom Classification Head
    
    架构流程：
    1. BERT (frozen): (batch, seq_len) -> (batch, seq_len, 768)
    2. Pooling: (batch, seq_len, 768) -> (batch, 768)
    3. Classifier: (batch, 768) -> (batch, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int,
        bert_model_name: str = 'bert-base-chinese',
        pooling_strategy: str = 'cls',  # 'cls', 'mean', 'max'
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        local_model_path: Optional[str] = None
    ):
        """
        Args:
            num_classes: 分类类别数
            bert_model_name: BERT模型名称
            pooling_strategy: 池化策略 ('cls', 'mean', 'max')
            hidden_dim: 分类头隐藏层维度
            dropout: Dropout概率
            freeze_bert: 是否冻结BERT参数
            local_model_path: 本地模型路径（用于离线环境）
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        
        # 1. BERT Feature Extractor
        try:
            from transformers import AutoModel, AutoConfig
            # 优先使用本地模型路径
            model_path = local_model_path if local_model_path else bert_model_name
            # 先加载配置获取隐藏维度
            config = AutoConfig.from_pretrained(model_path)
            self.bert_dim = config.hidden_size
            print(f"  模型隐藏维度: {self.bert_dim}")
            # 加载模型
            self.bert = AutoModel.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"无法加载BERT模型: {e}")
        
        # 冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 2. Custom Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 初始化分类头
        self._init_classifier()
    
    def _init_classifier(self):
        """初始化分类头权重"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _pool_output(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        对BERT输出进行池化
        
        Args:
            last_hidden_state: (batch, seq_len, 768)
            attention_mask: (batch, seq_len)
        
        Returns:
            pooled: (batch, 768)
        """
        if self.pooling_strategy == 'cls':
            # 使用[CLS] token的输出
            return last_hidden_state[:, 0, :]
        
        elif self.pooling_strategy == 'mean':
            # 平均池化（忽略padding）
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
            mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / mask_sum
        
        elif self.pooling_strategy == 'max':
            # 最大池化（忽略padding）
            mask_expanded = attention_mask.unsqueeze(-1).float()
            last_hidden_state = last_hidden_state.masked_fill(
                mask_expanded == 0, -1e9
            )
            return last_hidden_state.max(dim=1)[0]
        
        else:
            raise ValueError(f"未知的池化策略: {self.pooling_strategy}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) - 可选
        
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, seq_len) - 返回attention_mask作为占位符
        """
        # BERT特征提取
        with torch.no_grad() if not any(p.requires_grad for p in self.bert.parameters()) else torch.enable_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # 池化
        pooled = self._pool_output(last_hidden_state, attention_mask)  # (batch, 768)
        
        # 分类
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        # 返回attention_mask作为attention_weights的占位符（用于兼容接口）
        return logits, attention_mask


def test_model():
    """测试模型"""
    print("="*60)
    print("BERT Classifier 模型测试")
    print("="*60)
    
    # 超参数
    batch_size = 4
    seq_len = 20
    num_classes = 15
    
    # 创建模型
    model = BERTClassifier(
        num_classes=num_classes,
        pooling_strategy='cls',
        hidden_dim=256,
        dropout=0.3,
        freeze_bert=True
    )
    
    # 创建输入
    input_ids = torch.randint(0, 21128, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[0, 15:] = 0
    
    # 前向传播
    logits, attn_weights = model(input_ids, attention_mask)
    
    # 验证输出
    print(f"\n输入维度:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    print(f"\n输出维度:")
    print(f"  logits: {logits.shape}")
    print(f"  attention_weights: {attn_weights.shape}")
    
    assert logits.shape == (batch_size, num_classes)
    
    print("\n✅ 所有维度检查通过!")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  BERT参数: {total_params - trainable_params:,} (冻结)")


if __name__ == '__main__':
    test_model()
