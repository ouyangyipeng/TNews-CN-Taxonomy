#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-7B Feature Extractor + Custom Classifier

架构：
1. Qwen2.5-7B (frozen or finetune): 提取上下文嵌入
2. Pooling: 对Qwen输出进行池化
3. Custom Classifier: 自定义分类头

使用更大的模型（7B参数）来提升性能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class QwenClassifier(nn.Module):
    """
    Qwen2.5-7B Feature Extractor + Custom Classification Head
    
    架构流程：
    1. Qwen2.5-7B: (batch, seq_len) -> (batch, seq_len, 3584)
    2. Pooling: (batch, seq_len, 3584) -> (batch, 3584)
    3. Classifier: (batch, 3584) -> (batch, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int,
        model_path: str = '/workspace/NS-2026-03/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28',
        pooling_strategy: str = 'mean',  # 'last', 'mean', 'max'
        hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_model: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16
    ):
        """
        Args:
            num_classes: 分类类别数
            model_path: Qwen模型路径
            pooling_strategy: 池化策略 ('last', 'mean', 'max')
            hidden_dim: 分类头隐藏层维度
            dropout: Dropout概率
            freeze_model: 是否冻结模型参数
            use_lora: 是否使用LoRA
            lora_r: LoRA秩
            lora_alpha: LoRA alpha
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        self.model_dim = 3584  # Qwen2.5-7B hidden size
        
        # 1. Qwen2.5-7B Feature Extractor
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
        except Exception as e:
            raise RuntimeError(f"无法加载Qwen模型: {e}")
        
        # 冻结模型参数
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # LoRA适配
        if use_lora and not freeze_model:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )
                self.model = get_peft_model(self.model, lora_config)
                print(f"  LoRA配置: r={lora_r}, alpha={lora_alpha}")
            except ImportError:
                print("  警告: peft未安装，跳过LoRA配置")
        
        # 2. Custom Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
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
        对Qwen输出进行池化
        
        Args:
            last_hidden_state: (batch, seq_len, 3584)
            attention_mask: (batch, seq_len)
        
        Returns:
            pooled: (batch, 3584)
        """
        if self.pooling_strategy == 'last':
            # 使用最后一个非padding token的输出
            # 找到每个序列的最后一个有效位置
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            return last_hidden_state[batch_indices, seq_lengths]
        
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
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            logits: (batch, num_classes)
            attention_mask: (batch, seq_len) - 返回attention_mask作为占位符
        """
        # Qwen特征提取
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 获取最后一层隐藏状态
        last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, 3584)
        
        # 池化
        pooled = self._pool_output(last_hidden_state, attention_mask)  # (batch, 3584)
        
        # 分类
        logits = self.classifier(pooled.float())  # (batch, num_classes)
        
        # 返回attention_mask作为attention_weights的占位符（用于兼容接口）
        return logits, attention_mask


def test_model():
    """测试模型"""
    print("="*60)
    print("Qwen Classifier 模型测试")
    print("="*60)
    
    # 超参数
    batch_size = 2
    seq_len = 20
    num_classes = 15
    
    # 创建模型
    model = QwenClassifier(
        num_classes=num_classes,
        pooling_strategy='mean',
        hidden_dim=512,
        dropout=0.2,
        freeze_model=True
    )
    
    # 创建输入
    input_ids = torch.randint(0, 152064, (batch_size, seq_len))
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
    print(f"  冻结参数: {total_params - trainable_params:,}")


if __name__ == '__main__':
    test_model()
