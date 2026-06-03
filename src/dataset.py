#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset 和 DataLoader 模块

功能：
1. 加载清洗后的 JSONL 数据
2. 使用分词器将文本转换为 ID 序列
3. 实现动态 padding 的 collate_fn
4. 支持训练集/验证集划分（分层抽样）

核心类：
- TNEWSDataset: PyTorch Dataset 实现
- create_dataloaders: 创建训练/验证 DataLoader 的工厂函数
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.text_tokenizers import BaseTokenizer


class TNEWSDataset(Dataset):
    """
    TNEWS 数据集 PyTorch Dataset
    
    每个样本返回：
    - input_ids: List[int], token ID 序列（已 padding 到 max_len）
    - attention_mask: List[int], 1 表示有效 token，0 表示 padding
    - label: int, 类别标签
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BaseTokenizer,
        max_len: int,
        add_special_tokens: bool = False
    ):
        """
        Args:
            texts: 文本列表
            labels: 标签列表（整数）
            tokenizer: 分词器实例
            max_len: 最大序列长度（超过截断，不足 padding）
            add_special_tokens: 是否添加特殊 token（如 [CLS], [SEP]）
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_special_tokens = add_special_tokens
        
        # 预计算所有样本的 ID（加速训练）
        self.encoded_data = self._encode_all()
    
    def _encode_all(self) -> List[Tuple[List[int], List[int]]]:
        """预编码所有样本"""
        encoded = []
        for text in self.texts:
            # 分词并转换为 ID
            ids = self.tokenizer.encode_to_ids(text, max_len=None)
            
            # 截断
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            
            # 构建 attention mask
            attention_mask = [1] * len(ids)
            
            # Padding
            padding_length = self.max_len - len(ids)
            ids = ids + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
            encoded.append((ids, attention_mask))
        
        return encoded
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids, attention_mask = self.encoded_data[idx]
        label = self.labels[idx]
        
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_jsonl(file_path: Path) -> Tuple[List[str], List[int]]:
    """
    加载 JSONL 文件
    
    Returns:
        texts: 文本列表
        labels: 标签列表
    """
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                texts.append(item['sentence'])
                labels.append(item['label_id'])
    return texts, labels


def create_dataloaders(
    train_file: Path,
    tokenizer: BaseTokenizer,
    max_len: int,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证 DataLoader
    
    Args:
        train_file: 清洗后的训练集 JSONL 文件
        tokenizer: 分词器实例
        max_len: 最大序列长度
        batch_size: 批大小
        val_ratio: 验证集比例
        random_seed: 随机种子
        num_workers: DataLoader 工作进程数
    
    Returns:
        train_loader: 训练 DataLoader
        val_loader: 验证 DataLoader
        split_info: 划分信息字典
    """
    # 加载数据
    texts, labels = load_jsonl(train_file)
    print(f"  加载训练数据: {len(texts)} 条")
    
    # 分层抽样划分（保持类别分布一致）
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=labels  # 分层抽样
    )
    
    print(f"  训练集: {len(train_texts)} 条")
    print(f"  验证集: {len(val_texts)} 条")
    
    # 创建 Dataset
    train_dataset = TNEWSDataset(
        train_texts, train_labels, tokenizer, max_len
    )
    val_dataset = TNEWSDataset(
        val_texts, val_labels, tokenizer, max_len
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 划分信息
    split_info = {
        'total_samples': len(texts),
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'val_ratio': val_ratio,
        'random_seed': random_seed,
        'max_len': max_len,
        'batch_size': batch_size,
        'vocab_size': len(tokenizer)
    }
    
    return train_loader, val_loader, split_info


def create_test_dataloader(
    test_file: Path,
    tokenizer: BaseTokenizer,
    max_len: int,
    batch_size: int = 32,
    num_workers: int = 0
) -> tuple[DataLoader, list]:
    """
    创建测试集 DataLoader（dev.json 作为最终测试集）
    
    Args:
        test_file: 清洗后的测试集 JSONL 文件
        tokenizer: 分词器实例
        max_len: 最大序列长度
        batch_size: 批大小
        num_workers: DataLoader 工作进程数
    
    Returns:
        test_loader: 测试 DataLoader
        labels: 标签列表（如果没有标签则为空列表）
    """
    # 尝试加载测试数据，处理没有标签的情况
    texts = []
    labels = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['sentence'])
            if 'label_id' in data:
                labels.append(data['label_id'])
    
    print(f"  加载测试数据: {len(texts)} 条")
    print(f"  是否有标签: {len(labels) > 0}")
    
    # 如果没有标签，使用虚拟标签（-1）
    if len(labels) == 0:
        labels = [-1] * len(texts)
    
    test_dataset = TNEWSDataset(
        texts, labels, tokenizer, max_len
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return test_loader, labels


def demo():
    """演示 Dataset 和 DataLoader 的使用"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.tokenizers import get_tokenizer
    
    base_dir = Path(__file__).parent.parent
    train_file = base_dir / 'data' / 'processed' / 'train_clean.json'
    
    if not train_file.exists():
        print(f"错误: 请先运行数据清洗脚本生成 {train_file}")
        return
    
    # 使用字符级分词器
    tokenizer = get_tokenizer('char')
    
    # 加载数据构建词表
    texts, _ = load_jsonl(train_file)
    tokenizer.build_vocab(texts, min_freq=2)
    
    # 创建 DataLoader
    train_loader, val_loader, split_info = create_dataloaders(
        train_file, tokenizer,
        max_len=50, batch_size=32, val_ratio=0.1
    )
    
    print(f"\n划分信息: {split_info}")
    
    # 测试迭代
    print("\n测试 DataLoader 迭代:")
    for batch in train_loader:
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  label shape: {batch['label'].shape}")
        print(f"  input_ids[0][:10]: {batch['input_ids'][0][:10].tolist()}")
        print(f"  label[0]: {batch['label'][0].item()}")
        break


if __name__ == '__main__':
    demo()
