#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成学习预测脚本

支持多种集成策略：
1. 多数投票（Majority Voting）
2. 加权投票（Weighted Voting）
3. 平均概率（Average Probability）

用法：
    python src/ensemble_predict.py --checkpoints checkpoint1 checkpoint2 checkpoint3 --test_file data/processed/test_clean.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_tokenizers import get_tokenizer
from src.dataset import create_test_dataloader, load_jsonl
from src.models.bilstm_attention import BiLSTMAttention
from src.models.transformer import TransformerClassifier
from src.models.bert_classifier import BERTClassifier


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    加载模型和配置
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        model: 加载的模型
        config: 实验配置
    """
    # 加载配置
    config_path = checkpoint_path.parent / 'experiment_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model_type = config['model']
    num_classes = config['num_classes']
    
    # 创建模型
    if model_type == 'bilstm_attention':
        vocab_size = config.get('vocab_size', 5000)
        model = BiLSTMAttention(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=300,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            attention_dim=128,
            pad_token_id=0
        )
    elif model_type == 'transformer':
        vocab_size = config.get('vocab_size', 5000)
        model = TransformerClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=512,
            num_heads=8,
            num_layers=4,
            d_ff=2048,
            max_len=config.get('max_len', 128),
            dropout=0.1,
            pad_token_id=0
        )
    elif model_type == 'bert_classifier':
        # 从配置中读取本地模型路径
        local_model_path = config.get('local_model_path', '/workspace/models/bert-base-chinese')
        print(f"  加载 BERT 模型: {local_model_path}")
        model = BERTClassifier(
            num_classes=num_classes,
            pooling_strategy=config.get('pooling_strategy', 'cls'),
            hidden_dim=256,
            dropout=0.3,
            freeze_bert=False,
            local_model_path=local_model_path
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, config


def predict_batch(model: nn.Module, batch: Dict, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    对一批数据进行预测
    
    Args:
        model: 模型
        batch: 数据批次
        device: 设备
    
    Returns:
        predictions: 预测标签
        probabilities: 预测概率
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    
    return preds.cpu().numpy(), probs.cpu().numpy()


def ensemble_predict(
    models: List[nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    strategy: str = 'majority'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    集成预测
    
    Args:
        models: 模型列表
        test_loader: 测试数据加载器
        device: 设备
        strategy: 集成策略 ('majority', 'weighted', 'average')
    
    Returns:
        predictions: 集成预测标签
        probabilities: 集成预测概率
    """
    all_preds = []
    all_probs = []
    
    for batch in test_loader:
        batch_preds = []
        batch_probs = []
        
        for model in models:
            preds, probs = predict_batch(model, batch, device)
            batch_preds.append(preds)
            batch_probs.append(probs)
        
        # 转换为数组
        batch_preds = np.array(batch_preds)  # (num_models, batch_size)
        batch_probs = np.array(batch_probs)  # (num_models, batch_size, num_classes)
        
        if strategy == 'majority':
            # 多数投票
            ensemble_preds = []
            for i in range(batch_preds.shape[1]):
                votes = batch_preds[:, i]
                counter = Counter(votes)
                ensemble_preds.append(counter.most_common(1)[0][0])
            ensemble_preds = np.array(ensemble_preds)
            ensemble_probs = np.mean(batch_probs, axis=0)
        
        elif strategy == 'average':
            # 平均概率
            ensemble_probs = np.mean(batch_probs, axis=0)
            ensemble_preds = np.argmax(ensemble_probs, axis=-1)
        
        elif strategy == 'weighted':
            # 加权投票（假设模型权重相等）
            weights = np.ones(len(models)) / len(models)
            weighted_probs = np.average(batch_probs, axis=0, weights=weights)
            ensemble_probs = weighted_probs
            ensemble_preds = np.argmax(ensemble_probs, axis=-1)
        
        else:
            raise ValueError(f"未知的集成策略: {strategy}")
        
        all_preds.append(ensemble_preds)
        all_probs.append(ensemble_probs)
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    return all_preds, all_probs


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='集成学习预测脚本')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='检查点路径列表')
    parser.add_argument('--test_file', type=str, default='data/processed/test_clean.json',
                        help='测试数据文件')
    parser.add_argument('--strategy', type=str, default='majority',
                        choices=['majority', 'weighted', 'average'],
                        help='集成策略')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (auto/cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--max_len', type=int, default=128,
                        help='最大序列长度')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("集成学习预测")
    print("="*60)
    print(f"  检查点数量: {len(args.checkpoints)}")
    print(f"  集成策略: {args.strategy}")
    print(f"  设备: {device}")
    print()
    
    # 加载模型
    print("[1/3] 加载模型...")
    models = []
    configs = []
    for checkpoint_path in args.checkpoints:
        checkpoint_path = Path(checkpoint_path)
        print(f"  加载: {checkpoint_path}")
        model, config = load_model(checkpoint_path, device)
        models.append(model)
        configs.append(config)
    
    # 加载测试数据
    print("\n[2/3] 加载测试数据...")
    test_file = Path(args.test_file)
    
    # 尝试加载测试数据，处理没有标签的情况
    test_texts = []
    test_labels = None
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            test_texts.append(data['sentence'])
            if 'label_id' in data and test_labels is not None:
                test_labels.append(data['label_id'])
            elif 'label_id' in data and test_labels is None:
                test_labels = [data['label_id']]
    
    if test_labels is None:
        test_labels = []
    
    print(f"  测试样本数: {len(test_texts)}")
    print(f"  是否有标签: {len(test_labels) > 0}")
    
    # 创建分词器（使用第一个模型的配置）
    tokenizer_type = configs[0]['tokenizer']
    
    # 对于subword分词器，需要传递pretrained_model_name
    if tokenizer_type == 'subword':
        local_model_path = configs[0].get('local_model_path', '/workspace/models/bert-base-chinese')
        tokenizer = get_tokenizer(tokenizer_type, pretrained_model_name=local_model_path)
    else:
        tokenizer = get_tokenizer(tokenizer_type)
        vocab_path = Path(args.checkpoints[0]).parent / f'{tokenizer_type}_vocab.json'
        tokenizer.load_vocab(vocab_path)
    
    # 创建 DataLoader
    test_loader, _ = create_test_dataloader(
        test_file,
        tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size
    )
    
    # 集成预测
    print("\n[3/3] 集成预测...")
    predictions, probabilities = ensemble_predict(
        models, test_loader, device, args.strategy
    )
    
    # 计算指标（如果有标签）
    if len(test_labels) > 0 and all(l != -1 for l in test_labels):
        accuracy = accuracy_score(test_labels, predictions)
        macro_f1 = f1_score(test_labels, predictions, average='macro')
        
        print(f"\n集成预测结果:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro-F1: {macro_f1:.4f}")
        
        # 保存结果
        output_path = Path('ensemble_results.json')
        results = {
            'checkpoints': args.checkpoints,
            'strategy': args.strategy,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'predictions': predictions.tolist(),
            'true_labels': test_labels
        }
    else:
        print(f"\n集成预测结果:")
        print(f"  测试集没有标签，无法计算指标")
        print(f"  预测样本数: {len(predictions)}")
        
        # 保存结果
        output_path = Path('ensemble_results.json')
        results = {
            'checkpoints': args.checkpoints,
            'strategy': args.strategy,
            'predictions': predictions.tolist()
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
