#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版训练脚本

优化策略：
1. BERT冻结特征提取器
2. 类别权重平衡
3. 标签平滑
4. 优化学习率调度（warmup + cosine annealing）
5. 数据增强（可选）

用法：
    python src/train_enhanced.py --model bert_classifier --epochs 20 --lr 2e-4
    python src/train_enhanced.py --model bilstm_attention --use_class_weights --label_smoothing 0.1
"""

import argparse
import json
import sys
import time
import math
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import Counter

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

from src.text_tokenizers import get_tokenizer
from src.dataset import create_dataloaders, create_test_dataloader, load_jsonl
from src.models.bilstm_attention import BiLSTMAttention
from src.models.transformer import TransformerClassifier
from src.models.bert_classifier import BERTClassifier
from src.models.qwen_classifier import QwenClassifier


class EarlyStopping:
    """Early Stopping 机制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        elif mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            raise ValueError(f"mode 必须是 'max' 或 'min'，当前为 {mode}")
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def compute_class_weights(labels: list, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    计算类别权重（逆频率加权）
    
    Args:
        labels: 标签列表
        num_classes: 类别数
        device: 设备
    
    Returns:
        weights: (num_classes,) 类别权重张量
    """
    counter = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)  # 避免除零
        weight = total / (num_classes * count)
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # 归一化，使平均权重为1
    weights = weights / weights.mean()
    
    return weights


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Tuple[float, float, float, Dict]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * input_ids.size(0)
            
            # 预测
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 分类报告
    report = classification_report(
        all_labels, all_preds,
        output_dict=True,
        zero_division=0
    )
    
    return avg_loss, accuracy, macro_f1, report


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    gradient_clip: float = 1.0
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item() * input_ids.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5
):
    """
    带warmup的余弦退火学习率调度器
    
    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        num_training_steps: 总训练步数
        num_cycles: 余弦周期数
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    gradient_clip: float = 1.0,
    early_stopping_patience: int = 5,
    checkpoint_dir: Path = None,
    log_file: Path = None,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    warmup_ratio: float = 0.1
) -> Dict:
    """
    完整训练流程
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        epochs: 最大训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        gradient_clip: 梯度裁剪阈值
        early_stopping_patience: Early Stopping 容忍轮数
        checkpoint_dir: 模型保存目录
        log_file: 日志文件路径
        class_weights: 类别权重
        label_smoothing: 标签平滑系数
        warmup_ratio: warmup比例（占总epoch的比例）
    
    Returns:
        training_log: 训练日志字典
    """
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
    
    # 学习率调度器
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
    
    # 训练日志
    training_log = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'learning_rate': [],
        'best_epoch': 0,
        'best_macro_f1': 0.0
    }
    
    print(f"\n开始训练 (设备: {device})")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Gradient Clip: {gradient_clip}")
    print(f"  Early Stopping Patience: {early_stopping_patience}")
    print(f"  Label Smoothing: {label_smoothing}")
    print(f"  Class Weights: {'Enabled' if class_weights is not None else 'Disabled'}")
    print(f"  Warmup Steps: {num_warmup_steps}")
    print()
    
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, device, criterion, gradient_clip
        )
        
        # 更新学习率（按step）
        for _ in range(len(train_loader)):
            scheduler.step()
            global_step += 1
        
        # 验证
        val_loss, val_accuracy, val_macro_f1, _ = evaluate(
            model, val_loader, device, criterion
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录日志
        training_log['train_loss'].append(train_loss)
        training_log['val_loss'].append(val_loss)
        training_log['val_accuracy'].append(val_accuracy)
        training_log['val_macro_f1'].append(val_macro_f1)
        training_log['learning_rate'].append(current_lr)
        
        # 打印进度
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"Val Macro-F1: {val_macro_f1:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.2f}s")
        
        # 保存最佳模型
        if val_macro_f1 > training_log['best_macro_f1']:
            training_log['best_macro_f1'] = val_macro_f1
            training_log['best_epoch'] = epoch
            
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_macro_f1': val_macro_f1,
                    'val_accuracy': val_accuracy
                }, checkpoint_path)
                print(f"  ✓ 保存最佳模型 (Macro-F1: {val_macro_f1:.4f})")
        
        # Early Stopping 检查
        if early_stopping(val_macro_f1):
            print(f"\n⚠ Early Stopping at epoch {epoch}")
            break
    
    # 保存训练日志
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2)
        print(f"\n训练日志已保存: {log_file}")
    
    print(f"\n训练完成!")
    print(f"  最佳 Epoch: {training_log['best_epoch']}")
    print(f"  最佳 Macro-F1: {training_log['best_macro_f1']:.4f}")
    
    return training_log


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TNEWS 文本分类增强训练脚本')
    parser.add_argument('--model', type=str, default='bilstm_attention',
                        choices=['bilstm_attention', 'transformer', 'bert_classifier', 'qwen_classifier'],
                        help='模型类型')
    parser.add_argument('--tokenizer', type=str, default='char',
                        choices=['char', 'word', 'subword'],
                        help='分词器类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--max_len', type=int, default=128, help='最大序列长度')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='实验名称（用于保存文件）')
    
    # 增强选项
    parser.add_argument('--use_class_weights', action='store_true',
                        help='使用类别权重平衡')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='标签平滑系数 (0.0-0.2)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup比例 (0.0-0.3)')
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese',
                        help='BERT模型名称')
    parser.add_argument('--local_model_path', type=str, default=None,
                        help='本地BERT模型路径（用于离线环境）')
    parser.add_argument('--train_file', type=str, default='train_clean.json',
                        help='训练数据文件名（默认 train_clean.json）')
    parser.add_argument('--pooling_strategy', type=str, default='cls',
                        choices=['cls', 'mean', 'max', 'last'],
                        help='池化策略')
    parser.add_argument('--freeze_bert', action='store_true', default=False,
                        help='冻结BERT参数（默认解冻，进行全量微调）')
    
    # Qwen 模型参数
    parser.add_argument('--qwen_model_path', type=str,
                        default='/workspace/NS-2026-03/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28',
                        help='Qwen模型路径')
    parser.add_argument('--freeze_qwen', action='store_true', default=False,
                        help='冻结Qwen参数（默认解冻，进行全量微调）')
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='使用LoRA进行参数高效微调')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha参数')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 实验名称
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.tokenizer}_enhanced_seed{args.seed}"
    
    print("="*60)
    print(f"TNEWS 文本分类增强训练")
    print("="*60)
    print(f"  模型: {args.model}")
    print(f"  分词器: {args.tokenizer}")
    print(f"  设备: {device}")
    print(f"  实验名称: {args.experiment_name}")
    print()
    
    # 路径设置
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    checkpoint_dir = base_dir / 'checkpoints' / args.experiment_name
    log_dir = base_dir / 'logs' / args.experiment_name
    
    # 加载分词器
    print("[1/5] 加载分词器...")
    tokenizer_kwargs = {}
    if args.tokenizer == 'subword' and args.local_model_path:
        tokenizer_kwargs['pretrained_model_name'] = args.local_model_path
    tokenizer = get_tokenizer(args.tokenizer, **tokenizer_kwargs)
    
    # 加载训练数据构建词表
    train_file_path = data_dir / args.train_file
    print(f"  加载训练数据: {train_file_path}")
    train_texts, train_labels = load_jsonl(train_file_path)
    
    if args.model != 'bert_classifier':
        tokenizer.build_vocab(train_texts, min_freq=1)
        
        # 保存分词器词表
        vocab_path = checkpoint_dir / f'{args.tokenizer}_vocab.json'
        tokenizer.save_vocab(vocab_path)
        print(f"  词表已保存: {vocab_path}")
    
    # 创建 DataLoader
    print("\n[2/5] 创建 DataLoader...")
    train_loader, val_loader, split_info = create_dataloaders(
        data_dir / 'train_clean.json',
        tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        val_ratio=0.1,
        random_seed=args.seed
    )
    
    # 计算类别权重
    class_weights = None
    if args.use_class_weights:
        print("\n[3/5] 计算类别权重...")
        class_weights = compute_class_weights(train_labels, 15, device)
        print(f"  类别权重: {class_weights.cpu().numpy()}")
    else:
        print("\n[3/5] 跳过类别权重计算")
    
    # 创建模型
    print("\n[4/5] 创建模型...")
    num_classes = 15
    
    if args.model == 'bilstm_attention':
        vocab_size = len(tokenizer)
        model = BiLSTMAttention(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=300,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            attention_dim=128,
            pad_token_id=tokenizer.pad_id
        )
    elif args.model == 'transformer':
        vocab_size = len(tokenizer)
        model = TransformerClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=512,
            num_heads=8,
            num_layers=4,
            d_ff=2048,
            max_len=args.max_len,
            dropout=0.1,
            pad_token_id=tokenizer.pad_id
        )
    elif args.model == 'bert_classifier':
        model = BERTClassifier(
            num_classes=num_classes,
            bert_model_name=args.bert_model,
            pooling_strategy=args.pooling_strategy,
            hidden_dim=256,
            dropout=0.3,
            freeze_bert=args.freeze_bert,
            local_model_path=args.local_model_path
        )
    elif args.model == 'qwen_classifier':
        model = QwenClassifier(
            num_classes=num_classes,
            model_path=args.qwen_model_path,
            pooling_strategy=args.pooling_strategy,
            hidden_dim=512,
            dropout=0.2,
            freeze_model=args.freeze_qwen,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    if args.model == 'bert_classifier':
        print(f"  BERT参数: {total_params - trainable_params:,} (冻结)")
    
    # 训练
    print("\n[5/5] 开始训练...")
    training_log = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        log_file=log_dir / 'training_log.json',
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        warmup_ratio=args.warmup_ratio
    )
    
    # 保存实验配置
    experiment_config = {
        'model': args.model,
        'tokenizer': args.tokenizer,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'max_len': args.max_len,
        'device': str(device),
        'seed': args.seed,
        'num_classes': num_classes,
        'split_info': split_info,
        'best_epoch': training_log['best_epoch'],
        'best_macro_f1': training_log['best_macro_f1'],
        'use_class_weights': args.use_class_weights,
        'label_smoothing': args.label_smoothing,
        'warmup_ratio': args.warmup_ratio
    }
    
    # 保存 BERT 相关配置
    if args.model == 'bert_classifier':
        experiment_config['local_model_path'] = args.local_model_path
        experiment_config['pooling_strategy'] = args.pooling_strategy
    else:
        experiment_config['vocab_size'] = len(tokenizer)
    
    with open(checkpoint_dir / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)
    
    print(f"\n实验配置已保存: {checkpoint_dir / 'experiment_config.json'}")


if __name__ == '__main__':
    main()
