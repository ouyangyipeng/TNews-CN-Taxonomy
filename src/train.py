#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本

功能：
1. 加载数据和模型
2. 训练循环（含 Early Stopping）
3. 验证集评估（Accuracy + Macro-F1）
4. 保存最佳模型和训练日志
5. 支持多种分词器和模型类型的组合

用法：
    python src/train.py --model bilstm_attention --tokenizer char --epochs 50
    python src/train.py --model transformer --tokenizer word --epochs 30
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

from src.tokenizers import get_tokenizer
from src.dataset import create_dataloaders, create_test_dataloader, load_jsonl
from src.models.bilstm_attention import BiLSTMAttention
from src.models.transformer import TransformerClassifier


class EarlyStopping:
    """
    Early Stopping 机制
    
    当验证集指标在 patience 轮内没有提升时，停止训练
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: 容忍的轮数
            min_delta: 最小提升幅度
            mode: 'max' 表示指标越大越好，'min' 表示越小越好
        """
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
        """
        检查是否应该停止训练
        
        Args:
            score: 当前验证集指标
        
        Returns:
            True 表示应该停止训练
        """
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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, Dict]:
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        loss: 平均损失
        accuracy: 准确率
        macro_f1: Macro-F1 分数
        report: 分类报告字典
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
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
    gradient_clip: float = 1.0
) -> float:
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        gradient_clip: 梯度裁剪阈值
    
    Returns:
        avg_loss: 平均训练损失
    """
    model.train()
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
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
    log_file: Path = None
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
    
    Returns:
        training_log: 训练日志字典
    """
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器：使用 CosineAnnealingWarmRestarts 配合 warmup
    # 对于 Transformer，warmup 非常重要
    warmup_epochs = 5
    total_epochs = epochs
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
    print()
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, device, gradient_clip
        )
        
        # 验证
        val_loss, val_accuracy, val_macro_f1, _ = evaluate(
            model, val_loader, device
        )
        
        # 学习率调度
        scheduler.step()
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
    parser = argparse.ArgumentParser(description='TNEWS 文本分类训练脚本')
    parser.add_argument('--model', type=str, default='bilstm_attention',
                        choices=['bilstm_attention', 'transformer'],
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
        args.experiment_name = f"{args.model}_{args.tokenizer}_seed{args.seed}"
    
    print("="*60)
    print(f"TNEWS 文本分类训练")
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
    print("[1/4] 加载分词器...")
    tokenizer = get_tokenizer(args.tokenizer)
    
    # 加载训练数据构建词表
    train_texts, _ = load_jsonl(data_dir / 'train_clean.json')
    tokenizer.build_vocab(train_texts, min_freq=1)
    
    # 保存分词器词表
    vocab_path = checkpoint_dir / f'{args.tokenizer}_vocab.json'
    tokenizer.save_vocab(vocab_path)
    print(f"  词表已保存: {vocab_path}")
    
    # 创建 DataLoader
    print("\n[2/4] 创建 DataLoader...")
    train_loader, val_loader, split_info = create_dataloaders(
        data_dir / 'train_clean.json',
        tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        val_ratio=0.1,
        random_seed=args.seed
    )
    
    # 创建模型
    print("\n[3/4] 创建模型...")
    vocab_size = len(tokenizer)
    num_classes = 15
    
    if args.model == 'bilstm_attention':
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
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 训练
    print("\n[4/4] 开始训练...")
    training_log = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        log_file=log_dir / 'training_log.json'
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
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'split_info': split_info,
        'best_epoch': training_log['best_epoch'],
        'best_macro_f1': training_log['best_macro_f1']
    }
    
    with open(checkpoint_dir / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)
    
    print(f"\n实验配置已保存: {checkpoint_dir / 'experiment_config.json'}")


if __name__ == '__main__':
    main()
