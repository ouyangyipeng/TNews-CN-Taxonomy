#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化与分析脚本

功能：
1. 绘制训练曲线（Loss, Accuracy, Macro-F1）
2. 绘制混淆矩阵
3. 绘制 Attention Weights 热力图
4. 绘制 t-SNE 特征降维聚类图
5. Bad Case 分析

用法：
    python src/visualize.py --experiment bilstm_attention_char_seed42
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report

from src.tokenizers import get_tokenizer
from src.dataset import create_test_dataloader, load_jsonl, TNEWSDataset
from src.models.bilstm_attention import BiLSTMAttention
from src.models.transformer import TransformerClassifier


def plot_training_curves(log_file: Path, output_dir: Path):
    """
    绘制训练曲线
    
    Args:
        log_file: 训练日志 JSON 文件
        output_dir: 输出目录
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        log = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(log['train_loss']) + 1)
    
    # 1. Loss 曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, log['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, log['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy 曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, log['val_accuracy'], 'g-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Macro-F1 曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, log['val_macro_f1'], 'm-', label='Val Macro-F1', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Macro-F1', fontsize=11)
    ax3.set_title('Validation Macro-F1', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate 曲线
    ax4 = axes[1, 1]
    ax4.plot(epochs, log['learning_rate'], 'c-', label='Learning Rate', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Learning Rate', fontsize=11)
    ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存: {output_path}")
    plt.close()


def plot_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    output_dir: Path
):
    """
    绘制混淆矩阵
    
    Args:
        model: 模型
        dataloader: 测试数据加载器
        device: 设备
        id2label: ID 到标签名的映射
        output_dir: 输出目录
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 获取标签名
    label_names = [id2label[i] for i in range(len(id2label))]
    
    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存: {output_path}")
    plt.close()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))
    
    return all_preds, all_labels


def plot_attention_heatmap(
    model: nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    output_dir: Path,
    max_len: int = 128
):
    """
    绘制 Attention Weights 热力图
    
    Args:
        model: 模型
        tokenizer: 分词器
        text: 输入文本
        device: 设备
        output_dir: 输出目录
        max_len: 最大序列长度
    """
    model.eval()
    
    # 分词
    tokens = tokenizer.encode(text)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    
    # 转换为 ID 并手动 padding，确保长度一致
    input_ids = tokenizer.encode_to_ids(text, max_len=None)  # 不 padding
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    
    # 创建 attention mask（1 表示有效 token，0 表示 padding）
    actual_len = len(input_ids)
    attention_mask = [1] * actual_len + [0] * (max_len - actual_len)
    
    # Padding input_ids 到 max_len
    input_ids = input_ids + [tokenizer.pad_id] * (max_len - actual_len)
    
    # 转换为 tensor
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, attention_weights = model(input_ids_tensor, attention_mask_tensor)
    
    # 获取预测结果
    pred_id = torch.argmax(logits, dim=-1).item()
    
    # 处理 attention weights
    if isinstance(model, BiLSTMAttention):
        # BiLSTM: (batch, seq_len)
        attn_weights = attention_weights[0, :len(tokens)].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.heatmap(
            attn_weights.reshape(1, -1),
            annot=False,
            cmap='YlOrRd',
            xticklabels=tokens,
            yticklabels=['Attention'],
            ax=ax,
            cbar_kws={'label': 'Weight'}
        )
        ax.set_title(f'Attention Weights (Predicted: {pred_id})\nText: {text}', 
                     fontsize=11, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
    elif isinstance(model, TransformerClassifier):
        # Transformer: (batch, num_layers, num_heads, seq_len, seq_len)
        # 取最后一层、所有头的平均
        attn_weights = attention_weights[0, -1].mean(dim=0).cpu().numpy()
        attn_weights = attn_weights[:len(tokens), :len(tokens)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            attn_weights,
            annot=False,
            cmap='YlOrRd',
            xticklabels=tokens,
            yticklabels=tokens,
            ax=ax,
            cbar_kws={'label': 'Weight'}
        )
        ax.set_title(f'Self-Attention Weights (Last Layer, Avg Heads)\nText: {text}', 
                     fontsize=11, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / f'attention_heatmap_{text[:20]}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Attention 热力图已保存: {output_path}")
    plt.close()


def plot_tsne(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    output_dir: Path,
    max_samples: int = 1000
):
    """
    绘制 t-SNE 特征降维聚类图
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        id2label: ID 到标签名的映射
        output_dir: 输出目录
        max_samples: 最大样本数
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 获取特征（分类头之前的表示）
            if isinstance(model, BiLSTMAttention):
                # 获取 context + last_hidden 的拼接
                embedded = model.embedding(input_ids)
                lstm_out, (hidden, _) = model.lstm(embedded)
                last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
                context, _ = model.attention(last_hidden, lstm_out, lstm_out, attention_mask)
                features = torch.cat([context, last_hidden], dim=-1)
            elif isinstance(model, TransformerClassifier):
                # 获取 pooling 后的特征
                import math
                x = model.embedding(input_ids) * math.sqrt(model.d_model)
                x = model.positional_encoding(x)
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                for encoder_layer in model.encoder_layers:
                    x, _ = encoder_layer(x, mask)
                x = x * attention_mask.unsqueeze(-1)
                features = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if len(all_labels) >= max_samples:
                break
    
    # 拼接特征
    all_features = np.concatenate(all_features, axis=0)[:max_samples]
    all_labels = np.array(all_labels)[:max_samples]
    
    print(f"正在计算 t-SNE (样本数: {len(all_labels)})...")
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = np.unique(all_labels)
    colors = plt.cm.tab20(unique_labels / unique_labels.max())
    
    for label, color in zip(unique_labels, colors):
        mask = all_labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            label=id2label[label],
            alpha=0.6,
            s=30
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title('t-SNE Visualization of Sentence Features', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'tsne_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE 可视化已保存: {output_path}")
    plt.close()


def analyze_bad_cases(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    output_dir: Path,
    tokenizer,
    num_cases: int = 10
):
    """
    分析 Bad Cases，保存原始文本和预测详情。
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        id2label: ID 到标签名的映射
        output_dir: 输出目录
        tokenizer: 分词器（用于将 token ID 解码回文本）
        num_cases: 分析的 bad case 数量
    """
    model.eval()
    bad_cases = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            
            # 找出预测错误的样本
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    # 将 token ID 解码回原始文本
                    ids = input_ids[i].cpu().numpy()
                    mask = attention_mask[i].cpu().numpy()
                    # 只取有效 token（mask=1 的部分）
                    valid_ids = ids[mask == 1]
                    # 解码：id2token 映射
                    tokens = []
                    for tid in valid_ids:
                        tid_int = int(tid)
                        if tid_int in tokenizer.id2token:
                            tokens.append(tokenizer.id2token[tid_int])
                        else:
                            tokens.append(f'<{tid_int}>')
                    original_text = ''.join(tokens)
                    
                    bad_cases.append({
                        'text': original_text,
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'true_prob': probs[i, labels[i]].item(),
                        'pred_prob': probs[i, preds[i]].item()
                    })
            
            if len(bad_cases) >= num_cases:
                break
    
    # 保存 bad cases
    bad_cases = bad_cases[:num_cases]
    
    report = []
    report.append("# Bad Case 分析\n\n")
    
    for i, case in enumerate(bad_cases, 1):
        report.append(f"## Case {i}\n\n")
        report.append(f"- **原始文本**: {case['text']}\n")
        report.append(f"- **真实标签**: {id2label[case['true_label']]}\n")
        report.append(f"- **预测标签**: {id2label[case['pred_label']]}\n")
        report.append(f"- **真实标签概率**: {case['true_prob']:.4f}\n")
        report.append(f"- **预测标签概率**: {case['pred_prob']:.4f}\n\n")
    
    output_path = output_dir / 'bad_case_analysis.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"Bad Case 分析已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TNEWS 可视化与分析')
    parser.add_argument('--experiment', type=str, required=True,
                        help='实验名称')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print(f"TNEWS 可视化与分析")
    print("="*60)
    print(f"  实验: {args.experiment}")
    print(f"  设备: {device}")
    print()
    
    # 路径设置
    base_dir = Path(__file__).parent.parent
    checkpoint_dir = base_dir / 'checkpoints' / args.experiment
    log_dir = base_dir / 'logs' / args.experiment
    output_dir = base_dir / 'report' / 'figures' / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载实验配置
    with open(checkpoint_dir / 'experiment_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载标签映射
    with open(base_dir / 'data' / 'processed' / 'label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map['id2label'].items()}
    
    # 1. 绘制训练曲线
    print("[1/5] 绘制训练曲线...")
    plot_training_curves(log_dir / 'training_log.json', output_dir)
    
    # 加载分词器
    print("\n[2/5] 加载分词器和模型...")
    tokenizer = get_tokenizer(config['tokenizer'])
    tokenizer.load_vocab(checkpoint_dir / f"{config['tokenizer']}_vocab.json")
    
    # 创建测试 DataLoader
    test_loader = create_test_dataloader(
        base_dir / 'data' / 'processed' / 'dev_clean.json',
        tokenizer,
        max_len=config['max_len'],
        batch_size=config['batch_size']
    )
    
    # 加载模型
    vocab_size = len(tokenizer)
    num_classes = config['num_classes']
    
    if config['model'] == 'bilstm_attention':
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
    elif config['model'] == 'transformer':
        model = TransformerClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=512,
            num_heads=8,
            num_layers=4,
            d_ff=2048,
            max_len=config['max_len'],
            dropout=0.1,
            pad_token_id=tokenizer.pad_id
        )
    
    # 加载最佳模型权重
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"  模型加载完成 (Epoch {checkpoint['epoch']}, Macro-F1: {checkpoint['val_macro_f1']:.4f})")
    
    # 2. 绘制混淆矩阵
    print("\n[3/5] 绘制混淆矩阵...")
    plot_confusion_matrix(model, test_loader, device, id2label, output_dir)
    
    # 3. 绘制 Attention 热力图
    print("\n[4/5] 绘制 Attention 热力图...")
    sample_texts = [
        "北京今日天气多云转晴，适合外出游玩。",
        "华为发布新款手机，搭载最新芯片。",
        "股市今日大涨，科技股领涨。"
    ]
    for text in sample_texts:
        plot_attention_heatmap(model, tokenizer, text, device, output_dir)
    
    # 4. 绘制 t-SNE
    print("\n[5/5] 绘制 t-SNE 可视化...")
    plot_tsne(model, test_loader, device, id2label, output_dir, max_samples=1000)
    
    # 5. Bad Case 分析
    print("\n[6/6] Bad Case 分析...")
    analyze_bad_cases(model, test_loader, device, id2label, output_dir, tokenizer, num_cases=10)
    
    print("\n" + "="*60)
    print("可视化与分析完成!")
    print("="*60)


if __name__ == '__main__':
    main()
