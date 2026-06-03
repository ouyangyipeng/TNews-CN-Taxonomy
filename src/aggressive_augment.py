#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激进数据增强脚本

支持多种增强策略：
1. 回译（Back Translation）- 需要网络
2. 同义词替换（Synonym Replacement）
3. 随机删除（Random Deletion）
4. 随机交换（Random Swap）
5. 随机插入（Random Insertion）

用法：
    python src/aggressive_augment.py --input data/processed/train_clean.json --output data/processed/train_aggressive_augmented.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict

import jieba


def synonym_replacement(text: str, synonyms: Dict[str, List[str]], n: int = 2) -> str:
    """
    同义词替换
    
    Args:
        text: 原始文本
        synonyms: 同义词字典
        n: 替换次数
    
    Returns:
        增强后的文本
    """
    words = list(jieba.cut(text))
    new_words = words.copy()
    
    # 找到可以替换的词
    replaceable = [(i, w) for i, w in enumerate(words) if w in synonyms]
    
    if not replaceable:
        return text
    
    # 随机选择n个词进行替换
    n = min(n, len(replaceable))
    selected = random.sample(replaceable, n)
    
    for i, word in selected:
        syns = synonyms[word]
        if syns:
            new_words[i] = random.choice(syns)
    
    return ''.join(new_words)


def random_deletion(text: str, p: float = 0.2) -> str:
    """
    随机删除词
    
    Args:
        text: 原始文本
        p: 删除概率
    
    Returns:
        增强后的文本
    """
    words = list(jieba.cut(text))
    
    if len(words) <= 2:
        return text
    
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    # 如果删除后为空，返回原文本
    if not new_words:
        return text
    
    return ''.join(new_words)


def random_swap(text: str, n: int = 2) -> str:
    """
    随机交换词的位置
    
    Args:
        text: 原始文本
        n: 交换次数
    
    Returns:
        增强后的文本
    """
    words = list(jieba.cut(text))
    
    if len(words) <= 2:
        return text
    
    new_words = words.copy()
    
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
    return ''.join(new_words)


def random_insertion(text: str, n: int = 2) -> str:
    """
    随机插入同义词
    
    Args:
        text: 原始文本
        n: 插入次数
    
    Returns:
        增强后的文本
    """
    words = list(jieba.cut(text))
    
    # 简单的插入：复制一个词并插入到随机位置
    for _ in range(n):
        if words:
            word = random.choice(words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, word)
    
    return ''.join(words)


def load_synonyms() -> Dict[str, List[str]]:
    """
    加载同义词字典（简化版）
    
    Returns:
        同义词字典
    """
    # 简化的同义词字典（实际应用中应该使用更完整的资源）
    synonyms = {
        '好': ['优秀', '出色', '良好'],
        '大': ['巨大', '庞大', '宏大'],
        '小': ['微小', '细小', '渺小'],
        '多': ['许多', '众多', '大量'],
        '少': ['稀少', '少量', '不多'],
        '快': ['迅速', '快速', '飞速'],
        '慢': ['缓慢', '迟缓', '慢慢'],
        '新': ['崭新', '全新', '新颖'],
        '旧': ['陈旧', '老旧', '过时'],
        '高': ['高大', '高耸', '崇高'],
        '低': ['低矮', '低下', '卑微'],
        '长': ['漫长', '长久', '长远'],
        '短': ['短暂', '短小', '简短'],
        '美': ['美丽', '美好', '优美'],
        '丑': ['丑陋', '丑恶', '难看'],
    }
    return synonyms


def augment_sample(
    sample: Dict,
    synonyms: Dict[str, List[str]],
    strategies: List[str] = ['synonym', 'deletion', 'swap', 'insertion']
) -> List[Dict]:
    """
    对单个样本进行多种增强
    
    Args:
        sample: 原始样本
        synonyms: 同义词字典
        strategies: 增强策略列表
    
    Returns:
        增强后的样本列表
    """
    text = sample['sentence']
    augmented_samples = []
    
    for strategy in strategies:
        if strategy == 'synonym':
            new_text = synonym_replacement(text, synonyms, n=2)
        elif strategy == 'deletion':
            new_text = random_deletion(text, p=0.2)
        elif strategy == 'swap':
            new_text = random_swap(text, n=2)
        elif strategy == 'insertion':
            new_text = random_insertion(text, n=2)
        else:
            continue
        
        # 如果增强后的文本与原文本不同，添加到结果中
        if new_text != text and len(new_text) > 0:
            new_sample = sample.copy()
            new_sample['sentence'] = new_text
            augmented_samples.append(new_sample)
    
    return augmented_samples


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='激进数据增强脚本')
    parser.add_argument('--input', type=str, required=True,
                        help='输入数据文件')
    parser.add_argument('--output', type=str, required=True,
                        help='输出数据文件')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['synonym', 'deletion', 'swap', 'insertion'],
                        help='增强策略列表')
    parser.add_argument('--max_augment_per_sample', type=int, default=3,
                        help='每个样本最多增强次数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    print("="*60)
    print("激进数据增强")
    print("="*60)
    print(f"  输入文件: {args.input}")
    print(f"  输出文件: {args.output}")
    print(f"  增强策略: {args.strategies}")
    print(f"  每样本最大增强次数: {args.max_augment_per_sample}")
    print()
    
    # 加载数据
    print("[1/4] 加载数据...")
    input_path = Path(args.input)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"  原始样本数: {len(data)}")
    
    # 加载同义词字典
    print("\n[2/4] 加载同义词字典...")
    synonyms = load_synonyms()
    print(f"  同义词数量: {len(synonyms)}")
    
    # 数据增强
    print("\n[3/4] 数据增强...")
    augmented_data = data.copy()
    
    for i, sample in enumerate(data):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{len(data)}")
        
        # 对每个样本进行多次增强
        for _ in range(args.max_augment_per_sample):
            new_samples = augment_sample(sample, synonyms, args.strategies)
            augmented_data.extend(new_samples)
    
    print(f"  增强后样本数: {len(augmented_data)}")
    
    # 保存数据
    print("\n[4/4] 保存数据...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in augmented_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  已保存: {output_path}")
    print(f"  增强比例: {len(augmented_data) / len(data):.2f}x")


if __name__ == '__main__':
    main()
