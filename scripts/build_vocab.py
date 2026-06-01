#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建词表脚本

为三种分词器构建词表（仅使用训练集）：
1. Char-level: 字符级词表
2. Word-level: 词级词表（jieba 分词）
3. Subword-level: 使用预训练 BERT 词表

用法：
    python scripts/build_vocab.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizers import CharTokenizer, WordTokenizer, SubwordTokenizer
from src.dataset import load_jsonl


def main():
    """主函数"""
    base_dir = Path(__file__).parent.parent
    train_file = base_dir / 'data' / 'processed' / 'train_clean.json'
    vocab_dir = base_dir / 'data' / 'vocab'
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("构建词表")
    print("="*60)
    
    # 加载训练数据
    print("\n[1/4] 加载训练数据...")
    texts, labels = load_jsonl(train_file)
    print(f"  训练样本数: {len(texts)}")
    
    # 1. 构建字符级词表
    print("\n[2/4] 构建字符级词表...")
    char_tokenizer = CharTokenizer()
    char_tokenizer.build_vocab(texts, min_freq=1)
    char_vocab_path = vocab_dir / 'char_vocab.json'
    char_tokenizer.save_vocab(char_vocab_path)
    print(f"  词表大小: {len(char_tokenizer)}")
    print(f"  已保存: {char_vocab_path}")
    
    # 2. 构建词级词表
    print("\n[3/4] 构建词级词表 (jieba)...")
    word_tokenizer = WordTokenizer()
    word_tokenizer.build_vocab(texts, min_freq=2, max_vocab_size=50000)
    word_vocab_path = vocab_dir / 'word_vocab.json'
    word_tokenizer.save_vocab(word_vocab_path)
    print(f"  词表大小: {len(word_tokenizer)}")
    print(f"  已保存: {word_vocab_path}")
    
    # 3. 加载子词级词表（使用预训练 BERT）
    print("\n[4/4] 加载子词级词表 (BERT)...")
    subword_tokenizer = SubwordTokenizer()
    subword_tokenizer.build_vocab(texts)  # 使用预训练词表，无需构建
    subword_vocab_path = vocab_dir / 'subword_vocab.json'
    subword_tokenizer.save_vocab(subword_vocab_path)
    print(f"  词表大小: {len(subword_tokenizer)}")
    print(f"  已保存: {subword_vocab_path}")
    
    # 测试分词效果
    print("\n" + "="*60)
    print("分词效果测试")
    print("="*60)
    
    sample_text = texts[0]
    print(f"\n原始文本: {sample_text}")
    
    print(f"\nChar-level: {char_tokenizer.encode(sample_text)}")
    print(f"  长度: {len(char_tokenizer.encode(sample_text))}")
    
    print(f"\nWord-level: {word_tokenizer.encode(sample_text)}")
    print(f"  长度: {len(word_tokenizer.encode(sample_text))}")
    
    print(f"\nSubword-level: {subword_tokenizer.encode(sample_text)}")
    print(f"  长度: {len(subword_tokenizer.encode(sample_text))}")
    
    print("\n" + "="*60)
    print("词表构建完成!")
    print("="*60)


if __name__ == '__main__':
    main()
