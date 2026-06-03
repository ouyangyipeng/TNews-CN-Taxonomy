#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多粒度分词器模块

实现三种分词策略：
1. CharTokenizer: 字符级切分（每个汉字、字母、数字、标点为一个 token）
2. WordTokenizer: 词级切分（使用 jieba 分词）
3. SubwordTokenizer: 子词级切分（使用 BERT 的 WordPiece 分词器）

统一接口：
- encode(text: str) -> List[str]: 分词
- encode_to_ids(text: str) -> List[int]: 分词并转换为 ID
- build_vocab(texts: List[str]): 构建词表（仅用于 Char/Word）
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import jieba


class BaseTokenizer:
    """分词器基类"""
    
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.pad_id = 0
        self.unk_id = 1
    
    def encode(self, text: str) -> List[str]:
        """分词（子类实现）"""
        raise NotImplementedError
    
    def encode_to_ids(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """分词并转换为 ID"""
        tokens = self.encode(text)
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        
        # 截断
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        
        return ids
    
    def build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab_size: Optional[int] = None):
        """构建词表（子类实现）"""
        raise NotImplementedError
    
    def save_vocab(self, path: Path):
        """保存词表"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'id2token': {str(k): v for k, v in self.id2token.items()},
                'pad_id': self.pad_id,
                'unk_id': self.unk_id
            }, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, path: Path):
        """加载词表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.id2token = {int(k): v for k, v in data['id2token'].items()}
        self.pad_id = data['pad_id']
        self.unk_id = data['unk_id']
    
    def __len__(self):
        return len(self.vocab)


class CharTokenizer(BaseTokenizer):
    """字符级分词器"""
    
    def encode(self, text: str) -> List[str]:
        """将文本切分为字符"""
        return list(text)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1, max_vocab_size: Optional[int] = None):
        """
        构建字符级词表
        
        Args:
            texts: 训练文本列表
            min_freq: 最小字符频率
            max_vocab_size: 最大词表大小（可选）
        """
        # 统计字符频率
        char_counter = Counter()
        for text in texts:
            char_counter.update(self.encode(text))
        
        # 添加特殊 token
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.id2token = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }
        
        # 按频率排序，过滤低频字符
        idx = 2
        for char, freq in char_counter.most_common(max_vocab_size):
            if freq >= min_freq:
                self.vocab[char] = idx
                self.id2token[idx] = char
                idx += 1
        
        print(f"  CharTokenizer 词表大小: {len(self.vocab)}")


class WordTokenizer(BaseTokenizer):
    """词级分词器（基于 jieba）"""
    
    def __init__(self, cut_all: bool = False):
        """
        Args:
            cut_all: 是否使用全模式分词（默认精确模式）
        """
        super().__init__()
        self.cut_all = cut_all
    
    def encode(self, text: str) -> List[str]:
        """使用 jieba 分词"""
        if self.cut_all:
            return jieba.lcut(text, cut_all=True)
        else:
            return jieba.lcut(text)
    
    def build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab_size: Optional[int] = None):
        """
        构建词级词表
        
        Args:
            texts: 训练文本列表
            min_freq: 最小词频
            max_vocab_size: 最大词表大小（可选）
        """
        # 统计词频
        word_counter = Counter()
        for text in texts:
            words = self.encode(text)
            word_counter.update(words)
        
        # 添加特殊 token
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.id2token = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }
        
        # 按频率排序，过滤低频词
        idx = 2
        for word, freq in word_counter.most_common(max_vocab_size):
            if freq >= min_freq:
                self.vocab[word] = idx
                self.id2token[idx] = word
                idx += 1
        
        print(f"  WordTokenizer 词表大小: {len(self.vocab)}")


class SubwordTokenizer(BaseTokenizer):
    """子词级分词器（基于 BERT WordPiece）"""
    
    def __init__(self, pretrained_model_name: str = 'bert-base-chinese'):
        """
        Args:
            pretrained_model_name: 预训练模型名称（用于加载分词器）
        """
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self._tokenizer = None
    
    def _load_tokenizer(self):
        """延迟加载 transformers 分词器"""
        if self._tokenizer is None:
            try:
                from transformers import BertTokenizer
                self._tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
                # 同步词表
                self.vocab = self._tokenizer.get_vocab()
                self.id2token = {v: k for k, v in self.vocab.items()}
                self.pad_id = self._tokenizer.pad_token_id
                self.unk_id = self._tokenizer.unk_token_id
                print(f"  SubwordTokenizer 词表大小: {len(self.vocab)} (from {self.pretrained_model_name})")
            except Exception as e:
                print(f"  警告: 无法加载 {self.pretrained_model_name}，回退到 CharTokenizer")
                print(f"  错误信息: {e}")
                # 回退到字符级
                self._tokenizer = None
                self.vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
                self.id2token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
    
    def encode(self, text: str) -> List[str]:
        """使用 WordPiece 分词"""
        self._load_tokenizer()
        if self._tokenizer is not None:
            return self._tokenizer.tokenize(text)
        else:
            # 回退到字符级
            return list(text)
    
    def encode_to_ids(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """分词并转换为 ID（包含特殊 token）"""
        self._load_tokenizer()
        if self._tokenizer is not None:
            # 使用 transformers 的 encode 方法（自动添加 [CLS] 和 [SEP]）
            ids = self._tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True
            )
            return ids
        else:
            # 回退到基类方法
            return super().encode_to_ids(text, max_len)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1, max_vocab_size: Optional[int] = None):
        """子词分词器使用预训练词表，无需构建"""
        self._load_tokenizer()
        print(f"  SubwordTokenizer 使用预训练词表，无需构建")


def get_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    """
    工厂函数：获取分词器实例
    
    Args:
        tokenizer_type: 'char', 'word', 'subword'
        **kwargs: 传递给分词器构造函数的参数
    
    Returns:
        BaseTokenizer 实例
    """
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == 'char':
        return CharTokenizer(**kwargs)
    elif tokenizer_type == 'word':
        return WordTokenizer(**kwargs)
    elif tokenizer_type == 'subword':
        return SubwordTokenizer(**kwargs)
    else:
        raise ValueError(f"未知的分词器类型: {tokenizer_type}")


def demo():
    """演示三种分词器的使用"""
    sample_text = "北京今日天气多云转晴，适合外出游玩。"
    
    print("="*60)
    print("分词器演示")
    print("="*60)
    print(f"\n原始文本: {sample_text}\n")
    
    # 字符级
    char_tokenizer = CharTokenizer()
    char_tokens = char_tokenizer.encode(sample_text)
    print(f"Char-level: {char_tokens}")
    print(f"  长度: {len(char_tokens)}")
    
    # 词级
    word_tokenizer = WordTokenizer()
    word_tokens = word_tokenizer.encode(sample_text)
    print(f"\nWord-level (jieba): {word_tokens}")
    print(f"  长度: {len(word_tokens)}")
    
    # 子词级
    subword_tokenizer = SubwordTokenizer()
    subword_tokens = subword_tokenizer.encode(sample_text)
    print(f"\nSubword-level (BERT WordPiece): {subword_tokens}")
    print(f"  长度: {len(subword_tokens)}")


if __name__ == '__main__':
    demo()
