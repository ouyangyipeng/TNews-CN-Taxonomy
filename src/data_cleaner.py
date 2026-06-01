#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗模块

功能：
1. 去除无效样本（空文本、非法标签、重复样本）
2. 文本规范化（全角转半角、去除不可见字符、HTML 实体清理）
3. 标签映射（字符串标签 -> 整数索引）

输入：data/raw/train.json, data/raw/dev.json
输出：data/processed/train_clean.json, data/processed/dev_clean.json
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple


# 全角字符 -> 半角字符映射表
_FULLWIDTH_TO_HALFWIDTH = {
    # 全角数字
    **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},
    # 全角大写字母
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
    # 全角小写字母
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},
    # 全角标点
    '\uff01': '!',   # ！
    '\uff08': '(',   # （
    '\uff09': ')',   # ）
    '\uff0c': ',',   # ，
    '\uff1a': ':',   # ：
    '\uff1b': ';',   # ；
    '\uff1f': '?',   # ？
}

# HTML 实体映射
_HTML_ENTITIES = {
    '&': '&',
    '<': '<',
    '>': '>',
    '"': '"',
    ''': "'",
    '&nbsp;': ' ',
    ''': "'",
    '&#34;': '"',
}

# 不可见字符正则
_INVISIBLE_CHARS = re.compile(r'[\u200b\u200c\u200d\ufeff\xa0\u200e\u200f\u202a-\u202e\u2060-\u206f]')

# HTML 实体正则
_HTML_ENTITY_PATTERN = re.compile(r'&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);')


def fullwidth_to_halfwidth(text: str) -> str:
    """全角字符转半角"""
    result = []
    for char in text:
        if char in _FULLWIDTH_TO_HALFWIDTH:
            result.append(_FULLWIDTH_TO_HALFWIDTH[char])
        elif '\uff01' <= char <= '\uff5e':
            # 其他全角 ASCII 字符
            result.append(chr(ord(char) - 0xFEE0))
        else:
            result.append(char)
    return ''.join(result)


def clean_html_entities(text: str) -> str:
    """清理 HTML 实体"""
    # 先替换已知实体
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    
    # 处理数字实体 &#123;
    def replace_numeric_entity(match):
        entity = match.group(0)
        if entity.startswith('&#x') or entity.startswith('&#X'):
            try:
                code = int(entity[3:-1], 16)
                return chr(code)
            except (ValueError, OverflowError):
                return ''
        elif entity.startswith('&#'):
            try:
                code = int(entity[2:-1])
                return chr(code)
            except (ValueError, OverflowError):
                return ''
        return entity
    
    text = _HTML_ENTITY_PATTERN.sub(replace_numeric_entity, text)
    return text


def remove_invisible_chars(text: str) -> str:
    """去除不可见字符"""
    return _INVISIBLE_CHARS.sub('', text)


def normalize_whitespace(text: str) -> str:
    """规范化空白字符"""
    # 将各种空白字符统一为普通空格
    text = re.sub(r'[\t\r\n]+', ' ', text)
    # 合并连续空格
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def clean_text(text: str) -> str:
    """完整的文本清洗流水线"""
    # 1. 去除不可见字符
    text = remove_invisible_chars(text)
    # 2. 清理 HTML 实体
    text = clean_html_entities(text)
    # 3. 全角转半角
    text = fullwidth_to_halfwidth(text)
    # 4. Unicode NFKC 规范化
    text = unicodedata.normalize('NFKC', text)
    # 5. 规范化空白字符
    text = normalize_whitespace(text)
    return text


def load_labels(label_file: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    加载标签映射文件，构建 label2id 和 id2label
    
    Returns:
        label2id: {"100": 0, "101": 1, ...}
        id2label: {0: "news_story", 1: "news_culture", ...}
    """
    label_info = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                label_info.append((item['label'], item['label_desc']))
    
    # 按 label 数值排序，确保映射稳定
    label_info.sort(key=lambda x: int(x[0]))
    
    label2id = {}
    id2label = {}
    for idx, (label, label_desc) in enumerate(label_info):
        label2id[label] = idx
        id2label[idx] = label_desc
    
    return label2id, id2label


def clean_dataset(
    data: List[Dict],
    label2id: Dict[str, int],
    log_file: Path = None
) -> List[Dict]:
    """
    清洗数据集
    
    Args:
        data: 原始数据列表
        label2id: 标签到 ID 的映射
        log_file: 日志文件路径
    
    Returns:
        清洗后的数据列表
    """
    logs = []
    original_count = len(data)
    
    # 1. 去除空文本
    before = len(data)
    data = [item for item in data if item.get('sentence', '').strip()]
    empty_count = before - len(data)
    logs.append(f"去除空文本: {empty_count} 条")
    
    # 2. 去除非法标签
    before = len(data)
    data = [item for item in data if item.get('label', '') in label2id]
    invalid_label_count = before - len(data)
    logs.append(f"去除非法标签: {invalid_label_count} 条")
    
    # 3. 去除重复样本（基于 sentence 去重，保留第一个）
    before = len(data)
    seen_sentences = set()
    deduped_data = []
    for item in data:
        sentence = item['sentence'].strip()
        if sentence not in seen_sentences:
            seen_sentences.add(sentence)
            deduped_data.append(item)
    data = deduped_data
    duplicate_count = before - len(data)
    logs.append(f"去除重复样本: {duplicate_count} 条")
    
    # 4. 文本清洗
    for item in data:
        item['sentence'] = clean_text(item['sentence'])
        # 将标签映射为整数索引
        item['label_id'] = label2id[item['label']]
    
    # 5. 再次去除清洗后变为空的文本
    before = len(data)
    data = [item for item in data if item['sentence']]
    empty_after_clean = before - len(data)
    logs.append(f"清洗后变为空文本: {empty_after_clean} 条")
    
    final_count = len(data)
    logs.append(f"\n清洗完成: {original_count} -> {final_count} (减少 {original_count - final_count})")
    
    # 写入日志
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(logs))
    
    for log in logs:
        print(f"  {log}")
    
    return data


def save_jsonl(data: List[Dict], output_file: Path):
    """保存为 JSONL 格式"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 只保留必要字段，兼容无标签的测试集
            record = {
                'sentence': item.get('sentence', ''),
                'keywords': item.get('keywords', '')
            }
            # 有标签时才添加标签相关字段
            if 'label' in item:
                record['label'] = item['label']
                record['label_desc'] = item.get('label_desc', '')
            if 'label_id' in item:
                record['label_id'] = item['label_id']
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    """主函数：执行数据清洗流水线"""
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("数据清洗流水线")
    print("="*60)
    
    # 加载标签映射
    print("\n[1/4] 加载标签映射...")
    label2id, id2label = load_labels(raw_dir / 'labels.json')
    print(f"  类别数: {len(label2id)}")
    for label, idx in label2id.items():
        print(f"  {label} ({id2label[idx]}) -> {idx}")
    
    # 保存标签映射
    label_map = {
        'label2id': label2id,
        'id2label': {str(k): v for k, v in id2label.items()}
    }
    with open(processed_dir / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  标签映射已保存: {processed_dir / 'label_map.json'}")
    
    # 加载并清洗训练集
    print("\n[2/4] 清洗训练集...")
    with open(raw_dir / 'train.json', 'r', encoding='utf-8') as f:
        train_data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"  原始样本数: {len(train_data)}")
    
    train_clean = clean_dataset(
        train_data, label2id,
        log_file=processed_dir / 'train_clean_log.txt'
    )
    save_jsonl(train_clean, processed_dir / 'train_clean.json')
    print(f"  清洗后样本数: {len(train_clean)}")
    
    # 加载并清洗验证集（dev.json 作为最终测试集）
    print("\n[3/4] 清洗验证集 (dev.json)...")
    with open(raw_dir / 'dev.json', 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"  原始样本数: {len(dev_data)}")
    
    dev_clean = clean_dataset(
        dev_data, label2id,
        log_file=processed_dir / 'dev_clean_log.txt'
    )
    save_jsonl(dev_clean, processed_dir / 'dev_clean.json')
    print(f"  清洗后样本数: {len(dev_clean)}")
    
    # 清洗测试集（无标签，仅做文本清洗）
    print("\n[4/4] 清洗测试集 (test.json, 无标签)...")
    with open(raw_dir / 'test.json', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"  原始样本数: {len(test_data)}")
    
    # 测试集没有标签，只做文本清洗
    for item in test_data:
        item['sentence'] = clean_text(item.get('sentence', ''))
    
    save_jsonl(test_data, processed_dir / 'test_clean.json')
    print(f"  清洗后样本数: {len(test_data)}")
    
    print("\n" + "="*60)
    print("数据清洗完成!")
    print("="*60)


if __name__ == '__main__':
    main()
