#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TNEWS 数据集探索性分析 (Exploratory Data Analysis)

功能：
1. 统计类别分布，识别长尾类别
2. 分析文本长度分布（字符级 + 词级）
3. 计算分位数，确定 max_len
4. 检测数据质量问题（空文本、重复、异常字符）
5. 生成可视化图表

输出：
- report/figures/class_distribution.png
- report/figures/text_length_distribution.png
- report/figures/keyword_stats.png
- data/eda_report.md
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jieba

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置 seaborn 风格
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_labels(file_path: Path) -> Dict[str, str]:
    """加载标签映射"""
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                labels[item['label']] = item['label_desc']
    return labels


def analyze_class_distribution(data: List[Dict], labels: Dict[str, str]) -> pd.DataFrame:
    """分析类别分布"""
    label_counts = Counter([item['label'] for item in data])
    
    # 构建 DataFrame
    df = pd.DataFrame([
        {
            'label': label,
            'label_desc': labels.get(label, 'unknown'),
            'count': count
        }
        for label, count in label_counts.items()
    ])
    df = df.sort_values('count', ascending=False).reset_index(drop=True)
    
    # 计算统计量
    max_count = df['count'].max()
    min_count = df['count'].min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print("\n" + "="*60)
    print("类别分布统计")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\n最大类别样本数: {max_count}")
    print(f"最小类别样本数: {min_count}")
    print(f"类别不平衡比例: {imbalance_ratio:.2f}")
    
    return df


def plot_class_distribution(df: pd.DataFrame, output_path: Path):
    """绘制类别分布图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 创建标签（包含类别名和数量）
    labels = [f"{row['label_desc']}\n({row['count']})" 
              for _, row in df.iterrows()]
    
    bars = ax.bar(range(len(df)), df['count'], color=sns.color_palette("husl", len(df)))
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('News Category', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('TNEWS Class Distribution', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n类别分布图已保存: {output_path}")
    plt.close()


def analyze_text_length(data: List[Dict]) -> Dict[str, np.ndarray]:
    """分析文本长度分布"""
    # 字符级长度
    char_lengths = np.array([len(item['sentence']) for item in data])
    
    # 词级长度（使用 jieba 分词）
    print("\n正在进行 jieba 分词以统计词级长度...")
    word_lengths = np.array([len(jieba.lcut(item['sentence'])) for item in data])
    
    stats = {
        'char_lengths': char_lengths,
        'word_lengths': word_lengths
    }
    
    # 计算分位数
    percentiles = [50, 90, 95, 99]
    
    print("\n" + "="*60)
    print("文本长度分布统计")
    print("="*60)
    
    print("\n字符级长度 (Char-level):")
    print(f"  最小值: {char_lengths.min()}")
    print(f"  最大值: {char_lengths.max()}")
    print(f"  平均值: {char_lengths.mean():.2f}")
    print(f"  中位数: {np.median(char_lengths):.2f}")
    for p in percentiles:
        print(f"  {p}分位数: {np.percentile(char_lengths, p):.2f}")
    
    print("\n词级长度 (Word-level, jieba):")
    print(f"  最小值: {word_lengths.min()}")
    print(f"  最大值: {word_lengths.max()}")
    print(f"  平均值: {word_lengths.mean():.2f}")
    print(f"  中位数: {np.median(word_lengths):.2f}")
    for p in percentiles:
        print(f"  {p}分位数: {np.percentile(word_lengths, p):.2f}")
    
    return stats


def plot_text_length_distribution(stats: Dict[str, np.ndarray], output_path: Path):
    """绘制文本长度分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 字符级长度分布
    ax1 = axes[0]
    sns.histplot(stats['char_lengths'], bins=50, kde=True, ax=ax1, color='steelblue')
    ax1.axvline(np.percentile(stats['char_lengths'], 95), color='red', 
                linestyle='--', label=f'95th percentile: {np.percentile(stats["char_lengths"], 95):.0f}')
    ax1.axvline(np.percentile(stats['char_lengths'], 99), color='orange', 
                linestyle='--', label=f'99th percentile: {np.percentile(stats["char_lengths"], 99):.0f}')
    ax1.set_xlabel('Character Length', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Character-level Text Length Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 词级长度分布
    ax2 = axes[1]
    sns.histplot(stats['word_lengths'], bins=50, kde=True, ax=ax2, color='coral')
    ax2.axvline(np.percentile(stats['word_lengths'], 95), color='red', 
                linestyle='--', label=f'95th percentile: {np.percentile(stats["word_lengths"], 95):.0f}')
    ax2.axvline(np.percentile(stats['word_lengths'], 99), color='orange', 
                linestyle='--', label=f'99th percentile: {np.percentile(stats["word_lengths"], 99):.0f}')
    ax2.set_xlabel('Word Length (jieba)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Word-level Text Length Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n文本长度分布图已保存: {output_path}")
    plt.close()


def analyze_keywords(data: List[Dict]) -> Dict:
    """分析关键词字段"""
    total = len(data)
    empty_keywords = sum(1 for item in data if not item.get('keywords', '').strip())
    missing_rate = empty_keywords / total
    
    # 统计关键词数量分布
    keyword_counts = []
    for item in data:
        keywords = item.get('keywords', '').strip()
        if keywords:
            # 关键词以逗号分隔
            count = len([k for k in keywords.split(',') if k.strip()])
            keyword_counts.append(count)
        else:
            keyword_counts.append(0)
    
    keyword_counts = np.array(keyword_counts)
    
    stats = {
        'total': total,
        'empty_keywords': empty_keywords,
        'missing_rate': missing_rate,
        'keyword_counts': keyword_counts
    }
    
    print("\n" + "="*60)
    print("关键词字段统计")
    print("="*60)
    print(f"总样本数: {total}")
    print(f"关键词为空的样本数: {empty_keywords}")
    print(f"关键词缺失率: {missing_rate:.2%}")
    print(f"\n关键词数量分布（非空样本）:")
    non_zero = keyword_counts[keyword_counts > 0]
    if len(non_zero) > 0:
        print(f"  最小值: {non_zero.min()}")
        print(f"  最大值: {non_zero.max()}")
        print(f"  平均值: {non_zero.mean():.2f}")
        print(f"  中位数: {np.median(non_zero):.2f}")
    
    return stats


def plot_keyword_stats(stats: Dict, output_path: Path):
    """绘制关键词统计图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 关键词缺失率饼图
    ax1 = axes[0]
    sizes = [stats['empty_keywords'], stats['total'] - stats['empty_keywords']]
    labels = ['Empty', 'Non-empty']
    colors = ['#ff9999', '#66b3ff']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Keywords Field Missing Rate', fontsize=12, fontweight='bold')
    
    # 关键词数量分布
    ax2 = axes[1]
    non_zero = stats['keyword_counts'][stats['keyword_counts'] > 0]
    if len(non_zero) > 0:
        sns.histplot(non_zero, bins=30, kde=True, ax=ax2, color='green')
        ax2.set_xlabel('Number of Keywords', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Keywords Count Distribution (Non-empty)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n关键词统计图已保存: {output_path}")
    plt.close()


def check_data_quality(data: List[Dict]) -> Dict:
    """检查数据质量"""
    total = len(data)
    
    # 空文本检测
    empty_texts = [i for i, item in enumerate(data) if not item.get('sentence', '').strip()]
    
    # 重复样本检测
    sentences = [item.get('sentence', '').strip() for item in data]
    duplicates = len(sentences) - len(set(sentences))
    
    # 异常字符检测（HTML 实体、特殊符号）
    html_entity_pattern = re.compile(r'&[a-zA-Z]+;|&#\d+;')
    special_char_pattern = re.compile(r'[\u200b\u200c\u200d\ufeff\xa0]')
    
    html_entities = []
    special_chars = []
    
    for i, item in enumerate(data):
        sentence = item.get('sentence', '')
        if html_entity_pattern.search(sentence):
            html_entities.append(i)
        if special_char_pattern.search(sentence):
            special_chars.append(i)
    
    # 全角/半角混用检测
    fullwidth_pattern = re.compile(r'[\uff01-\uff5e]')
    fullwidth_count = sum(1 for item in data if fullwidth_pattern.search(item.get('sentence', '')))
    
    quality_report = {
        'total': total,
        'empty_texts': len(empty_texts),
        'duplicates': duplicates,
        'html_entities': len(html_entities),
        'special_chars': len(special_chars),
        'fullwidth_mixed': fullwidth_count
    }
    
    print("\n" + "="*60)
    print("数据质量检查")
    print("="*60)
    print(f"总样本数: {total}")
    print(f"空文本: {len(empty_texts)}")
    print(f"重复样本: {duplicates}")
    print(f"包含 HTML 实体: {len(html_entities)}")
    print(f"包含特殊不可见字符: {len(special_chars)}")
    print(f"包含全角字符（可能混用）: {fullwidth_count}")
    
    # 输出部分示例
    if html_entities:
        print(f"\nHTML 实体示例 (前3个):")
        for idx in html_entities[:3]:
            print(f"  [{idx}] {data[idx]['sentence'][:80]}...")
    
    if special_chars:
        print(f"\n特殊字符示例 (前3个):")
        for idx in special_chars[:3]:
            print(f"  [{idx}] {repr(data[idx]['sentence'][:80])}...")
    
    return quality_report


def generate_eda_report(class_df: pd.DataFrame, 
                       length_stats: Dict[str, np.ndarray],
                       keyword_stats: Dict,
                       quality_report: Dict,
                       output_path: Path):
    """生成 EDA 报告"""
    report = []
    report.append("# TNEWS 数据集探索性分析报告\n")
    report.append(f"生成时间: 2026-05-30\n")
    
    report.append("\n## 1. 数据集概览\n")
    report.append(f"- **训练集样本数**: {quality_report['total']}\n")
    report.append(f"- **类别数**: 15\n")
    
    report.append("\n## 2. 类别分布\n")
    report.append("| Label | Description | Count |\n")
    report.append("|-------|-------------|-------|\n")
    for _, row in class_df.iterrows():
        report.append(f"| {row['label']} | {row['label_desc']} | {row['count']} |\n")
    
    max_count = class_df['count'].max()
    min_count = class_df['count'].min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    report.append(f"\n- **最大类别样本数**: {max_count}\n")
    report.append(f"- **最小类别样本数**: {min_count}\n")
    report.append(f"- **类别不平衡比例**: {imbalance_ratio:.2f}\n")
    
    report.append("\n## 3. 文本长度分布\n")
    report.append("\n### 3.1 字符级长度\n")
    char_lengths = length_stats['char_lengths']
    report.append(f"- 最小值: {char_lengths.min()}\n")
    report.append(f"- 最大值: {char_lengths.max()}\n")
    report.append(f"- 平均值: {char_lengths.mean():.2f}\n")
    report.append(f"- 中位数: {np.median(char_lengths):.2f}\n")
    report.append(f"- 90分位数: {np.percentile(char_lengths, 90):.2f}\n")
    report.append(f"- 95分位数: {np.percentile(char_lengths, 95):.2f}\n")
    report.append(f"- 99分位数: {np.percentile(char_lengths, 99):.2f}\n")
    
    report.append("\n### 3.2 词级长度 (jieba 分词)\n")
    word_lengths = length_stats['word_lengths']
    report.append(f"- 最小值: {word_lengths.min()}\n")
    report.append(f"- 最大值: {word_lengths.max()}\n")
    report.append(f"- 平均值: {word_lengths.mean():.2f}\n")
    report.append(f"- 中位数: {np.median(word_lengths):.2f}\n")
    report.append(f"- 90分位数: {np.percentile(word_lengths, 90):.2f}\n")
    report.append(f"- 95分位数: {np.percentile(word_lengths, 95):.2f}\n")
    report.append(f"- 99分位数: {np.percentile(word_lengths, 99):.2f}\n")
    
    report.append("\n### 3.3 max_len 建议\n")
    report.append(f"基于 95 分位数，建议：\n")
    report.append(f"- **Char-level max_len**: {int(np.percentile(char_lengths, 95))}\n")
    report.append(f"- **Word-level max_len**: {int(np.percentile(word_lengths, 95))}\n")
    
    report.append("\n## 4. 关键词字段分析\n")
    report.append(f"- **关键词缺失率**: {keyword_stats['missing_rate']:.2%}\n")
    non_zero = keyword_stats['keyword_counts'][keyword_stats['keyword_counts'] > 0]
    if len(non_zero) > 0:
        report.append(f"- **平均关键词数（非空样本）**: {non_zero.mean():.2f}\n")
    report.append("\n**结论**: 关键词字段缺失率较高，不建议作为主要特征，但可作为辅助信息。\n")
    
    report.append("\n## 5. 数据质量检查\n")
    report.append(f"- **空文本**: {quality_report['empty_texts']}\n")
    report.append(f"- **重复样本**: {quality_report['duplicates']}\n")
    report.append(f"- **HTML 实体**: {quality_report['html_entities']}\n")
    report.append(f"- **特殊不可见字符**: {quality_report['special_chars']}\n")
    report.append(f"- **全角/半角混用**: {quality_report['fullwidth_mixed']}\n")
    
    report.append("\n## 6. 预处理建议\n")
    report.append("1. **去除空文本和重复样本**\n")
    report.append("2. **清洗 HTML 实体和特殊字符**\n")
    report.append("3. **统一全角字符为半角**\n")
    report.append("4. **根据分词方式选择合适的 max_len**\n")
    report.append("5. **考虑类别不平衡问题，可在训练时使用类别权重**\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"\nEDA 报告已保存: {output_path}")


def main():
    """主函数"""
    # 路径设置
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'raw'
    figures_dir = base_dir / 'report' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TNEWS 数据集探索性分析 (EDA)")
    print("="*60)
    
    # 加载数据
    print("\n正在加载数据...")
    train_data = load_jsonl(data_dir / 'train.json')
    labels = load_labels(data_dir / 'labels.json')
    print(f"训练集样本数: {len(train_data)}")
    print(f"类别数: {len(labels)}")
    
    # 1. 类别分布分析
    class_df = analyze_class_distribution(train_data, labels)
    plot_class_distribution(class_df, figures_dir / 'class_distribution.png')
    
    # 2. 文本长度分析
    length_stats = analyze_text_length(train_data)
    plot_text_length_distribution(length_stats, figures_dir / 'text_length_distribution.png')
    
    # 3. 关键词分析
    keyword_stats = analyze_keywords(train_data)
    plot_keyword_stats(keyword_stats, figures_dir / 'keyword_stats.png')
    
    # 4. 数据质量检查
    quality_report = check_data_quality(train_data)
    
    # 5. 生成报告
    generate_eda_report(
        class_df, 
        length_stats, 
        keyword_stats, 
        quality_report,
        data_dir.parent / 'eda_report.md'
    )
    
    print("\n" + "="*60)
    print("EDA 完成!")
    print("="*60)


if __name__ == '__main__':
    main()
