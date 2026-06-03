#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成优化效果累积图
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
strategies = ['BiLSTM+Attn', 'BERT-base', '+Augment', '+Simplified', 'Ensemble']
values = [47.00, 52.08, 53.30, 54.36, 54.11]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bars = ax.bar(strategies, values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

# 在柱子上方添加数值标签
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置坐标轴
ax.set_ylabel('Macro-F1 (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Optimization Strategy', fontsize=14, fontweight='bold')
ax.set_ylim(44, 57)

# 旋转x轴标签
plt.xticks(rotation=15, ha='right', fontsize=11)

# 添加网格
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加标题
ax.set_title('Cumulative Effect of Optimization Strategies', fontsize=16, fontweight='bold', pad=20)

# 调整布局
plt.tight_layout()

# 保存图表
output_dir = Path('report/figures')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'optimization_progress.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"优化效果图已保存: {output_path}")

plt.close()
