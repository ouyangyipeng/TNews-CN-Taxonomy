#!/bin/bash
# 打包脚本：打包 TNEWS 中文短新闻分类项目

set -e

PROJECT_NAME="2026ANN-NLP-23336188-欧阳易芃"
OUTPUT_ZIP="${PROJECT_NAME}.zip"

echo "=========================================="
echo "开始打包项目: ${PROJECT_NAME}"
echo "=========================================="

# 删除旧的 zip 文件
if [ -f "${OUTPUT_ZIP}" ]; then
    echo "删除旧的 zip 文件..."
    rm -f "${OUTPUT_ZIP}"
fi

# 创建临时目录
TEMP_DIR=$(mktemp -d)
PROJECT_DIR="${TEMP_DIR}/${PROJECT_NAME}"
mkdir -p "${PROJECT_DIR}"

echo "复制项目文件..."

# 复制源代码
cp -r src "${PROJECT_DIR}/"

# 复制脚本
cp -r scripts "${PROJECT_DIR}/"

# 复制数据（排除原始数据，只保留处理后的数据）
mkdir -p "${PROJECT_DIR}/data"
cp -r data/processed "${PROJECT_DIR}/data/" 2>/dev/null || true
cp -r data/vocab "${PROJECT_DIR}/data/" 2>/dev/null || true
cp -r data/splits "${PROJECT_DIR}/data/" 2>/dev/null || true
cp data/eda_report.md "${PROJECT_DIR}/data/" 2>/dev/null || true

# 复制报告
cp -r report "${PROJECT_DIR}/"

# 复制配置文件
cp -r configs "${PROJECT_DIR}/" 2>/dev/null || true

# 复制计划文档
cp -r plans "${PROJECT_DIR}/"

# 复制 README 和 ROADMAP
cp README.md "${PROJECT_DIR}/"
cp ROADMAP.md "${PROJECT_DIR}/"

# 复制日志（排除 A100 日志）
mkdir -p "${PROJECT_DIR}/logs"
cp -r logs/bilstm_char "${PROJECT_DIR}/logs/" 2>/dev/null || true
cp -r logs/bilstm_word "${PROJECT_DIR}/logs/" 2>/dev/null || true
cp -r logs/transformer_char "${PROJECT_DIR}/logs/" 2>/dev/null || true
cp -r logs/transformer_char_v2 "${PROJECT_DIR}/logs/" 2>/dev/null || true
cp -r logs/transformer_word "${PROJECT_DIR}/logs/" 2>/dev/null || true
cp -r logs/transformer_subword "${PROJECT_DIR}/logs/" 2>/dev/null || true

# 复制检查点（排除 A100 检查点）
mkdir -p "${PROJECT_DIR}/checkpoints"
cp -r checkpoints/bilstm_char "${PROJECT_DIR}/checkpoints/" 2>/dev/null || true
cp -r checkpoints/bilstm_word "${PROJECT_DIR}/checkpoints/" 2>/dev/null || true
cp -r checkpoints/transformer_char "${PROJECT_DIR}/checkpoints/" 2>/dev/null || true
cp -r checkpoints/transformer_char_v2 "${PROJECT_DIR}/checkpoints/" 2>/dev/null || true
cp -r checkpoints/transformer_word "${PROJECT_DIR}/checkpoints/" 2>/dev/null || true
cp -r checkpoints/transformer_subword "${PROJECT_DIR}/checkpoints/" 2>/dev/null || true

# 创建 zip 文件
echo "创建 zip 文件..."
cd "${TEMP_DIR}"
zip -r "${OUTPUT_ZIP}" "${PROJECT_NAME}" -x "*.pyc" -x "__pycache__/*" -x ".venv/*" -x ".git/*"

# 移动 zip 文件到项目根目录
mv "${OUTPUT_ZIP}" "/root/Course/NN/HW2/"

# 清理临时目录
cd /root/Course/NN/HW2
rm -rf "${TEMP_DIR}"

echo "=========================================="
echo "打包完成！"
echo "输出文件: ${OUTPUT_ZIP}"
echo "文件大小: $(ls -lh ${OUTPUT_ZIP} | awk '{print $5}')"
echo "=========================================="
