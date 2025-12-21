#!/bin/bash

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wqs

# 设置正确的CUDA路径（修复DeepSpeed找不到CUDA的问题）
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ============================================
# 配置参数
# ============================================
GPU_ID=0
DATASET=$1  # 必须指定数据集: sst2 或 imdb

# 检查是否提供了数据集参数
if [ -z "$DATASET" ]; then
    echo "错误: 请指定数据集！"
    echo "用法: ./run_test_baseline.sh sst2 或 ./run_test_baseline.sh imdb"
    exit 1
fi

echo "=========================================="
echo "测试原始DistilBERT模型（未微调）"
echo "数据集: $DATASET"
echo "使用GPU: $GPU_ID"
echo "=========================================="
echo ""

# 测试原始模型
python test_baseline.py \
    --dataset $DATASET \
    --model_name distilbert-base-uncased \
    --batch_size 32 \
    --max_length 128 \
    --gpu $GPU_ID \
    --output_dir ./baseline_results

echo ""
echo "=========================================="
echo "测试完成！"
echo "用法: ./run_test_baseline.sh sst2 或 ./run_test_baseline.sh imdb"
echo "=========================================="

