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
# 参数设置
# ============================================
GPU_ID=0
DATASET=${1:-"sst2"}  # 默认sst2，可通过参数传入imdb

# 根据数据集选择模型路径
if [ "$DATASET" == "imdb" ]; then
    MODEL_PATH="./results-hira_imdb/final_model"
else
    MODEL_PATH="./results-hira/final_model"
fi

echo "=========================================="
echo "测试LoRA微调后的模型"
echo "使用GPU: $GPU_ID"
echo "数据集: $DATASET"
echo "模型路径: $MODEL_PATH"
echo "=========================================="
echo ""

# 评估模型
CUDA_VISIBLE_DEVICES=0 python test_model_hira.py \
    --model_path $MODEL_PATH \
    --batch_size 32 \
    --max_length 128 \
    --gpu $GPU_ID \
    --dataset $DATASET

echo ""
echo "=========================================="
echo "测试完成！"
echo "用法: ./run_test.sh sst2 或 ./run_test.sh imdb"
echo "=========================================="

