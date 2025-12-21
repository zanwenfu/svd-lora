#!/bin/bash

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wqs

# 设置CUDA路径
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 参数设置
GPU_ID=0
DATASET=${1:-"sst2"}  # 默认 sst2，可传 imdb

# 测试参数（和 LoRA 保持一致，方便对比）
BATCH_SIZE=32
NUM_RUNS=30
WARMUP_RUNS=3

# 根据数据集选择 Sparse LoRA 模型路径和样本数
if [ "$DATASET" == "imdb" ]; then
    MODEL_PATH="./results_sparse_lora_imdb/final_sparse_lora_model"
    NUM_SAMPLES=1000     # 和 LoRA 的基准保持一致
else
    MODEL_PATH="./results_sparse_lora/final_sparse_lora_model"
    NUM_SAMPLES=872      # SST-2 验证集全部样本
fi

echo ""
echo "========================================================================"
echo "  测试 Sparse LoRA 微调模型推理速度"
echo "  数据集: $DATASET"
echo "========================================================================"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 未找到 Sparse LoRA 模型 ($MODEL_PATH)"
    echo "请先运行 Sparse LoRA 训练脚本"
    exit 1
fi

python benchmark_inference.py \
    --model_path $MODEL_PATH \
    --model_type sparse_lora \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --num_runs $NUM_RUNS \
    --warmup_runs $WARMUP_RUNS \
    --gpu $GPU_ID \
    --output_dir ./benchmark_results_sparse_lora \
    --dataset $DATASET

echo ""
echo "测试完成！用法: ./benchmark_sparse_lora.sh sst2 或 ./benchmark_sparse_lora.sh imdb"
echo ""
