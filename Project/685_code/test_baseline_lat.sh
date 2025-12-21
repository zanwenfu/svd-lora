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
DATASET=${1:-"sst2"}  # 默认sst2，可通过参数传入imdb

# 测试参数
BATCH_SIZE=32
NUM_RUNS=10
WARMUP_RUNS=3

# 根据数据集选择样本数
if [ "$DATASET" == "imdb" ]; then
    NUM_SAMPLES=1000  # IMDB测试集有25000样本，取1000
else
    NUM_SAMPLES=872   # SST-2验证集共872样本
fi

echo ""
echo "========================================================================"
echo "  测试原始DistilBERT模型推理速度"
echo "  数据集: $DATASET"
echo "========================================================================"
echo ""

python benchmark_inference.py \
    --model_path distilbert-base-uncased \
    --model_type baseline \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --num_runs $NUM_RUNS \
    --warmup_runs $WARMUP_RUNS \
    --gpu $GPU_ID \
    --output_dir ./benchmark_results \
    --dataset $DATASET

echo ""
echo "测试完成！用法: ./benchmark_baseline.sh sst2 或 ./benchmark_baseline.sh imdb"
echo ""

