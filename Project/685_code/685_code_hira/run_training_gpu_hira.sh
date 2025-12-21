#!/bin/bash

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
# conda activate wqs

# 设置正确的CUDA路径（修复DeepSpeed找不到CUDA的问题）
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ============================================
# 配置参数（可以修改这里）
# ============================================
# 指定使用哪张GPU（默认：0）
export CUDA_VISIBLE_DEVICES=0

# 数据集选择: sst2、imdb 或 wikitext2（必须指定）
DATASET=$1

# 检查是否提供了数据集参数
if [ -z "$DATASET" ]; then
    echo "错误: 请指定数据集！"
    echo "用法: ./run_training_gpu.sh sst2 或 ./run_training_gpu.sh imdb 或 ./run_training_gpu.sh wikitext2"
    exit 1
fi
# # 根据数据集选择模型路径
# if [ "$DATASET" == "imdb" ]; then
#     MODEL_PATH="./results-hira_imdb"
# else
#     MODEL_PATH="./results-hira"
# fi

echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "数据集: $DATASET"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i $CUDA_VISIBLE_DEVICES

# 运行训练脚本
python train_distilbert_hira.py \
    --dataset $DATASET \
    --model_name distilbert-base-uncased \
    --output_dir ./results-hira \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --max_length 128 \
    --save_steps 500 \
    --eval_steps 500 \
    --seed 42

echo "训练完成！"
echo "用法: ./run_training_gpu.sh sst2 或 ./run_training_gpu.sh imdb 或 ./run_training_gpu.sh wikitext2"

