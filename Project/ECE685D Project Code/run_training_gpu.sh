#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wqs

# Set correct CUDA path (fix DeepSpeed not finding CUDA)
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ============================================
# Configuration parameters (can be modified here)
# ============================================
# Specify GPU to use (default: 0)
export CUDA_VISIBLE_DEVICES=0

# Dataset selection: sst2 or imdb (must be specified)
DATASET=$1

# Check if dataset argument is provided
if [ -z "$DATASET" ]; then
    echo "Error: Please specify dataset!"
    echo "Usage: ./run_training_gpu.sh sst2 or ./run_training_gpu.sh imdb"
    exit 1
fi

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i $CUDA_VISIBLE_DEVICES

# Run training script
python train_distilbert_lora.py \
    --dataset $DATASET \
    --model_name distilbert-base-uncased \
    --output_dir ./results \
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

echo "Training completed!"
echo "Usage: ./run_training_gpu.sh sst2 or ./run_training_gpu.sh imdb"

