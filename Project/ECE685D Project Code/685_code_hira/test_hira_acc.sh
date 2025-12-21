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
# Parameter settings
# ============================================
GPU_ID=0
DATASET=${1:-"sst2"}  # Default sst2, can pass imdb via argument

# Select model path based on dataset
if [ "$DATASET" == "imdb" ]; then
    MODEL_PATH="./results-hira_imdb/final_model"
else
    MODEL_PATH="./results-hira/final_model"
fi

echo "=========================================="
echo "Test LoRA fine-tuned model"
echo "Using GPU: $GPU_ID"
echo "Dataset: $DATASET"
echo "Model path: $MODEL_PATH"
echo "=========================================="
echo ""

# Evaluate model
CUDA_VISIBLE_DEVICES=0 python test_model_hira.py \
    --model_path $MODEL_PATH \
    --batch_size 32 \
    --max_length 128 \
    --gpu $GPU_ID \
    --dataset $DATASET

echo ""
echo "=========================================="
echo "Test completed!"
echo "Usage: ./run_test.sh sst2 or ./run_test.sh imdb"
echo "=========================================="

