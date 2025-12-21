#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wqs

# Set CUDA path
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Parameter settings
GPU_ID=0
DATASET=${1:-"sst2"}  # Default sst2, can pass imdb

# Test parameters (consistent with LoRA for comparison)
BATCH_SIZE=32
NUM_RUNS=30
WARMUP_RUNS=3

# Select Sparse LoRA model path and sample count based on dataset
if [ "$DATASET" == "imdb" ]; then
    MODEL_PATH="./results_sparse_lora_imdb/final_sparse_lora_model"
    NUM_SAMPLES=1000     # Consistent with LoRA benchmark
else
    MODEL_PATH="./results_sparse_lora/final_sparse_lora_model"
    NUM_SAMPLES=872      # All samples from SST-2 validation set
fi

echo ""
echo "========================================================================"
echo "  Test Sparse LoRA fine-tuned model inference speed"
echo "  Dataset: $DATASET"
echo "========================================================================"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Sparse LoRA model not found ($MODEL_PATH)"
    echo "Please run Sparse LoRA training script first"
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
echo "Test completed! Usage: ./benchmark_sparse_lora.sh sst2 or ./benchmark_sparse_lora.sh imdb"
echo ""
