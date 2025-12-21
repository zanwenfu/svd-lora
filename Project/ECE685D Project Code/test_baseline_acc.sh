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
# Configuration parameters
# ============================================
GPU_ID=0
DATASET=$1  # Must specify dataset: sst2 or imdb

# Check if dataset argument is provided
if [ -z "$DATASET" ]; then
    echo "Error: Please specify dataset!"
    echo "Usage: ./run_test_baseline.sh sst2 or ./run_test_baseline.sh imdb"
    exit 1
fi

echo "=========================================="
echo "Test original DistilBERT model (untuned)"
echo "Dataset: $DATASET"
echo "Using GPU: $GPU_ID"
echo "=========================================="
echo ""

# Test original model
python test_baseline.py \
    --dataset $DATASET \
    --model_name distilbert-base-uncased \
    --batch_size 32 \
    --max_length 128 \
    --gpu $GPU_ID \
    --output_dir ./baseline_results

echo ""
echo "=========================================="
echo "Test completed!"
echo "Usage: ./run_test_baseline.sh sst2 or ./run_test_baseline.sh imdb"
echo "=========================================="

