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
DATASET=${1:-"sst2"}  # Default sst2, can pass imdb via argument

# Test parameters
BATCH_SIZE=32
NUM_RUNS=10
WARMUP_RUNS=3

# Select sample count based on dataset
if [ "$DATASET" == "imdb" ]; then
    NUM_SAMPLES=1000  # IMDB test set has 25000 samples, take 1000
else
    NUM_SAMPLES=872   # SST-2 validation set has 872 samples
fi

echo ""
echo "========================================================================"
echo "  Test original DistilBERT model inference speed"
echo "  Dataset: $DATASET"
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
echo "Test completed! Usage: ./benchmark_baseline.sh sst2 or ./benchmark_baseline.sh imdb"
echo ""

