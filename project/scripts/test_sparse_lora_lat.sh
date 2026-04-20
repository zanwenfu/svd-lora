#!/usr/bin/env bash
# Benchmark Sparse LoRA inference latency. Shares hyperparameters with LoRA so
# measurements are directly comparable.
#
# Usage: ./scripts/test_sparse_lora_lat.sh {sst2|imdb} [model_path]
# Optional env: CONDA_ENV, GPU_ID

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-sst2}"
MODEL_PATH="${2:-${RESULTS_DIR}/sparse_lora_${DATASET}/final_sparse_lora_model}"

if [[ "${DATASET}" == "imdb" ]]; then
    NUM_SAMPLES=1000
else
    NUM_SAMPLES=872
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Sparse LoRA adapter not found at ${MODEL_PATH}" >&2
    exit 1
fi

OUTPUT_DIR="${RESULTS_DIR}/benchmark_sparse_lora"
mkdir -p "${OUTPUT_DIR}"

python "${SRC_DIR}/benchmark_inference.py" \
    --model_path "${MODEL_PATH}" \
    --model_type sparse_lora \
    --batch_size 32 \
    --num_samples "${NUM_SAMPLES}" \
    --num_runs 30 \
    --warmup_runs 3 \
    --gpu "${CUDA_VISIBLE_DEVICES}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset "${DATASET}"
