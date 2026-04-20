#!/usr/bin/env bash
# Evaluate a trained LoRA adapter on SST-2 or IMDB test set.
#
# Usage: ./scripts/test_lora_acc.sh {sst2|imdb} [model_path]
# Optional env: CONDA_ENV, GPU_ID

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-sst2}"
MODEL_PATH="${2:-${RESULTS_DIR}/lora_${DATASET}/final_model}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "LoRA adapter not found at ${MODEL_PATH}" >&2
    echo "Train first:   ./scripts/train_lora.sh ${DATASET}" >&2
    exit 1
fi

python "${SRC_DIR}/test_model.py" \
    --model_path "${MODEL_PATH}" \
    --batch_size 32 \
    --max_length 128 \
    --gpu "${CUDA_VISIBLE_DEVICES}" \
    --dataset "${DATASET}"
