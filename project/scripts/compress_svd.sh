#!/usr/bin/env bash
# Run SVD-guided adaptive rank compression on a trained LoRA adapter.
#
# Usage: ./scripts/compress_svd.sh {sst2|imdb} [energy_threshold] [post_epochs]
#
# Example:
#   ./scripts/compress_svd.sh sst2           # compress only, tau=0.9
#   ./scripts/compress_svd.sh imdb 0.9 2     # compress + 2 epochs of fine-tuning

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-sst2}"
TAU="${2:-0.9}"
POST_EPOCHS="${3:-2}"

MODEL_PATH="${RESULTS_DIR}/lora_${DATASET}/final_model"
OUTPUT_DIR="${RESULTS_DIR}/lora_${DATASET}_svd"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "LoRA adapter not found at ${MODEL_PATH}" >&2
    echo "Train first:   ./scripts/train_lora.sh ${DATASET}" >&2
    exit 1
fi

cd "${SRC_DIR}"
python -m svd_compress.compress_lora \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --energy_threshold "${TAU}" \
    --dataset "${DATASET}" \
    --post_epochs "${POST_EPOCHS}" \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --max_length 128 \
    --seed 42
