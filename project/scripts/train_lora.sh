#!/usr/bin/env bash
# Train DistilBERT + LoRA on SST-2 or IMDB.
#
# Usage: ./scripts/train_lora.sh {sst2|imdb}
# Optional env: CONDA_ENV, GPU_ID

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
    echo "Usage: $0 {sst2|imdb}" >&2
    exit 1
fi

OUTPUT_DIR="${RESULTS_DIR}/lora_${DATASET}"
mkdir -p "${OUTPUT_DIR}"

echo "Training LoRA on ${DATASET} (GPU=${CUDA_VISIBLE_DEVICES}) -> ${OUTPUT_DIR}"

python "${SRC_DIR}/train_lora.py" \
    --dataset "${DATASET}" \
    --model_name distilbert-base-uncased \
    --output_dir "${OUTPUT_DIR}" \
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
