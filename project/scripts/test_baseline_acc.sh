#!/usr/bin/env bash
# Evaluate the unfinetuned DistilBERT baseline on SST-2 or IMDB.
#
# Usage: ./scripts/test_baseline_acc.sh {sst2|imdb}
# Optional env: CONDA_ENV, GPU_ID

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
    echo "Usage: $0 {sst2|imdb}" >&2
    exit 1
fi

OUTPUT_DIR="${RESULTS_DIR}/baseline_${DATASET}"
mkdir -p "${OUTPUT_DIR}"

python "${SRC_DIR}/test_baseline.py" \
    --dataset "${DATASET}" \
    --model_name distilbert-base-uncased \
    --batch_size 32 \
    --max_length 128 \
    --gpu "${CUDA_VISIBLE_DEVICES}" \
    --output_dir "${OUTPUT_DIR}"
