#!/usr/bin/env bash
# Benchmark HiRA inference latency.
#
# Usage: ./scripts/test_hira_lat.sh {sst2|imdb} [model_path]
# Optional env: CONDA_ENV, GPU_ID

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-sst2}"
MODEL_PATH="${2:-${RESULTS_DIR}/hira_${DATASET}/final_model}"

if [[ "${DATASET}" == "imdb" ]]; then
    NUM_SAMPLES=1000
else
    NUM_SAMPLES=872
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "HiRA adapter not found at ${MODEL_PATH}" >&2
    exit 1
fi

OUTPUT_DIR="${RESULTS_DIR}/benchmark_hira"
mkdir -p "${OUTPUT_DIR}"

python "${SRC_DIR}/benchmark_inference_hira.py" \
    --model_path "${MODEL_PATH}" \
    --model_type lora \
    --batch_size 32 \
    --num_samples "${NUM_SAMPLES}" \
    --num_runs 30 \
    --warmup_runs 3 \
    --gpu "${CUDA_VISIBLE_DEVICES}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset "${DATASET}"
