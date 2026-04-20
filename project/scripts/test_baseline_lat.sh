#!/usr/bin/env bash
# Benchmark inference latency of the unfinetuned DistilBERT baseline.
#
# Usage: ./scripts/test_baseline_lat.sh {sst2|imdb}
# Optional env: CONDA_ENV, GPU_ID

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-sst2}"

if [[ "${DATASET}" == "imdb" ]]; then
    NUM_SAMPLES=1000
else
    NUM_SAMPLES=872
fi

OUTPUT_DIR="${RESULTS_DIR}/benchmark_baseline"
mkdir -p "${OUTPUT_DIR}"

python "${SRC_DIR}/benchmark_inference.py" \
    --model_path distilbert-base-uncased \
    --model_type baseline \
    --batch_size 32 \
    --num_samples "${NUM_SAMPLES}" \
    --num_runs 10 \
    --warmup_runs 3 \
    --gpu "${CUDA_VISIBLE_DEVICES}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset "${DATASET}"
