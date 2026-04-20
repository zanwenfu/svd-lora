#!/usr/bin/env bash
# Common environment setup for all project scripts.
#
# By default, scripts run with whatever `python` is on the PATH. If you want to
# run them inside a specific conda environment, set CONDA_ENV before invoking:
#
#     CONDA_ENV=ece685 ./scripts/train_lora.sh sst2
#
# Or activate the env yourself before calling the script. We do NOT hardcode
# any env name or CUDA path — that's the operator's decision.

set -euo pipefail

if [[ -n "${CONDA_ENV:-}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

# Resolve repo paths relative to this script, regardless of cwd.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_DIR}/src"
RESULTS_DIR="${PROJECT_DIR}/results"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# GPU selection. Default to 0 if available; scripts can override.
: "${GPU_ID:=0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
