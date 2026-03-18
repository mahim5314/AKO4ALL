#!/bin/bash
# Quick benchmark runner
# Usage: bash scripts/bench.sh [label]
export FIB_DATASET_PATH="${FIB_DATASET_PATH:-/root/FlashInfer/mlsys26-contest}"
cd "$(dirname "$0")/.."

# Pass optional label parameter to run_local.py
if [ $# -eq 0 ]; then
    conda run -n fi-bench --no-capture-output python scripts/run_local.py
else
    conda run -n fi-bench --no-capture-output python scripts/run_local.py --label "$1"
fi
