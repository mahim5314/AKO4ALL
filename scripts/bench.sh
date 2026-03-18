#!/bin/bash
# Quick benchmark runner
# Usage: bash scripts/bench.sh [label] [--force-baseline]
if [ -z "$FIB_DATASET_PATH" ]; then
    echo "Error: FIB_DATASET_PATH is not set. Set it to the path of your flashinfer-bench trace set."
    exit 1
fi
cd "$(dirname "$0")/.."

# Parse arguments: label (positional) and --force-baseline (flag)
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--force-baseline" ]; then
        ARGS+=(--force-baseline)
    else
        ARGS+=(--label "$arg")
    fi
done

conda run -n fi-bench --no-capture-output python scripts/run_local.py "${ARGS[@]}"
