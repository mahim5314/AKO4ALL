#!/bin/bash
# Quick benchmark runner (Modal B200 backend)
# Usage: bash scripts/bench.sh [label] [--force-baseline]
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

conda run -n fi-bench --no-capture-output modal run scripts/run_modal.py "${ARGS[@]}"
