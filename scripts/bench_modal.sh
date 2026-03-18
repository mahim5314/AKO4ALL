#!/bin/bash
# Quick benchmark runner (Modal B200 backend)
# Usage: bash scripts/bench.sh [label]
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    conda run -n fi-bench --no-capture-output modal run scripts/run_modal.py
else
    conda run -n fi-bench --no-capture-output modal run scripts/run_modal.py --label "$1"
fi
