Optimize the {{LANGUAGE_NAME}} kernel in `solution/{{LANGUAGE}}/kernel.py` for maximum speedup on {{GPU_NAME}}.
This is a **{{OPERATOR_DESCRIPTION}}** operator.

The starting point is an **already partially-optimized {{LANGUAGE_NAME}} kernel**. Analyze it,
identify bottlenecks, and optimize for maximum speedup while preserving correctness.

The pure-Python reference is in `docs/definition.json` (under "reference") for understanding the computation.

**Recommended first step:** Run `bash scripts/bench.sh "baseline"` to establish current performance.
