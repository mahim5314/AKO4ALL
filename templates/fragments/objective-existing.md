Optimize the {{LANGUAGE_NAME}} kernel in `solution/{{LANGUAGE}}/` for maximum speedup on {{GPU_NAME}}.
This is a **{{OPERATOR_DESCRIPTION}}** operator.

The starting point is an **already partially-optimized {{LANGUAGE_NAME}} kernel**. Analyze it,
identify bottlenecks, and optimize for maximum speedup while preserving correctness.

The pure-Python reference implementation is in `docs/definition.json` (under "reference"). Your kernel may be a different implementation approach.

**Recommended first step:** Run `bash scripts/bench.sh "baseline"` to establish current performance.
