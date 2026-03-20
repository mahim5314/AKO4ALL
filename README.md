# AutoKernelForge

Automated GPU kernel optimization powered by Claude Code. Provide any kernel and its benchmark script — Claude Code analyzes them and iteratively rewrites the kernel for maximum performance.

Supports Triton, CUDA, C++, TileLang, Python — any kernel that can be benchmarked by a script.

## Prerequisites

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)
- Git
- A working benchmark environment (local GPU, [Modal](https://modal.com/), remote cluster, etc.) — make sure your bench script runs before starting

## Quick Start (no bench script)

If you just want to optimize a kernel without writing a benchmark, leave `bench/` empty and AutoKernelForge will use its built-in KernelBench evaluator (`bench/kernelbench/bench.py`) for correctness checking and performance timing automatically — no external dependencies beyond PyTorch.

### Scenario A: KernelBench-format kernel

Your input already has `class Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`:

```
AutoKernelForge/
├── input/
│   └── matmul.py                # class Model + get_inputs + get_init_inputs
├── bench/                       # Empty — uses KernelBench eval
└── HINTS.md
```

```bash
cd AutoKernelForge && claude
```

Claude detects the empty `bench/`, validates the KernelBench format, copies the kernel to `solution/`, and starts optimizing. The bench command becomes:
```
python bench/kernelbench/bench.py --ref input/matmul.py --solution solution/matmul.py --verbose
```

### Scenario B: Raw kernel (CUDA, Triton, etc.)

Your input is a raw kernel file (not KernelBench format):

```
AutoKernelForge/
├── input/
│   └── kernel.cu               # Raw CUDA kernel
├── bench/                       # Empty — uses KernelBench eval
└── HINTS.md
```

```bash
cd AutoKernelForge && claude
```

Claude detects the raw kernel, wraps it into KernelBench format (`input/kernel_kb.py` with `class Model` calling the CUDA code via `load_inline`, plus `get_inputs()` and `get_init_inputs()`), then proceeds as in Scenario A.

## Quick Start (with bench script)

1. Place your files:

```
AutoKernelForge/
├── input/                       # Your kernel files (required)
│   ├── kernel.py                # The kernel to optimize
│   └── reference.py             # Optional — correctness golden
├── bench/                       # Your benchmark script + deps (required)
│   └── bench.py
└── HINTS.md                     # Optimization hints (edit to add your own)
```

2. Run:

```bash
cd AutoKernelForge
claude
```

Claude Code reads your files, sets up the workspace, and begins optimizing.

If no `reference.py` is provided, the original kernel in `input/` is used as the correctness golden.

### Example

```
> cd AutoKernelForge && claude

Claude: [reads input/, bench/, and HINTS.md]
        [copies kernel to solution/, runs baseline benchmark]
        [rewrites kernel with tiling optimization]
        [benchmarks -> 1.8x speedup, all PASSED]
        [tries vectorized loads -> 2.3x speedup]
        [tries warp-level reduction -> 2.5x speedup, but 1 case FAILED]
        [reverts, tries alternative -> 2.4x, all PASSED]
        Final result: 2.4x mean speedup across all workloads
```

## Workspace Layout

After setup, the full repo looks like:

```
AutoKernelForge/
├── CLAUDE.md
├── input/                      # User-provided originals (read-only)
├── bench/                      # Benchmark script + deps (optional, read-only)
│   └── kernelbench/            # Built-in KernelBench evaluator (used when bench/ has no custom script)
│       ├── bench.py            # Self-contained eval script
│       └── GUIDE.md           # Output format, CLI args, tolerances
├── HINTS.md                    # Optimization hints
├── bench.sh                    # Generated benchmark wrapper (read-only)
├── solution/                   # Kernel files — only these are edited
├── scripts/                    # Workspace for profiling/debug tools
└── trajectory/                 # Auto-created on first benchmark run
    ├── 20260319_143022_baseline/
    │   ├── kernel.py           # Kernel snapshot
    │   └── output.txt          # Benchmark output
    └── ...
```

Every `bash bench.sh "label"` snapshots `solution/` and benchmark output into `trajectory/`.

## Hints

Edit `HINTS.md` in the repo root to add optimization hints or constraints:

```markdown
- Focus on the large-N workloads (N > 4096), they dominate runtime
- Use fp32 accumulation to avoid precision failures
- Try shared memory tiling with tile size 128
- Do not use inline PTX — keep it portable
```

### Web Search

Disabled by default. To enable when the agent hits a plateau, include in your hints:

```markdown
- If 3 consecutive optimization rounds show no speedup improvement, use WebSearch
  to research optimization techniques specific to this kernel type, then apply
  what you find.
```

## FAQ

**What if I don't have a bench script?**
Leave `bench/` empty. AutoKernelForge will use its built-in evaluator (`bench/kernelbench/bench.py`) for correctness checking and performance timing. Your kernel just needs to be in KernelBench format (`class Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`) — or Claude will wrap a raw kernel into that format for you.

**What if the benchmark fails after an optimization?**
The agent reads the failure, attempts fixes, and reverts if needed.

**My bench script uses a remote service (e.g., Modal). Does that work?**
Yes. As long as your bench script runs from the command line and prints results to stdout.

**Can I manually edit the kernel between runs?**
Edit files in `solution/`, then tell Claude to continue.
