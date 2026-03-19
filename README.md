# kernel-opt-agent

A framework for automated GPU kernel optimization using Claude Code. Provide any kernel and any benchmark script — Claude Code analyzes them, creates an isolated optimization environment, and iteratively improves the kernel.

## Prerequisites

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)

## Quick Start

```bash
cd kernel-opt-agent
claude
```

Claude Code will ask you for:
1. **Kernel path** — file or directory containing the kernel to optimize
2. **Bench script path** — the script that benchmarks the kernel
3. **Hints** (optional) — optimization hints or constraints

It then analyzes both files, creates an isolated child environment, and tells you to:

```bash
cd ../kernel-opt-agent-run-xxx
claude
```

Session 2 reads the generated `CLAUDE.md` and optimizes the kernel iteratively.

## How It Works

**Session 1** (this repo): Claude Code reads `CLAUDE.md`, gathers inputs, analyzes kernel + bench script, and creates a child environment as a sibling directory.

**Session 2** (child env): Claude Code reads the child's `CLAUDE.md`, runs benchmarks, modifies the kernel, and iterates until no further improvements are found.

### Child Environment Structure

```
kernel-opt-agent-run-xxx/
├── CLAUDE.md                   # Generated task spec (self-contained)
├── HINTS.md                    # Optimization hints
├── .gitignore
├── solution/                   # Kernel files (editable by agent)
├── bench/                      # Bench script + deps (read-only)
├── scripts/
│   └── bench.sh                # Benchmark wrapper with trajectory tracking
└── .claude/
    └── settings.local.json     # Agent permissions
```

Each `bash scripts/bench.sh "label"` run saves a snapshot to `trajectory/` for tracking progress.

## Customization

| File | Purpose |
|------|---------|
| `templates/agent/claude.json` | Agent permissions for child environments |
| `templates/hints.md` | Default hints template (copied when no hints provided) |
| `templates/bench-wrapper.sh` | Bench wrapper template with trajectory tracking |
| `templates/task.md` | Reference example of a child CLAUDE.md |
