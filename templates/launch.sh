#!/bin/bash
# One-click launch for Claude Code kernel optimization
cd "$(dirname "$0")"

claude \
  -p "Read CLAUDE.md and Info.md, then begin optimizing the kernel for maximum speedup." \
  2>&1 | tee run.log
