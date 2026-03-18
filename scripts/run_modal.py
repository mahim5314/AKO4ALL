"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal, with trajectory tracking.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy", "cupti-python")
    .pip_install("flashinfer-bench @ https://github.com/flashinfer-ai/flashinfer-bench/archive/refs/heads/main.tar.gz")
    .pip_install("torch-c-dlpack-ext")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=20, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def find_group_axis() -> str:
    """Find the first variable axis from the definition, for grouping results."""
    def_path = PROJECT_ROOT / "docs" / "definition.json"
    if def_path.exists():
        with open(def_path) as f:
            definition = json.load(f)
        for name, info in definition.get("axes", {}).items():
            if info.get("type") == "var":
                return name
    return ""


def compute_score(results: dict) -> dict:
    """Compute final score and per-group breakdown from benchmark results.

    If ANY workload is not PASSED, the kernel is invalid and final_score=None.
    """
    group_axis = find_group_axis()
    workloads_path = PROJECT_ROOT / "docs" / "workloads.jsonl"
    uuid_to_group = {}
    if workloads_path.exists() and group_axis:
        with open(workloads_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    w = json.loads(line)
                    uuid_to_group[w["workload"]["uuid"]] = w["workload"]["axes"].get(group_axis, "?")

    all_speedups = []
    by_group = {}
    passed = failed = error = 0
    all_valid = True

    for def_name, traces in results.items():
        for uuid, result in traces.items():
            status = result.get("status", "")
            if status == "PASSED" and result.get("speedup_factor") is not None:
                sf = result["speedup_factor"]
                all_speedups.append(sf)
                passed += 1
                grp = uuid_to_group.get(uuid, "?")
                by_group.setdefault(grp, []).append(sf)
            else:
                all_valid = False
                if status == "FAILED":
                    failed += 1
                else:
                    error += 1

    if all_valid and all_speedups:
        final_score = sum(all_speedups) / len(all_speedups)
        group_scores = {g: sum(sfs) / len(sfs) for g, sfs in sorted(by_group.items())}
    else:
        final_score = None
        group_scores = {}

    return {
        "final_score": final_score,
        "group_scores": group_scores,
        "group_axis": group_axis,
        "passed": passed,
        "failed": failed,
        "error": error,
        "total": passed + failed + error,
        "min_speedup": min(all_speedups) if all_speedups else None,
        "max_speedup": max(all_speedups) if all_speedups else None,
    }


def print_results(results: dict, score: dict = None):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

    if score:
        print()
        print("\u2550" * 50)
        if score["final_score"] is not None:
            print(f"  FINAL SCORE (mean speedup): {score['final_score']:.2f}x")
            print(f"  Passed: {score['passed']}/{score['total']} | Min: {score['min_speedup']:.2f}x | Max: {score['max_speedup']:.2f}x")
            group_axis = score.get("group_axis", "group")
            group_str = "  ".join(f"{g}\u2192{avg:.2f}x" for g, avg in score["group_scores"].items())
            print(f"  By {group_axis}:  {group_str}")
        else:
            parts = []
            if score["failed"]:
                parts.append(f"{score['failed']} workloads FAILED")
            if score["error"]:
                parts.append(f"{score['error']} ERROR")
            print(f"  FINAL SCORE: INVALID ({', '.join(parts)})")
            print(f"  All {score['total']} workloads must PASS for a valid score.")
            print(f"  Passed: {score['passed']}/{score['total']}")
        print("\u2550" * 50)


def save_trajectory(results: dict, solution: Solution, score: dict = None, label: str = None):
    """Save kernel and results to trajectory folder."""
    trajectory_dir = PROJECT_ROOT / "trajectory"
    trajectory_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if label:
        folder_name = f"{timestamp}_{label}"
    else:
        folder_name = timestamp

    run_dir = trajectory_dir / folder_name
    run_dir.mkdir(parents=True)

    kernel_path = PROJECT_ROOT / "solution/triton/kernel.py"
    if kernel_path.exists():
        shutil.copy2(kernel_path, run_dir / "kernel.py")
    config_path = PROJECT_ROOT / "config.toml"
    if config_path.exists():
        shutil.copy2(config_path, run_dir / "config.toml")

    trajectory_data = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "backend": "modal-b200",
        "solution_name": solution.name,
        "definition": solution.definition,
        "score": score,
        "results": results,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"\nTrajectory saved to: {run_dir}")


@app.local_entrypoint()
def main(label: str = None):
    """Pack solution and run benchmark on Modal B200, with trajectory tracking."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    score = compute_score(results)
    print_results(results, score)
    save_trajectory(results, solution, score, label)
