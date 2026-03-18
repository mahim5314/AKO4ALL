#!/usr/bin/env python3
"""Generate template variables from an operator's definition.json and workloads.jsonl.

Outputs a shell-sourceable file with key=value pairs for use by spawn_env.sh.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def first_sentence(text: str) -> str:
    """Extract the first sentence from a description."""
    for end in (".  ", ".\n", ". "):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1]
    # If no sentence boundary found, return up to first period
    idx = text.find(".")
    if idx != -1:
        return text[: idx + 1]
    return text


def format_shape(name: str, tensor_info: dict, axes: dict) -> str:
    """Format a single input/output shape string."""
    shape = tensor_info.get("shape")
    dtype = tensor_info.get("dtype", "?")
    if shape is None:
        return f"`{name}` {dtype} scalar"
    # Resolve const axes to values
    resolved = []
    for dim in shape:
        if dim in axes and axes[dim].get("type") == "const":
            resolved.append(str(axes[dim]["value"]))
        else:
            resolved.append(str(dim))
    return f"`{name}=[{','.join(resolved)}]` {dtype}"


def generate_shape_summary(definition: dict) -> str:
    """Generate a human-readable shape summary from definition."""
    axes = definition.get("axes", {})
    inputs = definition.get("inputs", {})
    outputs = definition.get("outputs", {})

    parts = []
    for name, info in inputs.items():
        parts.append(format_shape(name, info, axes))
    parts_str = "Key shapes: " + ", ".join(parts) + "."

    out_parts = []
    for name, info in outputs.items():
        out_parts.append(format_shape(name, info, axes))
    parts_str += "\nOutputs: " + ", ".join(out_parts) + "."

    return parts_str


def generate_workload_summary(workloads: list, axes: dict) -> str:
    """Generate a summary of workload distribution across variable axes."""
    if not workloads:
        return "No workloads found."

    # Find variable axes
    var_axes = [name for name, info in axes.items() if info.get("type") == "var"]
    if not var_axes:
        return f"Workloads: {len(workloads)} total (no variable axes)."

    parts = []
    for axis_name in var_axes:
        values = []
        for w in workloads:
            val = w.get("workload", {}).get("axes", {}).get(axis_name)
            if val is not None:
                values.append(val)
        if not values:
            continue
        counts = Counter(values)
        sorted_vals = sorted(counts.items())
        dist = ",".join(f"{v}(x{c})" for v, c in sorted_vals)
        parts.append(f"{axis_name} in {{{dist}}}")

    # Also include constant axis values from workloads
    const_parts = []
    for axis_name, info in axes.items():
        if info.get("type") == "const":
            continue
        # Already handled above
    # Check for axes that are var but constant across workloads
    first_workload_axes = workloads[0].get("workload", {}).get("axes", {})
    for axis_name in first_workload_axes:
        if axis_name in var_axes:
            continue
        # This is a workload axis that's not in var_axes list - include its value
        val = first_workload_axes[axis_name]
        all_same = all(
            w.get("workload", {}).get("axes", {}).get(axis_name) == val
            for w in workloads
        )
        if all_same:
            const_parts.append(f"{axis_name}={val}")

    summary = "Workloads: " + ", ".join(parts)
    if const_parts:
        summary += ", " + ", ".join(const_parts)
    summary += "."
    return summary


def find_group_axis(axes: dict) -> str:
    """Find the first variable axis name (used for scoring display)."""
    for name, info in axes.items():
        if info.get("type") == "var":
            return name
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate template context from operator definition")
    parser.add_argument("--definition", required=True, help="Path to definition.json")
    parser.add_argument("--workloads", required=True, help="Path to workloads.jsonl")
    parser.add_argument("--output", required=True, help="Output path for shell-sourceable variables file")
    args = parser.parse_args()

    # Load definition
    with open(args.definition) as f:
        definition = json.load(f)

    # Load workloads
    workloads = []
    with open(args.workloads) as f:
        for line in f:
            line = line.strip()
            if line:
                workloads.append(json.loads(line))

    operator = definition["name"]
    description = definition.get("description", operator)
    short_desc = first_sentence(description)
    axes = definition.get("axes", {})
    num_workloads = len(workloads)
    shape_summary = generate_shape_summary(definition)
    workload_summary = generate_workload_summary(workloads, axes)
    group_axis = find_group_axis(axes)

    # Write shell-sourceable output using single quotes to avoid
    # backtick/dollar interpretation. Escape embedded single quotes.
    def shell_quote(s: str) -> str:
        return "'" + s.replace("'", "'\\''") + "'"

    out = Path(args.output)
    with open(out, "w") as f:
        f.write(f'OPERATOR={shell_quote(operator)}\n')
        f.write(f'OPERATOR_DESCRIPTION={shell_quote(short_desc)}\n')
        f.write(f'NUM_WORKLOADS={shell_quote(str(num_workloads))}\n')
        f.write(f'SHAPE_SUMMARY={shell_quote(shape_summary)}\n')
        f.write(f'WORKLOAD_SUMMARY={shell_quote(workload_summary)}\n')
        f.write(f'GROUP_AXIS={shell_quote(group_axis)}\n')

    print(f"Generated context for operator: {operator}", file=sys.stderr)
    print(f"  Description: {short_desc}", file=sys.stderr)
    print(f"  Workloads: {num_workloads}", file=sys.stderr)
    print(f"  Group axis: {group_axis}", file=sys.stderr)


if __name__ == "__main__":
    main()
