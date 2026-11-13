"""CLI dispatch for workload generators.

Usage:
    python -m workloads.generators sharegpt --model <hf-id> --num-reqs 300 --sps 10 \
        --source <path-or-hf-id> --output workloads/sharegpt-<model>-<n>-sps<r>.jsonl
    python -m workloads.generators routing-capture --model <hf-id> \
        --dataset workloads/<workload>.jsonl --output workloads/<workload>.routing.npz
    python -m workloads.generators routing-synthetic --model <org>/<name> --skew 1.0 \
        --dataset workloads/<workload>.jsonl --output workloads/<workload>.zipf1.npz
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="workloads.generators")
    sub = parser.add_subparsers(dest="generator", required=True)

    sg = sub.add_parser("sharegpt", help="ShareGPT -> LLMServingSim JSONL")
    from workloads.generators.sharegpt import register_args as sg_register
    sg_register(sg)

    # Two routing-trace commands (new), for --expert-routing-policy TRACE:
    rc = sub.add_parser("routing-capture", help="HF model router decisions -> routing trace (.npz)")   # real routing
    rs = sub.add_parser("routing-synthetic", help="Zipf-skewed synthetic routing trace (.npz)")        # made-up skew
    # Imported here, not at the top, so the sharegpt command keeps working without numpy
    from workloads.generators.routing import register_capture_args, register_synthetic_args
    register_capture_args(rc)      # add routing-capture's own flags (--model, --dataset, ...)
    register_synthetic_args(rs)    # add routing-synthetic's flags (--skew, --phase-reqs, ...)

    args = parser.parse_args()

    if args.generator == "sharegpt":
        from workloads.generators.sharegpt import run
        return run(args)
    if args.generator == "routing-capture":
        # Lazy import: run_capture needs torch + transformers, only loaded when used
        from workloads.generators.routing import run_capture
        return run_capture(args)
    if args.generator == "routing-synthetic":
        from workloads.generators.routing import run_synthetic
        return run_synthetic(args)     # numpy only, runs without a GPU

    parser.error(f"Unknown generator: {args.generator}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
