"""Unified model evaluation entrypoint (Stage 1B/1C).

Usage:
    python evaluate_unified.py --mode full
    python evaluate_unified.py --mode quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import attention_fusion


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Unified Multi-Modal Crop Stress Model")
    parser.add_argument("--mode", choices=["full", "quick"], default="full",
                        help="Evaluation mode: full (complete test split) or quick (capped subset)")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Maximum number of test samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation DataLoader")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd_args = ["--eval_only", "--batch_size", str(args.batch_size)]
    if args.mode == "quick" and args.max_test is None:
        cmd_args.extend(["--max_test", "256"])
    elif args.max_test is not None:
        cmd_args.extend(["--max_test", str(args.max_test)])

    return attention_fusion.main(cmd_args)


if __name__ == "__main__":
    raise SystemExit(main())
