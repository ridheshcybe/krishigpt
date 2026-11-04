"""Unified model training entrypoint (Stage 1B).

Usage:
    python train_unified.py --mode train
    python train_unified.py --mode dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import attention_fusion


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Unified Multi-Modal Crop Stress Model")
    parser.add_argument("--mode", choices=["train", "dry-run"], default="train",
                        help="Execution mode: train (full 50 epochs) or dry-run (quick smoke test)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training DataLoader")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Cap number of training samples")
    parser.add_argument("--max-val", type=int, default=None,
                        help="Cap number of validation samples")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Cap batches per epoch")
    parser.add_argument("--stability-tol", type=float, default=None,
                        help="Early exit tolerance when metric changes <= tol (default 0.01 = ±1%%)")
    parser.add_argument("--stability-patience", type=int, default=None,
                        help="Consecutive stable epochs within ±tol before early exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd_args = ["--batch_size", str(args.batch_size)]
    if args.stability_tol is not None:
        cmd_args.extend(["--stability_tol", str(args.stability_tol)])
    if args.stability_patience is not None:
        cmd_args.extend(["--stability_patience", str(args.stability_patience)])
    if args.mode == "dry-run":
        cmd_args.extend([
            "--dry_run",
            "--epochs", str(args.epochs or 2),
            "--limit_batches", str(args.limit_batches or 10),
            "--max_train", str(args.max_train or 320),
            "--max_val", str(args.max_val or 128),
        ])
    else:
        if args.epochs is not None:
            cmd_args.extend(["--epochs", str(args.epochs)])
        if args.limit_batches is not None:
            cmd_args.extend(["--limit_batches", str(args.limit_batches)])
        if args.max_train is not None:
            cmd_args.extend(["--max_train", str(args.max_train)])
        if args.max_val is not None:
            cmd_args.extend(["--max_val", str(args.max_val)])

    return attention_fusion.main(cmd_args)


if __name__ == "__main__":
    raise SystemExit(main())
