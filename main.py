"""Single entrypoint for the multi-modal crop-stress system.

Modes
-----
* ``validate`` (alias ``evaluate``) -- check datasets, load the saved models,
  evaluate them on the held-out test splits, and write ``report.txt`` plus the
  validation plots;
* ``train``     -- retrain the XGBoost and MobileNetV2 models from scratch;
* ``inference`` -- one fused prediction from a tabular row and a leaf image.

Examples
--------
    python main.py --mode validate
    python main.py --mode evaluate      # alias for validate
    python main.py --mode train
    python main.py --mode inference --tabular sample.csv --image leaf.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from fusion_inference import (
    UNIFIED_CLASSES,
    UNIFIED_INDEX,
    fuse,
    load_image_bundle,
    load_tabular_bundle,
    predict_image_probs,
    predict_tabular_probs,
)
from generate_report import generate_all
from mobilenet_cnn import FINETUNE_EPOCHS, HEAD_EPOCHS, train as train_cnn
from tabular_xgb import DEFAULT_DATA_PATH, read_table, run as train_tabular
from validate_models import (
    DEFAULT_IMAGE_DIR,
    ValidationError,
    log,
    run_validation,
    validate_ches_dataset,
    validate_paddy_dataset,
)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_validate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate data, evaluate all three models, and write the report."""
    results = run_validation(
        data_path=args.data_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    report_path, _plot_paths = generate_all(results)

    tabular = results["tabular"]["metrics"]
    image = results["image"]["metrics"]
    fusion = results["fusion"]
    log(
        "Summary: XGBoost 4-class accuracy "
        f"{tabular['accuracy']:.4f} | CNN 5-class accuracy {image['accuracy']:.4f} "
        f"| fusion binary accuracy {fusion['fusion']['accuracy']:.4f}"
    )
    log(f"Report written to {report_path}")
    return results


def run_train(args: argparse.Namespace) -> None:
    """Retrain both models on the real datasets from scratch."""
    validate_ches_dataset(args.data_path)
    validate_paddy_dataset(args.image_dir)

    log("Training XGBoost tabular classifier")
    train_tabular(args.data_path)

    log("Training MobileNetV2 leaf classifier")
    train_cnn(
        image_dir=args.image_dir,
        head_epochs=args.epochs_head,
        finetune_epochs=args.epochs_ft,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    log("Retraining complete; run '--mode validate' to regenerate the report")


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Predict one crop-stress label from a tabular CSV row and a leaf image."""
    if args.tabular is None or args.image is None:
        raise ValidationError(
            "--mode inference requires --tabular <csv> and --image <image>"
        )
    if not args.tabular.exists():
        raise ValidationError(f"Tabular input not found: {args.tabular}")
    if not args.image.exists():
        raise ValidationError(f"Image input not found: {args.image}")

    dataframe = read_table(args.tabular)
    if dataframe.empty:
        raise ValidationError(f"Tabular input {args.tabular} has no rows")
    if not 0 <= args.row_index < len(dataframe):
        raise ValidationError(
            f"--row_index {args.row_index} is outside [0, {len(dataframe) - 1}]"
        )

    tabular_bundle = load_tabular_bundle()
    image_bundle = load_image_bundle()

    row = dataframe.iloc[[args.row_index]]
    tabular_probs = predict_tabular_probs(tabular_bundle, row)[0]
    image_probs = predict_image_probs(image_bundle, args.image)

    result = fuse(
        tabular_probs, tabular_bundle["class_order"], image_probs, image_bundle["classes"]
    )

    print(f"\n[inference] tabular row {args.row_index} of {args.tabular}")
    print(f"{'class':<18}{'probability':>12}")
    for name, value in result["tabular_probs"].items():
        print(f"{name:<18}{value:>12.4f}")
    print(f"\n[inference] image {args.image}")
    print(f"{'class':<18}{'probability':>12}")
    for name, value in result["image_probs"].items():
        print(f"{name:<18}{value:>12.4f}")
    print(f"\n[inference] unified 6-class space")
    print(f"{'class':<18}{'tabular':>10}{'image':>10}")
    for name in UNIFIED_CLASSES:
        index = UNIFIED_INDEX[name]
        print(
            f"{name:<18}{result['tabular_unified'][index]:>10.4f}"
            f"{result['image_unified'][index]:>10.4f}"
        )
    print(f"\n[inference] PREDICTION: {result['prediction']} ({result['rule']})")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for all three modes."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate, train, or run inference for the multi-modal crop-stress "
            "detection system."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "evaluate", "train", "inference"),
        required=True,
        help="validate = full report (evaluate is an alias), train = retrain "
        "both models, inference = single prediction",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CHES CSV/ZIP (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"paddy-leaf dataset root (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument("--tabular", type=Path, default=None, help="CSV row(s) for inference")
    parser.add_argument("--image", type=Path, default=None, help="leaf image for inference")
    parser.add_argument("--row_index", type=int, default=0, help="row of --tabular to use")
    parser.add_argument("--batch_size", type=int, default=32, help="inference batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest on Windows)",
    )
    parser.add_argument(
        "--epochs_head",
        type=int,
        default=HEAD_EPOCHS,
        help="phase-1 epochs when training",
    )
    parser.add_argument(
        "--epochs_ft",
        type=int,
        default=FINETUNE_EPOCHS,
        help="phase-2 epochs when training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested mode and print the total runtime.

    Returns:
        Process exit code (0 = success, 1 = validation/training error).
    """
    args = build_parser().parse_args(argv)
    if args.mode == "evaluate":
        # Documented spelling from the Stage-1 spec; same behaviour as validate.
        args.mode = "validate"
    started = time.perf_counter()
    try:
        if args.mode == "validate":
            run_validate(args)
        elif args.mode == "train":
            run_train(args)
        else:
            run_inference(args)
    except (ValidationError, FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    finally:
        elapsed = time.perf_counter() - started
        print(f"\n[main] mode={args.mode} total runtime: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
