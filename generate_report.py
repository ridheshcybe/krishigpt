"""Report and plot generation for the multi-modal crop-stress validation.

Consumes the dictionary produced by :func:`validate_models.run_validation` and
writes:

* ``report.txt``                                    -- full validation report
* ``plots/tabular_confusion_matrix.png``
* ``plots/image_confusion_matrix.png``
* ``plots/feature_importance.png``
* ``plots/training_curves.png``        (when a training history was saved)
* ``plots/fusion_rule_usage.png``

Usage
-----
    python generate_report.py               # reuse artifacts/validation_results.json
    python generate_report.py --refresh     # re-run validation first
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from validate_models import (
    DEFAULT_DATA_PATH,
    DEFAULT_IMAGE_DIR,
    PLOTS_DIR,
    REPORT_PATH,
    VALIDATION_JSON,
    ValidationError,
    run_validation,
)

PLOT_NAMES = {
    "tabular_confusion": "tabular_confusion_matrix.png",
    "image_confusion": "image_confusion_matrix.png",
    "feature_importance": "feature_importance.png",
    "training_curves": "training_curves.png",
    "fusion_rule_usage": "fusion_rule_usage.png",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _rule(char: str = "-", width: int = 88) -> str:
    """Return a horizontal rule."""
    return char * width


def _heading(title: str) -> str:
    """Return a numbered-section style heading block."""
    return f"\n{_rule()}\n{title}\n{_rule()}"


def _table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    align: Sequence[str] | None = None,
) -> str:
    """Format rows as a fixed-width text table.

    Args:
        headers: column titles.
        rows: cell values (converted to strings).
        align: per-column ``"left"`` / ``"right"``; defaults to left for the
            first column and right for the rest.
    """
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    alignment = list(align) if align else ["left"] + ["right"] * (len(headers) - 1)

    def line(row: Sequence[str]) -> str:
        """Left/right-align one row of cells using the computed widths."""
        parts = [
            cell.rjust(widths[index]) if alignment[index] == "right" else cell.ljust(widths[index])
            for index, cell in enumerate(row)
        ]
        return "  ".join(parts).rstrip()

    body = [line([str(header) for header in headers])]
    body.append("  ".join("-" * width for width in widths))
    body.extend(line(row) for row in rows)
    return "\n".join(body)


def _indent(text: str, spaces: int = 2) -> str:
    """Indent a multi-line block by ``spaces`` columns."""
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in text.splitlines())


def _confusion_table(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> str:
    """Render a confusion matrix as a text table with true/predicted axes."""
    rows = []
    for index, label in enumerate(labels):
        rows.append([label, *[f"{int(value):,}" for value in matrix[index]]])
    return "true \\ pred\n" + _table(["", *labels], rows)


def _format_metrics(block: dict[str, Any]) -> list[str]:
    """Format accuracy / macro / weighted metrics for one model."""
    return [
        f"  accuracy ......... {block.get('accuracy', float('nan')):.4f}",
        f"  macro precision .. {block.get('macro_precision', float('nan')):.4f}",
        f"  macro recall ..... {block.get('macro_recall', float('nan')):.4f}",
        f"  macro F1 ......... {block.get('macro_f1', float('nan')):.4f}",
        f"  weighted F1 ...... {block.get('weighted_f1', float('nan')):.4f}",
    ]


def _format_binary(block: dict[str, Any]) -> list[str]:
    """Format binary healthy-vs-disease metrics."""
    return [
        f"  binary accuracy .. {block.get('accuracy', float('nan')):.4f}",
        f"  binary macro F1 .. {block.get('macro_f1', float('nan')):.4f}",
        f"  healthy recall ... {block.get('healthy_recall', float('nan')):.4f}",
        f"  stress recall .... {block.get('stress_recall', float('nan')):.4f}",
    ]


def _per_class_table(per_class: dict[str, dict[str, Any]]) -> str:
    """Format a per-class precision/recall/F1/support table."""
    return _table(
        ["class", "precision", "recall", "f1", "support"],
        [
            [
                name,
                f"{values['precision']:.4f}",
                f"{values['recall']:.4f}",
                f"{values['f1']:.4f}",
                f"{values['support']:,}",
            ]
            for name, values in per_class.items()
        ],
    )


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def _data_section(results: dict[str, Any]) -> list[str]:
    """Dataset statistics section."""
    ches = results["data"]["ches"]
    paddy = results["data"]["paddy"]
    label_rules = ches.get("label_rules", {})

    lines = [_heading("1. DATASET STATISTICS")]
    lines += [
        f"CHES tabular dataset ({ches['path']})",
        f"  rows ................. {ches['n_rows']:,}",
        f"  columns .............. {ches['n_columns']}",
        f"  binary label column .. {ches['label_column']} "
        f"(0 = {ches['binary_counts']['0_healthy']:,} healthy, "
        f"1 = {ches['binary_counts']['1_stressed']:,} stressed)",
        f"  missing values ....... {ches['overall_missing_pct']:.2f}% overall "
        f"({ches['n_columns_with_missing']} column(s) with missing data)",
    ]
    if ches["missing_columns"]:
        lines.append("  columns with missing data (%):")
        for column, percentage in ches["missing_columns"].items():
            lines.append(f"    {column:<28}{percentage:>8.3f}%")
    else:
        lines.append("  columns with missing data (%): none")
    tabular_splits = ches.get("split_sizes", {})
    if tabular_splits:
        lines.append(
            "  split (70/15/15) ..... "
            f"train {tabular_splits['train']:,}, val {tabular_splits['val']:,}, "
            f"test {tabular_splits['test']:,}"
        )

    lines.append("")
    lines.append("  Multi-class label distribution (deterministic stress-type rules):")
    lines.append(
        _indent(
            _table(
                ["class", "count", "share"],
                [
                    [name, f"{count:,}", f"{ches['class_share'][name] * 100:.2f}%"]
                    for name, count in ches["class_counts"].items()
                ],
            )
        )
    )
    lines.append("")
    lines.append("  Label rules applied:")
    lines.append(
        "    healthy         binary = 0 and soil_moisture in "
        f"{label_rules.get('thresholds', {}).get('healthy_moisture', [20, 60])} "
        "and rainfall in "
        f"{label_rules.get('thresholds', {}).get('healthy_rainfall', [50, 200])}"
    )
    lines.append(
        "    water_stress    binary = 1 and soil_moisture < "
        f"{label_rules.get('thresholds', {}).get('water_moisture_max', 20)}"
    )
    lines.append(
        "    nutrient_stress binary = 1 and soil pH outside "
        f"{label_rules.get('thresholds', {}).get('nutrient_ph', [6.0, 7.5])}"
    )
    lines.append("    stress_generic  remaining samples (residual bucket)")
    lines.append(
        f"    threshold relaxation step used: {label_rules.get('relaxation_step', 0)}; "
        f"binary=0 rows outside the strict healthy window: "
        f"{label_rules.get('unassigned_binary0_rows', 0):,}"
    )
    lines.append(
        "    residual policy: those binary=0 rows are routed to stress_generic "
        "as a documented residual class (Option B). Widening the window was "
        "evaluated and rejected: the CHES binary flag is almost independent of "
        "the features inside either window, so the wider window only relabels "
        "rows and does not make the healthy/stress boundary learnable."
    )

    lines.append("")
    lines.append(f"Paddy leaf dataset ({paddy['root']})")
    lines.append(
        f"  readable images ...... {paddy['total_images']:,} "
        f"(size {paddy['img_size']}x{paddy['img_size']}, "
        f"{paddy['corrupted_skipped']} corrupted skipped)"
    )
    lines.append("  class distribution:")
    lines.append(
        _indent(
            _table(
                ["class", "images", "folders"],
                [
                    [
                        name,
                        f"{count:,}",
                        ", ".join(paddy["class_folders"][name]) or "-",
                    ]
                    for name, count in paddy["class_counts"].items()
                ],
                align=["left", "right", "left"],
            )
        )
    )
    lines.append(
        "  ImageNet normalisation: mean="
        f"{paddy['imagenet_mean']}, std={paddy['imagenet_std']}"
    )
    image_splits = paddy.get("split_sizes", {})
    if image_splits:
        lines.append(
            "  split (70/15/15) ..... "
            f"train {image_splits['train']:,}, val {image_splits['val']:,}, "
            f"test {image_splits['test']:,}"
        )
    return lines


def _architecture_section(results: dict[str, Any]) -> list[str]:
    """Model architecture summary."""
    tabular = results["tabular"]
    image = results["image"]
    xgb_hp = tabular.get("hyperparameters", {})
    cnn_hp = image.get("hyperparameters", {})

    lines = [_heading("2. MODEL ARCHITECTURES")]
    lines += [
        "XGBoost tabular classifier (artifacts/xgb_stress_model.json)",
        f"  objective ............ {xgb_hp.get('objective', 'multi:softprob')}",
        f"  trees (n_estimators) . {xgb_hp.get('n_estimators', 'n/a')}",
        f"  max_depth ............ {xgb_hp.get('max_depth', 'n/a')}",
        f"  learning_rate ........ {xgb_hp.get('learning_rate', 'n/a')}",
        f"  colsample_bytree ..... {xgb_hp.get('colsample_bytree', 'n/a')}",
        f"  subsample ............ {xgb_hp.get('subsample', 'n/a')}",
        "  early stopping ....... patience="
        f"{xgb_hp.get('early_stopping_rounds', 30)} on validation mlogloss",
        f"  best iteration ....... {tabular.get('best_iteration', -1) + 1}",
        f"  tree budget .......... {xgb_hp.get('n_estimators', 'n/a')}",
        f"  device ............... {xgb_hp.get('device', 'cpu')}",
        f"  healthy class weight . inverse-frequency x"
        f"{xgb_hp.get('healthy_weight_boost', 1.0):.2f} "
        "(focal-style; scale_pos_weight is binary-only)",
        f"  input features ....... {tabular.get('n_features', 'n/a')} "
        "(25 scaled features + 2 ordinal encodings)",
        f"  output classes ....... {', '.join(tabular.get('class_order', []))}",
        "",
        "MobileNetV2 image classifier (artifacts/mobilenet_paddy.pth)",
        "  backbone ............. torchvision mobilenet_v2 (ImageNet weights)",
        "  head ................. AdaptiveAvgPool2d(1) -> Flatten -> "
        "Linear(1280, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 5)",
        "  output activations ... logits -- no in-head softmax, because "
        "CrossEntropyLoss consumes logits; softmax is applied at inference "
        "(predict_proba, fusion_inference, validation)",
        "  input ................ 224x224 RGB, ImageNet mean/std normalisation",
        f"  output classes ....... {', '.join(image.get('classes', []))}",
        f"  phase 1 .............. frozen backbone, "
        f"{cnn_hp.get('epochs_head', 'n/a')} epochs",
        f"  phase 2 .............. top {cnn_hp.get('unfrozen_blocks', 'n/a')} "
        f"blocks unfrozen, {cnn_hp.get('epochs_finetune', 'n/a')} epochs",
        f"  optimiser ............ {cnn_hp.get('optimizer', 'Adam')} "
        f"lr={cnn_hp.get('learning_rate', 'n/a')}",
        f"  scheduler ............ {cnn_hp.get('scheduler', 'n/a')}",
        f"  loss ................. {cnn_hp.get('loss', 'CrossEntropyLoss')}",
        f"  batch size ........... {cnn_hp.get('batch_size', 'n/a')}",
        "",
        "Rule-based fusion (fusion_inference.py)",
        "  unified space ........ [healthy, nutrient_stress, water_stress, "
        "disease_stress, pest_stress, stress_generic]",
        "  image-gated order .... the image model arbitrates before any tabular "
        "rule may force a stress label",
        "  R1 ................... image disease_stress > 0.60 -> disease_stress",
        "  R2 ................... image healthy > 0.50 and tabular healthy < 0.40 "
        "-> healthy (override)",
        "  R3 ................... tabular healthy > 0.50 and image healthy > 0.50 "
        "-> healthy",
        "  R4 ................... tabular nutrient_stress > 0.70 and image "
        "disease_stress < 0.50 -> nutrient_stress",
        "  R5 ................... tabular water_stress > 0.70 and image "
        "disease_stress < 0.50 -> water_stress",
        "  R6 ................... tabular argmax, preferring a specific stress class",
    ]
    return lines


def _metric_section(results: dict[str, Any]) -> list[str]:
    """Metrics for all three models."""
    tabular, image, fusion = results["tabular"], results["image"], results["fusion"]
    lines = [_heading("3. MODEL METRICS (held-out test splits)")]

    lines.append(f"3.1 XGBoost tabular model -- {tabular['n_test']:,} test rows")
    lines += _format_metrics(tabular["metrics"])
    lines.append("  binary healthy vs stress:")
    lines += ["  " + line.strip() for line in _format_binary(tabular["binary"])]
    lines.append("  per-class:")
    lines.append(_indent(_per_class_table(tabular["metrics"]["per_class"])))

    lines.append("")
    lines.append(f"3.2 MobileNetV2 image model -- {image['n_test']:,} test images")
    lines += _format_metrics(image["metrics"])
    lines.append("  binary healthy vs disease_stress:")
    lines += ["  " + line.strip() for line in _format_binary(image["binary"])]
    lines.append("  per-class:")
    lines.append(_indent(_per_class_table(image["metrics"]["per_class"])))

    lines.append("")
    lines.append(
        f"3.3 Rule-based fusion -- {fusion['n']:,} image test samples "
        "(tabular probabilities simulated)"
    )
    lines.append("  tabular-only (simulated) healthy vs stress:")
    lines += ["  " + line.strip() for line in _format_binary(fusion["tabular_only_simulated"])]
    lines.append("  CNN-only healthy vs disease:")
    lines += ["  " + line.strip() for line in _format_binary(fusion["cnn_only"])]
    lines.append("  fusion healthy vs disease:")
    lines += ["  " + line.strip() for line in _format_binary(fusion["fusion"])]
    lines.append("  fused label distribution:")
    lines.append(
        _indent(
            _table(
                ["unified class", "predictions"],
                [
                    [name, f"{count:,}"]
                    for name, count in fusion["fused_label_counts"].items()
                ],
            )
        )
    )
    simulation = fusion["simulation"]
    lines.append(
        f"  simulation: {simulation['method']} "
        f"(seed={simulation['seed']}, concentration={simulation['concentration']})"
    )
    simulated = simulation.get("simulated_prediction_counts", {})
    if simulated:
        lines.append(
            "  simulated tabular predictions: "
            + ", ".join(f"{name}={count:,}" for name, count in simulated.items())
        )
    return lines


def _confusion_section(results: dict[str, Any]) -> list[str]:
    """Confusion matrices as text tables."""
    tabular, image = results["tabular"], results["image"]
    lines = [_heading("4. CONFUSION MATRICES (text)")]
    lines.append(f"XGBoost 4-class, n={tabular['n_test']:,}")
    lines.append(
        _confusion_table(tabular["metrics"]["confusion_matrix"], tabular["class_order"])
    )
    lines.append("")
    lines.append(f"MobileNetV2 5-class, n={image['n_test']:,}")
    lines.append(_confusion_table(image["metrics"]["confusion_matrix"], image["classes"]))
    return lines


def _importance_section(results: dict[str, Any]) -> list[str]:
    """Top-10 XGBoost feature importances."""
    lines = [_heading("5. TOP-10 XGBOOST FEATURE IMPORTANCES (gain)")]
    lines.append(
        _table(
            ["rank", "feature", "gain", "share"],
            [
                [
                    index + 1,
                    item["feature"],
                    f"{item['gain']:,.2f}",
                    f"{item['share'] * 100:.2f}%",
                ]
                for index, item in enumerate(results["tabular"]["importances"])
            ],
            align=["right", "left", "right", "right"],
        )
    )
    return lines


def _rule_section(results: dict[str, Any]) -> list[str]:
    """Fusion rule usage counts."""
    fusion = results["fusion"]
    lines = [_heading("6. FUSION RULE USAGE")]
    lines.append(
        _indent(
            _table(
                ["rule", "samples", "share"],
                [
                    [rule, f"{count:,}", f"{fusion['rule_usage_share'][rule] * 100:.2f}%"]
                    for rule, count in fusion["rule_usage"].items()
                ],
            )
        )
    )
    lines.append("")
    lines.append("Per-sample log (first 15 rows; full log in the validation JSON):")
    per_sample = fusion["per_sample"][:15]
    if per_sample:
        lines.append(
            _indent(
                _table(
                    ["#", "image", "true", "cnn", "tabular", "fused", "rule"],
                    [
                        [
                            row["index"],
                            Path(row["image"]).name[:28],
                            row["true_class"],
                            row["cnn_prediction"],
                            row["tabular_prediction"],
                            row["fused_prediction"],
                            row["rule"],
                        ]
                        for row in per_sample
                    ],
                    align=["right", "left", "left", "left", "left", "left", "left"],
                )
            )
        )
    else:
        lines.append("  (no samples recorded)")
    return lines


def _comparison_section(results: dict[str, Any]) -> list[str]:
    """Side-by-side accuracy comparison."""
    comparison = results["comparison"]
    tabular, image, fusion = results["tabular"], results["image"], results["fusion"]
    lines = [_heading("7. COMPARISON: XGBOOST vs CNN vs FUSION")]
    lines.append(
        _table(
            ["evaluation", "n", "accuracy", "macro F1"],
            [
                [
                    "XGBoost, tabular test, 4-class",
                    f"{tabular['n_test']:,}",
                    f"{comparison['tabular_test_4class_accuracy']:.4f}",
                    f"{comparison['tabular_test_4class_macro_f1']:.4f}",
                ],
                [
                    "XGBoost, tabular test, healthy vs stress",
                    f"{tabular['n_test']:,}",
                    f"{comparison['tabular_test_binary_accuracy']:.4f}",
                    f"{tabular['binary']['macro_f1']:.4f}",
                ],
                [
                    "MobileNetV2, image test, 5-class",
                    f"{image['n_test']:,}",
                    f"{comparison['image_test_5class_accuracy']:.4f}",
                    f"{comparison['image_test_5class_macro_f1']:.4f}",
                ],
                [
                    "MobileNetV2, image test, healthy vs disease",
                    f"{image['n_test']:,}",
                    f"{comparison['image_test_binary_accuracy']:.4f}",
                    f"{image['binary']['macro_f1']:.4f}",
                ],
                [
                    "Tabular-only (simulated), image test",
                    f"{fusion['n']:,}",
                    f"{comparison['simulated_tabular_binary_accuracy_on_image_test']:.4f}",
                    f"{fusion['tabular_only_simulated']['macro_f1']:.4f}",
                ],
                [
                    "Fusion (R1-R6), image test, healthy vs disease",
                    f"{fusion['n']:,}",
                    f"{comparison['fusion_binary_accuracy']:.4f}",
                    f"{fusion['fusion']['macro_f1']:.4f}",
                ],
            ],
        )
    )
    delta = fusion["fusion"]["accuracy"] - image["binary"]["accuracy"]
    lines.append("")
    lines.append(
        "Note: the tabular and image test sets are different datasets, so only "
        "rows sharing an evaluation set are directly comparable. The fusion row "
        "uses simulated tabular probabilities (see section 3.3)."
    )
    if delta >= 0:
        tabular_decisions = sum(
            fusion["rule_usage"][rule] for rule in ("R4", "R5", "R6")
        )
        lines.append(
            f"Finding: fusion changes binary accuracy by {delta:+.4f} versus "
            "CNN-only on the image test set; its healthy recall moves from "
            f"{image['binary']['healthy_recall']:.4f} to "
            f"{fusion['fusion']['healthy_recall']:.4f} and its stress recall from "
            f"{image['binary']['stress_recall']:.4f} to "
            f"{fusion['fusion']['stress_recall']:.4f}. "
            f"{tabular_decisions:,} of {fusion['n']:,} samples were decided by a "
            "tabular rule (R4-R6); when that count is zero the image-gated "
            "order has reduced fusion to the CNN decision."
        )
    else:
        lines.append(
            f"Finding: fusion changes binary accuracy by {delta:+.4f} versus "
            "CNN-only on the image test set. The image-gated rules only defer "
            "to the tabular model while image disease_stress < 0.50, so the "
            "residual gap comes from tabular stress calls on leaves the image "
            "model scored below that gate; fusion healthy recall is "
            f"{fusion['fusion']['healthy_recall']:.4f} versus "
            f"{image['binary']['healthy_recall']:.4f} for CNN-only. The fusion "
            "still raises simulated tabular healthy recall from "
            f"{fusion['tabular_only_simulated']['healthy_recall']:.4f} to "
            f"{fusion['fusion']['healthy_recall']:.4f}."
        )
    return lines


def _verification_section(results: dict[str, Any], plot_paths: list[Path]) -> list[str]:
    """Reproduction checks, plot inventory, and known limitations."""
    tabular, image = results["tabular"], results["image"]
    lines = [_heading("8. VERIFICATION AND ARTIFACTS")]
    lines.append(f"  runtime ...................... {results['runtime_seconds']:.1f}s")
    lines.append(f"  started ...................... {results['started_at']}")
    lines.append(f"  finished ..................... {results['finished_at']}")
    lines.append(
        "  test splits .................. rebuilt with the training seeds "
        "(70/15/15 stratified)"
    )
    lines.append(
        "  XGBoost metrics match saved .. "
        f"{_yes_no(tabular.get('reproduces_saved_metrics'))}"
    )
    lines.append(
        "  CNN metrics match saved ...... "
        f"{_yes_no(image.get('reproduces_saved_metrics'))}"
    )
    lines.append(f"  torch device ................. {image.get('device', 'n/a')}")
    lines.append("")
    lines.append("  plots written:")
    for path in plot_paths:
        lines.append(f"    {path}")
    lines.append("")
    lines.append("")
    lines.append("  Limitations:")
    lines.append(
        "    - Fusion tabular inputs are simulated from the XGBoost confusion "
        "matrix (Dirichlet noise) because no paired sensor/leaf dataset exists; "
        "fusion accuracy therefore measures rule arbitration, not field data."
    )
    lines.append(
        "    - The XGBoost 'healthy' class excludes "
        f"{results['data']['ches']['label_rules'].get('unassigned_binary0_rows', 0):,} "
        "binary=0 rows that fall outside the strict moisture/rainfall window; "
        "they are bucketed into stress_generic, which caps healthy precision."
    )
    lines.append(
        "    - The paddy archive contains augmented flips/rotations of the same "
        "leaves, so the CNN score is a within-dataset estimate."
    )
    lines.append(
        "    - XGBoost per-round loss history was not saved, so "
        "training_curves.png shows the MobileNetV2 phases only."
    )
    lines.append(
        "    - MobileNetV2 emits logits (no in-head softmax): CrossEntropyLoss "
        "consumes logits, so softmax is applied at inference "
        "(predict_proba, fusion_inference, validation) instead."
    )
    return lines


def _yes_no(value: Any) -> str:
    """Render an optional boolean."""
    if value is None:
        return "not checked"
    return "yes" if value else "no (metrics differ)"


def build_report(results: dict[str, Any], plot_paths: list[Path]) -> str:
    """Assemble the complete report text.

    Args:
        results: output of :func:`validate_models.run_validation`.
        plot_paths: figures written for this run.

    Returns:
        The report as a single string.
    """
    lines = [
        _rule("="),
        " CROP STRESS DETECTION - VALIDATION REPORT",
        _rule("="),
        f" generated ....... {datetime.now():%Y-%m-%d %H:%M:%S}",
        f" validation ...... {results['started_at']} -> {results['finished_at']} "
        f"({results['runtime_seconds']:.1f}s)",
        " datasets ........ data/ches_dataset.csv, data/paddy_images",
        " artifacts ....... artifacts/xgb_stress_model.json, "
        "artifacts/mobilenet_paddy.pth",
        " mode ............ real data only (no dummy samples)",
    ]
    lines += _data_section(results)
    lines += _architecture_section(results)
    lines += _metric_section(results)
    lines += _confusion_section(results)
    lines += _importance_section(results)
    lines += _rule_section(results)
    lines += _comparison_section(results)
    lines += _verification_section(results, plot_paths)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _annotate_cells(ax: plt.Axes, matrix: np.ndarray) -> None:
    """Write counts into confusion-matrix cells with readable contrast."""
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{int(matrix[row, column]):,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=9,
            )


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    path: Path,
    title: str,
    cmap: str = "Blues",
) -> Path:
    """Save an annotated confusion-matrix figure."""
    matrix_array = np.asarray(matrix)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix_array, interpolation="nearest", cmap=cmap)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    _annotate_cells(ax, matrix_array)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    importances: Sequence[dict[str, Any]], path: Path, top_k: int = 10
) -> Path:
    """Save a horizontal bar chart of the top-K feature importances."""
    items = list(importances)[:top_k]
    if not items:
        raise ValidationError("No feature importances available to plot.")

    names = [item["feature"] for item in items][::-1]
    values = [item["share"] * 100.0 for item in items][::-1]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(np.arange(len(names)), values, color="#3572a5")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title(f"XGBoost top {len(names)} feature importances")
    ax.set_xlim(0, max(values) * 1.12)
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curves(
    history: Sequence[dict[str, Any]], path: Path, xgb_note: str
) -> Path:
    """Save MobileNetV2 loss/accuracy curves (XGBoost history is not stored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    if history:
        phase1_epochs = max(
            (row["epoch"] for row in history if row.get("phase") == 1), default=0
        )
        epochs = [
            row["epoch"] + (0 if row.get("phase") == 1 else phase1_epochs)
            for row in history
        ]
        ax_loss.plot(epochs, [row["train_loss"] for row in history], label="train")
        ax_loss.plot(epochs, [row["val_loss"] for row in history], label="val")
        ax_acc.plot(epochs, [row["train_acc"] for row in history], label="train")
        ax_acc.plot(epochs, [row["val_acc"] for row in history], label="val")
        if phase1_epochs:
            for ax in (ax_loss, ax_acc):
                ax.axvline(phase1_epochs + 0.5, color="grey", linestyle="--", linewidth=1)
            fig.text(
                0.5,
                0.055,
                f"phase 1 (frozen, {phase1_epochs} epochs) | phase 2 (fine-tune)",
                ha="center",
                fontsize=8,
            )
    else:
        ax_loss.text(0.5, 0.5, "no training history saved", ha="center", va="center")
        ax_acc.text(0.5, 0.5, "no training history saved", ha="center", va="center")

    ax_loss.set(xlabel="epoch", ylabel="loss", title="MobileNetV2 loss")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="MobileNetV2 accuracy")
    for ax in (ax_loss, ax_acc):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Training curves (MobileNetV2, 50 epochs)")
    fig.text(0.01, 0.008, xgb_note, fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_fusion_rule_usage(
    usage: dict[str, int], path: Path, total: int | None = None
) -> Path:
    """Save a bar chart of how often each fusion rule fired."""
    rules = sorted(usage)
    counts = [usage[rule] for rule in rules]
    total = total or sum(counts) or 1

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(rules, counts, color="#4a7c59")
    ax.set_ylabel("samples")
    ax.set_title("Fusion rule usage on the image test set")
    ax.grid(axis="y", alpha=0.3)
    for bar, rule in zip(bars, rules):
        share = usage[rule] / total * 100.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{usage[rule]:,}\n({share:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all(
    results: dict[str, Any],
    report_path: Path = REPORT_PATH,
    plots_dir: Path = PLOTS_DIR,
) -> tuple[Path, list[Path]]:
    """Write every requested plot and the text report.

    Args:
        results: output of :func:`validate_models.run_validation`.
        report_path: destination of ``report.txt``.
        plots_dir: directory for the five figures.

    Returns:
        ``(report_path, plot_paths)``.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    tabular, image, fusion = results["tabular"], results["image"], results["fusion"]

    tabular_plot = plot_confusion_matrix(
        tabular["metrics"]["confusion_matrix"],
        tabular["class_order"],
        plots_dir / PLOT_NAMES["tabular_confusion"],
        "XGBoost crop-stress confusion matrix (test)",
        cmap="Blues",
    )
    image_plot = plot_confusion_matrix(
        image["metrics"]["confusion_matrix"],
        image["classes"],
        plots_dir / PLOT_NAMES["image_confusion"],
        "MobileNetV2 paddy-leaf confusion matrix (test)",
        cmap="Greens",
    )
    importance_plot = plot_feature_importance(
        tabular["importances"], plots_dir / PLOT_NAMES["feature_importance"]
    )
    curves_plot = plot_training_curves(
        image.get("history", []),
        plots_dir / PLOT_NAMES["training_curves"],
        xgb_note=(
            "XGBoost: single fit with a "
            f"{tabular.get('hyperparameters', {}).get('n_estimators', 'n/a')}-tree "
            "budget plus early stopping (best iteration "
            f"{tabular.get('best_iteration', -1) + 1}); per-round loss history was "
            "not saved by the training script."
        ),
    )
    rules_plot = plot_fusion_rule_usage(
        fusion["rule_usage"], plots_dir / PLOT_NAMES["fusion_rule_usage"], fusion["n"]
    )

    plot_paths = [
        tabular_plot,
        image_plot,
        importance_plot,
        curves_plot,
        rules_plot,
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(results, plot_paths), encoding="utf-8")
    print(f"[report] {report_path}")
    for path in plot_paths:
        print(f"[plot] {path}")
    return report_path, plot_paths


def load_or_run_validation(
    refresh: bool,
    data_path: Path = DEFAULT_DATA_PATH,
    image_dir: Path = DEFAULT_IMAGE_DIR,
) -> dict[str, Any]:
    """Return cached validation results, or run validation when needed."""
    if not refresh and VALIDATION_JSON.exists():
        with VALIDATION_JSON.open("r", encoding="utf-8") as fh:
            print(f"[report] reusing {VALIDATION_JSON} (use --refresh to re-validate)")
            return json.load(fh)
    return run_validation(data_path, image_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate report.txt and validation plots for the crop-stress models."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-run validation instead of reusing artifacts/validation_results.json",
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = parse_args(argv)
    try:
        results = load_or_run_validation(args.refresh, args.data_path, args.image_dir)
        generate_all(results)
    except ValidationError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
