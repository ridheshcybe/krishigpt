"""Validation pipeline for the multi-modal crop-stress system.

This module checks that the real datasets and trained artifacts are present and
usable, rebuilds the *same* stratified test splits used during training, and
evaluates all three models on them:

* XGBoost (4-class tabular) -- accuracy, macro/weighted scores, confusion
  matrix, top-10 feature importances;
* MobileNetV2 (5-class leaf) -- accuracy, per-class scores, binary
  healthy-vs-disease accuracy and healthy recall;
* rule-based fusion -- run on the image test set with tabular probabilities
  simulated from the XGBoost confusion matrix via Dirichlet noise (the approach
  used in the paper), including per-sample rule attribution.

Nothing here fabricates data: a missing dataset or artifact raises
``ValidationError`` with instructions.  Reporting lives in
``generate_report.py``; the CLI lives in ``main.py``.

Usage
-----
    python -c "from validate_models import run_validation; run_validation()"
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion_inference import (
    UNIFIED_CLASSES,
    fuse,
    load_image_bundle,
    load_tabular_bundle,
    predict_tabular_probs,
)
from mobilenet_cnn import (
    BATCH_SIZE,
    CLASSES as IMAGE_CLASSES,
    IMAGE_SUFFIXES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    METRICS_PATH as CNN_METRICS_PATH,
    PaddyDataset,
    _class_from_dir,
    build_transforms,
    scan_images,
    stratified_split,
)
from tabular_xgb import (
    ARTIFACTS_DIR,
    CLASS_ORDER as XGB_CLASSES,
    DEFAULT_DATA_PATH,
    XGB_PARAMS,
    METRICS_PATH as XGB_METRICS_PATH,
    MODEL_PATH as XGB_MODEL_PATH,
    PREPROCESSOR_PATH,
    build_labels,
    evaluate,
    read_table,
    resolve_columns,
)

PROJECT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_DIR / "plots"
DEFAULT_IMAGE_DIR = PROJECT_DIR / "data" / "paddy_images"
REPORT_PATH = PROJECT_DIR / "report.txt"
VALIDATION_JSON = ARTIFACTS_DIR / "validation_results.json"

SEED = 42
IMAGE_BATCH_SIZE = BATCH_SIZE
NUM_WORKERS = 0 if os.name == "nt" else 4

#: Dirichlet concentration (sum of alpha) for the simulated tabular
#: probabilities.  25 keeps the simulated vectors close to the confusion-matrix
#: row while still producing confident and ambiguous samples.
DIRICHLET_CONCENTRATION = 25.0

#: Core CHES columns the validation insists on, by semantic role.
REQUIRED_CHES_COLUMNS: dict[str, tuple[str, ...]] = {
    "soil_moisture": ("soil_moisture", "soilmoisture"),
    "soil_ph": ("ph", "soil_ph", "soilph"),
    "temperature": ("temperature", "temp"),
    "humidity": ("humidity", "relative_humidity"),
    "rainfall": ("rainfall", "rain"),
    "wind_speed": ("wind_speed", "windspeed"),
    "ndvi": ("ndvi",),
    "savi": ("savi",),
}

BINARY_LABEL_ALIASES = (
    "crop_health_label",
    "crophealthlabel",
    "stress_label",
    "health_label",
    "binary_label",
    "label",
)
BINARY_TEXT_VALUES = {
    "healthy": 0,
    "health": 0,
    "0": 0,
    "stressed": 1,
    "stress": 1,
    "diseased": 1,
    "1": 1,
}


class ValidationError(RuntimeError):
    """Raised when required data, artifacts, or inputs are missing/invalid."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    """Print ``message`` prefixed with a local timestamp."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _norm(name: str) -> str:
    """Normalise a column name for case/underscore-insensitive matching."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _match_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    """Return the first column whose normalised name matches an alias."""
    lookup = {_norm(column): column for column in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


# ---------------------------------------------------------------------------
# 1. Dataset validation
# ---------------------------------------------------------------------------
def validate_ches_dataset(path: Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load and validate the real CHES tabular dataset.

    Args:
        path: CSV or ZIP containing the CHES table.

    Returns:
        The dataframe (with the binary label normalised to 0/1) and the
        resolved column names for the required fields.

    Raises:
        ValidationError: if the file is missing, empty, lacks required
            columns, or has a label that is not binary.
    """
    if not path.exists():
        raise ValidationError(
            f"Download CHES dataset from Kaggle first (expected {path})."
        )

    try:
        df = read_table(path)
    except Exception as exc:  # noqa: BLE001 - surface unreadable files clearly
        raise ValidationError(f"Could not read CHES dataset {path}: {exc}") from exc

    if df.empty:
        raise ValidationError(f"CHES dataset {path} contains no rows.")

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for role, aliases in REQUIRED_CHES_COLUMNS.items():
        column = _match_column(df.columns, aliases)
        if column is None:
            missing.append(role)
        else:
            resolved[role] = column

    label_column = _match_column(df.columns, BINARY_LABEL_ALIASES)
    if label_column is None:
        missing.append("binary stress label (0/1 or healthy/stressed)")
    else:
        resolved["label"] = label_column

    if missing:
        raise ValidationError(
            "CHES dataset is missing required columns: "
            f"{missing}. Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df[resolved["label"]] = normalise_binary_label(df[resolved["label"]], resolved["label"])

    try:
        resolve_columns(df)
    except KeyError as exc:
        raise ValidationError(
            f"CHES dataset lacks a column needed by the label rules: {exc}"
        ) from exc

    log(f"CHES data OK: {len(df):,} rows x {len(df.columns)} columns ({path})")
    return df, resolved


def normalise_binary_label(series: pd.Series, name: str) -> np.ndarray:
    """Return a 0/1 integer label array, accepting numeric or text labels.

    Args:
        series: raw label column.
        name: column name, used in error messages.

    Returns:
        Integer array with 0 = healthy and 1 = stressed.

    Raises:
        ValidationError: if values are not binary.
    """
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        text = series.astype("string").str.strip().str.lower()
        mapped = text.map(BINARY_TEXT_VALUES)
        if mapped.isna().any() and text.notna().any():
            numeric = pd.to_numeric(series, errors="coerce")
        else:
            numeric = mapped.astype("float64")

    if numeric.isna().any():
        raise ValidationError(
            f"Binary label column {name!r} has missing/unparseable values "
            f"({int(numeric.isna().sum()):,} row(s))."
        )
    unique = set(np.unique(numeric.astype("int64")).tolist())
    if not unique.issubset({0, 1}):
        raise ValidationError(
            f"Label column {name!r} is not binary (found values {sorted(unique)})."
        )
    return numeric.astype("int64").to_numpy()


def ches_statistics(
    df: pd.DataFrame, resolved: dict[str, str], y: np.ndarray, label_meta: dict[str, Any]
) -> dict[str, Any]:
    """Summarise the CHES dataset: sizes, class distribution, missing values."""
    label_column = resolved["label"]
    binary_counts = {
        "0_healthy": int((df[label_column] == 0).sum()),
        "1_stressed": int((df[label_column] == 1).sum()),
    }
    class_counts = {
        name: int((y == index).sum()) for index, name in enumerate(XGB_CLASSES)
    }

    missing_pct = df.isna().mean().mul(100.0)
    missing_columns = {
        str(column): round(float(value), 4)
        for column, value in missing_pct.items()
        if value > 0
    }

    stats = {
        "path": str(DEFAULT_DATA_PATH),
        "n_rows": int(len(df)),
        "n_columns": int(df.columns.shape[0]),
        "label_column": label_column,
        "binary_counts": binary_counts,
        "class_counts": class_counts,
        "class_share": {
            name: round(count / max(len(df), 1), 4)
            for name, count in class_counts.items()
        },
        "missing_columns": missing_columns,
        "n_columns_with_missing": len(missing_columns),
        "overall_missing_pct": round(float(df.isna().to_numpy().mean() * 100.0), 4),
        "label_rules": label_meta,
    }

    log(
        f"CHES stats: {stats['n_rows']:,} rows, classes={class_counts}, "
        f"missing columns={len(missing_columns)}"
    )
    return stats


def validate_paddy_dataset(image_dir: Path = DEFAULT_IMAGE_DIR) -> dict[str, Any]:
    """Validate the real paddy-leaf image dataset and summarise its classes.

    Args:
        image_dir: dataset root; class folders may sit directly under it or
            under ``train`` / ``val`` / ``test`` split folders.

    Returns:
        Statistics dictionary with class folder paths, counts, and totals.

    Raises:
        ValidationError: if the directory is missing, a class folder is
            absent, or no readable image exists.
    """
    if not image_dir.is_dir():
        raise ValidationError(
            "Download Plant Stress Identification Paddy Leaves dataset from "
            f"Kaggle first (expected {image_dir})."
        )

    folders: dict[str, list[str]] = {name: [] for name in IMAGE_CLASSES}
    for dirpath, _dirnames, _filenames in os.walk(image_dir):
        canonical = _class_from_dir(Path(dirpath).name)
        if canonical is None:
            continue
        folders[canonical].append(str(Path(dirpath).relative_to(image_dir)))

    missing = [name for name in IMAGE_CLASSES if not folders[name]]
    if missing:
        raise ValidationError(
            "Plant Stress Identification dataset is incomplete: missing class "
            f"folder(s) {missing} under {image_dir}. Expected the five classes "
            f"{IMAGE_CLASSES} (bacterial_leaf_blight/BLB, blast, healthy, "
            "hispa, leaf_spot)."
        )

    samples = scan_images(image_dir)
    counts = np.bincount(
        [label for _path, label in samples], minlength=len(IMAGE_CLASSES)
    )
    total_files = sum(
        1
        for folder in folders.values()
        for relative in folder
        for child in (image_dir / relative).iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )
    corrupted = total_files - len(samples)

    stats = {
        "root": str(image_dir),
        "class_folders": {name: folders[name] for name in IMAGE_CLASSES},
        "class_counts": {
            name: int(counts[index]) for index, name in enumerate(IMAGE_CLASSES)
        },
        "total_images": int(len(samples)),
        "corrupted_skipped": int(max(corrupted, 0)),
        "img_size": IMG_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }
    log(
        f"Paddy images OK: {stats['total_images']:,} readable images, "
        f"counts={stats['class_counts']}"
    )
    return stats


def print_dataset_statistics(ches: dict[str, Any], paddy: dict[str, Any]) -> None:
    """Print sample counts, class distributions, missing values, and splits."""
    print()
    log("Dataset statistics")
    print(
        f"  CHES: {ches['n_rows']:,} rows x {ches['n_columns']} columns, "
        f"{ches['overall_missing_pct']:.2f}% missing "
        f"({ches['n_columns_with_missing']} column(s) affected)"
    )
    print(
        f"    binary {ches['label_column']}: "
        f"0={ches['binary_counts']['0_healthy']:,} "
        f"1={ches['binary_counts']['1_stressed']:,}"
    )
    for name, count in ches["class_counts"].items():
        share = count / max(ches["n_rows"], 1) * 100.0
        print(f"    {name:<18}{count:>9,}  {share:5.2f}%")
    if ches["missing_columns"]:
        for column, percentage in ches["missing_columns"].items():
            print(f"    missing: {column:<24}{percentage:>7.3f}%")
    else:
        print("    missing: no column has missing values")
    tabular_splits = ches.get("split_sizes", {})
    if tabular_splits:
        print(
            "    split 70/15/15: "
            f"train={tabular_splits['train']:,} val={tabular_splits['val']:,} "
            f"test={tabular_splits['test']:,}"
        )

    print(f"  Paddy images: {paddy['total_images']:,} readable "
          f"({paddy['corrupted_skipped']} corrupted skipped)")
    for name, count in paddy["class_counts"].items():
        print(f"    {name:<18}{count:>9,}")
    image_splits = paddy.get("split_sizes", {})
    if image_splits:
        print(
            "    split 70/15/15: "
            f"train={image_splits['train']:,} val={image_splits['val']:,} "
            f"test={image_splits['test']:,}"
        )
    print()


# ---------------------------------------------------------------------------
# 2. Test-split reconstruction (same seeds/labels as training)
# ---------------------------------------------------------------------------
def tabular_test_split(df: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    """Rebuild the 70/15/15 stratified tabular split used during training.

    Args:
        df: validated CHES dataframe.
        seed: split seed used during training.

    Returns:
        Dictionary with ``y`` (4-class labels), ``test_index``, ``label_meta``,
        and ``split_sizes``.

    Raises:
        ValidationError: if a class is too small for a stratified split.
    """
    cols = resolve_columns(df)
    y, label_meta = build_labels(df, cols)

    class_counts = np.bincount(y, minlength=len(XGB_CLASSES))
    if np.any(class_counts < 2):
        raise ValidationError(
            "Every class needs at least two samples for a stratified split; "
            f"got {dict(zip(XGB_CLASSES, class_counts.tolist()))}"
        )

    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=seed
    )
    split_sizes = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }
    log(
        "Tabular split reconstructed: "
        f"train={split_sizes['train']:,} val={split_sizes['val']:,} "
        f"test={split_sizes['test']:,}"
    )
    return {
        "y": y,
        "test_index": test_idx,
        "label_meta": label_meta,
        "split_sizes": split_sizes,
    }


def image_test_split(
    image_dir: Path, seed: int = SEED
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Rebuild the 70/15/15 image split and return ``(train, val, test)``."""
    samples = scan_images(image_dir)
    train, val, test = stratified_split(samples, seed=seed)
    log(
        f"Image split reconstructed: train={len(train):,} val={len(val):,} "
        f"test={len(test):,} of {len(samples):,} images"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# 3a. XGBoost validation
# ---------------------------------------------------------------------------
def feature_importances(
    booster: Any, feature_names: list[str], top_k: int = 10
) -> list[dict[str, Any]]:
    """Return the top-``k`` gain-based importances, normalised to sum to 1."""
    gain = booster.get_score(importance_type="gain")
    names = list(booster.feature_names) if booster.feature_names else []
    if len(names) != len(feature_names):
        names = [f"f{i}" for i in range(len(feature_names))]

    values = np.zeros(len(feature_names), dtype=float)
    for key, value in gain.items():
        index: int | None = None
        if key in names:
            index = names.index(key)
        elif key.startswith("f") and key[1:].isdigit():
            index = int(key[1:])
        if index is not None and 0 <= index < len(values):
            values[index] = float(value)

    total = float(values.sum())
    if total <= 0:
        values = np.ones_like(values)
        total = float(len(values))

    order = np.argsort(values)[-min(top_k, len(values)):][::-1]
    return [
        {
            "feature": feature_names[index],
            "gain": float(values[index]),
            "share": float(values[index] / total),
        }
        for index in order
    ]


def validate_tabular(
    df: pd.DataFrame, y: np.ndarray, test_idx: np.ndarray
) -> dict[str, Any]:
    """Evaluate the saved XGBoost booster on the tabular test split."""
    if not XGB_MODEL_PATH.exists():
        raise ValidationError(
            f"XGBoost model missing: {XGB_MODEL_PATH}. Run "
            "'python tabular_xgb.py --data_path data/ches_dataset.csv' first."
        )
    if not PREPROCESSOR_PATH.exists():
        raise ValidationError(
            f"XGBoost preprocessor missing: {PREPROCESSOR_PATH}. Retrain the "
            "tabular model to regenerate it."
        )

    bundle = load_tabular_bundle(XGB_MODEL_PATH, PREPROCESSOR_PATH)
    test_frame = df.iloc[test_idx]
    probabilities = predict_tabular_probs(bundle, test_frame)
    y_true = y[test_idx]
    y_pred = probabilities.argmax(axis=1)
    class_order = list(bundle["class_order"])

    metrics = evaluate(y_true, y_pred, class_order)
    healthy_index = class_order.index("healthy")
    binary = binary_metrics(
        (y_true != healthy_index).astype(int),
        (y_pred != healthy_index).astype(int),
    )

    preprocessor = bundle["preprocessor"]
    importances = feature_importances(
        bundle["booster"], list(preprocessor["feature_names"])
    )

    saved_match: bool | None = None
    hyperparameters: dict[str, Any] = dict(XGB_PARAMS)
    if XGB_METRICS_PATH.exists():
        with XGB_METRICS_PATH.open("r", encoding="utf-8") as fh:
            saved = json.load(fh)
        saved_accuracy = float(saved.get("test", {}).get("accuracy", float("nan")))
        saved_match = bool(abs(saved_accuracy - metrics["accuracy"]) < 1e-9)
        if not saved_match:
            log(
                "WARNING: recomputed XGBoost accuracy "
                f"{metrics['accuracy']:.6f} != saved {saved_accuracy:.6f}"
            )
        hyperparameters = dict(saved.get("hyperparameters", hyperparameters))

    log(
        f"XGBoost: n={len(test_idx):,} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    return {
        "n_test": int(len(test_idx)),
        "class_order": class_order,
        "metrics": metrics,
        "binary": binary,
        "importances": importances,
        "best_iteration": int(preprocessor.get("best_iteration", -1)),
        "hyperparameters": hyperparameters,
        "n_features": len(preprocessor["feature_names"]),
        "feature_names": list(preprocessor["feature_names"]),
        "reproduces_saved_metrics": saved_match,
    }


# ---------------------------------------------------------------------------
# 3b. MobileNetV2 validation
# ---------------------------------------------------------------------------
def build_image_test_loader(
    image_dir: Path,
    batch_size: int = IMAGE_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
    test_samples: list[tuple[Path, int]] | None = None,
    split_sizes: dict[str, int] | None = None,
) -> tuple[DataLoader, list[str], np.ndarray, dict[str, int]]:
    """Build a test-split DataLoader and return ``(loader, paths, labels, sizes)``.

    Args:
        image_dir: paddy dataset root.
        batch_size: inference batch size.
        num_workers: DataLoader workers.
        seed: split seed used during training.
        test_samples: pre-computed test samples (skips re-scanning the dataset).
        split_sizes: pre-computed split sizes, stored in the metrics.
    """
    if test_samples is None or split_sizes is None:
        train_samples, val_samples, test_samples = image_test_split(image_dir, seed=seed)
        split_sizes = {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        }
    _train_transform, eval_transform = build_transforms()
    loader = DataLoader(
        PaddyDataset(test_samples, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    paths = [
        str(path.relative_to(image_dir)) if path.is_relative_to(image_dir) else str(path)
        for path, _label in test_samples
    ]
    labels = np.asarray([label for _path, label in test_samples], dtype=np.int64)
    return loader, paths, labels, dict(split_sizes)


def predict_cnn(
    loader: DataLoader, bundle: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred, probabilities)`` for the image test loader."""
    model = bundle["model"]
    device = bundle["device"]
    model.eval()

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="[cnn] test batches", unit="batch"):
            images = images.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1).cpu().numpy()
            all_prob.append(probabilities)
            all_pred.append(probabilities.argmax(axis=1))
            all_true.append(labels.numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def validate_image(
    image_dir: Path,
    batch_size: int = IMAGE_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    test_samples: list[tuple[Path, int]] | None = None,
    split_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate the saved MobileNetV2 checkpoint on the image test split.

    Args:
        image_dir: paddy dataset root.
        batch_size: inference batch size.
        num_workers: DataLoader workers.
        test_samples: optional pre-computed test samples.
        split_sizes: optional pre-computed split sizes.
    """
    bundle = load_image_bundle()
    classes = list(bundle["classes"])
    if classes != IMAGE_CLASSES:
        raise ValidationError(
            f"Checkpoint classes {classes} do not match the dataset classes "
            f"{IMAGE_CLASSES}."
        )

    loader, paths, labels, split_sizes = build_image_test_loader(
        image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        test_samples=test_samples,
        split_sizes=split_sizes,
    )
    y_true, y_pred, probabilities = predict_cnn(loader, bundle)
    if not np.array_equal(y_true, labels):
        raise ValidationError("DataLoader order does not match the sample labels.")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(classes)
    }
    healthy_index = classes.index("healthy")
    binary = binary_metrics(
        (y_true != healthy_index).astype(int),
        (y_pred != healthy_index).astype(int),
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )[2]
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(classes))
        ).tolist(),
        "binary_healthy_vs_disease": binary,
    }

    saved_match: bool | None = None
    history: list[dict[str, Any]] = []
    hyperparameters: dict[str, Any] = {}
    if CNN_METRICS_PATH.exists():
        with CNN_METRICS_PATH.open("r", encoding="utf-8") as fh:
            saved = json.load(fh)
        saved_accuracy = float(saved.get("test", {}).get("accuracy", float("nan")))
        saved_match = bool(abs(saved_accuracy - metrics["accuracy"]) < 1e-9)
        if not saved_match:
            log(
                "WARNING: recomputed CNN accuracy "
                f"{metrics['accuracy']:.6f} != saved {saved_accuracy:.6f}"
            )
        history = list(saved.get("history", []))
        hyperparameters = dict(saved.get("hyperparameters", {}))

    log(
        f"MobileNetV2: n={len(y_true):,} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} binary_acc={binary['accuracy']:.4f}"
    )
    return {
        "n_test": int(len(y_true)),
        "split_sizes": split_sizes,
        "classes": classes,
        "labels": y_true,
        "device": str(bundle["device"]),
        "metrics": metrics,
        "binary": binary,
        "paths": paths,
        "probabilities": probabilities,
        "history": history,
        "hyperparameters": hyperparameters,
        "reproduces_saved_metrics": saved_match,
    }


# ---------------------------------------------------------------------------
# 3c. Fusion validation
# ---------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy, macro F1, and per-class recalls for a 0/1 problem."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )[2]
        ),
        "healthy_recall": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "stress_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "support_healthy": int((y_true == 0).sum()),
        "support_stress": int((y_true == 1).sum()),
    }


def simulate_tabular_probabilities(
    confusion: list[list[int]] | np.ndarray,
    image_is_healthy: np.ndarray,
    tabular_classes: list[str],
    healthy_index: int,
    seed: int = SEED,
    concentration: float = DIRICHLET_CONCENTRATION,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Simulate 4-class tabular probabilities for each image-test sample.

    The tabular and image datasets are not paired, so (following the paper) the
    tabular model's behaviour on the image test set is simulated from its own
    test confusion matrix:

    * a healthy leaf draws from the confusion matrix's true-healthy row;
    * a diseased leaf draws from one of the stressed rows, chosen from the
      marginal distribution of true stressed classes;
    * the row yields a categorical draw of the tabular prediction, which is
      then perturbed with Dirichlet noise, so the simulated probabilities
      reproduce the model's per-class error rates.

    Args:
        confusion: 4x4 XGBoost test confusion matrix.
        image_is_healthy: boolean mask marking healthy image-test samples.
        tabular_classes: tabular class order of the confusion matrix.
        healthy_index: index of ``healthy`` in ``tabular_classes``.
        seed: RNG seed for reproducibility.
        concentration: Dirichlet concentration (alpha sums to this value).

    Returns:
        ``(probabilities, metadata)`` with shape ``(n_samples, 4)``.
    """
    image_is_healthy = np.asarray(image_is_healthy, dtype=bool)
    matrix = np.asarray(confusion, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(f"Confusion matrix must be square, got {matrix.shape}")
    if matrix.shape[0] != len(tabular_classes):
        raise ValidationError(
            f"Confusion matrix has {matrix.shape[0]} rows but "
            f"{len(tabular_classes)} tabular classes."
        )
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValidationError("Confusion matrix has an all-zero row.")
    matrix = matrix / row_sums

    stress_indices = [i for i in range(len(tabular_classes)) if i != healthy_index]
    stress_marginal = matrix[stress_indices].mean(axis=0)[stress_indices]
    stress_marginal = stress_marginal / stress_marginal.sum()

    rng = np.random.default_rng(seed)
    n_classes = len(tabular_classes)
    probabilities = np.zeros((len(image_is_healthy), n_classes), dtype=float)
    predicted_counts = np.zeros(n_classes, dtype=int)
    for index, is_healthy in enumerate(image_is_healthy):
        if is_healthy:
            share = matrix[healthy_index]
        else:
            row = stress_indices[int(rng.choice(len(stress_indices), p=stress_marginal))]
            share = matrix[row]
        predicted = int(rng.choice(n_classes, p=share))
        predicted_counts[predicted] += 1
        # Most Dirichlet mass on the simulated prediction, the rest spread
        # according to that row of the confusion matrix.
        alpha = share * concentration * 0.2 + 0.05
        alpha[predicted] += concentration * 0.8
        probabilities[index] = rng.dirichlet(alpha)

    metadata = {
        "method": (
            "categorical draw from the XGBoost test confusion matrix, then "
            "Dirichlet noise around the drawn class"
        ),
        "seed": int(seed),
        "concentration": float(concentration),
        "healthy_row_class": tabular_classes[healthy_index],
        "simulated_prediction_counts": {
            name: int(predicted_counts[i]) for i, name in enumerate(tabular_classes)
        },
        "confusion_matrix": matrix.tolist(),
        "note": (
            "Tabular probabilities on the image test set are simulated; the "
            "two datasets are not paired."
        ),
    }
    return probabilities, metadata


def validate_fusion(
    xgb_confusion: list[list[int]],
    xgb_classes: list[str],
    image_probs: np.ndarray,
    image_true: np.ndarray,
    image_classes: list[str],
    image_paths: list[str],
    seed: int = SEED,
) -> dict[str, Any]:
    """Run the rule-based fusion on the image test set with simulated tabular data."""
    healthy_index = image_classes.index("healthy")
    tabular_probs, simulation = simulate_tabular_probabilities(
        xgb_confusion,
        image_true == healthy_index,
        xgb_classes,
        xgb_classes.index("healthy"),
        seed=seed,
    )

    usage = {f"R{i}": 0 for i in range(1, 7)}
    per_sample: list[dict[str, Any]] = []
    fused_labels: list[str] = []
    for index in tqdm(range(len(image_true)), desc="[fusion] samples", unit="sample"):
        result = fuse(
            tabular_probs[index], xgb_classes, image_probs[index], image_classes
        )
        usage[result["rule"]] += 1
        fused_labels.append(result["prediction"])
        per_sample.append(
            {
                "index": int(index),
                "image": image_paths[index] if index < len(image_paths) else "",
                "true_class": image_classes[int(image_true[index])],
                "cnn_prediction": image_classes[int(image_probs[index].argmax())],
                "tabular_prediction": xgb_classes[int(tabular_probs[index].argmax())],
                "fused_prediction": result["prediction"],
                "rule": result["rule"],
            }
        )

    y_bin = (image_true != healthy_index).astype(int)
    cnn_bin = (image_probs.argmax(axis=1) != healthy_index).astype(int)
    tabular_bin = (tabular_probs.argmax(axis=1) != xgb_classes.index("healthy")).astype(int)
    fused_bin = np.asarray([0 if label == "healthy" else 1 for label in fused_labels])

    result = {
        "n": int(len(image_true)),
        "unified_classes": UNIFIED_CLASSES,
        "simulation": simulation,
        "rule_usage": usage,
        "rule_usage_share": {
            rule: round(count / max(len(image_true), 1), 4)
            for rule, count in usage.items()
        },
        "tabular_only_simulated": binary_metrics(y_bin, tabular_bin),
        "cnn_only": binary_metrics(y_bin, cnn_bin),
        "fusion": binary_metrics(y_bin, fused_bin),
        "fused_label_counts": {
            name: int(sum(1 for label in fused_labels if label == name))
            for name in UNIFIED_CLASSES
        },
        "per_sample": per_sample,
    }
    log(
        f"Fusion: n={result['n']:,} accuracy={result['fusion']['accuracy']:.4f} "
        f"cnn_only={result['cnn_only']['accuracy']:.4f} usage={usage}"
    )
    return result


# ---------------------------------------------------------------------------
# 4. Orchestration
# ---------------------------------------------------------------------------
def run_validation(
    data_path: Path = DEFAULT_DATA_PATH,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    batch_size: int = IMAGE_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    save_results: bool = True,
) -> dict[str, Any]:
    """Run the full data + model validation sequence.

    Args:
        data_path: CHES CSV/ZIP path.
        image_dir: paddy-leaf dataset root.
        batch_size: image inference batch size.
        num_workers: DataLoader workers.
        save_results: write ``artifacts/validation_results.json``.

    Returns:
        Nested dictionary with dataset stats and all model metrics.
    """
    started = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    log("=" * 72)
    log("Starting full validation")
    log("=" * 72)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- data ------------------------------------------------------------
    df, resolved = validate_ches_dataset(data_path)
    paddy = validate_paddy_dataset(image_dir)

    split = tabular_test_split(df)
    y, test_idx = split["y"], split["test_index"]
    ches = ches_statistics(df, resolved, y, split["label_meta"])
    ches["path"] = str(data_path)
    ches["split_sizes"] = split["split_sizes"]

    image_train, image_val, image_test = image_test_split(image_dir)
    image_sizes = {
        "train": len(image_train),
        "val": len(image_val),
        "test": len(image_test),
    }
    paddy["split_sizes"] = image_sizes
    print_dataset_statistics(ches, paddy)

    # ---- models ----------------------------------------------------------
    tabular = validate_tabular(df, y, test_idx)
    image = validate_image(
        image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        test_samples=image_test,
        split_sizes=image_sizes,
    )

    fusion = validate_fusion(
        tabular["metrics"]["confusion_matrix"],
        tabular["class_order"],
        image["probabilities"],
        image["labels"],
        image["classes"],
        image["paths"],
    )

    comparison = {
        "tabular_test_4class_accuracy": tabular["metrics"]["accuracy"],
        "tabular_test_4class_macro_f1": tabular["metrics"]["macro_f1"],
        "tabular_test_binary_accuracy": tabular["binary"]["accuracy"],
        "image_test_5class_accuracy": image["metrics"]["accuracy"],
        "image_test_5class_macro_f1": image["metrics"]["macro_f1"],
        "image_test_binary_accuracy": image["binary"]["accuracy"],
        "fusion_binary_accuracy": fusion["fusion"]["accuracy"],
        "simulated_tabular_binary_accuracy_on_image_test": fusion[
            "tabular_only_simulated"
        ]["accuracy"],
    }

    runtime = time.perf_counter() - started
    results = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(runtime, 2),
        "data": {"ches": ches, "paddy": paddy},
        "tabular": tabular,
        "image": image,
        "fusion": fusion,
        "comparison": comparison,
    }

    if save_results:
        save_validation_results(results)

    log(f"Validation finished in {runtime:.1f}s")
    return results


def save_validation_results(results: dict[str, Any], path: Path = VALIDATION_JSON) -> Path:
    """Persist validation results as JSON (numpy-aware)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(results), fh, indent=2)
    log(f"Saved validation results -> {path}")
    return path


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas/Path values to JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    try:
        run_validation()
    except ValidationError as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1) from error
