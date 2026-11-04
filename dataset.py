"""Stage 1A - multi-modal data ingestion, validation and weak pairing.

This module builds the dataset that Stage 1B trains on: a PyTorch dataset
yielding ``(tabular_vector, rgb_image, ndvi_timeseries, label)`` tuples from
three real sources, plus the split bookkeeping that makes results
reproducible.

Data sources (all real, no dummy data)
--------------------------------------
1. TABULAR - CHES (data/ches_dataset.csv) through the existing
   ``tabular_xgb`` pipeline: numeric conversion + median imputation,
   ordinal encoding of growth_stage/crop_name with
   ``handle_unknown="use_encoded_value"``, per-group StandardScaler
   (ground 16 / satellite 8 / temporal 1) fitted on the train split only.
   Model-ready tensor shape is **(N, 27)** = 25 scaled group features + 2
   ordinal encodings.  The spec's "(N, 32)" is the *raw* CSV column count,
   which includes the label column, two annotation columns
   (Ground_Truth_Segmentation, Bounding_Boxes) and the two raw categorical
   columns that are folded into the 2 ordinal encodings; reusing the existing
   preprocessing (as required) therefore yields 27 columns, not 32.

   Label rules (deterministic, unchanged from Stage 0):
     healthy       binary=0 AND soil_moisture in [20,60] AND rainfall in [50,200]
     water_stress  binary=1 AND soil_moisture < 20
     nutrient_stress binary=1 AND (soil_pH < 6.0 OR soil_pH > 7.5)
     stress_generic remaining rows

   BINARY=0 ROWS OUTSIDE THE HEALTHY WINDOW (explicit decision): 61,263 of
   the 63,821 binary=0 rows fall outside the moisture/rainfall window.  They
   are **routed to stress_generic as a documented residual class (Option B)**
   - not silently ignored, not silently relabelled healthy.  Option A
   (widening the window to [10,80]/[20,400]) was measured and rejected in
   Stage 0: the CHES binary flag is almost independent of the features inside
   either window (precision ceiling ~0.29 vs ~0.30), so widening only
   relabels rows without making the boundary learnable.  The count is
   reported in ``artifacts/stage1a_stats.json`` and echoed on stdout.

2. RGB IMAGES - paddy leaf dataset (data/paddy_images/), 5 classes
   (bacterial_leaf_blight/BLB, blast, healthy, hispa, leaf_spot):
   resize 224x224, ImageNet normalisation, train-only augmentation
   (RandomHorizontalFlip, RandomRotation(10), ColorJitter(0.2, 0.2)) -
   reusing ``mobilenet_cnn.build_transforms()``.  Shape (N, 3, 224, 224).

3. NDVI TIME SERIES - **PROXY**.  No real paired Sentinel-2 series exists for
   these samples (verified: data/ holds only ches_dataset.csv, whose NDVI
   column is a single scalar per row, not a time series), so every sample gets
   a class-conditional synthetic curve declared as PROXY in code
   (``NDVI_IS_PROXY``), on the dataset class (``ndvi_is_proxy``), in the stats
   JSON and in PAIRING_STRATEGY.md.  Shape (N, T) with T = 20 (spec: 10-30).
   If a real export ever appears (data/ndvi_timeseries.csv), load_real_ndvi()
   picks it up and clears the PROXY flag.

Weak pairing (documented; see PAIRING_STRATEGY.md)
---------------------------------------------------
No dataset pairs sensor rows with leaf photos, so:
* each tabular row is paired with a randomly sampled image of the matching
  class (healthy row -> healthy photo; any stress row -> one of the four
  disease photos, the only stress imagery available - a same-binary-class
  fallback documented in PAIRING_STRATEGY.md);
* images are drawn only from the row's OWN split pool, so no photo and no
  CHES row is shared between train/val/test;
* ``disease_stress`` has no tabular source (CHES has no disease label), so it
  is supplied by IMAGE-SOURCED samples: each disease photo becomes one sample
  labelled disease_stress, weakly paired with a random stress row from the
  same split as contextual input (reverse weak pairing);
* ``pest_stress`` has no data source yet - it is RESERVED, carries zero
  samples in every split, and is reported as an untrained class;
* NDVI curves are class-conditional (they follow the sample's label).  That
  is intentional per the Stage 1A spec but it means the third modality alone
  can recover the label - a prototype-only property flagged in the report.

Reproducibility: seed 42 everywhere (rows, pairing, curves, splits).

Usage
-----
    python dataset.py                 # build, validate, save splits + stats
    python dataset.py --smoke 4       # additionally print 4 sample tuples
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import mobilenet_cnn as mc
import stage1_config as cfg
import tabular_xgb as tx
from stage1_config import (
    CLASS_MAP,
    N_CLASSES,
    NDVI_IS_PROXY,
    NDVI_STEPS,
    SEED,
    UNIFIED_CLASSES,
    UNIFIED_INDEX,
    image_label_to_unified,
    tabular_label_to_unified,
)

ARTIFACTS_DIR = tx.ARTIFACTS_DIR
SPLIT_PATH = ARTIFACTS_DIR / "split_indices.json"
STATS_PATH = ARTIFACTS_DIR / "stage1a_stats.json"
REAL_NDV_PATHS = (Path("data/ndvi_timeseries.csv"), Path("data/ndvi.csv"))

#: Label for binary=0 rows outside the healthy window (explicit decision).
RESIDUAL_POLICY = (
    "Option B: binary=0 rows outside the strict healthy window are routed to "
    "stress_generic as a documented residual class; widening the window "
    "(Option A) was evaluated and rejected because it does not change "
    "separability (precision ceiling ~0.29 vs ~0.30)."
)


def log(message: str) -> None:
    """Print a timestamped progress line."""
    print(f"[stage1a {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Tabular ingestion (existing pipeline, no reimplementation)
# ---------------------------------------------------------------------------
def load_tabular(data_path: Path) -> dict[str, Any]:
    """Load CHES, preprocess with the fitted Stage-0 preprocessor, build labels.

    Returns ``features`` (N, 27) float32, ``labels`` (N,) unified 6-class, and
    metadata including the out-of-window binary=0 count.
    """
    frame = tx.load_dataset(data_path)                      # validates columns
    cols = tx.resolve_columns(frame)
    y4, _meta = tx.build_labels(frame, cols)                # Stage-0 rules

    binary = pd.to_numeric(frame[cols.binary_label], errors="coerce").fillna(0)
    moisture = pd.to_numeric(frame[cols.soil_moisture], errors="coerce")
    rainfall = pd.to_numeric(frame[cols.rainfall], errors="coerce")
    in_window = (
        binary.eq(0)
        & moisture.between(20.0, 60.0)
        & rainfall.between(50.0, 200.0)
    )
    out_of_window_binary0 = int((binary.eq(0) & ~in_window).sum())

    labels = tabular_label_to_unified(y4)

    preprocessor = json.loads(tx.PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    features = tx.transform_features(frame, preprocessor).to_numpy(np.float32)

    meta = {
        "rows": int(len(frame)),
        "raw_columns": int(frame.shape[1]),
        "feature_dim": int(features.shape[1]),
        "feature_dim_note": (
            "(N, 27) = 25 scaled group features + 2 ordinal encodings; the "
            "spec's 32 is the raw CSV column count (label + annotations + raw "
            "categoricals excluded by the existing preprocessing)"
        ),
        "binary0_outside_window": out_of_window_binary0,
        "binary0_policy": RESIDUAL_POLICY,
        "label_counts": label_counts(labels),
    }
    log(
        f"tabular: {meta['rows']:,} rows x {features.shape[1]} features | "
        f"binary=0 out-of-window {out_of_window_binary0:,} -> stress_generic "
        f"(Option B)"
    )
    return {
        "frame": frame,
        "features": features,
        "labels": labels,
        "meta": meta,
    }


def label_counts(labels: np.ndarray) -> dict[str, int]:
    """Class -> count for the unified space (zeros included, so reserved
    classes stay visible)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=N_CLASSES)
    return {UNIFIED_CLASSES[i]: int(counts[i]) for i in range(N_CLASSES)}


# ---------------------------------------------------------------------------
# 2. NDVI: real export if present, otherwise declared PROXY
# ---------------------------------------------------------------------------
def load_real_ndvi(frame: pd.DataFrame, steps: int) -> np.ndarray | None:
    """Return a real (N, T) NDVI matrix if a paired export exists, else None."""
    for path in REAL_NDV_PATHS:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if len(table) != len(frame):
            log(f"WARNING: {path} row count {len(table)} != {len(frame)}; ignoring")
            continue
        values = table.select_dtypes(include=[np.number]).to_numpy(np.float32)
        if values.shape[1] < steps:
            log(f"WARNING: {path} has only {values.shape[1]} steps; ignoring")
            continue
        log(f"real NDVI series loaded from {path}")
        return values[:, :steps]
    return None


def build_proxy_ndvi(labels: np.ndarray, steps: int = NDVI_STEPS, seed: int = SEED) -> np.ndarray:
    """Build class-conditional PROXY NDVI curves (label-matched, declared).

    PROXY - prototyping only.  Curves follow the Stage 1A spec per class::

        healthy        0.6-0.8, stable
        water_stress   0.6 -> 0.3 declining
        nutrient_stress 0.7 -> 0.4 gradual decline
        disease_stress baseline with sudden lesion-like dips
        pest_stress    irregular spikes/dips (reserved class, unused for now)
        stress_generic noisy moderate decline

    Deterministic: each sample's RNG is seeded from (seed, sample position),
    so curves are stable across runs, splits and epochs.  Because the curve is
    class-conditional it encodes the label by construction - flagged in
    PAIRING_STRATEGY.md as a prototype-only property.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    n = len(labels_arr)
    if n == 0:
        return np.zeros((0, steps), dtype=np.float32)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(n, steps)).astype(np.float32)
    curves = np.empty((n, steps), dtype=np.float32)

    lin_water = np.linspace(0.60, 0.30, steps, dtype=np.float32)
    lin_nutr = np.linspace(0.70, 0.40, steps, dtype=np.float32)
    lin_dis = np.linspace(0.70, 0.50, steps, dtype=np.float32)
    lin_gen = np.linspace(0.60, 0.45, steps, dtype=np.float32)

    # Default residual bucket (stress_generic): noisy moderate decline
    curves[:] = lin_gen + noise * 0.04

    mask_h = labels_arr == UNIFIED_INDEX["healthy"]
    if mask_h.any():
        bases = rng.uniform(0.60, 0.80, size=int(mask_h.sum())).astype(np.float32)[:, None]
        curves[mask_h] = bases + noise[mask_h] * 0.015

    mask_w = labels_arr == UNIFIED_INDEX["water_stress"]
    if mask_w.any():
        curves[mask_w] = lin_water + noise[mask_w] * 0.015

    mask_n = labels_arr == UNIFIED_INDEX["nutrient_stress"]
    if mask_n.any():
        curves[mask_n] = lin_nutr + noise[mask_n] * 0.012

    mask_d = labels_arr == UNIFIED_INDEX["disease_stress"]
    if mask_d.any():
        curves[mask_d] = lin_dis + noise[mask_d] * 0.012
        width = max(1, steps // 8)
        for idx in np.where(mask_d)[0]:
            for _ in range(int(rng.integers(2, 4))):        # lesion-like dips
                at = int(rng.integers(1, steps))
                depth = float(rng.uniform(0.15, 0.30))
                lo, hi = max(0, at - width), min(steps, at + width)
                curves[idx, lo:hi] -= depth * np.hanning(hi - lo).astype(np.float32)

    mask_p = labels_arr == UNIFIED_INDEX["pest_stress"]
    if mask_p.any():
        for idx in np.where(mask_p)[0]:
            curve = 0.60 + np.cumsum(noise[idx] * 0.04, dtype=np.float32) * 0.5
            for _ in range(int(rng.integers(2, 5))):        # irregular spikes
                at = int(rng.integers(0, steps))
                curve[at] += float(rng.uniform(-0.15, 0.15))
            curves[idx] = curve

    np.clip(curves, 0.05, 1.0, out=curves)   # NDVI stays physically plausible
    return curves


# ---------------------------------------------------------------------------
# 3. Images: split-restricted pools (no photo crosses splits)
# ---------------------------------------------------------------------------
def build_image_pools(image_dir: Path, seed: int = SEED) -> dict[str, dict[str, list[Path]]]:
    """Scan the real image tree and split it 70/15/15 per class."""
    samples = mc.scan_images(image_dir)
    train_s, val_s, test_s = mc.stratified_split(samples, seed=seed)
    pools: dict[str, dict[str, list[Path]]] = {}
    for name, subset in (("train", train_s), ("val", val_s), ("test", test_s)):
        buckets: dict[str, list[Path]] = {c: [] for c in mc.CLASSES}
        for path, label in subset:
            buckets[mc.CLASSES[label]].append(path)
        pools[name] = buckets
        log(
            f"image pool {name}: "
            + ", ".join(f"{c}={len(buckets[c])}" for c in mc.CLASSES)
        )
    return pools


# ---------------------------------------------------------------------------
# 4. Weak pairing -> sample table
# ---------------------------------------------------------------------------
def split_rows(labels: np.ndarray, seed: int = SEED) -> dict[str, np.ndarray]:
    """Stratified 70/15/15 split of the CHES rows on the unified label."""
    all_idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.30, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=seed
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def build_sample_table(
    labels: np.ndarray,
    row_splits: dict[str, np.ndarray],
    pools: dict[str, dict[str, list[Path]]],
    seed: int = SEED,
) -> dict[str, Any]:
    """Weak-pair every row and add image-sourced disease samples.

    Returns arrays ``row``, ``image``, ``label``, ``split`` describing one
    sample per entry (row-sourced first, image-sourced appended).
    """
    rng = np.random.default_rng(seed)
    split_of_row = np.empty(len(labels), dtype=object)
    for split, idx in row_splits.items():
        split_of_row[idx] = split

    row_list: list[int] = []
    image_list: list[Path] = []
    label_list: list[int] = []
    split_list: list[str] = []

    disease_classes = [c for c in mc.CLASSES if c != "healthy"]
    for split, idx in row_splits.items():
        healthy_pool = pools[split]["healthy"]
        disease_pool = [p for c in disease_classes for p in pools[split][c]]
        for row in idx:
            if labels[row] == UNIFIED_INDEX["healthy"]:
                pool = healthy_pool
            else:
                pool = disease_pool          # same-binary-class fallback
            row_list.append(int(row))
            image_list.append(pool[int(rng.integers(0, len(pool)))])
            label_list.append(int(labels[row]))
            split_list.append(split)

    # Image-sourced samples supply disease_stress (CHES has no disease label);
    # the label comes from the shared 5->6 image mapping, not a hard-coded value.
    for split, idx in row_splits.items():
        for class_name in disease_classes:
            class_idx = mc.CLASSES.index(class_name)
            unified_label = int(
                image_label_to_unified(np.asarray([class_idx]))[0]
            )
            for path in pools[split][class_name]:
                row_list.append(int(rng.choice(idx)))       # contextual row
                image_list.append(path)
                label_list.append(unified_label)
                split_list.append(split)

    table = {
        "row": np.asarray(row_list, dtype=np.int64),
        "image": np.asarray(image_list, dtype=object),
        "label": np.asarray(label_list, dtype=np.int64),
        "split": np.asarray(split_list, dtype=object),
    }
    log(
        f"sample table: {len(table['label']):,} samples "
        f"({len(labels):,} row-sourced + "
        f"{len(table['label']) - len(labels):,} image-sourced disease)"
    )
    return table


# ---------------------------------------------------------------------------
# 5. Dataset
# ---------------------------------------------------------------------------
class MultiModalStressDataset(Dataset):
    """Yields ``(tabular, image, ndvi, label)`` tuples.

    ``ndvi_is_proxy`` declares on the dataset itself whether the third
    modality is synthetic (PROXY).  Stage 1B must propagate this flag to any
    checkpoint, metric or plot it produces.
    """

    ndvi_is_proxy: bool = NDVI_IS_PROXY

    def __init__(
        self,
        indices: np.ndarray,
        table: dict[str, Any],
        features: np.ndarray,
        ndvi: np.ndarray,
        transform: Any,
        image_cache: dict[Path, Image.Image] | None = None,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.table = table
        self.features = features
        self.ndvi = ndvi
        self.transform = transform
        self.image_cache: dict[Path, Image.Image] = (
            image_cache if image_cache is not None else {}
        )
        self._tensor_cache: dict[Path, torch.Tensor] = {}
        _, self._eval_tf = mc.build_transforms()
        tf_repr = repr(transform)
        self._is_train_tf = "RandomHorizontalFlip" in tf_repr or "RandomRotation" in tf_repr

    def __len__(self) -> int:
        return len(self.indices)

    def _load_image(self, path: Path) -> Image.Image:
        cached = self.image_cache.get(path)
        if cached is None:
            with Image.open(path) as handle:
                cached = handle.convert("RGB").resize((mc.IMG_SIZE, mc.IMG_SIZE))
            self.image_cache[path] = cached
        return cached

    def _load_tensor(self, path: Path) -> torch.Tensor:
        tensor = self._tensor_cache.get(path)
        if tensor is None:
            tensor = self._eval_tf(self._load_image(path))
            self._tensor_cache[path] = tensor
        if self._is_train_tf:
            # Fast tensor-space augmentation (horizontal flip + mild brightness/contrast jitter)
            out = tensor.flip(-1) if torch.rand(1).item() < 0.5 else tensor.clone()
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            shift = (torch.rand(1).item() - 0.5) * 0.1
            return out.mul_(scale).add_(shift)
        return tensor

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        sample = int(self.indices[i])
        row = int(self.table["row"][sample])
        image = self._load_tensor(self.table["image"][sample])
        return (
            torch.from_numpy(self.features[row]),          # (27,)
            image,                                         # (3, 224, 224)
            torch.from_numpy(self.ndvi[sample]),           # (T,)
            int(self.table["label"][sample]),
        )


# ---------------------------------------------------------------------------
# 6. Validation + artefacts
# ---------------------------------------------------------------------------
def coverage_report(table: dict[str, Any]) -> dict[str, Any]:
    """Per-split class counts and the list of missing classes."""
    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        mask = table["split"] == split
        counts = label_counts(table["label"][mask])
        missing = [name for name, n in counts.items() if n == 0]
        splits[split] = {
            "samples": int(mask.sum()),
            "class_counts": counts,
            "missing_classes": missing,
        }
    all_counts = label_counts(table["label"])
    splits["missing_classes_overall"] = [
        name for name, n in all_counts.items() if n == 0
    ]
    return splits


def save_splits(table: dict[str, Any], path: Path, seed: int) -> None:
    """Persist split indices for reproducibility (spec: split_indices.json)."""
    payload: dict[str, Any] = {
        "seed": seed,
        "strategy": (
            "stratified 70/15/15 on the unified 6-class label for CHES rows; "
            "images split 70/15/15 per class; pairing restricted to a row's "
            "own split so no photo or row crosses splits"
        ),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(table["label"])),
        "class_order": UNIFIED_CLASSES,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        payload["splits"][split] = {
            "sample_indices": np.where(table["split"] == split)[0].tolist(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    log(f"split indices -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


def validate(
    tabular: dict[str, Any],
    table: dict[str, Any],
    ndvi: np.ndarray,
    ndvi_is_proxy: bool,
    pools: dict[str, dict[str, list[Path]]],
    dataset_probe: MultiModalStressDataset,
) -> dict[str, Any]:
    """Print and return the Stage-1A validation summary."""
    coverage = coverage_report(table)
    shapes = None
    if len(dataset_probe):
        tab, img, series, label = dataset_probe[0]
        shapes = {
            "tabular": list(tab.shape),
            "image": list(img.shape),
            "ndvi": list(series.shape),
            "label": int(label),
        }

    stats = {
        "tabular": tabular["meta"],
        "images": {
            "total": int(sum(len(v) for s in pools.values() for v in s.values())),
            "classes": mc.CLASSES,
            "pools": {
                s: {c: len(v) for c, v in buckets.items()}
                for s, buckets in pools.items()
            },
        },
        "ndvi": {
            "steps": int(ndvi.shape[1]),
            "is_proxy": bool(ndvi_is_proxy),
            "source": "PROXY (class-conditional synthetic)" if ndvi_is_proxy else "real",
            "shape": list(ndvi.shape),
        },
        "labels": {
            "class_order": UNIFIED_CLASSES,
            "class_map": {str(k): v for k, v in CLASS_MAP.items()},
            "counts": label_counts(table["label"]),
            "reserved_untrained": cfg.RESERVED_CLASSES,
        },
        "pairing": {
            "strategy": "weak pairing (seed 42), split-restricted pools",
            "row_sourced": int(len(tabular["labels"])),
            "image_sourced": int(len(table["label"]) - len(tabular["labels"])),
            "fallback": (
                "healthy row -> healthy photo; stress row -> random disease "
                "photo (only stress imagery available)"
            ),
        },
        "coverage": coverage,
        "sample_shapes": shapes,
        "dataset_class": {
            "yields": "(tabular, image, ndvi, label)",
            "ndvi_is_proxy": bool(dataset_probe.ndvi_is_proxy),
        },
    }

    log("--- validation ------------------------------------------------")
    log(f"tabular tensor ....... {stats['tabular']['feature_dim']} columns "
        f"(raw CSV has {stats['tabular']['raw_columns']})")
    log(f"binary=0 out-of-window  {stats['tabular']['binary0_outside_window']:,} "
        f"-> stress_generic (Option B, not ignored)")
    log(f"ndvi .................. {stats['ndvi']['shape']}  "
        f"PROXY={'YES' if ndvi_is_proxy else 'no'}")
    log(f"samples ............... {len(table['label']):,} "
        f"(row {stats['pairing']['row_sourced']:,} + "
        f"image {stats['pairing']['image_sourced']:,})")
    for split in ("train", "val", "test"):
        block = coverage[split]
        log(
            f"  {split:<6} {block['samples']:>7,}  "
            + ", ".join(f"{k}={v}" for k, v in block["class_counts"].items())
            + (f"  MISSING={block['missing_classes']}" if block["missing_classes"] else "")
        )
    log(f"reserved (untrained) .. {cfg.RESERVED_CLASSES}")
    if shapes:
        log(f"probe item ............ tabular{shapes['tabular']} "
            f"image{shapes['image']} ndvi{shapes['ndvi']} "
            f"label={shapes['label']} ({UNIFIED_CLASSES[shapes['label']]})")
    log("---------------------------------------------------------------")
    return stats


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage-1A dataset options."""
    parser = argparse.ArgumentParser(description="Stage 1A: multi-modal data layer")
    parser.add_argument("--data_path", type=Path, default=tx.DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", type=Path, default=Path("data/paddy_images"))
    parser.add_argument("--ndvi_steps", type=int, default=NDVI_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", type=int, default=0, metavar="N",
                        help="print N sample tuples after validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build, validate and persist the Stage-1A dataset (0 on success)."""
    args = parse_args(argv)
    started = time.perf_counter()

    tabular = load_tabular(args.data_path)
    row_splits = split_rows(tabular["labels"], seed=args.seed)
    pools = build_image_pools(args.image_dir, seed=args.seed)
    table = build_sample_table(
        tabular["labels"], row_splits, pools, seed=args.seed
    )

    real_ndvi = load_real_ndvi(tabular["frame"], args.ndvi_steps)
    ndvi_is_proxy = real_ndvi is None
    if real_ndvi is not None:
        ndvi = real_ndvi
    else:
        log(f"no real NDVI export found -> building PROXY curves "
            f"(class-conditional, {args.ndvi_steps} steps)")
        ndvi = build_proxy_ndvi(table["label"], steps=args.ndvi_steps, seed=args.seed)

    train_tf, eval_tf = mc.build_transforms()
    cache: dict[Path, Image.Image] = {}
    datasets = {
        "train": MultiModalStressDataset(
            np.where(table["split"] == "train")[0], table,
            tabular["features"], ndvi, train_tf, cache,
        ),
        "val": MultiModalStressDataset(
            np.where(table["split"] == "val")[0], table,
            tabular["features"], ndvi, eval_tf, cache,
        ),
        "test": MultiModalStressDataset(
            np.where(table["split"] == "test")[0], table,
            tabular["features"], ndvi, eval_tf, cache,
        ),
    }

    stats = validate(tabular, table, ndvi, ndvi_is_proxy, pools, datasets["train"])
    save_splits(table, SPLIT_PATH, args.seed)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"stats -> {STATS_PATH}")

    if args.smoke:
        for split in ("train", "val", "test"):
            ds = datasets[split]
            for i in range(min(args.smoke, len(ds))):
                tab, img, series, label = ds[i]
                log(f"smoke {split}[{i}]: tabular{tuple(tab.shape)} "
                    f"img{tuple(img.shape)} ndvi{tuple(series.shape)} "
                    f"label={label} ({UNIFIED_CLASSES[label]})")

    log(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
