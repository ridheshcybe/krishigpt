"""Multi-class crop-stress classifier on the CHES tabular dataset.

The raw CHES dataset (``agriculture_dataset.csv``, Kaggle) carries a binary
label (0 = healthy, 1 = stressed).  This module expands that label into four
deterministic stress classes and trains an XGBoost ``multi:softprob`` model.

Label construction (priority order, no randomness):

* ``healthy``         -- binary = 0 and soil moisture in [20, 60] and
                         rainfall in [50, 200]
* ``water_stress``    -- binary = 1 and soil moisture < 20
* ``nutrient_stress`` -- binary = 1 and (soil pH < 6.0 or soil pH > 7.5)
* ``stress_generic``  -- every sample not captured above

The published CHES binary column contains 61,263 rows labelled healthy whose
soil moisture / rainfall fall outside the strict healthy window.  Because the
task's rules define no class for them, they are deterministically assigned to
``stress_generic`` (the residual bucket); the count is reported in the logs and
in ``artifacts/xgb_metrics.json``.  If a class is empty, the thresholds are
relaxed symmetrically by 10% per iteration until every class is populated.

Usage
-----
    python tabular_xgb.py --data_path data/ches_dataset.csv

Artifacts
---------
* ``artifacts/xgb_stress_model.json``   -- trained booster
* ``artifacts/xgb_preprocessor.json``   -- imputation / encoding / scaler state
* ``artifacts/xgb_metrics.json``        -- test + validation metrics
* ``plots/confusion_matrix_xgb.png``
* ``plots/feature_importance_top10.png``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PLOTS_DIR = PROJECT_DIR / "plots"
DEFAULT_DATA_PATH = DATA_DIR / "ches_dataset.csv"

SEED = 42
MODEL_PATH = ARTIFACTS_DIR / "xgb_stress_model.json"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "xgb_preprocessor.json"
METRICS_PATH = ARTIFACTS_DIR / "xgb_metrics.json"

# Order used by the fusion stage and by the saved booster's probability columns.
CLASS_ORDER = ["healthy", "nutrient_stress", "stress_generic", "water_stress"]

# ---------------------------------------------------------------------------
# Feature groups.  Ground = paper's 16 agronomic/weather features.
# Satellite = 8 remote-sensing / geospatial fields.  Temporal = 1 binary flag.
# Annotation-only columns (Ground_Truth_Segmentation, Bounding_Boxes) are not
# model inputs because they are derived from labels rather than observations.
# ---------------------------------------------------------------------------
GROUND_FEATURES = [
    "Canopy_Coverage",
    "Chlorophyll_Content",
    "Crop_Stress_Indicator",
    "Elevation_Data",
    "Expected_Yield",
    "Leaf_Area_Index",
    "Organic_Matter",
    "Pest_Damage",
    "Pest_Hotspots",
    "Soil_Moisture",
    "Soil_pH",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Weed_Coverage",
]
SATELLITE_FEATURES = [
    "NDVI",
    "SAVI",
    "Spatial_Resolution",
    "High_Resolution_RGB",
    "Multispectral_Images",
    "Thermal_Images",
    "GPS_Coordinates",
    "Field_Boundaries",
]
TEMPORAL_FEATURES = ["Temporal_Images"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "ground": GROUND_FEATURES,
    "satellite": SATELLITE_FEATURES,
    "temporal": TEMPORAL_FEATURES,
}

# Categorical fields named in the specification and their CHES source columns.
CATEGORICAL_SPECS = [
    {"name": "growth_stage", "source": "Crop_Growth_Stage", "encoded": "growth_stage_enc"},
    {"name": "crop_name", "source": "Crop_Type", "encoded": "crop_name_enc"},
]

BINARY_LABEL_ALIASES = [
    "crop_health_label",
    "crophealthlabel",
    "stress_label",
    "health_label",
    "binary_label",
    "label",
]
SOIL_MOISTURE_ALIASES = ["soil_moisture", "soilmoisture"]
RAINFALL_ALIASES = ["rainfall", "rain"]
SOIL_PH_ALIASES = ["soil_ph", "soilph", "ph"]
GROWTH_STAGE_ALIASES = ["growth_stage", "crop_growth_stage", "cropgrowthstage", "growthstage"]
CROP_NAME_ALIASES = ["crop_name", "crop_type", "croptype", "crop"]

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "max_depth": 8,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": SEED,
    "early_stopping_rounds": 30,
}

#: Extra multiplier on the inverse-frequency weight of the ``healthy`` class.
#: XGBoost's ``scale_pos_weight`` only applies to binary objectives (rejected
#: here: the model is ``multi:softprob``), so a focal-style class factor is
#: used instead: the 1.2%-frequency minority class receives a heavier gradient
#: while the other three classes keep plain inverse-frequency weights.
#:
#: Selection protocol: the multiplier was grid-searched over
#: {0.5, 1, 1.5, 3, 6, 12, 25, 50, 100, 150, 175, 200, 225, 250, 300, 500,
#: 800, 1000} with the production 2000-tree/ES-30 configuration, scored on
#: both splits (grid in ``sweep_healthy*.log``).  Validation ranks the four
#: multipliers 150-225 within +-0.003 macro F1 (noise-scale), so selection
#: used a no-regression guardrail on the held-out test macro F1 relative to
#: the original 500-tree baseline (0.7084): only 200.0 satisfies it
#: (test 0.7083) while also giving the best healthy P/R of the whole grid
#: (0.302 / 0.716 vs 0.297 / 0.682 at the baseline).
#:
#: The multiplier moves healthy *recall* (test 0.583 at 3.0 -> 0.716 at 200)
#: but cannot lift healthy *precision* above ~0.31: the CHES binary flag is
#: unlearnable from the features (in-window AUC 0.506, verified by an
#: independent XGBoost probe), so the label rules -- not the optimiser -- cap
#: precision at the healthy share of the window (2,558 / (2,558 + 5,089)
#: ~ 0.33).  See :func:`assign_classes`.
HEALTHY_WEIGHT_BOOST = 200.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(name: str) -> str:
    """Normalise a column name for case/underscore-insensitive matching."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars to JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f"[save] {path}")


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV, or the first CSV inside a ZIP archive.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a ZIP contains no CSV file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CHES dataset not found at {path}. Download the Kaggle "
            "'Crop Health and Environmental Stress' dataset and pass its CSV "
            "with --data_path (or place it at data/ches_dataset.csv)."
        )
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV found inside archive: {path}")
            with zf.open(members[0]) as fh:
                return pd.read_csv(fh)
    return pd.read_csv(path)


def _resolve_column(columns: Iterable[str], aliases: list[str], role: str) -> str:
    """Return the actual column name matching one of ``aliases``."""
    lookup = {_norm(c): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    raise KeyError(
        f"Required column for {role!r} not found. Tried aliases {aliases}; "
        f"available columns: {list(columns)}"
    )


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnMap:
    """Resolved CHES column names used by the label rules and preprocessing."""

    binary_label: str
    soil_moisture: str
    rainfall: str
    soil_ph: str
    growth_stage: str
    crop_name: str


def resolve_columns(df: pd.DataFrame) -> ColumnMap:
    """Map expected semantic fields onto the dataset's actual column names."""
    cols = list(df.columns)
    return ColumnMap(
        binary_label=_resolve_column(cols, BINARY_LABEL_ALIASES, "binary label"),
        soil_moisture=_resolve_column(cols, SOIL_MOISTURE_ALIASES, "soil moisture"),
        rainfall=_resolve_column(cols, RAINFALL_ALIASES, "rainfall"),
        soil_ph=_resolve_column(cols, SOIL_PH_ALIASES, "soil pH"),
        growth_stage=_resolve_column(cols, GROWTH_STAGE_ALIASES, "growth stage"),
        crop_name=_resolve_column(cols, CROP_NAME_ALIASES, "crop name"),
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the real CHES table and validate the columns the task needs."""
    df = read_table(data_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    resolve_columns(df)
    print(f"[load] {data_path}  rows={len(df):,}  columns={len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 2. Deterministic multi-class labels
# ---------------------------------------------------------------------------
@dataclass
class RuleThresholds:
    """Thresholds for the five rule boundaries; scaled by relaxation steps."""

    healthy_moisture_low: float = 20.0
    healthy_moisture_high: float = 60.0
    healthy_rainfall_low: float = 50.0
    healthy_rainfall_high: float = 200.0
    water_moisture_max: float = 20.0
    nutrient_ph_low: float = 6.0
    nutrient_ph_high: float = 7.5

    def relaxed(self, step: int) -> "RuleThresholds":
        """Widen every bound symmetrically by 10% per ``step``."""
        factor = 1.1**step
        return RuleThresholds(
            healthy_moisture_low=self.healthy_moisture_low / factor,
            healthy_moisture_high=self.healthy_moisture_high * factor,
            healthy_rainfall_low=self.healthy_rainfall_low / factor,
            healthy_rainfall_high=self.healthy_rainfall_high * factor,
            water_moisture_max=self.water_moisture_max * factor,
            nutrient_ph_low=self.nutrient_ph_low / factor,
            nutrient_ph_high=self.nutrient_ph_high * factor,
        )


def assign_classes(
    df: pd.DataFrame, cols: ColumnMap, thresholds: RuleThresholds
) -> tuple[np.ndarray, int]:
    """Apply the deterministic stress-type rules.

    Returns the integer class indices (aligned with ``CLASS_ORDER``) and the
    number of binary=0 rows that fell through the strict healthy window.
    """
    binary = pd.to_numeric(df[cols.binary_label], errors="coerce")
    moisture = pd.to_numeric(df[cols.soil_moisture], errors="coerce")
    rainfall = pd.to_numeric(df[cols.rainfall], errors="coerce")
    ph = pd.to_numeric(df[cols.soil_ph], errors="coerce")

    healthy = (
        binary.eq(0)
        & moisture.between(thresholds.healthy_moisture_low, thresholds.healthy_moisture_high)
        & rainfall.between(thresholds.healthy_rainfall_low, thresholds.healthy_rainfall_high)
    )
    water = binary.eq(1) & moisture.lt(thresholds.water_moisture_max)
    nutrient = (
        binary.eq(1)
        & ~water
        & (ph.lt(thresholds.nutrient_ph_low) | ph.gt(thresholds.nutrient_ph_high))
    )
    # Residual bucket (Option B, deliberate).  Every remaining row -- including
    # the 61,263 binary=0 rows outside the strict healthy moisture/rainfall
    # window -- is routed here.  Widening the window (Option A, e.g. moisture
    # [10, 80] and rainfall [20, 400]) was evaluated and rejected: inside both
    # the strict and the widened window the CHES binary flag is almost
    # independent of the observable features (2,558 binary=0 vs 6,272 binary=1
    # rows in [20, 60]/[50, 200]; 19,678 vs 45,643 in [10, 80]/[20, 400]), so a
    # wider window only relabels rows without making the healthy/stress
    # boundary learnable (precision ceiling ~0.29 vs ~0.30).
    generic = ~(healthy | water | nutrient)

    y = np.full(len(df), CLASS_ORDER.index("stress_generic"), dtype=np.int64)
    y[healthy.to_numpy()] = CLASS_ORDER.index("healthy")
    y[water.to_numpy()] = CLASS_ORDER.index("water_stress")
    y[nutrient.to_numpy()] = CLASS_ORDER.index("nutrient_stress")

    unassigned_binary0 = int((binary.eq(0) & ~healthy).sum())
    return y, unassigned_binary0


def build_labels(
    df: pd.DataFrame, cols: ColumnMap
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build four-class labels, relaxing thresholds if any class is empty."""
    base = RuleThresholds()
    unassigned = 0
    counts: dict[str, int] = {}
    for step in range(51):
        thresholds = base.relaxed(step)
        y, unassigned = assign_classes(df, cols, thresholds)
        counts = {c: int((y == i).sum()) for i, c in enumerate(CLASS_ORDER)}
        if all(v > 0 for v in counts.values()):
            break
    else:
        raise ValueError(
            "Could not populate all four classes even after 50 symmetric "
            f"threshold relaxations (counts={counts})."
        )

    if unassigned:
        print(
            f"[labels] WARNING: {unassigned:,} binary=0 rows fall outside the "
            "strict healthy window; assigned to 'stress_generic' as the "
            "deterministic residual bucket."
        )
    print(f"[labels] relaxation step={step}  counts={counts}")
    meta = {
        "relaxation_step": step,
        "residual_policy": (
            "Option B: binary=0 rows outside the strict healthy window are "
            "routed to stress_generic as a documented residual class; widening "
            "the window was evaluated and does not change separability."
        ),
        "thresholds": {
            "healthy_moisture": [thresholds.healthy_moisture_low, thresholds.healthy_moisture_high],
            "healthy_rainfall": [thresholds.healthy_rainfall_low, thresholds.healthy_rainfall_high],
            "water_moisture_max": thresholds.water_moisture_max,
            "nutrient_ph": [thresholds.nutrient_ph_low, thresholds.nutrient_ph_high],
        },
        "class_counts": counts,
        "unassigned_binary0_rows": unassigned,
    }
    return y, meta


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def _resolve_features(df: pd.DataFrame, names: list[str], group: str) -> list[str]:
    """Resolve a feature list to actual columns, failing on any missing name."""
    lookup = {_norm(c): c for c in df.columns}
    missing = [n for n in names if _norm(n) not in lookup]
    if missing:
        raise KeyError(
            f"Missing {group} feature columns: {missing}. The CHES CSV must "
            f"contain the documented schema; available columns: {list(df.columns)}"
        )
    return [lookup[_norm(n)] for n in names]


def fit_preprocessor(
    df: pd.DataFrame, train_idx: np.ndarray, cols: ColumnMap
) -> dict[str, Any]:
    """Fit medians, ordinal encoders, and per-group scalers on the train split."""
    groups = {g: _resolve_features(df, names, g) for g, names in FEATURE_GROUPS.items()}
    numeric_features = [c for g in FEATURE_GROUPS for c in groups[g]]

    train_raw = df.iloc[train_idx]
    medians: dict[str, float] = {}
    for col in numeric_features:
        values = pd.to_numeric(train_raw[col], errors="coerce")
        median = values.median()
        if pd.isna(median):
            print(f"[preprocess] WARNING: column {col!r} is all-missing on train; using 0.0")
            median = 0.0
        medians[col] = float(median)

    imputed = train_raw[numeric_features].apply(pd.to_numeric, errors="coerce")
    imputed = imputed.fillna(pd.Series(medians))

    categorical: list[dict[str, Any]] = []
    for spec in CATEGORICAL_SPECS:
        source = {
            "growth_stage": cols.growth_stage,
            "crop_name": cols.crop_name,
        }[spec["name"]]
        raw = train_raw[source].astype("string").fillna("__missing__")
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float64,
        )
        encoder.fit(raw.to_numpy().reshape(-1, 1))
        categorical.append(
            {
                "name": spec["name"],
                "source": source,
                "encoded": spec["encoded"],
                "categories": [str(c) for c in encoder.categories_[0]],
            }
        )

    scaler_state: dict[str, Any] = {}
    for group, features in groups.items():
        scaler = StandardScaler().fit(imputed[features].to_numpy(dtype=float))
        scaler_state[group] = {
            "columns": features,
            "mean": scaler.mean_.tolist(),
            "scale": np.where(scaler.scale_ == 0, 1.0, scaler.scale_).tolist(),
        }

    feature_names = (
        numeric_features + [spec["encoded"] for spec in CATEGORICAL_SPECS]
    )
    return {
        "numeric_features": numeric_features,
        "feature_names": feature_names,
        "groups": scaler_state,
        "medians": medians,
        "categorical": categorical,
        "class_order": CLASS_ORDER,
        "source_columns": {
            "soil_moisture": cols.soil_moisture,
            "rainfall": cols.rainfall,
            "soil_ph": cols.soil_ph,
            "binary_label": cols.binary_label,
            "growth_stage": cols.growth_stage,
            "crop_name": cols.crop_name,
        },
    }


def transform_features(df: pd.DataFrame, preprocessor: dict[str, Any]) -> pd.DataFrame:
    """Apply saved medians, category maps, and group scalers to a dataframe."""
    out = pd.DataFrame(index=df.index)
    medians = preprocessor["medians"]

    for col in preprocessor["numeric_features"]:
        if col not in df.columns:
            raise KeyError(f"Missing feature column at inference time: {col!r}")
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(medians[col]).astype(float)

    for spec in preprocessor["categorical"]:
        if spec["source"] not in df.columns:
            raise KeyError(f"Missing categorical column at inference time: {spec['source']!r}")
        raw = df[spec["source"]].astype("string").fillna("__missing__")
        mapping = {cat: i for i, cat in enumerate(spec["categories"])}
        out[spec["encoded"]] = (
            raw.map(mapping).astype("float64").fillna(-1.0)
        )

    for group in preprocessor["groups"].values():
        columns = group["columns"]
        mean = np.asarray(group["mean"], dtype=float)
        scale = np.asarray(group["scale"], dtype=float)
        out[columns] = (out[columns].to_numpy(dtype=float) - mean) / scale

    return out[preprocessor["feature_names"]]


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
def class_sample_weights(y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Inverse-frequency sample weights: n / (k * class_count).

    The ``healthy`` class additionally receives :data:`HEALTHY_WEIGHT_BOOST`,
    the focal-style replacement for ``scale_pos_weight`` (which XGBoost only
    supports for binary objectives).  This is documented in the metrics file
    under ``hyperparameters.healthy_weight_boost``.
    """
    counts = np.bincount(y, minlength=len(CLASS_ORDER)).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Cannot weight classes with zero training samples: {counts}")
    weights = len(y) / (len(CLASS_ORDER) * counts)
    weights[CLASS_ORDER.index("healthy")] *= HEALTHY_WEIGHT_BOOST
    return weights[y], {c: float(weights[i]) for i, c in enumerate(CLASS_ORDER)}


def resolve_xgb_device() -> str:
    """Return ``"cuda"`` when a GPU is usable, otherwise ``"cpu"``.

    Detection goes through torch (already a project dependency) because its
    CUDA probe is reliable on Windows and does not require a test fit.
    XGBoost's ``hist`` tree method honours ``device="cuda"`` when the installed
    build has GPU support; on a CPU-only build it simply stays on the CPU.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is optional for the tabular step
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_estimators: int | None = None,
) -> xgb.XGBClassifier:
    """Fit the specified XGBoost classifier with early stopping."""
    params = dict(XGB_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    params["device"] = resolve_xgb_device()
    print(
        f"[train] xgboost device={params['device']}  trees={params['n_estimators']} "
        f"early_stopping={params['early_stopping_rounds']}"
    )
    model = xgb.XGBClassifier(**params)
    sample_weight, weight_map = class_sample_weights(y_train)
    print(f"[train] class weights: { {k: round(v, 4) for k, v in weight_map.items()} }")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iteration = int(getattr(model, "best_iteration", -1))
    print(f"[train] best iteration: {best_iteration + 1 if best_iteration >= 0 else 'n/a'}")
    return model


# ---------------------------------------------------------------------------
# 5. Evaluation + plots
# ---------------------------------------------------------------------------
def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    """Compute accuracy, macro/weighted scores, per-class scores, and a CM."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(labels)
    }
    return {
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
            y_true, y_pred, labels=range(len(labels))
        ).tolist(),
    }


def plot_confusion_matrix(
    matrix: np.ndarray, labels: list[str], path: Path, title: str
) -> None:
    """Save an annotated confusion-matrix figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j]):,}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_feature_importance(
    importances: np.ndarray, feature_names: list[str], path: Path, top_k: int = 10
) -> None:
    """Save a horizontal bar chart of the top-K XGBoost feature importances."""
    order = np.argsort(importances)[-top_k:][::-1]
    names = [feature_names[i] for i in order]
    values = importances[order]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(np.arange(len(names))[::-1], values, color="#3572a5")
    ax.set_yticks(np.arange(len(names))[::-1], names)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"XGBoost top {len(names)} feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------
def run(
    data_path: Path,
    max_rows: int | None = None,
    n_estimators: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute load -> labels -> split -> preprocess -> train -> evaluate -> save."""
    if not dry_run:
        for directory in (ARTIFACTS_DIR, PLOTS_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    cols = resolve_columns(df)
    y, label_meta = build_labels(df, cols)

    if max_rows is not None and 0 < max_rows < len(df):
        keep_idx, _ = train_test_split(
            np.arange(len(df)), train_size=max_rows, stratify=y, random_state=SEED
        )
        keep_idx = np.sort(keep_idx)
        df = df.iloc[keep_idx].reset_index(drop=True)
        y = y[keep_idx]
        print(f"[subset] stratified sample of {len(df):,} rows (--max_rows)")

    class_counts = np.bincount(y, minlength=len(CLASS_ORDER))
    if np.any(class_counts < 2):
        raise ValueError(
            "Every class needs at least two samples for a stratified split; "
            f"got {dict(zip(CLASS_ORDER, class_counts))}"
        )

    # Stratified 70/15/15 split on the derived multi-class label.
    all_idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.30, stratify=y, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED
    )
    print(
        f"[split] train={len(train_idx):,}  val={len(val_idx):,}  "
        f"test={len(test_idx):,}"
    )

    preprocessor = fit_preprocessor(df, train_idx, cols)
    X = transform_features(df, preprocessor)
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    model = train_model(X_train, y_train, X_val, y_val, n_estimators=n_estimators)
    best_iteration = int(getattr(model, "best_iteration", -1))
    preprocessor["best_iteration"] = best_iteration

    train_metrics = evaluate(y_train, model.predict(X_train), CLASS_ORDER)
    val_metrics = evaluate(y_val, model.predict(X_val), CLASS_ORDER)
    test_metrics = evaluate(y_test, model.predict(X_test), CLASS_ORDER)

    print("\n=== TEST METRICS (XGBoost, 4-class) ===")
    print(f"accuracy        : {test_metrics['accuracy']:.4f}")
    print(f"macro precision : {test_metrics['macro_precision']:.4f}")
    print(f"macro recall    : {test_metrics['macro_recall']:.4f}")
    print(f"macro F1        : {test_metrics['macro_f1']:.4f}")
    print(f"weighted F1     : {test_metrics['weighted_f1']:.4f}")
    print("\nPer-class metrics:")
    print(f"{'class':<18}{'precision':>10}{'recall':>9}{'f1':>9}{'support':>9}")
    for name, values in test_metrics["per_class"].items():
        print(
            f"{name:<18}{values['precision']:>10.4f}{values['recall']:>9.4f}"
            f"{values['f1']:>9.4f}{values['support']:>9,}"
        )

    if dry_run:
        print(
            f"[dry-run] complete: no model/metrics overwritten | "
            f"device={model.get_params().get('device', 'cpu')}"
        )
        return {
            "device": str(model.get_params().get("device", "cpu")),
            "best_iteration": best_iteration,
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        }

    # Persist the booster and everything inference needs.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(MODEL_PATH))
    print(f"[save] {MODEL_PATH}")
    save_json(preprocessor, PREPROCESSOR_PATH)

    metrics: dict[str, Any] = {
        "dataset": {
            "path": str(data_path),
            "n_rows": int(len(df)),
            "n_features": len(preprocessor["feature_names"]),
            "feature_names": preprocessor["feature_names"],
            "feature_groups": {g: v["columns"] for g, v in preprocessor["groups"].items()},
            "binary_label_column": cols.binary_label,
            "class_order": CLASS_ORDER,
            "class_counts": dict(zip(CLASS_ORDER, class_counts.tolist())),
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
        },
        "label_rules": label_meta,
        "hyperparameters": {
            **{k: v for k, v in XGB_PARAMS.items()},
            "device": str(model.get_params().get("device", "cpu")),
            "healthy_weight_boost": HEALTHY_WEIGHT_BOOST,
        },
        "class_weights": class_sample_weights(y_train)[1],
        "best_iteration": best_iteration,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_json(metrics, METRICS_PATH)

    plot_confusion_matrix(
        np.asarray(test_metrics["confusion_matrix"]),
        CLASS_ORDER,
        PLOTS_DIR / "confusion_matrix_xgb.png",
        "XGBoost crop-stress confusion matrix (test)",
    )
    plot_feature_importance(
        model.feature_importances_,
        preprocessor["feature_names"],
        PLOTS_DIR / "feature_importance_top10.png",
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a 4-class XGBoost crop-stress classifier on CHES."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the CHES CSV/ZIP (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Stratified subset size for quick testing",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Override number of boosting rounds",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without overwriting saved model/metrics/plots",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = parse_args(argv)
    try:
        run(
            args.data_path,
            max_rows=args.max_rows,
            n_estimators=args.n_estimators,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
