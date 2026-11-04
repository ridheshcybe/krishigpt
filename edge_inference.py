"""Stage 4.1 - lightweight edge inference (no torch, no torchvision).

    python edge_inference.py --image leaf.jpg --tabular sensors.json --ndvi ndvi.csv

Runs anywhere ``onnxruntime``, ``numpy`` and ``pillow`` are installed -
Raspberry Pi, drone companion computer, mobile/backend service - because it
never imports PyTorch (only ``config.py`` constants and the *saved* preprocessor
JSON are read, both of which are torch-free).

Inputs
------
* ``--image``   any PIL-readable leaf photo (RGB)
* ``--tabular`` a JSON object (``{"soil_moisture": 18.2, ...}``) or a CSV with
                a header row and exactly one data row of raw sensor columns
* ``--ndvi``    a CSV containing an ``ndvi`` column (or a single column) with
                >= 1 values, or a JSON list / ``{"ndvi": [...]}``.  Values are
                padded (edge-repeat) or truncated to ``NDVI_STEPS``.

Preprocessing contract
----------------------
Image:   Resize(224x224, PIL BILINEAR) -> /255 -> ImageNet mean/std   (Stage-1 eval transform)
Tabular: saved medians -> categorical index map -> group z-scores      (artifacts/xgb_preprocessor.json)
NDVI:    float series, PROXY by default (declared, see stage1_config.NDVI_IS_PROXY)

All three are re-implemented *data-driven* from the saved artifacts so this
file stays torch-free; the JSON preprocessor, not this script, is the source of
truth for the tabular recipe.

Output: predicted class, confidence, softmax probabilities for all 6 classes,
and the per-modality attention weights (extractable from ONNX - the model
returns ``(logits, attention)``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Torch-free config access: stage1_config imports numpy only.
import config as cfg

IMAGE_PATH_DEFAULT = None
ONNX_DEFAULT = Path("artifacts/unified_model.onnx")
PREPROCESSOR_PATH = Path("artifacts/xgb_preprocessor.json")

#: Input order of the exported graph (see edge_deploy.py export_onnx).
ATTENTION_MODALITIES = ("image", "tabular", "ndvi")


def die(message: str, hint: str = "") -> None:
    """Print a clear error + fix-it hint, then exit 1 (never a traceback)."""
    print(f"ERROR: {message}", file=sys.stderr)
    if hint:
        print(f"  fix: {hint}", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# 1. Preprocessing (torch-free, mirrors the Stage-1 eval pipeline)
# ---------------------------------------------------------------------------
def preprocess_image(path: Path) -> np.ndarray:
    """PIL image -> (1, 3, 224, 224) float32, same recipe as Stage-1 eval."""
    try:
        from PIL import Image
    except ImportError:
        die("pillow is not installed", "pip install pillow")

    try:
        with Image.open(path) as handle:
            img = handle.convert("RGB")
    except Exception as err:
        die(f"cannot read image {path}: {err}",
            "pass a valid .jpg/.png leaf photo via --image")
    # Resize((224,224)) with PIL BILINEAR = torchvision's PIL eval path.
    img = img.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(img, dtype=np.float32) / 255.0            # HWC, [0,1]
    array = (array - np.asarray(cfg.IMAGENET_MEAN, dtype=np.float32)) / \
        np.asarray(cfg.IMAGENET_STD, dtype=np.float32)
    return array.transpose(2, 0, 1)[None]                          # CHW


def load_tabular_raw(path: Path) -> dict[str, Any]:
    """Read one raw sensor row from JSON or CSV-with-header."""
    if not path.exists():
        die(f"tabular file not found: {path}",
            'example: {"soil_moisture": 18.2, "ph": 6.5, "temperature": 31.0, '
            '"humidity": 78.0, "rainfall": 0.0}')
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if not data:
                die(f"{path} is an empty JSON list")
            data = data[0]
        if not isinstance(data, dict):
            die(f"{path} must be a JSON object of column -> value")
        return data
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        die(f"{path} has no data rows (needs a header row + one data row)")
    if len(rows) > 1:
        print(f"NOTE: {len(rows)} rows in {path}; using the first.", file=sys.stderr)
    return dict(rows[0])


def apply_saved_preprocessor(raw: dict[str, Any],
                             preprocessor: dict[str, Any]) -> np.ndarray:
    """Saved-median impute -> categorical index map -> group z-score.

    Faithful, JSON-driven mirror of ``tabular_xgb.transform_features`` (which
    cannot be imported here because it pulls in xgboost/torch).  The preprocessor
    JSON is the single source of truth shared with training.
    """
    medians = preprocessor["medians"]
    numeric = preprocessor["numeric_features"]
    out: dict[str, float] = {}

    for col in numeric:
        if col not in raw:
            # Row may legitimately omit columns; training used medians there.
            out[col] = float(medians.get(col, 0.0))
            continue
        try:
            value = float(raw[col])
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value):
            value = float(medians[col])
        out[col] = value

    for spec in preprocessor["categorical"]:
        source, encoded = spec["source"], spec["encoded"]
        mapping = {cat: i for i, cat in enumerate(spec["categories"])}
        raw_value = str(raw.get(source, "__missing__"))
        out[encoded] = float(mapping.get(raw_value, -1.0))

    for group in preprocessor["groups"].values():
        columns = group["columns"]
        mean = np.asarray(group["mean"], dtype=np.float64)
        scale = np.asarray(group["scale"], dtype=np.float64)
        values = np.array([out[c] for c in columns], dtype=np.float64)
        scaled = (values - mean) / scale
        for c, v in zip(columns, scaled):
            out[c] = float(v)

    feature_names = preprocessor["feature_names"]
    missing = [f for f in feature_names if f not in out]
    if missing:
        die(f"tabular row is missing derived features: {missing[:5]} ...")
    return np.asarray([out[f] for f in feature_names], dtype=np.float32)[None]


def load_ndvi(path: Path) -> np.ndarray:
    """Read an NDVI series from CSV/JSON and fit it to ``NDVI_STEPS``."""
    if not path.exists():
        die(f"ndvi file not found: {path}",
            "expected a CSV with an 'ndvi' column or a JSON list of values")
    values: list[float] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("ndvi", "values", "series"):
                if key in data:
                    data = data[key]
                    break
            else:
                die(f"{path}: JSON object needs an 'ndvi'/'values' key")
        values = [float(v) for v in data]
    else:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            key = next(
                (c for c in columns if c.strip().lower() in ("ndvi", "value", "values")),
                columns[0] if columns else None,
            )
            if key is None:
                die(f"{path}: no columns found")
            for row in reader:
                try:
                    values.append(float(row[key]))
                except (TypeError, ValueError):
                    continue
        if not values:
            die(f"{path}: no numeric NDVI values found in column {key!r}")

    series = np.asarray(values, dtype=np.float32)
    target = cfg.NDVI_STEPS
    if len(series) >= target:
        series = series[-target:]                       # keep the most recent days
    else:                                               # edge-repeat pad
        series = np.pad(series, (target - len(series), 0), mode="edge")
    return series[None]


def resolve_preprocessor() -> dict[str, Any] | None:
    """Load the saved Stage-0 preprocessor; None if absent (tabular = raw)."""
    if not PREPROCESSOR_PATH.exists():
        print(
            f"WARNING: {PREPROCESSOR_PATH} missing - passing raw sensor values "
            "without median/scaling. Train the Stage-0 model first for parity.",
            file=sys.stderr,
        )
        return None
    return json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 2. ONNX inference
# ---------------------------------------------------------------------------
def run_onnx(model_path: Path, image: np.ndarray, tabular: np.ndarray,
             ndvi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits (1,6), attention (1,3)) from the ONNX graph."""
    try:
        import onnxruntime as ort
    except ImportError:
        die("onnxruntime is not installed", "pip install onnxruntime")

    if not model_path.exists():
        die(f"ONNX model not found: {model_path}",
            "python edge_deploy.py --benchmark   (exports artifacts/unified_model.onnx)")
    try:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        outputs = session.run(None, {
            "images": image, "tabular": tabular, "ndvi": ndvi,
        })
    except Exception as err:
        die(f"ONNX inference failed: {err}",
            "re-export with: python edge_deploy.py --export-only")
    logits = np.asarray(outputs[0], dtype=np.float32)
    attention = np.asarray(outputs[1], dtype=np.float32) if len(outputs) > 1 \
        else np.full((1, 3), np.nan, dtype=np.float32)
    return logits, attention


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the class axis."""
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse edge-inference options."""
    parser = argparse.ArgumentParser(
        description="KisanGPT edge inference (onnxruntime only - no torch)."
    )
    parser.add_argument("--image", type=Path, required=True,
                        help="leaf photo (jpg/png)")
    parser.add_argument("--tabular", type=Path, required=True,
                        help="sensor row: JSON object or CSV header + 1 row")
    parser.add_argument("--ndvi", type=Path, required=True,
                        help="NDVI series: CSV with ndvi column or JSON list")
    parser.add_argument("--model", type=Path, default=ONNX_DEFAULT,
                        help=f"ONNX graph (default: {ONNX_DEFAULT}; "
                             "pass artifacts/unified_model_int8.onnx for INT8)")
    parser.add_argument("--json", action="store_true",
                        help="print the result as JSON instead of a summary")
    parser.add_argument("--top-k", type=int, default=0,
                        help="also print the k most confident classes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: preprocess -> ONNX -> report."""
    args = parse_args(argv)
    started = time.perf_counter()

    if not args.image.exists():
        die(f"image not found: {args.image}",
            "pass a real leaf photo, e.g. --image data/paddy_images/test/xxx.jpg")

    preprocessor = resolve_preprocessor()
    image = preprocess_image(args.image)
    if preprocessor is not None:
        tabular = apply_saved_preprocessor(load_tabular_raw(args.tabular), preprocessor)
    else:
        # Minimal fallback: named core sensors, remaining slots zero-filled.
        raw = load_tabular_raw(args.tabular)
        tabular = np.zeros((1, cfg.TABULAR_DIM), dtype=np.float32)
        for i, key in enumerate(("soil_moisture", "ph", "temperature",
                                 "humidity", "rainfall")):
            if key in raw:
                try:
                    tabular[0, i] = float(raw[key])
                except (TypeError, ValueError):
                    pass
    ndvi = load_ndvi(args.ndvi)

    logits, attention = run_onnx(args.model, image, tabular, ndvi)
    probs = softmax(logits)[0]
    predicted = int(probs.argmax())
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    attention_flat = attention[0]
    total = float(np.nansum(attention_flat))
    attention_map = {
        name: (float(value / total) if total > 0 else None)
        for name, value in zip(ATTENTION_MODALITIES, attention_flat)
    }

    result = {
        "predicted_class": cfg.CLASS_MAP[predicted],
        "confidence": round(float(probs[predicted]), 4),
        "probabilities": {
            cfg.CLASS_MAP[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
        "attention_weights": {
            k: (round(v, 4) if v is not None else None)
            for k, v in attention_map.items()
        },
        "model": str(args.model),
        "ndvi_is_proxy": bool(cfg.NDVI_IS_PROXY),
        "latency_ms": round(elapsed_ms, 1),
        "inputs": {
            "image": str(args.image),
            "tabular": str(args.tabular),
            "ndvi": str(args.ndvi),
        },
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print()
    print("=" * 62)
    print("  KisanGPT edge inference")
    print("=" * 62)
    print(f"  prediction : {result['predicted_class']}")
    print(f"  confidence : {result['confidence'] * 100:.1f}%")
    print(f"  latency    : {result['latency_ms']:.1f} ms (CPU, "
          f"{args.model})")
    print("  attention  : " + ", ".join(
        f"{k}={v:.2f}" if v is not None else f"{k}=n/a"
        for k, v in result["attention_weights"].items()
    ))
    print("  all classes:")
    order = np.argsort(-probs)[: max(args.top_k, 6)]
    for i in order:
        bar = "#" * int(round(probs[i] * 30))
        print(f"    {cfg.CLASS_MAP[i]:<16} {probs[i] * 100:5.1f}%  {bar}")
    if cfg.NDVI_IS_PROXY:
        print("  NOTE: NDVI input treated as PROXY (no real Sentinel-2 series "
              "for this sample).")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
