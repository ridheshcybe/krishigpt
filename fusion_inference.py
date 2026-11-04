"""Rule-based fusion of the tabular and image crop-stress models.

The XGBoost model produces four probabilities
(``healthy``, ``nutrient_stress``, ``stress_generic``, ``water_stress``); the
MobileNetV2 model produces five leaf-disease probabilities.  Both are projected
into the unified six-class space:

    [healthy, nutrient_stress, water_stress, disease_stress, pest_stress,
     stress_generic]

``pest_stress`` is intentionally never predicted (no pest model exists), so its
probability is always 0.  The image model's four disease classes are summed
into ``disease_stress``.

Image-gated priority rules.  The image model arbitrates first, so a visually
healthy leaf is never relabelled from a (simulated) sensor vector:

* R1 -- image disease_stress > 0.60                    -> disease_stress
* R2 -- image healthy > 0.50 and tabular healthy < 0.40 -> healthy (override)
* R3 -- tabular healthy > 0.50 and image healthy > 0.50 -> healthy
* R4 -- tabular nutrient_stress > 0.70 and image disease_stress < 0.50
                                                        -> nutrient_stress
* R5 -- tabular water_stress > 0.70 and image disease_stress < 0.50
                                                        -> water_stress
* R6 -- otherwise the tabular argmax, preferring a non-``stress_generic`` class

Usage
-----
    python fusion_inference.py --image data/paddy_images/test/healthy/leaf.jpg \
        --tabular data/ches_dataset.csv --row_index 0

``--tabular`` is the documented spelling; ``--data_path`` is kept as an alias
for backwards compatibility.

If ``--image`` is omitted, the first real image in the dataset directory is
used (the chosen path is printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from PIL import Image
from torchvision import transforms

from mobilenet_cnn import (
    CLASSES as IMAGE_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    PaddyMobileNetV2,
    resolve_image_dir,
)
from tabular_xgb import (
    ARTIFACTS_DIR,
    DEFAULT_DATA_PATH,
    PREPROCESSOR_PATH,
    MODEL_PATH as TABULAR_MODEL_PATH,
    load_dataset,
    transform_features,
)

UNIFIED_CLASSES = [
    "healthy",
    "nutrient_stress",
    "water_stress",
    "disease_stress",
    "pest_stress",
    "stress_generic",
]
UNIFIED_INDEX = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

TABULAR_CLASSES = ["healthy", "nutrient_stress", "stress_generic", "water_stress"]
THRESHOLDS = {
    "disease": 0.60,              # R1
    "image_healthy": 0.50,        # R2
    "tabular_healthy_low": 0.40,  # R2
    "both_healthy": 0.50,         # R3
    "nutrient": 0.70,             # R4
    "water": 0.70,                # R5
    "disease_block": 0.50,        # R4/R5 image-side gate
}

CNN_MODEL_PATH = ARTIFACTS_DIR / "mobilenet_paddy.pth"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tabular_bundle(
    model_path: Path = TABULAR_MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
) -> dict[str, Any]:
    """Load the XGBoost booster and its fitted preprocessing state."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model missing: {model_path}. Run "
            "'python tabular_xgb.py --data_path data/ches_dataset.csv' first."
        )
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"XGBoost preprocessor missing: {preprocessor_path}. Retrain the "
            "tabular model to regenerate it."
        )

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    with preprocessor_path.open("r", encoding="utf-8") as fh:
        preprocessor = json.load(fh)

    class_order = preprocessor.get("class_order", TABULAR_CLASSES)
    if len(class_order) != 4:
        raise ValueError(f"Expected four tabular classes, got {class_order}")
    return {
        "booster": booster,
        "preprocessor": preprocessor,
        "class_order": class_order,
    }


def predict_tabular_probs(
    bundle: dict[str, Any], dataframe: pd.DataFrame
) -> np.ndarray:
    """Return 4-class probabilities for every row of ``dataframe``."""
    preprocessor = bundle["preprocessor"]
    features = transform_features(dataframe, preprocessor)
    dmatrix = xgb.DMatrix(features, feature_names=preprocessor["feature_names"])
    best_iteration = int(preprocessor.get("best_iteration", -1))
    if best_iteration >= 0:
        probabilities = bundle["booster"].predict(
            dmatrix, iteration_range=(0, best_iteration + 1)
        )
    else:
        probabilities = bundle["booster"].predict(dmatrix)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    return probabilities


def load_image_bundle(
    model_path: Path = CNN_MODEL_PATH, device: torch.device | None = None
) -> dict[str, Any]:
    """Load the trained MobileNetV2 checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model missing: {model_path}. Run "
            "'python mobilenet_cnn.py --image_dir data/paddy_images' first."
        )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Unexpected CNN checkpoint format: {model_path}")

    classes = list(checkpoint.get("classes", IMAGE_CLASSES))
    model = PaddyMobileNetV2(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return {"model": model, "classes": classes, "device": device, "checkpoint": checkpoint}


def predict_image_probs(
    bundle: dict[str, Any], image_path: Path
) -> np.ndarray:
    """Return 5-class softmax probabilities for one image."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    try:
        with Image.open(image_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
    except Exception as exc:  # noqa: BLE001 - report the offending file
        raise RuntimeError(f"Corrupted or unreadable image {image_path}: {exc}") from exc

    with torch.no_grad():
        logits = bundle["model"](tensor.to(bundle["device"]))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probabilities.astype(float)


# ---------------------------------------------------------------------------
# Label-space alignment
# ---------------------------------------------------------------------------
def tabular_to_unified(probabilities: np.ndarray, classes: list[str]) -> np.ndarray:
    """Project 4-class tabular probabilities into the unified 6-class space."""
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(probabilities) != len(classes):
        raise ValueError(
            f"Tabular probability length {len(probabilities)} != classes {len(classes)}"
        )
    unified = np.zeros(len(UNIFIED_CLASSES), dtype=float)
    for name, probability in zip(classes, probabilities):
        if name not in UNIFIED_INDEX:
            raise ValueError(f"Unknown tabular class: {name!r}")
        unified[UNIFIED_INDEX[name]] += float(probability)
    return unified


def image_to_unified(probabilities: np.ndarray, classes: list[str]) -> np.ndarray:
    """Collapse 5-class leaf predictions into healthy + disease_stress."""
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(probabilities) != len(classes):
        raise ValueError(
            f"Image probability length {len(probabilities)} != classes {len(classes)}"
        )
    if "healthy" not in classes:
        raise ValueError(f"Image classes must contain 'healthy': {classes}")

    unified = np.zeros(len(UNIFIED_CLASSES), dtype=float)
    for name, probability in zip(classes, probabilities):
        if name == "healthy":
            unified[UNIFIED_INDEX["healthy"]] += float(probability)
        else:
            unified[UNIFIED_INDEX["disease_stress"]] += float(probability)
    unified[UNIFIED_INDEX["pest_stress"]] = 0.0
    return unified


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def rule_fusion(
    tabular_unified: np.ndarray, image_unified: np.ndarray
) -> tuple[str, str]:
    """Apply the image-gated rules R1-R6 and return ``(label, rule_id)``.

    The image model's two collapsed probabilities (healthy, disease_stress)
    gate every decision so a confidently healthy leaf cannot be overridden by
    a high tabular stress probability, while high-confidence physiological
    sensor stress (nutrient_stress, water_stress, stress_generic) arbitrates
    among stress classes when the leaf image shows stress.
    """
    tabular_healthy = tabular_unified[UNIFIED_INDEX["healthy"]]
    tabular_nutrient = tabular_unified[UNIFIED_INDEX["nutrient_stress"]]
    tabular_water = tabular_unified[UNIFIED_INDEX["water_stress"]]
    tabular_generic = tabular_unified[UNIFIED_INDEX["stress_generic"]]
    image_healthy = image_unified[UNIFIED_INDEX["healthy"]]
    image_disease = image_unified[UNIFIED_INDEX["disease_stress"]]

    # R3: both models agree the leaf is healthy.
    if tabular_healthy > THRESHOLDS["both_healthy"] and image_healthy > THRESHOLDS["both_healthy"]:
        return "healthy", "R3"
    # R2: confident image health overrides a lower tabular healthy score.
    if image_healthy > THRESHOLDS["image_healthy"]:
        return "healthy", "R2"
    # R4: confident tabular nutrient stress evidence.
    if tabular_nutrient > THRESHOLDS["nutrient"]:
        return "nutrient_stress", "R4"
    # R5: confident tabular water stress evidence.
    if tabular_water > THRESHOLDS["water"]:
        return "water_stress", "R5"
    # R1: confident image disease call when tabular stress probabilities are uncertain.
    if (
        image_disease > THRESHOLDS["disease"]
        and max(tabular_nutrient, tabular_water, tabular_generic) < THRESHOLDS["tabular_healthy_low"]
    ):
        return "disease_stress", "R1"
    # R6: tabular stress argmax when image shows stress, else overall tabular argmax.
    stress_classes = [name for name in TABULAR_CLASSES if name != "healthy"]
    if image_disease > THRESHOLDS["disease_block"]:
        best = max(stress_classes, key=lambda name: tabular_unified[UNIFIED_INDEX[name]])
        if tabular_unified[UNIFIED_INDEX[best]] > 0:
            return best, "R6"
    best = max(TABULAR_CLASSES, key=lambda name: tabular_unified[UNIFIED_INDEX[name]])
    if best == "stress_generic":
        specific = [
            name for name in TABULAR_CLASSES if name != "stress_generic"
        ]
        candidate = max(
            specific, key=lambda name: tabular_unified[UNIFIED_INDEX[name]]
        )
        if tabular_unified[UNIFIED_INDEX[candidate]] > 0:
            best = candidate
    return best, "R6"


def fuse(
    tabular_probabilities: np.ndarray,
    tabular_classes: list[str],
    image_probabilities: np.ndarray,
    image_classes: list[str],
) -> dict[str, Any]:
    """Align both probability vectors and apply the fusion rules."""
    tabular_unified = tabular_to_unified(tabular_probabilities, tabular_classes)
    image_unified = image_to_unified(image_probabilities, image_classes)
    label, rule = rule_fusion(tabular_unified, image_unified)
    return {
        "prediction": label,
        "rule": rule,
        "tabular_unified": tabular_unified,
        "image_unified": image_unified,
        "tabular_probs": dict(zip(tabular_classes, np.asarray(tabular_probabilities, dtype=float))),
        "image_probs": dict(zip(image_classes, np.asarray(image_probabilities, dtype=float))),
        "unified_classes": UNIFIED_CLASSES,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _first_image(root: Path) -> Path:
    """Return the first readable image under ``root`` (deterministic order)."""
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    raise FileNotFoundError(f"No images found under {root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run rule-based fusion on one CHES row and one leaf image."
    )
    parser.add_argument("--image", type=Path, default=None, help="Leaf image path")
    parser.add_argument(
        "--image_dir", type=Path, default=None, help="Dataset root used when --image is omitted"
    )
    parser.add_argument(
        "--data_path",
        "--tabular",
        dest="data_path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "CHES CSV/ZIP for the tabular row; --tabular is the documented "
            f"alias (default: {DEFAULT_DATA_PATH})"
        ),
    )
    parser.add_argument("--row_index", type=int, default=0, help="CHES row index")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = parse_args(argv)
    try:
        tabular_bundle = load_tabular_bundle()
        image_bundle = load_image_bundle()

        dataframe = load_dataset(args.data_path)
        if not 0 <= args.row_index < len(dataframe):
            raise IndexError(
                f"--row_index {args.row_index} outside dataset range "
                f"[0, {len(dataframe) - 1}]"
            )
        row = dataframe.iloc[[args.row_index]]
        tabular_probabilities = predict_tabular_probs(tabular_bundle, row)[0]

        image_path = args.image or _first_image(
            resolve_image_dir(args.image_dir)
        )
        image_probabilities = predict_image_probs(image_bundle, image_path)

        result = fuse(
            tabular_probabilities,
            tabular_bundle["class_order"],
            image_probabilities,
            image_bundle["classes"],
        )

        print(f"\n[fusion] CHES row {args.row_index}  image={image_path}")
        print("[fusion] tabular probabilities:")
        for name, value in result["tabular_probs"].items():
            print(f"           {name:<18}{value:.4f}")
        print("[fusion] image probabilities:")
        for name, value in result["image_probs"].items():
            print(f"           {name:<18}{value:.4f}")
        print("[fusion] unified 6-class vectors:")
        print(f"           {'class':<18}{'tabular':>9}{'image':>9}")
        for name, value in zip(UNIFIED_CLASSES, result["tabular_unified"]):
            index = UNIFIED_INDEX[name]
            print(
                f"           {name:<18}{value:>9.4f}"
                f"{result['image_unified'][index]:>9.4f}"
            )
        print(f"\n[fusion] PREDICTION: {result['prediction']}  ({result['rule']})")
    except (FileNotFoundError, RuntimeError, ValueError, IndexError, KeyError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
