"""Stage 2A CLI - model output -> Layer 1 diagnosis -> optional LLM polish.

Pipeline for one sample::

    Stage-1A dataset  ->  unified model (attention_fusion checkpoint)
                      ->  structured output {class, confidence, attention,
                          tabular attributions, Grad-CAM summary, NDVI trend}
                      ->  Layer 1 rule/template diagnosis (diagnosis_engine)
                      ->  Layer 2 LLM polish (llm_polish, optional)
                      ->  printed report / diagnoses.csv

Attribution method (documented, no invented numbers):
* tabular  - gradient x input of the predicted-class logit, computed
             batch-wide in one backward pass (samples are independent in
             eval mode), so contributions are per-sample;
* image    - Grad-CAM over MobileNetV2's last feature map, summarised as
             activation coverage and peak region (top/bottom x left/right);
* ndvi     - start/end/direction/steepest single-step drop read off the
             actual 20-step series (flagged PROXY while the proxy is in use).

Modes
-----
    python diagnosis_cli.py --sample_idx 42
    python diagnosis_cli.py --sample_idx 42 --no-llm
    python diagnosis_cli.py --interactive
    python diagnosis_cli.py --batch --output diagnoses.csv [--limit 500]

``--sample_idx`` indexes the chosen split in Stage-1A order (the same order
stored in artifacts/split_indices.json).
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
import torch
from torch.nn import functional as F

import dataset as data
import stage1_config as cfg
import tabular_xgb as tx
from attention_fusion import AttentionFusion, CKPT_PATH as MODEL_PATH
from mobilenet_cnn import build_transforms
from diagnosis_engine import (
    Diagnosis,
    ModelOutput,
    TopFeature,
    diagnose,
    load_knowledge_base,
)
from llm_polish import describe_provider, polish
from stage1_config import UNIFIED_CLASSES, UNIFIED_INDEX

KB_PATH = Path("knowledge_base.json")


def log(message: str) -> None:
    """Print a timestamped progress line."""
    print(f"[stage2a {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_unified_model(
    checkpoint: Path | str, device: torch.device, tabular_dim: int
) -> AttentionFusion:
    """Load the Stage-1B attention-fusion checkpoint (clear errors on failure)."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train the unified model "
            "first: python attention_fusion.py"
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = AttentionFusion(tabular_dim=tabular_dim).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    saved_order = payload.get("class_order")
    if saved_order and list(saved_order) != list(UNIFIED_CLASSES):
        raise ValueError(
            f"Checkpoint class order {saved_order} does not match "
            f"stage1_config {list(UNIFIED_CLASSES)} - retrain or fix the config"
        )
    epoch = payload.get("epoch", "?")
    log(f"model loaded from {checkpoint.name} (saved at epoch {epoch})")
    return model


# ---------------------------------------------------------------------------
# Model output extraction
# ---------------------------------------------------------------------------
def describe_ndvi(series: np.ndarray, is_proxy: bool) -> str:
    """Describe the NDVI trend read off the actual series (no invention)."""
    start, end = float(series[0]), float(series[-1])
    delta = end - start
    if abs(delta) < 0.03:
        trend = f"stable ({start:.2f} -> {end:.2f})"
    elif delta < 0:
        trend = f"declining from {start:.2f} to {end:.2f}"
    else:
        trend = f"rising from {start:.2f} to {end:.2f}"
    steps = np.diff(series.astype(float))
    steepest = float(steps.min()) if len(steps) else 0.0
    extra = (
        f", steepest single-step drop {abs(steepest):.2f}"
        if steepest < -0.05
        else ""
    )
    prefix = "PROXY NDVI" if is_proxy else "NDVI"
    return f"{prefix} {trend} over {len(series)} time steps{extra}"


def gradcam_summary(cam: np.ndarray, threshold: float = 0.5) -> str:
    """Summarise a Grad-CAM map as coverage % + peak region (7x7 grid)."""
    peak = float(cam.max())
    if peak <= 1e-8:
        return "image branch produced no usable activation for this sample"
    mask = cam >= threshold * peak
    coverage = 100.0 * float(mask.mean())
    rows, cols = np.unravel_index(int(np.argmax(cam)), cam.shape)
    vertical = "upper" if rows < cam.shape[0] / 2 else "lower"
    horizontal = "left" if cols < cam.shape[1] / 2 else "right"
    return (
        f"model attention covers {coverage:.0f}% of the leaf region, "
        f"strongest in the {vertical}-{horizontal} area"
    )


def analyze_batch(
    model: AttentionFusion,
    tabular: torch.Tensor,
    images: torch.Tensor,
    ndvi: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run the model with gradients and return per-sample attributions.

    One backward pass per batch is enough: samples do not interact in eval
    mode (BatchNorm uses running statistics), so ``grad`` of the summed score
    equals each sample's own gradient.
    """
    tab = tabular.to(device).clone().requires_grad_(True)
    images = images.to(device)
    ndvi = ndvi.to(device)

    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logits, attention = model(images, tab, ndvi)
        probs = F.softmax(logits.detach(), dim=1)
        pred = probs.argmax(dim=1)
        confidence = probs.gather(1, pred[:, None]).squeeze(1)
        scores = logits.gather(1, pred[:, None]).squeeze(1)
        activations = model.image_encoder.activations
        grad_tab, grad_acts = torch.autograd.grad(
            scores.sum(), [tab, activations], retain_graph=False, allow_unused=True
        )

    contrib = (grad_tab * tab).detach().cpu().numpy()
    tabular_values = tab.detach().cpu().numpy()

    cams: list[np.ndarray] = []
    if grad_acts is not None:
        weights = grad_acts.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()
        cams = [cam[i] for i in range(cam.shape[0])]
    else:
        cams = [np.zeros((1, 1), dtype=np.float32)] * logits.size(0)

    return {
        "pred": pred.detach().cpu().numpy(),
        "confidence": confidence.detach().cpu().numpy(),
        "attention": attention.detach().cpu().numpy(),
        "contrib": contrib,
        "tabular_values": tabular_values,
        "cams": np.asarray(cams, dtype=object),
    }


def build_value_map(preprocessor: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Map scaled feature -> (mean, scale) so values display in raw units.

    Attributions stay in model (standardised) space, which is what the
    gradient is computed against; only the *displayed value* is converted
    back to the CSV's native units so a farmer-facing line reads
    ``soil moisture = 18.2`` instead of a z-score.  Ordinal encodings have
    no scaler and are passed through unchanged.
    """
    mapping: dict[str, tuple[float, float]] = {}
    for group in preprocessor.get("groups", {}).values():
        columns = group.get("columns", [])
        means = group.get("mean", [])
        scales = group.get("scale", [])
        for name, mean, scale in zip(columns, means, scales):
            mapping[name] = (float(mean), float(scale))
    return mapping


def raw_value(feature: str, scaled: float, value_map: dict[str, tuple[float, float]]) -> float:
    """Convert one standardised feature value back to its original units."""
    entry = value_map.get(feature)
    if entry is None:
        return scaled                      # ordinal encoding, already raw
    mean, scale = entry
    return scaled * scale + mean


def build_model_output(
    index: int,
    split: str,
    sample: dict[str, Any],
    analysis: dict[str, Any],
    position: int,
    feature_names: list[str],
    value_map: dict[str, tuple[float, float]],
) -> ModelOutput:
    """Assemble the Layer 1 input dict for one sample."""
    contribution = np.asarray(analysis["contrib"][position], dtype=float)
    values = np.asarray(analysis["tabular_values"][position], dtype=float)
    top_positions = np.argsort(-np.abs(contribution))[:3]
    top = [
        TopFeature(
            feature=feature_names[i],
            value=raw_value(feature_names[i], float(values[i]), value_map),
            contribution=float(contribution[i]),
        )
        for i in top_positions
    ]
    pred_idx = int(analysis["pred"][position])
    attention = {
        name: float(analysis["attention"][position, i])
        for i, name in enumerate(("tabular", "image", "ndvi"))
    }
    return ModelOutput(
        predicted_class=UNIFIED_CLASSES[pred_idx],
        confidence=float(analysis["confidence"][position]),
        attention_weights=attention,
        tabular_top_features=top,
        gradcam_summary=gradcam_summary(np.asarray(analysis["cams"][position],
                                                    dtype=np.float32)),
        ndvi_trend=describe_ndvi(
            np.asarray(sample["ndvi"], dtype=float), sample["ndvi_is_proxy"]
        ),
        ndvi_is_proxy=sample["ndvi_is_proxy"],
        true_class=sample["true_class"],
        sample_id=f"{split}:{index} (row {sample['row_index']})",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(
    model_output: ModelOutput,
    layer1: Diagnosis,
    layer2: str | None,
    sample: dict[str, Any],
    llm_status: str,
) -> None:
    """Print the five required blocks for a single sample."""
    print("\n" + "=" * 88)
    print(f"SAMPLE {model_output.sample_id}")
    print(f"image: {sample['image']}")
    print("-" * 88)
    print("1. RAW MODEL OUTPUT")
    print(f"   predicted class .... {model_output.predicted_class}")
    print(f"   confidence ......... {model_output.confidence:.4f}")
    print(
        "   attention weights ... "
        + ", ".join(f"{k}={v:.3f}" for k, v in model_output.attention_weights.items())
    )
    print(f"   tabular attributions:")
    for feat in model_output.tabular_top_features:
        print(
            f"     - {feat.feature} = {feat.value:g} "
            f"(contribution {feat.contribution:+.3f})"
        )
    print(f"   grad-cam ............ {model_output.gradcam_summary}")
    print(f"   ndvi trend .......... {model_output.ndvi_trend}")
    if model_output.true_class is not None:
        print(f"   ground truth ........ {model_output.true_class}")
    print("-" * 88)
    print("2. LAYER 1 DIAGNOSIS (deterministic template)")
    print(layer1.draft)
    print("-" * 88)
    print("3. LAYER 2 POLISH")
    print(f"   [{llm_status}]")
    print(f"   {layer2}" if layer2 else "   (kept Layer 1 text - LLM unavailable)")
    print("-" * 88)
    print("4. SEVERITY + RECOMMENDED ACTION")
    print(f"   severity ............ {layer1.severity.upper()}")
    for reason in layer1.severity_reasons:
        print(f"     - {reason}")
    print(f"   first action ........ {layer1.actions[0]}")
    print(f"   escalation .......... {layer1.escalation}")
    print("=" * 88)


def row_from(
    index: int,
    split: str,
    sample: dict[str, Any],
    model_output: ModelOutput,
    layer1: Diagnosis,
    layer2: str | None,
) -> dict[str, Any]:
    """Flatten one diagnosis into a CSV row."""
    attention = model_output.attention_weights
    return {
        "sample_idx": index,
        "split": split,
        "row_index": sample["row_index"],
        "image": str(sample["image"]),
        "true_class": model_output.true_class or "",
        "predicted_class": model_output.predicted_class,
        "confidence": f"{model_output.confidence:.4f}",
        "severity": layer1.severity,
        "driver_modality": layer1.driver_modality,
        "attention_tabular": f"{attention.get('tabular', 0.0):.4f}",
        "attention_image": f"{attention.get('image', 0.0):.4f}",
        "attention_ndvi": f"{attention.get('ndvi', 0.0):.4f}",
        "gradcam_summary": model_output.gradcam_summary,
        "ndvi_trend": model_output.ndvi_trend,
        "contributing_factors": " | ".join(layer1.factors),
        "recommended_actions": " | ".join(layer1.actions),
        "escalation": layer1.escalation,
        "layer1_draft": layer1.draft.replace("\n", " | "),
        "layer2_polished": (layer2 or "").replace("\n", " "),
        "llm_used": "yes" if layer2 else "no",
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def sample_context(
    dataset: data.MultiModalStressDataset, index: int, split: str
) -> dict[str, Any]:
    """Non-tensor context for a dataset position (row, image, labels)."""
    sample = int(dataset.indices[index])
    return {
        "row_index": int(dataset.table["row"][sample]),
        "image": Path(dataset.table["image"][sample]),
        "true_class": UNIFIED_CLASSES[int(dataset.table["label"][sample])],
        "ndvi": np.asarray(dataset.ndvi[sample], dtype=np.float32),
        "ndvi_is_proxy": bool(dataset.ndvi_is_proxy),
        "split": split,
        "index": index,
    }


def run_indices(
    model: AttentionFusion,
    dataset: data.MultiModalStressDataset,
    indices: list[int],
    split: str,
    kb: dict[str, Any],
    feature_names: list[str],
    value_map: dict[str, tuple[float, float]],
    device: torch.device,
    batch_size: int,
    use_llm: bool,
    interactive: bool,
) -> list[dict[str, Any]]:
    """Analyze ``indices`` in batches; print singles, return all CSV rows."""
    rows: list[dict[str, Any]] = []
    llm_status = describe_provider() if use_llm else "disabled (--no-llm)"

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        tensors = [dataset[i] for i in chunk]
        tabular = torch.stack([t[0] for t in tensors])
        images = torch.stack([t[1] for t in tensors])
        ndvi = torch.stack([t[2] for t in tensors])
        analysis = analyze_batch(model, tabular, images, ndvi, device)

        for position, index in enumerate(chunk):
            context = sample_context(dataset, index, split)
            output = build_model_output(
                index, split, context, analysis, position, feature_names, value_map
            )
            layer1 = diagnose(output, kb)
            layer2 = polish(layer1.draft) if use_llm else None
            rows.append(row_from(index, split, context, output, layer1, layer2))
            # Full report only in single/interactive mode; batch mode stays
            # quiet and writes everything to CSV.
            if interactive:
                print_report(output, layer1, layer2, context, llm_status)
        if not interactive and len(indices) > 20:
            log(f"processed {min(start + batch_size, len(indices))}/{len(indices)}")
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write batch results to CSV (UTF-8, header from the first row)."""
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {len(rows)} diagnoses -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_datasets(
    args: argparse.Namespace,
) -> tuple[dict[str, data.MultiModalStressDataset], np.ndarray, list[str]]:
    """Rebuild the Stage-1A dataset exactly as Stage 1B splits it."""
    tabular = data.load_tabular(args.data_path)
    row_splits = data.split_rows(tabular["labels"], seed=args.seed)
    pools = data.build_image_pools(args.image_dir, seed=args.seed)
    table = data.build_sample_table(tabular["labels"], row_splits, pools, seed=args.seed)

    real = data.load_real_ndvi(tabular["frame"], cfg.NDVI_STEPS)
    ndvi_is_proxy = real is None
    ndvi = (
        real
        if real is not None
        else data.build_proxy_ndvi(table["label"], seed=args.seed)
    )

    train_tf, eval_tf = build_transforms()
    cache: dict[Path, Any] = {}
    datasets = {
        name: data.MultiModalStressDataset(
            np.where(table["split"] == name)[0],
            table, tabular["features"], ndvi,
            train_tf if name == "train" else eval_tf, cache,
        )
        for name in ("train", "val", "test")
    }
    preprocessor = json.loads(tx.PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    feature_names = list(preprocessor["feature_names"])
    return (
        datasets,
        tabular["features"],
        feature_names,
        build_value_map(preprocessor),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage-2A CLI options."""
    parser = argparse.ArgumentParser(
        description="Stage 2A: farmer-friendly diagnosis from the unified model"
    )
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="index within the chosen split (Stage-1A order)")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--no-llm", action="store_true",
                        help="Layer 1 only (skip LLM polish)")
    parser.add_argument("--interactive", action="store_true",
                        help="loop: read sample indices from stdin")
    parser.add_argument("--batch", action="store_true",
                        help="run over the split and save CSV")
    parser.add_argument("--output", type=Path, default=Path("diagnoses.csv"))
    parser.add_argument("--limit", type=int, default=None,
                        help="cap samples in batch mode")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--llm_provider", default=None,
                        help="ollama | openai | anthropic | auto (default: auto)")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--data_path", type=Path, default=tx.DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", type=Path, default=Path("data/paddy_images"))
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Stage-2A pipeline (0 on success, 1 on a clean failure)."""
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # Windows consoles default to a legacy codepage; diagnoses contain
        # UTF-8 characters (degree signs, arrows) from the LLM output.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        kb = load_knowledge_base(KB_PATH)
        datasets, _features, feature_names, value_map = build_datasets(args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_unified_model(
            args.checkpoint, device, tabular_dim=len(feature_names)
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    use_llm = not args.no_llm
    log(describe_provider() if use_llm else "LLM polish disabled (--no-llm)")
    dataset = datasets[args.split]
    available = len(dataset)
    if available == 0:
        print(f"[error] split {args.split!r} is empty", file=sys.stderr)
        return 1

    try:
        if args.interactive:
            log(f"interactive mode on split {args.split!r} "
                f"({available:,} samples); enter an index or 'q' to quit")
            for line in sys.stdin:
                text = line.strip()
                if text.lower() in {"q", "quit", "exit"}:
                    break
                if not text.lstrip("-").isdigit():
                    print("enter a numeric sample index or 'q'")
                    continue
                index = int(text)
                if not 0 <= index < available:
                    print(f"index must be in [0, {available - 1}]")
                    continue
                run_indices(
                    model, dataset, [index], args.split, kb, feature_names,
                    value_map, device, 1, use_llm, True,
                )
        elif args.batch:
            count = available if args.limit is None else min(args.limit, available)
            indices = list(range(count))
            rows = run_indices(
                model, dataset, indices, args.split, kb, feature_names,
                value_map, device, args.batch_size, use_llm, False,
            )
            write_csv(rows, args.output)
            correct = sum(
                1 for r in rows if r["true_class"] == r["predicted_class"]
            )
            log(f"accuracy on this batch: {correct}/{len(rows)} "
                f"({correct / max(len(rows), 1):.4f})")
        else:
            if not 0 <= args.sample_idx < available:
                print(
                    f"[error] --sample_idx must be in [0, {available - 1}] "
                    f"for split {args.split!r}",
                    file=sys.stderr,
                )
                return 1
            run_indices(
                model, dataset, [args.sample_idx], args.split, kb,
                feature_names, value_map, device, 1, use_llm, True,
            )
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
