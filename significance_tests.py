"""Stage 1C/4 - paired statistical significance tests for the final report.

    python significance_tests.py            # writes artifacts/significance_tests.json

Compares the unified multi-modal model against the strongest single-modality
baseline (XGBoost tabular-only) on **the same CHES rows**, using McNemar's
exact test (two-sided binomial on the discordant pairs).

Pairing rules (methodology, no leakage):
* Unified predictions come from rows in the Stage-1A *test* split (row-sourced
  samples only - image-sourced disease samples carry a random contextual row
  and have no tabular ground truth of their own).
* XGBoost predictions are restricted to rows in *its* test split (its own
  70/15/15 stratified split, seed 42), so neither model is scored on rows it
  trained or validated on.
* The paired set is the intersection: rows unseen by both models.
* XGBoost's 4-class output is mapped to the unified 6-class space through
  ``stage1_config.tabular_label_to_unified`` (the documented conversion).

Outputs ``artifacts/significance_tests.json`` with per-model accuracy on the
paired set, the 2x2 discordant counts, the exact p-value and McNemar's
chi-square with continuity correction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import stage1_config as cfg

OUTPUT = Path("artifacts/significance_tests.json")
ALPHA = 0.05


def log(message: str) -> None:
    """Print a timestamped progress line."""
    print(f"[stage4 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired correctness vectors."""
    from scipy.stats import binomtest

    a_only = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    b_only = int(np.sum(~correct_a & correct_b))   # B right, A wrong
    discordant = a_only + b_only
    if discordant == 0:
        return {"b": a_only, "c": b_only, "discordant": 0,
                "p_value": 1.0, "chi2_correction": 0.0,
                "note": "no discordant pairs - models make identical errors"}
    p_value = float(
        binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue
    )
    chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
    # When every discordant pair favours one model, exact p = 2^(1-n), which
    # underflows float64 for n > ~1024 - report log10(p) so the artifact can
    # still state *how* significant the result is.
    import math

    if p_value > 0.0:
        p_log10 = round(math.log10(p_value), 4)
        note = None
    else:
        p_log10 = round((1 - discordant) * math.log10(2), 2)
        note = f"exact two-sided p = 2^({1 - discordant}) ~ 10^{p_log10}; "
        note += "underflows float64, so p_value is reported as 0.0"
    result = {
        "b": a_only,
        "c": b_only,
        "discordant": discordant,
        "p_value": round(p_value, 8),
        "p_value_log10": p_log10,
        "chi2_correction": round(float(chi2), 4),
        "significant_at_0.05": bool(p_value < ALPHA),
    }
    if note:
        result["p_value_note"] = note
    return result


def build_paired_predictions(seed: int) -> dict[str, Any]:
    """Unified + XGBoost predictions on rows unseen by both models."""
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    import dataset as data
    import fusion_inference as fi
    import tabular_xgb as tx
    from mobilenet_cnn import build_transforms

    # --- Stage-1A test split (unified model's held-out rows) ---------------
    tabular = data.load_tabular(tx.DEFAULT_DATA_PATH)
    row_splits = data.split_rows(tabular["labels"], seed=seed)
    pools = data.build_image_pools(Path("data/paddy_images"), seed=seed)
    table = data.build_sample_table(tabular["labels"], row_splits, pools, seed=seed)
    real = data.load_real_ndvi(tabular["frame"], cfg.NDVI_STEPS)
    ndvi = real if real is not None else data.build_proxy_ndvi(table["label"], seed=seed)
    _, eval_tf = build_transforms()
    test_indices = np.where(table["split"] == "test")[0]
    # row-sourced samples occupy table indices [0, n_rows); image-sourced
    # samples follow and carry a random contextual row -> excluded.
    n_rows = len(tabular["labels"])
    row_sourced = test_indices[test_indices < n_rows]
    log(f"unified test: {len(test_indices)} samples "
        f"({len(row_sourced)} row-sourced, eligible for pairing)")

    dataset = data.MultiModalStressDataset(
        row_sourced, table, tabular["features"], ndvi, eval_tf, {}
    )
    # dataset position order == row_sourced order (np.where is sorted)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    import attention_fusion as af
    import torch

    ckpt_path = (
        Path("artifacts/unified_model.pth")
        if Path("artifacts/unified_model.pth").exists()
        else af.CKPT_PATH
    )
    if ckpt_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = blob.get("model_state", blob.get("state_dict", blob))
        model = af.AttentionFusion(tabular_dim=tabular["features"].shape[1]).to(device)
        model.load_state_dict(state)
        unified_pred = af.predict(model, loader, device)["y_pred"].astype(np.int64)
    else:
        import onnxruntime as ort

        session = ort.InferenceSession(
            "artifacts/unified_model.onnx", providers=["CPUExecutionProvider"]
        )
        unified_pred_list: list[int] = []
        for _tab, img, ndvi_t, _label in loader:
            logits = session.run(None, {
                "images": img.numpy().astype(np.float32),
                "tabular": _tab.numpy().astype(np.float32),
                "ndvi": ndvi_t.numpy().astype(np.float32),
            })[0]
            unified_pred_list.extend(logits.argmax(-1).tolist())
        unified_pred = np.asarray(unified_pred_list, dtype=np.int64)
    unified_true = table["label"][row_sourced].astype(np.int64)
    unified_rows = table["row"][row_sourced].astype(np.int64)

    # --- XGBoost's own held-out rows ---------------------------------------
    df = tx.load_dataset(tx.DEFAULT_DATA_PATH)
    cols = tx.resolve_columns(df)
    y4, _ = tx.build_labels(df, cols)
    all_idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.30, stratify=y4, random_state=cfg.SEED
    )
    _val_idx, xgb_test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y4[temp_idx], random_state=cfg.SEED
    )
    log(f"xgb test split: {len(xgb_test_idx)} rows; "
        f"unified train/val excluded: {len(train_idx)} rows")

    # --- intersection: rows unseen by BOTH ---------------------------------
    xgb_test_set = np.zeros(len(df), dtype=bool)
    xgb_test_set[xgb_test_idx] = True
    paired_mask = xgb_test_set[unified_rows]
    paired_rows = unified_rows[paired_mask]
    if len(paired_rows) == 0:
        raise RuntimeError("empty paired set - check split definitions")

    bundle = fi.load_tabular_bundle()
    probs = fi.predict_tabular_probs(bundle, df.iloc[paired_rows])
    xgb4 = probs.argmax(axis=1)
    xgb_pred = cfg.tabular_label_to_unified(xgb4)

    true = unified_true[paired_mask]
    pred_unified = unified_pred[paired_mask]
    correct_u = pred_unified == true
    correct_x = xgb_pred == true
    log(f"paired set: {len(paired_rows)} rows unseen by both models")
    return {
        "n_paired": int(len(paired_rows)),
        "unified_correct": correct_u,
        "xgb_correct": correct_x,
        "unified_accuracy": round(float(correct_u.mean()), 4),
        "xgb_accuracy": round(float(correct_x.mean()), 4),
        "xgb_pred_accuracy_paired": round(float((xgb_pred == true).mean()), 4),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Paired McNemar test: unified model vs XGBoost baseline."
    )
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args(argv)

    try:
        paired = build_paired_predictions(args.seed)
        test = mcnemar_exact(paired["unified_correct"], paired["xgb_correct"])
    except FileNotFoundError as err:
        log(f"ERROR: missing artifact - {err}")
        return 1

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "comparison": "unified multi-modal (AttentionFusion) vs XGBoost tabular-only",
        "test": "McNemar exact (two-sided binomial on discordant pairs)",
        "alpha": ALPHA,
        "pairing": (
            "intersection of the Stage-1A test split (row-sourced samples) and "
            "the XGBoost test split; neither model is scored on training rows"
        ),
        "n_paired": paired["n_paired"],
        "accuracy": {
            "unified": paired["unified_accuracy"],
            "xgb_tabular": paired["xgb_accuracy"],
            "difference": round(
                paired["unified_accuracy"] - paired["xgb_accuracy"], 4
            ),
        },
        "mcnemar": test,
        "verdict": (
            "significant (reject H0: equal error rates)"
            if test.get("significant_at_0.05")
            else "not significant at alpha=0.05"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"saved -> {OUTPUT}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
