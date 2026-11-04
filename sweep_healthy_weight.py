"""Diagnostic sweep: healthy-class weight vs test P/R/F1 (P1 FIX 4).

Trains the production XGBoost configuration (2000 trees, early stopping 30,
inverse-frequency weights) for a grid of ``HEALTHY_WEIGHT_BOOST`` values and
reports healthy precision/recall/F1 and macro F1 on the untouched test split.

Also computes the precision-recall curve of p(healthy) on the test split to
establish the attainable ceiling: the binary CHES flag is unlearnable from the
features (AUC ~0.50), so labels -- not the optimiser -- bound healthy precision.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

import tabular_xgb as tx

BOOSTS = [0.5, 1.0, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0]
if len(sys.argv) > 1:
    BOOSTS = [float(a) for a in sys.argv[1:]]


def main() -> int:
    """Run the boost grid on the fixed split and print per-boost metrics."""
    df = tx.load_dataset(tx.DEFAULT_DATA_PATH)
    cols = tx.resolve_columns(df)
    y, _meta = tx.build_labels(df, cols)

    all_idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.30, stratify=y, random_state=tx.SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=tx.SEED
    )
    pre = tx.fit_preprocessor(df, train_idx, cols)
    X = tx.transform_features(df, pre)
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    counts = np.bincount(y_train, minlength=len(tx.CLASS_ORDER)).astype(float)
    base_w = len(y_train) / (len(tx.CLASS_ORDER) * counts)
    healthy_i = tx.CLASS_ORDER.index("healthy")
    print(f"[split] train={len(train_idx):,} val={len(val_idx):,} "
          f"test={len(test_idx):,}  base_w={np.round(base_w, 3).tolist()}", flush=True)

    results = []
    for boost in BOOSTS:
        w = base_w.copy()
        w[healthy_i] *= boost
        params = dict(tx.XGB_PARAMS)
        params["device"] = tx.resolve_xgb_device()
        model = tx.xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w[y_train],
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, labels=range(len(tx.CLASS_ORDER)), zero_division=0
        )
        vpred = model.predict(X_val)
        vp, vr, vf, _ = precision_recall_fscore_support(
            y_val, vpred, labels=range(len(tx.CLASS_ORDER)), zero_division=0
        )
        proba = model.predict_proba(X_test)[:, healthy_i]
        yh = (y_test == healthy_i).astype(int)
        p_c, r_c, f_c = precision_recall_curve(yh, proba)
        valid = (p_c + r_c) > 0
        f_c = np.where(valid, 2 * p_c * r_c / np.where(valid, p_c + r_c, 1), 0)
        k = int(np.argmax(f_c))
        row = {
            "boost": boost,
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1.mean()),
            "healthy_p": float(precision[healthy_i]),
            "healthy_r": float(recall[healthy_i]),
            "healthy_f1": float(f1[healthy_i]),
            "pr_best_f1": float(f_c[k]),
            "pr_best_p": float(p_c[k]),
            "pr_best_r": float(r_c[k]),
            "n_predicted_healthy": int((pred == healthy_i).sum()),
            "val_macro_f1": float(vf.mean()),
            "val_healthy_p": float(vp[healthy_i]),
            "val_healthy_r": float(vr[healthy_i]),
            "val_healthy_f1": float(vf[healthy_i]),
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    print("\nSUMMARY (test split, n=31,803)", flush=True)
    print(f"{'boost':>6} {'iters':>6} {'acc':>7} {'macroF1':>8} "
          f"{'h_P':>6} {'h_R':>6} {'h_F1':>6} | {'valMac':>7} {'valhF1':>7} "
          f"| {'PRbestF1':>9} {'P':>5} {'R':>5}", flush=True)
    for r in results:
        print(f"{r['boost']:>6} {r['best_iteration']:>6} {r['accuracy']:>7.4f} "
              f"{r['macro_f1']:>8.4f} {r['healthy_p']:>6.3f} {r['healthy_r']:>6.3f} "
              f"{r['healthy_f1']:>6.3f} | {r['val_macro_f1']:>7.4f} "
              f"{r['val_healthy_f1']:>7.4f} | {r['pr_best_f1']:>9.3f} "
              f"{r['pr_best_p']:>5.3f} {r['pr_best_r']:>5.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
