"""Stage 4.2b - feedback-driven fine-tuning (v1 -> v2 -> ...) .

    python retrain.py --run       # gated by artifacts/retrain_flag.json
    python retrain.py --compare   # evaluate existing versions + curve

Pipeline
--------
1. Gate on ``artifacts/retrain_flag.json`` (written by
   ``feedback_loop.py --check-retrain``); if the flag is missing or false the
   CLI exits with a message instead of training.
2. Read ``feedback_log.csv`` and convert it into training signal:
   * ``incorrect`` + ``user_actual_class`` -> **correction** (label overridden
     to the agronomist's class);
   * ``correct`` -> **reinforcement** sample (lower sampler weight).
   Each row is matched to the *real* Stage-1A sample whose image basename
   matches, so the tabular/NDVI/image values are the genuine pipeline values -
   nothing is synthesised.  Rows that cannot be matched are skipped and
   counted.
3. Fine-tune the latest checkpoint on (original train split + matched samples)
   for a few epochs at a low LR, then save as the NEXT version
   (``unified_model_v2.pth``, ``_v3``, ...).  The base checkpoint
   ``unified_model.pth`` is never overwritten.
4. Evaluate every existing version on the same seeded test subset, save
   ``artifacts/retrain_comparison.json`` and plot the improvement curve to
   ``plots/retrain_curve.png``.

Design decisions (measured rationale, not defaults)
----------------------------------------------------
* **Test-split exclusions.**  ``dataset.build_image_pools`` puts every photo
  in exactly one split pool (no photo crosses splits), so the sample table's
  ``split`` field unambiguously maps an image name to its split.   Corrections whose photo belongs to the *test* pool are skipped:
  training on them would poison the v1-vs-v2 comparison.
* **Oversampling corrections.**  The train split has ~150k rows; a handful of
  corrections would otherwise almost never be drawn.  The sampler weights are
  original = 1.0, correction = ``CORRECTION_OVERSAMPLE``, reinforcement =
  ``REINFORCE_WEIGHT`` (spec: lower weight for reinforcements), so each
  correction is seen about ``CORRECTION_OVERSAMPLE`` times per epoch while the
  epoch size stays that of the original train split.
* **Constant low LR** (spec: 1e-5 to avoid catastrophic forgetting) - no
  cosine decay, because decaying an already tiny LR to zero in 8 epochs would
  stop the corrections from landing at all.
* Loss keeps Stage 1B's inverse-frequency class weighting
  (``attention_fusion.inverse_frequency_weights``) recomputed on the combined
  label set.

Requires the retrain flag first::

    python feedback_loop.py --check-retrain
    python retrain.py --run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

import dataset as data
import stage1_config as cfg
import tabular_xgb as tx
from attention_fusion import (
    inverse_frequency_weights,
    metrics_block,
    predict,
    run_epoch,
    set_seed,
)
from edge_deploy import load_model
from feedback_loop import load_log
from mobilenet_cnn import build_transforms

# ---------------------------------------------------------------------------:
# Paths / defaults
# ---------------------------------------------------------------------------:
ARTIFACTS = Path("artifacts")
BASE_CHECKPOINT = ARTIFACTS / "unified_model.pth"
FLAG_JSON = ARTIFACTS / "retrain_flag.json"
COMPARISON_JSON = ARTIFACTS / "retrain_comparison.json"
CURVE_PLOT = Path("plots/retrain_curve.png")
FEEDBACK_LOG = Path("feedback_log.csv")
IMAGE_DIR = Path("data/paddy_images")

#: Spec: fine-tune 5-10 epochs.
DEFAULT_EPOCHS = 8
#: Spec: low LR to avoid catastrophic forgetting.
DEFAULT_LR = 1e-5
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 1.0
#: Each correction drawn ~20x per epoch (see module docstring).
CORRECTION_OVERSAMPLE = 20.0
#: Spec: reinforcements added with a lower weight.
REINFORCE_WEIGHT = 0.25
#: Seeded test subset for version comparison (0 = full test split).
DEFAULT_EVAL_LIMIT = 4096
DEFAULT_BATCH = 32
#: Feedback verdicts (components/feedback.py writes these).
VERDICT_CORRECT = "correct"
VERDICT_INCORRECT = "incorrect"


def log(message: str) -> None:
    """Print a timestamped progress line."""
    print(f"[stage4 {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Version management
# ---------------------------------------------------------------------------
def version_checkpoints() -> list[tuple[str, Path]]:
    """Existing checkpoints ordered v1 (base) -> v2 -> ..."""
    versions: list[tuple[str, Path]] = []
    if BASE_CHECKPOINT.exists():
        versions.append(("v1", BASE_CHECKPOINT))
    pattern = ARTIFACTS / "unified_model_v*.pth"
    found = []
    for path in pattern.parent.glob("unified_model_v*.pth"):
        stem = path.stem.replace("unified_model_", "")
        number = int(stem[1:]) if stem[1:].isdigit() else None
        if number is not None:
            found.append((number, stem, path))
    for _, stem, path in sorted(found):
        versions.append((stem, path))   # numeric order: v2, v3, ... v10
    return versions


def next_checkpoint() -> Path:
    """Path for the next retrain cycle: v1 -> v2 -> v3 ..."""
    existing = {name for name, _ in version_checkpoints()}
    cycle = 2
    while f"v{cycle}" in existing:
        cycle += 1
    return ARTIFACTS / f"unified_model_v{cycle}.pth"


def read_flag() -> dict[str, Any] | None:
    """Load the retrain flag; None when the file does not exist yet."""
    if not FLAG_JSON.exists():
        return None
    try:
        return json.loads(FLAG_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Stage-1A rebuild (same recipe as diagnosis_cli - reuse, no copy of logic)
# ---------------------------------------------------------------------------
def build_stage1(seed: int) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """Rebuild (datasets, table, features, ndvi) exactly as Stage 1B split them."""
    tabular = data.load_tabular(tx.DEFAULT_DATA_PATH)
    row_splits = data.split_rows(tabular["labels"], seed=seed)
    pools = data.build_image_pools(IMAGE_DIR, seed=seed)
    table = data.build_sample_table(tabular["labels"], row_splits, pools, seed=seed)

    real = data.load_real_ndvi(tabular["frame"], cfg.NDVI_STEPS)
    ndvi = real if real is not None else data.build_proxy_ndvi(table["label"], seed=seed)

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
    return datasets, table, tabular["features"], ndvi


def image_index(table: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """basename (lowercase) -> (split, sample index); first occurrence wins."""
    index: dict[str, tuple[str, int]] = {}
    for sample, path in enumerate(table["image"]):
        key = Path(str(path)).name.lower()
        if key not in index:
            index[key] = (str(table["split"][sample]), sample)
    return index


def split_positions(table: dict[str, Any], split: str) -> np.ndarray:
    """Table indices belonging to ``split`` (dataset position = rank here)."""
    return np.where(table["split"] == split)[0]


# ---------------------------------------------------------------------------
# 3. Feedback -> training samples
# ---------------------------------------------------------------------------
class CombinedDataset(Dataset):
    """Original train split plus relabelled correction/reinforcement views.

    Extras are ``(base_dataset, position, target_label)`` triples: the base
    dataset supplies the real tabular/image/NDVI tuple, only the label changes
    (correction) or is kept (reinforcement).
    """

    def __init__(self, base: Any, extras: list[tuple[Any, int, int]]) -> None:
        self.base = base
        self.extras = extras

    def __len__(self) -> int:
        return len(self.base) + len(self.extras)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if i < len(self.base):
            return self.base[i]
        dataset, position, label = self.extras[i - len(self.base)]
        tabular, image, ndvi, _ = dataset[position]
        return tabular, image, ndvi, label


def match_feedback(
    df: Any, table: dict[str, Any], datasets: dict[str, Any]
) -> tuple[list[tuple[Any, int, int]], list[tuple[Any, int, int]], dict[str, int]]:
    """Split feedback rows into (corrections, reinforcements, skip counts).

    Corrections carry the agronomist's label; reinforcements keep the sample's
    own label.  Test-split rows are skipped so the held-out comparison stays
    honest, unknown images/classes are skipped with counts (never guessed).
    """
    index = image_index(table)
    positions = {split: split_positions(table, split) for split in datasets}
    corrections: list[tuple[Any, int, int]] = []
    reinforcements: list[tuple[Any, int, int]] = []
    skipped = {"test_split": 0, "image_not_in_pipeline": 0, "unknown_class": 0,
               "empty_image_name": 0, "unlabelled_correction": 0}

    if df is None or df.empty:
        return corrections, reinforcements, skipped

    for _, row in df.iterrows():
        verdict = str(row.get("feedback", "")).strip().lower()
        name = str(row.get("image_name", "") or "").strip()
        if not name:
            skipped["empty_image_name"] += 1
            continue
        hit = index.get(Path(name).name.lower())
        if hit is None:
            skipped["image_not_in_pipeline"] += 1
            continue
        split, sample = hit
        if split == "test":
            skipped["test_split"] += 1
            continue
        position = int(np.searchsorted(positions[split], sample))
        base = datasets[split]

        if verdict == VERDICT_INCORRECT:
            actual = str(row.get("user_actual_class", "") or "").strip()
            label = cfg.UNIFIED_INDEX.get(actual)
            if not actual:
                skipped["unlabelled_correction"] += 1
                continue
            if label is None:
                skipped["unknown_class"] += 1
                continue
            corrections.append((base, position, int(label)))
        elif verdict == VERDICT_CORRECT:
            reinforcements.append(
                (base, position, int(table["label"][sample]))
            )
    return corrections, reinforcements, skipped


# ---------------------------------------------------------------------------
# 4. Fine-tuning
# ---------------------------------------------------------------------------
def fine_tune(
    base_path: Path,
    combined: CombinedDataset,
    weights: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    """Low-LR fine-tune of the latest checkpoint; saves the next version."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = load_model(base_path)
    model.to(device)

    labels = np.asarray([int(combined[i][3]) for i in range(len(combined))])
    criterion = nn.CrossEntropyLoss(
        weight=inverse_frequency_weights(labels).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(combined.base),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        combined, batch_size=DEFAULT_BATCH, sampler=sampler,
        num_workers=0, drop_last=False,
    )

    log(f"fine-tuning {base_path.name} on {len(combined.base)} original + "
        f"{len(combined.extras)} feedback samples | epochs={epochs} lr={lr:g} "
        f"device={device.type}")
    history = []
    started = time.perf_counter()
    ckpt_dir = ARTIFACTS / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prev_acc: float | None = None
    prev_loss: float | None = None
    stable_epochs = 0
    stability_tol = 0.01
    stability_patience = 2

    for epoch in tqdm(range(1, epochs + 1), desc="retrain", unit="epoch"):
        stats = run_epoch(
            model, loader, criterion, optimizer, device, scaler, use_amp,
            max_grad_norm=MAX_GRAD_NORM,
        )
        history.append({"epoch": epoch, "loss": round(stats["loss"], 5),
                        "accuracy": round(stats["accuracy"], 4)})
        epoch_ckpt = ckpt_dir / f"{out_path.stem}_epoch_{epoch:02d}.pth"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_order": cfg.UNIFIED_CLASSES,
                "ndvi_is_proxy": bool(cfg.NDVI_IS_PROXY),
                "epoch": epoch,
                "retrain": {
                    "from": str(base_path),
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": WEIGHT_DECAY,
                    "feedback_samples": len(combined.extras),
                    "base_samples": len(combined.base),
                    "history": history,
                    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seed": seed,
                },
            },
            epoch_ckpt,
        )
        tqdm.write(f"  epoch {epoch}/{epochs}  loss {stats['loss']:.4f}  "
                   f"acc {stats['accuracy']:.4f}  ckpt={epoch_ckpt.name}")
        if prev_acc is not None and prev_loss is not None:
            acc_delta = abs(stats["accuracy"] - prev_acc)
            loss_rel_delta = abs(stats["loss"] - prev_loss) / max(abs(prev_loss), 1e-8)
            if acc_delta <= stability_tol and loss_rel_delta <= max(stability_tol * 5.0, 0.05):
                stable_epochs += 1
            else:
                stable_epochs = 0
            if stable_epochs >= stability_patience:
                tqdm.write(
                    f"  early exit at epoch {epoch}: model metric stable within "
                    f"+-{stability_tol * 100:.1f}% (delta_acc={acc_delta * 100:.2f}%, "
                    f"delta_loss_rel={loss_rel_delta * 100:.2f}%)"
                )
                break
        prev_acc = stats["accuracy"]
        prev_loss = stats["loss"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_order": cfg.UNIFIED_CLASSES,
            "ndvi_is_proxy": bool(cfg.NDVI_IS_PROXY),
            "retrain": {
                "from": str(base_path),
                "epochs": epochs,
                "lr": lr,
                "weight_decay": WEIGHT_DECAY,
                "feedback_samples": len(combined.extras),
                "base_samples": len(combined.base),
                "history": history,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": seed,
            },
        },
        out_path,
    )
    log(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) "
        f"in {time.perf_counter() - started:.0f}s")
    return {"history": history, "seconds": round(time.perf_counter() - started, 1)}


# ---------------------------------------------------------------------------
# 5. Evaluation across versions
# ---------------------------------------------------------------------------
def eval_loader(dataset: Any, limit: int, seed: int, batch_size: int) -> DataLoader:
    """Seeded subset of the test split (same rows for every version)."""
    if limit and 0 < limit < len(dataset):
        rng = np.random.default_rng(seed)
        indices = sorted(rng.permutation(len(dataset))[:limit].tolist())
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_versions(
    versions: list[tuple[str, Path]], test_ds: Any, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Accuracy/macro-F1 of every checkpoint on the identical test subset."""
    loader = eval_loader(test_ds, limit, seed, DEFAULT_BATCH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name, path in versions:
        model = load_model(path)
        out = predict(model, loader, device)
        metrics = metrics_block(out["y_true"], out["y_pred"], title=name)
        results.append({
            "version": name,
            "checkpoint": str(path),
            "n_eval": int(len(out["y_true"])),
            "accuracy": round(float(metrics["accuracy"]), 4),
            "macro_f1": round(float(metrics["macro_f1"]), 4),
            "per_class_f1": {
                cls: round(float(values["f1"]), 4)
                for cls, values in metrics["per_class"].items()
            },
        })
        log(f"  {name}: acc={results[-1]['accuracy']} "
            f"macro_f1={results[-1]['macro_f1']} (n={results[-1]['n_eval']})")
    return results


def plot_curve(results: list[dict[str, Any]], path: Path) -> None:
    """Accuracy / macro-F1 across retraining cycles."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["version"] for r in results]
        accuracy = [r["accuracy"] for r in results]
        f1 = [r["macro_f1"] for r in results]
        x = range(len(names))

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(x, accuracy, marker="o", color="#2E7D32", label="accuracy")
        ax.plot(x, f1, marker="s", color="#EF6C00", label="macro F1")
        for xi, a, f in zip(x, accuracy, f1):
            ax.annotate(f"{a:.3f}", (xi, a), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color="#2E7D32")
            ax.annotate(f"{f:.3f}", (xi, f), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=8, color="#EF6C00")
        ax.set_xticks(list(x), names)
        ax.set_ylabel("score on held-out test subset")
        ax.set_xlabel("model version (v1 = pre-feedback baseline)")
        ax.set_title("Feedback-driven retraining: improvement over cycles")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_ylim(min(min(accuracy), min(f1)) - 0.05,
                    max(max(accuracy), max(f1)) + 0.05)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log(f"curve -> {path}")
    except Exception as err:
        log(f"WARNING: could not write curve ({err})")


def print_comparison(results: list[dict[str, Any]]) -> None:
    """Console table for --compare / post-run."""
    print()
    print("-" * 62)
    print(f"{'version':<10}{'accuracy':>12}{'macro F1':>12}{'eval n':>10}")
    print("-" * 62)
    for row in results:
        print(f"{row['version']:<10}{row['accuracy']:>12.4f}"
              f"{row['macro_f1']:>12.4f}{row['n_eval']:>10}")
    print("-" * 62)
    if len(results) >= 2:
        first, last = results[0], results[-1]
        print(f"delta since {first['version']}: "
              f"accuracy {last['accuracy'] - first['accuracy']:+.4f}, "
              f"macro F1 {last['macro_f1'] - first['macro_f1']:+.4f}")
    print()


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage-4.2b CLI options."""
    parser = argparse.ArgumentParser(
        description="Fine-tune KisanGPT on farmer feedback (v1 -> v2 -> ...)."
    )
    parser.add_argument("--run", action="store_true",
                        help="gate on retrain_flag, fine-tune, evaluate, save")
    parser.add_argument("--compare", action="store_true",
                        help="evaluate existing versions + write curve/comparison")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="fine-tune epochs, 5-10 (default 8)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="low learning rate (default 1e-5)")
    parser.add_argument("--max-original", type=int, default=0,
                        help="cap original train samples for a quick cycle "
                             "(0 = full train split)")
    parser.add_argument("--eval-limit", type=int, default=DEFAULT_EVAL_LIMIT,
                        help="seeded test subset per version (0 = full split)")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    return parser.parse_args(argv)


def run_compare(args: argparse.Namespace, datasets: dict[str, Any] | None = None) -> int:
    """Evaluate all existing versions and write comparison + curve."""
    versions = version_checkpoints()
    if not versions:
        log(f"ERROR: no checkpoints found (expected {BASE_CHECKPOINT}). "
            "Train first: python attention_fusion.py")
        return 1
    if datasets is None:
        log("rebuilding Stage-1A test split for evaluation ...")
        datasets, _, _, _ = build_stage1(args.seed)
    results = evaluate_versions(versions, datasets["test"],
                                args.eval_limit, args.seed)
    print_comparison(results)
    plot_curve(results, CURVE_PLOT)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "eval": {
            "n_per_version": results[0]["n_eval"],
            "split": "test",
            "note": "identical seeded subset for every version",
        },
        "versions": results,
        "latest": results[-1],
        "delta_vs_v1": {
            "accuracy": round(results[-1]["accuracy"] - results[0]["accuracy"], 4),
            "macro_f1": round(results[-1]["macro_f1"] - results[0]["macro_f1"], 4),
        },
        "curve_plot": str(CURVE_PLOT),
        "ndvi_is_proxy": bool(cfg.NDVI_IS_PROXY),
    }
    COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"comparison -> {COMPARISON_JSON}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    if not args.run and not args.compare:
        args.run = True  # default action = full retrain (spec: python retrain.py --run)

    if args.run:
        # --- 1. gate -------------------------------------------------------
        flag = read_flag()
        if flag is None:
            log(f"ERROR: {FLAG_JSON} not found. Collect feedback in the "
                "dashboard, then run: python feedback_loop.py --check-retrain")
            return 1
        if not flag.get("retrain_flag", False):
            log("retrain_flag is not set - nothing to do. "
                f"({flag.get('next_step', 'Collect more feedback first.')})")
            log(f"rule: {flag.get('rule', 'n/a')}")
            return 1

        # --- 2. feedback -> samples ---------------------------------------
        set_seed(args.seed)
        log("rebuilding Stage-1A split and matching feedback rows ...")
        datasets, table, _, _ = build_stage1(args.seed)
        df = load_log(FEEDBACK_LOG)
        corrections, reinforcements, skipped = match_feedback(df, table, datasets)
        log(f"matched: {len(corrections)} corrections, "
            f"{len(reinforcements)} reinforcements | skipped: {skipped}")
        if not corrections and not reinforcements:
            log("ERROR: the retrain flag is set but no feedback row matches a "
                "train/val sample. Check image_name values in feedback_log.csv.")
            return 1

        # --- 3. fine-tune --------------------------------------------------
        base = datasets["train"]
        if args.max_original and args.max_original < len(base):
            rng = np.random.default_rng(args.seed)
            base = Subset(base, sorted(
                rng.permutation(len(base))[:args.max_original].tolist()))
            log(f"original train capped at {len(base)} samples (--max-original)")
        combined = CombinedDataset(base, corrections + reinforcements)
        weights = np.concatenate([
            np.full(len(base), 1.0),
            np.full(len(corrections), CORRECTION_OVERSAMPLE),
            np.full(len(reinforcements), REINFORCE_WEIGHT),
        ])
        versions = version_checkpoints()
        source_name, source_path = versions[-1]
        out_path = next_checkpoint()
        training = fine_tune(source_path, combined, weights, args.epochs,
                             args.lr, args.seed, out_path)

        # --- 4. compare + save --------------------------------------------
        results = evaluate_versions(
            [(name, path) for name, path in version_checkpoints()],
            datasets["test"], args.eval_limit, args.seed,
        )
        print_comparison(results)
        plot_curve(results, CURVE_PLOT)
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
            "training": {
                "from_version": source_name,
                "to_version": out_path.stem.replace("unified_model_", ""),
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": WEIGHT_DECAY,
                "correction_oversample": CORRECTION_OVERSAMPLE,
                "reinforce_weight": REINFORCE_WEIGHT,
                "original_samples": len(base),
                "corrections": len(corrections),
                "reinforcements": len(reinforcements),
                "skipped": skipped,
                "seconds": training["seconds"],
                "history": training["history"],
                "eval_note": "test rows excluded from training "
                             "(split-restricted image pools)",
            },
            "eval": {
                "n_per_version": results[0]["n_eval"],
                "split": "test",
                "note": "identical seeded subset for every version",
            },
            "versions": results,
            "latest": results[-1],
            "delta_vs_v1": {
                "accuracy": round(results[-1]["accuracy"] - results[0]["accuracy"], 4),
                "macro_f1": round(results[-1]["macro_f1"] - results[0]["macro_f1"], 4),
            },
            "curve_plot": str(CURVE_PLOT),
            "next_step": "python feedback_loop.py --check-retrain "
                         "(recomputes the flag from the current log)",
        }
        COMPARISON_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"comparison -> {COMPARISON_JSON}")
        return 0

    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
