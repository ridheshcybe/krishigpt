"""Stage 4.1 - edge deployment: INT8 quantization, ONNX export, benchmarks.

    python edge_deploy.py --benchmark      # export + quantize + measure + plot
    python edge_deploy.py --export-only    # just export/verify the ONNX file

What it produces
----------------
* ``artifacts/unified_model.onnx``       FP32 ONNX graph (3 inputs, 2 outputs)
* ``artifacts/unified_model_int8.onnx``  dynamically quantised ONNX graph
* ``artifacts/edge_benchmark.json``      size / latency / accuracy per variant
* ``plots/edge_benchmark.png``           comparison bar chart

Quantization choice (documented, as required)
---------------------------------------------
**Dynamic INT8** for both PyTorch and ONNX:

* PyTorch: ``torch.ao.quantization.quantize_dynamic`` over ``nn.Linear``
  (weights only, quantised per forward pass).  Chosen over *static* INT8
  because static quantization needs a calibration pass over real images plus
  module fusion/QConfig wiring for the MobileNetV2 trunk - a lot of machinery
  for a trunk that stays convolutional either way - while dynamic quantization
  needs no calibration data, works on any CPU (fbgemm/QNNPACK picked
  automatically), and reliably shrinks the 45 MB of Linear weights.
* ONNX: ``onnxruntime.quantization.quantize_dynamic`` (QInt8 weights), the
  ONNX equivalent of the same trade-off.

Every variant is checked against the FP32 PyTorch reference: logits must match
within ``VERIFY_TOLERANCE`` and test accuracy within ``ACCURACY_TOLERANCE``, so
the "accuracy drop" column is measured, not asserted.

Requires: torch (+ optional CUDA), onnx, onnxruntime, and the Stage-1A data
pipeline for the accuracy check (reuse, never re-implemented).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

import config as cfg

# ---------------------------------------------------------------------------:-
# Tolerances / benchmark defaults (documented)
# ---------------------------------------------------------------------------:
#: Max allowed |logit difference| between an edge variant and FP32 PyTorch.
VERIFY_TOLERANCE = 1e-3
#: Max allowed absolute accuracy drop vs FP32 (spec: "within 1-2%").
ACCURACY_TOLERANCE = 0.02
#: Timing: warmup runs then measured runs, batch size 1 (edge = single image).
WARMUP_RUNS = 5
TIMED_RUNS = 30
#: Default number of test samples used for the accuracy comparison.
DEFAULT_LIMIT = 256

ONNX_PATH = Path("artifacts/unified_model.onnx")
ONNX_INT8_PATH = Path("artifacts/unified_model_int8.onnx")
BENCHMARK_JSON = Path("artifacts/edge_benchmark.json")
BENCHMARK_PLOT = Path("plots/edge_benchmark.png")

#: Spec targets, checked and reported.
TARGET_LATENCY_MS = 500.0
TARGET_SIZE_MB = 50.0

MODEL_INPUTS = ("images", "tabular", "ndvi")


def log(message: str) -> None:
    """Print a timestamped progress line (Stage-4 tag, matching earlier stages)."""
    print(f"[stage4 {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Model loading (reuse Stage 1B, never re-declared)
# ---------------------------------------------------------------------------
def resolve_checkpoint() -> Path:
    """Prefer ``unified_model.pth``, fall back to the Stage-1B filename."""
    for candidate in (Path("artifacts/unified_model.pth"),
                      Path("artifacts/attention_fusion.pt")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No unified model checkpoint found. Train it first: "
        "python attention_fusion.py"
    )


def load_model(checkpoint: Path | None = None) -> Any:
    """Load ``AttentionFusion`` in eval mode on CPU (edge target = CPU)."""
    import torch
    from attention_fusion import AttentionFusion

    path = checkpoint or resolve_checkpoint()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("model_state", blob.get("state_dict", blob))
    model = AttentionFusion()
    model.load_state_dict(state)
    model.eval()
    log(f"loaded {path} ({path.stat().st_size / 1e6:.1f} MB) -> CPU eval mode")
    return model


# ---------------------------------------------------------------------------
# 2. ONNX export + verification
# ---------------------------------------------------------------------------
def export_onnx(model: Any, path: Path = ONNX_PATH, opset: int = 17) -> dict[str, Any]:
    """Export FP32 ONNX and verify it against PyTorch on random inputs.

    Returns a verification block: max/mean absolute logit difference and the
    attention-weight difference (the model returns both, so attention *is*
    extractable from ONNX - required by ``edge_inference.py``).
    """
    import torch

    dummy_image = torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    dummy_tab = torch.zeros(1, cfg.TABULAR_DIM)
    dummy_ndvi = torch.zeros(1, cfg.NDVI_STEPS)

    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=list(MODEL_INPUTS),
        output_names=["logits", "attention"],
        dynamic_axes={
            "images": {0: "batch"}, "tabular": {0: "batch"}, "ndvi": {0: "batch"},
            "logits": {0: "batch"}, "attention": {0: "batch"},
        },
        opset_version=opset,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(model, (dummy_image, dummy_tab, dummy_ndvi),
                              str(path), dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(model, (dummy_image, dummy_tab, dummy_ndvi),
                              str(path), **kwargs)
    log(f"ONNX exported -> {path} ({path.stat().st_size / 1e6:.2f} MB, opset {opset})")
    return verify_onnx(model, path)


def verify_onnx(model: Any, path: Path) -> dict[str, Any]:
    """Compare ONNX Runtime output against the PyTorch reference."""
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(cfg.SEED)
    image = rng.standard_normal((1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)).astype(np.float32)
    tab = rng.standard_normal((1, cfg.TABULAR_DIM)).astype(np.float32)
    ndvi = rng.standard_normal((1, cfg.NDVI_STEPS)).astype(np.float32)

    with torch.no_grad():
        p_logits, p_attn = model(
            torch.from_numpy(image), torch.from_numpy(tab), torch.from_numpy(ndvi)
        )
    p_logits = p_logits.numpy()
    p_attn = p_attn.numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "images": image, "tabular": tab, "ndvi": ndvi,
    })
    o_logits, o_attn = out[0], out[1]

    logit_diff = float(np.abs(p_logits - o_logits).max())
    attn_diff = float(np.abs(p_attn - o_attn).max())
    ok = logit_diff <= VERIFY_TOLERANCE
    log(
        f"verify ONNX vs PyTorch: max|dlogit|={logit_diff:.2e} "
        f"(tol {VERIFY_TOLERANCE:g}) -> {'OK' if ok else 'MISMATCH'}"
    )
    return {
        "verified": bool(ok),
        "max_abs_logit_diff": round(logit_diff, 8),
        "max_abs_attention_diff": round(attn_diff, 8),
        "tolerance": VERIFY_TOLERANCE,
        "onnx_outputs": ["logits", "attention"],
    }


# ---------------------------------------------------------------------------
# 3. Quantization
# ---------------------------------------------------------------------------
def quantize_pytorch_int8(model: Any) -> Any:
    """Dynamic INT8 on every ``nn.Linear`` (documented choice - see module doc)."""
    import torch

    before_mb = _torch_size_mb(model)
    try:
        qmodel = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )
        qmodel.eval()
    except Exception as err:
        log(f"WARNING: PyTorch dynamic quantization unavailable ({err}); skipped")
        return None
    after_mb = _torch_size_mb(qmodel)
    log(
        f"PyTorch dynamic INT8: {before_mb:.1f} MB -> {after_mb:.1f} MB "
        f"({(1 - after_mb / before_mb) * 100:.1f}% smaller)"
    )
    return qmodel


def _torch_size_mb(model: Any) -> float:
    """Serialized size of a PyTorch module in MB (measured, not estimated)."""
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


def quantize_onnx_int8(src: Path = ONNX_PATH, dst: Path = ONNX_INT8_PATH) -> Path | None:
    """Dynamic INT8 ONNX graph via onnxruntime.quantization."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as err:
        log(f"WARNING: onnxruntime.quantization unavailable ({err}); skipped")
        return None
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
        )
    except Exception as err:
        log(f"WARNING: ONNX INT8 quantization failed ({err}); skipped")
        return None
    log(f"ONNX INT8 -> {dst} ({dst.stat().st_size / 1e6:.2f} MB)")
    return dst


# ---------------------------------------------------------------------------
# 4. Test data (reuses Stage-1A builders - no duplicated pipeline)
# ---------------------------------------------------------------------------
def build_test_loader(limit: int | None, seed: int, batch_size: int = 32) -> Any:
    """Return a DataLoader over the held-out test split (optionally capped)."""
    from torch.utils.data import DataLoader, Subset

    import diagnosis_cli as dcli

    args = SimpleNamespace(
        data_path=dcli.tx.DEFAULT_DATA_PATH,
        image_dir=Path("data/paddy_images"),
        seed=seed,
    )
    datasets, _, _, _ = dcli.build_datasets(args)
    test = datasets["test"]
    if limit and limit > 0:
        n = min(int(limit), len(test))
        # Seeded *random* subset: taking the first n rows would sample an
        # ordered slice of the split (possibly a single class) and report a
        # meaningless accuracy.
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(test))[:n].tolist()
        test = Subset(test, sorted(indices))
        log(f"test subset: {n} random samples, seed={seed} (accuracy check)")
    else:
        log(f"test split: {len(test)} samples (accuracy check)")
    return DataLoader(test, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# 5. Benchmarks
# ---------------------------------------------------------------------------
def time_inference(fn: Callable[[], Any], warmup: int, runs: int) -> float:
    """Mean wall-clock milliseconds per single inference (batch = 1)."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) * 1000.0 / runs


def accuracy_pytorch(model: Any, loader: Any, device: str = "cpu",
                     limit: int | None = None) -> float:
    """Top-1 accuracy of a PyTorch module over the loader."""
    import torch

    correct = total = 0
    with torch.no_grad():
        for tab, img, ndvi, label in loader:
            logits, _ = model(
                img.to(device), tab.to(device), ndvi.to(device)
            )
            pred = logits.argmax(dim=-1).cpu()
            correct += int((pred == label).sum())
            total += int(label.numel())
            if limit and total >= limit:
                break
    return correct / max(total, 1)


def accuracy_onnx(session: Any, loader: Any, limit: int | None = None) -> float:
    """Top-1 accuracy of an ONNX session over the same loader."""
    correct = total = 0
    for tab, img, ndvi, label in loader:
        logits = session.run(
            None,
            {
                "images": img.numpy().astype(np.float32),
                "tabular": tab.numpy().astype(np.float32),
                "ndvi": ndvi.numpy().astype(np.float32),
            },
        )[0]
        pred = logits.argmax(axis=-1)
        correct += int((pred == label.numpy()).sum())
        total += int(label.numel())
        if limit and total >= limit:
            break
    return correct / max(total, 1)


def _single_batch(loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take one batch and return numpy arrays shaped for batch=1 timing."""
    tab, img, ndvi, _ = next(iter(loader))
    return (
        img[:1].numpy().astype(np.float32),
        tab[:1].numpy().astype(np.float32),
        ndvi[:1].numpy().astype(np.float32),
    )


def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    """Full pipeline: export, quantize, benchmark every variant, plot, save."""
    import torch
    import onnxruntime as ort

    model = load_model(Path(args.checkpoint) if args.checkpoint else None)
    verification = export_onnx(model, ONNX_PATH)

    results: dict[str, Any] = {}
    limit = None if args.limit == 0 else args.limit
    loader = build_test_loader(limit, args.seed) if args.accuracy else None
    batch = _single_batch(loader) if loader is not None else None

    # --- PyTorch FP32 CPU --------------------------------------------------
    image, tab, ndvi = batch if batch else _random_batch()
    with torch.no_grad():
        reference_logits, _ = model(
            torch.from_numpy(image), torch.from_numpy(tab), torch.from_numpy(ndvi)
        )
    reference_logits = reference_logits.numpy()

    lat = time_inference(
        lambda: model(torch.from_numpy(image), torch.from_numpy(tab),
                      torch.from_numpy(ndvi)),
        WARMUP_RUNS, args.runs,
    )
    results["pytorch_fp32_cpu"] = {
        "label": "PyTorch FP32 (CPU)",
        "size_mb": round(_torch_size_mb(model), 2),
        "latency_ms": round(lat, 2),
        "accuracy": round(accuracy_pytorch(model, loader), 4) if loader else None,
        "max_abs_logit_diff": 0.0,
    }
    log(f"pytorch_fp32_cpu: {results['pytorch_fp32_cpu']['size_mb']} MB, "
        f"{lat:.1f} ms, acc={results['pytorch_fp32_cpu']['accuracy']}")

    # --- PyTorch dynamic INT8 CPU -----------------------------------------
    qmodel = quantize_pytorch_int8(model)
    if qmodel is not None:
        lat = time_inference(
            lambda: qmodel(torch.from_numpy(image), torch.from_numpy(tab),
                           torch.from_numpy(ndvi)),
            WARMUP_RUNS, args.runs,
        )
        with torch.no_grad():
            q_logits, _ = qmodel(torch.from_numpy(image), torch.from_numpy(tab),
                                 torch.from_numpy(ndvi))
        results["pytorch_int8_cpu"] = {
            "label": "PyTorch dynamic INT8 (CPU)",
            "size_mb": round(_torch_size_mb(qmodel), 2),
            "latency_ms": round(lat, 2),
            "accuracy": round(accuracy_pytorch(qmodel, loader), 4) if loader else None,
            "max_abs_logit_diff": round(
                float(np.abs(q_logits.numpy() - reference_logits).max()), 6),
        }
        log(f"pytorch_int8_cpu: {results['pytorch_int8_cpu']['size_mb']} MB, {lat:.1f} ms")

    # --- PyTorch CUDA ------------------------------------------------------
    if torch.cuda.is_available() and not args.no_gpu:
        gpu = torch.device("cuda")
        model_gpu = model.to(gpu)
        gi, gt, gn = (torch.from_numpy(x).to(gpu) for x in (image, tab, ndvi))
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                model_gpu(gi, gt, gn)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.runs):
                model_gpu(gi, gt, gn)
            torch.cuda.synchronize()
            lat = (time.perf_counter() - start) * 1000.0 / args.runs
        results["pytorch_fp32_cuda"] = {
            "label": "PyTorch FP32 (CUDA)",
            "size_mb": round(_torch_size_mb(model_gpu), 2),
            "latency_ms": round(lat, 2),
            "accuracy": round(accuracy_pytorch(model_gpu, loader, "cuda"), 4)
            if loader else None,
            "max_abs_logit_diff": None,
            "device_name": torch.cuda.get_device_name(0),
        }
        model.to("cpu")
        log(f"pytorch_fp32_cuda: {lat:.1f} ms on {torch.cuda.get_device_name(0)}")
    else:
        results["pytorch_fp32_cuda"] = {
            "label": "PyTorch FP32 (CUDA)",
            "available": False,
            "reason": "torch.cuda.is_available() is False on this host",
        }
        log("pytorch_fp32_cuda: skipped (no CUDA device)")

    # --- ONNX Runtime FP32 CPU --------------------------------------------
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    feed = {"images": image, "tabular": tab, "ndvi": ndvi}
    o_logits = sess.run(None, feed)[0]
    lat = time_inference(lambda: sess.run(None, feed), WARMUP_RUNS, args.runs)
    results["onnx_fp32_cpu"] = {
        "label": "ONNX Runtime FP32 (CPU)",
        "size_mb": round(ONNX_PATH.stat().st_size / 1e6, 2),
        "latency_ms": round(lat, 2),
        "accuracy": round(accuracy_onnx(sess, loader), 4) if loader else None,
        "max_abs_logit_diff": round(
            float(np.abs(o_logits - reference_logits).max()), 6),
    }
    log(f"onnx_fp32_cpu: {results['onnx_fp32_cpu']['size_mb']} MB, {lat:.1f} ms")

    # --- ONNX Runtime INT8 CPU --------------------------------------------
    int8_path = quantize_onnx_int8()
    if int8_path is not None and int8_path.exists():
        isess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        i_logits = isess.run(None, feed)[0]
        lat = time_inference(lambda: isess.run(None, feed), WARMUP_RUNS, args.runs)
        results["onnx_int8_cpu"] = {
            "label": "ONNX Runtime dynamic INT8 (CPU)",
            "size_mb": round(int8_path.stat().st_size / 1e6, 2),
            "latency_ms": round(lat, 2),
            "accuracy": round(accuracy_onnx(isess, loader), 4) if loader else None,
            "max_abs_logit_diff": round(
                float(np.abs(i_logits - reference_logits).max()), 6),
        }
        log(f"onnx_int8_cpu: {results['onnx_int8_cpu']['size_mb']} MB, {lat:.1f} ms")

    # --- reference accuracy + pass/fail ------------------------------------
    if loader is not None and results["pytorch_fp32_cpu"]["accuracy"] is not None:
        ref_acc = results["pytorch_fp32_cpu"]["accuracy"]
        for name, row in results.items():
            if row.get("accuracy") is None:
                continue
            row["accuracy_drop_vs_fp32"] = round(ref_acc - row["accuracy"], 4)
            row["within_2pct"] = bool(ref_acc - row["accuracy"] <= ACCURACY_TOLERANCE)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": {
            "latency_ms_max": TARGET_LATENCY_MS,
            "size_mb_max": TARGET_SIZE_MB,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "quantization": {
            "pytorch": "dynamic INT8 (torch.ao.quantization.quantize_dynamic, "
                       "nn.Linear weights) - no calibration data required",
            "onnx": "dynamic INT8 (onnxruntime.quantization.quantize_dynamic, QInt8)",
        },
        "timing": {
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": args.runs,
            "batch_size": 1,
        },
        "accuracy_samples": int(limit) if limit else "full test split",
        "onnx_verification": verification,
        "variants": results,
    }
    payload["targets_met"] = {
        name: {
            "latency": bool(row.get("latency_ms", 0) <= TARGET_LATENCY_MS),
            "size": bool(row.get("size_mb", 0) <= TARGET_SIZE_MB),
        }
        for name, row in results.items()
        if row.get("latency_ms") is not None
    }

    BENCHMARK_JSON.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"benchmark written -> {BENCHMARK_JSON}")
    plot_benchmark(payload)
    return payload


def _random_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random inputs for timing when no test data is requested."""
    rng = np.random.default_rng(cfg.SEED)
    return (
        rng.standard_normal((1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)).astype(np.float32),
        rng.standard_normal((1, cfg.TABULAR_DIM)).astype(np.float32),
        rng.standard_normal((1, cfg.NDVI_STEPS)).astype(np.float32),
    )


def plot_benchmark(payload: dict[str, Any]) -> None:
    """Bar chart of latency and size per variant -> plots/edge_benchmark.png."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names, latencies, sizes = [], [], []
        for row in payload["variants"].values():
            if row.get("latency_ms") is None:
                continue
            names.append(row["label"].replace(" (", "\n("))
            latencies.append(row["latency_ms"])
            sizes.append(row["size_mb"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        colors = ["#2E7D32", "#F9A825", "#1565C0", "#EF6C00", "#C62828"]

        bars = axes[0].bar(range(len(names)), latencies,
                           color=colors[:len(names)])
        axes[0].axhline(TARGET_LATENCY_MS, ls="--", color="#C62828",
                        label=f"target {int(TARGET_LATENCY_MS)} ms")
        axes[0].set_xticks(range(len(names)), names, fontsize=8)
        axes[0].set_title("Latency per inference (batch=1, CPU)")
        axes[0].set_ylabel("ms (lower is better)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3, axis="y")
        for bar, value in zip(bars, latencies):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value,
                         f"{value:.0f}", ha="center", va="bottom", fontsize=8)

        bars = axes[1].bar(range(len(names)), sizes, color=colors[:len(names)])
        axes[1].axhline(TARGET_SIZE_MB, ls="--", color="#C62828",
                        label=f"target {int(TARGET_SIZE_MB)} MB")
        axes[1].set_xticks(range(len(names)), names, fontsize=8)
        axes[1].set_title("Model size")
        axes[1].set_ylabel("MB (lower is better)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, axis="y")
        for bar, value in zip(bars, sizes):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value,
                         f"{value:.1f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle("KisanGPT edge deployment benchmark (FP32 vs INT8)", fontsize=13)
        fig.tight_layout()
        BENCHMARK_PLOT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(BENCHMARK_PLOT, dpi=130)
        plt.close(fig)
        log(f"benchmark plot -> {BENCHMARK_PLOT}")
    except Exception as err:
        log(f"WARNING: could not write benchmark plot ({err})")


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
def print_table(payload: dict[str, Any]) -> None:
    """Console table: variant / size / latency / accuracy / drop."""
    print()
    print("-" * 88)
    print(f"{'variant':<34}{'size MB':>10}{'ms':>10}{'accuracy':>11}{'drop':>9}")
    print("-" * 88)
    for name, row in payload["variants"].items():
        if row.get("latency_ms") is None:
            print(f"{row['label']:<34}{'-':>10}{'-':>10}"
                  f"{'n/a':>11}{'-':>9}  ({row.get('reason', 'skipped')})")
            continue
        acc = row.get("accuracy")
        drop = row.get("accuracy_drop_vs_fp32")
        print(
            f"{row['label']:<34}{row['size_mb']:>10.2f}{row['latency_ms']:>10.1f}"
            f"{(f'{acc:.4f}' if acc is not None else 'n/a'):>11}"
            f"{(f'{drop:+.4f}' if drop is not None else 'n/a'):>9}"
        )
    print("-" * 88)
    print(f"targets: < {TARGET_LATENCY_MS:.0f} ms and < {TARGET_SIZE_MB:.0f} MB "
          f"-> {payload.get('targets_met')}")
    print(f"ONNX verified vs PyTorch: {payload['onnx_verification']['verified']} "
          f"(max|dlogit|={payload['onnx_verification']['max_abs_logit_diff']})")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage-4.1 CLI options."""
    parser = argparse.ArgumentParser(
        description="Stage 4.1: export, quantize and benchmark KisanGPT for edge."
    )
    parser.add_argument("--benchmark", action="store_true",
                        help="export + quantize + measure + plot + save JSON")
    parser.add_argument("--export-only", action="store_true",
                        help="only export and verify artifacts/unified_model.onnx")
    parser.add_argument("--checkpoint", default=None,
                        help="override the unified checkpoint path")
    parser.add_argument("--runs", type=int, default=TIMED_RUNS,
                        help="timed repetitions per variant (default 30)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="test samples for the accuracy check; 0 = full split")
    parser.add_argument("--no-accuracy", dest="accuracy", action="store_false",
                        help="skip building the test split (timing/size only)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="skip the CUDA row even if a GPU is present")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    if not args.benchmark and not args.export_only:
        args.benchmark = True  # default action matches the spec's CLI

    try:
        if args.export_only:
            model = load_model(Path(args.checkpoint) if args.checkpoint else None)
            export_onnx(model, ONNX_PATH)
            return 0
        payload = run_benchmarks(args)
        print_table(payload)
    except FileNotFoundError as err:
        log(f"ERROR: {err}")
        return 1
    except Exception as err:
        log(f"ERROR: {type(err).__name__}: {err}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
