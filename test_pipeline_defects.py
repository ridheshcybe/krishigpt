"""Comprehensive test suite for trajectory pipeline fixes and narrow verification pass.

Covers:
1. Fallback onset timing conversion:
   - actual_onsets is 1-based; fallback alert['day'] is 0-based step k.
   - k=1 (predicted day 2) and actual_onset=2 are the SAME day, producing timing difference 0, not 1.
   - Distinction between dedicated binary head (issuance lead time) and fallback scan (onset timing difference).
2. Production compute_alert_metrics test covering all denominator combinations:
   - No alerts, events present: precision undefined (None), recall 0.0.
   - Alerts, no events: precision 0.0, recall undefined (None).
   - Neither: precision and recall both undefined (None).
   - Matched true positives: normal defined values.
   - No matched true positives: mean_lead_days, median, std, pct_before undefined (None) even when precision/recall have defined zero values.
   - Format test verifying write_report handles each undefined metric gracefully with "N/A (undefined denominator)".
3. Evaluator production code tests:
   - Evaluator's real build_parser().
   - Real TrajectoryDataset on a tiny fixture for history/target slicing and Day-1 indexing.
   - Real load_split with a temporary split fixture, asserting loaded split path, field membership, and hash association.
   - write_report formatting with realistic stub payloads.
4. Strict read-only loader safety:
   - Missing data cache raises FileNotFoundError without generating data or writing files.
   - Missing/mismatched split file raises FileNotFoundError/ValueError in read_only mode without generating files.

Boundaries:
No training, no model checkpoint modifications, no full evaluations, zero writes to artifacts/.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import config as cfg
from stage1_config import UNIFIED_CLASSES
from trajectory_model import detect_onset_alerts
from evaluate_trajectory import (
    build_class_mapping,
    build_parser,
    compute_alert_metrics,
    write_report,
)
from trajectory_data import TrajectoryDataset, build_dataset, load_split


def test_1_fallback_timing_and_origin_distinction():
    """Verify fallback onset timing conversion and head distinction."""
    # Case A: Fallback scan
    # Model predicts stress on step index 1 (which is 1-based Day 2).
    # Actual onset occurs on 1-based Day 2.
    probs_fb = torch.zeros((1, 3, 4))
    probs_fb[0, 0, 0] = 1.0  # day 0 healthy
    probs_fb[0, 1, 1] = 0.9  # day 1 water stress
    probs_fb[0, 2, 1] = 0.9  # day 2 water stress

    alerts_fb = detect_onset_alerts(probs_fb, horizon=3, early_probs=None, threshold=0.50)
    assert len(alerts_fb) == 1
    assert alerts_fb[0]["alerted"] is True
    assert alerts_fb[0]["alert_origin"] == "fallback_scan"
    assert alerts_fb[0]["day"] == 1  # 0-based step index
    assert alerts_fb[0]["predicted_onset_day"] == 2  # 1-based predicted onset day

    # Metric evaluation for Fallback scan:
    actual_onsets = [2]
    actual_pos = np.array([True], dtype=bool)
    m_fb = compute_alert_metrics(alerts_fb, actual_onsets, actual_pos)
    # The actual onset is Day 2, predicted onset is Day 2 -> difference is exactly 0.0!
    assert m_fb["lead_times"] == [0.0]
    assert m_fb["mean_lead_days"] == 0.0
    assert m_fb["pct_on_onset"] == 100.0
    assert m_fb["alert_origin"] == "fallback_scan"
    assert "actual_onset_day - (predicted_onset_step + 1)" in m_fb["lead_definition"]

    # Case B: Dedicated binary head
    # Alert is issued at t=0 (forecast issuance time), actual onset is Day 2.
    probs_bin = torch.zeros((1, 3, 4))
    probs_bin[0, :, 0] = 1.0
    early_prob = torch.tensor([0.85])

    alerts_bin = detect_onset_alerts(probs_bin, horizon=3, early_probs=early_prob, threshold=0.50)
    assert alerts_bin[0]["alerted"] is True
    assert alerts_bin[0]["alert_origin"] == "binary_head"
    assert alerts_bin[0]["day"] == 0

    m_bin = compute_alert_metrics(alerts_bin, actual_onsets, actual_pos)
    # Lead time = actual_onset - issuance_day = 2 - 0 = 2 days
    assert m_bin["lead_times"] == [2.0]
    assert m_bin["mean_lead_days"] == 2.0
    assert m_bin["pct_before_onset"] == 100.0
    assert m_bin["alert_origin"] == "binary_head"
    assert "actual_onset_day - alert_issuance_day" in m_bin["lead_definition"]


def test_2_production_metric_denominators_and_formatting():
    """Exercise compute_alert_metrics on all combinations of denominators."""
    dummy_alert_on = [{"alerted": True, "day": 0, "alert_origin": "binary_head", "message": "Alert"}]
    dummy_alert_off = [{"alerted": False, "day": None, "alert_origin": "binary_head", "message": "None"}]

    # 1. No alerts, events present: precision undefined (None), recall 0.0
    m1 = compute_alert_metrics(dummy_alert_off, [2], np.array([True]))
    assert m1["tp"] == 0 and m1["fp"] == 0 and m1["fn"] == 1 and m1["tn"] == 0
    assert m1["precision"] is None
    assert m1["recall"] == 0.0
    assert m1["f1"] == 0.0  # TP=0 with FN>0: F1 = 2TP/(2TP+FP+FN) = 0.0, defined
    assert m1["mean_lead_days"] is None
    assert m1["pct_before_onset"] is None

    # 2. Alerts, no events: precision 0.0, recall undefined (None)
    m2 = compute_alert_metrics(dummy_alert_on, [None], np.array([False]))
    assert m2["tp"] == 0 and m2["fp"] == 1 and m2["fn"] == 0 and m2["tn"] == 0
    assert m2["precision"] == 0.0
    assert m2["recall"] is None
    assert m2["f1"] == 0.0  # TP=0 with FP>0: defined zero, not undefined
    assert m2["mean_lead_days"] is None
    assert m2["pct_before_onset"] is None

    # 3. Neither: both undefined (None)
    m3 = compute_alert_metrics(dummy_alert_off, [None], np.array([False]))
    assert m3["tp"] == 0 and m3["fp"] == 0 and m3["fn"] == 0 and m3["tn"] == 1
    assert m3["precision"] is None
    assert m3["recall"] is None
    assert m3["f1"] is None  # TP=FP=FN=0: the only undefined-F1 case
    assert m3["mean_lead_days"] is None
    assert m3["pct_before_onset"] is None

    # 4. Matched true positives: normal defined values
    m4 = compute_alert_metrics(dummy_alert_on, [3], np.array([True]))
    assert m4["tp"] == 1 and m4["fp"] == 0 and m4["fn"] == 0 and m4["tn"] == 0
    assert m4["precision"] == 1.0
    assert m4["recall"] == 1.0
    assert m4["f1"] == 1.0
    assert m4["mean_lead_days"] == 3.0
    assert m4["pct_before_onset"] == 100.0

    # 5. Report formatting test with undefined values
    payload = {
        "generated_at": "2026-11-02 12:00:00",
        "split": "test",
        "t_history": 21,
        "t_future": 14,
        "classes": UNIFIED_CLASSES,
        "encoder": "transformer",
        "pooling": "last",
        "best_epoch": 0,
        "proxy_flags": {"ndvi_is_proxy": True, "weather_is_proxy": True, "weather_source": "proxy"},
        "ndvi_regression": {"methods": {"trajectory_model": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "day1_mae": 0.0}}},
        "classification": {"trajectory_model": {"accuracy": 0.0, "macro_f1": 0.0, "mean_step_accuracy": 0.0, "per_class_f1": {}, "step_accuracy": []}},
        "onset_detection": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "n": 3},
        "early_warning": {"best_threshold": 0.5, "selection_rule": "default", "baseline_precision": 0.0},
        "alert_quality": m1,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.txt"
        write_report(payload, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "alert precision (TP/pred): N/A (undefined denominator)" in content
        assert "event recall (TP/actual) : 0.0000" in content
        assert "mean lead time           : N/A (undefined denominator)" in content
        assert "alerts fired BEFORE onset: N/A (undefined denominator)" in content


def test_3_evaluator_real_parser():
    """Exercise evaluate_trajectory's real build_parser()."""
    parser = build_parser()
    args = parser.parse_args([
        "--split", "val",
        "--alert_horizon", "5",
        "--checkpoint", "custom/traj.pth",
        "--data", "custom/traj_data.npz",
        "--split_file", "custom/traj_split.json",
    ])
    assert args.split == "val"
    assert args.alert_horizon == 5
    assert args.checkpoint == "custom/traj.pth"
    assert args.data == "custom/traj_data.npz"
    assert args.split_file == "custom/traj_split.json"


def test_4_real_trajectory_dataset_slicing():
    """Exercise real TrajectoryDataset on a tiny fixture for slicing and Day-1 delta."""
    n_fields = 2
    t_history = 21
    t_future = 14
    t_total = t_history + t_future

    ndvi = np.zeros((n_fields, t_total), dtype=np.float32)
    ndvi[0] = np.linspace(0.8, 0.4, t_total)
    ndvi[1] = np.linspace(0.7, 0.3, t_total)
    weather = np.zeros((n_fields, t_total, 4), dtype=np.float32)
    labels = np.zeros((n_fields, t_total), dtype=np.int64)

    ds = TrajectoryDataset(
        ndvi=ndvi,
        weather=weather,
        labels=labels,
        field_indices=[0, 1],
        t_history=t_history,
        t_future=t_future,
        use_weather=False,
    )
    assert len(ds) == 2
    item0 = ds[0]

    # History is first 21 days (indices 0..20)
    assert item0["x"].shape == (21, 1)
    # Last history ndvi is day index 20
    assert np.isclose(item0["last_ndvi"].item(), ndvi[0, 20])
    # Target ndvi is days 21..34
    assert item0["y_ndvi"].shape == (14,)
    assert np.isclose(item0["y_ndvi"][0].item(), ndvi[0, 21])
    # Delta ndvi is target - last_ndvi
    expected_delta = ndvi[0, 21:] - ndvi[0, 20]
    assert np.allclose(item0["delta_ndvi"].numpy(), expected_delta)


def test_5_real_load_split_and_provenance_hash():
    """Call real load_split with a temporary split fixture and verify hash association."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        split_fixture = tmp_path / "test_split.json"
        split_dict = {
            "train": [0, 1],
            "val": [2],
            "test": [3],
            "seed": 42,
            "split_by": "field",
            "fractions": [0.5, 0.25, 0.25],
            "n_fields": 4,
        }
        split_fixture.write_text(json.dumps(split_dict), encoding="utf-8")

        # Tiny dataset fixture so the split can be validated against the ACTUAL n_fields
        data_fixture = tmp_path / "data.npz"
        np.savez(
            data_fixture,
            ndvi=np.zeros((4, 35), dtype=np.float32),
            meta=np.asarray([4, 21, 14, 42], dtype=np.int64),
        )

        loaded = load_split(seed=42, split_path=split_fixture, dataset_path=data_fixture, read_only=True)
        assert loaded["loaded_path"] == str(split_fixture.resolve())
        assert loaded["train"] == [0, 1]
        assert loaded["val"] == [2]
        assert loaded["test"] == [3]

        # Verify hash calculation of the loaded file
        h = hashlib.sha256(split_fixture.read_bytes()).hexdigest()
        assert hashlib.sha256(Path(loaded["loaded_path"]).read_bytes()).hexdigest() == h


def test_6_strict_read_only_loader_safety():
    """Verify evaluation fails clearly in read_only mode without generating files or modifying disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        missing_data = tmp_path / "non_existent_data.npz"
        missing_split = tmp_path / "non_existent_split.json"

        # 1. Missing data cache in read_only mode must raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Strict read-only load failed"):
            build_dataset(seed=42, cache_path=missing_data, read_only=True)
        assert not missing_data.exists()

        # 2. Missing split file in read_only mode must raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Strict read-only load failed"):
            load_split(seed=42, split_path=missing_split, read_only=True)
        assert not missing_split.exists()

        # 3. Seed mismatch in read_only mode must raise ValueError without modifying file
        bad_seed_split = tmp_path / "mismatch_split.json"
        bad_seed_split.write_text(json.dumps({"seed": 999, "train": [0], "val": [1], "test": [2]}), encoding="utf-8")
        with pytest.raises(ValueError, match="split seed mismatch"):
            load_split(seed=42, split_path=bad_seed_split, read_only=True)




def test_7_origin_stamping_and_mixed_rejection():
    """Every record carries alert_origin; mixed origins are rejected, legacy records excluded."""
    # Fallback no-fire record carries origin
    probs = torch.zeros((1, 3, 4))
    probs[0, :, 0] = 1.0  # all healthy -> no fire
    alerts = detect_onset_alerts(probs, horizon=3, early_probs=None, threshold=0.50)
    assert alerts[0]["alerted"] is False
    assert alerts[0]["alert_origin"] == "fallback_scan"

    # Binary no-fire record carries origin
    alerts_b = detect_onset_alerts(probs, horizon=3, early_probs=torch.tensor([0.10]), threshold=0.50)
    assert alerts_b[0]["alerted"] is False
    assert alerts_b[0]["alert_origin"] == "binary_head"

    # Mixed origins among alerted records are rejected, never pooled
    mixed = [
        {"alerted": True, "day": 0, "alert_origin": "binary_head", "message": ""},
        {"alerted": True, "day": 1, "predicted_onset_step": 1, "predicted_onset_day": 2,
         "alert_origin": "fallback_scan", "message": ""},
    ]
    with pytest.raises(ValueError, match="mixed alert origins"):
        compute_alert_metrics(mixed, [1, 2], np.array([True, True]))

    # A KEYLESS record carrying predicted_onset_day is inferred fallback; combined with an
    # explicit binary_head alert it is a conflict and must be rejected, not pooled.
    conflict = [
        {"alerted": True, "day": 0, "alert_origin": "binary_head", "message": ""},
        {"alerted": True, "day": 1, "predicted_onset_day": 2, "message": ""},
    ]
    with pytest.raises(ValueError, match="mixed alert origins"):
        compute_alert_metrics(conflict, [1, 2], np.array([True, True]))

    # Keyless fallback record alone: inferred fallback, timing math correct
    fb_keyless = [{"alerted": True, "day": 1, "predicted_onset_day": 2, "message": ""}]
    m_fb = compute_alert_metrics(fb_keyless, [2], np.array([True]))
    assert m_fb["lead_times"] == [0.0]
    assert m_fb["alert_origin"] == "fallback_scan"

    # Legacy keyless matched record with no onset fields: excluded from timing,
    # counted, never scored as issuance
    legacy = [{"alerted": True, "day": 0, "message": ""}]
    m = compute_alert_metrics(legacy, [2], np.array([True]))
    assert m["lead_times"] == []
    assert m["n_unknown_origin"] == 1
    assert m["mean_lead_days"] is None
    assert "unknown origin" in m["warning"]


def test_8_read_only_split_structure_validation():
    """load_split(read_only=True) rejects corrupt splits even with a matching seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data = tmp_path / "data.npz"
        np.savez(
            data,
            ndvi=np.zeros((4, 35), dtype=np.float32),
            meta=np.asarray([4, 21, 14, 42], dtype=np.int64),
        )

        def write(name, obj):
            p = tmp_path / name
            p.write_text(json.dumps(obj), encoding="utf-8")
            return p

        base = {"seed": 42, "train": [0, 1], "val": [2], "test": [3], "n_fields": 4}

        with pytest.raises(ValueError, match="overlap"):
            load_split(seed=42, split_path=write("ov.json", dict(base, val=[1, 2])), dataset_path=data, read_only=True)
        with pytest.raises(ValueError, match="integer"):
            load_split(seed=42, split_path=write("b.json", dict(base, train=[True, 1])), dataset_path=data, read_only=True)
        with pytest.raises(ValueError, match="negative"):
            load_split(seed=42, split_path=write("n.json", dict(base, train=[-1, 1])), dataset_path=data, read_only=True)
        with pytest.raises(ValueError, match="duplicate"):
            load_split(seed=42, split_path=write("d.json", dict(base, train=[1, 1])), dataset_path=data, read_only=True)
        with pytest.raises(ValueError, match="out of range"):
            load_split(seed=42, split_path=write("r.json", dict(base, test=[7])), dataset_path=data, read_only=True)
        with pytest.raises(ValueError, match="declares"):
            load_split(seed=42, split_path=write("m.json", dict(base, n_fields=5)), dataset_path=data, read_only=True)
        for bad_decl in (True, 4.0, "4"):
            with pytest.raises(ValueError, match="must be an integer"):
                load_split(seed=42, split_path=write("dt.json", dict(base, n_fields=bad_decl)), dataset_path=data, read_only=True)
        with pytest.raises(ValueError, match="missing keys"):
            load_split(seed=42, split_path=write("k.json", {"seed": 42, "train": [0]}), dataset_path=data, read_only=True)

        # A valid split in the existing format still loads
        ok = load_split(seed=42, split_path=write("ok.json", base), dataset_path=data, read_only=True)
        assert ok["test"] == [3]
        assert ok["loaded_path"].endswith("ok.json")


if __name__ == "__main__":
    test_1_fallback_timing_and_origin_distinction()
    print("test_1_fallback_timing_and_origin_distinction: PASSED")
    test_2_production_metric_denominators_and_formatting()
    print("test_2_production_metric_denominators_and_formatting: PASSED")
    test_3_evaluator_real_parser()
    print("test_3_evaluator_real_parser: PASSED")
    test_4_real_trajectory_dataset_slicing()
    print("test_4_real_trajectory_dataset_slicing: PASSED")
    test_5_real_load_split_and_provenance_hash()
    print("test_5_real_load_split_and_provenance_hash: PASSED")
    test_6_strict_read_only_loader_safety()
    print("test_6_strict_read_only_loader_safety: PASSED")
    test_7_origin_stamping_and_mixed_rejection()
    print("test_7_origin_stamping_and_mixed_rejection: PASSED")
    test_8_read_only_split_structure_validation()
    print("test_8_read_only_split_structure_validation: PASSED")
    print("\nALL 8 PRODUCTION CONSOLIDATION TESTS PASSED SUCCESSFULLY!")
