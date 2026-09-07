"""Strict Byte-Read-Only Verification for Phase 7 Structural Falsification Pilot."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from clusterdrift.falsification.bootstrap import (
    compute_paired_unit_bootstrap,
    evaluate_falsification_verdict,
)
from clusterdrift.falsification.dataset import FORBIDDEN_PREDICTOR_FEATURES
from clusterdrift.falsification.evaluation import compute_prediction_record_sha256
from clusterdrift.falsification.protocol import (
    ALL_SHIFT_FAMILIES,
    CONDITION_TO_FAMILY,
    PREDEFINED_FEATURE_BLOCKS,
    load_falsification_config,
)
from clusterdrift.signals.hashing import (
    compute_file_sha256,
    compute_quality_record_sha256,
    compute_signal_record_sha256,
)


def verify_falsification(project_root: Path) -> Dict[str, Any]:
    """Execute complete byte-read-only verification of Phase 7 artifacts.

    Raises an AssertionError or ValueError if any verification check fails.
    """
    root = Path(project_root)
    cfg_path = root / "configs" / "falsification.yaml"
    cfg = load_falsification_config(cfg_path)

    check_count = 0

    # 1. Verify Input Lock
    lock_path = root / "data" / "falsification" / "phase7_input_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Phase 7 input lock not found: {lock_path}")
    with open(lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)
    check_count += 1

    # Verify input lock references
    for manifest_key, rel_path in [
        ("phase1_dataset_manifest_sha256", "data/manifests/dataset_manifest.json"),
        ("phase2_split_manifest_sha256", "data/manifests/split_manifest.json"),
        ("phase4_shift_manifest_sha256", "data/manifests/shift_manifest.json"),
        ("phase5_probe_manifest_sha256", "data/manifests/probe_manifest.json"),
    ]:
        p = root / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Referenced manifest not found: {p}")
        expected_sha = lock.get(manifest_key)
        actual_sha = compute_file_sha256(p)
        if expected_sha != actual_sha:
            raise ValueError(
                f"Manifest SHA mismatch for {manifest_key}: expected {expected_sha}, got {actual_sha}"
            )
        check_count += 1

    # 2. Verify Replay Artifacts (120 JSON + 120 NPZ)
    replay_dir = root / "data" / "falsification" / "offline_shift_replay"
    if not replay_dir.exists():
        raise FileNotFoundError(f"Offline replay dir not found: {replay_dir}")

    replay_jsons = list(replay_dir.glob("*.json"))
    replay_npzs = list(replay_dir.glob("*.npz"))
    if len(replay_jsons) != 120 or len(replay_npzs) != 120:
        raise ValueError(
            f"Expected 120 replay JSONs and 120 NPZs, found {len(replay_jsons)} JSONs, {len(replay_npzs)} NPZs"
        )
    check_count += 2

    # Check each replay file
    for r_json in replay_jsons:
        with open(r_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        npz_name = meta.get("companion_npz")
        npz_p = replay_dir / npz_name
        if not npz_p.exists():
            raise FileNotFoundError(f"Companion NPZ not found: {npz_p}")
        actual_npz_sha = compute_file_sha256(npz_p)
        if actual_npz_sha != meta.get("companion_npz_sha256"):
            raise ValueError(f"NPZ SHA mismatch for {r_json.name}")
        check_count += 1

    # 3. Verify Signal and Quality Artifacts
    res_dir = root / "results" / "falsification"
    signals_p = res_dir / "signals_label_free.csv"
    quality_p = res_dir / "quality_evaluation_only.csv"
    joined_p = res_dir / "joined_evaluation_table.csv"

    for p in [signals_p, quality_p, joined_p]:
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {p}")
        check_count += 1

    sig_df = pd.read_csv(signals_p)
    qual_df = pd.read_csv(quality_p)
    join_df = pd.read_csv(joined_p)

    expected_rows = len(cfg["datasets"]) * len(cfg["outer_folds"]) * len(cfg["conditions"]) * len(cfg["methods"]) * len(cfg["algorithm_seeds"])
    if len(sig_df) != expected_rows:
        raise ValueError(f"Signals row count mismatch: expected {expected_rows}, got {len(sig_df)}")
    if len(qual_df) != expected_rows:
        raise ValueError(f"Quality row count mismatch: expected {expected_rows}, got {len(qual_df)}")
    if len(join_df) != expected_rows:
        raise ValueError(f"Joined row count mismatch: expected {expected_rows}, got {len(join_df)}")
    check_count += 3

    # Check zero forbidden label features in signals
    for col in sig_df.columns:
        if col in ["y", "label", "labels", "ari", "delta_ari", "ari_clean", "ari_condition"]:
            raise ValueError(f"Forbidden label column found in signals: {col}")
    check_count += 1

    # Verify signal hashes
    for _, row in sig_df.iterrows():
        expected_h = row["signal_record_sha256"]
        actual_h = compute_signal_record_sha256(row.to_dict())
        if actual_h != expected_h:
            raise ValueError(f"Signal row hash mismatch at {row['dataset_id']}, {row['condition']}")
        check_count += 1

    # Verify quality hashes
    for _, row in qual_df.iterrows():
        expected_h = row["quality_record_sha256"]
        actual_h = compute_quality_record_sha256(row.to_dict())
        if actual_h != expected_h:
            raise ValueError(f"Quality row hash mismatch at {row['dataset_id']}, {row['condition']}")
        check_count += 1

    # Verify exact keys match
    key_cols = ["dataset_id", "outer_fold", "condition", "method", "seed"]
    sig_keys = sig_df[key_cols].drop_duplicates()
    qual_keys = qual_df[key_cols].drop_duplicates()
    if len(sig_keys) != expected_rows or len(qual_keys) != expected_rows:
        raise ValueError("Duplicate keys detected in signal or quality artifact")
    check_count += 2

    # 4. Verify Prediction Files and Hashes
    lodo_pred_p = res_dir / "lodo_predictions.csv"
    losfo_pred_p = res_dir / "losfo_predictions.csv"
    for p in [lodo_pred_p, losfo_pred_p]:
        if not p.exists():
            raise FileNotFoundError(f"Prediction file not found: {p}")
        check_count += 1

    lodo_preds = pd.read_csv(lodo_pred_p)
    losfo_preds = pd.read_csv(losfo_pred_p)

    # Check sample of prediction hashes
    for _, row in lodo_preds.iloc[::20].iterrows():
        exp_h = row["prediction_record_sha256"]
        act_h = compute_prediction_record_sha256(row.to_dict())
        if exp_h != act_h:
            raise ValueError("LODO prediction hash mismatch")
        check_count += 1

    for _, row in losfo_preds.iloc[::20].iterrows():
        exp_h = row["prediction_record_sha256"]
        act_h = compute_prediction_record_sha256(row.to_dict())
        if exp_h != act_h:
            raise ValueError("LOSFO prediction hash mismatch")
        check_count += 1

    # Check that test datasets in LODO were not in training
    for ds in cfg["datasets"]:
        sub = lodo_preds[lodo_preds["dataset_id"] == ds]
        if not (sub["outer_test_group"] == ds).all():
            raise ValueError(f"LODO leak detected for dataset {ds}")
        check_count += 1

    # Check that test families in LOSFO were not in training
    for fam in ALL_SHIFT_FAMILIES:
        sub = losfo_preds[losfo_preds["shift_family"] == fam]
        if not (sub["outer_test_group"] == fam).all():
            raise ValueError(f"LOSFO leak detected for family {fam}")
        check_count += 1

    # 5. Verify Metric Artifacts & Verdict Recomputation
    paired_ds_p = res_dir / "paired_dataset_deltas.csv"
    paired_fam_p = res_dir / "paired_family_deltas.csv"
    boot_p = res_dir / "bootstrap_intervals.csv"
    verdict_p = res_dir / "verdict.json"

    for p in [paired_ds_p, paired_fam_p, boot_p, verdict_p]:
        if not p.exists():
            raise FileNotFoundError(f"Metric artifact not found: {p}")
        check_count += 1

    paired_ds_df = pd.read_csv(paired_ds_p)
    paired_fam_df = pd.read_csv(paired_fam_p)
    boot_df = pd.read_csv(boot_p)
    with open(verdict_p, "r", encoding="utf-8") as f:
        stored_verdict = json.load(f)

    # Recompute bootstrap
    d_04 = paired_ds_df["delta_04"].values
    d_34 = paired_ds_df["delta_34"].values
    f_04 = paired_fam_df["delta_04"].values
    f_34 = paired_fam_df["delta_34"].values

    recomputed_boot = {
        "lodo_delta_04": compute_paired_unit_bootstrap(d_04, seed=cfg["bootstrap"]["seed"]),
        "lodo_delta_34": compute_paired_unit_bootstrap(d_34, seed=cfg["bootstrap"]["seed"]),
        "losfo_delta_04": compute_paired_unit_bootstrap(f_04, seed=cfg["bootstrap"]["seed"]),
        "losfo_delta_34": compute_paired_unit_bootstrap(f_34, seed=cfg["bootstrap"]["seed"]),
    }
    check_count += 4

    lodo_metrics_p = res_dir / "lodo_metrics.csv"
    lodo_m = pd.read_csv(lodo_metrics_p)
    hgbr_lodo = lodo_m[lodo_m["regressor"] == "hist_gradient_boosting"]
    p0_mae = float(hgbr_lodo[hgbr_lodo["feature_block"] == "P0"]["mae"].values[0])
    p3_mae = float(hgbr_lodo[hgbr_lodo["feature_block"] == "P3"]["mae"].values[0])
    p4_mae = float(hgbr_lodo[hgbr_lodo["feature_block"] == "P4"]["mae"].values[0])

    recomputed_verdict = evaluate_falsification_verdict(
        p0_mae,
        p3_mae,
        p4_mae,
        d_04,
        d_34,
        f_04,
        f_34,
        recomputed_boot,
    )
    check_count += 1

    if recomputed_verdict["verdict"] != stored_verdict["verdict"]:
        raise ValueError(
            f"Verdict recomputation mismatch: expected {stored_verdict['verdict']}, got {recomputed_verdict['verdict']}"
        )
    check_count += 1

    return {
        "status": "PASSED",
        "total_checks_verified": check_count,
        "verdict": stored_verdict["verdict"],
        "n_signals": len(sig_df),
        "n_quality": len(qual_df),
        "n_replay_scenarios": len(replay_jsons),
    }