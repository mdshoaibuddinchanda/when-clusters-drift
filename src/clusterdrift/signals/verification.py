"""Strictly Byte-Read-Only Verification Engine for Phase 6.

Validates:
    - Phase-6 input lock and upstream cryptographic hashes
    - Exact 192-key universes for signal and quality records
    - Cryptographic recomputation of all signal and quality record SHA-256 hashes
    - Parameter invariance (D_X across method/seed, sigma across conditions)
    - Mathematical bounds on all primary signals and validity controls
    - Absolute label isolation (no labels in signal artifact)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from clusterdrift.signals.hashing import (
    compute_canonical_dict_sha256,
    compute_file_sha256,
    compute_quality_record_sha256,
    compute_signal_protocol_sha256,
    compute_signal_record_sha256,
)


def verify_phase6_integrity(
    project_root: Path,
    expected_producer_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete byte-read-only verification of Phase 6 artifacts and configuration.

    Parameters
    ----------
    project_root : Path
        Root directory of repository.
    expected_producer_commit : Optional[str]
        Optional commit SHA to verify against lock.

    Returns
    -------
    audit_report : Dict[str, Any]
        Summary of verification results.

    Raises
    ------
    ValueError
        If any integrity, cryptographic, key-universe, or mathematical invariant check fails.
    """
    root = Path(project_root)
    audit: Dict[str, Any] = {"checks": [], "errors": []}

    def _assert(cond: bool, msg: str) -> None:
        if not cond:
            audit["errors"].append(msg)
            raise ValueError(f"[VERIFY FAILURE] {msg}")
        audit["checks"].append(msg)

    # 1. Config and input-lock files exist
    sig_cfg_path = root / "configs" / "signals.yaml"
    lock_path = root / "data" / "signals" / "phase6_input_lock.json"
    signals_csv_path = root / "results" / "signal_validation" / "signals_label_free.csv"
    quality_csv_path = root / "results" / "signal_validation" / "quality_evaluation_only.csv"

    _assert(sig_cfg_path.exists(), "configs/signals.yaml exists")
    _assert(lock_path.exists(), "data/signals/phase6_input_lock.json exists")
    _assert(signals_csv_path.exists(), "results/signal_validation/signals_label_free.csv exists")
    _assert(quality_csv_path.exists(), "results/signal_validation/quality_evaluation_only.csv exists")

    # Load configuration
    with open(sig_cfg_path, "r", encoding="utf-8") as f:
        sig_cfg = yaml.safe_load(f)

    # 2. Input Lock Verification
    with open(lock_path, "r", encoding="utf-8") as f:
        lock_doc = json.load(f)

    _assert("phase6_input_lock_sha256" in lock_doc, "phase6_input_lock_sha256 present in lock")
    stored_lock_sha = lock_doc["phase6_input_lock_sha256"]
    lock_copy = dict(lock_doc)
    del lock_copy["phase6_input_lock_sha256"]
    recomputed_lock_sha = compute_canonical_dict_sha256(lock_copy)
    _assert(stored_lock_sha == recomputed_lock_sha, "phase6_input_lock_sha256 matches recomputed hash")

    # Verify upstream hashes in lock
    _assert(
        lock_doc["phase5_input_lock_sha256"] == compute_file_sha256(root / "data" / "probes" / "phase5_input_lock.json"),
        "phase5_input_lock_sha256 matches upstream file",
    )
    _assert(
        lock_doc["phase4_input_lock_sha256"] == compute_file_sha256(root / "data" / "shifts" / "phase4_input_lock.json"),
        "phase4_input_lock_sha256 matches upstream file",
    )
    _assert(
        lock_doc["signals_config_sha256"] == compute_file_sha256(sig_cfg_path),
        "signals_config_sha256 matches configs/signals.yaml",
    )

    proto_sha = compute_signal_protocol_sha256(sig_cfg)
    _assert(
        lock_doc["signal_protocol_sha256"] == proto_sha,
        "signal_protocol_sha256 matches current config",
    )

    if expected_producer_commit:
        _assert(
            lock_doc["generated_from_commit"] == expected_producer_commit,
            f"generated_from_commit {lock_doc['generated_from_commit']} matches expected {expected_producer_commit}",
        )

    # 3. Key Universe Verification
    expected_datasets = sig_cfg["validation"]["datasets"]
    expected_fold = sig_cfg["validation"]["outer_fold"]
    expected_conditions = sig_cfg["validation"]["conditions"]
    expected_methods = sig_cfg["validation"]["methods"]
    expected_seeds = sig_cfg["validation"]["algorithm_seeds"]

    expected_keys = set()
    for ds in expected_datasets:
        for cond in expected_conditions:
            for meth in expected_methods:
                for seed in expected_seeds:
                    expected_keys.add((ds, expected_fold, cond, meth, seed))

    expected_count = len(expected_keys)  # 6 * 8 * 2 * 2 = 192
    _assert(expected_count == 192, f"Expected key count is 192 (got {expected_count})")

    # Read signal rows
    df_sig = pd.read_csv(signals_csv_path)
    _assert(len(df_sig) == expected_count, f"Signal CSV row count is {len(df_sig)}, expected {expected_count}")

    signal_keys = set(
        zip(df_sig["dataset_id"], df_sig["outer_fold"], df_sig["condition"], df_sig["method"], df_sig["seed"])
    )
    _assert(len(signal_keys) == len(df_sig), "No duplicate signal records")
    _assert(signal_keys == expected_keys, "Signal keys exactly match expected 192-key universe")

    # Read quality rows
    df_qual = pd.read_csv(quality_csv_path)
    _assert(len(df_qual) == expected_count, f"Quality CSV row count is {len(df_qual)}, expected {expected_count}")

    quality_keys = set(
        zip(df_qual["dataset_id"], df_qual["outer_fold"], df_qual["condition"], df_qual["method"], df_qual["seed"])
    )
    _assert(len(quality_keys) == len(df_qual), "No duplicate quality records")
    _assert(quality_keys == expected_keys, "Quality keys exactly match expected 192-key universe")
    _assert(signal_keys == quality_keys, "Signal keys and quality keys are strictly identical")

    # 4. Recompute Signal Record SHA256 Hashes
    for _, row in df_sig.iterrows():
        rec_dict = row.to_dict()
        recomputed_sha = compute_signal_record_sha256(rec_dict)
        _assert(
            recomputed_sha == rec_dict["signal_record_sha256"],
            f"Signal record hash mismatch for {rec_dict['dataset_id']} {rec_dict['condition']} {rec_dict['method']} {rec_dict['seed']}",
        )

    # 5. Recompute Quality Record SHA256 Hashes
    for _, row in df_qual.iterrows():
        q_dict = row.to_dict()
        recomputed_q_sha = compute_quality_record_sha256(q_dict)
        _assert(
            recomputed_q_sha == q_dict["quality_record_sha256"],
            f"Quality record hash mismatch for {q_dict['dataset_id']} {q_dict['condition']} {q_dict['method']} {q_dict['seed']}",
        )

    # 6. D_X Invariance Across Method/Seed (48 unique scenario calculations)
    scenario_dx_map: Dict[Tuple[str, int, str], List[float]] = {}
    for _, row in df_sig.iterrows():
        key = (row["dataset_id"], int(row["outer_fold"]), row["condition"])
        scenario_dx_map.setdefault(key, []).append(float(row["D_X"]))

    _assert(len(scenario_dx_map) == 48, f"Unique D_X scenario count is {len(scenario_dx_map)}, expected 48")
    for sc_key, dx_vals in scenario_dx_map.items():
        _assert(len(dx_vals) == 4, f"Scenario {sc_key} has {len(dx_vals)} rows, expected 4")
        _assert(
            np.allclose(dx_vals, dx_vals[0], atol=1e-7),
            f"D_X varies across method/seed for scenario {sc_key}: {dx_vals}",
        )

    # 7. MMD Bandwidth Invariance Across Conditions for same dataset/fold
    dataset_sigmas: Dict[Tuple[str, int], List[float]] = {}
    for _, row in df_sig.iterrows():
        key = (row["dataset_id"], int(row["outer_fold"]))
        dataset_sigmas.setdefault(key, []).append(float(row["mmd_sigma"]))

    _assert(len(dataset_sigmas) == 6, f"Unique dataset/fold sigma count is {len(dataset_sigmas)}, expected 6")
    for ds_key, sig_vals in dataset_sigmas.items():
        _assert(
            np.allclose(sig_vals, sig_vals[0], atol=1e-7),
            f"MMD sigma varies across conditions for dataset/fold {ds_key}: min={min(sig_vals)}, max={max(sig_vals)}",
        )

    # 8. Mathematical Bounds and Usability
    for _, row in df_sig.iterrows():
        if row["usable"]:
            _assert(0.0 <= row["D_U_R"] <= 1.000001, f"D_U_R out of [0, 1]: {row['D_U_R']}")
            _assert(0.0 <= row["D_U_C"] <= 1.000001, f"D_U_C out of [0, 1]: {row['D_U_C']}")
            _assert(0.0 <= row["D_H"] <= 1.000001, f"D_H out of [0, 1]: {row['D_H']}")
            _assert(0.0 <= row["D_M"] <= 1.000001, f"D_M out of [0, 1]: {row['D_M']}")
            _assert(row["D_V"] >= -1e-10, f"D_V negative: {row['D_V']}")
            _assert(row["D_X"] >= -1e-10, f"D_X negative: {row['D_X']}")

            if np.isfinite(row["PE_norm"]):
                _assert(0.0 <= row["PE_norm"] <= 1.000001, f"PE_norm out of [0, 1]: {row['PE_norm']}")
            if np.isfinite(row["FPC"]):
                K = int(row["K"])
                _assert(row["FPC"] >= (1.0 / K) - 1e-5, f"FPC below 1/K: {row['FPC']} vs 1/{K}")
                _assert(row["FPC"] <= 1.000001, f"FPC above 1.0: {row['FPC']}")

    # 9. Absolute Label Isolation Check in Signal Artifact
    forbidden_signal_cols = ["y", "label", "labels", "ari", "nmi", "ami", "delta_ari"]
    col_names_lower = [c.lower() for c in df_sig.columns]
    for forb in forbidden_signal_cols:
        _assert(forb not in col_names_lower, f"Forbidden label/evaluation column '{forb}' found in signal CSV")

    audit["status"] = "PASSED"
    audit["n_checks"] = len(audit["checks"])
    audit["signal_rows"] = len(df_sig)
    audit["quality_rows"] = len(df_qual)
    audit["unique_dx_scenarios"] = len(scenario_dx_map)
    audit["unique_sigmas"] = len(dataset_sigmas)
    return audit
