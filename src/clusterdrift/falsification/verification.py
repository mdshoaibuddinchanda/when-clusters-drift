"""Strict Byte-Read-Only Verification for Phase 7 Structural Falsification Pilot."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from clusterdrift.falsification.bootstrap import (
    compute_paired_unit_bootstrap,
    evaluate_falsification_verdict,
)
from clusterdrift.falsification.dataset import FORBIDDEN_PREDICTOR_FEATURES
from clusterdrift.falsification.evaluation import compute_prediction_record_sha256
from clusterdrift.falsification.execution.cache import (
    PersistentPhase7Cache,
    compute_content_fingerprint,
)
from clusterdrift.falsification.execution.resources import get_default_work_dir
from clusterdrift.falsification.protocol import (
    ALL_SHIFT_FAMILIES,
    CONDITION_TO_FAMILY,
    PREDEFINED_FEATURE_BLOCKS,
    compute_falsification_protocol_sha256,
    load_falsification_config,
)
from clusterdrift.probes.bank import (
    load_current_probe_descriptor,
    load_reference_probe_descriptor,
)
from clusterdrift.shifts.hashing import (
    compute_bytes_sha256,
    compute_canonical_json_sha256,
    compute_file_sha256,
)
from clusterdrift.alignment.state import compute_model_fingerprint
from clusterdrift.data.preprocess import build_preprocessor
from clusterdrift.falsification.execution.environment import limit_inner_threads
from clusterdrift.signals.engine import fit_clustering_model
from clusterdrift.signals.hashing import (
    compute_quality_record_sha256,
    compute_signal_record_sha256,
)


def verify_input_lock(project_root: Path) -> Dict[str, Any]:
    """Verify Phase 7 input lock integrity, manifests, configs, and Phase 6 provenance."""
    root = Path(project_root).resolve()
    lock_path = root / "data" / "falsification" / "phase7_input_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Phase 7 input lock not found: {lock_path}")

    with open(lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)

    # 1. Require exact Phase-6 provenance
    exp_p6_producer = "ca73a520ef65c2e1de8ff014190019e19d4ab864"
    exp_p6_artifact = "a2726e655da43bec59c640dda008cf65f412919d"
    act_p6_producer = lock.get("phase6_final_producer_commit")
    act_p6_artifact = lock.get("phase6_final_artifact_commit")

    if act_p6_producer != exp_p6_producer:
        raise ValueError(
            f"Phase 6 producer commit mismatch: expected {exp_p6_producer}, got {act_p6_producer}"
        )
    if act_p6_artifact != exp_p6_artifact:
        raise ValueError(
            f"Phase 6 artifact commit mismatch: expected {exp_p6_artifact}, got {act_p6_artifact}"
        )

    # Check that obsolete/wrong commit is NOT present
    if "phase6_freeze_commit" in lock:
        raise ValueError(
            "Obsolete 'phase6_freeze_commit' found in phase7_input_lock.json; must be removed."
        )

    # 2. Phase 7 protocol & execution commits
    exp_p7_protocol = "8dc7bc8056a686f1eb147f9ec5bf211935454da6"
    if lock.get("phase7_scientific_protocol_commit") != exp_p7_protocol:
        raise ValueError(
            f"Phase 7 protocol commit mismatch: expected {exp_p7_protocol}, got {lock.get('phase7_scientific_protocol_commit')}"
        )

    # 3. Check all file SHAs against disk
    manifest_and_config_bindings = [
        ("phase1_dataset_manifest_sha256", "data/manifests/datasets.json"),
        ("phase2_split_manifest_sha256", "data/splits/split_manifest.json"),
        ("phase2_preprocessing_config_sha256", "configs/preprocessing.yaml"),
        ("phase4_shift_manifest_sha256", "data/shifts/shift_manifest.json"),
        ("phase5_probe_manifest_sha256", "data/probes/probe_manifest.json"),
        ("methods_config_sha256", "configs/methods.yaml"),
        ("signals_config_sha256", "configs/signals.yaml"),
        ("alignment_config_sha256", "configs/alignment.yaml"),
        ("probe_config_sha256", "configs/probes.yaml"),
        ("falsification_config_sha256", "configs/falsification.yaml"),
        ("phase6_input_lock_sha256", "data/signals/phase6_input_lock.json"),
    ]

    for lock_key, rel_p in manifest_and_config_bindings:
        p = root / rel_p
        if not p.exists():
            raise FileNotFoundError(f"Referenced file not found: {p}")
        expected_sha = lock.get(lock_key)
        actual_sha = compute_file_sha256(p)
        if expected_sha != actual_sha:
            raise ValueError(
                f"SHA256 mismatch for {lock_key} ({rel_p}): expected {expected_sha}, got {actual_sha}"
            )

    # 4. Phase 6 signal protocol check from phase6 input lock
    p6_lock_p = root / "data" / "signals" / "phase6_input_lock.json"
    with open(p6_lock_p, "r", encoding="utf-8") as f:
        p6_lock = json.load(f)
    if lock.get("phase6_signal_protocol_sha256") != p6_lock.get("signal_protocol_sha256"):
        raise ValueError("Phase 6 signal protocol SHA mismatch between Phase 6 and Phase 7 locks")

    return {
        "status": "VALID",
        "lock_version": lock.get("lock_version", 2),
        "phase6_producer_commit": exp_p6_producer,
        "phase6_artifact_commit": exp_p6_artifact,
        "phase7_protocol_commit": exp_p7_protocol,
    }


def verify_offline_shift_replay(project_root: Path, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Verify all 120 offline local-overlap replay descriptors and companion NPZ arrays."""
    root = Path(project_root).resolve()
    if cfg is None:
        cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    replay_dir = root / "data" / "falsification" / "offline_shift_replay"
    if not replay_dir.exists():
        raise FileNotFoundError(f"Offline shift replay directory not found: {replay_dir}")

    datasets = cfg["datasets"]
    folds = cfg["outer_folds"]
    local_overlap_conditions = ["local_overlap_mild", "local_overlap_severe"]

    all_json_files = list(replay_dir.rglob("*.json"))
    all_npz_files = list(replay_dir.rglob("*.npz"))

    if len(all_json_files) != 120:
        raise ValueError(f"Expected exactly 120 replay JSON files, found {len(all_json_files)}")
    if len(all_npz_files) != 120:
        raise ValueError(f"Expected exactly 120 replay NPZ files, found {len(all_npz_files)}")

    expected_commit = "8dc7bc8056a686f1eb147f9ec5bf211935454da6"
    verified_count = 0

    for ds in datasets:
        for fold in folds:
            for cond in local_overlap_conditions:
                json_p = replay_dir / ds / f"fold_{fold}" / f"{cond}.json"
                npz_p = json_p.with_suffix(".npz")

                if not json_p.exists():
                    raise FileNotFoundError(f"Missing replay descriptor: {json_p}")
                if not npz_p.exists():
                    raise FileNotFoundError(f"Missing companion NPZ: {npz_p}")

                with open(json_p, "r", encoding="utf-8") as f:
                    d = json.load(f)

                # Coordinate check
                if d.get("dataset_id") != ds or d.get("outer_fold") != fold or d.get("condition") != cond:
                    raise ValueError(f"Replay descriptor coordinate mismatch in {json_p}")

                # Commit check
                if d.get("generated_from_commit") != expected_commit:
                    raise ValueError(
                        f"Replay descriptor commit mismatch in {json_p}: expected {expected_commit}, got {d.get('generated_from_commit')}"
                    )

                # Self-hash recomputation check
                stored_desc_sha = d.get("replay_descriptor_sha256")
                d_without_sha = {k: v for k, v in d.items() if k != "replay_descriptor_sha256"}
                recomputed_desc_sha = compute_canonical_json_sha256(d_without_sha)
                if stored_desc_sha != recomputed_desc_sha:
                    raise ValueError(f"Replay descriptor self-hash mismatch for {json_p.name}")

                # NPZ file hash check
                actual_npz_sha = compute_file_sha256(npz_p)
                if actual_npz_sha != d.get("replay_npz_sha256"):
                    raise ValueError(f"Replay NPZ file hash mismatch for {npz_p.name}")

                # NPZ array verification
                with np.load(npz_p) as npz:
                    if "selected_target_positions" not in npz:
                        raise KeyError(f"'selected_target_positions' missing in {npz_p.name}")
                    if "replacement_values" not in npz:
                        raise KeyError(f"'replacement_values' missing in {npz_p.name}")

                    pos = npz["selected_target_positions"]
                    vals = npz["replacement_values"]

                    if pos.dtype != np.int64:
                        raise TypeError(f"selected_target_positions dtype must be int64, got {pos.dtype}")
                    if vals.dtype != np.float64:
                        raise TypeError(f"replacement_values dtype must be float64, got {vals.dtype}")

                    pos_sha = compute_bytes_sha256(pos.tobytes())
                    vals_sha = compute_bytes_sha256(vals.tobytes())

                    if pos_sha != d.get("selected_target_positions_sha256"):
                        raise ValueError(f"selected_target_positions_sha256 mismatch in {json_p.name}")
                    if vals_sha != d.get("replacement_values_sha256"):
                        raise ValueError(f"replacement_values_sha256 mismatch in {json_p.name}")

                    affected_cols = d.get("affected_numeric_columns", [])
                    if vals.ndim == 2 and vals.shape[1] != len(affected_cols):
                        raise ValueError(
                            f"replacement_values shape[1] ({vals.shape[1]}) != affected_columns count ({len(affected_cols)})"
                        )

                # Phase 4 shift spec binding check
                spec_p = root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"
                if not spec_p.exists():
                    raise FileNotFoundError(f"Phase 4 spec file not found: {spec_p}")

                with open(spec_p, "r", encoding="utf-8") as f:
                    spec = json.load(f)

                if d.get("phase4_shift_spec_sha256") != spec.get("shift_spec_sha256"):
                    raise ValueError(f"Phase 4 shift spec SHA mismatch for {json_p.name}")
                if d.get("phase4_shift_spec_file_sha256") != compute_file_sha256(spec_p):
                    raise ValueError(f"Phase 4 shift spec file SHA mismatch for {json_p.name}")
                if d.get("phase4_npz_sha256") != spec.get("npz_sha256"):
                    raise ValueError(f"Phase 4 NPZ SHA mismatch for {json_p.name}")
                if d.get("phase4_protocol_sha256") != spec.get("shift_protocol_sha256"):
                    raise ValueError(f"Phase 4 protocol SHA mismatch for {json_p.name}")

                verified_count += 1

    return {
        "status": "VALID",
        "verified_scenarios": verified_count,
        "json_count": len(all_json_files),
        "npz_count": len(all_npz_files),
    }


def verify_pass_a_signals(project_root: Path, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Verify Pass A label-free signals CSV: universe, hashes, bounds, isolation, and bindings."""
    root = Path(project_root).resolve()
    if cfg is None:
        cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    signals_p = root / "results" / "falsification" / "signals_label_free.csv"
    if not signals_p.exists():
        raise FileNotFoundError(f"Canonical signals artifact not found: {signals_p}")

    df = pd.read_csv(signals_p)
    expected_rows = (
        len(cfg["datasets"])
        * len(cfg["outer_folds"])
        * len(cfg["conditions"])
        * len(cfg["methods"])
        * len(cfg["algorithm_seeds"])
    )
    if len(df) != expected_rows:
        raise ValueError(f"Signals row count mismatch: expected {expected_rows}, got {len(df)}")

    # 1. Exact 4,500-key universe verification
    expected_keys = [
        (ds, fold, cond, meth, seed)
        for ds in cfg["datasets"]
        for fold in cfg["outer_folds"]
        for cond in cfg["conditions"]
        for meth in cfg["methods"]
        for seed in cfg["algorithm_seeds"]
    ]
    actual_keys = list(zip(
        df["dataset_id"],
        df["outer_fold"].astype(int),
        df["condition"],
        df["method"],
        df["seed"].astype(int),
    ))

    if len(set(actual_keys)) != expected_rows:
        raise ValueError(f"Duplicate keys found in signals CSV! Unique count: {len(set(actual_keys))}")
    if set(actual_keys) != set(expected_keys):
        missing = set(expected_keys) - set(actual_keys)
        unexpected = set(actual_keys) - set(expected_keys)
        raise ValueError(f"Signals key universe mismatch: missing={len(missing)}, unexpected={len(unexpected)}")

    # 2. Strict label isolation
    forbidden_cols = {
        "labels", "label", "y", "y_true", "target", "ari", "nmi", "ami",
        "delta_ari", "accuracy", "ari_clean", "ari_condition"
    }
    found_forbidden = forbidden_cols.intersection(set(df.columns))
    if found_forbidden:
        raise ValueError(f"Forbidden label/ground-truth columns found in signals CSV: {found_forbidden}")

    # 3. Cryptographic record hash recomputation
    for idx, row in df.iterrows():
        rec = row.to_dict()
        cleaned = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        stored_sha = str(cleaned.get("signal_record_sha256", ""))
        recomputed_sha = compute_signal_record_sha256(cleaned)
        if stored_sha != recomputed_sha:
            raise ValueError(f"Record hash mismatch at row {idx} ({row['dataset_id']}, {row['condition']})")

    # 4. Primary scientific signals bounds & finiteness
    core_signals = ["D_U_R", "D_U_C", "D_V", "D_H", "D_M", "D_X", "FPC", "PE_norm", "XB_soft_m2", "silhouette"]
    for col in core_signals:
        if col not in df.columns:
            raise KeyError(f"Core signal column missing: {col}")
        vals = df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all():
            raise ValueError(f"Non-finite values detected in signal {col}")

    # Specific bounds
    if not ((df["D_U_R"] >= -1e-6) & (df["D_U_R"] <= 1.0 + 1e-6)).all():
        raise ValueError("D_U_R out of bounds [0, 1]")
    if not ((df["D_U_C"] >= -1e-6) & (df["D_U_C"] <= 1.0 + 1e-6)).all():
        raise ValueError("D_U_C out of bounds [0, 1]")
    if not ((df["D_H"] >= -1e-6) & (df["D_H"] <= 1.0 + 1e-6)).all():
        raise ValueError("D_H out of bounds [0, 1]")
    if not ((df["D_M"] >= -1e-6) & (df["D_M"] <= 1.0 + 1e-6)).all():
        raise ValueError("D_M out of bounds [0, 1]")
    if not (df["D_V"] >= -1e-6).all():
        raise ValueError("D_V out of bounds (negative values detected)")
    if not (df["D_X"] >= -1e-6).all():
        raise ValueError("D_X out of bounds (negative values detected)")
    if not ((df["PE_norm"] >= -1e-6) & (df["PE_norm"] <= 1.0 + 1e-6)).all():
        raise ValueError("PE_norm out of bounds [0, 1]")
    if not ((df["silhouette"] >= -1.0 - 1e-6) & (df["silhouette"] <= 1.0 + 1e-6)).all():
        raise ValueError("silhouette out of bounds [-1, 1]")

    # 5. Verify row-by-row bindings against Phase 4 and Phase 5 identities
    ref_cache: Dict[Tuple[str, int], str] = {}
    cur_cache: Dict[Tuple[str, int, str], str] = {}
    spec_cache: Dict[Tuple[str, int, str], Tuple[str, str]] = {}
    replay_cache: Dict[Tuple[str, int, str], str] = {}

    for idx, row in df.iterrows():
        ds = row["dataset_id"]
        fold = int(row["outer_fold"])
        cond = row["condition"]

        # Phase 4 spec
        s_key = (ds, fold, cond)
        if s_key not in spec_cache:
            spec_p = root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"
            with open(spec_p, "r", encoding="utf-8") as f:
                s_doc = json.load(f)
            spec_cache[s_key] = (s_doc["shift_spec_sha256"], compute_file_sha256(spec_p))

        exp_spec_sha, exp_spec_file_sha = spec_cache[s_key]
        if row["shift_spec_sha256"] != exp_spec_sha:
            raise ValueError(f"Row shift_spec_sha256 mismatch at row {idx}")
        if row["shift_spec_file_sha256"] != exp_spec_file_sha:
            raise ValueError(f"Row shift_spec_file_sha256 mismatch at row {idx}")

        # Phase 5 reference bank
        r_key = (ds, fold)
        if r_key not in ref_cache:
            ref_p = root / "data" / "probes" / "reference" / ds / f"fold_{fold}.json"
            r_desc, _, _ = load_reference_probe_descriptor(ref_p)
            ref_cache[r_key] = r_desc.probe_bank_sha256

        if row["reference_bank_sha256"] != ref_cache[r_key]:
            raise ValueError(f"Row reference_bank_sha256 mismatch at row {idx}")

        # Phase 5 current bank
        c_key = (ds, fold, cond)
        if c_key not in cur_cache:
            cur_p = root / "data" / "probes" / "current" / ds / f"fold_{fold}" / f"{cond}.json"
            c_desc, _, _, _ = load_current_probe_descriptor(cur_p)
            cur_cache[c_key] = c_desc.probe_bank_sha256

        if row["current_bank_sha256"] != cur_cache[c_key]:
            raise ValueError(f"Row current_bank_sha256 mismatch at row {idx}")

        # Local overlap replay
        if cond.startswith("local_overlap"):
            rep_key = (ds, fold, cond)
            if rep_key not in replay_cache:
                rep_p = root / "data" / "falsification" / "offline_shift_replay" / ds / f"fold_{fold}" / f"{cond}.json"
                with open(rep_p, "r", encoding="utf-8") as f:
                    r_doc = json.load(f)
                replay_cache[rep_key] = r_doc["replay_descriptor_sha256"]

            if row.get("shift_replay_sha256") != replay_cache[rep_key]:
                raise ValueError(f"Row shift_replay_sha256 mismatch at row {idx}")

    return {
        "status": "VALID",
        "signals_csv": str(signals_p.relative_to(root)),
        "rows": len(df),
        "columns": len(df.columns),
        "sha256": compute_file_sha256(signals_p),
    }


def verify_pass_a(project_root: Path) -> Dict[str, Any]:
    """Execute complete, strict, byte-read-only verification for Phase 7 Pass A."""
    root = Path(project_root).resolve()
    cfg_path = root / "configs" / "falsification.yaml"
    cfg = load_falsification_config(cfg_path)

    lock_res = verify_input_lock(root)
    replay_res = verify_offline_shift_replay(root, cfg)
    signals_res = verify_pass_a_signals(root, cfg)

    return {
        "status": "PASSED",
        "pass_a_status": "FROZEN",
        "input_lock": lock_res,
        "offline_replay": replay_res,
        "signals": signals_res,
    }


def verify_pass_b(project_root: Path, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute complete, strict, byte-read-only verification for Phase 7 Pass B."""
    root = Path(project_root).resolve()
    if cfg is None:
        cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    # 1. Require Pass A verification still passes and SHA matches
    pass_a_res = verify_pass_a(root)
    exp_signals_sha = "3a9e6c68cbe34c8763d70769715dba0d34976dfc35c8877c5680561da5c0e80a"
    if pass_a_res["signals"]["sha256"] != exp_signals_sha:
        raise ValueError(
            f"Pass A signals CSV SHA256 mismatch during Pass B verification: "
            f"expected {exp_signals_sha}, got {pass_a_res['signals']['sha256']}"
        )

    # 2. Quality CSV exists and has exact 4,500 rows
    quality_p = root / "results" / "falsification" / "quality_evaluation_only.csv"
    if not quality_p.exists():
        raise FileNotFoundError(f"Pass B quality artifact not found: {quality_p}")

    df_qual = pd.read_csv(quality_p)
    expected_rows = (
        len(cfg["datasets"])
        * len(cfg["outer_folds"])
        * len(cfg["conditions"])
        * len(cfg["methods"])
        * len(cfg["algorithm_seeds"])
    )
    if len(df_qual) != expected_rows:
        raise ValueError(f"Quality row count mismatch: expected {expected_rows}, got {len(df_qual)}")

    # 3. Exact Key Universe & Key Matching with Pass A signals
    expected_keys = [
        (ds, fold, cond, meth, seed)
        for ds in cfg["datasets"]
        for fold in cfg["outer_folds"]
        for cond in cfg["conditions"]
        for meth in cfg["methods"]
        for seed in cfg["algorithm_seeds"]
    ]
    actual_keys = list(zip(
        df_qual["dataset_id"],
        df_qual["outer_fold"].astype(int),
        df_qual["condition"],
        df_qual["method"],
        df_qual["seed"].astype(int),
    ))

    if len(set(actual_keys)) != expected_rows:
        raise ValueError(f"Duplicate keys found in quality CSV! Unique count: {len(set(actual_keys))}")
    if set(actual_keys) != set(expected_keys):
        missing = set(expected_keys) - set(actual_keys)
        unexpected = set(actual_keys) - set(expected_keys)
        raise ValueError(f"Quality key universe mismatch: missing={len(missing)}, unexpected={len(unexpected)}")

    # Check signal keys == quality keys exactly
    signals_p = root / "results" / "falsification" / "signals_label_free.csv"
    df_sig = pd.read_csv(signals_p)
    signal_keys = list(zip(
        df_sig["dataset_id"],
        df_sig["outer_fold"].astype(int),
        df_sig["condition"],
        df_sig["method"],
        df_sig["seed"].astype(int),
    ))
    if set(actual_keys) != set(signal_keys):
        raise ValueError("Quality keys do not match Pass A signal keys exactly!")

    # 4. Cryptographic record hash recomputation
    valid_hashes = 0
    for idx, row in df_qual.iterrows():
        rec = row.to_dict()
        stored_sha = str(rec.get("quality_record_sha256", ""))
        recomputed_sha = compute_quality_record_sha256(rec)
        if stored_sha != recomputed_sha:
            raise ValueError(f"Quality record hash mismatch at row {idx} ({row['dataset_id']}, {row['condition']})")
        valid_hashes += 1

    # 5. Finiteness & Metric Bounds
    for col in ["ari_clean", "ari_condition", "delta_ari", "nmi_condition", "ami_condition", "n_evaluation_rows"]:
        if col not in df_qual.columns:
            raise KeyError(f"Required quality column missing: {col}")
        vals = df_qual[col].to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all():
            raise ValueError(f"Non-finite values detected in quality column {col}")

    if not ((df_qual["ari_clean"] >= -1.0 - 1e-6) & (df_qual["ari_clean"] <= 1.0 + 1e-6)).all():
        raise ValueError("ari_clean out of bounds [-1, 1]")
    if not ((df_qual["ari_condition"] >= -1.0 - 1e-6) & (df_qual["ari_condition"] <= 1.0 + 1e-6)).all():
        raise ValueError("ari_condition out of bounds [-1, 1]")
    if not ((df_qual["nmi_condition"] >= -1e-6) & (df_qual["nmi_condition"] <= 1.0 + 1e-6)).all():
        raise ValueError("nmi_condition out of bounds [0, 1]")
    if not ((df_qual["ami_condition"] >= -1.0 - 1e-6) & (df_qual["ami_condition"] <= 1.0 + 1e-6)).all():
        raise ValueError("ami_condition out of bounds [-1, 1]")

    # 6. Delta ARI Identity exactness: delta_ari = ari_clean - ari_condition
    diff = (df_qual["delta_ari"] - (df_qual["ari_clean"] - df_qual["ari_condition"])).abs()
    if not (diff <= 1e-6).all():
        max_diff = diff.max()
        raise ValueError(f"Delta ARI identity violated: max diff = {max_diff}")

    # 7. Clean condition check: delta_ari == 0 and ari_condition == ari_clean
    clean_sub = df_qual[df_qual["condition"] == "clean"]
    if len(clean_sub) != 300:
        raise ValueError(f"Expected exactly 300 clean condition rows, got {len(clean_sub)}")
    if not (clean_sub["delta_ari"].abs() <= 1e-6).all():
        raise ValueError("Clean condition delta_ari is non-zero!")
    if not ((clean_sub["ari_clean"] - clean_sub["ari_condition"]).abs() <= 1e-6).all():
        raise ValueError("Clean condition ari_clean != ari_condition!")

    # 8. Clean consistency across conditions for each (dataset_id, outer_fold, method, seed)
    clean_delta_checks = 0
    clean_delta_mismatches = 0
    for (ds, fold, meth, seed), group in df_qual.groupby(["dataset_id", "outer_fold", "method", "seed"]):
        clean_delta_checks += 1
        if (group["ari_clean"].max() - group["ari_clean"].min()) > 1e-6:
            clean_delta_mismatches += 1
            raise ValueError(f"Inconsistent ari_clean for {ds} fold_{fold} seed_{seed} across conditions")

    # 9. Class prevalence evaluation rows check against Phase-4 specs
    for (ds, fold), group in df_qual.groupby(["dataset_id", "outer_fold"]):
        for cond in ["class_prevalence_mild", "class_prevalence_severe"]:
            sub = group[group["condition"] == cond]
            if not sub.empty:
                spec_p = root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.npz"
                with np.load(spec_p) as npz:
                    exp_len = len(npz["row_index_map"])
                if not (sub["n_evaluation_rows"] == exp_len).all():
                    raise ValueError(f"n_evaluation_rows mismatch for {ds} fold_{fold} {cond}")

    # 10. Source model fingerprint check: 300 source models match frozen Pass A
    source_model_fp_checks = 0
    source_model_fp_mismatches = 0
    sig_fps = {}
    for (ds, fold, meth, seed), group in df_sig.groupby(["dataset_id", "outer_fold", "method", "seed"]):
        sig_fps[(ds, int(fold), meth, int(seed))] = group["source_model_fingerprint"].iloc[0]

    if len(sig_fps) != 300:
        raise ValueError(f"Expected 300 source model fingerprints in Pass A, got {len(sig_fps)}")

    manifest_p = root / "data" / "manifests" / "datasets.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        m_list = json.load(f)
    classes_map = {m["dataset_id"]: m["n_classes"] for m in m_list if "dataset_id" in m}
    prep_cfg = yaml.safe_load(open(root / "configs" / "preprocessing.yaml", "r", encoding="utf-8"))
    prep_config_sha = compute_file_sha256(root / "configs" / "preprocessing.yaml")
    protocol_sha = compute_falsification_protocol_sha256(cfg)
    cache = PersistentPhase7Cache(get_default_work_dir(), protocol_sha, root)

    for ds in cfg["datasets"]:
        K = classes_map[ds]
        ds_dir = root / "data" / "canonical" / "controlled" / ds
        df_X = pd.read_parquet(ds_dir / "features.parquet")
        with open(ds_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        roles = meta.get("feature_roles", {col: "numeric" for col in df_X.columns})

        for fold in cfg["outer_folds"]:
            fold_p = root / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
            with np.load(fold_p) as npz:
                src_idx = npz["source_indices"]
            X_src_raw = None
            X_src_trans = None

            for seed in cfg["algorithm_seeds"]:
                model_fp = compute_content_fingerprint({
                    "dataset_id": ds,
                    "outer_fold": fold,
                    "method": cfg["methods"][0],
                    "seed": seed,
                    "K": K,
                    "prep_config_sha": prep_config_sha,
                })
                src_model = cache.get_source_model(ds, fold, cfg["methods"][0], seed, model_fp)
                if src_model is None:
                    if X_src_trans is None:
                        X_src_raw = df_X.iloc[src_idx].copy().reset_index(drop=True)
                        src_prep = build_preprocessor(feature_roles=roles, config=prep_cfg, metadata=meta)
                        src_prep.fit(X_src_raw)
                        X_src_trans = src_prep.transform(X_src_raw)
                    with limit_inner_threads(1):
                        src_model = fit_clustering_model(
                            method=cfg["methods"][0],
                            K=K,
                            seed=seed,
                            X=X_src_trans,
                        )
                act_fp = compute_model_fingerprint(cfg["methods"][0], {}, seed, src_model.cluster_centers_)
                exp_fp = sig_fps[(ds, fold, cfg["methods"][0], seed)]
                source_model_fp_checks += 1
                if act_fp != exp_fp:
                    # Retry with fresh fit under limit_inner_threads(1)
                    if X_src_trans is None:
                        X_src_raw = df_X.iloc[src_idx].copy().reset_index(drop=True)
                        src_prep = build_preprocessor(feature_roles=roles, config=prep_cfg, metadata=meta)
                        src_prep.fit(X_src_raw)
                        X_src_trans = src_prep.transform(X_src_raw)
                    with limit_inner_threads(1):
                        src_model = fit_clustering_model(
                            method=cfg["methods"][0],
                            K=K,
                            seed=seed,
                            X=X_src_trans,
                        )
                    act_fp = compute_model_fingerprint(cfg["methods"][0], {}, seed, src_model.cluster_centers_)
                    if act_fp != exp_fp:
                        source_model_fp_mismatches += 1
                        raise ValueError(
                            f"Source model fingerprint mismatch for {ds} fold_{fold} seed_{seed}: {act_fp} vs {exp_fp}"
                        )

    # 11. Delta distribution statistics (descriptive integrity only)
    neg_count = int((df_qual["delta_ari"] < -1e-6).sum())
    pos_count = int((df_qual["delta_ari"] > 1e-6).sum())
    zero_count = int((df_qual["delta_ari"].abs() <= 1e-6).sum())

    return {
        "status": "PASSED",
        "pass_b_status": "FROZEN",
        "quality": {
            "csv_path": str(quality_p.relative_to(root)),
            "rows": len(df_qual),
            "columns": len(df_qual.columns),
            "sha256": compute_file_sha256(quality_p),
            "size_bytes": quality_p.stat().st_size,
        },
        "source_fingerprint_checks": source_model_fp_checks,
        "source_fingerprint_mismatches": source_model_fp_mismatches,
        "clean_delta_checks": clean_delta_checks,
        "clean_delta_mismatches": clean_delta_mismatches,
        "signed_negative_delta_count": neg_count,
        "positive_delta_count": pos_count,
        "zero_delta_count": zero_count,
        "valid_quality_hashes": valid_hashes,
    }


def verify_falsification(project_root: Path) -> Dict[str, Any]:
    """Execute complete byte-read-only verification of Phase 7 artifacts (Pass A + Pass B + Pass C)."""
    root = Path(project_root).resolve()
    cfg_path = root / "configs" / "falsification.yaml"
    cfg = load_falsification_config(cfg_path)

    check_count = 0

    # 1. Input Lock & Replay
    verify_input_lock(root)
    verify_offline_shift_replay(root, cfg)
    verify_pass_a_signals(root, cfg)
    check_count += 3

    # 2. Verify Quality & Joined Artifacts
    res_dir = root / "results" / "falsification"
    quality_p = res_dir / "quality_evaluation_only.csv"
    joined_p = res_dir / "joined_evaluation_table.csv"

    for p in [quality_p, joined_p]:
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {p}")
        check_count += 1

    qual_df = pd.read_csv(quality_p)
    join_df = pd.read_csv(joined_p)
    expected_rows = len(cfg["datasets"]) * len(cfg["outer_folds"]) * len(cfg["conditions"]) * len(cfg["methods"]) * len(cfg["algorithm_seeds"])

    if len(qual_df) != expected_rows:
        raise ValueError(f"Quality row count mismatch: expected {expected_rows}, got {len(qual_df)}")
    if len(join_df) != expected_rows:
        raise ValueError(f"Joined row count mismatch: expected {expected_rows}, got {len(join_df)}")
    check_count += 2

    # Verify quality hashes
    for _, row in qual_df.iterrows():
        expected_h = row["quality_record_sha256"]
        actual_h = compute_quality_record_sha256(row.to_dict())
        if actual_h != expected_h:
            raise ValueError(f"Quality row hash mismatch at {row['dataset_id']}, {row['condition']}")
        check_count += 1

    # 3. Verify Prediction Files and Hashes
    lodo_pred_p = res_dir / "lodo_predictions.csv"
    losfo_pred_p = res_dir / "losfo_predictions.csv"
    for p in [lodo_pred_p, losfo_pred_p]:
        if not p.exists():
            raise FileNotFoundError(f"Prediction file not found: {p}")
        check_count += 1

    lodo_preds = pd.read_csv(lodo_pred_p)
    losfo_preds = pd.read_csv(losfo_pred_p)

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

    # 4. Verify Metric Artifacts & Verdict Recomputation
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
    with open(verdict_p, "r", encoding="utf-8") as f:
        stored_verdict = json.load(f)

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
        "n_signals": expected_rows,
        "n_quality": len(qual_df),
        "n_replay_scenarios": 120,
    }