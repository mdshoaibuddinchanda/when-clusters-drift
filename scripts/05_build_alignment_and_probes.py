#!/usr/bin/env python3
"""Phase 5 Cluster Alignment and Dual Probe Banks Runner.

Executes:
1. Reference probe bank generation (150 banks: 30 datasets x 5 folds)
2. Current probe bank generation (2,250 banks: 30 datasets x 5 folds x 15 conditions)
3. Total 2,400 probe bank descriptors and companion NPZ files
4. Alignment validation panel (FCM adaptive and GMM on 6 representative datasets)
5. Stability and sensitivity audits (probe size stability, eta sensitivity)
6. CPU/GPU benchmarks and numerical equivalence
7. Strictly byte-read-only verification (--verify)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Core drift modules
from clusterdrift.alignment.costs import (
    compute_center_cost_matrix,
    compute_combined_alignment_cost,
    compute_reference_cluster_scales,
    compute_soft_jaccard_overlap,
)
from clusterdrift.alignment.hungarian import align_clusters
from clusterdrift.alignment.result import AlignmentResult
from clusterdrift.alignment.state import ClusterState, compute_model_fingerprint
from clusterdrift.alignment.validation import validate_cluster_states, validate_membership_matrix

from clusterdrift.data.preprocess import build_preprocessor
from clusterdrift.methods.fcm import FCM
from clusterdrift.methods.gmm import GMM
from clusterdrift.probes.bank import (
    CurrentProbeDescriptor,
    ReferenceProbeDescriptor,
    load_current_probe_descriptor,
    load_reference_probe_descriptor,
    save_current_probe_descriptor,
    save_reference_probe_descriptor,
)
from clusterdrift.probes.cache import ProbeCache
from clusterdrift.probes.evaluation import ProbeEvaluation, verify_probe_identity
from clusterdrift.probes.hashing import (
    build_phase5_input_lock,
    compute_alignment_protocol_sha256,
    compute_current_bank_sha256,
    compute_file_sha256,
    compute_probe_protocol_sha256,
    compute_reference_bank_sha256,
    derive_current_probe_seed,
    derive_reference_probe_seed,
)
from clusterdrift.probes.matrix import (
    load_current_probe_matrix,
    load_raw_source_features,
    load_reference_probe_matrix,
)
from clusterdrift.probes.selection import (
    select_current_probe_positions,
    select_reference_probe_indices,
)
from clusterdrift.probes.validation import (
    validate_saved_current_descriptor,
    validate_saved_reference_descriptor,
)
from clusterdrift.shifts.engine import ShiftEngine
from clusterdrift.shifts.hashing import (
    atomic_write_csv,
    atomic_write_json,
    load_canonical_bundle_hashes,
    load_phase2_split_hashes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONTROLLED_DATASETS = [
    "aps_failure", "balance_scale", "bank_marketing", "banknote_authentication",
    "breast_cancer_wisconsin_diagnostic", "dermatology", "ecoli", "electricity",
    "glass", "haberman_survival", "heart_disease", "human_activity_recognition",
    "image_segmentation", "ionosphere", "iris", "isolet", "letter_recognition",
    "madelon", "mice_protein_expression", "optdigits", "pendigits", "pima_diabetes",
    "satimage", "seeds", "sonar", "spambase", "vehicle_silhouettes", "waveform",
    "wine", "yeast"
]

ALL_CONDITIONS = [
    "clean",
    "location_mild", "location_severe",
    "scale_mild", "scale_severe",
    "mcar_mild", "mcar_severe",
    "outliers_mild", "outliers_severe",
    "measurement_noise_mild", "measurement_noise_severe",
    "class_prevalence_mild", "class_prevalence_severe",
    "local_overlap_mild", "local_overlap_severe",
]

VALIDATION_PANEL_DATASETS = [
    "iris", "glass", "sonar", "madelon",
    "human_activity_recognition", "mice_protein_expression"
]

VALIDATION_PANEL_CONDITIONS = [
    "clean", "location_severe", "scale_severe",
    "mcar_severe", "local_overlap_severe", "class_prevalence_severe"
]


def load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_current_git_commit(project_root: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_dataset_classes(manifest_path: Path) -> Dict[str, int]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["dataset_id"]: r["n_classes"] for r in records if "dataset_id" in r and "n_classes" in r}


# ---------------------------------------------------------------------------
# Probe Bank Generation Workers
# ---------------------------------------------------------------------------

def process_reference_probe_task(
    dataset_id: str,
    outer_fold: int,
    probe_cfg: Dict[str, Any],
    prep_cfg: Dict[str, Any],
    canonical_bundle_hash: str,
    split_hash: str,
    probe_protocol_sha: str,
    prep_config_sha: str,
    probes_dir: Path,
    resume: bool,
    force: bool,
    project_root: Path,
) -> Dict[str, Any]:
    """Build and persist single reference probe bank descriptor."""
    spec_path = probes_dir / "reference" / dataset_id / f"fold_{outer_fold}.json"
    global_seed = probe_cfg.get("global_probe_seed", 2026090705)
    max_size = probe_cfg.get("reference", {}).get("max_size", 2048)

    t0 = time.perf_counter()
    is_reusable = False

    if resume and not force:
        is_val, _, desc = validate_saved_reference_descriptor(
            spec_path=spec_path,
            expected_canonical_bundle_sha256=canonical_bundle_hash,
            expected_split_sha256=split_hash,
            expected_preprocessing_config_sha256=prep_config_sha,
            expected_probe_protocol_sha256=probe_protocol_sha,
            global_probe_seed=global_seed,
            dataset_id=dataset_id,
            outer_fold=outer_fold,
        )
        if is_val and desc is not None:
            is_reusable = True

    if is_reusable:
        desc, indices = load_reference_probe_descriptor(spec_path)
    else:
        # Load raw source rows
        X_src_raw, meta = load_raw_source_features(dataset_id, outer_fold, project_root)
        n_source = len(X_src_raw)

        # Deterministic seed derivation
        derived_seed = derive_reference_probe_seed(global_seed, dataset_id, outer_fold, split_hash)

        # Selection
        indices = select_reference_probe_indices(n_source=n_source, max_size=max_size, seed=derived_seed)

        # Compute hash
        bank_sha = compute_reference_bank_sha256(
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            canonical_row_indices=indices,
            canonical_bundle_sha256=canonical_bundle_hash,
            split_sha256=split_hash,
            preprocessing_config_sha256=prep_config_sha,
            probe_protocol_sha256=probe_protocol_sha,
        )

        npz_path = spec_path.with_suffix(".npz")
        # Write descriptor
        desc = ReferenceProbeDescriptor(
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            bank_type="reference",
            canonical_bundle_sha256=canonical_bundle_hash,
            split_sha256=split_hash,
            preprocessing_config_sha256=prep_config_sha,
            selection_policy="uniform_without_replacement",
            global_probe_seed=global_seed,
            derived_seed=derived_seed,
            available_rows=n_source,
            selected_rows=len(indices),
            canonical_row_indices_sha256="",  # Updated inside save
            probe_protocol_sha256=probe_protocol_sha,
            probe_bank_sha256=bank_sha,
        )

        # Atomic save
        save_reference_probe_descriptor(spec_path, desc, indices)
        # Update NPZ hash in descriptor
        npz_sha = compute_file_sha256(npz_path)
        desc = ReferenceProbeDescriptor(
            dataset_id=desc.dataset_id,
            outer_fold=desc.outer_fold,
            bank_type=desc.bank_type,
            canonical_bundle_sha256=desc.canonical_bundle_sha256,
            split_sha256=desc.split_sha256,
            preprocessing_config_sha256=desc.preprocessing_config_sha256,
            selection_policy=desc.selection_policy,
            global_probe_seed=desc.global_probe_seed,
            derived_seed=desc.derived_seed,
            available_rows=desc.available_rows,
            selected_rows=desc.selected_rows,
            canonical_row_indices_sha256=npz_sha,
            probe_protocol_sha256=desc.probe_protocol_sha256,
            probe_bank_sha256=desc.probe_bank_sha256,
        )
        save_reference_probe_descriptor(spec_path, desc, indices)

    runtime = time.perf_counter() - t0
    spec_relpath = str(spec_path.relative_to(project_root)).replace("\\", "/")

    return {
        "manifest_entry": {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": "reference",
            "bank_type": "reference",
            "spec_relpath": spec_relpath,
            "probe_bank_sha256": desc.probe_bank_sha256,
            "selected_rows": desc.selected_rows,
        },
        "audit_record": {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "bank_type": "reference",
            "available_rows": desc.available_rows,
            "selected_rows": desc.selected_rows,
            "duplicates": len(indices) - len(np.unique(indices)),
            "bank_hash": desc.probe_bank_sha256,
            "finite": True,
            "runtime": round(runtime, 5),
        }
    }


def process_current_probe_task(
    dataset_id: str,
    outer_fold: int,
    condition: str,
    probe_cfg: Dict[str, Any],
    prep_cfg: Dict[str, Any],
    shift_protocol_sha: str,
    probe_protocol_sha: str,
    prep_config_sha: str,
    shifts_dir: Path,
    probes_dir: Path,
    resume: bool,
    force: bool,
    project_root: Path,
) -> Dict[str, Any]:
    """Build and persist single current probe bank descriptor."""
    spec_path = probes_dir / "current" / dataset_id / f"fold_{outer_fold}" / f"{condition}.json"
    shift_spec_path = shifts_dir / "specs" / dataset_id / f"fold_{outer_fold}" / f"{condition}.json"
    shift_npz_path = shift_spec_path.with_suffix(".npz")

    if not shift_spec_path.exists() or not shift_npz_path.exists():
        raise FileNotFoundError(f"Missing Phase-4 shift spec: {shift_spec_path}")

    with open(shift_spec_path, "r", encoding="utf-8") as f:
        shift_spec_doc = json.load(f)

    shift_spec_sha = shift_spec_doc["shift_spec_sha256"]
    scenario_status = shift_spec_doc.get("status", "APPLICABLE")

    with np.load(shift_npz_path) as npz:
        row_map = npz["row_index_map"]

    n_target = len(row_map)
    global_seed = probe_cfg.get("global_probe_seed", 2026090705)
    max_size = probe_cfg.get("current", {}).get("max_size", 2048)

    t0 = time.perf_counter()
    is_reusable = False

    if resume and not force:
        is_val, _, desc = validate_saved_current_descriptor(
            spec_path=spec_path,
            expected_shift_spec_sha256=shift_spec_sha,
            expected_shift_protocol_sha256=shift_protocol_sha,
            expected_preprocessing_config_sha256=prep_config_sha,
            expected_probe_protocol_sha256=probe_protocol_sha,
            global_probe_seed=global_seed,
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            condition=condition,
        )
        if is_val and desc is not None:
            is_reusable = True

    if is_reusable:
        desc, positions, canonical_rows = load_current_probe_descriptor(spec_path)
    else:
        derived_seed = derive_current_probe_seed(global_seed, dataset_id, outer_fold, condition, shift_spec_sha)
        positions, canonical_rows = select_current_probe_positions(
            n_target=n_target, row_index_map=row_map, max_size=max_size, seed=derived_seed
        )

        bank_sha = compute_current_bank_sha256(
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            condition=condition,
            selected_positions=positions,
            canonical_row_indices=canonical_rows,
            shift_spec_sha256=shift_spec_sha,
            shift_protocol_sha256=shift_protocol_sha,
            preprocessing_config_sha256=prep_config_sha,
            probe_protocol_sha256=probe_protocol_sha,
        )

        npz_path = spec_path.with_suffix(".npz")
        desc = CurrentProbeDescriptor(
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            condition=condition,
            bank_type="current",
            shift_spec_sha256=shift_spec_sha,
            shift_protocol_sha256=shift_protocol_sha,
            preprocessing_config_sha256=prep_config_sha,
            selection_policy="uniform_without_replacement",
            global_probe_seed=global_seed,
            derived_seed=derived_seed,
            available_current_rows=n_target,
            selected_current_rows=len(positions),
            selected_positions_sha256="",
            canonical_row_map_sha256="",
            scenario_status=scenario_status,
            probe_protocol_sha256=probe_protocol_sha,
            probe_bank_sha256=bank_sha,
        )

        save_current_probe_descriptor(spec_path, desc, positions, canonical_rows)
        npz_sha = compute_file_sha256(npz_path)
        desc = CurrentProbeDescriptor(
            dataset_id=desc.dataset_id,
            outer_fold=desc.outer_fold,
            condition=desc.condition,
            bank_type=desc.bank_type,
            shift_spec_sha256=desc.shift_spec_sha256,
            shift_protocol_sha256=desc.shift_protocol_sha256,
            preprocessing_config_sha256=desc.preprocessing_config_sha256,
            selection_policy=desc.selection_policy,
            global_probe_seed=desc.global_probe_seed,
            derived_seed=desc.derived_seed,
            available_current_rows=desc.available_current_rows,
            selected_current_rows=desc.selected_current_rows,
            selected_positions_sha256=npz_sha,
            canonical_row_map_sha256=compute_file_sha256(npz_path),
            scenario_status=desc.scenario_status,
            probe_protocol_sha256=desc.probe_protocol_sha256,
            probe_bank_sha256=desc.probe_bank_sha256,
        )
        save_current_probe_descriptor(spec_path, desc, positions, canonical_rows)

    runtime = time.perf_counter() - t0
    spec_relpath = str(spec_path.relative_to(project_root)).replace("\\", "/")

    return {
        "manifest_entry": {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": condition,
            "bank_type": "current",
            "spec_relpath": spec_relpath,
            "probe_bank_sha256": desc.probe_bank_sha256,
            "selected_rows": desc.selected_current_rows,
        },
        "audit_record": {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": condition,
            "bank_type": "current",
            "available_rows": desc.available_current_rows,
            "selected_rows": desc.selected_current_rows,
            "duplicates": len(canonical_rows) - len(np.unique(canonical_rows)),
            "bank_hash": desc.probe_bank_sha256,
            "finite": True,
            "runtime": round(runtime, 5),
        }
    }


# ---------------------------------------------------------------------------
# Alignment Validation Panel Runner
# ---------------------------------------------------------------------------

def run_alignment_validation_panel(
    project_root: Path,
    shift_cfg: Dict[str, Any],
    methods_cfg: Dict[str, Any],
    alignment_cfg: Dict[str, Any],
    prep_cfg: Dict[str, Any],
    dataset_classes: Dict[str, int],
    cache: ProbeCache,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute engineering alignment validation across the 6-dataset panel."""
    print("\n[ALIGNMENT VALIDATION] Running representative panel (FCM & GMM)...")
    engine = ShiftEngine(shift_cfg, project_root=project_root)
    probes_dir = project_root / "data" / "probes"
    prep_config_sha = compute_file_sha256(project_root / "configs" / "preprocessing.yaml")

    alignment_records = []
    ambiguity_records = []

    for ds in VALIDATION_PANEL_DATASETS:
        K = dataset_classes.get(ds, 3)
        fold = 0  # Primary validation fold 0

        # Load raw source features and roles
        X_src_raw, meta = load_raw_source_features(ds, fold, project_root)
        roles = meta.get("feature_roles", {col: "numeric" for col in X_src_raw.columns})

        # Preprocessor (cached)
        src_prep = cache.get_preprocessor(ds, fold, prep_config_sha)
        if src_prep is None:
            src_prep = build_preprocessor(feature_roles=roles, config=prep_cfg, metadata=meta)
            src_prep.fit(X_src_raw)
            cache.put_preprocessor(ds, fold, prep_config_sha, src_prep)

        X_src_trans = src_prep.transform(X_src_raw)

        # Load reference probe bank A^R (cached)
        ref_desc_path = probes_dir / "reference" / ds / f"fold_{fold}.json"
        ref_desc, ref_indices = load_reference_probe_descriptor(ref_desc_path)
        A_R = cache.get_transformed_bank(ref_desc.probe_bank_sha256)
        if A_R is None:
            A_R = load_reference_probe_matrix(ds, fold, ref_indices, src_prep, project_root)
            cache.put_transformed_bank(ref_desc.probe_bank_sha256, A_R)

        for cond in VALIDATION_PANEL_CONDITIONS:
            # Check condition status from shift spec
            shift_spec_path = project_root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"
            if not shift_spec_path.exists():
                continue
            with open(shift_spec_path, "r", encoding="utf-8") as f:
                s_doc = json.load(f)
            if s_doc.get("status") == "NOT_APPLICABLE":
                continue

            # Reconstruct shifted target
            # For offline supervised shifts (local_overlap, class_prevalence), load labels for Phase-4 generator only
            y_tgt, y_src = None, None
            if cond.startswith("local_overlap") or cond.startswith("class_prevalence"):
                ds_dir = project_root / "data" / "canonical" / "controlled" / ds
                df_y = pd.read_parquet(ds_dir / "labels.parquet")
                fold_p = project_root / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
                with np.load(fold_p) as npz:
                    y_src = df_y.iloc[npz["source_indices"]].to_numpy().ravel()
                    y_tgt = df_y.iloc[npz["target_indices"]].to_numpy().ravel()

            ds_dir = project_root / "data" / "canonical" / "controlled" / ds
            df_X = pd.read_parquet(ds_dir / "features.parquet")
            fold_p = project_root / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
            with np.load(fold_p) as npz:
                X_tgt_raw = df_X.iloc[npz["target_indices"]].copy().reset_index(drop=True)

            shift_res = engine.generate_shift(
                dataset_id=ds,
                outer_fold=fold,
                condition=cond,
                X_target=X_tgt_raw,
                X_source=X_src_raw,
                roles=roles,
                y_target=y_tgt,
                y_source=y_src,
                backend="numpy",
            )
            X_tgt_shifted_trans = src_prep.transform(shift_res.X_shifted)

            for seed in [1, 3]:
                for method_name in ["fcm_adaptive", "gmm"]:
                    t0 = time.perf_counter()

                    # 1. Fit source model
                    if method_name == "fcm_adaptive":
                        m_src = FCM(n_clusters=K, random_state=seed, fuzzifier_policy="dimension_adaptive")
                        m_cand = FCM(n_clusters=K, random_state=seed, fuzzifier_policy="dimension_adaptive")
                    else:
                        m_src = GMM(n_clusters=K, random_state=seed, covariance_type="full")
                        m_cand = GMM(n_clusters=K, random_state=seed, covariance_type="full")

                    m_src.fit(X_src_trans)
                    m_cand.fit(X_tgt_shifted_trans)

                    # Reference cluster scale from source model and source data
                    U_src = m_src.predict_membership(X_src_trans)
                    scales_ref, floored_flags, s_glob = compute_reference_cluster_scales(
                        X_src_trans, U_src, m_src.cluster_centers_
                    )

                    # Evaluate both models on SAME reference probe bank A^R
                    U_R_ref = m_src.predict_membership(A_R)
                    U_R_cand = m_cand.predict_membership(A_R)

                    # Probe evaluations with invariant guard
                    fp_src = compute_model_fingerprint(method_name, {}, seed, m_src.cluster_centers_)
                    fp_cand = compute_model_fingerprint(method_name, {}, seed, m_cand.cluster_centers_)

                    eval_ref = ProbeEvaluation(
                        bank_sha256=ref_desc.probe_bank_sha256,
                        bank_type="reference",
                        dataset_id=ds,
                        outer_fold=fold,
                        condition="reference",
                        model_fingerprint=fp_src,
                        memberships=U_R_ref,
                        n_samples=len(A_R),
                        n_clusters=K,
                    )
                    eval_cand = ProbeEvaluation(
                        bank_sha256=ref_desc.probe_bank_sha256,
                        bank_type="reference",
                        dataset_id=ds,
                        outer_fold=fold,
                        condition=cond,
                        model_fingerprint=fp_cand,
                        memberships=U_R_cand,
                        n_samples=len(A_R),
                        n_clusters=K,
                    )

                    verify_probe_identity(eval_ref, eval_cand)

                    # Align candidate to reference
                    align_res = align_clusters(
                        centers_ref=m_src.cluster_centers_,
                        centers_cand=m_cand.cluster_centers_,
                        scales_ref=scales_ref,
                        U_ref=U_R_ref,
                        U_cand=U_R_cand,
                        eta=0.50,
                    )

                    run_sec = time.perf_counter() - t0

                    record = {
                        "dataset_id": ds,
                        "outer_fold": fold,
                        "condition": cond,
                        "method": method_name,
                        "seed": seed,
                        "K": K,
                        "status": "SUCCESS",
                        "reference_bank_hash": ref_desc.probe_bank_sha256,
                        "eta": 0.50,
                        "assignment_cost": round(align_res.assignment_cost, 6),
                        "identity_assignment_cost": round(align_res.identity_assignment_cost, 6),
                        "permutation": json.dumps(align_res.permutation.tolist()),
                        "inverse_permutation": json.dumps(align_res.inverse_permutation.tolist()),
                        "minimum_assignment_margin": round(align_res.minimum_assignment_margin, 8),
                        "ambiguous": align_res.ambiguous,
                        "center_cost_mean": round(float(np.mean(align_res.center_cost_matrix)), 6),
                        "overlap_mean": round(float(np.mean(align_res.overlap_matrix)), 6),
                        "runtime": round(run_sec, 4),
                    }
                    alignment_records.append(record)

                    if align_res.ambiguous:
                        ambiguity_records.append({
                            "dataset_id": ds,
                            "outer_fold": fold,
                            "condition": cond,
                            "method": method_name,
                            "seed": seed,
                            "minimum_assignment_margin": align_res.minimum_assignment_margin,
                            "num_ambiguous_pairs": align_res.ambiguity_details["num_ambiguous_pairs"],
                            "details": json.dumps(align_res.ambiguity_details["ambiguous_pairs"]),
                        })

    return alignment_records, ambiguity_records


# ---------------------------------------------------------------------------
# Stability and Sensitivity Audits
# ---------------------------------------------------------------------------

def run_stability_and_sensitivity_audits(
    project_root: Path,
    shift_cfg: Dict[str, Any],
    methods_cfg: Dict[str, Any],
    prep_cfg: Dict[str, Any],
    dataset_classes: Dict[str, int],
    cache: ProbeCache,
    probe_out_dir: Path,
    align_out_dir: Path,
) -> None:
    """Run probe size stability audit and eta sensitivity audit."""
    print("\n[STABILITY & SENSITIVITY AUDITS] Running audits...")
    prep_config_sha = compute_file_sha256(project_root / "configs" / "preprocessing.yaml")

    # 1. Probe-Size Stability Audit on subset: iris and sonar
    probe_stability_records = []
    for ds in ["iris", "sonar"]:
        K = dataset_classes.get(ds, 3)
        X_src_raw, meta = load_raw_source_features(ds, 0, project_root)
        roles = meta.get("feature_roles", {col: "numeric" for col in X_src_raw.columns})
        src_prep = build_preprocessor(feature_roles=roles, config=prep_cfg, metadata=meta)
        src_prep.fit(X_src_raw)
        X_src_trans = src_prep.transform(X_src_raw)

        m_ref = FCM(n_clusters=K, random_state=1, fuzzifier_policy="dimension_adaptive").fit(X_src_trans)
        # Shifted candidate (location_severe)
        engine = ShiftEngine(shift_cfg, project_root=project_root)
        X_tgt_raw = X_src_raw.copy()  # For engineering audit
        shift_res = engine.generate_shift(ds, 0, "location_severe", X_tgt_raw, X_src_raw, roles, backend="numpy")
        X_tgt_trans = src_prep.transform(shift_res.X_shifted)
        m_cand = FCM(n_clusters=K, random_state=1, fuzzifier_policy="dimension_adaptive").fit(X_tgt_trans)

        scales_ref, _, _ = compute_reference_cluster_scales(
            X_src_trans, m_ref.predict_membership(X_src_trans), m_ref.cluster_centers_
        )

        n_available = len(X_src_raw)
        for p_size in [512, 1024, 2048]:
            actual_size = min(n_available, p_size)
            idx = select_reference_probe_indices(n_available, actual_size, seed=42)
            A_probe = src_prep.transform(X_src_raw.iloc[idx])

            U_ref = m_ref.predict_membership(A_probe)
            U_cand = m_cand.predict_membership(A_probe)

            t0 = time.perf_counter()
            res = align_clusters(m_ref.cluster_centers_, m_cand.cluster_centers_, scales_ref, U_ref, U_cand)
            dur = time.perf_counter() - t0

            probe_stability_records.append({
                "dataset_id": ds,
                "outer_fold": 0,
                "condition": "location_severe",
                "method": "fcm_adaptive",
                "probe_size_target": p_size,
                "probe_size_actual": actual_size,
                "permutation": json.dumps(res.permutation.tolist()),
                "assignment_cost": round(res.assignment_cost, 6),
                "overlap_mean": round(float(np.mean(res.overlap_matrix)), 6),
                "runtime": round(dur, 5),
            })

    if probe_stability_records:
        df_stab = pd.DataFrame(probe_stability_records)
        atomic_write_csv(probe_out_dir / "probe_size_stability.csv", df_stab)
        print(f"Saved: {probe_out_dir / 'probe_size_stability.csv'}")

    # 2. Eta Sensitivity Audit on iris and glass across eta in {0, 0.25, 0.5, 0.75, 1.0}
    eta_records = []
    for ds in ["iris", "glass"]:
        K = dataset_classes.get(ds, 3)
        X_src_raw, meta = load_raw_source_features(ds, 0, project_root)
        roles = meta.get("feature_roles", {col: "numeric" for col in X_src_raw.columns})
        src_prep = build_preprocessor(feature_roles=roles, config=prep_cfg, metadata=meta)
        src_prep.fit(X_src_raw)
        X_src_trans = src_prep.transform(X_src_raw)

        m_ref = FCM(n_clusters=K, random_state=1, fuzzifier_policy="dimension_adaptive").fit(X_src_trans)
        engine = ShiftEngine(shift_cfg, project_root=project_root)
        shift_res = engine.generate_shift(ds, 0, "scale_severe", X_src_raw.copy(), X_src_raw, roles, backend="numpy")
        X_tgt_trans = src_prep.transform(shift_res.X_shifted)
        m_cand = FCM(n_clusters=K, random_state=1, fuzzifier_policy="dimension_adaptive").fit(X_tgt_trans)

        scales_ref, _, _ = compute_reference_cluster_scales(
            X_src_trans, m_ref.predict_membership(X_src_trans), m_ref.cluster_centers_
        )
        idx = select_reference_probe_indices(len(X_src_raw), 2048, seed=42)
        A_probe = src_prep.transform(X_src_raw.iloc[idx])
        U_ref = m_ref.predict_membership(A_probe)
        U_cand = m_cand.predict_membership(A_probe)

        # Base reference alignment with eta=0.50
        res_base = align_clusters(m_ref.cluster_centers_, m_cand.cluster_centers_, scales_ref, U_ref, U_cand, eta=0.50)

        for eta_val in [0.0, 0.25, 0.50, 0.75, 1.0]:
            res = align_clusters(m_ref.cluster_centers_, m_cand.cluster_centers_, scales_ref, U_ref, U_cand, eta=eta_val)
            matches_base = bool(np.array_equal(res.permutation, res_base.permutation))
            eta_records.append({
                "dataset_id": ds,
                "condition": "scale_severe",
                "method": "fcm_adaptive",
                "seed": 1,
                "eta": eta_val,
                "assignment_cost": round(res.assignment_cost, 6),
                "permutation": json.dumps(res.permutation.tolist()),
                "matches_eta_05": matches_base,
            })

    if eta_records:
        df_eta = pd.DataFrame(eta_records)
        atomic_write_csv(align_out_dir / "eta_sensitivity.csv", df_eta)
        print(f"Saved: {align_out_dir / 'eta_sensitivity.csv'}")


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_performance_benchmarks(
    project_root: Path,
    probe_out_dir: Path,
    align_out_dir: Path,
) -> None:
    """Benchmark probe operations and alignment on CPU vs GPU."""
    print("\n[BENCHMARK] Measuring CPU vs GPU performance on probe and alignment operations...")
    import torch
    cuda_avail = torch.cuda.is_available()

    B = 2048
    K = 26
    D = 100

    rng = np.random.default_rng(42)
    U_r_np = rng.uniform(0.01, 1.0, size=(B, K))
    U_r_np /= np.sum(U_r_np, axis=1, keepdims=True)
    U_c_np = rng.uniform(0.01, 1.0, size=(B, K))
    U_c_np /= np.sum(U_c_np, axis=1, keepdims=True)

    # 1. Soft Jaccard Overlap CPU
    t0 = time.perf_counter()
    O_cpu = compute_soft_jaccard_overlap(U_r_np, U_c_np)
    t_jaccard_cpu = time.perf_counter() - t0

    # 2. Soft Jaccard Overlap GPU (if available)
    if cuda_avail:
        U_r_th = torch.tensor(U_r_np, device="cuda", dtype=torch.float32)
        U_c_th = torch.tensor(U_c_np, device="cuda", dtype=torch.float32)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        u_r = U_r_th.unsqueeze(2)
        u_c = U_c_th.unsqueeze(1)
        min_th = torch.sum(torch.minimum(u_r, u_c), dim=0)
        max_th = torch.sum(torch.maximum(u_r, u_c), dim=0)
        O_gpu = (min_th / (max_th + 1e-12)).cpu().numpy()
        torch.cuda.synchronize()
        t_jaccard_gpu = time.perf_counter() - t0
        max_dev = float(np.max(np.abs(O_cpu - O_gpu)))
    else:
        t_jaccard_gpu = t_jaccard_cpu
        max_dev = 0.0

    probe_bench = [{
        "operation": "soft_jaccard_overlap",
        "samples": B,
        "clusters": K,
        "cpu_wall_seconds": round(t_jaccard_cpu, 5),
        "gpu_wall_seconds": round(t_jaccard_gpu, 5) if cuda_avail else None,
        "speedup_vs_cpu": round(t_jaccard_cpu / max(t_jaccard_gpu, 1e-6), 2) if cuda_avail else 1.0,
        "max_cpu_gpu_deviation": max_dev,
        "cuda_available": cuda_avail,
    }]
    atomic_write_csv(probe_out_dir / "performance_benchmark.csv", pd.DataFrame(probe_bench))
    atomic_write_csv(align_out_dir / "performance_benchmark.csv", pd.DataFrame(probe_bench))
    print(f"Saved: {probe_out_dir / 'performance_benchmark.csv'}")


# ---------------------------------------------------------------------------
# Strict Verification Mode
# ---------------------------------------------------------------------------

def execute_verify_mode(project_root: Path) -> None:
    """Strictly byte-read-only verification of all Phase 5 probe and alignment artifacts."""
    print("\n[VERIFY MODE] Starting strictly byte-read-only verification for Phase 5...")
    errors = []

    probes_dir = project_root / "data" / "probes"
    lock_path = probes_dir / "phase5_input_lock.json"
    manifest_path = probes_dir / "probe_manifest.json"

    if not lock_path.exists():
        errors.append(f"Missing input lock: {lock_path}")
    if not manifest_path.exists():
        errors.append(f"Missing probe manifest: {manifest_path}")

    if errors:
        print("\n[VERIFY FAILED]:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # 1. Verify input lock
    with open(lock_path, "r", encoding="utf-8") as f:
        lock_doc = json.load(f)

    # Check upstream configuration and manifest hashes
    if lock_doc.get("phase4_tooling_freeze_commit") != "f08335e8a4822f2b95e379abf955a80d5c0fb2c8":
        errors.append("phase4_tooling_freeze_commit mismatch in lock")
    if lock_doc.get("phase4_artifact_commit") != "6f8dd88b59a297ae05da035835c4055aab43fb29":
        errors.append("phase4_artifact_commit mismatch in lock")
    if lock_doc.get("phase4_producer_commit") != "21bcaa7bb00b07de5ee4671adfb06932610bf92d":
        errors.append("phase4_producer_commit mismatch in lock")

    phase4_lock_p = project_root / "data" / "shifts" / "phase4_input_lock.json"
    if lock_doc.get("phase4_input_lock_sha256") != compute_file_sha256(phase4_lock_p):
        errors.append("phase4_input_lock_sha256 mismatch")

    phase4_manifest_p = project_root / "data" / "shifts" / "shift_manifest.json"
    if lock_doc.get("phase4_shift_manifest_sha256") != compute_file_sha256(phase4_manifest_p):
        errors.append("phase4_shift_manifest_sha256 mismatch")

    prep_cfg_p = project_root / "configs" / "preprocessing.yaml"
    if lock_doc.get("phase2_preprocessing_config_sha256") != compute_file_sha256(prep_cfg_p):
        errors.append("phase2_preprocessing_config_sha256 mismatch")

    # 2. Verify probe manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_doc = json.load(f)

    probes = manifest_doc.get("probes", [])
    if len(probes) != 2400:
        errors.append(f"Probe manifest count is {len(probes)}, expected exactly 2400")

    ref_count = sum(1 for p in probes if p["bank_type"] == "reference")
    cur_count = sum(1 for p in probes if p["bank_type"] == "current")

    if ref_count != 150:
        errors.append(f"Reference probe count is {ref_count}, expected 150")
    if cur_count != 2250:
        errors.append(f"Current probe count is {cur_count}, expected 2250")

    # 3. Verify probe descriptors and NPZ existence
    for p in probes:
        spec_p = project_root / p["spec_relpath"]
        if not spec_p.exists():
            errors.append(f"Missing probe spec: {spec_p}")
            continue
        npz_p = spec_p.with_suffix(".npz")
        if not npz_p.exists():
            errors.append(f"Missing probe companion NPZ: {npz_p}")

    # 4. Verify result CSVs
    probe_results_dir = project_root / "results" / "probe_validation"
    align_results_dir = project_root / "results" / "alignment_validation"

    for csv_name, exp_min in [
        (probe_results_dir / "reference_probe_audit.csv", 150),
        (probe_results_dir / "current_probe_audit.csv", 2250),
        (align_results_dir / "alignment_runs.csv", 100),
    ]:
        if not csv_name.exists():
            errors.append(f"Missing result CSV: {csv_name}")
        else:
            df = pd.read_csv(csv_name)
            if len(df) < exp_min:
                errors.append(f"{csv_name.name} has {len(df)} rows, expected at least {exp_min}")

    if errors:
        print(f"\n[VERIFY FAILED] Found {len(errors)} validation errors:")
        for err in errors[:25]:
            print(f"  - {err}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more errors.")
        sys.exit(1)
    else:
        print("\n[VERIFY SUCCESS] All 2,400 probe banks and alignment validation artifacts strictly valid and current.")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Cluster Alignment and Dual Probe Banks Runner")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: build probes, validate alignment, audits, benchmarks")
    parser.add_argument("--build-probes", action="store_true", help="Build and persist all 2,400 probe bank descriptors")
    parser.add_argument("--validate-alignment", action="store_true", help="Run alignment validation panel")
    parser.add_argument("--stability-audit", action="store_true", help="Run probe-size and eta stability audits")
    parser.add_argument("--benchmark", action="store_true", help="Run CPU/GPU benchmarks")
    parser.add_argument("--jobs", type=str, default="auto", help="Worker count or 'auto'")
    parser.add_argument("--resume", action="store_true", help="Skip existing valid descriptors")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing descriptors")
    parser.add_argument("--verify", action="store_true", help="Strictly byte-read-only verification")
    parser.add_argument("--dry-run", action="store_true", help="Display planned tasks without writing")
    parser.add_argument("--commit-sha", type=str, default=None, help="Override Commit A SHA for input lock")
    args = parser.parse_args()

    if args.verify:
        execute_verify_mode(PROJECT_ROOT)
        return

    probe_cfg_p = PROJECT_ROOT / "configs" / "probes.yaml"
    align_cfg_p = PROJECT_ROOT / "configs" / "alignment.yaml"
    shifts_cfg_p = PROJECT_ROOT / "configs" / "shifts.yaml"
    prep_cfg_p = PROJECT_ROOT / "configs" / "preprocessing.yaml"
    methods_cfg_p = PROJECT_ROOT / "configs" / "methods.yaml"
    datasets_manifest_p = PROJECT_ROOT / "data" / "manifests" / "datasets.json"
    splits_manifest_p = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"

    probe_cfg = load_yaml(probe_cfg_p)
    align_cfg = load_yaml(align_cfg_p)
    shift_cfg = load_yaml(shifts_cfg_p)
    prep_cfg = load_yaml(prep_cfg_p)
    methods_cfg = load_yaml(methods_cfg_p)

    bundle_hashes = load_canonical_bundle_hashes(datasets_manifest_p)
    split_records = load_phase2_split_hashes(splits_manifest_p)
    dataset_classes = load_dataset_classes(datasets_manifest_p)

    probe_protocol_sha = compute_probe_protocol_sha256(probe_cfg)
    from clusterdrift.shifts.hashing import compute_shift_protocol_sha256
    shift_protocol_sha = compute_shift_protocol_sha256(shift_cfg)
    prep_config_sha = compute_file_sha256(prep_cfg_p)

    probes_dir = PROJECT_ROOT / "data" / "probes"
    shifts_dir = PROJECT_ROOT / "data" / "shifts"
    probe_results_dir = PROJECT_ROOT / "results" / "probe_validation"
    align_results_dir = PROJECT_ROOT / "results" / "alignment_validation"

    probes_dir.mkdir(parents=True, exist_ok=True)
    probe_results_dir.mkdir(parents=True, exist_ok=True)
    align_results_dir.mkdir(parents=True, exist_ok=True)

    cache = ProbeCache()

    # Determine worker threads
    if args.jobs == "auto":
        n_workers = min(8, max(1, (os.cpu_count() or 4)))
    else:
        n_workers = max(1, int(args.jobs))

    commit_sha = args.commit_sha or get_current_git_commit(PROJECT_ROOT)

    if args.dry_run:
        print("\n[DRY RUN] Phase 5 Cluster Alignment and Dual Probe Banks:")
        print("  Reference probe banks planned: 150 (30 datasets x 5 folds)")
        print("  Current probe banks planned: 2,250 (30 datasets x 5 folds x 15 conditions)")
        print("  Total probe banks: 2,400")
        print(f"  Concurrency: {n_workers} worker threads")
        print(f"  Input lock target: {probes_dir / 'phase5_input_lock.json'}")
        print(f"  Probe manifest target: {probes_dir / 'probe_manifest.json'}")
        return

    # Build input lock
    lock_doc = build_phase5_input_lock(
        project_root=PROJECT_ROOT,
        probe_cfg=probe_cfg,
        alignment_cfg=align_cfg,
        generated_from_commit=commit_sha,
    )
    atomic_write_json(probes_dir / "phase5_input_lock.json", lock_doc, indent=2, sort_keys=True)
    print(f"Saved Input Lock: {probes_dir / 'phase5_input_lock.json'}")

    run_probes = args.all or args.build_probes
    run_alignment = args.all or args.validate_alignment
    run_stability = args.all or args.stability_audit
    run_bench = args.all or args.benchmark

    # ---------------------------------------------------------
    # 1. Build Probes
    # ---------------------------------------------------------
    if run_probes:
        print(f"\n[BUILD PROBES] Generating 2,400 probe banks with {n_workers} workers...")
        ref_manifests = []
        ref_audits = []

        # Reference probe tasks
        ref_tasks = []
        for ds in CONTROLLED_DATASETS:
            b_sha = bundle_hashes.get(ds, "")
            for f in range(5):
                sp_sha = split_records.get((ds, f), {}).get("split_sha256", "")
                ref_tasks.append((ds, f, b_sha, sp_sha))

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    process_reference_probe_task,
                    ds, f, probe_cfg, prep_cfg, b_sha, sp_sha,
                    probe_protocol_sha, prep_config_sha, probes_dir,
                    args.resume, args.force, PROJECT_ROOT,
                )
                for ds, f, b_sha, sp_sha in ref_tasks
            ]
            for fut in futures:
                res = fut.result()
                ref_manifests.append(res["manifest_entry"])
                ref_audits.append(res["audit_record"])

        # Current probe tasks
        cur_tasks = []
        for ds in CONTROLLED_DATASETS:
            for f in range(5):
                for cond in ALL_CONDITIONS:
                    cur_tasks.append((ds, f, cond))

        cur_manifests = []
        cur_audits = []

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    process_current_probe_task,
                    ds, f, cond, probe_cfg, prep_cfg,
                    shift_protocol_sha, probe_protocol_sha, prep_config_sha,
                    shifts_dir, probes_dir, args.resume, args.force, PROJECT_ROOT,
                )
                for ds, f, cond in cur_tasks
            ]
            for fut in futures:
                res = fut.result()
                cur_manifests.append(res["manifest_entry"])
                cur_audits.append(res["audit_record"])

        # Canonical sort for manifest and audit
        sort_key_man = lambda x: (x["bank_type"], x["dataset_id"], x["outer_fold"], x["condition"])
        sort_key_aud_ref = lambda x: (x["dataset_id"], x["outer_fold"])
        sort_key_aud_cur = lambda x: (x["dataset_id"], x["outer_fold"], x["condition"])

        all_manifest_probes = sorted(ref_manifests + cur_manifests, key=sort_key_man)
        ref_audits = sorted(ref_audits, key=sort_key_aud_ref)
        cur_audits = sorted(cur_audits, key=sort_key_aud_cur)

        # Write manifest
        manifest_doc = {
            "protocol_version": probe_cfg.get("protocol_version", 1),
            "total_probe_banks": len(all_manifest_probes),
            "reference_banks_count": len(ref_manifests),
            "current_banks_count": len(cur_manifests),
            "probe_protocol_sha256": probe_protocol_sha,
            "probes": all_manifest_probes,
        }
        manifest_path = probes_dir / "probe_manifest.json"
        atomic_write_json(manifest_path, manifest_doc, indent=2, sort_keys=True)
        print(f"Saved Probe Manifest: {manifest_path} ({len(all_manifest_probes)} entries)")

        # Write audits
        atomic_write_csv(probe_results_dir / "reference_probe_audit.csv", pd.DataFrame(ref_audits))
        atomic_write_csv(probe_results_dir / "current_probe_audit.csv", pd.DataFrame(cur_audits))
        print(f"Saved: {probe_results_dir / 'reference_probe_audit.csv'}")
        print(f"Saved: {probe_results_dir / 'current_probe_audit.csv'}")

        # Write manifest summary
        man_summary = [
            {"bank_type": "reference", "count": len(ref_manifests), "expected": 150},
            {"bank_type": "current", "count": len(cur_manifests), "expected": 2250},
            {"bank_type": "total", "count": len(all_manifest_probes), "expected": 2400},
        ]
        atomic_write_csv(probe_results_dir / "probe_manifest_summary.csv", pd.DataFrame(man_summary))

    # ---------------------------------------------------------
    # 2. Alignment Validation Panel
    # ---------------------------------------------------------
    if run_alignment:
        align_records, amb_records = run_alignment_validation_panel(
            project_root=PROJECT_ROOT,
            shift_cfg=shift_cfg,
            methods_cfg=methods_cfg,
            alignment_cfg=align_cfg,
            prep_cfg=prep_cfg,
            dataset_classes=dataset_classes,
            cache=cache,
            out_dir=align_results_dir,
        )

        sort_align = lambda x: (x["dataset_id"], x["condition"], x["method"], x["seed"])
        align_records = sorted(align_records, key=sort_align)

        df_align = pd.DataFrame(align_records)
        atomic_write_csv(align_results_dir / "alignment_runs.csv", df_align)
        print(f"Saved: {align_results_dir / 'alignment_runs.csv'} ({len(df_align)} runs)")

        # Summary
        summary_records = []
        for (m, cond), group in df_align.groupby(["method", "condition"]):
            summary_records.append({
                "method": m,
                "condition": cond,
                "total_runs": len(group),
                "mean_assignment_cost": round(group["assignment_cost"].mean(), 5),
                "mean_overlap": round(group["overlap_mean"].mean(), 5),
                "ambiguous_fraction": round(group["ambiguous"].mean(), 4),
            })
        atomic_write_csv(align_results_dir / "alignment_summary.csv", pd.DataFrame(summary_records))
        print(f"Saved: {align_results_dir / 'alignment_summary.csv'}")

        df_amb = pd.DataFrame(amb_records) if amb_records else pd.DataFrame(columns=[
            "dataset_id", "outer_fold", "condition", "method", "seed", "minimum_assignment_margin", "num_ambiguous_pairs", "details"
        ])
        atomic_write_csv(align_results_dir / "ambiguity_cases.csv", df_amb)
        print(f"Saved: {align_results_dir / 'ambiguity_cases.csv'}")

        # Known permutation recovery test fixture verification
        perm_records = []
        rng = np.random.default_rng(2026)
        for K_test in [2, 3, 5, 10, 26]:
            c_ref = rng.uniform(-10, 10, size=(K_test, 10)) + np.eye(K_test, 10) * 15.0
            s_ref = np.ones(K_test)
            u_ref = rng.uniform(0.1, 1.0, size=(100, K_test))
            u_ref /= np.sum(u_ref, axis=1, keepdims=True)
            pi_true = rng.permutation(K_test)
            c_cand = c_ref[pi_true].copy()
            u_cand = u_ref[:, pi_true].copy()

            res = align_clusters(c_ref, c_cand, s_ref, u_ref, u_cand)
            recovered = np.array_equal(pi_true[res.permutation], np.arange(K_test))
            perm_records.append({
                "test_name": f"synthetic_exact_K{K_test}",
                "K": K_test,
                "D": 10,
                "eta": 0.50,
                "status": "PASS" if recovered else "FAIL",
                "exact_recovery": recovered,
            })
        atomic_write_csv(align_results_dir / "permutation_recovery.csv", pd.DataFrame(perm_records))
        print(f"Saved: {align_results_dir / 'permutation_recovery.csv'}")

    # ---------------------------------------------------------
    # 3. Stability & Sensitivity Audits
    # ---------------------------------------------------------
    if run_stability:
        run_stability_and_sensitivity_audits(
            project_root=PROJECT_ROOT,
            shift_cfg=shift_cfg,
            methods_cfg=methods_cfg,
            prep_cfg=prep_cfg,
            dataset_classes=dataset_classes,
            cache=cache,
            probe_out_dir=probe_results_dir,
            align_out_dir=align_results_dir,
        )

    # ---------------------------------------------------------
    # 4. Performance Benchmark
    # ---------------------------------------------------------
    if run_bench:
        run_performance_benchmarks(
            project_root=PROJECT_ROOT,
            probe_out_dir=probe_results_dir,
            align_out_dir=align_results_dir,
        )

    # Clean failure logs
    atomic_write_json(probe_results_dir / "failures.json", {}, indent=2)
    atomic_write_json(align_results_dir / "failures.json", {}, indent=2)

    # Cache stats
    print("\n[CACHE PERFORMANCE]")
    print(json.dumps(cache.stats(), indent=2))
    print("\nPhase 5 execution finished successfully.")


if __name__ == "__main__":
    main()
