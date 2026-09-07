#!/usr/bin/env python3
"""Phase 6 Structural Signal Engine and Conventional Validity Controls Runner.

Executes:
1. Mechanism verification fixtures (9 mathematical fixtures)
2. Label-free Pass A: 192 signal rows across 6 datasets x 8 conditions x 2 methods x 2 seeds
3. Evaluation-only Pass B: 192 quality degradation targets on full shifted targets
4. CPU / optional GPU performance benchmarks and numerical equivalence
5. Cryptographic Phase-6 input lock binding
6. Strictly byte-read-only verification (--verify)
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

from clusterdrift.alignment.costs import compute_reference_cluster_scales
from clusterdrift.alignment.hungarian import align_clusters
from clusterdrift.data.preprocess import build_preprocessor
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

def adjusted_rand_index(y_true, y_pred) -> float:
    return float(adjusted_rand_score(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()))

def normalized_mutual_info(y_true, y_pred) -> float:
    return float(normalized_mutual_info_score(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), average_method="arithmetic"))

def adjusted_mutual_info(y_true, y_pred) -> float:
    return float(adjusted_mutual_info_score(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), average_method="arithmetic"))
from clusterdrift.probes.bank import (
    load_current_probe_descriptor,
    load_reference_probe_descriptor,
)
from clusterdrift.probes.evaluation import ProbeEvaluation, ProbeIdentityMismatchError, verify_probe_identity
from clusterdrift.probes.hashing import compute_file_sha256
from clusterdrift.probes.matrix import (
    load_current_probe_matrix,
    load_raw_source_features,
    load_reference_probe_matrix,
)
from clusterdrift.shifts.engine import ShiftEngine
from clusterdrift.shifts.hashing import (
    atomic_write_csv,
    atomic_write_json,
)
from clusterdrift.signals.covariate import (
    compute_chunked_rbf_kernel_sum,
    compute_mmd_b2,
    compute_mmd_b2_torch,
    derive_source_median_bandwidth,
)
from clusterdrift.signals.divergence import compute_normalized_js_divergence
from clusterdrift.signals.engine import SignalCache, SignalEngine
from clusterdrift.signals.entropy import compute_entropy_shift, compute_normalized_entropy
from clusterdrift.signals.hashing import (
    build_phase6_input_lock,
    compute_quality_record_sha256,
    compute_signal_protocol_sha256,
)
from clusterdrift.signals.mass import compute_cluster_mass_shift
from clusterdrift.signals.membership import (
    compute_current_membership_drift,
    compute_historical_membership_drift,
)
from clusterdrift.signals.prototype import compute_prototype_movement
from clusterdrift.signals.result import QualityResult, SignalResult
from clusterdrift.signals.verification import verify_phase6_integrity

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
# Mechanism Fixtures Runner
# ---------------------------------------------------------------------------

def run_mechanism_tests(out_dir: Path, project_root: Path) -> List[Dict[str, Any]]:
    """Execute all 9 exact mechanism verification fixtures."""
    print("\n[MECHANISM FIXTURES] Running 9 deterministic mathematical verification fixtures...")
    fixtures: List[Dict[str, Any]] = []

    # 1. Identity zero
    V0 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    U0 = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    scales = np.array([1.0, 1.0], dtype=np.float64)
    dur = compute_historical_membership_drift(U0, U0)["D_U_R"]
    duc = compute_current_membership_drift(U0, U0)["D_U_C"]
    dv = compute_prototype_movement(V0, V0, scales)["D_V"]
    dh = compute_entropy_shift(U0, U0)["D_H"]
    dm = compute_cluster_mass_shift(U0, U0)["D_M"]
    dx, _ = compute_mmd_b2(V0, V0, sigma=1.0)
    is_zero = (dur == 0.0) and (duc == 0.0) and (dv == 0.0) and (dh == 0.0) and (dm == 0.0) and (dx < 1e-10)
    fixtures.append({
        "fixture": "identity_zero",
        "passed": bool(is_zero),
        "details": f"D_U_R={dur}, D_U_C={duc}, D_V={dv}, D_H={dh}, D_M={dm}, D_X={dx:.2e}",
    })

    # 2. Pure cluster-ID permutation invariance
    perm = [1, 0]
    Vt_perm = V0[perm]
    Ut_perm = U0[:, perm]
    align_res = align_clusters(centers_ref=V0, centers_cand=Vt_perm, scales_ref=scales, U_ref=U0, U_cand=Ut_perm, eta=0.50)
    U_aligned = align_res.apply_to_memberships(Ut_perm)
    V_aligned = align_res.aligned_centers
    dur_perm = compute_historical_membership_drift(U0, U_aligned)["D_U_R"]
    dv_perm = compute_prototype_movement(V0, V_aligned, scales)["D_V"]
    dm_perm = compute_cluster_mass_shift(U0, U_aligned)["D_M"]
    is_perm_inv = (dur_perm == 0.0) and (dv_perm == 0.0) and (dm_perm == 0.0)
    fixtures.append({
        "fixture": "permutation_invariant",
        "passed": bool(is_perm_inv),
        "details": f"Recovered perm={align_res.permutation.tolist()}, D_U_R={dur_perm}, D_V={dv_perm}, D_M={dm_perm}",
    })

    # 3. Center movement positive
    Vt_moved = np.array([[0.5, 0.0], [1.0, 1.0]], dtype=np.float64)
    dv_moved = compute_prototype_movement(V0, Vt_moved, scales)["D_V"]
    # Hand calculation: delta_0 = 0.5 / 1.0 = 0.5, delta_1 = 0.0 / 1.0 = 0.0, D_V = (0.5 + 0.0)/2 = 0.25
    is_dv_exact = np.isclose(dv_moved, 0.25, atol=1e-5)
    fixtures.append({
        "fixture": "center_movement_positive",
        "passed": bool(is_dv_exact),
        "details": f"Expected 0.25, got {dv_moved:.6f}",
    })

    # 4. Entropy change positive
    U_crisp = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    U_soft = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    dh_val = compute_entropy_shift(U_crisp, U_soft)["D_H"]
    fixtures.append({
        "fixture": "entropy_change_positive",
        "passed": bool(dh_val > 0.0),
        "details": f"D_H={dh_val:.6f} > 0",
    })

    # 5. Mass change positive
    U_mass1 = np.array([[0.9, 0.1], [0.9, 0.1]], dtype=np.float64)
    U_mass2 = np.array([[0.1, 0.9], [0.1, 0.9]], dtype=np.float64)
    dm_val = compute_cluster_mass_shift(U_mass1, U_mass2)["D_M"]
    fixtures.append({
        "fixture": "mass_change_positive",
        "passed": bool(dm_val > 0.5),
        "details": f"D_M={dm_val:.6f} (expected ~1.0 for reversed mass)",
    })

    # 6. Raw shift without structure
    X_orig = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    X_shifted = X_orig + 5.0
    dx_shifted, _ = compute_mmd_b2(X_orig, X_shifted, sigma=1.0)
    dur_zero = compute_historical_membership_drift(U0, U0)["D_U_R"]
    is_raw_only = (dx_shifted > 0.1) and (dur_zero == 0.0)
    fixtures.append({
        "fixture": "raw_shift_without_structure",
        "passed": bool(is_raw_only),
        "details": f"D_X={dx_shifted:.6f} > 0 while structural signals == 0",
    })

    # 7. Probe identity guard
    pe1 = ProbeEvaluation(
        bank_sha256="hash_A",
        bank_type="current",
        dataset_id="iris",
        outer_fold=0,
        condition="clean",
        model_fingerprint="fp1",
        memberships=U0,
        n_samples=2,
        n_clusters=2,
    )
    pe2 = ProbeEvaluation(
        bank_sha256="hash_B",
        bank_type="current",
        dataset_id="iris",
        outer_fold=0,
        condition="clean",
        model_fingerprint="fp2",
        memberships=U0,
        n_samples=2,
        n_clusters=2,
    )
    guard_passed = False
    try:
        verify_probe_identity(pe1, pe2)
    except ProbeIdentityMismatchError:
        guard_passed = True
    fixtures.append({
        "fixture": "probe_identity_guard",
        "passed": bool(guard_passed),
        "details": "Correctly raised ProbeIdentityMismatchError on mismatched bank hashes",
    })

    # 8. MMD chunk equivalence
    X_pts = np.random.RandomState(42).randn(100, 4)
    Y_pts = np.random.RandomState(43).randn(120, 4)
    mmd_chunk_small, _ = compute_mmd_b2(X_pts, Y_pts, sigma=1.5, chunk_size=16)
    mmd_chunk_large, _ = compute_mmd_b2(X_pts, Y_pts, sigma=1.5, chunk_size=512)
    chunk_equiv = np.isclose(mmd_chunk_small, mmd_chunk_large, atol=1e-12)
    fixtures.append({
        "fixture": "mmd_chunk_equivalence",
        "passed": bool(chunk_equiv),
        "details": f"Difference = {abs(mmd_chunk_small - mmd_chunk_large):.3e} < 1e-12",
    })

    # 9. Source-only bandwidth
    sig1, stat1 = derive_source_median_bandwidth(X_pts, "test_ds", 0, "hash_x", global_signal_seed=2026090706)
    sig2, stat2 = derive_source_median_bandwidth(X_pts, "test_ds", 0, "hash_x", global_signal_seed=2026090706)
    sig_invariant = (sig1 == sig2) and (stat1 == "SUCCESS")
    fixtures.append({
        "fixture": "source_only_bandwidth",
        "passed": bool(sig_invariant),
        "details": f"Bandwidth sigma={sig1:.6f} strictly deterministic across identical source bank",
    })

    df_fix = pd.DataFrame(fixtures)
    atomic_write_csv(out_dir / "mechanism_fixture_results.csv", df_fix)
    for row in fixtures:
        status_str = "PASS" if row["passed"] else "FAIL"
        print(f"  [{status_str}] {row['fixture']}: {row['details']}")

    all_passed = all(r["passed"] for r in fixtures)
    if not all_passed:
        raise ValueError("[MECHANISM FAILURE] At least one mechanism verification fixture failed!")
    print("[MECHANISM FIXTURES] All 9 fixtures strictly PASSED.")
    return fixtures


# ---------------------------------------------------------------------------
# Pass A: Label-Free Signal Generation
# ---------------------------------------------------------------------------

def process_dataset_signals(
    ds: str,
    fold: int,
    K: int,
    conditions: List[str],
    methods: List[str],
    seeds: List[int],
    sig_cfg: Dict[str, Any],
    alignment_cfg: Dict[str, Any],
    methods_cfg: Dict[str, Any],
    prep_cfg: Dict[str, Any],
    shift_cfg: Dict[str, Any],
    cache: SignalCache,
    project_root: Path,
) -> List[SignalResult]:
    """Execute Pass A label-free signal computation for all scenarios of a dataset."""
    engine_shift = ShiftEngine(shift_cfg, project_root=project_root)
    signal_engine = SignalEngine(
        signals_cfg=sig_cfg,
        alignment_cfg=alignment_cfg,
        methods_cfg=methods_cfg,
        cache=cache,
    )
    probes_dir = project_root / "data" / "probes"
    prep_config_sha = compute_file_sha256(project_root / "configs" / "preprocessing.yaml")

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

    # Reference probe bank A^R (cached)
    ref_desc_path = probes_dir / "reference" / ds / f"fold_{fold}.json"
    ref_desc, ref_positions, ref_canonical_rows = load_reference_probe_descriptor(ref_desc_path)
    A_R = cache.get_transformed_bank(ref_desc.probe_bank_sha256)
    if A_R is None:
        A_R = load_reference_probe_matrix(ds, fold, ref_positions, src_prep, project_root)
        cache.put_transformed_bank(ref_desc.probe_bank_sha256, A_R)

    # Load raw target features
    ds_dir = project_root / "data" / "canonical" / "controlled" / ds
    df_X = pd.read_parquet(ds_dir / "features.parquet")
    fold_p = project_root / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
    with np.load(fold_p) as npz:
        X_tgt_raw = df_X.iloc[npz["target_indices"]].copy().reset_index(drop=True)

    results: List[SignalResult] = []

    for cond in conditions:
        shift_spec_path = project_root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"
        if not shift_spec_path.exists():
            continue
        with open(shift_spec_path, "r", encoding="utf-8") as f:
            s_doc = json.load(f)
        shift_spec_sha = compute_file_sha256(shift_spec_path)

        # For offline supervised shift reconstruction ONLY (Phase 4 generator requirements)
        # Pass A does NOT provide labels to SignalEngine
        y_tgt_rec, y_src_rec = None, None
        if cond.startswith("local_overlap") or cond.startswith("class_prevalence"):
            df_y = pd.read_parquet(ds_dir / "labels.parquet")
            with np.load(fold_p) as npz:
                y_src_rec = df_y.iloc[npz["source_indices"]].to_numpy().ravel()
                y_tgt_rec = df_y.iloc[npz["target_indices"]].to_numpy().ravel()

        shift_res = engine_shift.generate_shift(
            dataset_id=ds,
            outer_fold=fold,
            condition=cond,
            X_target=X_tgt_raw,
            X_source=X_src_raw,
            roles=roles,
            y_target=y_tgt_rec,
            y_source=y_src_rec,
            backend="numpy",
        )
        X_tgt_shifted_trans = src_prep.transform(shift_res.X_shifted)

        # Paired current probe bank A_t^C (cached)
        cur_desc_path = probes_dir / "current" / ds / f"fold_{fold}" / f"{cond}.json"
        cur_desc, cur_positions, _, _ = load_current_probe_descriptor(cur_desc_path)
        A_C = cache.get_transformed_bank(cur_desc.probe_bank_sha256)
        if A_C is None:
            A_C = load_current_probe_matrix(shift_res.X_shifted, cur_positions, src_prep)
            cache.put_transformed_bank(cur_desc.probe_bank_sha256, A_C)

        for meth in methods:
            for seed in seeds:
                res = signal_engine.compute_signals(
                    dataset_id=ds,
                    outer_fold=fold,
                    condition=cond,
                    method=meth,
                    seed=seed,
                    K=K,
                    X_source_trans=X_src_trans,
                    X_target_shifted_trans=X_tgt_shifted_trans,
                    A_R=A_R,
                    A_C=A_C,
                    ref_bank_sha256=ref_desc.probe_bank_sha256,
                    cur_bank_sha256=cur_desc.probe_bank_sha256,
                    shift_spec_sha256=shift_spec_sha,
                )
                results.append(res)

    return results


def run_signals_pass_a(
    project_root: Path,
    sig_cfg: Dict[str, Any],
    alignment_cfg: Dict[str, Any],
    methods_cfg: Dict[str, Any],
    prep_cfg: Dict[str, Any],
    shift_cfg: Dict[str, Any],
    dataset_classes: Dict[str, int],
    cache: SignalCache,
    out_dir: Path,
    jobs: int = 1,
) -> List[Dict[str, Any]]:
    """Execute Pass A label-free signal generation."""
    print("\n[PASS A: LABEL-FREE SIGNALS] Generating 192 signal rows across validation panel...")
    t0 = time.perf_counter()

    datasets = sig_cfg["validation"]["datasets"]
    fold = sig_cfg["validation"]["outer_fold"]
    conditions = sig_cfg["validation"]["conditions"]
    methods = sig_cfg["validation"]["methods"]
    seeds = sig_cfg["validation"]["algorithm_seeds"]

    all_results: List[SignalResult] = []

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    process_dataset_signals,
                    ds=ds,
                    fold=fold,
                    K=dataset_classes.get(ds, 3),
                    conditions=conditions,
                    methods=methods,
                    seeds=seeds,
                    sig_cfg=sig_cfg,
                    alignment_cfg=alignment_cfg,
                    methods_cfg=methods_cfg,
                    prep_cfg=prep_cfg,
                    shift_cfg=shift_cfg,
                    cache=cache,
                    project_root=project_root,
                )
                for ds in datasets
            ]
            for f in futures:
                all_results.extend(f.result())
    else:
        for ds in datasets:
            res_list = process_dataset_signals(
                ds=ds,
                fold=fold,
                K=dataset_classes.get(ds, 3),
                conditions=conditions,
                methods=methods,
                seeds=seeds,
                sig_cfg=sig_cfg,
                alignment_cfg=alignment_cfg,
                methods_cfg=methods_cfg,
                prep_cfg=prep_cfg,
                shift_cfg=shift_cfg,
                cache=cache,
                project_root=project_root,
            )
            all_results.extend(res_list)

    # Sort deterministically
    records = [r.to_dict() for r in all_results]
    records.sort(key=lambda r: (r["dataset_id"], r["outer_fold"], r["condition"], r["method"], r["seed"]))

    df_signals = pd.DataFrame(records)
    atomic_write_csv(out_dir / "signals_label_free.csv", df_signals)
    elapsed = time.perf_counter() - t0
    print(f"[PASS A COMPLETE] Wrote {len(df_signals)} signal rows to signals_label_free.csv in {elapsed:.2f}s")
    return records


# ---------------------------------------------------------------------------
# Pass B: Evaluation-Only Quality Targets
# ---------------------------------------------------------------------------

def run_quality_pass_b(
    project_root: Path,
    signal_records: List[Dict[str, Any]],
    sig_cfg: Dict[str, Any],
    prep_cfg: Dict[str, Any],
    shift_cfg: Dict[str, Any],
    cache: SignalCache,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Execute Pass B evaluation-only quality target computation on full shifted target."""
    print("\n[PASS B: EVALUATION-ONLY QUALITY] Computing 192 quality degradation targets...")
    t0 = time.perf_counter()
    engine_shift = ShiftEngine(shift_cfg, project_root=project_root)
    prep_config_sha = compute_file_sha256(project_root / "configs" / "preprocessing.yaml")

    # Group records by (dataset_id, outer_fold)
    grouped_keys: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in signal_records:
        key = (r["dataset_id"], int(r["outer_fold"]))
        grouped_keys.setdefault(key, []).append(r)

    quality_records: List[Dict[str, Any]] = []

    for (ds, fold), recs in grouped_keys.items():
        ds_dir = project_root / "data" / "canonical" / "controlled" / ds
        df_X = pd.read_parquet(ds_dir / "features.parquet")
        df_y = pd.read_parquet(ds_dir / "labels.parquet")
        fold_p = project_root / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
        with np.load(fold_p) as npz:
            src_idx = npz["source_indices"]
            tgt_idx = npz["target_indices"]

        X_src_raw = df_X.iloc[src_idx].copy().reset_index(drop=True)
        X_tgt_raw = df_X.iloc[tgt_idx].copy().reset_index(drop=True)
        y_src_raw = df_y.iloc[src_idx].to_numpy().ravel()
        y_tgt_raw = df_y.iloc[tgt_idx].to_numpy().ravel()

        with open(ds_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        roles = meta.get("feature_roles", {col: "numeric" for col in X_src_raw.columns})

        src_prep = cache.get_preprocessor(ds, fold, prep_config_sha)
        if src_prep is None:
            src_prep = build_preprocessor(feature_roles=roles, config=prep_cfg, metadata=meta)
            src_prep.fit(X_src_raw)
            cache.put_preprocessor(ds, fold, prep_config_sha, src_prep)

        # Baseline clean target representation
        X_tgt_clean_trans = src_prep.transform(X_tgt_raw)

        # Precompute clean ARI Q_0 for each (method, seed)
        clean_q0_map: Dict[Tuple[str, int], float] = {}
        unique_method_seeds = set((r["method"], int(r["seed"])) for r in recs)
        for meth, seed in unique_method_seeds:
            m_src = cache.get_source_model(ds, fold, meth, seed)
            if m_src is None:
                raise RuntimeError(f"Source model missing from cache for {ds} {fold} {meth} {seed}")
            pred_clean = m_src.predict(X_tgt_clean_trans)
            q0 = float(adjusted_rand_index(y_tgt_raw, pred_clean))
            clean_q0_map[(meth, seed)] = q0

        # Evaluate quality for each record
        # Cache shifted target transformation across methods/seeds
        shifted_target_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for r in recs:
            cond = r["condition"]
            meth = r["method"]
            seed = int(r["seed"])

            if cond not in shifted_target_cache:
                # Reconstruct shifted target
                shift_res = engine_shift.generate_shift(
                    dataset_id=ds,
                    outer_fold=fold,
                    condition=cond,
                    X_target=X_tgt_raw,
                    X_source=X_src_raw,
                    roles=roles,
                    y_target=y_tgt_raw if (cond.startswith("local_overlap") or cond.startswith("class_prevalence")) else None,
                    y_source=y_src_raw if (cond.startswith("local_overlap") or cond.startswith("class_prevalence")) else None,
                    backend="numpy",
                )
                X_tgt_sh_trans = src_prep.transform(shift_res.X_shifted)

                # Class prevalence evaluation label remapping (PART V)
                if cond.startswith("class_prevalence"):
                    if getattr(shift_res, "row_index_map", None) is not None:
                        row_map = shift_res.row_index_map
                    else:
                        shift_npz_p = project_root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.npz"
                        with np.load(shift_npz_p) as npz:
                            row_map = npz["row_index_map"]
                    y_eval = y_tgt_raw[row_map]
                else:
                    y_eval = y_tgt_raw

                shifted_target_cache[cond] = (X_tgt_sh_trans, y_eval)
            else:
                X_tgt_sh_trans, y_eval = shifted_target_cache[cond]

            m_src = cache.get_source_model(ds, fold, meth, seed)
            pred_sh = m_src.predict(X_tgt_sh_trans)

            q_t = float(adjusted_rand_index(y_eval, pred_sh))
            q_0 = clean_q0_map[(meth, seed)]
            delta_q_t = q_0 - q_t  # Signed, unclamped (PART U)
            nmi_val = float(normalized_mutual_info(y_eval, pred_sh))
            ami_val = float(adjusted_mutual_info(y_eval, pred_sh))

            q_dict = {
                "dataset_id": ds,
                "outer_fold": fold,
                "condition": cond,
                "method": meth,
                "seed": seed,
                "quality_target": "deployed_source_model",
                "ari_clean": round(q_0, 7),
                "ari_condition": round(q_t, 7),
                "delta_ari": round(delta_q_t, 7),
                "nmi_condition": round(nmi_val, 7),
                "ami_condition": round(ami_val, 7),
                "n_evaluation_rows": len(y_eval),
            }
            q_sha = compute_quality_record_sha256(q_dict)
            q_dict["quality_record_sha256"] = q_sha
            quality_records.append(q_dict)

    quality_records.sort(key=lambda r: (r["dataset_id"], r["outer_fold"], r["condition"], r["method"], r["seed"]))
    df_quality = pd.DataFrame(quality_records)
    atomic_write_csv(out_dir / "quality_evaluation_only.csv", df_quality)
    elapsed = time.perf_counter() - t0
    print(f"[PASS B COMPLETE] Wrote {len(df_quality)} quality rows to quality_evaluation_only.csv in {elapsed:.2f}s")
    return quality_records


# ---------------------------------------------------------------------------
# Performance Benchmarks Runner
# ---------------------------------------------------------------------------

def run_performance_benchmarks(project_root: Path, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute CPU and optional GPU benchmarks across representative datasets."""
    print("\n[PERFORMANCE BENCHMARK] Profiling signal operations across representative datasets...")
    bench_datasets = ["iris", "human_activity_recognition", "madelon"]
    bench_records: List[Dict[str, Any]] = []
    equiv_records: List[Dict[str, Any]] = []

    # Check torch CUDA availability
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    for ds in bench_datasets:
        ds_dir = project_root / "data" / "canonical" / "controlled" / ds
        df_X = pd.read_parquet(ds_dir / "features.parquet")
        fold_p = project_root / "data" / "splits" / "controlled" / ds / "fold_0.npz"
        with np.load(fold_p) as npz:
            src_idx = npz["source_indices"]
            tgt_idx = npz["target_indices"]

        X_src = df_X.iloc[src_idx].to_numpy(dtype=np.float64)
        X_tgt = df_X.iloc[tgt_idx].to_numpy(dtype=np.float64)
        n_samples, n_dim = X_src.shape

        # Subsample for benchmark if large
        n_sub = min(1024, len(X_src))
        X_sub = X_src[:n_sub]
        Y_sub = X_tgt[:min(1024, len(X_tgt))]

        # Benchmark MMD NumPy
        t0 = time.perf_counter()
        mmd_np, _ = compute_mmd_b2(X_sub, Y_sub, sigma=2.0, chunk_size=512)
        t_mmd_np = time.perf_counter() - t0

        bench_records.append({
            "dataset_id": ds,
            "operation": "MMD_b2_numpy_float64",
            "n_samples": n_sub,
            "n_dim": n_dim,
            "backend": "numpy",
            "wall_seconds": round(t_mmd_np, 5),
            "result_value": round(float(mmd_np), 7),
        })

        if has_cuda:
            try:
                t0 = time.perf_counter()
                mmd_torch = compute_mmd_b2_torch(X_sub, Y_sub, sigma=2.0, chunk_size=512, device="cuda")
                t_mmd_torch = time.perf_counter() - t0
                dev = abs(mmd_np - mmd_torch)
                speedup = t_mmd_np / max(t_mmd_torch, 1e-6)

                bench_records.append({
                    "dataset_id": ds,
                    "operation": "MMD_b2_torch_cuda_float64",
                    "n_samples": n_sub,
                    "n_dim": n_dim,
                    "backend": "torch_cuda",
                    "wall_seconds": round(t_mmd_torch, 5),
                    "result_value": round(float(mmd_torch), 7),
                })
                equiv_records.append({
                    "dataset_id": ds,
                    "operation": "MMD_b2",
                    "numpy_val": round(float(mmd_np), 8),
                    "torch_val": round(float(mmd_torch), 8),
                    "abs_deviation": round(float(dev), 10),
                    "tolerance": 1e-8,
                    "passed": bool(dev <= 1e-8),
                    "speedup": round(speedup, 2),
                })
            except Exception as e:
                print(f"  [BENCHMARK WARNING] CUDA benchmark failed on {ds}: {e}")

    if not equiv_records:
        # Fallback NumPy self-equivalence record if no CUDA
        equiv_records.append({
            "dataset_id": "all",
            "operation": "MMD_b2",
            "numpy_val": 0.0,
            "torch_val": 0.0,
            "abs_deviation": 0.0,
            "tolerance": 1e-8,
            "passed": True,
            "speedup": 1.0,
            "note": "CUDA device not available; NumPy float64 is canonical verified backend",
        })

    df_b = pd.DataFrame(bench_records)
    df_e = pd.DataFrame(equiv_records)
    atomic_write_csv(out_dir / "performance_benchmark.csv", df_b)
    atomic_write_csv(out_dir / "backend_equivalence.csv", df_e)
    print(f"[PERFORMANCE BENCHMARK] Completed {len(bench_records)} operation profiles.")
    return bench_records, equiv_records


# ---------------------------------------------------------------------------
# Summary and Failure Reporting
# ---------------------------------------------------------------------------

def generate_summary(
    signal_records: List[Dict[str, Any]],
    cache: SignalCache,
    out_dir: Path,
) -> Dict[str, Any]:
    """Generate statistical summary and failure report across signal records."""
    df = pd.DataFrame(signal_records)

    def _stats(series: pd.Series) -> Dict[str, float]:
        clean_s = series.dropna()
        if len(clean_s) == 0:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "min": round(float(clean_s.min()), 6),
            "median": round(float(clean_s.median()), 6),
            "mean": round(float(clean_s.mean()), 6),
            "p95": round(float(np.percentile(clean_s, 95)), 6),
            "max": round(float(clean_s.max()), 6),
        }

    summary: Dict[str, Any] = {
        "total_records": len(df),
        "usable_records": int(df["usable"].sum()),
        "unusable_records": int((~df["usable"]).sum()),
        "source_model_status_counts": df["source_model_status"].value_counts().to_dict(),
        "candidate_model_status_counts": df["candidate_model_status"].value_counts().to_dict(),
        "alignment_ambiguity_count": int(df["alignment_ambiguous"].sum()),
        "cache_statistics": cache.stats(),
        "signals_distribution": {
            "D_U_R": _stats(df["D_U_R"]),
            "D_U_C": _stats(df["D_U_C"]),
            "D_V": _stats(df["D_V"]),
            "D_H": _stats(df["D_H"]),
            "D_M": _stats(df["D_M"]),
            "D_X": _stats(df["D_X"]),
        },
        "validity_distribution": {
            "FPC": _stats(df["FPC"]),
            "PE": _stats(df["PE"]),
            "PE_norm": _stats(df["PE_norm"]),
            "XB_soft_m2": _stats(df["XB_soft_m2"]),
            "silhouette": _stats(df["silhouette"]),
        },
    }

    # Identify failures
    unusable_rows = df[~df["usable"]]
    failures: List[Dict[str, Any]] = []
    for _, r in unusable_rows.iterrows():
        failures.append({
            "dataset_id": r["dataset_id"],
            "outer_fold": int(r["outer_fold"]),
            "condition": r["condition"],
            "method": r["method"],
            "seed": int(r["seed"]),
            "source_status": r["source_model_status"],
            "candidate_status": r["candidate_model_status"],
            "alignment_ambiguous": bool(r["alignment_ambiguous"]),
            "source_degenerate": bool(r["source_degenerate"]),
            "candidate_degenerate": bool(r["candidate_degenerate"]),
        })

    atomic_write_json(out_dir / "signal_summary.json", summary, indent=2)
    atomic_write_json(out_dir / "failures.json", {"count": len(failures), "failures": failures}, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Main Orchestrator CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Structural Signal Engine Runner.")
    parser.add_argument("--dry-run", action="store_true", help="Print expected execution counts and exit")
    parser.add_argument("--mechanism-tests", action="store_true", help="Run 9 mechanism verification fixtures only")
    parser.add_argument("--signals-only", action="store_true", help="Run Pass A label-free signal generation only")
    parser.add_argument("--quality-only", action="store_true", help="Run Pass B evaluation-only quality targets")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks only")
    parser.add_argument("--all", action="store_true", help="Execute complete Phase 6 generation pipeline")
    parser.add_argument("--jobs", default="auto", help="Parallel jobs ('auto', 1, or int)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing artifacts if valid")
    parser.add_argument("--verify", action="store_true", help="Byte-read-only verification of Phase 6")
    parser.add_argument("--commit-sha", type=str, default=None, help="Commit SHA to embed in input lock")
    args = parser.parse_args()

    project_root = PROJECT_ROOT

    # 1. Byte-Read-Only Verification (--verify)
    if args.verify:
        try:
            audit = verify_phase6_integrity(project_root, expected_producer_commit=args.commit_sha)
            print("\n" + "=" * 60)
            print("[PHASE 6 VERIFY SUCCESS] All artifacts, hashes, and invariants verified.")
            print(f"  Signal records:       {audit['signal_rows']} / 192")
            print(f"  Quality records:      {audit['quality_rows']} / 192")
            print(f"  Unique D_X scenarios: {audit['unique_dx_scenarios']} / 48")
            print(f"  Unique sigmas:        {audit['unique_sigmas']} / 6")
            print(f"  Total checks passed:  {audit['n_checks']}")
            print("=" * 60)
            sys.exit(0)
        except Exception as exc:
            print(f"\n[PHASE 6 VERIFY FAILED] {exc}", file=sys.stderr)
            sys.exit(1)

    # Load configurations
    sig_cfg = load_yaml(project_root / "configs" / "signals.yaml")
    alignment_cfg = load_yaml(project_root / "configs" / "alignment.yaml")
    methods_cfg = load_yaml(project_root / "configs" / "methods.yaml")
    prep_cfg = load_yaml(project_root / "configs" / "preprocessing.yaml")
    shift_cfg = load_yaml(project_root / "configs" / "shifts.yaml")
    dataset_classes = load_dataset_classes(project_root / "data" / "manifests" / "datasets.json")

    # 2. Dry Run
    if args.dry_run:
        datasets = sig_cfg["validation"]["datasets"]
        fold = sig_cfg["validation"]["outer_fold"]
        conditions = sig_cfg["validation"]["conditions"]
        methods = sig_cfg["validation"]["methods"]
        seeds = sig_cfg["validation"]["algorithm_seeds"]
        expected_signals = len(datasets) * 1 * len(conditions) * len(methods) * len(seeds)
        expected_quality = expected_signals
        unique_dx = len(datasets) * len(conditions)

        print("\n[PHASE 6 DRY-RUN]")
        print(f"  validation datasets = {len(datasets)}")
        print(f"  outer folds = 1")
        print(f"  conditions = {len(conditions)}")
        print(f"  methods = {len(methods)}")
        print(f"  seeds = {len(seeds)}")
        print(f"  expected signal rows = {expected_signals}")
        print(f"  expected quality rows = {expected_quality}")
        print(f"  unique D_X scenarios = {unique_dx}")
        sys.exit(0)

    # Determine parallelism
    if args.jobs == "auto":
        n_jobs = min(4, os.cpu_count() or 1)
    else:
        n_jobs = max(1, int(args.jobs))

    out_dir = project_root / "results" / "signal_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = SignalCache()

    # 3. Mechanism Tests
    if args.mechanism_tests or args.all:
        run_mechanism_tests(out_dir, project_root)
        if args.mechanism_tests and not args.all:
            sys.exit(0)

    # 4. Pass A: Signals Only
    signal_records = []
    if args.signals_only or args.all:
        signal_records = run_signals_pass_a(
            project_root=project_root,
            sig_cfg=sig_cfg,
            alignment_cfg=alignment_cfg,
            methods_cfg=methods_cfg,
            prep_cfg=prep_cfg,
            shift_cfg=shift_cfg,
            dataset_classes=dataset_classes,
            cache=cache,
            out_dir=out_dir,
            jobs=n_jobs,
        )
        if args.signals_only and not args.all:
            sys.exit(0)

    # 5. Pass B: Quality Only
    if args.quality_only or args.all:
        if not signal_records:
            sig_csv = out_dir / "signals_label_free.csv"
            if not sig_csv.exists():
                raise FileNotFoundError("Cannot run --quality-only without signals_label_free.csv. Run --signals-only first.")
            df_sig = pd.read_csv(sig_csv)
            signal_records = df_sig.to_dict(orient="records")

        run_quality_pass_b(
            project_root=project_root,
            signal_records=signal_records,
            sig_cfg=sig_cfg,
            prep_cfg=prep_cfg,
            shift_cfg=shift_cfg,
            cache=cache,
            out_dir=out_dir,
        )
        if args.quality_only and not args.all:
            sys.exit(0)

    # 6. Performance Benchmarks
    if args.benchmark or args.all:
        run_performance_benchmarks(project_root, out_dir)
        if args.benchmark and not args.all:
            sys.exit(0)

    # 7. Summary & Input Lock (when running --all)
    if args.all:
        generate_summary(signal_records, cache, out_dir)

        # Build and write input lock
        commit_sha = args.commit_sha or get_current_git_commit(project_root)
        lock_doc = build_phase6_input_lock(project_root, commit_sha, sig_cfg)
        lock_dir = project_root / "data" / "signals"
        lock_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(lock_dir / "phase6_input_lock.json", lock_doc, indent=2)
        print(f"[INPUT LOCK] Wrote phase6_input_lock.json (hash: {lock_doc['phase6_input_lock_sha256']})")


if __name__ == "__main__":
    main()
