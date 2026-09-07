#!/usr/bin/env python3
"""Phase 7 Structural Signal Falsification Pilot Runner.

Pre-registered protocol execution:
1. --prepare-offline-replay: Generates 120 local-overlap replay descriptors (JSON+NPZ)
2. --signals-only: Generates 4,500 label-free FCM signal rows (labels strictly blocked)
3. --quality-only: Generates 4,500 evaluation-only quality degradation rows
4. --evaluate: Joins artifacts, runs LODO, LOSFO, Ridge, ablations, diagnostics, bootstrap, verdict
5. --all: Executes steps 1 to 4 sequentially
6. --verify: Byte-read-only integrity and verification audit
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
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
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from clusterdrift.alignment.costs import compute_reference_cluster_scales
from clusterdrift.alignment.hungarian import align_clusters
from clusterdrift.data.preprocess import build_preprocessor
from clusterdrift.falsification.bootstrap import (
    compute_paired_unit_bootstrap,
    evaluate_falsification_verdict,
)
from clusterdrift.falsification.dataset import (
    get_dataset_dimensions,
    join_signals_and_quality,
)
from clusterdrift.falsification.evaluation import (
    run_lodo_evaluation,
    run_losfo_evaluation,
)
from clusterdrift.falsification.protocol import (
    ALL_SHIFT_FAMILIES,
    CONDITION_TO_FAMILY,
    compute_falsification_protocol_sha256,
    load_falsification_config,
)
from clusterdrift.falsification.verification import verify_falsification
from clusterdrift.probes.bank import (
    load_current_probe_descriptor,
    load_reference_probe_descriptor,
)
from clusterdrift.probes.matrix import (
    load_current_probe_matrix,
    load_raw_source_features,
    load_reference_probe_matrix,
)
from clusterdrift.shifts.hashing import (
    atomic_write_csv,
    atomic_write_json,
    compute_file_sha256,
)
from clusterdrift.shifts.replay import (
    build_local_overlap_replay_descriptor,
    replay_frozen_shift,
)
from clusterdrift.signals.engine import SignalCache, SignalEngine, fit_clustering_model
from clusterdrift.signals.hashing import (
    compute_quality_record_sha256,
    compute_signal_protocol_sha256,
)
from clusterdrift.signals.result import QualityResult, SignalResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
# Offline Replay Generation (120 Scenarios)
# ---------------------------------------------------------------------------

def prepare_offline_replay(cfg: Dict[str, Any], project_root: Path) -> List[Path]:
    """Materialize 120 local-overlap replay descriptors (JSON+NPZ)."""
    t0 = time.perf_counter()
    print("[OFFLINE REPLAY] Generating 120 local-overlap replay pairs...")
    commit_sha = get_current_git_commit(project_root)
    out_base = project_root / "data" / "falsification" / "offline_shift_replay"
    out_base.mkdir(parents=True, exist_ok=True)

    datasets = cfg["datasets"]
    folds = cfg["outer_folds"]
    local_overlap_conditions = ["local_overlap_mild", "local_overlap_severe"]

    created_paths: List[Path] = []

    for ds in datasets:
        ds_dir = project_root / "data" / "canonical" / "controlled" / ds
        df_X = pd.read_parquet(ds_dir / "features.parquet")
        df_y = pd.read_parquet(ds_dir / "labels.parquet")

        with open(ds_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        roles = meta.get("feature_roles", {c: "numeric" for c in df_X.columns})

        for fold in folds:
            fold_p = project_root / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
            with np.load(fold_p) as npz:
                src_idx = npz["source_indices"]
                tgt_idx = npz["target_indices"]

            X_src_raw = df_X.iloc[src_idx].copy().reset_index(drop=True)
            X_tgt_raw = df_X.iloc[tgt_idx].copy().reset_index(drop=True)
            y_src_raw = df_y.iloc[src_idx].to_numpy().ravel()
            y_tgt_raw = df_y.iloc[tgt_idx].to_numpy().ravel()

            for cond in local_overlap_conditions:
                spec_path = project_root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"
                if not spec_path.exists():
                    raise FileNotFoundError(f"Spec not found: {spec_path}")

                desc = build_local_overlap_replay_descriptor(
                    X_target_raw=X_tgt_raw,
                    y_target_raw=y_tgt_raw,
                    X_source_raw=X_src_raw,
                    y_source_raw=y_src_raw,
                    roles=roles,
                    dataset_id=ds,
                    outer_fold=fold,
                    condition=cond,
                    phase4_spec_path=spec_path,
                    project_root=project_root,
                    output_root=out_base,
                    commit_sha=commit_sha,
                )
                out_dir = out_base / ds / f"fold_{fold}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{cond}.json"
                atomic_write_json(out_path, desc, indent=2)
                created_paths.append(out_path)

    elapsed = time.perf_counter() - t0
    print(f"[OFFLINE REPLAY COMPLETE] Created {len(created_paths)} replay descriptors in {elapsed:.2f}s")
    return created_paths


# ---------------------------------------------------------------------------
# Pass A: Label-Free Signal Generation (4,500 Rows)
# ---------------------------------------------------------------------------

def process_single_task_signals(
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
    cache: SignalCache,
    project_root: Path,
) -> List[SignalResult]:
    """Execute Pass A label-free signal computation for all scenarios of a dataset fold."""
    signal_engine = SignalEngine(
        signals_cfg=sig_cfg,
        alignment_cfg=alignment_cfg,
        methods_cfg=methods_cfg,
        cache=cache,
    )
    probes_dir = project_root / "data" / "probes"
    prep_config_sha = compute_file_sha256(project_root / "configs" / "preprocessing.yaml")
    replay_root = project_root / "data" / "falsification" / "offline_shift_replay"

    # Load raw source features (NO labels)
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
    ref_desc, ref_positions, _ = load_reference_probe_descriptor(ref_desc_path)
    A_R = cache.get_transformed_bank(ref_desc.probe_bank_sha256)
    if A_R is None:
        A_R = load_reference_probe_matrix(ds, fold, ref_positions, src_prep, project_root)
        cache.put_transformed_bank(ref_desc.probe_bank_sha256, A_R)

    # Load raw target features (NO labels)
    ds_dir = project_root / "data" / "canonical" / "controlled" / ds
    df_X = pd.read_parquet(ds_dir / "features.parquet")
    fold_p = project_root / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
    with np.load(fold_p) as npz:
        X_tgt_raw = df_X.iloc[npz["target_indices"]].copy().reset_index(drop=True)

    results: List[SignalResult] = []

    for cond in conditions:
        shift_spec_path = project_root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"
        if not shift_spec_path.exists():
            raise FileNotFoundError(f"Shift spec not found: {shift_spec_path}")
        with open(shift_spec_path, "r", encoding="utf-8") as f:
            s_doc = json.load(f)
        shift_spec_sha = s_doc["shift_spec_sha256"]
        shift_spec_file_sha = compute_file_sha256(shift_spec_path)

        # Replay shift using falsification offline replay root
        shift_res = replay_frozen_shift(
            X_target_raw=X_tgt_raw,
            dataset_id=ds,
            outer_fold=fold,
            condition=cond,
            phase4_spec_path=shift_spec_path,
            X_source_raw=X_src_raw,
            roles=roles,
            project_root=project_root,
            output_root=replay_root,
        )
        shift_replay_sha = shift_res.metadata.get("replay_descriptor_sha256")
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
                    reference_probe_bank_sha256=ref_desc.probe_bank_sha256,
                    current_probe_bank_sha256=cur_desc.probe_bank_sha256,
                    shift_spec_sha256=shift_spec_sha,
                    shift_spec_file_sha256=shift_spec_file_sha,
                    shift_replay_sha256=shift_replay_sha,
                )
                results.append(res)

    return results


def run_label_free_signals_pass(cfg: Dict[str, Any], project_root: Path, max_workers: int = 4) -> pd.DataFrame:
    """Execute Pass A label-free signal computation with strict label blocking."""
    print("[PASS A] Starting label-free signal computation (labels strictly blocked)...")
    t0 = time.perf_counter()

    # Monkeypatch pd.read_parquet to enforce label blocking
    orig_read_parquet = pd.read_parquet

    def blocked_read_parquet(path, *args, **kwargs):
        p_str = str(path).lower()
        if "labels.parquet" in p_str:
            raise PermissionError(f"CRITICAL: Access to labels is strictly forbidden in Pass A: {path}")
        return orig_read_parquet(path, *args, **kwargs)

    pd.read_parquet = blocked_read_parquet

    try:
        # Load configs
        sig_cfg = yaml.safe_load(open(project_root / "configs" / "signals.yaml", "r", encoding="utf-8"))
        alignment_cfg = yaml.safe_load(open(project_root / "configs" / "alignment.yaml", "r", encoding="utf-8"))
        methods_cfg = yaml.safe_load(open(project_root / "configs" / "methods.yaml", "r", encoding="utf-8"))
        prep_cfg = yaml.safe_load(open(project_root / "configs" / "preprocessing.yaml", "r", encoding="utf-8"))

        manifest_path = project_root / "data" / "manifests" / "dataset_manifest.json"
        classes_map = load_dataset_classes(manifest_path)

        cache = SignalCache()
        datasets = cfg["datasets"]
        folds = cfg["outer_folds"]
        conditions = cfg["conditions"]
        methods = cfg["methods"]
        seeds = cfg["algorithm_seeds"]

        tasks = [(ds, fold) for ds in datasets for fold in folds]
        print(f"[PASS A] Dispatching {len(tasks)} dataset-fold tasks across {max_workers} workers...")

        all_results: List[SignalResult] = []

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_single_task_signals,
                        ds,
                        fold,
                        classes_map[ds],
                        conditions,
                        methods,
                        seeds,
                        sig_cfg,
                        alignment_cfg,
                        methods_cfg,
                        prep_cfg,
                        cache,
                        project_root,
                    )
                    for ds, fold in tasks
                ]
                for fut in futures:
                    all_results.extend(fut.result())
        else:
            for ds, fold in tasks:
                res = process_single_task_signals(
                    ds,
                    fold,
                    classes_map[ds],
                    conditions,
                    methods,
                    seeds,
                    sig_cfg,
                    alignment_cfg,
                    methods_cfg,
                    prep_cfg,
                    cache,
                    project_root,
                )
                all_results.extend(res)

        df = pd.DataFrame([r.to_dict() for r in all_results])

        # Canonical sort
        df.sort_values(
            by=["dataset_id", "outer_fold", "condition", "method", "seed"],
            inplace=True,
        )

        out_path = project_root / "results" / "falsification" / "signals_label_free.csv"
        atomic_write_csv(out_path, df)

        elapsed = time.perf_counter() - t0
        print(f"[PASS A COMPLETE] Generated {len(df)} signal rows in {elapsed:.2f}s -> {out_path.relative_to(project_root)}")

        # Save cache execution summary
        summary = {
            "elapsed_seconds": elapsed,
            "total_signal_rows": len(df),
            "cache_metrics": cache.stats(),
        }
        atomic_write_json(project_root / "results" / "falsification" / "execution_summary.json", summary, indent=2)

        return df

    finally:
        # Restore pd.read_parquet
        pd.read_parquet = orig_read_parquet


# ---------------------------------------------------------------------------
# Pass B: Evaluation-Only Quality Targets (4,500 Rows)
# ---------------------------------------------------------------------------

def run_evaluation_quality_pass(cfg: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Execute Pass B quality computation on full shifted test targets (labels allowed)."""
    print("[PASS B] Starting evaluation-only quality computation (labels allowed)...")
    t0 = time.perf_counter()

    methods_cfg = yaml.safe_load(open(project_root / "configs" / "methods.yaml", "r", encoding="utf-8"))
    prep_cfg = yaml.safe_load(open(project_root / "configs" / "preprocessing.yaml", "r", encoding="utf-8"))
    manifest_path = project_root / "data" / "manifests" / "dataset_manifest.json"
    classes_map = load_dataset_classes(manifest_path)
    prep_config_sha = compute_file_sha256(project_root / "configs" / "preprocessing.yaml")
    replay_root = project_root / "data" / "falsification" / "offline_shift_replay"

    cache = SignalCache()
    datasets = cfg["datasets"]
    folds = cfg["outer_folds"]
    conditions = cfg["conditions"]
    methods = cfg["methods"]
    seeds = cfg["algorithm_seeds"]

    results: List[QualityResult] = []

    for ds in datasets:
        K = classes_map[ds]
        ds_dir = project_root / "data" / "canonical" / "controlled" / ds
        df_X = pd.read_parquet(ds_dir / "features.parquet")
        df_y = pd.read_parquet(ds_dir / "labels.parquet")

        with open(ds_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        roles = meta.get("feature_roles", {col: "numeric" for col in df_X.columns})

        for fold in folds:
            fold_p = project_root / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
            with np.load(fold_p) as npz:
                src_idx = npz["source_indices"]
                tgt_idx = npz["target_indices"]

            X_src_raw = df_X.iloc[src_idx].copy().reset_index(drop=True)
            X_tgt_raw = df_X.iloc[tgt_idx].copy().reset_index(drop=True)
            y_src_raw = df_y.iloc[src_idx].to_numpy().ravel()
            y_tgt_raw = df_y.iloc[tgt_idx].to_numpy().ravel()

            src_prep = build_preprocessor(feature_roles=roles, config=prep_cfg, metadata=meta)
            src_prep.fit(X_src_raw)
            X_src_trans = src_prep.transform(X_src_raw)

            # Fit source models once per seed
            src_models: Dict[int, Any] = {}
            ari_cleans: Dict[int, float] = {}

            for seed in seeds:
                src_model = fit_clustering_model(
                    method=methods[0],
                    X=X_src_trans,
                    K=K,
                    seed=seed,
                    methods_cfg=methods_cfg,
                )
                src_models[seed] = src_model

                # Clean test prediction
                X_tgt_clean_trans = src_prep.transform(X_tgt_raw)
                U_clean = src_model.predict_proba(X_tgt_clean_trans)
                y_pred_clean = np.argmax(U_clean, axis=1)
                ari_cleans[seed] = float(adjusted_rand_score(y_tgt_raw, y_pred_clean))

            for cond in conditions:
                shift_spec_path = project_root / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"
                shift_res = replay_frozen_shift(
                    X_target_raw=X_tgt_raw,
                    dataset_id=ds,
                    outer_fold=fold,
                    condition=cond,
                    phase4_spec_path=shift_spec_path,
                    X_source_raw=X_src_raw,
                    roles=roles,
                    project_root=project_root,
                    output_root=replay_root,
                )
                X_tgt_shifted_trans = src_prep.transform(shift_res.X_shifted)

                # Ground truth shifted labels
                if cond.startswith("class_prevalence"):
                    y_tgt_shifted = y_tgt_raw[shift_res.row_index_map]
                else:
                    y_tgt_shifted = y_tgt_raw.copy()

                for meth in methods:
                    for seed in seeds:
                        src_model = src_models[seed]
                        U_cond = src_model.predict_proba(X_tgt_shifted_trans)
                        y_pred_cond = np.argmax(U_cond, axis=1)

                        ari_clean = ari_cleans[seed]
                        ari_cond = float(adjusted_rand_score(y_tgt_shifted, y_pred_cond))
                        delta_ari = ari_clean - ari_cond

                        nmi_cond = float(normalized_mutual_info_score(y_tgt_shifted, y_pred_cond, average_method="arithmetic"))
                        ami_cond = float(adjusted_mutual_info_score(y_tgt_shifted, y_pred_cond, average_method="arithmetic"))

                        rec_for_hash = {
                            "dataset_id": ds,
                            "outer_fold": fold,
                            "condition": cond,
                            "method": meth,
                            "seed": seed,
                            "quality_target": "delta_ari",
                            "ari_clean": ari_clean,
                            "ari_condition": ari_cond,
                            "delta_ari": delta_ari,
                            "nmi_condition": nmi_cond,
                            "ami_condition": ami_cond,
                            "n_evaluation_rows": len(y_tgt_shifted),
                        }
                        q_sha = compute_quality_record_sha256(rec_for_hash)

                        q_res = QualityResult(
                            dataset_id=ds,
                            outer_fold=fold,
                            condition=cond,
                            method=meth,
                            seed=seed,
                            quality_target="delta_ari",
                            ari_clean=ari_clean,
                            ari_condition=ari_cond,
                            delta_ari=delta_ari,
                            nmi_condition=nmi_cond,
                            ami_condition=ami_cond,
                            n_evaluation_rows=len(y_tgt_shifted),
                            quality_record_sha256=q_sha,
                        )
                        results.append(q_res)

    df = pd.DataFrame([r.to_dict() for r in results])
    df.sort_values(
        by=["dataset_id", "outer_fold", "condition", "method", "seed"],
        inplace=True,
    )

    out_path = project_root / "results" / "falsification" / "quality_evaluation_only.csv"
    atomic_write_csv(out_path, df)

    elapsed = time.perf_counter() - t0
    print(f"[PASS B COMPLETE] Generated {len(df)} quality rows in {elapsed:.2f}s -> {out_path.relative_to(project_root)}")
    return df


# ---------------------------------------------------------------------------
# Evaluation & Falsification Analysis
# ---------------------------------------------------------------------------

def run_evaluation(cfg: Dict[str, Any], project_root: Path) -> None:
    """Join signals and quality, run LODO, LOSFO, Ridge, ablations, bootstrap, verdict."""
    print("[EVALUATION] Joining signals and quality...")
    t0 = time.perf_counter()

    res_dir = project_root / "results" / "falsification"
    sig_p = res_dir / "signals_label_free.csv"
    qual_p = res_dir / "quality_evaluation_only.csv"

    if not sig_p.exists() or not qual_p.exists():
        raise FileNotFoundError("Signals and quality CSVs must exist before running evaluation")

    sig_df = pd.read_csv(sig_p)
    qual_df = pd.read_csv(qual_p)

    dims = get_dataset_dimensions(project_root, cfg["datasets"])
    joined_df = join_signals_and_quality(sig_df, qual_df, dims)

    joined_p = res_dir / "joined_evaluation_table.csv"
    atomic_write_csv(joined_p, joined_df)
    print(f"[EVALUATION] Joined evaluation table saved -> {joined_p.relative_to(project_root)}")

    # 1. Run LODO Evaluation
    print("[EVALUATION] Executing Leave-One-Dataset-Out (LODO) nested evaluation...")
    lodo_res = run_lodo_evaluation(joined_df, cfg)

    atomic_write_csv(res_dir / "lodo_predictions.csv", lodo_res["predictions_df"])
    atomic_write_csv(res_dir / "lodo_metrics.csv", lodo_res["lodo_metrics_df"])
    atomic_write_csv(res_dir / "lodo_dataset_metrics.csv", lodo_res["lodo_dataset_metrics_df"])
    atomic_write_csv(res_dir / "paired_dataset_deltas.csv", lodo_res["paired_dataset_deltas_df"])
    atomic_write_csv(res_dir / "ridge_control_metrics.csv", lodo_res["ridge_control_metrics_df"])
    atomic_write_csv(res_dir / "structural_ablation.csv", lodo_res["structural_ablation_df"])
    atomic_write_csv(res_dir / "single_signal_incremental.csv", lodo_res["single_signal_incremental_df"])
    atomic_write_csv(res_dir / "dimension_strata_audit.csv", lodo_res["dimension_strata_audit_df"])

    # 2. Run LOSFO Evaluation
    print("[EVALUATION] Executing Leave-One-Shift-Family-Out (LOSFO) nested evaluation...")
    losfo_res = run_losfo_evaluation(joined_df, cfg)

    atomic_write_csv(res_dir / "losfo_predictions.csv", losfo_res["predictions_df"])
    atomic_write_csv(res_dir / "losfo_metrics.csv", losfo_res["losfo_metrics_df"])
    atomic_write_csv(res_dir / "losfo_family_metrics.csv", losfo_res["losfo_family_metrics_df"])
    atomic_write_csv(res_dir / "paired_family_deltas.csv", losfo_res["paired_family_deltas_df"])

    # Combine hyperparameter selections
    all_hypers = pd.concat([lodo_res["hyperparameters_df"], losfo_res["hyperparameters_df"]], ignore_index=True)
    atomic_write_csv(res_dir / "hyperparameter_selections.csv", all_hypers)

    # 3. Paired Bootstrap on Independent Units
    print("[EVALUATION] Executing paired bootstrap on 12 dataset units and 7 family units...")
    boot_seed = cfg["bootstrap"]["seed"]
    boot_reps = cfg["bootstrap"]["repetitions"]

    paired_ds = lodo_res["paired_dataset_deltas_df"]
    paired_fam = losfo_res["paired_family_deltas_df"]

    d_04 = paired_ds["delta_04"].values
    d_34 = paired_ds["delta_34"].values
    f_04 = paired_fam["delta_04"].values
    f_34 = paired_fam["delta_34"].values

    boot_res = {
        "lodo_delta_04": compute_paired_unit_bootstrap(d_04, n_repetitions=boot_reps, seed=boot_seed),
        "lodo_delta_34": compute_paired_unit_bootstrap(d_34, n_repetitions=boot_reps, seed=boot_seed),
        "losfo_delta_04": compute_paired_unit_bootstrap(f_04, n_repetitions=boot_reps, seed=boot_seed),
        "losfo_delta_34": compute_paired_unit_bootstrap(f_34, n_repetitions=boot_reps, seed=boot_seed),
    }

    boot_rows = []
    for comp_name, b in boot_res.items():
        r = dict(b)
        r["comparison"] = comp_name
        boot_rows.append(r)
    atomic_write_csv(res_dir / "bootstrap_intervals.csv", pd.DataFrame(boot_rows))

    # 4. Mechanical Verdict
    print("[EVALUATION] Mechanically evaluating pre-registered falsification rules...")
    hgbr_lodo = lodo_res["lodo_metrics_df"][lodo_res["lodo_metrics_df"]["regressor"] == "hist_gradient_boosting"]
    p0_mae = float(hgbr_lodo[hgbr_lodo["feature_block"] == "P0"]["mae"].values[0])
    p3_mae = float(hgbr_lodo[hgbr_lodo["feature_block"] == "P3"]["mae"].values[0])
    p4_mae = float(hgbr_lodo[hgbr_lodo["feature_block"] == "P4"]["mae"].values[0])

    verdict_dict = evaluate_falsification_verdict(
        p0_mae,
        p3_mae,
        p4_mae,
        d_04,
        d_34,
        f_04,
        f_34,
        boot_res,
    )
    atomic_write_json(res_dir / "verdict.json", verdict_dict, indent=2)
    atomic_write_json(res_dir / "failures.json", {}, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"[EVALUATION COMPLETE] Finished all evaluations in {elapsed:.2f}s")
    print(f"============================================================")
    print(f"VERDICT: {verdict_dict['verdict']}")
    print(f"P0 MAE: {p0_mae:.5f} | P3 MAE: {p3_mae:.5f} | P4 MAE: {p4_mae:.5f}")
    print(f"R_04: {verdict_dict['metrics']['r_04']:.4f} | R_34: {verdict_dict['metrics']['r_34']:.4f}")
    print(f"Dataset wins: {verdict_dict['metrics']['dataset_wins_04']} | Family wins: {verdict_dict['metrics']['family_wins_04']}")
    print(f"============================================================")


# ---------------------------------------------------------------------------
# Phase 7 Input Lock Creation
# ---------------------------------------------------------------------------

def create_phase7_input_lock(project_root: Path, producer_commit: str) -> Dict[str, Any]:
    """Cryptographically bind all Phase 1-6 inputs into phase7_input_lock.json."""
    lock_data = {
        "lock_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase6_freeze_commit": "a2726e6423405c93d9319eebbe0cbdfdfc7ecbb4",
        "phase7_producer_commit": producer_commit,
        "phase1_dataset_manifest_sha256": compute_file_sha256(project_root / "data" / "manifests" / "dataset_manifest.json"),
        "phase2_split_manifest_sha256": compute_file_sha256(project_root / "data" / "manifests" / "split_manifest.json"),
        "phase4_shift_manifest_sha256": compute_file_sha256(project_root / "data" / "manifests" / "shift_manifest.json"),
        "phase5_probe_manifest_sha256": compute_file_sha256(project_root / "data" / "manifests" / "probe_manifest.json"),
        "methods_config_sha256": compute_file_sha256(project_root / "configs" / "methods.yaml"),
        "signals_config_sha256": compute_file_sha256(project_root / "configs" / "signals.yaml"),
        "falsification_config_sha256": compute_file_sha256(project_root / "configs" / "falsification.yaml"),
    }
    out_p = project_root / "data" / "falsification" / "phase7_input_lock.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_p, lock_data, indent=2)
    return lock_data


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Structural Signal Falsification Pilot")
    parser.add_argument("--prepare-offline-replay", action="store_true", help="Materialize 120 local-overlap replay descriptors")
    parser.add_argument("--signals-only", action="store_true", help="Generate 4,500 label-free signal rows (labels blocked)")
    parser.add_argument("--quality-only", action="store_true", help="Generate 4,500 evaluation-only quality degradation rows")
    parser.add_argument("--evaluate", action="store_true", help="Join tables and run LODO/LOSFO/ablation/bootstrap evaluation")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: replay, signals, quality, evaluate")
    parser.add_argument("--verify", action="store_true", help="Run strict byte-read-only verification")
    parser.add_argument("--workers", type=int, default=4, help="Max workers for parallel signal generation")
    args = parser.parse_args()

    cfg = load_falsification_config(PROJECT_ROOT / "configs" / "falsification.yaml")

    if args.verify:
        res = verify_falsification(PROJECT_ROOT)
        print(f"[VERIFY SUCCESS] All {res['total_checks_verified']} checks passed! Verdict: {res['verdict']}")
        return

    if args.prepare_offline_replay or args.all:
        prepare_offline_replay(cfg, PROJECT_ROOT)

    if args.signals_only or args.all:
        run_label_free_signals_pass(cfg, PROJECT_ROOT, max_workers=args.workers)

    if args.quality_only or args.all:
        run_evaluation_quality_pass(cfg, PROJECT_ROOT)

    if args.evaluate or args.all:
        run_evaluation(cfg, PROJECT_ROOT)


if __name__ == "__main__":
    main()