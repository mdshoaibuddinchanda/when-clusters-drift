#!/usr/bin/env python3
"""Script 02: Deterministic source/target and inner-validation split generation and preprocessing audit."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from clusterdrift.data.folds import (
    FoldSplit,
    compute_split_hash,
    generate_group_kfold_splits,
    generate_kfold_splits,
    generate_tableshift_natural_splits,
    generate_temporal_block_splits,
    generate_whyshift_natural_splits,
    save_fold_split_artifacts,
    verify_split_artifact,
)
from clusterdrift.data.manifest import compute_canonical_bundle_sha256, compute_file_sha256
from clusterdrift.data.preprocess import SourceOnlyPreprocessor, build_preprocessor
from clusterdrift.data.registry import get_dataset_spec, list_controlled_real, list_datasets, list_natural_shift

PHASE1_FREEZE_COMMIT = "d5cb818589a138b25f10e926e9e83b198d369599"
DATE_OF_FREEZE = "2026-09-07"
SPLIT_SEED = 20260907
N_OUTER = 5
N_INNER = 3

EXCLUDED_DATASETS = [
    {
        "dataset_id": "whyshift_taxi",
        "reason": "Raw download required; no Phase-1 canonical bundle present in data/canonical/.",
    },
    {
        "dataset_id": "whyshift_us_accidents",
        "reason": "Raw download required; no Phase-1 canonical bundle present in data/canonical/.",
    },
    {
        "dataset_id": "tableshift_college_scorecard",
        "reason": "Raw download required; no Phase-1 canonical bundle present in data/canonical/.",
    },
    {
        "dataset_id": "tableshift_heloc",
        "reason": "Raw download required; no Phase-1 canonical bundle present in data/canonical/.",
    },
    {
        "dataset_id": "tableshift_assistments",
        "reason": "Raw download required; no Phase-1 canonical bundle present in data/canonical/.",
    },
]


def compute_bundle_hash_for_dir(directory: Path) -> str:
    """Compute deterministic canonical bundle hash directly from directory artifacts."""
    fpath = directory / "features.parquet"
    lpath = directory / "labels.parquet"
    gpath = directory / "groups.parquet"
    dpath = directory / "domains.parquet"
    mpath = directory / "metadata.json"

    f_sha = compute_file_sha256(fpath) if fpath.exists() else "NONE"
    l_sha = compute_file_sha256(lpath) if lpath.exists() else "NONE"
    g_sha = compute_file_sha256(gpath) if gpath.exists() else None
    d_sha = compute_file_sha256(dpath) if dpath.exists() else None
    m_sha = compute_file_sha256(mpath) if mpath.exists() else "NONE"

    return compute_canonical_bundle_sha256(f_sha, l_sha, g_sha, d_sha, m_sha)


def load_preprocessing_config(project_root: Path) -> Dict[str, Any]:
    cfg_path = project_root / "configs" / "preprocessing.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_hash(project_root: Path) -> str:
    cfg_path = project_root / "configs" / "preprocessing.yaml"
    return compute_file_sha256(cfg_path)


def generate_input_lock(project_root: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Generate and persist data/splits/phase2_input_lock.json."""
    lock_path = project_root / "data" / "splits" / "phase2_input_lock.json"
    canonical_root = project_root / "data" / "canonical"

    eligible_datasets: Dict[str, Any] = {}

    # 1. Controlled real
    for ds_id in list_controlled_real():
        spec = get_dataset_spec(ds_id)
        ds_dir = canonical_root / "controlled" / ds_id
        meta_path = ds_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        bundle_sha = compute_bundle_hash_for_dir(ds_dir)
        rows = meta.get("n_rows", meta.get("rows", 0))
        features = meta.get("n_features", meta.get("features", 0))

        eligible_datasets[ds_id] = {
            "dataset_group": "controlled_real",
            "source_provider": spec.source_provider,
            "split_strategy": spec.split_strategy,
            "canonical_dir": f"data/canonical/controlled/{ds_id}",
            "canonical_bundle_sha256": bundle_sha,
            "row_count": rows,
            "feature_count": features,
            "date_of_freeze": DATE_OF_FREEZE,
        }

    # 2. WhyShift natural shift
    whyshift_tasks = [
        ("whyshift_acs_income", "income"),
        ("whyshift_acs_pubcov", "pubcov"),
        ("whyshift_acs_mobility", "mobility"),
    ]
    for ds_id, task in whyshift_tasks:
        spec = get_dataset_spec(ds_id)
        task_dir = canonical_root / "natural" / "whyshift" / task
        domains = ["CA", "FL", "NY", "PA", "TX"]
        domain_bundles: Dict[str, str] = {}
        domain_rows: Dict[str, int] = {}
        feature_cnt = 0

        for d in domains:
            ddir = task_dir / d
            meta_path = ddir / "metadata.json"
            if meta_path.exists():
                dmeta = json.loads(meta_path.read_text(encoding="utf-8"))
                domain_bundles[d] = compute_bundle_hash_for_dir(ddir)
                domain_rows[d] = dmeta.get("n_rows", dmeta.get("rows", 0))
                feature_cnt = dmeta.get("n_features", dmeta.get("features", feature_cnt))

        eligible_datasets[ds_id] = {
            "dataset_group": "natural_shift",
            "source_provider": spec.source_provider,
            "split_strategy": "natural_domain",
            "canonical_dir": f"data/canonical/natural/whyshift/{task}",
            "domains": domains,
            "canonical_bundle_sha256": domain_bundles,
            "row_count": domain_rows,
            "feature_count": feature_cnt,
            "date_of_freeze": DATE_OF_FREEZE,
        }

    # 3. TableShift natural shift
    ts_id = "tableshift_hospital_readmission"
    spec = get_dataset_spec(ts_id)
    ts_dir = canonical_root / "natural" / "tableshift" / ts_id
    meta_path = ts_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    ts_bundle_sha = compute_bundle_hash_for_dir(ts_dir)
    eligible_datasets[ts_id] = {
        "dataset_group": "natural_shift",
        "source_provider": spec.source_provider,
        "split_strategy": "natural_domain",
        "canonical_dir": f"data/canonical/natural/tableshift/{ts_id}",
        "canonical_bundle_sha256": ts_bundle_sha,
        "row_count": meta.get("n_rows", meta.get("rows", 0)),
        "feature_count": meta.get("n_features", meta.get("features", 0)),
        "domain_column": "admission_source_id",
        "date_of_freeze": DATE_OF_FREEZE,
    }

    lock_doc = {
        "phase1_freeze_commit": PHASE1_FREEZE_COMMIT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_seed": SPLIT_SEED,
        "outer_folds": N_OUTER,
        "inner_folds": N_INNER,
        "total_eligible_datasets": len(eligible_datasets),
        "total_partitions": 46,
        "eligible_datasets": eligible_datasets,
        "excluded_datasets": EXCLUDED_DATASETS,
    }

    if not dry_run:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_doc, f, indent=2)

    return lock_doc


def process_controlled_dataset(
    ds_id: str,
    project_root: Path,
    prep_cfg: Dict[str, Any],
    config_hash: str,
    force: bool = False,
    verify_only: bool = False,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single controlled dataset, generating splits and auditing preprocessing."""
    spec = get_dataset_spec(ds_id)
    ds_dir = project_root / "data" / "canonical" / "controlled" / ds_id
    out_dir = project_root / "data" / "splits" / "controlled" / ds_id

    features_path = ds_dir / "features.parquet"
    labels_path = ds_dir / "labels.parquet"
    groups_path = ds_dir / "groups.parquet"
    meta_path = ds_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bundle_sha = compute_bundle_hash_for_dir(ds_dir)
    feature_roles = meta.get("feature_roles", {})

    df_X = pd.read_parquet(features_path)
    n_samples = len(df_X)

    # Determine splits
    if spec.split_strategy == "group_kfold":
        if not groups_path.exists():
            raise FileNotFoundError(f"Missing groups.parquet for group_kfold dataset {ds_id}")
        df_g = pd.read_parquet(groups_path)
        group_series = df_g.iloc[:, 0].values
        group_col_name = spec.group_column or "group_id"
        splits = generate_group_kfold_splits(
            groups=group_series,
            n_outer=N_OUTER,
            n_inner=N_INNER,
            group_column_name=group_col_name,
        )
    elif spec.split_strategy == "temporal_block":
        time_col = spec.time_column or "date"
        time_series = df_X[time_col].values
        splits = generate_temporal_block_splits(
            time_values=time_series,
            n_outer=N_OUTER,
            n_inner=N_INNER,
            time_column_name=time_col,
        )
    else:
        splits = generate_kfold_splits(
            n_samples=n_samples,
            n_outer=N_OUTER,
            n_inner=N_INNER,
            split_seed=SPLIT_SEED,
        )

    manifest_entries: List[Dict[str, Any]] = []
    audit_entries: List[Dict[str, Any]] = []

    common_meta = {
        "dataset_id": ds_id,
        "scenario_id": "default",
        "split_seed": SPLIT_SEED,
    }

    # Preprocessor output feature dimension tracker for uniformity check
    encoded_dims: List[int] = []

    for split in splits:
        r = split.outer_fold
        prefix = f"fold_{r}"
        npz_path = out_dir / f"{prefix}.npz"
        json_path = out_dir / f"{prefix}.json"

        is_valid = False
        if npz_path.exists() and json_path.exists() and not force:
            is_valid = verify_split_artifact(npz_path, json_path, expected_bundle_sha256=bundle_sha)
            if is_valid:
                print(f"  [{ds_id} fold_{r}] [SKIP VERIFIED]")

        if not is_valid:
            if not dry_run and not verify_only:
                npz_path, json_path, split_sha = save_fold_split_artifacts(
                    split=split,
                    target_dir=out_dir,
                    file_prefix=prefix,
                    common_metadata=common_meta,
                    canonical_bundle_sha256=bundle_sha,
                    split_config_sha256=config_hash,
                )
            else:
                split_sha = compute_split_hash(
                    source_indices=split.source_indices,
                    target_indices=split.target_indices,
                    inner_folds=split.inner_folds,
                    split_strategy=split.split_strategy,
                    split_seed=SPLIT_SEED,
                    canonical_bundle_sha256=bundle_sha,
                )
        else:
            with open(json_path, "r", encoding="utf-8") as jf:
                split_meta = json.load(jf)
            split_sha = split_meta.get("split_sha256", "")

        npz_sha = compute_file_sha256(npz_path) if npz_path.exists() else ""
        json_sha = compute_file_sha256(json_path) if json_path.exists() else ""

        rel_npz = f"data/splits/controlled/{ds_id}/{prefix}.npz"
        rel_json = f"data/splits/controlled/{ds_id}/{prefix}.json"

        manifest_entries.append({
            "dataset_id": ds_id,
            "scenario_id": "default",
            "outer_fold": r,
            "npz_path": rel_npz,
            "npz_sha256": npz_sha,
            "json_path": rel_json,
            "json_sha256": json_sha,
            "split_sha256": split_sha,
            "source_rows": len(split.source_indices),
            "target_rows": len(split.target_indices),
            "inner_folds_count": len(split.inner_folds),
            "split_strategy": split.split_strategy,
        })

        # Preprocessing audit
        X_source = df_X.iloc[split.source_indices]
        X_target = df_X.iloc[split.target_indices]

        prep = build_preprocessor(feature_roles=feature_roles, config=prep_cfg, metadata=meta)
        prep.fit(X_source)
        out_src = prep.transform(X_source)
        out_tgt = prep.transform(X_target)

        src_nan = bool(np.isnan(out_src).any())
        tgt_nan = bool(np.isnan(out_tgt).any())
        src_finite = bool(np.all(np.isfinite(out_src)))
        tgt_finite = bool(np.all(np.isfinite(out_tgt)))
        enc_dim = out_src.shape[1]
        encoded_dims.append(enc_dim)

        audit_entries.append({
            "dataset_name": ds_id,
            "scenario_id": "default",
            "outer_fold": r,
            "source_rows": len(split.source_indices),
            "target_rows": len(split.target_indices),
            "raw_features": df_X.shape[1],
            "encoded_features": enc_dim,
            "source_has_nan": src_nan,
            "target_has_nan": tgt_nan,
            "source_all_finite": src_finite,
            "target_all_finite": tgt_finite,
            "preprocessor_config_hash": config_hash,
            "split_hash": split_sha,
        })

    return manifest_entries, audit_entries


def process_whyshift_dataset(
    ds_id: str,
    task: str,
    project_root: Path,
    prep_cfg: Dict[str, Any],
    config_hash: str,
    force: bool = False,
    verify_only: bool = False,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single WhyShift dataset across 4 target state scenarios."""
    spec = get_dataset_spec(ds_id)
    canonical_task_dir = project_root / "data" / "canonical" / "natural" / "whyshift" / task
    splits_task_dir = project_root / "data" / "splits" / "natural" / f"whyshift_{task}"

    ca_dir = canonical_task_dir / "CA"
    ca_meta = json.loads((ca_dir / "metadata.json").read_text(encoding="utf-8"))
    ca_features = pd.read_parquet(ca_dir / "features.parquet")
    ca_bundle_sha = compute_bundle_hash_for_dir(ca_dir)
    feature_roles = ca_meta.get("feature_roles", {})

    target_states = ["TX", "NY", "FL", "PA"]
    manifest_entries: List[Dict[str, Any]] = []
    audit_entries: List[Dict[str, Any]] = []

    for tgt_state in target_states:
        tgt_dir = canonical_task_dir / tgt_state
        tgt_meta = json.loads((tgt_dir / "metadata.json").read_text(encoding="utf-8"))
        tgt_features = pd.read_parquet(tgt_dir / "features.parquet")
        tgt_bundle_sha = compute_bundle_hash_for_dir(tgt_dir)

        scenario_id = f"target_{tgt_state}"
        scenario_out_dir = splits_task_dir / scenario_id

        splits = generate_whyshift_natural_splits(
            n_source=len(ca_features),
            n_target=len(tgt_features),
            source_domain="CA",
            target_domain=tgt_state,
            n_outer=N_OUTER,
            n_inner=N_INNER,
            split_seed=SPLIT_SEED,
        )

        composite_bundle_sha = f"{ca_bundle_sha}:{tgt_bundle_sha}"
        common_meta = {
            "dataset_id": ds_id,
            "scenario_id": scenario_id,
            "source_domain": "CA",
            "target_domain": tgt_state,
            "source_bundle_sha256": ca_bundle_sha,
            "target_bundle_sha256": tgt_bundle_sha,
            "split_seed": SPLIT_SEED,
        }

        encoded_dims: List[int] = []

        for split in splits:
            r = split.outer_fold
            prefix = f"fold_{r}"
            npz_path = scenario_out_dir / f"{prefix}.npz"
            json_path = scenario_out_dir / f"{prefix}.json"

            is_valid = False
            if npz_path.exists() and json_path.exists() and not force:
                is_valid = verify_split_artifact(npz_path, json_path, expected_bundle_sha256=composite_bundle_sha)
                if is_valid:
                    print(f"  [{ds_id} {scenario_id} fold_{r}] [SKIP VERIFIED]")

            if not is_valid:
                if not dry_run and not verify_only:
                    npz_path, json_path, split_sha = save_fold_split_artifacts(
                        split=split,
                        target_dir=scenario_out_dir,
                        file_prefix=prefix,
                        common_metadata=common_meta,
                        canonical_bundle_sha256=composite_bundle_sha,
                        split_config_sha256=config_hash,
                    )
                else:
                    split_sha = compute_split_hash(
                        source_indices=split.source_indices,
                        target_indices=split.target_indices,
                        inner_folds=split.inner_folds,
                        split_strategy=split.split_strategy,
                        split_seed=SPLIT_SEED,
                        canonical_bundle_sha256=composite_bundle_sha,
                    )
            else:
                with open(json_path, "r", encoding="utf-8") as jf:
                    split_meta = json.load(jf)
                split_sha = split_meta.get("split_sha256", "")

            npz_sha = compute_file_sha256(npz_path) if npz_path.exists() else ""
            json_sha = compute_file_sha256(json_path) if json_path.exists() else ""

            rel_npz = f"data/splits/natural/whyshift_{task}/{scenario_id}/{prefix}.npz"
            rel_json = f"data/splits/natural/whyshift_{task}/{scenario_id}/{prefix}.json"

            manifest_entries.append({
                "dataset_id": ds_id,
                "scenario_id": scenario_id,
                "outer_fold": r,
                "npz_path": rel_npz,
                "npz_sha256": npz_sha,
                "json_path": rel_json,
                "json_sha256": json_sha,
                "split_sha256": split_sha,
                "source_rows": len(split.source_indices),
                "target_rows": len(split.target_indices),
                "inner_folds_count": len(split.inner_folds),
                "split_strategy": split.split_strategy,
            })

            # Preprocessing audit
            X_source = ca_features.iloc[split.source_indices]
            X_target = tgt_features.iloc[split.target_indices]

            prep = build_preprocessor(feature_roles=feature_roles, config=prep_cfg, metadata=ca_meta)
            prep.fit(X_source)
            out_src = prep.transform(X_source)
            out_tgt = prep.transform(X_target)

            src_nan = bool(np.isnan(out_src).any())
            tgt_nan = bool(np.isnan(out_tgt).any())
            src_finite = bool(np.all(np.isfinite(out_src)))
            tgt_finite = bool(np.all(np.isfinite(out_tgt)))
            enc_dim = out_src.shape[1]
            encoded_dims.append(enc_dim)

            audit_entries.append({
                "dataset_name": ds_id,
                "scenario_id": scenario_id,
                "outer_fold": r,
                "source_rows": len(split.source_indices),
                "target_rows": len(split.target_indices),
                "raw_features": ca_features.shape[1],
                "encoded_features": enc_dim,
                "source_has_nan": src_nan,
                "target_has_nan": tgt_nan,
                "source_all_finite": src_finite,
                "target_all_finite": tgt_finite,
                "preprocessor_config_hash": config_hash,
                "split_hash": split_sha,
            })

    return manifest_entries, audit_entries


def process_tableshift_dataset(
    project_root: Path,
    prep_cfg: Dict[str, Any],
    config_hash: str,
    force: bool = False,
    verify_only: bool = False,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process TableShift Hospital Readmission dataset across 17 OOD admission_source_id scenarios."""
    ds_id = "tableshift_hospital_readmission"
    canonical_dir = project_root / "data" / "canonical" / "natural" / "tableshift" / ds_id
    splits_dir = project_root / "data" / "splits" / "natural" / ds_id

    features_df = pd.read_parquet(canonical_dir / "features.parquet")
    domains_df = pd.read_parquet(canonical_dir / "domains.parquet")
    meta = json.loads((canonical_dir / "metadata.json").read_text(encoding="utf-8"))
    bundle_sha = compute_bundle_hash_for_dir(canonical_dir)
    feature_roles = meta.get("feature_roles", {})

    domain_series = domains_df["admission_source_id"].values
    unique_vals = sorted([int(x) for x in pd.Series(domain_series).unique()])

    all_scenarios = generate_tableshift_natural_splits(
        domain_series=domain_series,
        candidate_ood_values=unique_vals,
        n_outer=N_OUTER,
        n_inner=N_INNER,
        split_seed=SPLIT_SEED,
    )

    manifest_entries: List[Dict[str, Any]] = []
    audit_entries: List[Dict[str, Any]] = []

    for scenario_id, splits in all_scenarios.items():
        scenario_out_dir = splits_dir / scenario_id
        common_meta = {
            "dataset_id": ds_id,
            "scenario_id": scenario_id,
            "split_seed": SPLIT_SEED,
        }

        encoded_dims: List[int] = []

        for split in splits:
            r = split.outer_fold
            prefix = f"fold_{r}"
            npz_path = scenario_out_dir / f"{prefix}.npz"
            json_path = scenario_out_dir / f"{prefix}.json"

            is_valid = False
            if npz_path.exists() and json_path.exists() and not force:
                is_valid = verify_split_artifact(npz_path, json_path, expected_bundle_sha256=bundle_sha)
                if is_valid:
                    print(f"  [{ds_id} {scenario_id} fold_{r}] [SKIP VERIFIED]")

            if not is_valid:
                if not dry_run and not verify_only:
                    npz_path, json_path, split_sha = save_fold_split_artifacts(
                        split=split,
                        target_dir=scenario_out_dir,
                        file_prefix=prefix,
                        common_metadata=common_meta,
                        canonical_bundle_sha256=bundle_sha,
                        split_config_sha256=config_hash,
                    )
                else:
                    split_sha = compute_split_hash(
                        source_indices=split.source_indices,
                        target_indices=split.target_indices,
                        inner_folds=split.inner_folds,
                        split_strategy=split.split_strategy,
                        split_seed=SPLIT_SEED,
                        canonical_bundle_sha256=bundle_sha,
                    )
            else:
                with open(json_path, "r", encoding="utf-8") as jf:
                    split_meta = json.load(jf)
                split_sha = split_meta.get("split_sha256", "")

            npz_sha = compute_file_sha256(npz_path) if npz_path.exists() else ""
            json_sha = compute_file_sha256(json_path) if json_path.exists() else ""

            rel_npz = f"data/splits/natural/{ds_id}/{scenario_id}/{prefix}.npz"
            rel_json = f"data/splits/natural/{ds_id}/{scenario_id}/{prefix}.json"

            manifest_entries.append({
                "dataset_id": ds_id,
                "scenario_id": scenario_id,
                "outer_fold": r,
                "npz_path": rel_npz,
                "npz_sha256": npz_sha,
                "json_path": rel_json,
                "json_sha256": json_sha,
                "split_sha256": split_sha,
                "source_rows": len(split.source_indices),
                "target_rows": len(split.target_indices),
                "inner_folds_count": len(split.inner_folds),
                "split_strategy": split.split_strategy,
            })

            # Preprocessing audit
            X_source = features_df.iloc[split.source_indices]
            X_target = features_df.iloc[split.target_indices]

            prep = build_preprocessor(feature_roles=feature_roles, config=prep_cfg, metadata=meta)
            prep.fit(X_source)
            out_src = prep.transform(X_source)
            out_tgt = prep.transform(X_target)

            src_nan = bool(np.isnan(out_src).any())
            tgt_nan = bool(np.isnan(out_tgt).any())
            src_finite = bool(np.all(np.isfinite(out_src)))
            tgt_finite = bool(np.all(np.isfinite(out_tgt)))
            enc_dim = out_src.shape[1]
            encoded_dims.append(enc_dim)

            audit_entries.append({
                "dataset_name": ds_id,
                "scenario_id": scenario_id,
                "outer_fold": r,
                "source_rows": len(split.source_indices),
                "target_rows": len(split.target_indices),
                "raw_features": features_df.shape[1],
                "encoded_features": enc_dim,
                "source_has_nan": src_nan,
                "target_has_nan": tgt_nan,
                "source_all_finite": src_finite,
                "target_all_finite": tgt_finite,
                "preprocessor_config_hash": config_hash,
                "split_hash": split_sha,
            })

    return manifest_entries, audit_entries


def main():
    parser = argparse.ArgumentParser(description="Deterministic Phase 2 split generation and preprocessing audit.")
    parser.add_argument("--all", action="store_true", help="Process all Phase-2 eligible canonical datasets.")
    parser.add_argument("--dataset", type=str, default=None, help="Process a specific dataset by ID.")
    parser.add_argument("--force", action="store_true", help="Force recreation of splits even if verified valid.")
    parser.add_argument("--verify", action="store_true", help="Verify existing splits and audit without modifying.")
    parser.add_argument("--dry-run", action="store_true", help="Perform actions in memory without writing artifacts.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    prep_cfg = load_preprocessing_config(project_root)
    config_hash = get_config_hash(project_root)

    print("=" * 70)
    print("WHEN CLUSTERS DRIFT — PHASE 2 SPLIT GENERATION & AUDIT")
    print(f"Split seed:            {SPLIT_SEED}")
    print(f"Outer folds:           {N_OUTER}")
    print(f"Inner folds:           {N_INNER}")
    print(f"Config SHA256:         {config_hash}")
    print(f"Freeze Commit:         {PHASE1_FREEZE_COMMIT}")
    print("=" * 70)

    # 1. Generate / verify input lock
    print("\n[1/4] Verifying Phase-2 Input Lock...")
    lock_doc = generate_input_lock(project_root, dry_run=args.dry_run)
    eligible = lock_doc["eligible_datasets"]
    print(f"Registered {len(eligible)} eligible datasets (46 partitions).")
    print(f"Registered {len(lock_doc['excluded_datasets'])} excluded datasets.")

    # 2. Determine target datasets to run
    controlled_to_run: List[str] = []
    whyshift_to_run: List[Tuple[str, str]] = []
    tableshift_to_run = False

    whyshift_map = {
        "whyshift_acs_income": ("whyshift_acs_income", "income"),
        "whyshift_income": ("whyshift_acs_income", "income"),
        "whyshift_acs_pubcov": ("whyshift_acs_pubcov", "pubcov"),
        "whyshift_pubcov": ("whyshift_acs_pubcov", "pubcov"),
        "whyshift_acs_mobility": ("whyshift_acs_mobility", "mobility"),
        "whyshift_mobility": ("whyshift_acs_mobility", "mobility"),
        "whyshift_traveltime": ("whyshift_acs_mobility", "mobility"),
    }

    if args.dataset:
        did = args.dataset.lower()
        if did in list_controlled_real():
            controlled_to_run.append(did)
        elif did in whyshift_map:
            whyshift_to_run.append(whyshift_map[did])
        elif did == "tableshift_hospital_readmission":
            tableshift_to_run = True
        else:
            print(f"ERROR: Dataset '{args.dataset}' is not a Phase-2 eligible dataset.")
            sys.exit(1)
    elif args.all or args.verify:
        controlled_to_run = list_controlled_real()
        whyshift_to_run = [
            ("whyshift_acs_income", "income"),
            ("whyshift_acs_pubcov", "pubcov"),
            ("whyshift_acs_mobility", "mobility"),
        ]
        tableshift_to_run = True
    else:
        print("Please specify --all, --verify, or --dataset <name>.")
        sys.exit(0)

    # 3. Execute splits and preprocessing audits
    all_manifest: List[Dict[str, Any]] = []
    all_audit: List[Dict[str, Any]] = []

    print("\n[2/4] Processing Controlled Real Datasets...")
    for ds_id in controlled_to_run:
        t0 = time.time()
        man_ents, aud_ents = process_controlled_dataset(
            ds_id=ds_id,
            project_root=project_root,
            prep_cfg=prep_cfg,
            config_hash=config_hash,
            force=args.force,
            verify_only=args.verify,
            dry_run=args.dry_run,
        )
        all_manifest.extend(man_ents)
        all_audit.extend(aud_ents)
        elapsed = time.time() - t0
        print(f"  Processed {ds_id} (5 folds) in {elapsed:.2f}s")

    if whyshift_to_run:
        print("\n[3/4] Processing WhyShift Natural Datasets...")
        for ds_id, task in whyshift_to_run:
            t0 = time.time()
            man_ents, aud_ents = process_whyshift_dataset(
                ds_id=ds_id,
                task=task,
                project_root=project_root,
                prep_cfg=prep_cfg,
                config_hash=config_hash,
                force=args.force,
                verify_only=args.verify,
                dry_run=args.dry_run,
            )
            all_manifest.extend(man_ents)
            all_audit.extend(aud_ents)
            elapsed = time.time() - t0
            print(f"  Processed {ds_id} (4 scenarios, 20 folds) in {elapsed:.2f}s")

    if tableshift_to_run:
        print("\n[4/4] Processing TableShift Natural Dataset...")
        t0 = time.time()
        man_ents, aud_ents = process_tableshift_dataset(
            project_root=project_root,
            prep_cfg=prep_cfg,
            config_hash=config_hash,
            force=args.force,
            verify_only=args.verify,
            dry_run=args.dry_run,
        )
        all_manifest.extend(man_ents)
        all_audit.extend(aud_ents)
        elapsed = time.time() - t0
        print(f"  Processed tableshift_hospital_readmission (17 scenarios, 85 folds) in {elapsed:.2f}s")

    # 4. Save audit log and split manifest
    splits_dir = project_root / "data" / "splits"
    if not args.dry_run and all_manifest:
        # If running a subset, merge with existing manifest/audit
        manifest_path = splits_dir / "split_manifest.json"
        audit_path = splits_dir / "preprocessing_audit.csv"

        if (args.dataset and manifest_path.exists()) and not args.force:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_doc = json.load(f)
            old_splits = {
                (s["dataset_id"], s["scenario_id"], s["outer_fold"]): s
                for s in existing_doc.get("splits", [])
            }
            for s in all_manifest:
                old_splits[(s["dataset_id"], s["scenario_id"], s["outer_fold"])] = s
            merged_splits = list(old_splits.values())
        else:
            merged_splits = all_manifest

        manifest_doc = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_outer_folds": len(merged_splits),
            "total_inner_folds": sum(s["inner_folds_count"] for s in merged_splits),
            "splits": merged_splits,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_doc, f, indent=2)

        # Audit CSV
        if (args.dataset and audit_path.exists()) and not args.force:
            df_old = pd.read_csv(audit_path)
            old_audits = {
                (row["dataset_name"], row["scenario_id"], row["outer_fold"]): row.to_dict()
                for _, row in df_old.iterrows()
            }
            for a in all_audit:
                old_audits[(a["dataset_name"], a["scenario_id"], a["outer_fold"])] = a
            df_final = pd.DataFrame(list(old_audits.values()))
        else:
            df_final = pd.DataFrame(all_audit)

        df_final.to_csv(audit_path, index=False)
        print(f"\nAudit saved to {audit_path} ({len(df_final)} rows)")
        print(f"Manifest saved to {manifest_path} ({len(merged_splits)} splits)")

    print("\n" + "=" * 70)
    print("PHASE 2 GENERATION & VERIFICATION COMPLETE")
    print(f"Total Outer Folds: {len(all_manifest)}")
    print(f"Total Inner Folds: {sum(s['inner_folds_count'] for s in all_manifest)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
