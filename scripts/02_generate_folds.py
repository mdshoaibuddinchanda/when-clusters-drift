#!/usr/bin/env python3
"""Script 02: Deterministic source/target and inner-validation split generation and preprocessing audit."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from clusterdrift.data.folds import (
    FoldSplit,
    compute_preprocessing_config_sha256,
    compute_scenario_input_sha256,
    compute_split_hash,
    compute_split_protocol_sha256,
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
from clusterdrift.shifts.hashing import atomic_write_csv, atomic_write_json

PHASE1_FREEZE_COMMIT = "d5cb818589a138b25f10e926e9e83b198d369599"
DATE_OF_FREEZE = "2026-09-07"

EXCLUDED_DATASETS = [
    {
        "dataset_id": "whyshift_taxi",
        "reason": "not canonicalized and validated at Phase-1 experiment freeze.",
    },
    {
        "dataset_id": "whyshift_us_accidents",
        "reason": "not canonicalized and validated at Phase-1 experiment freeze.",
    },
    {
        "dataset_id": "tableshift_college_scorecard",
        "reason": "not canonicalized and validated at Phase-1 experiment freeze.",
    },
    {
        "dataset_id": "tableshift_heloc",
        "reason": "not canonicalized and validated at Phase-1 experiment freeze.",
    },
    {
        "dataset_id": "tableshift_assistments",
        "reason": "not canonicalized and validated at Phase-1 experiment freeze.",
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


def generate_input_lock(
    project_root: Path,
    prep_cfg: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Generate and persist data/splits/phase2_input_lock.json."""
    lock_path = project_root / "data" / "splits" / "phase2_input_lock.json"
    canonical_root = project_root / "data" / "canonical"

    datasets_manifest_path = project_root / "data" / "manifests" / "datasets.json"
    synthetic_manifest_path = project_root / "data" / "manifests" / "synthetic_manifest.json"

    datasets_manifest_sha = (
        compute_file_sha256(datasets_manifest_path) if datasets_manifest_path.exists() else ""
    )
    synthetic_manifest_sha = (
        compute_file_sha256(synthetic_manifest_path) if synthetic_manifest_path.exists() else ""
    )

    split_protocol_sha = compute_split_protocol_sha256(prep_cfg)
    prep_config_sha = compute_preprocessing_config_sha256(prep_cfg)

    split_cfg = prep_cfg.get("splitting", {})
    outer_folds = int(split_cfg.get("outer_folds", 5))
    inner_folds = int(split_cfg.get("inner_folds", 3))
    split_seed = int(split_cfg.get("split_seed", 20260907))

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

    # Preserve existing timestamp if present
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if lock_path.exists():
        try:
            old_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            if "created_at" in old_lock:
                created_at = old_lock["created_at"]
        except Exception:
            pass

    lock_doc = {
        "phase2_protocol_version": prep_cfg.get("protocol_version", 1),
        "source_git_commit": PHASE1_FREEZE_COMMIT,
        "created_at": created_at,
        "datasets_manifest_sha256": datasets_manifest_sha,
        "synthetic_manifest_sha256": synthetic_manifest_sha,
        "split_protocol_sha256": split_protocol_sha,
        "preprocessing_config_sha256": prep_config_sha,
        "split_seed": split_seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "total_eligible_datasets": len(eligible_datasets),
        "total_partitions": 46,
        "eligible_datasets": eligible_datasets,
        "excluded_datasets": EXCLUDED_DATASETS,
    }

    if not dry_run:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(lock_path, lock_doc, indent=2, sort_keys=False)

    return lock_doc


def process_controlled_dataset(
    ds_id: str,
    project_root: Path,
    prep_cfg: Dict[str, Any],
    split_protocol_sha: str,
    prep_config_sha: str,
    force: bool = False,
    verify_only: bool = False,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single controlled dataset, generating splits and auditing preprocessing."""
    spec = get_dataset_spec(ds_id)
    ds_dir = project_root / "data" / "canonical" / "controlled" / ds_id
    out_dir = project_root / "data" / "splits" / "controlled" / ds_id

    features_path = ds_dir / "features.parquet"
    groups_path = ds_dir / "groups.parquet"
    meta_path = ds_dir / "metadata.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bundle_sha = compute_bundle_hash_for_dir(ds_dir)
    feature_roles = meta.get("feature_roles", {})

    df_X = pd.read_parquet(features_path)
    n_samples = len(df_X)

    split_cfg = prep_cfg.get("splitting", {})
    n_outer = int(split_cfg.get("outer_folds", 5))
    n_inner = int(split_cfg.get("inner_folds", 3))
    split_seed = int(split_cfg.get("split_seed", 20260907))

    # Determine splits
    if spec.split_strategy == "group_kfold":
        if not groups_path.exists():
            raise FileNotFoundError(f"Missing groups.parquet for group_kfold dataset {ds_id}")
        df_g = pd.read_parquet(groups_path)
        group_series = df_g.iloc[:, 0].values
        group_col_name = "mouse_subject_id" if ds_id == "mice_protein_expression" else (spec.group_column or "group_id")
        splits = generate_group_kfold_splits(
            groups=group_series,
            n_outer=n_outer,
            n_inner=n_inner,
            group_column_name=group_col_name,
        )
    elif spec.split_strategy == "temporal_block":
        time_cols = split_cfg.get("temporal", {}).get("order_columns", ["date", "period"])
        splits = generate_temporal_block_splits(
            features_df=df_X,
            time_columns=time_cols,
            n_outer=n_outer,
            n_inner=n_inner,
        )
    else:
        splits = generate_kfold_splits(
            n_samples=n_samples,
            n_outer=n_outer,
            n_inner=n_inner,
            split_seed=split_seed,
        )

    manifest_entries: List[Dict[str, Any]] = []
    audit_entries: List[Dict[str, Any]] = []

    common_meta = {
        "dataset_id": ds_id,
        "scenario_id": "default",
        "split_seed": split_seed,
    }

    for split in splits:
        r = split.outer_fold
        prefix = f"fold_{r}"
        npz_path = out_dir / f"{prefix}.npz"
        json_path = out_dir / f"{prefix}.json"

        is_valid = False
        if npz_path.exists() and json_path.exists() and not force:
            is_valid = verify_split_artifact(
                npz_path=npz_path,
                json_path=json_path,
                expected_bundle_sha256=bundle_sha,
                expected_protocol_sha256=split_protocol_sha,
                expected_split_seed=split_seed,
                expected_strategy=split.split_strategy,
            )
            if is_valid:
                print(f"  [{ds_id} fold_{r}] [SKIP VERIFIED]")

        if not is_valid:
            if verify_only:
                raise RuntimeError(f"Split artifact invalid or missing for {ds_id} fold_{r} during verify.")
            if not dry_run:
                npz_path, json_path, split_sha = save_fold_split_artifacts(
                    split=split,
                    target_dir=out_dir,
                    file_prefix=prefix,
                    common_metadata=common_meta,
                    canonical_bundle_sha256=bundle_sha,
                    split_config_sha256=prep_config_sha,
                    split_protocol_sha256=split_protocol_sha,
                    scenario_input_sha256=bundle_sha,
                )
            else:
                split_sha = compute_split_hash(
                    source_indices=split.source_indices,
                    target_indices=split.target_indices,
                    inner_folds=split.inner_folds,
                    split_strategy=split.split_strategy,
                    split_seed=split_seed,
                    canonical_bundle_sha256=bundle_sha,
                    split_protocol_sha256=split_protocol_sha,
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

        src_miss_before = int(X_source.isna().sum().sum())
        tgt_miss_before = int(X_target.isna().sum().sum())

        prep = build_preprocessor(feature_roles=feature_roles, config=prep_cfg, metadata=meta)
        prep.fit(X_source)
        out_src = prep.transform(X_source)
        out_tgt = prep.transform(X_target)

        src_miss_after = int(np.isnan(out_src).sum())
        tgt_miss_after = int(np.isnan(out_tgt).sum())
        unseen_count, unseen_cols = prep.get_unseen_categories(X_target)

        audit_entries.append({
            "dataset_id": ds_id,
            "scenario_id": "default",
            "outer_fold": r,
            "source_rows": len(split.source_indices),
            "target_rows": len(split.target_indices),
            "raw_features": df_X.shape[1],
            "encoded_features": out_src.shape[1],
            "numeric_features": len(prep.numeric_cols),
            "categorical_features": len(prep.categorical_cols),
            "ordinal_features": len(prep.ordinal_cols),
            "boolean_features": len(prep.boolean_cols),
            "source_missing_before": src_miss_before,
            "target_missing_before": tgt_miss_before,
            "source_missing_after": src_miss_after,
            "target_missing_after": tgt_miss_after,
            "unseen_target_category_occurrences": unseen_count,
            "unseen_target_category_columns": json.dumps(unseen_cols),
            "source_all_finite": bool(np.all(np.isfinite(out_src))),
            "target_all_finite": bool(np.all(np.isfinite(out_tgt))),
            "canonical_or_scenario_input_sha256": bundle_sha,
            "split_protocol_sha256": split_protocol_sha,
            "preprocessing_config_sha256": prep_config_sha,
            "split_sha256": split_sha,
        })

    return manifest_entries, audit_entries


def process_whyshift_dataset(
    ds_id: str,
    task: str,
    project_root: Path,
    prep_cfg: Dict[str, Any],
    split_protocol_sha: str,
    prep_config_sha: str,
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

    split_cfg = prep_cfg.get("splitting", {})
    n_outer = int(split_cfg.get("outer_folds", 5))
    n_inner = int(split_cfg.get("inner_folds", 3))
    split_seed = int(split_cfg.get("split_seed", 20260907))

    target_states = ["TX", "NY", "FL", "PA"]
    manifest_entries: List[Dict[str, Any]] = []
    audit_entries: List[Dict[str, Any]] = []

    for tgt_state in target_states:
        tgt_dir = canonical_task_dir / tgt_state
        tgt_features = pd.read_parquet(tgt_dir / "features.parquet")
        tgt_bundle_sha = compute_bundle_hash_for_dir(tgt_dir)

        scenario_id = f"target_{tgt_state}"
        scenario_out_dir = splits_task_dir / scenario_id

        splits = generate_whyshift_natural_splits(
            n_source=len(ca_features),
            n_target=len(tgt_features),
            source_domain="CA",
            target_domain=tgt_state,
            n_outer=n_outer,
            n_inner=n_inner,
            split_seed=split_seed,
        )

        scenario_input_sha = compute_scenario_input_sha256(ca_bundle_sha, tgt_bundle_sha)

        common_meta = {
            "dataset_id": ds_id,
            "scenario_id": scenario_id,
            "source_domain": "CA",
            "target_domain": tgt_state,
            "source_bundle_sha256": ca_bundle_sha,
            "target_bundle_sha256": tgt_bundle_sha,
            "scenario_input_sha256": scenario_input_sha,
            "split_seed": split_seed,
        }

        for split in splits:
            r = split.outer_fold
            prefix = f"fold_{r}"
            npz_path = scenario_out_dir / f"{prefix}.npz"
            json_path = scenario_out_dir / f"{prefix}.json"

            is_valid = False
            if npz_path.exists() and json_path.exists() and not force:
                is_valid = verify_split_artifact(
                    npz_path=npz_path,
                    json_path=json_path,
                    expected_bundle_sha256=scenario_input_sha,
                    expected_protocol_sha256=split_protocol_sha,
                    expected_split_seed=split_seed,
                    expected_strategy=split.split_strategy,
                )
                if is_valid:
                    print(f"  [{ds_id} {scenario_id} fold_{r}] [SKIP VERIFIED]")

            if not is_valid:
                if verify_only:
                    raise RuntimeError(f"Split artifact invalid or missing for {ds_id} {scenario_id} fold_{r} during verify.")
                if not dry_run:
                    npz_path, json_path, split_sha = save_fold_split_artifacts(
                        split=split,
                        target_dir=scenario_out_dir,
                        file_prefix=prefix,
                        common_metadata=common_meta,
                        canonical_bundle_sha256=ca_bundle_sha,
                        split_config_sha256=prep_config_sha,
                        split_protocol_sha256=split_protocol_sha,
                        scenario_input_sha256=scenario_input_sha,
                    )
                else:
                    split_sha = compute_split_hash(
                        source_indices=split.source_indices,
                        target_indices=split.target_indices,
                        inner_folds=split.inner_folds,
                        split_strategy=split.split_strategy,
                        split_seed=split_seed,
                        canonical_bundle_sha256=scenario_input_sha,
                        split_protocol_sha256=split_protocol_sha,
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

            src_miss_before = int(X_source.isna().sum().sum())
            tgt_miss_before = int(X_target.isna().sum().sum())

            prep = build_preprocessor(feature_roles=feature_roles, config=prep_cfg, metadata=ca_meta)
            prep.fit(X_source)
            out_src = prep.transform(X_source)
            out_tgt = prep.transform(X_target)

            src_miss_after = int(np.isnan(out_src).sum())
            tgt_miss_after = int(np.isnan(out_tgt).sum())
            unseen_count, unseen_cols = prep.get_unseen_categories(X_target)

            audit_entries.append({
                "dataset_id": ds_id,
                "scenario_id": scenario_id,
                "outer_fold": r,
                "source_rows": len(split.source_indices),
                "target_rows": len(split.target_indices),
                "raw_features": ca_features.shape[1],
                "encoded_features": out_src.shape[1],
                "numeric_features": len(prep.numeric_cols),
                "categorical_features": len(prep.categorical_cols),
                "ordinal_features": len(prep.ordinal_cols),
                "boolean_features": len(prep.boolean_cols),
                "source_missing_before": src_miss_before,
                "target_missing_before": tgt_miss_before,
                "source_missing_after": src_miss_after,
                "target_missing_after": tgt_miss_after,
                "unseen_target_category_occurrences": unseen_count,
                "unseen_target_category_columns": json.dumps(unseen_cols),
                "source_all_finite": bool(np.all(np.isfinite(out_src))),
                "target_all_finite": bool(np.all(np.isfinite(out_tgt))),
                "canonical_or_scenario_input_sha256": scenario_input_sha,
                "split_protocol_sha256": split_protocol_sha,
                "preprocessing_config_sha256": prep_config_sha,
                "split_sha256": split_sha,
            })

    return manifest_entries, audit_entries


def process_tableshift_dataset(
    project_root: Path,
    prep_cfg: Dict[str, Any],
    split_protocol_sha: str,
    prep_config_sha: str,
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

    split_cfg = prep_cfg.get("splitting", {})
    n_outer = int(split_cfg.get("outer_folds", 5))
    n_inner = int(split_cfg.get("inner_folds", 3))
    split_seed = int(split_cfg.get("split_seed", 20260907))

    all_scenarios = generate_tableshift_natural_splits(
        domain_series=domain_series,
        candidate_ood_values=unique_vals,
        n_outer=n_outer,
        n_inner=n_inner,
        split_seed=split_seed,
    )

    manifest_entries: List[Dict[str, Any]] = []
    audit_entries: List[Dict[str, Any]] = []

    for scenario_id, splits in all_scenarios.items():
        scenario_out_dir = splits_dir / scenario_id
        target_sample_count = len(splits[0].target_indices)

        common_meta = {
            "dataset_id": ds_id,
            "scenario_id": scenario_id,
            "target_sample_count": target_sample_count,
            "split_seed": split_seed,
        }

        for split in splits:
            r = split.outer_fold
            prefix = f"fold_{r}"
            npz_path = scenario_out_dir / f"{prefix}.npz"
            json_path = scenario_out_dir / f"{prefix}.json"

            is_valid = False
            if npz_path.exists() and json_path.exists() and not force:
                is_valid = verify_split_artifact(
                    npz_path=npz_path,
                    json_path=json_path,
                    expected_bundle_sha256=bundle_sha,
                    expected_protocol_sha256=split_protocol_sha,
                    expected_split_seed=split_seed,
                    expected_strategy=split.split_strategy,
                )
                if is_valid:
                    print(f"  [{ds_id} {scenario_id} fold_{r}] [SKIP VERIFIED]")

            if not is_valid:
                if verify_only:
                    raise RuntimeError(f"Split artifact invalid or missing for {ds_id} {scenario_id} fold_{r} during verify.")
                if not dry_run:
                    npz_path, json_path, split_sha = save_fold_split_artifacts(
                        split=split,
                        target_dir=scenario_out_dir,
                        file_prefix=prefix,
                        common_metadata=common_meta,
                        canonical_bundle_sha256=bundle_sha,
                        split_config_sha256=prep_config_sha,
                        split_protocol_sha256=split_protocol_sha,
                        scenario_input_sha256=bundle_sha,
                    )
                else:
                    split_sha = compute_split_hash(
                        source_indices=split.source_indices,
                        target_indices=split.target_indices,
                        inner_folds=split.inner_folds,
                        split_strategy=split.split_strategy,
                        split_seed=split_seed,
                        canonical_bundle_sha256=bundle_sha,
                        split_protocol_sha256=split_protocol_sha,
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

            src_miss_before = int(X_source.isna().sum().sum())
            tgt_miss_before = int(X_target.isna().sum().sum())

            prep = build_preprocessor(feature_roles=feature_roles, config=prep_cfg, metadata=meta)
            prep.fit(X_source)
            out_src = prep.transform(X_source)
            out_tgt = prep.transform(X_target)

            src_miss_after = int(np.isnan(out_src).sum())
            tgt_miss_after = int(np.isnan(out_tgt).sum())
            unseen_count, unseen_cols = prep.get_unseen_categories(X_target)

            audit_entries.append({
                "dataset_id": ds_id,
                "scenario_id": scenario_id,
                "outer_fold": r,
                "source_rows": len(split.source_indices),
                "target_rows": len(split.target_indices),
                "raw_features": features_df.shape[1],
                "encoded_features": out_src.shape[1],
                "numeric_features": len(prep.numeric_cols),
                "categorical_features": len(prep.categorical_cols),
                "ordinal_features": len(prep.ordinal_cols),
                "boolean_features": len(prep.boolean_cols),
                "source_missing_before": src_miss_before,
                "target_missing_before": tgt_miss_before,
                "source_missing_after": src_miss_after,
                "target_missing_after": tgt_miss_after,
                "unseen_target_category_occurrences": unseen_count,
                "unseen_target_category_columns": json.dumps(unseen_cols),
                "source_all_finite": bool(np.all(np.isfinite(out_src))),
                "target_all_finite": bool(np.all(np.isfinite(out_tgt))),
                "canonical_or_scenario_input_sha256": bundle_sha,
                "split_protocol_sha256": split_protocol_sha,
                "preprocessing_config_sha256": prep_config_sha,
                "split_sha256": split_sha,
            })

    return manifest_entries, audit_entries


def main():
    parser = argparse.ArgumentParser(description="Deterministic Phase 2 split generation and preprocessing audit.")
    parser.add_argument("--all", action="store_true", help="Process all Phase-2 eligible canonical datasets.")
    parser.add_argument("--dataset", type=str, default=None, help="Process a specific dataset by ID.")
    parser.add_argument("--force", action="store_true", help="Force recreation of splits even if verified valid.")
    parser.add_argument("--verify", action="store_true", help="Verify existing splits without modifying any files.")
    parser.add_argument("--dry-run", action="store_true", help="Perform actions in memory without writing artifacts.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    prep_cfg = load_preprocessing_config(project_root)

    split_cfg = prep_cfg.get("splitting", {})
    outer_folds = int(split_cfg.get("outer_folds", 5))
    inner_folds = int(split_cfg.get("inner_folds", 3))
    split_seed = int(split_cfg.get("split_seed", 20260907))

    split_protocol_sha = compute_split_protocol_sha256(prep_cfg)
    prep_config_sha = compute_preprocessing_config_sha256(prep_cfg)

    print("=" * 70)
    print("WHEN CLUSTERS DRIFT — PHASE 2 SPLIT GENERATION & AUDIT")
    print(f"Split seed:            {split_seed}")
    print(f"Outer folds:           {outer_folds}")
    print(f"Inner folds:           {inner_folds}")
    print(f"Split Protocol SHA:    {split_protocol_sha}")
    print(f"Prep Config SHA:       {prep_config_sha}")
    print(f"Freeze Commit:         {PHASE1_FREEZE_COMMIT}")
    if args.verify:
        print("MODE:                  STRICTLY READ-ONLY VERIFICATION")
    print("=" * 70)

    # 1. Generate / verify input lock
    print("\n[1/4] Verifying Phase-2 Input Lock...")
    lock_doc = generate_input_lock(project_root, prep_cfg, dry_run=(args.dry_run or args.verify))
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

    try:
        print("\n[2/4] Processing Controlled Real Datasets...")
        for ds_id in controlled_to_run:
            t0 = time.time()
            man_ents, aud_ents = process_controlled_dataset(
                ds_id=ds_id,
                project_root=project_root,
                prep_cfg=prep_cfg,
                split_protocol_sha=split_protocol_sha,
                prep_config_sha=prep_config_sha,
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
                    split_protocol_sha=split_protocol_sha,
                    prep_config_sha=prep_config_sha,
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
                split_protocol_sha=split_protocol_sha,
                prep_config_sha=prep_config_sha,
                force=args.force,
                verify_only=args.verify,
                dry_run=args.dry_run,
            )
            all_manifest.extend(man_ents)
            all_audit.extend(aud_ents)
            elapsed = time.time() - t0
            print(f"  Processed tableshift_hospital_readmission (17 scenarios, 85 folds) in {elapsed:.2f}s")

    except Exception as e:
        print(f"\nERROR during processing/verification: {e}")
        sys.exit(1)

    # 4. Save audit log and split manifest (ONLY if NOT in verify_only or dry_run mode)
    splits_dir = project_root / "data" / "splits"
    if not args.dry_run and not args.verify and all_manifest:
        manifest_path = splits_dir / "split_manifest.json"
        audit_path = splits_dir / "preprocessing_audit.csv"

        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if manifest_path.exists():
            try:
                old_man = json.loads(manifest_path.read_text(encoding="utf-8"))
                if "created_at" in old_man:
                    created_at = old_man["created_at"]
            except Exception:
                pass

        if args.dataset and manifest_path.exists() and not args.force:
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
            "created_at": created_at,
            "total_outer_folds": len(merged_splits),
            "total_inner_folds": sum(s["inner_folds_count"] for s in merged_splits),
            "splits": merged_splits,
        }
        atomic_write_json(manifest_path, manifest_doc, indent=2, sort_keys=False)

        if args.dataset and audit_path.exists() and not args.force:
            df_old = pd.read_csv(audit_path)
            old_audits = {
                (row["dataset_id"], row["scenario_id"], row["outer_fold"]): row.to_dict()
                for _, row in df_old.iterrows()
            }
            for a in all_audit:
                old_audits[(a["dataset_id"], a["scenario_id"], a["outer_fold"])] = a
            df_final = pd.DataFrame(list(old_audits.values()))
        else:
            df_final = pd.DataFrame(all_audit)

        atomic_write_csv(audit_path, df_final, index=False)
        print(f"\nAudit saved to {audit_path} ({len(df_final)} rows)")
        print(f"Manifest saved to {manifest_path} ({len(merged_splits)} splits)")

    print("\n" + "=" * 70)
    print("PHASE 2 GENERATION & VERIFICATION COMPLETE")
    print(f"Total Outer Folds: {len(all_manifest)}")
    print(f"Total Inner Folds: {sum(s['inner_folds_count'] for s in all_manifest)}")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
