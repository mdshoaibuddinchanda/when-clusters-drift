"""Rigorous test suite for Phase 2 zero-leakage, split reproducibility, and preprocessing isolation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.data.folds import (
    compute_split_hash,
    generate_group_kfold_splits,
    generate_kfold_splits,
    generate_tableshift_natural_splits,
    generate_temporal_block_splits,
    generate_whyshift_natural_splits,
    verify_split_artifact,
)
from clusterdrift.data.manifest import compute_canonical_bundle_sha256, compute_file_sha256
from clusterdrift.data.preprocess import SourceOnlyPreprocessor, build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_split_index_reproducibility():
    """Verify that calling split generation twice with the same seed produces bit-identical indices and hashes."""
    s1 = generate_kfold_splits(n_samples=500, n_outer=5, n_inner=3, split_seed=20260907)
    s2 = generate_kfold_splits(n_samples=500, n_outer=5, n_inner=3, split_seed=20260907)

    for fold1, fold2 in zip(s1, s2):
        np.testing.assert_array_equal(fold1.source_indices, fold2.source_indices)
        np.testing.assert_array_equal(fold1.target_indices, fold2.target_indices)

        h1 = compute_split_hash(
            source_indices=fold1.source_indices,
            target_indices=fold1.target_indices,
            inner_folds=fold1.inner_folds,
            split_strategy=fold1.split_strategy,
            split_seed=20260907,
            canonical_bundle_sha256="dummy_bundle_hash",
        )
        h2 = compute_split_hash(
            source_indices=fold2.source_indices,
            target_indices=fold2.target_indices,
            inner_folds=fold2.inner_folds,
            split_strategy=fold2.split_strategy,
            split_seed=20260907,
            canonical_bundle_sha256="dummy_bundle_hash",
        )
        assert h1 == h2


def test_source_target_disjointness_across_all_manifest_splits():
    """Verify that every generated fold in split_manifest.json has completely disjoint source and target indices."""
    manifest_path = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"
    assert manifest_path.exists(), "split_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    for entry in doc["splits"]:
        npz_path = PROJECT_ROOT / entry["npz_path"]
        assert npz_path.exists(), f"Missing split NPZ: {npz_path}"
        with np.load(npz_path) as npz:
            src = npz["source_indices"]
            tgt = npz["target_indices"]

            # For Whyshift, source is CA indices (0..N_CA-1) and target is Target state indices (0..N_Tgt-1).
            # Within single-partition datasets, indices must be strictly disjoint sets:
            if "whyshift" not in entry["dataset_id"]:
                assert len(np.intersect1d(src, tgt)) == 0, (
                    f"Leakage detected in {entry['dataset_id']} scenario {entry['scenario_id']} fold {entry['outer_fold']}"
                )


def test_group_isolation_mice_and_har():
    """Verify that GroupKFold strictly prevents any mouse or subject from appearing in both source and target."""
    # 1. Mice Protein
    mice_groups_path = PROJECT_ROOT / "data" / "canonical" / "controlled" / "mice_protein_expression" / "groups.parquet"
    assert mice_groups_path.exists()
    mice_groups = pd.read_parquet(mice_groups_path).iloc[:, 0].values
    assert len(np.unique(mice_groups)) == 72

    mice_splits = generate_group_kfold_splits(groups=mice_groups, n_outer=5, n_inner=3, group_column_name="MouseID")
    for s in mice_splits:
        src_mice = set(mice_groups[s.source_indices])
        tgt_mice = set(mice_groups[s.target_indices])
        assert src_mice.isdisjoint(tgt_mice), f"Mice group leakage detected in fold {s.outer_fold}"
        for inf in s.inner_folds:
            in_tr_mice = set(mice_groups[inf.train_indices])
            in_val_mice = set(mice_groups[inf.val_indices])
            assert in_tr_mice.isdisjoint(in_val_mice), f"Inner mice group leakage in fold {s.outer_fold}"

    # 2. HAR
    har_groups_path = PROJECT_ROOT / "data" / "canonical" / "controlled" / "human_activity_recognition" / "groups.parquet"
    assert har_groups_path.exists()
    har_groups = pd.read_parquet(har_groups_path).iloc[:, 0].values
    assert len(np.unique(har_groups)) == 30

    har_splits = generate_group_kfold_splits(groups=har_groups, n_outer=5, n_inner=3, group_column_name="subject")
    for s in har_splits:
        src_subjects = set(har_groups[s.source_indices])
        tgt_subjects = set(har_groups[s.target_indices])
        assert src_subjects.isdisjoint(tgt_subjects), f"HAR subject leakage detected in fold {s.outer_fold}"
        for inf in s.inner_folds:
            in_tr_subj = set(har_groups[inf.train_indices])
            in_val_subj = set(har_groups[inf.val_indices])
            assert in_tr_subj.isdisjoint(in_val_subj), f"Inner HAR subject leakage in fold {s.outer_fold}"


def test_temporal_ordering_electricity():
    """Verify that temporal splits for Electricity strictly prevent future-to-past leakage."""
    elec_path = PROJECT_ROOT / "data" / "canonical" / "controlled" / "electricity" / "features.parquet"
    assert elec_path.exists()
    df_elec = pd.read_parquet(elec_path)
    time_series = df_elec["date"].values

    splits = generate_temporal_block_splits(time_values=time_series, n_outer=5, n_inner=3, time_column_name="date")

    for s in splits:
        max_source_date = np.max(time_series[s.source_indices])
        min_target_date = np.min(time_series[s.target_indices])
        assert max_source_date <= min_target_date, (
            f"Future leakage in Electricity fold {s.outer_fold}: max source {max_source_date} > min target {min_target_date}"
        )

        for inf in s.inner_folds:
            max_inner_tr = np.max(time_series[inf.train_indices])
            min_inner_val = np.min(time_series[inf.val_indices])
            assert max_inner_tr <= min_inner_val, (
                f"Future leakage in Electricity inner fold {inf.inner_fold} of outer {s.outer_fold}"
            )


def test_inner_fold_isolation_and_no_target_leakage():
    """Verify that inner folds are strictly sub-partitions of source and disjoint from target."""
    n_samples = 300
    splits = generate_kfold_splits(n_samples=n_samples, n_outer=5, n_inner=3, split_seed=20260907)

    for s in splits:
        source_set = set(s.source_indices)
        target_set = set(s.target_indices)

        for inf in s.inner_folds:
            tr_set = set(inf.train_indices)
            val_set = set(inf.val_indices)

            # 1. Disjointness
            assert tr_set.isdisjoint(val_set)
            # 2. Strict subsets of source
            assert tr_set.issubset(source_set)
            assert val_set.issubset(source_set)
            # 3. Union equals source
            assert tr_set.union(val_set) == source_set
            # 4. Zero target contamination
            assert tr_set.isdisjoint(target_set)
            assert val_set.isdisjoint(target_set)


def test_source_only_scaler_isolation_outlier_injection():
    """Verify that extreme target outliers do not alter source-fitted standard scalers."""
    cfg = {"numeric": {"imputer": "median", "scaler": "standard"}}
    roles = {"f1": "numeric", "f2": "numeric"}

    X_source = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "f2": [10.0, 20.0, 30.0, 40.0, 50.0]})
    X_target_clean = pd.DataFrame({"f1": [2.5, 3.5], "f2": [25.0, 35.0]})
    X_target_corrupted = pd.DataFrame({"f1": [1e9, -1e9], "f2": [1e12, -1e12]})

    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    # Record source-fitted parameters
    mean_before = prep.numeric_scaler.mean_.copy()
    scale_before = prep.numeric_scaler.scale_.copy()

    # Transform clean target and corrupted target
    out_src1 = prep.transform(X_source)
    prep.transform(X_target_clean)
    prep.transform(X_target_corrupted)
    out_src2 = prep.transform(X_source)

    # Parameters and source outputs must remain identical
    np.testing.assert_array_equal(prep.numeric_scaler.mean_, mean_before)
    np.testing.assert_array_equal(prep.numeric_scaler.scale_, scale_before)
    np.testing.assert_array_equal(out_src1, out_src2)


def test_source_only_imputer_isolation_missing_value_injection():
    """Verify that target missing values are imputed strictly using source statistics."""
    cfg = {"numeric": {"imputer": "median", "scaler": "standard"}}
    roles = {"val": "numeric"}

    # Source has median = 10.0
    X_source = pd.DataFrame({"val": [5.0, 10.0, 15.0]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    # Imputer statistics must equal [10.0]
    assert prep.numeric_imputer.statistics_[0] == 10.0

    # Target has NaNs and extreme values
    X_target = pd.DataFrame({"val": [np.nan, 9999.0, np.nan]})
    out_tgt = prep.transform(X_target)

    # Source-fitted median (10.0) scaled by (val - 10.0) / std(source) -> 0.0
    # NaNs imputed with 10.0 will scale to 0.0
    assert np.isclose(out_tgt[0, 0], 0.0)
    assert np.isclose(out_tgt[2, 0], 0.0)
    assert prep.numeric_imputer.statistics_[0] == 10.0


def test_unseen_target_categories_handling():
    """Verify that unseen categorical levels in target data are handled gracefully without errors or NaNs."""
    cfg = {
        "categorical": {
            "imputer": "most_frequent",
            "encoder": "onehot",
            "handle_unknown": "ignore",
        }
    }
    roles = {"city": "categorical"}

    X_source = pd.DataFrame({"city": ["Paris", "London", "Paris", "Berlin"]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    # Target contains unseen categories: 'Tokyo', 'Sydney'
    X_target = pd.DataFrame({"city": ["London", "Tokyo", "Paris", "Sydney", np.nan]})

    unseen_count = prep.count_unseen_categories(X_target)
    assert unseen_count == 2

    out_tgt = prep.transform(X_target)
    assert not np.isnan(out_tgt).any()
    assert np.all(np.isfinite(out_tgt))
    # Unseen categories Tokyo and Sydney should be encoded to all zeros
    assert np.all(out_tgt[1] == 0.0)
    assert np.all(out_tgt[3] == 0.0)


def test_output_finiteness_and_float32_dtype():
    """Verify that output matrices are strictly finite and typed as float32."""
    cfg = {
        "output_dtype": "float32",
        "numeric": {"imputer": "median", "scaler": "standard"},
        "categorical": {"imputer": "most_frequent", "encoder": "onehot", "handle_unknown": "ignore"},
        "boolean": {"imputer": "most_frequent", "encoding": "zero_one"},
    }
    roles = {"num": "numeric", "cat": "categorical", "flag": "boolean"}

    X = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["x", "y", "x", np.nan],
        "flag": [True, False, True, np.nan],
    })

    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X)
    out = prep.transform(X)

    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert not np.isnan(out).any()


def test_verify_split_artifact_detects_tampering():
    """Verify that verify_split_artifact correctly detects corrupted or mismatched split files."""
    iris_npz = PROJECT_ROOT / "data" / "splits" / "controlled" / "iris" / "fold_0.npz"
    iris_json = PROJECT_ROOT / "data" / "splits" / "controlled" / "iris" / "fold_0.json"

    assert iris_npz.exists() and iris_json.exists()
    # 1. Valid check
    assert verify_split_artifact(iris_npz, iris_json) is True

    # 2. Tampered bundle hash
    assert verify_split_artifact(iris_npz, iris_json, expected_bundle_sha256="corrupted_hash") is False


def test_phase1_canonical_immutability():
    """Verify that all 46 Phase-1 canonical bundles remain bit-identical to snapshot."""
    snapshot_path = PROJECT_ROOT / "data" / "splits" / "phase1_bundle_snapshot.json"
    assert snapshot_path.exists(), "Snapshot must exist"

    with open(snapshot_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    for k, expected_hash in snapshot.items():
        if k.startswith("whyshift_acs_"):
            parts = k.split("_")
            domain = parts[-1]
            task = parts[-2]
            p = PROJECT_ROOT / "data" / "canonical" / "natural" / "whyshift" / task / domain
        elif k.startswith("tableshift_"):
            p = PROJECT_ROOT / "data" / "canonical" / "natural" / "tableshift" / k
        else:
            p = PROJECT_ROOT / "data" / "canonical" / "controlled" / k

        fpath = p / "features.parquet"
        lpath = p / "labels.parquet"
        gpath = p / "groups.parquet"
        dpath = p / "domains.parquet"
        mpath = p / "metadata.json"

        f_sha = compute_file_sha256(fpath) if fpath.exists() else "NONE"
        l_sha = compute_file_sha256(lpath) if lpath.exists() else "NONE"
        g_sha = compute_file_sha256(gpath) if gpath.exists() else None
        d_sha = compute_file_sha256(dpath) if dpath.exists() else None
        m_sha = compute_file_sha256(mpath) if mpath.exists() else "NONE"

        bundle_sha = compute_canonical_bundle_sha256(f_sha, l_sha, g_sha, d_sha, m_sha)
        assert bundle_sha == expected_hash, f"Canonical bundle changed for {k}!"


def test_split_manifest_and_audit_completeness():
    """Verify that split_manifest.json and preprocessing_audit.csv are complete and valid."""
    manifest_path = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"
    audit_path = PROJECT_ROOT / "data" / "splits" / "preprocessing_audit.csv"

    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    assert man["total_outer_folds"] == 295
    assert man["total_inner_folds"] == 885
    assert len(man["splits"]) == 295

    df_audit = pd.read_csv(audit_path)
    assert len(df_audit) == 295
    assert not df_audit["source_has_nan"].any()
    assert not df_audit["target_has_nan"].any()
    assert df_audit["source_all_finite"].all()
    assert df_audit["target_all_finite"].all()
