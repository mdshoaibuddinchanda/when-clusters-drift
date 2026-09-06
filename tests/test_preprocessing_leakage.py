"""Rigorous test suite for Phase 2.1 zero-leakage, split reproducibility, and preprocessing isolation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from clusterdrift.data.folds import (
    compute_preprocessing_config_sha256,
    compute_scenario_input_sha256,
    compute_split_hash,
    compute_split_protocol_sha256,
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


# ---------------------------------------------------------------------------
# Item 1: Categorical & Ordinal Missing Handling
# ---------------------------------------------------------------------------

def test_categorical_nan_is_actually_imputed():
    """Verify that categorical NaN is imputed to source mode rather than encoded as literal 'nan'."""
    cfg = {"categorical": {"imputer": "most_frequent", "encoder": "onehot", "handle_unknown": "ignore", "min_frequency": 0.0}}
    roles = {"cat": "categorical"}

    # Source has mode 'A'
    X_source = pd.DataFrame({"cat": ["A", "A", "B"]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    # Target has a true NaN
    X_target = pd.DataFrame({"cat": [np.nan]})
    out_tgt = prep.transform(X_target)
    out_src_A = prep.transform(pd.DataFrame({"cat": ["A"]}))

    # Transformed NaN must be identical to transformed 'A'
    np.testing.assert_array_equal(out_tgt, out_src_A)


def test_categorical_none_is_actually_imputed():
    """Verify that categorical None is imputed to source mode rather than encoded as literal 'None'."""
    cfg = {"categorical": {"imputer": "most_frequent", "encoder": "onehot", "handle_unknown": "ignore", "min_frequency": 0.0}}
    roles = {"cat": "categorical"}

    X_source = pd.DataFrame({"cat": ["X", "X", "Y"]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    X_target = pd.DataFrame({"cat": [None]})
    out_tgt = prep.transform(X_target)
    out_src_X = prep.transform(pd.DataFrame({"cat": ["X"]}))

    np.testing.assert_array_equal(out_tgt, out_src_X)


def test_missing_marker_not_encoded_as_literal_category():
    """Verify that missing values in source do not create literal 'nan', 'None', '<NA>' category columns."""
    cfg = {"categorical": {"imputer": "most_frequent", "encoder": "onehot", "handle_unknown": "ignore", "min_frequency": 0.0}}
    roles = {"cat": "categorical"}

    X_source = pd.DataFrame({"cat": ["apple", "apple", "banana", np.nan, None]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    feature_names = prep.get_feature_names_out()
    for name in feature_names:
        assert not name.lower().endswith("_nan"), f"Literal nan category found: {name}"
        assert not name.lower().endswith("_none"), f"Literal none category found: {name}"
        assert not name.lower().endswith("_<na>"), f"Literal <na> category found: {name}"


def test_ordinal_missing_is_actually_imputed():
    """Verify that ordinal missing values are imputed to source mode before ordinal encoding."""
    cfg = {"ordinal": {"policy": "preserve_order"}}
    roles = {"stage": "ordinal"}
    meta = {"category_orderings": {"stage": ["low", "medium", "high"]}}

    # Source has mode 'medium'
    X_source = pd.DataFrame({"stage": ["medium", "medium", "high"]})
    prep = build_preprocessor(feature_roles=roles, config=cfg, metadata=meta)
    prep.fit(X_source)

    X_target = pd.DataFrame({"stage": [np.nan]})
    out_tgt = prep.transform(X_target)
    out_src_med = prep.transform(pd.DataFrame({"stage": ["medium"]}))

    np.testing.assert_array_equal(out_tgt, out_src_med)


# ---------------------------------------------------------------------------
# Item 2: All-Missing Feature Preservation
# ---------------------------------------------------------------------------

def test_all_missing_numeric_feature_preserved():
    """Verify that an entirely-missing numeric column in source is preserved as exactly 1 output feature."""
    cfg = {"numeric": {"imputer": "median", "scaler": "standard", "keep_empty_features": True, "all_missing_fallback": 0.0}}
    roles = {"num": "numeric"}

    X_source = pd.DataFrame({"num": [np.nan, np.nan, np.nan]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    assert prep.get_feature_names_out() == ["num"]
    out_src = prep.transform(X_source)
    assert out_src.shape == (3, 1)
    assert not np.isnan(out_src).any()
    assert np.all(out_src == 0.0)


def test_all_missing_categorical_feature_preserved():
    """Verify that an entirely-missing categorical column uses __MISSING_SOURCE__ and preserves 1 feature."""
    cfg = {"categorical": {"imputer": "most_frequent", "encoder": "onehot", "handle_unknown": "ignore", "all_missing_sentinel": "__MISSING_SOURCE__"}}
    roles = {"cat": "categorical"}

    X_source = pd.DataFrame({"cat": [np.nan, None, np.nan]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    feature_names = prep.get_feature_names_out()
    assert len(feature_names) == 1
    assert "__MISSING_SOURCE__" in feature_names[0]
    out_src = prep.transform(X_source)
    assert out_src.shape == (3, 1)
    assert not np.isnan(out_src).any()


def test_all_missing_boolean_feature_preserved():
    """Verify that an entirely-missing boolean column preserves 1 output feature with fallback False (0.0)."""
    cfg = {"boolean": {"imputer": "most_frequent", "encoding": "zero_one", "all_missing_fallback": False}}
    roles = {"flag": "boolean"}

    X_source = pd.DataFrame({"flag": [np.nan, None, np.nan]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    assert prep.get_feature_names_out() == ["flag"]
    out_src = prep.transform(X_source)
    assert out_src.shape == (3, 1)
    assert np.all(out_src == 0.0)


def test_all_missing_feature_output_dimension_stable():
    """Verify that combining multiple all-missing features maintains exact output dimensions across source and target."""
    cfg = {
        "output_dtype": "float32",
        "numeric": {"imputer": "median", "scaler": "standard", "all_missing_fallback": 0.0},
        "categorical": {"imputer": "most_frequent", "encoder": "onehot", "handle_unknown": "ignore", "all_missing_sentinel": "__MISSING_SOURCE__"},
        "boolean": {"imputer": "most_frequent", "encoding": "zero_one", "all_missing_fallback": False},
    }
    roles = {"num": "numeric", "cat": "categorical", "bool": "boolean"}

    X_source = pd.DataFrame({"num": [np.nan, np.nan], "cat": [None, np.nan], "bool": [np.nan, None]})
    X_target = pd.DataFrame({"num": [10.0, np.nan], "cat": ["val", None], "bool": [True, np.nan]})

    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_source)

    out_src = prep.transform(X_source)
    out_tgt = prep.transform(X_target)

    assert out_src.shape == (2, 3)
    assert out_tgt.shape == (2, 3)
    assert np.all(np.isfinite(out_src))
    assert np.all(np.isfinite(out_tgt))


# ---------------------------------------------------------------------------
# Items 3, 4, 5: Configuration as Single Truth & Protocol Hashing
# ---------------------------------------------------------------------------

def test_config_seed_change_modifies_splits_deterministically():
    """Verify that changing the split seed in configuration alters split indices deterministically."""
    s1 = generate_kfold_splits(n_samples=100, n_outer=5, n_inner=3, split_seed=20260907)
    s2 = generate_kfold_splits(n_samples=100, n_outer=5, n_inner=3, split_seed=99999999)

    # Different seeds must produce different fold indices
    assert not np.array_equal(s1[0].source_indices, s2[0].source_indices)

    # Re-running with the same seed must produce identical indices
    s1_repeat = generate_kfold_splits(n_samples=100, n_outer=5, n_inner=3, split_seed=20260907)
    np.testing.assert_array_equal(s1[0].source_indices, s1_repeat[0].source_indices)


def test_split_protocol_change_invalidates_stale_splits():
    """Verify that changing split protocol SHA invalidates existing split artifacts."""
    iris_npz = PROJECT_ROOT / "data" / "splits" / "controlled" / "iris" / "fold_0.npz"
    iris_json = PROJECT_ROOT / "data" / "splits" / "controlled" / "iris" / "fold_0.json"
    assert iris_npz.exists() and iris_json.exists()

    with open(iris_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    current_proto_sha = meta["split_protocol_sha256"]
    # Matching protocol verifies
    assert verify_split_artifact(iris_npz, iris_json, expected_protocol_sha256=current_proto_sha) is True
    # Altered protocol rejects
    assert verify_split_artifact(iris_npz, iris_json, expected_protocol_sha256="stale_protocol_hash_12345") is False


def test_preprocessing_only_change_preserves_splits_updates_audit():
    """Verify that changing preprocessing options leaves split indices unchanged while altering config hash."""
    cfg1 = {
        "protocol_version": 1,
        "splitting": {"outer_folds": 5, "inner_folds": 3, "split_seed": 20260907},
        "preprocessing": {"categorical": {"min_frequency": 0.01}},
    }
    cfg2 = {
        "protocol_version": 1,
        "splitting": {"outer_folds": 5, "inner_folds": 3, "split_seed": 20260907},
        "preprocessing": {"categorical": {"min_frequency": 0.05}},
    }

    proto1 = compute_split_protocol_sha256(cfg1)
    proto2 = compute_split_protocol_sha256(cfg2)
    assert proto1 == proto2, "Split protocol SHA must not change when only preprocessing changes."

    prep_sha1 = compute_preprocessing_config_sha256(cfg1)
    prep_sha2 = compute_preprocessing_config_sha256(cfg2)
    assert prep_sha1 != prep_sha2, "Preprocessing config SHA must update when preprocessing changes."


# ---------------------------------------------------------------------------
# Item 6: Strictly Read-Only --verify
# ---------------------------------------------------------------------------

def test_verify_mode_is_byte_read_only():
    """Verify that running python scripts/02_generate_folds.py --verify modifies zero files on disk."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    target_files = [
        splits_dir / "phase2_input_lock.json",
        splits_dir / "split_manifest.json",
        splits_dir / "preprocessing_audit.csv",
        splits_dir / "controlled" / "iris" / "fold_0.json",
        splits_dir / "controlled" / "iris" / "fold_0.npz",
    ]

    hashes_before = {f: compute_file_sha256(f) for f in target_files if f.exists()}

    # Execute --verify
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "02_generate_folds.py"), "--verify", "--dataset", "iris"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"--verify failed with output: {res.stderr}\n{res.stdout}"

    hashes_after = {f: compute_file_sha256(f) for f in target_files if f.exists()}

    assert hashes_before == hashes_after, "Files were modified during read-only --verify mode!"


# ---------------------------------------------------------------------------
# Item 9: Electricity Lexicographic Chronology
# ---------------------------------------------------------------------------

def test_electricity_strict_date_period_chronology():
    """Verify that Electricity temporal splits satisfy strict (date, period) tuple ordering."""
    elec_path = PROJECT_ROOT / "data" / "canonical" / "controlled" / "electricity" / "features.parquet"
    assert elec_path.exists()
    df_elec = pd.read_parquet(elec_path)

    splits = generate_temporal_block_splits(features_df=df_elec, time_columns=["date", "period"], n_outer=5, n_inner=3)

    time_cols = ["date", "period"]
    for s in splits:
        src_keys = [tuple(df_elec.loc[i, time_cols]) for i in s.source_indices]
        tgt_keys = [tuple(df_elec.loc[i, time_cols]) for i in s.target_indices]
        assert max(src_keys) < min(tgt_keys), f"Outer temporal leakage in fold {s.outer_fold}"

        for inf in s.inner_folds:
            in_tr_keys = [tuple(df_elec.loc[i, time_cols]) for i in inf.train_indices]
            in_val_keys = [tuple(df_elec.loc[i, time_cols]) for i in inf.val_indices]
            assert max(in_tr_keys) < min(in_val_keys), f"Inner temporal leakage in fold {s.outer_fold}, inner {inf.inner_fold}"


# ---------------------------------------------------------------------------
# Item 11: Inner-Train-Only Preprocessing Isolation
# ---------------------------------------------------------------------------

def test_inner_preprocessing_fits_inner_train_only():
    """Verify that inner preprocessing fits only on inner_train and is completely unaffected by inner_val."""
    cfg = {
        "numeric": {"imputer": "median", "scaler": "standard"},
        "categorical": {"imputer": "most_frequent", "encoder": "onehot", "handle_unknown": "ignore", "min_frequency": 0.0},
    }
    roles = {"f1": "numeric", "cat": "categorical"}

    iris_npz = PROJECT_ROOT / "data" / "splits" / "controlled" / "iris" / "fold_0.npz"
    with np.load(iris_npz) as npz:
        in_train_idx = npz["inner_0_train"]
        in_val_idx = npz["inner_0_val"]

    # Synthesize dataset matching indices
    N = max(np.max(in_train_idx), np.max(in_val_idx)) + 1
    df = pd.DataFrame({
        "f1": np.linspace(1.0, 10.0, N),
        "cat": ["A" if i % 2 == 0 else "B" for i in range(N)],
    })

    X_train = df.iloc[in_train_idx]
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_train)

    mean_before = prep.numeric_scaler.mean_.copy()
    scale_before = prep.numeric_scaler.scale_.copy()
    vocab_before = prep.source_category_vocabularies_["cat"].copy()

    # Corrupt inner_val with extreme numeric outlier and unseen category
    X_val_corrupted = df.iloc[in_val_idx].copy()
    X_val_corrupted.iloc[0, 0] = 1e12
    X_val_corrupted.iloc[0, 1] = "UNSEEN_VAL_CATEGORY"

    out_val = prep.transform(X_val_corrupted)

    # Parameters and source-fitted state must remain 100% unchanged
    np.testing.assert_array_equal(prep.numeric_scaler.mean_, mean_before)
    np.testing.assert_array_equal(prep.numeric_scaler.scale_, scale_before)
    assert prep.source_category_vocabularies_["cat"] == vocab_before
    assert np.all(np.isfinite(out_val))


# ---------------------------------------------------------------------------
# Items 12 & 13: Hardened Boolean Mapping & Infinity Rejection
# ---------------------------------------------------------------------------

def test_boolean_encoding_and_invalid_value_rejection():
    """Verify that boolean encoding accepts canonical representations and raises on unknown values."""
    cfg = {"boolean": {"imputer": "most_frequent", "encoding": "zero_one"}}
    roles = {"flag": "boolean"}

    # Valid inputs
    X_valid = pd.DataFrame({"flag": [True, False, 1, 0, "true", "false", "yes", "no", np.nan]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X_valid)
    out = prep.transform(X_valid)
    assert out.shape == (9, 1)
    assert not np.isnan(out).any()

    # Invalid input
    X_invalid = pd.DataFrame({"flag": ["maybe", "unknown", 42]})
    prep_invalid = build_preprocessor(feature_roles=roles, config=cfg)
    with pytest.raises(ValueError, match="Invalid"):
        prep_invalid.fit(X_invalid)


def test_infinite_values_fail_loudly():
    """Verify that infinite values in raw input or transformed output raise an explicit ValueError."""
    cfg = {"numeric": {"imputer": "median", "scaler": "standard"}}
    roles = {"val": "numeric"}

    X_inf = pd.DataFrame({"val": [1.0, 2.0, np.inf]})
    prep = build_preprocessor(feature_roles=roles, config=cfg)

    with pytest.raises(ValueError, match="Infinite values"):
        prep.fit(X_inf)


# ---------------------------------------------------------------------------
# Items 14-22: Baseline Phase 2 Split, Manifest, and Provenance Integrity
# ---------------------------------------------------------------------------

def test_split_index_reproducibility():
    """Verify that calling split generation twice produces bit-identical indices and hashes."""
    s1 = generate_kfold_splits(n_samples=200, n_outer=5, n_inner=3, split_seed=20260907)
    s2 = generate_kfold_splits(n_samples=200, n_outer=5, n_inner=3, split_seed=20260907)

    for fold1, fold2 in zip(s1, s2):
        np.testing.assert_array_equal(fold1.source_indices, fold2.source_indices)
        np.testing.assert_array_equal(fold1.target_indices, fold2.target_indices)


def test_source_target_disjointness_across_all_manifest_splits():
    """Verify that all manifest splits have disjoint source and target index sets."""
    manifest_path = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    for entry in doc["splits"]:
        npz_path = PROJECT_ROOT / entry["npz_path"]
        with np.load(npz_path) as npz:
            src = npz["source_indices"]
            tgt = npz["target_indices"]
            if "whyshift" not in entry["dataset_id"]:
                assert len(np.intersect1d(src, tgt)) == 0


def test_group_isolation_mice_and_har():
    """Verify that mice and subjects never cross split boundaries."""
    mice_groups_path = PROJECT_ROOT / "data" / "canonical" / "controlled" / "mice_protein_expression" / "groups.parquet"
    mice_groups = pd.read_parquet(mice_groups_path).iloc[:, 0].values
    mice_splits = generate_group_kfold_splits(groups=mice_groups, n_outer=5, n_inner=3, group_column_name="mouse_subject_id")
    for s in mice_splits:
        assert set(mice_groups[s.source_indices]).isdisjoint(set(mice_groups[s.target_indices]))

    har_groups_path = PROJECT_ROOT / "data" / "canonical" / "controlled" / "human_activity_recognition" / "groups.parquet"
    har_groups = pd.read_parquet(har_groups_path).iloc[:, 0].values
    har_splits = generate_group_kfold_splits(groups=har_groups, n_outer=5, n_inner=3, group_column_name="subject")
    for s in har_splits:
        assert set(har_groups[s.source_indices]).isdisjoint(set(har_groups[s.target_indices]))


def test_inner_fold_isolation_and_no_target_leakage():
    """Verify inner fold isolation and zero target overlap."""
    splits = generate_kfold_splits(n_samples=200, n_outer=5, n_inner=3, split_seed=20260907)
    for s in splits:
        src = set(s.source_indices)
        tgt = set(s.target_indices)
        for inf in s.inner_folds:
            tr = set(inf.train_indices)
            val = set(inf.val_indices)
            assert tr.isdisjoint(val)
            assert tr.issubset(src) and val.issubset(src)
            assert tr.isdisjoint(tgt) and val.isdisjoint(tgt)


def test_phase1_canonical_immutability():
    """Verify that all 46 Phase-1 canonical bundles remain bit-identical to snapshot."""
    snapshot_path = PROJECT_ROOT / "data" / "splits" / "phase1_bundle_snapshot.json"
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

        f_sha = compute_file_sha256(p / "features.parquet") if (p / "features.parquet").exists() else "NONE"
        l_sha = compute_file_sha256(p / "labels.parquet") if (p / "labels.parquet").exists() else "NONE"
        g_sha = compute_file_sha256(p / "groups.parquet") if (p / "groups.parquet").exists() else None
        d_sha = compute_file_sha256(p / "domains.parquet") if (p / "domains.parquet").exists() else None
        m_sha = compute_file_sha256(p / "metadata.json") if (p / "metadata.json").exists() else "NONE"

        bundle_sha = compute_canonical_bundle_sha256(f_sha, l_sha, g_sha, d_sha, m_sha)
        assert bundle_sha == expected_hash, f"Canonical bundle mutated for {k}!"


def test_split_manifest_and_audit_completeness():
    """Verify that split_manifest.json and preprocessing_audit.csv are complete with all 23 columns."""
    manifest_path = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"
    audit_path = PROJECT_ROOT / "data" / "splits" / "preprocessing_audit.csv"

    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    assert man["total_outer_folds"] == 295
    assert man["total_inner_folds"] == 885
    assert len(man["splits"]) == 295

    df_audit = pd.read_csv(audit_path)
    assert len(df_audit) == 295
    assert df_audit.shape[1] == 23

    # Check 0 NaNs after preprocessing
    assert (df_audit["source_missing_after"] == 0).all()
    assert (df_audit["target_missing_after"] == 0).all()
    assert df_audit["source_all_finite"].all()
    assert df_audit["target_all_finite"].all()
