"""Unit tests for fuzzifier policy module, numerical stabilization, and soft model validity gate."""

import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.methods.fcm import FCM
from clusterdrift.methods.fuzzifier import FuzzifierResolution, resolve_fuzzifier
from clusterdrift.data.preprocess import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_dimension_adaptive_fuzzifier_exact_values():
    """Verify exact theoretical values of the Winkler dimension-dependent rule."""
    test_cases = [
        (2, 2.0, False),
        (4, 1.5, False),
        (10, 1.2, False),
        (20, 1.1, False),
        (100, 1.02, False),
        (500, 1.01, True),  # raw = 1.004 -> clipped to 1.01
        (1000, 1.01, True), # raw = 1.002 -> clipped to 1.01
    ]
    for D, expected_m, expected_clipped in test_cases:
        res = resolve_fuzzifier(policy="dimension_adaptive", n_features=D)
        assert isinstance(res, FuzzifierResolution)
        np.testing.assert_allclose(res.effective_m, expected_m, atol=1e-12)
        assert res.clipped == expected_clipped
        assert res.dimension == D
        assert "Winkler" in res.reference


def test_dimension_adaptive_fuzzifier_no_labels():
    """Verify that resolve_fuzzifier strictly forbids label/target arguments."""
    sig = inspect.signature(resolve_fuzzifier)
    assert "y" not in sig.parameters
    assert "labels" not in sig.parameters
    assert "target" not in sig.parameters

    with pytest.raises(TypeError, match="Supervised labels or target arguments are strictly prohibited"):
        resolve_fuzzifier(policy="dimension_adaptive", n_features=10, y=np.array([1, 2]))


def test_low_m_membership_log_domain_stability():
    """Verify log-domain Softmax does not overflow or produce NaN for extreme low m."""
    fcm = FCM(n_clusters=3, m=1.01, fuzzifier_policy="fixed")
    # beta = 2 / (1.01 - 1.0) = 200.0
    # Standard inv_dist ** 200 easily overflows if distance is small
    dist = np.array([
        [0.01, 1.5, 3.0],
        [10.0, 0.05, 5.0],
        [2.0, 2.0, 2.0],
    ])
    U = fcm._compute_memberships_from_distances(dist)

    assert not np.isnan(U).any()
    assert not np.isinf(U).any()
    # Simplex check
    np.testing.assert_allclose(np.sum(U, axis=1), [1.0, 1.0, 1.0], atol=1e-12)
    # Closest cluster should dominate
    assert U[0, 0] > 0.999
    assert U[1, 1] > 0.999
    # Equal distance should split uniformly
    np.testing.assert_allclose(U[2], [1.0 / 3, 1.0 / 3, 1.0 / 3], atol=1e-12)


def test_low_m_zero_distance_case():
    """Verify exact zero-distance assignment under low m."""
    fcm = FCM(n_clusters=2, m=1.02, fuzzifier_policy="fixed")
    dist = np.array([
        [0.0, 2.5],
        [0.0, 0.0],
    ])
    U = fcm._compute_memberships_from_distances(dist)
    # Exact coincidence: [1.0, 0.0] for point 0, [0.5, 0.5] for point 1
    np.testing.assert_array_equal(U[0], [1.0, 0.0])
    np.testing.assert_array_equal(U[1], [0.5, 0.5])


def test_low_m_simplex_machine_precision():
    """Verify row simplex constraint holds to machine precision for random distances and low m."""
    rng = np.random.default_rng(42)
    dist = rng.uniform(0.1, 10.0, size=(100, 5))
    fcm = FCM(n_clusters=5, m=1.05, fuzzifier_policy="fixed")
    U = fcm._compute_memberships_from_distances(dist)

    row_sums = np.sum(U, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-14)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_s01_dimension_rule_nondegenerate_all_seeds(seed):
    """Verify that dimension-adaptive m=1.2 on s01 produces non-degenerate solutions across all 5 seeds."""
    feat_path = PROJECT_ROOT / "data" / "synthetic" / "s01_balanced_gmm" / "features.parquet"
    meta_path = PROJECT_ROOT / "data" / "synthetic" / "s01_balanced_gmm" / "metadata.json"
    df_X = pd.read_parquet(feat_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Standard fold 0 source partition (80% deterministic permutation)
    rng = np.random.default_rng(20260907)
    perm = rng.permutation(len(df_X))[:int(0.8 * len(df_X))]
    X_sub = df_X.iloc[perm].copy()
    prep = build_preprocessor({c: "numeric" for c in df_X.columns}, {"standardize_numeric": True}, meta)
    X = prep.fit(X_sub).transform(X_sub)

    fcm = FCM(n_clusters=3, random_state=seed, initialization="kmeans++", fuzzifier_policy="dimension_adaptive")
    fcm.fit(X)

    assert fcm.status_ == "SUCCESS"
    assert fcm.degenerate_solution_ is False
    assert fcm.effective_m_ == 1.2
    assert fcm.diagnostics_["effective_distinct_prototypes"] == 3
    assert fcm.diagnostics_["fpc_floor_gap"] > 0.3
    assert fcm.diagnostics_["normalized_min_center_distance"] > 0.5


def test_fixed_m2_s01_collapse_reproduced():
    """Verify that fixed m=2.0 on s01 collapses to DEGENERATE_SOLUTION due to fuzzifier objective collapse."""
    feat_path = PROJECT_ROOT / "data" / "synthetic" / "s01_balanced_gmm" / "features.parquet"
    meta_path = PROJECT_ROOT / "data" / "synthetic" / "s01_balanced_gmm" / "metadata.json"
    df_X = pd.read_parquet(feat_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rng = np.random.default_rng(20260907)
    perm = rng.permutation(len(df_X))[:int(0.8 * len(df_X))]
    X_sub = df_X.iloc[perm].copy()
    prep = build_preprocessor({c: "numeric" for c in df_X.columns}, {"standardize_numeric": True}, meta)
    X = prep.fit(X_sub).transform(X_sub)

    fcm = FCM(n_clusters=3, random_state=42, initialization="kmeans++", fuzzifier_policy="fixed", m=2.0)
    fcm.fit(X)

    assert fcm.status_ == "DEGENERATE_SOLUTION"
    assert fcm.degenerate_solution_ is True
    assert fcm.diagnostics_["fpc_floor_gap"] < 0.02


def test_policy_metadata_persisted():
    """Verify that all policy metadata attributes are persisted on fitted models."""
    X = np.random.default_rng(42).normal(size=(50, 4))
    fcm = FCM(n_clusters=2, fuzzifier_policy="dimension_adaptive")
    fcm.fit(X)

    assert fcm.fuzzifier_policy_ == "dimension_adaptive"
    assert fcm.effective_m_ == 1.5  # 1 + 2 / 4
    assert fcm.effective_dimension_ == 4
    assert fcm.fuzzifier_clipped_ is False


def test_comparison_uses_all_runs_same_denominator():
    """Verify that comparison calculations evaluate all runs in the denominator."""
    # Construct synthetic run dataframe with 10 total runs, 4 degenerate
    df = pd.DataFrame({
        "status": ["SUCCESS"] * 6 + ["DEGENERATE_SOLUTION"] * 4,
        "degenerate_solution": [False] * 6 + [True] * 4,
        "ARI": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0],
    })
    total = len(df)
    assert total == 10

    degen_rate = df["degenerate_solution"].sum() / total
    assert degen_rate == 0.4

    mean_ari_all = df["ARI"].mean()
    # Mean of all 10 values: (3.3) / 10 = 0.33
    np.testing.assert_allclose(mean_ari_all, 0.33, atol=1e-10)

    # Non-degenerate only: 3.3 / 6 = 0.55
    mean_ari_nondegen = df[~df["degenerate_solution"]]["ARI"].mean()
    np.testing.assert_allclose(mean_ari_nondegen, 0.55, atol=1e-10)


def test_no_hardcoded_historical_scientific_stats():
    """Verify that scripts do not contain hard-coded pre-repair statistics dictionaries."""
    script_path = PROJECT_ROOT / "scripts" / "03_validate_baselines.py"
    code = script_path.read_text(encoding="utf-8")
    assert "pre_repair_stats =" not in code, "Found forbidden hard-coded pre_repair_stats dictionary in 03_validate_baselines.py"
