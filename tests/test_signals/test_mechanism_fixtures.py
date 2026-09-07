"""Targeted tests executing all 9 Phase 6 mechanism verification fixtures."""

from pathlib import Path
import numpy as np
import pytest

from clusterdrift.alignment.costs import compute_reference_cluster_scales
from clusterdrift.alignment.hungarian import align_clusters
from clusterdrift.probes.evaluation import ProbeEvaluation, ProbeIdentityMismatchError, verify_probe_identity
from clusterdrift.signals.covariate import (
    compute_mmd_b2,
    derive_source_median_bandwidth,
)
from clusterdrift.signals.entropy import compute_entropy_shift
from clusterdrift.signals.mass import compute_cluster_mass_shift
from clusterdrift.signals.membership import (
    compute_current_membership_drift,
    compute_historical_membership_drift,
)
from clusterdrift.signals.prototype import compute_prototype_movement


def test_fixture_1_identity_zero():
    """Fixture 1: Identical representations produce zero structural signals."""
    V0 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    U0 = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    scales = np.array([1.0, 1.0], dtype=np.float64)

    dur = compute_historical_membership_drift(U0, U0)["D_U_R"]
    duc = compute_current_membership_drift(U0, U0)["D_U_C"]
    dv = compute_prototype_movement(V0, V0, scales)["D_V"]
    dh = compute_entropy_shift(U0, U0)["D_H"]
    dm = compute_cluster_mass_shift(U0, U0)["D_M"]
    dx, _ = compute_mmd_b2(V0, V0, sigma=1.0)

    assert dur == 0.0
    assert duc == 0.0
    assert dv == 0.0
    assert dh == 0.0
    assert dm == 0.0
    assert dx < 1e-10


def test_fixture_2_permutation_invariance():
    """Fixture 2: Pure cluster-ID permutation is strictly resolved by alignment."""
    V0 = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    U0 = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    scales = np.array([1.0, 1.0], dtype=np.float64)

    perm = [1, 0]
    Vt_perm = V0[perm]
    Ut_perm = U0[:, perm]

    align_res = align_clusters(centers_ref=V0, centers_cand=Vt_perm, scales_ref=scales, U_ref=U0, U_cand=Ut_perm, eta=0.50)
    assert align_res.permutation.tolist() == [1, 0]

    U_aligned = align_res.apply_to_memberships(Ut_perm)
    V_aligned = align_res.aligned_centers

    dur = compute_historical_membership_drift(U0, U_aligned)["D_U_R"]
    dv = compute_prototype_movement(V0, V_aligned, scales)["D_V"]
    dm = compute_cluster_mass_shift(U0, U_aligned)["D_M"]

    assert np.isclose(dur, 0.0, atol=1e-12)
    assert np.isclose(dv, 0.0, atol=1e-12)
    assert np.isclose(dm, 0.0, atol=1e-12)


def test_fixture_3_center_movement_positive():
    """Fixture 3: Displaced prototype produces hand-verifiable positive D_V."""
    V0 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    Vt = np.array([[0.5, 0.0], [1.0, 1.0]], dtype=np.float64)
    scales = np.array([1.0, 1.0], dtype=np.float64)

    dv = compute_prototype_movement(V0, Vt, scales)["D_V"]
    assert np.isclose(dv, 0.25, atol=1e-6)


def test_fixture_4_entropy_change_positive():
    """Fixture 4: Softening crisp memberships yields positive D_H."""
    U_crisp = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    U_soft = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)

    dh = compute_entropy_shift(U_crisp, U_soft)["D_H"]
    assert dh > 0.0


def test_fixture_5_mass_change_positive():
    """Fixture 5: Reallocating population masses yields positive D_M."""
    U1 = np.array([[0.9, 0.1], [0.9, 0.1]], dtype=np.float64)
    U2 = np.array([[0.1, 0.9], [0.1, 0.9]], dtype=np.float64)

    dm = compute_cluster_mass_shift(U1, U2)["D_M"]
    assert dm > 0.5


def test_fixture_6_raw_shift_without_structure():
    """Fixture 6: Shift in feature space alters D_X without changing structural signals."""
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    Y = X + 4.0
    U0 = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)

    dx, _ = compute_mmd_b2(X, Y, sigma=1.0)
    dur = compute_historical_membership_drift(U0, U0)["D_U_R"]
    duc = compute_current_membership_drift(U0, U0)["D_U_C"]

    assert dx > 0.5
    assert dur == 0.0
    assert duc == 0.0


def test_fixture_7_probe_identity_guard():
    """Fixture 7: Comparing evaluations with different bank hashes raises ProbeIdentityMismatchError."""
    U = np.array([[0.5, 0.5]])
    pe1 = ProbeEvaluation("bank_A", "current", "iris", 0, "clean", "fp1", U, 1, 2)
    pe2 = ProbeEvaluation("bank_B", "current", "iris", 0, "clean", "fp2", U, 1, 2)

    with pytest.raises(ProbeIdentityMismatchError):
        verify_probe_identity(pe1, pe2)


def test_fixture_8_mmd_chunk_equivalence():
    """Fixture 8: Chunked MMD equals unchunked to numerical precision."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    Y = rng.randn(80, 4) + 1.0

    v_small, _ = compute_mmd_b2(X, Y, sigma=1.5, chunk_size=16)
    v_large, _ = compute_mmd_b2(X, Y, sigma=1.5, chunk_size=512)
    assert np.isclose(v_small, v_large, atol=1e-12)


def test_fixture_9_source_only_bandwidth():
    """Fixture 9: Source-only bandwidth is invariant across repeated derivations on reference bank."""
    rng = np.random.RandomState(99)
    A_R = rng.randn(150, 5)

    s1, st1 = derive_source_median_bandwidth(A_R, "ds1", 0, "hash_ref", global_signal_seed=2026090706)
    s2, st2 = derive_source_median_bandwidth(A_R, "ds1", 0, "hash_ref", global_signal_seed=2026090706)

    assert s1 == s2
    assert st1 == "SUCCESS"
