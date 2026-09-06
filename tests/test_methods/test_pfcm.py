"""Tests for Possibilistic Fuzzy C-Means (PFCM) clustering implementation."""

import numpy as np
import pytest

from clusterdrift.methods.pfcm import PFCM


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(777)
    c1 = rng.normal(loc=[-4.0, 0.0], scale=0.5, size=(20, 2))
    c2 = rng.normal(loc=[4.0, 0.0], scale=0.5, size=(20, 2))
    return np.vstack([c1, c2])


def test_pfcm_published_update_equations():
    """Verify Pal et al. (2005) update equations for v_k, u_ik, and t_ik analytically."""
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
    ])
    pfcm = PFCM(n_clusters=2, a=1.0, b=1.0, m=2.0, eta=2.0, k_scale=1.0)

    # Hand-chosen U, T, and centers
    U = np.array([[0.8, 0.2], [0.3, 0.7]])
    T = np.array([[0.9, 0.1], [0.4, 0.6]])

    # 1. Test Prototype update: v_k = sum_i(w_ik * x_i) / sum_i(w_ik)
    # w_ik = a*u_ik^m + b*t_ik^eta = u_ik^2 + t_ik^2
    # w_00 = 0.8^2 + 0.9^2 = 0.64 + 0.81 = 1.45
    # w_10 = 0.3^2 + 0.4^2 = 0.09 + 0.16 = 0.25
    # Mass cluster 0 = 1.45 + 0.25 = 1.70
    # expected_v0 = (1.45 * [0, 0] + 0.25 * [2, 0]) / 1.70 = [0.5 / 1.70, 0.0] = [5/17, 0.0]
    centers, valid = pfcm._update_centers(X, U, T)
    assert valid
    np.testing.assert_allclose(centers[0, 0], 5.0 / 17.0, atol=1e-10)
    np.testing.assert_allclose(centers[0, 1], 0.0, atol=1e-10)

    # 2. Test Typicality update: t_ik = [1 + (b / gamma_k * d_ik^2)^(1/(eta-1))]^(-1)
    # For b=1, eta=2, power=1: t_ik = [1 + (1 / gamma_k) * d_ik^2]^(-1)
    dist = np.array([[2.0, 4.0]])
    gamma = np.array([4.0, 16.0])
    # For point 0:
    # cluster 0: d^2 = 4, gamma = 4 => ratio = 4/4 = 1 => t_00 = 1 / (1 + 1) = 0.5
    # cluster 1: d^2 = 16, gamma = 16 => ratio = 16/16 = 1 => t_01 = 1 / (1 + 1) = 0.5
    T_computed = pfcm._compute_typicalities(dist, gamma)
    np.testing.assert_allclose(T_computed[0], [0.5, 0.5], atol=1e-10)


def test_pfcm_typicality_semantics(sample_data):
    """Verify that typicalities are in [0, 1] and are NOT constrained to sum to 1."""
    X = sample_data
    pfcm = PFCM(n_clusters=2, random_state=42)
    pfcm.fit(X)

    # Memberships must sum to 1
    U = pfcm.membership_
    np.testing.assert_allclose(np.sum(U, axis=1), np.ones(len(X)), atol=1e-5)

    # Typicalities must be in [0, 1]
    T = pfcm.typicality_
    assert (T >= 0.0).all()
    assert (T <= 1.0).all()

    # Typicalities generally do NOT sum to 1 (they are independent possibilities)
    row_sums_T = np.sum(T, axis=1)
    assert not np.allclose(row_sums_T, np.ones(len(X))), (
        "Typicalities incorrectly constrained to sum to 1!"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_pfcm_reproducibility(sample_data, seed):
    """Verify that PFCM is strictly reproducible for fixed algorithm seeds."""
    X = sample_data
    p1 = PFCM(n_clusters=2, random_state=seed)
    p1.fit(X)

    p2 = PFCM(n_clusters=2, random_state=seed)
    p2.fit(X)

    np.testing.assert_array_equal(p1.cluster_centers_, p2.cluster_centers_)
    np.testing.assert_array_equal(p1.membership_, p2.membership_)
    np.testing.assert_array_equal(p1.typicality_, p2.typicality_)


def test_pfcm_fcm_warm_start_initialization():
    """Verify that PFCM properly warm-starts from FCM and records warm-start metrics."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-4.0, 0.0], scale=0.5, size=(40, 2))
    c2 = rng.normal(loc=[4.0, 0.0], scale=0.5, size=(40, 2))
    X = np.vstack([c1, c2])

    pfcm = PFCM(n_clusters=2, random_state=42, initialization="fcm_warm_start")
    pfcm.fit(X)

    assert pfcm.initialization_method_ == "fcm_warm_start"
    assert pfcm.init_fcm_iterations_ > 0
    assert pfcm.init_fcm_objective_ is not None
    assert pfcm.warm_start_runtime_seconds_ > 0.0
    assert pfcm.gamma_ is not None
    assert len(pfcm.gamma_) == 2
    assert (pfcm.gamma_ > 0).all()


def test_pfcm_gamma_validation():
    """Verify that PFCM gamma_k exactly matches the documented formulation from known X, U, V."""
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
    ])
    # Known centers at [0, 0] and [2, 0]
    centers = np.array([[0.0, 0.0], [2.0, 0.0]])
    U = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    m = 2.0
    k_scale = 1.0
    pfcm = PFCM(n_clusters=2, m=m, k_scale=k_scale)

    gamma = pfcm._estimate_gamma(X, centers, U)

    # Independent calculation:
    # d_00^2 = 0, d_01^2 = 4
    # d_10^2 = 4, d_11^2 = 0
    # U^m = [[0.81, 0.01], [0.01, 0.81]]
    # cluster 0: sum_i u_i0^m * d_i0^2 = 0.81 * 0 + 0.01 * 4 = 0.04
    # mass_0 = 0.81 + 0.01 = 0.82
    # gamma_0 = 0.04 / 0.82 = 4 / 82 = 2 / 41
    # cluster 1: sum_i u_i1^m * d_i1^2 = 0.01 * 4 + 0.81 * 0 = 0.04
    # mass_1 = 0.01 + 0.81 = 0.82
    # gamma_1 = 0.04 / 0.82 = 2 / 41
    expected_gamma_0 = 0.04 / 0.82
    expected_gamma_1 = 0.04 / 0.82

    np.testing.assert_allclose(gamma[0], expected_gamma_0, atol=1e-10)
    np.testing.assert_allclose(gamma[1], expected_gamma_1, atol=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_pfcm_easy_balanced_synthetic_not_degenerate(seed):
    """Verify that PFCM on 3 well-separated balanced Gaussians is strictly non-degenerate and achieves ARI > 0.5."""
    from clusterdrift.metrics.external import adjusted_rand_index

    rng = np.random.default_rng(2000 + seed)
    c1 = rng.normal(loc=[-5.0, 0.0], scale=0.5, size=(100, 2))
    c2 = rng.normal(loc=[0.0, 5.0], scale=0.5, size=(100, 2))
    c3 = rng.normal(loc=[5.0, 0.0], scale=0.5, size=(100, 2))
    X = np.vstack([c1, c2, c3])
    y_true = np.array([0] * 100 + [1] * 100 + [2] * 100)

    pfcm = PFCM(n_clusters=3, random_state=seed, initialization="fcm_warm_start")
    pfcm.fit(X)

    assert pfcm.status_ == "SUCCESS"
    assert pfcm.degenerate_solution_ is False
    assert pfcm.diagnostics_["effective_distinct_prototypes"] == 3
    assert pfcm.diagnostics_["fpc_floor_gap"] > 0.2
    assert pfcm.diagnostics_["normalized_min_center_distance"] > 0.3

    ari = adjusted_rand_index(y_true, pfcm.predict(X))
    assert ari > 0.8, f"Expected clean cluster recovery, got ARI={ari}"

