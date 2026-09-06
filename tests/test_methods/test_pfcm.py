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
