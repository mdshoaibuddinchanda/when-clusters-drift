"""Tests for Fuzzy C-Means (FCM) clustering implementation."""

import numpy as np
import pytest

from clusterdrift.methods.fcm import FCM


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(2026)
    c1 = rng.normal(loc=[-3.0, 0.0], scale=0.4, size=(25, 2))
    c2 = rng.normal(loc=[3.0, 0.0], scale=0.4, size=(25, 2))
    return np.vstack([c1, c2])


def test_fcm_membership_simplex(sample_data):
    """Verify that final FCM memberships lie in [0, 1] and sum to 1."""
    X = sample_data
    fcm = FCM(n_clusters=2, m=2.0, random_state=1)
    fcm.fit(X)

    U = fcm.membership_
    assert (U >= 0.0).all()
    assert (U <= 1.0).all()
    np.testing.assert_allclose(np.sum(U, axis=1), np.ones(len(X)), atol=1e-5)


def test_fcm_center_update():
    """Verify prototype update equation v_k = sum_i(u_ik^m * x_i) / sum_i(u_ik^m) analytically."""
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
    ])
    # Hand-constructed memberships for K=2, m=2
    U = np.array([
        [0.8, 0.2],
        [0.6, 0.4],
        [0.1, 0.9],
    ])
    m = 2.0
    fcm = FCM(n_clusters=2, m=m)

    centers, valid = fcm._update_centers(X, U)
    assert valid

    # Calculate expected centers manually
    U_m = U ** 2  # [[0.64, 0.04], [0.36, 0.16], [0.01, 0.81]]
    # cluster 0 mass: 0.64 + 0.36 + 0.01 = 1.01
    # cluster 1 mass: 0.04 + 0.16 + 0.81 = 1.01
    expected_v0 = (0.64 * X[0] + 0.36 * X[1] + 0.01 * X[2]) / 1.01
    expected_v1 = (0.04 * X[0] + 0.16 * X[1] + 0.81 * X[2]) / 1.01

    np.testing.assert_allclose(centers[0], expected_v0, atol=1e-10)
    np.testing.assert_allclose(centers[1], expected_v1, atol=1e-10)


def test_fcm_membership_update():
    """Verify membership update equation u_ik = [sum_j (d_ik / d_ij)^(2/(m-1))]^(-1) analytically."""
    fcm = FCM(n_clusters=2, m=2.0)
    # Point at (1, 0), centers at (0, 0) and (4, 0)
    # d1 = 1.0, d2 = 3.0
    # For m=2, 2/(m-1) = 2
    # u1 = [ (1/1)^2 + (1/3)^2 ]^(-1) = [ 1 + 1/9 ]^(-1) = [10/9]^(-1) = 9/10 = 0.9
    # u2 = [ (3/1)^2 + (3/3)^2 ]^(-1) = [ 9 + 1 ]^(-1) = [10]^(-1) = 0.1
    dist = np.array([[1.0, 3.0]])
    U = fcm._compute_memberships_from_distances(dist)

    np.testing.assert_allclose(U[0, 0], 0.9, atol=1e-10)
    np.testing.assert_allclose(U[0, 1], 0.1, atol=1e-10)


def test_fcm_zero_distance_case():
    """Verify that a point coincident with a prototype gets 1.0 and 0.0 to others."""
    fcm = FCM(n_clusters=3, m=2.0)
    # Point 0 coincides with prototype 1: dist = [2.0, 0.0, 5.0]
    # Point 1 coincides with prototype 0 and 2: dist = [0.0, 3.0, 0.0]
    dist = np.array([
        [2.0, 0.0, 5.0],
        [0.0, 3.0, 0.0],
    ])
    U = fcm._compute_memberships_from_distances(dist)

    # Point 0
    np.testing.assert_allclose(U[0], [0.0, 1.0, 0.0])
    # Point 1 (coincident with 2 prototypes -> 0.5 each)
    np.testing.assert_allclose(U[1], [0.5, 0.0, 0.5])


def test_fcm_objective_monotonic_nonincreasing(sample_data):
    """Verify that the FCM objective function is monotonically non-increasing up to tolerance."""
    X = sample_data
    fcm = FCM(n_clusters=2, m=2.0, random_state=42, max_iter=50, tol=1e-8)
    fcm.fit(X)

    history = fcm.objective_history_
    assert len(history) > 2
    for t in range(1, len(history)):
        # Objective at iteration t must be <= objective at iteration t-1 + numerical tol
        assert history[t] <= history[t - 1] + 1e-6, (
            f"FCM objective increased at iteration {t}: {history[t-1]} -> {history[t]}"
        )


def test_fcm_predict_does_not_modify_centers(sample_data):
    """Verify that out-of-sample predict_membership() leaves cluster_centers_ bit-identical."""
    X = sample_data
    fcm = FCM(n_clusters=2, m=2.0, random_state=42)
    fcm.fit(X)

    centers_before = fcm.cluster_centers_.copy()

    # Out-of-sample prediction
    X_target = np.array([[-10.0, 10.0], [10.0, -10.0]])
    U_target = fcm.predict_membership(X_target)

    # Validate output
    assert U_target.shape == (2, 2)
    np.testing.assert_allclose(np.sum(U_target, axis=1), [1.0, 1.0])

    # Prototypes must be completely unchanged (bit-identical)
    np.testing.assert_array_equal(fcm.cluster_centers_, centers_before)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_fcm_reproducibility(sample_data, seed):
    """Verify deterministic reproduction across 5 algorithm seeds."""
    X = sample_data
    f1 = FCM(n_clusters=2, random_state=seed)
    f1.fit(X)

    f2 = FCM(n_clusters=2, random_state=seed)
    f2.fit(X)

    np.testing.assert_array_equal(f1.cluster_centers_, f2.cluster_centers_)
    np.testing.assert_array_equal(f1.membership_, f2.membership_)
