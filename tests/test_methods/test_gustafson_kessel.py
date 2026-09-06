"""Tests for Gustafson-Kessel (GK) fuzzy clustering implementation."""

import numpy as np
import pytest

from clusterdrift.methods.gustafson_kessel import GustafsonKessel


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(333)
    # 2 elongated anisotropic clusters
    c1 = rng.normal(loc=[-3.0, 0.0], scale=[1.5, 0.3], size=(30, 2))
    c2 = rng.normal(loc=[3.0, 0.0], scale=[0.3, 1.5], size=(30, 2))
    return np.vstack([c1, c2])


def test_gk_metric_positive_definite(sample_data):
    """Verify that Gustafson-Kessel produces symmetric positive-definite metric matrices with det(A_k) ~ 1."""
    X = sample_data
    gk = GustafsonKessel(n_clusters=2, random_state=42)
    gk.fit(X)

    assert gk.metric_matrices_ is not None
    assert len(gk.metric_matrices_) == 2

    for k, A_k in enumerate(gk.metric_matrices_):
        # Symmetry
        np.testing.assert_allclose(A_k, A_k.T, atol=1e-8)

        # Positive definiteness (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(A_k)
        assert (eigvals > 0).all(), f"Cluster {k} metric matrix not positive definite: {eigvals}"

        # Volume constraint: det(A_k) should be ~ 1
        sign, logdet = np.linalg.slogdet(A_k)
        det_val = np.exp(logdet)
        np.testing.assert_allclose(det_val, 1.0, rtol=0.2, err_msg=f"det(A_{k}) deviates from 1: {det_val}")


def test_gk_singular_covariance_protection():
    """Verify that GK handles rank-deficient / collinear data safely without crashing."""
    rng = np.random.default_rng(555)
    # Linearly dependent 3D data (column 2 is exact 2*col 0 + 3*col 1)
    x0 = rng.normal(size=40)
    x1 = rng.normal(size=40)
    x2 = 2.0 * x0 + 3.0 * x1
    X_singular = np.column_stack([x0, x1, x2])

    gk = GustafsonKessel(n_clusters=2, random_state=42, min_eig=1e-4)
    # Should fit with regularization applied rather than crashing with SingularMatrix error
    gk.fit(X_singular)

    assert gk.cluster_centers_ is not None
    assert any(gk.regularization_applied_), "Expected regularization to be triggered on singular data."
    assert gk.status_ in ("SUCCESS", "MAX_ITER_REACHED")


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_gk_reproducibility(sample_data, seed):
    """Verify that GK is strictly reproducible for fixed algorithm seeds."""
    X = sample_data
    g1 = GustafsonKessel(n_clusters=2, random_state=seed)
    g1.fit(X)

    g2 = GustafsonKessel(n_clusters=2, random_state=seed)
    g2.fit(X)

    np.testing.assert_array_equal(g1.cluster_centers_, g2.cluster_centers_)
    np.testing.assert_array_equal(g1.membership_, g2.membership_)


def test_gk_kmeans_plusplus_initialization():
    """Verify that GustafsonKessel records kmeans++ initialization method and initial centers."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-4.0, 0.0], scale=0.5, size=(40, 2))
    c2 = rng.normal(loc=[4.0, 0.0], scale=0.5, size=(40, 2))
    X = np.vstack([c1, c2])

    gk = GustafsonKessel(n_clusters=2, random_state=42, initialization="kmeans++")
    gk.fit(X)

    assert gk.initialization_method_ == "kmeans++"
    assert gk.initial_centers_ is not None
    assert gk.initial_centers_.shape == (2, 2)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_gk_easy_balanced_synthetic_not_degenerate(seed):
    """Verify that GK on 3 well-separated balanced Gaussians is strictly non-degenerate and achieves ARI > 0.5."""
    from clusterdrift.metrics.external import adjusted_rand_index

    rng = np.random.default_rng(3000 + seed)
    c1 = rng.normal(loc=[-5.0, 0.0], scale=0.5, size=(100, 2))
    c2 = rng.normal(loc=[0.0, 5.0], scale=0.5, size=(100, 2))
    c3 = rng.normal(loc=[5.0, 0.0], scale=0.5, size=(100, 2))
    X = np.vstack([c1, c2, c3])
    y_true = np.array([0] * 100 + [1] * 100 + [2] * 100)

    gk = GustafsonKessel(n_clusters=3, random_state=seed, initialization="kmeans++")
    gk.fit(X)

    assert gk.status_ == "SUCCESS"
    assert gk.degenerate_solution_ is False
    assert gk.diagnostics_["effective_distinct_prototypes"] == 3
    assert gk.diagnostics_["fpc_floor_gap"] > 0.2
    assert gk.diagnostics_["normalized_min_center_distance"] > 0.3

    ari = adjusted_rand_index(y_true, gk.predict(X))
    assert ari > 0.8, f"Expected clean cluster recovery, got ARI={ari}"

