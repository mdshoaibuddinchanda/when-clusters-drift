"""Tests for Gaussian Mixture Model (GMM) clustering implementation."""

import numpy as np
import pytest

from clusterdrift.methods.gmm import GMM


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(999)
    c1 = rng.normal(loc=[-2.0, 0.0], scale=0.5, size=(30, 2))
    c2 = rng.normal(loc=[2.0, 0.0], scale=0.5, size=(30, 2))
    return np.vstack([c1, c2])


def test_gmm_probability_simplex(sample_data):
    """Verify that GMM posterior probabilities satisfy row simplex constraints."""
    X = sample_data
    gmm = GMM(n_clusters=2, random_state=42)
    gmm.fit(X)

    proba = gmm.membership_
    assert proba.shape == (len(X), 2)
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()
    np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(len(X)), atol=1e-5)


def test_gmm_attributes_exposed(sample_data):
    """Verify that GMM exposes means, weights, covariances, converged, and cluster_centers_."""
    X = sample_data
    gmm = GMM(n_clusters=2, random_state=42)
    gmm.fit(X)

    assert gmm.means_ is not None
    assert gmm.cluster_centers_ is not None
    np.testing.assert_array_equal(gmm.cluster_centers_, gmm.means_)
    assert gmm.weights_ is not None
    assert gmm.weights_.shape == (2,)
    np.testing.assert_allclose(np.sum(gmm.weights_), 1.0)
    assert gmm.covariances_ is not None
    assert gmm.covariances_.shape == (2, 2, 2)
    assert gmm.converged_
    assert gmm.n_iter_ > 0


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_gmm_reproducibility(sample_data, seed):
    """Verify that GMM produces identical results for fixed algorithm seeds."""
    X = sample_data
    g1 = GMM(n_clusters=2, random_state=seed)
    g1.fit(X)

    g2 = GMM(n_clusters=2, random_state=seed)
    g2.fit(X)

    np.testing.assert_array_equal(g1.cluster_centers_, g2.cluster_centers_)
    np.testing.assert_array_equal(g1.weights_, g2.weights_)
    np.testing.assert_array_equal(g1.covariances_, g2.covariances_)
