"""Tests for KMeans clustering wrapper."""

import numpy as np
import pytest

from clusterdrift.methods.kmeans import KMeans


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(12345)
    return rng.normal(size=(50, 3))


def test_kmeans_explicit_parameters():
    """Verify that KMeans initializes with explicit locked parameters."""
    km = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300, tol=1e-4)
    assert km.n_clusters == 4
    assert km.random_state == 42
    assert km.n_init == 10
    assert km.max_iter == 300
    assert km.tol == 1e-4
    assert km.membership_semantics == "hard_one_hot"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_kmeans_reproducibility(sample_data, seed):
    """Verify that KMeans produces bit-identical cluster centers for the same algorithm seed."""
    X = sample_data
    km1 = KMeans(n_clusters=3, random_state=seed, n_init=5)
    km1.fit(X)

    km2 = KMeans(n_clusters=3, random_state=seed, n_init=5)
    km2.fit(X)

    np.testing.assert_array_equal(km1.cluster_centers_, km2.cluster_centers_)
    np.testing.assert_array_equal(km1.membership_, km2.membership_)
    assert km1.inertia_ == km2.inertia_


def test_kmeans_out_of_sample_predict(sample_data):
    """Verify KMeans predictions on out-of-sample data do not alter prototypes."""
    X = sample_data
    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X)

    centers_before = km.cluster_centers_.copy()
    X_target = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    preds = km.predict(X_target)
    memberships = km.predict_membership(X_target)

    assert preds.shape == (2,)
    assert memberships.shape == (2, 3)
    np.testing.assert_array_equal(km.cluster_centers_, centers_before)
