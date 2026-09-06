"""Tests validating the common unsupervised clustering interface across all 5 methods."""

import numpy as np
import pytest

from clusterdrift.methods.base import BaseClusteringMethod
from clusterdrift.methods.fcm import FCM
from clusterdrift.methods.gmm import GMM
from clusterdrift.methods.gustafson_kessel import GustafsonKessel
from clusterdrift.methods.kmeans import KMeans
from clusterdrift.methods.pfcm import PFCM


@pytest.fixture
def sample_data():
    """Deterministic 2D synthetic dataset with 3 clear clusters."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(30, 2))
    c2 = rng.normal(loc=[5.0, 5.0], scale=0.5, size=(30, 2))
    c3 = rng.normal(loc=[-5.0, 5.0], scale=0.5, size=(30, 2))
    X = np.vstack([c1, c2, c3])
    return X


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (KMeans, {"n_clusters": 3, "random_state": 1}),
        (FCM, {"n_clusters": 3, "random_state": 1}),
        (GMM, {"n_clusters": 3, "random_state": 1}),
        (PFCM, {"n_clusters": 3, "random_state": 1}),
        (GustafsonKessel, {"n_clusters": 3, "random_state": 1}),
    ],
)
def test_common_method_interface(sample_data, model_cls, kwargs):
    """Verify that every method implements the standard unsupervised interface."""
    X = sample_data
    model = model_cls(**kwargs)

    assert isinstance(model, BaseClusteringMethod)
    assert model.cluster_centers_ is None

    # Fit unsupervised
    fitted = model.fit(X)
    assert fitted is model

    # Check fitted attributes
    assert model.cluster_centers_ is not None
    assert model.cluster_centers_.shape == (3, 2)
    assert model.n_clusters == 3
    assert isinstance(model.converged_, bool)
    assert isinstance(model.n_iter_, int)
    assert model.n_iter_ > 0
    assert len(model.objective_history_) > 0
    assert model.status_ in ("SUCCESS", "MAX_ITER_REACHED")

    # Hard predictions
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)).issubset({0, 1, 2})

    # Out-of-sample prediction
    X_new = np.array([[0.1, 0.1], [4.9, 5.1]])
    pred_new = model.predict(X_new)
    assert pred_new.shape == (2,)

    # Membership prediction
    U = model.predict_membership(X)
    assert isinstance(U, np.ndarray)
    assert U.shape == (len(X), 3)


def test_soft_methods_membership_shape_and_simplex(sample_data):
    """Verify that FCM, GMM, PFCM, and GK produce valid soft memberships satisfying simplex constraints."""
    X = sample_data
    soft_models = [
        FCM(n_clusters=3, random_state=42),
        GMM(n_clusters=3, random_state=42),
        PFCM(n_clusters=3, random_state=42),
        GustafsonKessel(n_clusters=3, random_state=42),
    ]

    for model in soft_models:
        model.fit(X)
        U = model.membership_
        assert U.shape == (len(X), 3)

        # Non-negative
        assert (U >= -1e-6).all(), f"{model.__class__.__name__} has negative memberships"

        # Row sum == 1
        row_sums = np.sum(U, axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(len(X)),
            atol=1e-4,
            err_msg=f"{model.__class__.__name__} row sums violate simplex constraint",
        )


def test_hard_membership_one_hot(sample_data):
    """Verify that KMeans predict_membership returns one-hot matrix and sets semantics flag."""
    X = sample_data
    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X)

    U = km.predict_membership(X)
    assert km.membership_semantics == "hard_one_hot"
    assert U.shape == (len(X), 3)

    # Values must be exactly 0.0 or 1.0
    assert set(np.unique(U)).issubset({0.0, 1.0})
    # Exactly one 1.0 per row
    np.testing.assert_allclose(np.sum(U, axis=1), np.ones(len(X)))
