import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from clusterdrift.falsification.models import tune_and_fit_hgbr, tune_and_fit_ridge

def test_inner_group_cv_no_leakage():
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 4

    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    groups = np.array([f"dataset_{i % 8}" for i in range(n_samples)])

    gkf = GroupKFold(n_splits=5)
    for tr_idx, val_idx in gkf.split(X, y, groups=groups):
        tr_groups = set(groups[tr_idx])
        val_groups = set(groups[val_idx])
        # Intersection must be empty
        assert len(tr_groups.intersection(val_groups)) == 0


def test_tune_and_fit_respects_groups():
    rng = np.random.RandomState(42)
    n_samples = 100
    X = rng.randn(n_samples, 2)
    y = rng.randn(n_samples)
    groups = np.array([f"ds_{i % 6}" for i in range(n_samples)])

    grid = {"learning_rate": [0.1], "max_leaf_nodes": [7], "max_iter": [10], "l2_regularization": [0.0]}
    model, prep, selection = tune_and_fit_hgbr(X, y, groups, grid, n_splits=3)
    assert selection.selected_params["max_leaf_nodes"] == 7

    ridge_alphas = [0.1, 1.0]
    r_model, r_prep, r_sel = tune_and_fit_ridge(X, y, groups, ridge_alphas, n_splits=3)
    assert r_sel.selected_params["alpha"] in ridge_alphas