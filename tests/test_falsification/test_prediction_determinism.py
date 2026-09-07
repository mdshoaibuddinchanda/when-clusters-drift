import numpy as np
import pytest
from clusterdrift.falsification.models import tune_and_fit_hgbr, tune_and_fit_ridge

def test_prediction_determinism():
    rng = np.random.RandomState(2026090707)
    X = rng.randn(120, 3)
    y = rng.randn(120)
    groups = np.array([f"ds_{i % 5}" for i in range(120)])

    grid = {"learning_rate": [0.05], "max_leaf_nodes": [7], "max_iter": [50], "l2_regularization": [0.0]}

    m1, p1, s1 = tune_and_fit_hgbr(X, y, groups, grid, random_state=2026090707)
    pred1 = m1.predict(p1.transform(X[:20]))

    m2, p2, s2 = tune_and_fit_hgbr(X, y, groups, grid, random_state=2026090707)
    pred2 = m2.predict(p2.transform(X[:20]))

    np.testing.assert_array_equal(pred1, pred2)