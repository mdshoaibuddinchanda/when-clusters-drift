"""Targeted unit tests for Normalized Jensen-Shannon Divergence."""

import numpy as np
import pytest

from clusterdrift.signals.divergence import (
    check_probability_simplex,
    compute_normalized_js_divergence,
)


def test_js_divergence_hand_fixture():
    """Test orthogonal distributions hand fixture p=[1, 0], q=[0, 1] yields exact 1.0."""
    p = np.array([1.0, 0.0], dtype=np.float64)
    q = np.array([0.0, 1.0], dtype=np.float64)
    val = compute_normalized_js_divergence(p, q)
    assert np.isclose(val, 1.0, atol=1e-12)


def test_js_divergence_identity():
    """Test JS divergence of distribution with itself is strictly 0.0."""
    p = np.array([0.4, 0.3, 0.3], dtype=np.float64)
    val = compute_normalized_js_divergence(p, p)
    assert np.isclose(val, 0.0, atol=1e-12)


def test_js_divergence_symmetry():
    """Test JS divergence is symmetric: JS_N(p, q) == JS_N(q, p)."""
    p = np.array([0.7, 0.2, 0.1], dtype=np.float64)
    q = np.array([0.1, 0.5, 0.4], dtype=np.float64)
    val_pq = compute_normalized_js_divergence(p, q)
    val_qp = compute_normalized_js_divergence(q, p)
    assert np.isclose(val_pq, val_qp, atol=1e-12)


def test_js_divergence_range():
    """Test JS_N is strictly bounded in [0, 1] across random probability vectors."""
    rng = np.random.RandomState(42)
    for _ in range(50):
        K = rng.randint(2, 10)
        p = rng.dirichlet(np.ones(K))
        q = rng.dirichlet(np.ones(K))
        val = compute_normalized_js_divergence(p, q)
        assert 0.0 <= val <= 1.0


def test_js_divergence_vectorized_2d():
    """Test vectorized evaluation across 2D membership batches."""
    P = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.8, 0.2],
    ], dtype=np.float64)
    Q = np.array([
        [0.0, 1.0],
        [0.5, 0.5],
        [0.2, 0.8],
    ], dtype=np.float64)
    vals = compute_normalized_js_divergence(P, Q)
    assert vals.shape == (3,)
    assert np.isclose(vals[0], 1.0, atol=1e-12)
    assert np.isclose(vals[1], 0.0, atol=1e-12)
    assert 0.0 < vals[2] < 1.0


def test_js_divergence_simplex_validation():
    """Test simplex violation detection."""
    p_invalid_neg = np.array([-0.1, 1.1])
    q_valid = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="negative values"):
        compute_normalized_js_divergence(p_invalid_neg, q_valid)

    p_invalid_sum = np.array([0.5, 0.6])
    with pytest.raises(ValueError, match="rows do not sum to 1"):
        compute_normalized_js_divergence(p_invalid_sum, q_valid)
