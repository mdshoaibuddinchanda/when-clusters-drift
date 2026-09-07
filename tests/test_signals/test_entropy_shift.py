"""Targeted unit tests for Normalized Entropy and Entropy Shift (D_H)."""

import numpy as np
import pytest

from clusterdrift.signals.entropy import (
    compute_entropy_shift,
    compute_normalized_entropy,
)


def test_entropy_crisp_is_zero():
    """Test normalized entropy of crisp one-hot distribution is strictly 0.0."""
    u_crisp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    h = compute_normalized_entropy(u_crisp)
    assert np.isclose(h, 0.0, atol=1e-12)


def test_entropy_uniform_is_one():
    """Test normalized entropy of uniform distribution is strictly 1.0."""
    for K in [2, 3, 5, 10]:
        u_unif = np.ones(K, dtype=np.float64) / float(K)
        h = compute_normalized_entropy(u_unif)
        assert np.isclose(h, 1.0, atol=1e-12)


def test_entropy_k1_is_zero():
    """Test K=1 case gracefully returns 0.0."""
    u_one = np.array([1.0], dtype=np.float64)
    assert compute_normalized_entropy(u_one) == 0.0


def test_entropy_shift_direction_and_magnitude():
    """Test entropy shift statistics between crisp source and soft candidate."""
    U_src = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float64)
    U_cand = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ], dtype=np.float64)

    res = compute_entropy_shift(U_src, U_cand)
    assert np.isclose(res["entropy_source_current"], 0.0, atol=1e-6)
    assert np.isclose(res["entropy_candidate_current"], 1.0, atol=1e-6)
    assert np.isclose(res["entropy_signed_change"], 1.0, atol=1e-6)
    assert np.isclose(res["D_H"], 1.0, atol=1e-6)


def test_entropy_shift_negative_signed_change():
    """Test when candidate is crisper than source, signed change is negative while D_H is positive."""
    U_src = np.array([[0.5, 0.5]], dtype=np.float64)
    U_cand = np.array([[1.0, 0.0]], dtype=np.float64)

    res = compute_entropy_shift(U_src, U_cand)
    assert res["entropy_signed_change"] < 0
    assert res["D_H"] > 0
    assert np.isclose(res["D_H"], abs(res["entropy_signed_change"]))
