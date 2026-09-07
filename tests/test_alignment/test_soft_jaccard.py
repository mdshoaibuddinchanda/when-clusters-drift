"""Unit tests for soft-Jaccard membership overlap calculation."""

import numpy as np
import pytest

from clusterdrift.alignment.costs import compute_soft_jaccard_overlap


def test_soft_jaccard_hand_computable():
    """Verify soft-Jaccard against exact analytical hand calculation."""
    # B = 2, K = 2
    U_ref = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
    ], dtype=np.float64)

    U_cand = np.array([
        [0.7, 0.3],
        [0.4, 0.6],
    ], dtype=np.float64)

    # O_00 = (min(0.8, 0.7) + min(0.1, 0.4)) / (max(0.8, 0.7) + max(0.1, 0.4)) = 0.8 / 1.2 = 2/3
    # O_01 = (min(0.8, 0.3) + min(0.1, 0.6)) / (max(0.8, 0.3) + max(0.1, 0.6)) = 0.4 / 1.4 = 2/7
    # O_10 = (min(0.2, 0.7) + min(0.9, 0.4)) / (max(0.2, 0.7) + max(0.9, 0.4)) = 0.6 / 1.6 = 3/8
    # O_11 = (min(0.2, 0.3) + min(0.9, 0.6)) / (max(0.2, 0.3) + max(0.9, 0.6)) = 0.8 / 1.2 = 2/3
    expected = np.array([
        [2.0 / 3.0, 2.0 / 7.0],
        [3.0 / 8.0, 2.0 / 3.0],
    ])

    O = compute_soft_jaccard_overlap(U_ref, U_cand, epsilon=0.0)
    np.testing.assert_allclose(O, expected, atol=1e-12)


def test_soft_jaccard_self_overlap_identity():
    """Self-overlap must have exact 1.0 on the diagonal."""
    rng = np.random.default_rng(42)
    B, K = 100, 5
    raw = rng.uniform(0.1, 1.0, size=(B, K))
    U = raw / np.sum(raw, axis=1, keepdims=True)

    O = compute_soft_jaccard_overlap(U, U, epsilon=1e-12)
    diag = np.diag(O)
    np.testing.assert_allclose(diag, 1.0, atol=1e-6)


def test_soft_jaccard_boundedness_and_symmetry():
    """Overlap must be bounded in [0, 1] and symmetric under transposition of swapped inputs."""
    rng = np.random.default_rng(123)
    B, K = 50, 4
    U_a = rng.uniform(0, 1, size=(B, K))
    U_a /= np.sum(U_a, axis=1, keepdims=True)
    U_b = rng.uniform(0, 1, size=(B, K))
    U_b /= np.sum(U_b, axis=1, keepdims=True)

    O_ab = compute_soft_jaccard_overlap(U_a, U_b)
    O_ba = compute_soft_jaccard_overlap(U_b, U_a)

    assert np.all(O_ab >= 0.0)
    assert np.all(O_ab <= 1.0)
    np.testing.assert_allclose(O_ab, O_ba.T, atol=1e-12)


def test_soft_jaccard_one_hot_memberships():
    """Test soft-Jaccard with discrete one-hot memberships."""
    # Two samples, perfectly identical partition
    U1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    U2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    O = compute_soft_jaccard_overlap(U1, U2, epsilon=1e-12)
    np.testing.assert_allclose(O, np.eye(2), atol=1e-6)

    # Inverted partition
    U3 = np.array([[0.0, 1.0], [1.0, 0.0]])
    O_inv = compute_soft_jaccard_overlap(U1, U3, epsilon=1e-12)
    np.testing.assert_allclose(O_inv, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-6)
