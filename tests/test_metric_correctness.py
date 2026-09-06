"""Tests validating external and internal clustering evaluation metrics."""

import numpy as np
import pytest

from clusterdrift.metrics.external import (
    adjusted_mutual_info,
    adjusted_rand_index,
    normalized_mutual_info,
)
from clusterdrift.metrics.internal import (
    fuzzy_partition_coefficient,
    partition_entropy,
    silhouette_metric,
    xie_beni_index,
)


def test_ari_permutation_invariance():
    """Verify that ARI, NMI, and AMI are invariant to label permutations."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    # Permute cluster labels: 0 -> 2, 1 -> 0, 2 -> 1
    y_pred2 = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])

    # Invariant to label permutation
    assert adjusted_rand_index(y_true, y_pred1) == pytest.approx(1.0)
    assert adjusted_rand_index(y_true, y_pred2) == pytest.approx(1.0)

    assert normalized_mutual_info(y_true, y_pred1) == pytest.approx(1.0)
    assert normalized_mutual_info(y_true, y_pred2) == pytest.approx(1.0)

    assert adjusted_mutual_info(y_true, y_pred1) == pytest.approx(1.0)
    assert adjusted_mutual_info(y_true, y_pred2) == pytest.approx(1.0)


def test_fpc_hand_computable():
    """Verify Fuzzy Partition Coefficient on an analytically solved 2x2 matrix."""
    # U = [[0.8, 0.2], [0.6, 0.4]]
    # sum of squares: 0.8^2 + 0.2^2 + 0.6^2 + 0.4^2 = 0.64 + 0.04 + 0.36 + 0.16 = 1.20
    # FPC = 1.20 / 2 = 0.60
    U = np.array([[0.8, 0.2], [0.6, 0.4]])
    fpc = fuzzy_partition_coefficient(U)
    assert fpc == pytest.approx(0.60, abs=1e-6)


def test_partition_entropy_hand_computable():
    """Verify Partition Entropy on an analytically solved 2x2 matrix."""
    # U = [[0.5, 0.5], [0.5, 0.5]]
    # PE = - (1/2) * 4 * (0.5 * ln(0.5)) = -2 * (0.5 * (-ln 2)) = ln 2 ≈ 0.693147
    U = np.array([[0.5, 0.5], [0.5, 0.5]])
    pe = partition_entropy(U)
    assert pe == pytest.approx(np.log(2.0), abs=1e-5)


def test_xie_beni_hand_computable():
    """Verify Xie-Beni index on a hand-constructed 1D 2-cluster example."""
    # 2 points in 1D: x_0 = 0.0, x_1 = 4.0
    # 2 centers: v_0 = 0.0, v_1 = 4.0
    # Min center distance squared: (4.0 - 0.0)^2 = 16.0
    # Let U = [[1.0, 0.0], [0.0, 1.0]], m=2
    # Distances to own center are 0. Numerator = 0.
    # XB = 0.0
    X = np.array([[0.0], [4.0]])
    U_crisp = np.array([[1.0, 0.0], [0.0, 1.0]])
    centers = np.array([[0.0], [4.0]])
    xb_crisp = xie_beni_index(X, U_crisp, centers, m=2.0)
    assert xb_crisp == pytest.approx(0.0, abs=1e-8)

    # Now let U = [[0.5, 0.5], [0.5, 0.5]], m=2
    # Point 0: d00^2 = 0, d01^2 = 16. u00^2 * 0 + u01^2 * 16 = 0.25 * 16 = 4.0
    # Point 1: d10^2 = 16, d11^2 = 0. u10^2 * 16 + u11^2 * 0 = 0.25 * 16 = 4.0
    # Numerator = 4.0 + 4.0 = 8.0
    # Denominator = N * min_dist_sq = 2 * 16 = 32.0
    # XB = 8.0 / 32.0 = 0.25
    U_fuzzy = np.array([[0.5, 0.5], [0.5, 0.5]])
    xb_fuzzy = xie_beni_index(X, U_fuzzy, centers, m=2.0)
    assert xb_fuzzy == pytest.approx(0.25, abs=1e-6)


def test_silhouette_metric():
    """Verify silhouette metric computation on well-separated clusters."""
    X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    labels = np.array([0, 0, 1, 1])
    score = silhouette_metric(X, labels)
    assert score is not None
    assert score > 0.8
