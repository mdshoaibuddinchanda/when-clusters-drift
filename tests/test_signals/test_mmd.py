"""Targeted unit tests for MMD_b^2, Block Chunking, and Source-Only Bandwidth."""

import numpy as np
import pytest

from clusterdrift.signals.covariate import (
    compute_chunked_rbf_kernel_sum,
    compute_mmd_b2,
    derive_source_median_bandwidth,
)


def test_mmd_identical_samples():
    """Test MMD_b^2(X, X) is numerically zero."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    val, _ = compute_mmd_b2(X, X, sigma=1.0)
    assert np.isclose(val, 0.0, atol=1e-10)


def test_mmd_symmetry():
    """Test MMD_b^2(X, Y) == MMD_b^2(Y, X)."""
    rng = np.random.RandomState(42)
    X = rng.randn(80, 4)
    Y = rng.randn(120, 4) + 1.5
    val_xy, _ = compute_mmd_b2(X, Y, sigma=2.0)
    val_yx, _ = compute_mmd_b2(Y, X, sigma=2.0)
    assert np.isclose(val_xy, val_yx, atol=1e-12)


def test_mmd_row_permutation_invariance():
    """Test permuting rows of X does not alter MMD."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 3)
    Y = rng.randn(70, 3)
    perm = rng.permutation(len(X))
    X_perm = X[perm]

    val1, _ = compute_mmd_b2(X, Y, sigma=1.0)
    val2, _ = compute_mmd_b2(X_perm, Y, sigma=1.0)
    assert np.isclose(val1, val2, atol=1e-12)


def test_mmd_chunking_equivalence():
    """Test chunked exact GEMM summation matches across different chunk sizes."""
    rng = np.random.RandomState(123)
    X = rng.randn(150, 6)
    Y = rng.randn(130, 6)
    sigma = 1.8

    val_c16, _ = compute_mmd_b2(X, Y, sigma=sigma, chunk_size=16)
    val_c64, _ = compute_mmd_b2(X, Y, sigma=sigma, chunk_size=64)
    val_c512, _ = compute_mmd_b2(X, Y, sigma=sigma, chunk_size=512)

    assert np.isclose(val_c16, val_c64, atol=1e-12)
    assert np.isclose(val_c64, val_c512, atol=1e-12)


def test_source_only_bandwidth_invariance():
    """Test RBF bandwidth sigma derived from reference bank does not depend on target."""
    rng = np.random.RandomState(456)
    A_R = rng.randn(200, 8)

    sig1, stat1 = derive_source_median_bandwidth(A_R, "ds_test", 0, "hash_1", global_signal_seed=2026090706)
    sig2, stat2 = derive_source_median_bandwidth(A_R, "ds_test", 0, "hash_1", global_signal_seed=2026090706)
    assert sig1 == sig2
    assert stat1 == "SUCCESS"
    assert sig1 > 0.0


def test_mmd_location_shift_monotonicity():
    """Test larger synthetic location shift produces larger MMD."""
    rng = np.random.RandomState(789)
    X = rng.randn(200, 4)
    Y_mild = X + 1.0
    Y_severe = X + 5.0

    sigma = 2.0
    mmd_mild, _ = compute_mmd_b2(X, Y_mild, sigma=sigma)
    mmd_severe, _ = compute_mmd_b2(X, Y_severe, sigma=sigma)

    assert mmd_severe > mmd_mild
