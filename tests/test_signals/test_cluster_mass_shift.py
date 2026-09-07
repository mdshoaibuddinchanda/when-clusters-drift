"""Targeted unit tests for Fuzzy Cluster-Mass Redistribution Signal (D_M)."""

import numpy as np
import pytest

from clusterdrift.signals.mass import (
    compute_cluster_mass_distribution,
    compute_cluster_mass_shift,
)


def test_cluster_mass_simplex():
    """Test average cluster mass lies on simplex."""
    U = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ], dtype=np.float64)
    p = compute_cluster_mass_distribution(U)
    assert len(p) == 3
    assert np.all(p >= 0.0)
    assert np.isclose(np.sum(p), 1.0, atol=1e-12)


def test_cluster_mass_shift_identical():
    """Test D_M is 0.0 when mass distributions are identical."""
    U = np.array([[0.6, 0.4], [0.8, 0.2]], dtype=np.float64)
    res = compute_cluster_mass_shift(U, U)
    assert np.isclose(res["D_M"], 0.0, atol=1e-6)


def test_cluster_mass_shift_divergent():
    """Test D_M is strictly positive when mass redistribution occurs."""
    U1 = np.array([[0.9, 0.1], [0.9, 0.1]], dtype=np.float64)
    U2 = np.array([[0.1, 0.9], [0.1, 0.9]], dtype=np.float64)
    res = compute_cluster_mass_shift(U1, U2)
    assert res["D_M"] > 0.5
    assert 0.0 <= res["D_M"] <= 1.0
