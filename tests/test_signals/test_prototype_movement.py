"""Targeted unit tests for Normalized Prototype Movement (D_V)."""

import numpy as np
import pytest

from clusterdrift.signals.prototype import compute_prototype_movement


def test_prototype_movement_identical():
    """Test D_V is 0.0 when prototype centers are identical."""
    V = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    scales = np.array([1.0, 1.0], dtype=np.float64)
    res = compute_prototype_movement(V, V, scales)
    assert np.isclose(res["D_V"], 0.0, atol=1e-12)
    assert np.isclose(res["D_V_median"], 0.0, atol=1e-12)
    assert np.isclose(res["D_V_max"], 0.0, atol=1e-12)


def test_prototype_movement_hand_calculation():
    """Test exact hand-calculated displacement."""
    # V0: c0=[0, 0], c1=[0, 0]
    # Vt: c0=[3, 4] (norm 5), c1=[0, 0] (norm 0)
    # scales: s0=2.0, s1=1.0
    # delta_0 = 5.0 / 2.0 = 2.5
    # delta_1 = 0.0 / 1.0 = 0.0
    # D_V = (2.5 + 0.0) / 2 = 1.25
    V0 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    Vt = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
    scales = np.array([2.0, 1.0], dtype=np.float64)

    res = compute_prototype_movement(V0, Vt, scales, epsilon=0.0)
    assert np.isclose(res["D_V"], 1.25, atol=1e-6)
    assert np.isclose(res["D_V_max"], 2.5, atol=1e-6)
    assert np.isclose(res["D_V_median"], 1.25, atol=1e-6)


def test_prototype_movement_mismatched_dimensions():
    """Test error raised if center counts or dimensions mismatch."""
    V0 = np.zeros((3, 2))
    Vt = np.zeros((2, 2))
    scales = np.ones(3)
    with pytest.raises(ValueError, match="shapes mismatch"):
        compute_prototype_movement(V0, Vt, scales)
