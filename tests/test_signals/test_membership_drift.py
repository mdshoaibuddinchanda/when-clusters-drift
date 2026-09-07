"""Targeted unit tests for Historical and Current Probe Membership Drift (D_U_R, D_U_C)."""

import numpy as np
import pytest

from clusterdrift.signals.membership import (
    compute_current_membership_drift,
    compute_historical_membership_drift,
    compute_pointwise_membership_drift,
)


def test_membership_drift_identical():
    """Test pointwise and aggregate drift are strictly 0 for identical memberships."""
    U = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.5, 0.5],
    ], dtype=np.float64)
    pw = compute_pointwise_membership_drift(U, U)
    assert np.allclose(pw, 0.0, atol=1e-12)

    dur = compute_historical_membership_drift(U, U)
    assert np.isclose(dur["D_U_R"], 0.0, atol=1e-12)
    assert np.isclose(dur["D_U_R_median"], 0.0, atol=1e-12)
    assert np.isclose(dur["D_U_R_p95"], 0.0, atol=1e-12)

    duc = compute_current_membership_drift(U, U)
    assert np.isclose(duc["D_U_C"], 0.0, atol=1e-12)


def test_membership_drift_orthogonal():
    """Test orthogonal reassignments yield exact 1.0."""
    U1 = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    U2 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    res = compute_historical_membership_drift(U1, U2)
    assert np.isclose(res["D_U_R"], 1.0, atol=1e-12)


def test_membership_drift_shape_mismatch():
    """Test shape mismatch raises ValueError."""
    U1 = np.ones((5, 2)) * 0.5
    U2 = np.ones((6, 2)) * 0.5
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_historical_membership_drift(U1, U2)
