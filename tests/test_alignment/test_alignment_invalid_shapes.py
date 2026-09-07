"""Test error handling for invalid shapes, dimensions, and non-simplex memberships."""

import numpy as np
import pytest

from clusterdrift.alignment.hungarian import align_clusters


def test_different_cluster_count_fails():
    """Alignment between different K must raise ValueError explicitly."""
    c_ref = np.array([[0.0, 0.0], [1.0, 1.0]])  # K = 2
    c_cand = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # K = 3
    s_ref = np.array([1.0, 1.0])
    u_ref = np.array([[0.5, 0.5]])
    u_cand = np.array([[0.33, 0.33, 0.34]])

    with pytest.raises(ValueError, match="Cluster count mismatch"):
        align_clusters(c_ref, c_cand, s_ref, u_ref, u_cand)


def test_different_feature_dimension_fails():
    """Alignment between different feature dimensions must raise ValueError explicitly."""
    c_ref = np.array([[0.0, 0.0], [1.0, 1.0]])  # D = 2
    c_cand = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # D = 3
    s_ref = np.array([1.0, 1.0])
    u_ref = np.array([[0.5, 0.5]])
    u_cand = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        align_clusters(c_ref, c_cand, s_ref, u_ref, u_cand)


def test_nan_or_inf_fails():
    """Non-finite values in centers or memberships must raise ValueError."""
    c_ref = np.array([[np.nan, 0.0], [1.0, 1.0]])
    c_cand = np.array([[0.0, 0.0], [1.0, 1.0]])
    s_ref = np.array([1.0, 1.0])
    u_ref = np.array([[0.5, 0.5]])
    u_cand = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError, match="Membership matrix contains non-finite"):
        align_clusters(
            c_cand, c_cand, s_ref,
            np.array([[np.nan, 0.5]]), u_cand
        )


def test_non_simplex_membership_fails():
    """Memberships that do not sum to 1.0 must raise ValueError."""
    c = np.array([[0.0, 0.0], [1.0, 1.0]])
    s = np.array([1.0, 1.0])
    u_bad = np.array([[0.8, 0.8]])  # Sums to 1.6 != 1.0

    with pytest.raises(ValueError, match="Membership rows must sum to 1.0"):
        align_clusters(c, c, s, u_bad, np.array([[0.5, 0.5]]))
