"""Targeted unit tests for Conventional Validity Controls (FPC, PE, PE_norm, XB_soft_m2, Silhouette)."""

import numpy as np
import pytest

from clusterdrift.signals.validity import (
    compute_fpc,
    compute_pe,
    compute_silhouette_control,
    compute_validity_controls,
    compute_xb_soft_m2,
)


def test_fpc_bounds_and_extremes():
    """Test FPC is 1.0 for crisp memberships and 1/K for uniform memberships."""
    K = 4
    B = 10
    # Crisp
    U_crisp = np.zeros((B, K))
    U_crisp[:, 0] = 1.0
    assert np.isclose(compute_fpc(U_crisp), 1.0, atol=1e-12)

    # Uniform
    U_unif = np.ones((B, K)) / float(K)
    assert np.isclose(compute_fpc(U_unif), 1.0 / K, atol=1e-12)


def test_pe_bounds_and_extremes():
    """Test PE is 0 for crisp and 1.0 (normalized) for uniform."""
    K = 3
    B = 8
    U_crisp = np.zeros((B, K))
    U_crisp[:, 1] = 1.0
    pe_c, pe_norm_c = compute_pe(U_crisp)
    assert np.isclose(pe_c, 0.0, atol=1e-12)
    assert np.isclose(pe_norm_c, 0.0, atol=1e-12)

    U_unif = np.ones((B, K)) / float(K)
    pe_u, pe_norm_u = compute_pe(U_unif)
    assert np.isclose(pe_norm_u, 1.0, atol=1e-12)


def test_xb_soft_m2_hand_fixture():
    """Test Xie-Beni calculation against hand calculation."""
    # 2 samples, 2 clusters in 1D
    # Centers: v0 = 0.0, v1 = 10.0 => min_dist2 = 100.0
    # X: x0 = 0.0, x1 = 10.0
    # U: u0 = [1.0, 0.0], u1 = [0.0, 1.0] (crisp at centers)
    # numerator: 1.0^2 * 0^2 + 1.0^2 * 0^2 = 0
    # XB = 0
    X = np.array([[0.0], [10.0]])
    V = np.array([[0.0], [10.0]])
    U = np.array([[1.0, 0.0], [0.0, 1.0]])

    xb, status = compute_xb_soft_m2(X, V, U)
    assert status == "SUCCESS"
    assert np.isclose(xb, 0.0, atol=1e-12)


def test_xb_soft_m2_collapsed_centers():
    """Test XB returns NOT_DEFINED and NaN when prototype centers collapse."""
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    V_collapsed = np.array([[1.0, 1.0], [1.0, 1.0]])  # Collapsed centers
    U = np.array([[0.5, 0.5], [0.5, 0.5]])

    xb, status = compute_xb_soft_m2(X, V_collapsed, U)
    assert status == "NOT_DEFINED"
    assert np.isnan(xb)


def test_silhouette_single_cluster_undefined():
    """Test Silhouette returns NOT_DEFINED and NaN when only one cluster is assigned."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # All samples assign to cluster 0
    U_single = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])

    sil, status = compute_silhouette_control(X, U_single)
    assert status == "NOT_DEFINED"
    assert np.isnan(sil)


def test_silhouette_valid_assignment():
    """Test Silhouette succeeds with distinct well-separated clusters."""
    X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
    U = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])

    sil, status = compute_silhouette_control(X, U)
    assert status == "SUCCESS"
    assert sil > 0.8
