"""Unit tests for combined alignment cost formulation."""

import numpy as np
import pytest

from clusterdrift.alignment.costs import (
    compute_center_cost_matrix,
    compute_combined_alignment_cost,
    compute_soft_jaccard_overlap,
)


def test_combined_cost_interpolation():
    """Verify combined cost respects convex combination C = eta * D + (1 - eta) * (1 - O)."""
    centers_ref = np.array([[0.0, 0.0], [5.0, 5.0]])
    centers_cand = np.array([[0.1, 0.1], [5.2, 5.0]])
    scales_ref = np.array([1.0, 2.0])

    U_ref = np.array([[0.9, 0.1], [0.1, 0.9]])
    U_cand = np.array([[0.8, 0.2], [0.2, 0.8]])

    D = compute_center_cost_matrix(centers_ref, centers_cand, scales_ref)
    O = compute_soft_jaccard_overlap(U_ref, U_cand)

    # eta = 0.0 -> pure (1 - O)
    C_0, _, _ = compute_combined_alignment_cost(
        centers_ref, centers_cand, scales_ref, U_ref, U_cand, eta=0.0
    )
    np.testing.assert_allclose(C_0, 1.0 - O, atol=1e-12)

    # eta = 1.0 -> pure D
    C_1, _, _ = compute_combined_alignment_cost(
        centers_ref, centers_cand, scales_ref, U_ref, U_cand, eta=1.0
    )
    np.testing.assert_allclose(C_1, D, atol=1e-12)

    # eta = 0.5 -> 0.5 * D + 0.5 * (1 - O)
    C_half, D_out, O_out = compute_combined_alignment_cost(
        centers_ref, centers_cand, scales_ref, U_ref, U_cand, eta=0.5
    )
    expected_half = 0.5 * D + 0.5 * (1.0 - O)
    np.testing.assert_allclose(C_half, expected_half, atol=1e-12)
    np.testing.assert_allclose(D_out, D, atol=1e-12)
    np.testing.assert_allclose(O_out, O, atol=1e-12)
