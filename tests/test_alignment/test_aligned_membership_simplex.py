"""Test that aligned membership matrices strictly preserve the unit simplex."""

import numpy as np

from clusterdrift.alignment.hungarian import align_clusters


def test_aligned_membership_preserves_simplex():
    """Verify that after permutation, aligned memberships remain on the unit simplex."""
    rng = np.random.default_rng(888)
    K = 4
    B = 150
    D = 5

    centers_ref = rng.normal(0, 3, size=(K, D))
    centers_cand = centers_ref[::-1]  # Inverted order
    scales_ref = np.ones(K)

    raw_ref = rng.exponential(1.0, size=(B, K))
    U_ref = raw_ref / np.sum(raw_ref, axis=1, keepdims=True)
    U_cand = U_ref[:, ::-1]

    res = align_clusters(
        centers_ref=centers_ref,
        centers_cand=centers_cand,
        scales_ref=scales_ref,
        U_ref=U_ref,
        U_cand=U_cand,
    )

    aligned_U = res.aligned_memberships
    assert aligned_U is not None
    assert aligned_U.shape == (B, K)

    # 1. Non-negativity
    assert np.all(aligned_U >= 0.0)
    assert np.all(aligned_U <= 1.0)

    # 2. Row sum to 1.0
    row_sums = np.sum(aligned_U, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    # 3. Exact column match
    np.testing.assert_allclose(aligned_U, U_ref, atol=1e-12)
