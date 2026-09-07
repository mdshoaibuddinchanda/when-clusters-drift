"""Test non-mutation guarantees of the alignment infrastructure."""

import numpy as np

from clusterdrift.alignment.hungarian import align_clusters


def test_alignment_does_not_mutate_inputs():
    """Verify input arrays are strictly unmodified by align_clusters."""
    centers_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    centers_cand = np.array([[3.1, 4.1], [0.9, 1.9]])
    scales_ref = np.array([1.5, 1.5])
    U_ref = np.array([[0.8, 0.2], [0.3, 0.7]])
    U_cand = np.array([[0.2, 0.8], [0.7, 0.3]])

    ref_copy = centers_ref.copy()
    cand_copy = centers_cand.copy()
    scales_copy = scales_ref.copy()
    u_ref_copy = U_ref.copy()
    u_cand_copy = U_cand.copy()

    res = align_clusters(
        centers_ref=centers_ref,
        centers_cand=centers_cand,
        scales_ref=scales_ref,
        U_ref=U_ref,
        U_cand=U_cand,
        eta=0.50,
    )

    # All inputs must be byte-identical to before alignment
    np.testing.assert_array_equal(centers_ref, ref_copy)
    np.testing.assert_array_equal(centers_cand, cand_copy)
    np.testing.assert_array_equal(scales_ref, scales_copy)
    np.testing.assert_array_equal(U_ref, u_ref_copy)
    np.testing.assert_array_equal(U_cand, u_cand_copy)

    # Result arrays must be separate copies, not views of inputs
    assert res.aligned_centers is not centers_cand
    assert res.aligned_memberships is not U_cand
