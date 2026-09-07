"""Test reproducibility and deterministic stability of alignment."""

import numpy as np

from clusterdrift.alignment.hungarian import align_clusters


def test_alignment_reproducibility_repeated():
    """Verify repeated execution produces bitwise identical results."""
    rng = np.random.default_rng(777)
    centers_ref = rng.normal(0, 1, size=(5, 3))
    centers_cand = centers_ref[np.array([2, 0, 4, 1, 3])] + rng.normal(0, 0.01, size=(5, 3))
    scales_ref = np.ones(5)
    U_raw = rng.uniform(0.1, 1.0, size=(50, 5))
    U_ref = U_raw / np.sum(U_raw, axis=1, keepdims=True)
    U_cand = U_ref[:, np.array([2, 0, 4, 1, 3])]

    res1 = align_clusters(centers_ref, centers_cand, scales_ref, U_ref, U_cand)
    res2 = align_clusters(centers_ref, centers_cand, scales_ref, U_ref, U_cand)

    np.testing.assert_array_equal(res1.permutation, res2.permutation)
    np.testing.assert_array_equal(res1.aligned_centers, res2.aligned_centers)
    np.testing.assert_array_equal(res1.aligned_memberships, res2.aligned_memberships)
    assert res1.assignment_cost == res2.assignment_cost
    assert res1.ambiguous == res2.ambiguous
