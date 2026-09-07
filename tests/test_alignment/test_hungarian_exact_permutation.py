"""Permutation recovery tests for Hungarian bipartite alignment.

Tests exact recovery of known permutations across K in {2, 3, 5, 10, 26}.
"""

import numpy as np
import pytest

from clusterdrift.alignment.hungarian import align_clusters


@pytest.mark.parametrize("K", [2, 3, 5, 10, 26])
def test_hungarian_exact_known_permutation_recovery(K):
    """Verify that Hungarian alignment recovers exact synthetic permutations with 100% accuracy."""
    rng = np.random.default_rng(1000 + K)
    D = 8
    B = 200

    # 1. Distinct reference centers well-separated
    centers_ref = rng.uniform(-10.0, 10.0, size=(K, D))
    # Ensure distinctness by adding diagonal offset
    centers_ref += np.eye(K, D) * 15.0

    scales_ref = np.ones(K, dtype=np.float64) * 2.0

    # Distinct reference memberships
    raw_U = rng.uniform(0.01, 1.0, size=(B, K))
    # Sharpen so clusters are distinct
    for k in range(K):
        raw_U[k * (B // K) : (k + 1) * (B // K), k] += 10.0
    U_ref = raw_U / np.sum(raw_U, axis=1, keepdims=True)

    # 2. Apply random permutation pi to create candidate
    pi = rng.permutation(K)
    centers_cand = centers_ref[pi].copy()
    U_cand = U_ref[:, pi].copy()

    # Small perturbation to simulate real-world estimation noise
    centers_cand += rng.normal(0, 0.05, size=(K, D))
    U_cand += rng.normal(0, 0.005, size=(B, K))
    U_cand = np.clip(U_cand, 0.0, 1.0)
    U_cand /= np.sum(U_cand, axis=1, keepdims=True)

    # 3. Align candidate back to reference
    result = align_clusters(
        centers_ref=centers_ref,
        centers_cand=centers_cand,
        scales_ref=scales_ref,
        U_ref=U_ref,
        U_cand=U_cand,
        eta=0.50,
    )

    # The Hungarian permutation maps reference k -> candidate j
    # Since candidate j came from reference pi[j], we expect permutation[k] to equal the position of k in pi!
    # That is: centers_cand[permutation[k]] matches centers_ref[k]
    # Because centers_cand = centers_ref[pi], centers_cand[j] = centers_ref[pi[j]]
    # Thus pi[permutation[k]] must equal k!
    reconstructed_ref = pi[result.permutation]
    np.testing.assert_array_equal(
        reconstructed_ref,
        np.arange(K),
        err_msg=f"Failed to recover exact permutation for K={K}. Permutation was {result.permutation}, pi was {pi}"
    )

    # Check that aligned centers closely match reference centers
    np.testing.assert_allclose(result.aligned_centers, centers_ref, atol=0.25)
