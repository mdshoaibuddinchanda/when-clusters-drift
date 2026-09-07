"""Hungarian (Kuhn-Munkres) minimum-cost bipartite cluster alignment.

Provides exact O(K^3) bipartite matching between reference and candidate cluster
representations with orientation consistency, ambiguity detection, and non-mutation guarantees.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from clusterdrift.alignment.costs import compute_combined_alignment_cost
from clusterdrift.alignment.result import AlignmentResult
from clusterdrift.alignment.state import ClusterState
from clusterdrift.alignment.validation import validate_cluster_states, validate_membership_matrix


def align_clusters(
    centers_ref: np.ndarray,
    centers_cand: np.ndarray,
    scales_ref: np.ndarray,
    U_ref: np.ndarray,
    U_cand: np.ndarray,
    eta: float = 0.50,
    cost_margin_tolerance: float = 1e-8,
    U_cand_to_align: Optional[np.ndarray] = None,
    epsilon: float = 1e-12,
    metadata: Optional[Dict[str, Any]] = None,
) -> AlignmentResult:
    """Perform exact minimum-cost bipartite alignment of candidate clusters to reference clusters.

    Orientation convention:
        permutation[k_ref] = j_cand
        aligned_candidate_centers = centers_cand[permutation]
        aligned_candidate_memberships = U_cand[:, permutation]

    Parameters
    ----------
    centers_ref : np.ndarray, shape (K, D)
        Reference cluster prototype centers.
    centers_cand : np.ndarray, shape (K, D)
        Candidate cluster prototype centers.
    scales_ref : np.ndarray, shape (K,)
        Reference cluster RMS radius scales.
    U_ref : np.ndarray, shape (B, K)
        Reference model memberships on historical reference probe bank A^R.
    U_cand : np.ndarray, shape (B, K)
        Candidate model memberships on historical reference probe bank A^R.
    eta : float, default 0.50
        Weight between center distance and membership overlap: C = eta * D + (1 - eta) * (1 - O).
    cost_margin_tolerance : float, default 1e-8
        Threshold for detecting low-margin/ambiguous assignments.
    U_cand_to_align : np.ndarray, optional
        Additional candidate membership matrix to permute (e.g. evaluated on current probe bank A^C).
        If None, permutes U_cand.
    epsilon : float, default 1e-12
        Small positive constant for numerical stability.
    metadata : dict, optional
        Diagnostic metadata to attach to result.

    Returns
    -------
    result : AlignmentResult
        Aligned centers, permuted memberships, permutations, costs, and ambiguity diagnostics.
    """
    # 1. Input validations
    K_ref, D_ref = centers_ref.shape
    K_cand, D_cand = centers_cand.shape

    if K_ref != K_cand:
        raise ValueError(
            f"Cluster count mismatch: reference has {K_ref}, candidate has {K_cand}. "
            "Alignment requires equal cluster count."
        )
    if D_ref != D_cand:
        raise ValueError(
            f"Feature dimension mismatch: reference has {D_ref}, candidate has {D_cand}."
        )
    if len(scales_ref) != K_ref:
        raise ValueError(f"scales_ref length {len(scales_ref)} does not match K {K_ref}")

    validate_membership_matrix(U_ref, expected_n_clusters=K_ref, check_simplex=True)
    validate_membership_matrix(U_cand, expected_n_samples=U_ref.shape[0], expected_n_clusters=K_cand, check_simplex=True)

    # 2. Compute cost matrices
    combined_cost, center_cost, overlap = compute_combined_alignment_cost(
        centers_ref=centers_ref,
        centers_cand=centers_cand,
        scales_ref=scales_ref,
        U_ref=U_ref,
        U_cand=U_cand,
        eta=eta,
        epsilon=epsilon,
    )

    # 3. Exact Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(combined_cost)
    # By convention of linear_sum_assignment, row_ind is [0, 1, ..., K-1]
    permutation = np.zeros(K_ref, dtype=np.int64)
    permutation[row_ind] = col_ind

    inverse_permutation = np.zeros(K_ref, dtype=np.int64)
    inverse_permutation[permutation] = np.arange(K_ref, dtype=np.int64)

    # 4. Total and identity assignment costs
    assignment_cost = float(np.sum(combined_cost[row_ind, col_ind]))
    identity_assignment_cost = float(np.trace(combined_cost))

    # 5. Permute candidate centers and memberships (DO NOT MUTATE ORIGINALS)
    aligned_centers = centers_cand[permutation].copy()

    target_membership = U_cand_to_align if U_cand_to_align is not None else U_cand
    aligned_memberships = target_membership[:, permutation].copy() if target_membership is not None else None

    # 6. Ambiguity analysis
    ambiguous = False
    min_margin = float("inf")
    ambiguous_pairs = []

    if K_ref > 1:
        for k in range(K_ref):
            assigned_j = permutation[k]
            assigned_cost = combined_cost[k, assigned_j]

            # Best alternative candidate for this reference cluster
            other_j_costs = [combined_cost[k, j] for j in range(K_ref) if j != assigned_j]
            row_margin = min(other_j_costs) - assigned_cost

            # Best alternative reference for this candidate cluster
            other_k_costs = [combined_cost[k_prime, assigned_j] for k_prime in range(K_ref) if k_prime != k]
            col_margin = min(other_k_costs) - assigned_cost

            pair_margin = float(min(row_margin, col_margin))
            if pair_margin < min_margin:
                min_margin = pair_margin

            if pair_margin < cost_margin_tolerance:
                ambiguous = True
                ambiguous_pairs.append({
                    "ref_cluster": int(k),
                    "assigned_cand_cluster": int(assigned_j),
                    "cost": float(assigned_cost),
                    "margin": pair_margin,
                })
    else:
        min_margin = float("inf")

    ambiguity_details = {
        "cost_margin_tolerance": cost_margin_tolerance,
        "minimum_assignment_margin": float(min_margin) if np.isfinite(min_margin) else 1e9,
        "ambiguous_pairs": ambiguous_pairs,
        "num_ambiguous_pairs": len(ambiguous_pairs),
    }

    return AlignmentResult(
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        aligned_centers=aligned_centers,
        aligned_memberships=aligned_memberships,
        center_cost_matrix=center_cost,
        overlap_matrix=overlap,
        combined_cost_matrix=combined_cost,
        assignment_cost=assignment_cost,
        identity_assignment_cost=identity_assignment_cost,
        cluster_scales=scales_ref.copy(),
        ambiguous=ambiguous,
        minimum_assignment_margin=float(min_margin) if np.isfinite(min_margin) else 1e9,
        ambiguity_details=ambiguity_details,
        eta=eta,
        metadata=metadata or {},
    )
