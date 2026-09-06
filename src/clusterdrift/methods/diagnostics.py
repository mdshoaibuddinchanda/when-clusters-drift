"""Formal fuzzy partition degeneracy diagnostics.

Detects degenerate fuzzy solutions where prototypes collapse onto identical locations
and memberships degrade to near-uniform assignments across clusters.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd

from clusterdrift.methods.utils import ensure_feature_array
from clusterdrift.metrics.internal import fuzzy_partition_coefficient, partition_entropy


def assess_fuzzy_partition_degeneracy(
    X: np.ndarray | pd.DataFrame,
    centers: np.ndarray,
    U: np.ndarray,
    K: int,
    tol_fpc: float = 0.02,
    tol_dist: float = 0.02,
) -> Dict[str, Any]:
    """Assess whether a soft/fuzzy clustering partition has collapsed into a degenerate solution.

    A degenerate solution requires joint evidence from BOTH membership uniformity AND prototype collapse:
    1. Membership near-uniformity: FPC is close to theoretical floor (1/K) AND mean maximum membership
       is close to 1/K (lack of crispness/contrast).
    2. Prototype collapse: normalized minimum pairwise prototype distance is near zero, or fewer than
       K effectively distinct prototypes exist.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        Input feature matrix of shape (N, d).
    centers : np.ndarray
        Cluster prototypes/centers of shape (K, d).
    U : np.ndarray
        Soft membership matrix of shape (N, K).
    K : int
        Number of clusters.
    tol_fpc : float
        Tolerance above 1/K for FPC floor gap to be considered near-uniform (default 0.02).
    tol_dist : float
        Tolerance for normalized minimum center distance to be considered collapsed (default 0.02).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all diagnostic metrics and the boolean flag 'is_degenerate'.
    """
    arr_X = ensure_feature_array(X)
    centers = np.asarray(centers, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    N, d = arr_X.shape

    # 1. Membership partition metrics
    fpc = fuzzy_partition_coefficient(U)
    pe = partition_entropy(U)
    fpc_floor = 1.0 / float(K)
    fpc_floor_gap = max(0.0, float(fpc - fpc_floor))
    pe_ceiling = float(np.log(K))
    pe_ceiling_gap = max(0.0, float(pe_ceiling - pe))

    mean_max_membership = float(np.mean(np.max(U, axis=1)))
    hard_labels = np.argmax(U, axis=1)
    effective_hard_clusters = int(len(np.unique(hard_labels)))

    # 2. Prototype geometry & data scale
    global_mean = np.mean(arr_X, axis=0)
    data_variance = np.mean(np.sum((arr_X - global_mean) ** 2, axis=1))
    data_scale = float(np.sqrt(max(data_variance, 1e-12)))

    # Pairwise center distances
    if K > 1:
        diff_centers = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (K, K, d)
        dist_centers = np.sqrt(np.maximum(np.sum(diff_centers ** 2, axis=-1), 0.0))  # (K, K)
        np.fill_diagonal(dist_centers, np.inf)
        min_center_dist = float(np.min(dist_centers))
    else:
        min_center_dist = 0.0

    normalized_min_center_dist = min_center_dist / max(data_scale, 1e-12)

    # Count distinct prototypes (at least tol_dist * data_scale from each other)
    distinct_count = 0
    if K > 1:
        reps = []
        for c in centers:
            if not any(np.linalg.norm(c - r) < (tol_dist * data_scale) for r in reps):
                reps.append(c)
        distinct_count = len(reps)
    else:
        distinct_count = 1

    # 3. Joint evidence for degeneracy
    membership_is_near_uniform = (
        fpc_floor_gap < tol_fpc and mean_max_membership < (fpc_floor + 0.15)
    )
    prototype_is_collapsed = (
        normalized_min_center_dist < tol_dist or distinct_count < K
    )

    is_degenerate = bool(membership_is_near_uniform and prototype_is_collapsed)

    return {
        "fpc": round(fpc, 6),
        "pe": round(pe, 6),
        "fpc_floor": round(fpc_floor, 6),
        "fpc_floor_gap": round(fpc_floor_gap, 6),
        "pe_ceiling": round(pe_ceiling, 6),
        "pe_ceiling_gap": round(pe_ceiling_gap, 6),
        "min_center_distance": round(min_center_dist, 6),
        "normalized_min_center_distance": round(normalized_min_center_dist, 6),
        "data_scale": round(data_scale, 6),
        "mean_max_membership": round(mean_max_membership, 6),
        "effective_distinct_prototypes": distinct_count,
        "effective_hard_clusters": effective_hard_clusters,
        "is_degenerate": is_degenerate,
    }
