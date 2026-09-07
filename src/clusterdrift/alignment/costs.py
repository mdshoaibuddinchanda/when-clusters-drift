"""Cost matrix formulations for cluster alignment.

Computes:
1. Reference cluster RMS radius scales with numerical floor.
2. Pairwise center distance matrix normalized by reference scale.
3. Soft-Jaccard membership overlap matrix on historical probe bank.
4. Convex combination alignment cost matrix.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist


def compute_reference_cluster_scales(
    X_source: np.ndarray,
    U_source: np.ndarray,
    centers_ref: np.ndarray,
    epsilon: float = 1e-12,
    scale_floor_ratio: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute membership-weighted RMS radius for each reference cluster.

    Parameters
    ----------
    X_source : np.ndarray, shape (N, D)
        Source training feature matrix.
    U_source : np.ndarray, shape (N, K)
        Source membership matrix (fuzzy, responsibilities, or one-hot).
    centers_ref : np.ndarray, shape (K, D)
        Reference cluster prototype centers.
    epsilon : float
        Small positive constant for numerical stability.
    scale_floor_ratio : float
        Ratio rho for flooring: s_k >= rho * s_global.

    Returns
    -------
    scales : np.ndarray, shape (K,)
        Floored RMS radius for each cluster.
    floored : np.ndarray, shape (K,)
        Boolean flags indicating whether flooring was applied.
    s_global : float
        Global unweighted RMS radius around source mean.
    """
    N, D = X_source.shape
    K = centers_ref.shape[0]

    # Global unweighted RMS radius
    mean_global = np.mean(X_source, axis=0)
    global_sq_dists = np.sum((X_source - mean_global) ** 2, axis=1)
    s_global = float(np.sqrt(np.mean(global_sq_dists) + epsilon))
    floor_val = scale_floor_ratio * s_global

    scales = np.zeros(K, dtype=np.float64)
    floored = np.zeros(K, dtype=bool)

    for k in range(K):
        # Squared Euclidean distance from all source points to center k
        sq_dists_k = np.sum((X_source - centers_ref[k]) ** 2, axis=1)  # Shape: (N,)
        weights_k = U_source[:, k]  # Shape: (N,)
        sum_weights = np.sum(weights_k)

        weighted_var = np.sum(weights_k * sq_dists_k) / (sum_weights + epsilon)
        raw_scale = np.sqrt(weighted_var)

        if raw_scale < floor_val:
            scales[k] = floor_val
            floored[k] = True
        else:
            scales[k] = raw_scale
            floored[k] = False

    return scales, floored, s_global


def compute_soft_jaccard_overlap(
    U_ref: np.ndarray,
    U_cand: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute pairwise soft-Jaccard overlap matrix between reference and candidate models.

    Both membership matrices must be evaluated on the EXACT SAME historical reference probe bank A^R.

    O_{kj} = \\frac{\\sum_r \\min(U^R_{rk}, U^C_{rj})}{\\sum_r \\max(U^R_{rk}, U^C_{rj}) + \\epsilon}

    Parameters
    ----------
    U_ref : np.ndarray, shape (B, K)
        Reference model memberships on historical probe bank.
    U_cand : np.ndarray, shape (B, K)
        Candidate model memberships on historical probe bank.
    epsilon : float
        Small positive constant for numerical stability.

    Returns
    -------
    overlap : np.ndarray, shape (K, K)
        Symmetric bounded overlap matrix in [0, 1].
    """
    B, K_ref = U_ref.shape
    B_c, K_cand = U_cand.shape

    if B != B_c:
        raise ValueError(f"Probe sample size mismatch: U_ref has {B}, U_cand has {B_c}")
    if K_ref != K_cand:
        raise ValueError(f"Cluster count mismatch: U_ref has {K_ref}, U_cand has {K_cand}")

    # Vectorized computation across B probe rows
    u_r = U_ref[:, :, None]  # Shape: (B, K, 1)
    u_c = U_cand[:, None, :]  # Shape: (B, 1, K)

    min_sum = np.sum(np.minimum(u_r, u_c), axis=0)  # Shape: (K, K)
    max_sum = np.sum(np.maximum(u_r, u_c), axis=0)  # Shape: (K, K)

    overlap = min_sum / (max_sum + epsilon)
    return np.clip(overlap, 0.0, 1.0)


def compute_center_cost_matrix(
    centers_ref: np.ndarray,
    centers_cand: np.ndarray,
    scales_ref: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute pairwise normalized prototype center distance matrix.

    D^{center}_{kj} = \\frac{\\|v_k^{ref} - v_j^{cand}\\|_2}{s_k^{ref} + \\epsilon}

    Parameters
    ----------
    centers_ref : np.ndarray, shape (K, D)
        Reference cluster prototype centers.
    centers_cand : np.ndarray, shape (K, D)
        Candidate cluster prototype centers.
    scales_ref : np.ndarray, shape (K,)
        Reference cluster RMS radius scales.
    epsilon : float
        Small positive constant for numerical stability.

    Returns
    -------
    D_center : np.ndarray, shape (K, K)
        Normalized center distance matrix.
    """
    K_ref, D_ref = centers_ref.shape
    K_cand, D_cand = centers_cand.shape

    if K_ref != K_cand:
        raise ValueError(f"Cluster count mismatch: centers_ref has {K_ref}, centers_cand has {K_cand}")
    if D_ref != D_cand:
        raise ValueError(f"Feature dimension mismatch: centers_ref has {D_ref}, centers_cand has {D_cand}")

    # Pairwise Euclidean distance
    dist_matrix = cdist(centers_ref, centers_cand, metric="euclidean")  # Shape: (K, K)
    d_center = dist_matrix / (scales_ref[:, None] + epsilon)
    return d_center


def compute_combined_alignment_cost(
    centers_ref: np.ndarray,
    centers_cand: np.ndarray,
    scales_ref: np.ndarray,
    U_ref: np.ndarray,
    U_cand: np.ndarray,
    eta: float = 0.50,
    epsilon: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute combined bipartite matching cost matrix: C = eta * D_center + (1 - eta) * (1 - O).

    Parameters
    ----------
    centers_ref : np.ndarray, shape (K, D)
        Reference cluster prototype centers.
    centers_cand : np.ndarray, shape (K, D)
        Candidate cluster prototype centers.
    scales_ref : np.ndarray, shape (K,)
        Reference cluster RMS radius scales.
    U_ref : np.ndarray, shape (B, K)
        Reference model memberships on historical probe bank.
    U_cand : np.ndarray, shape (B, K)
        Candidate model memberships on historical probe bank.
    eta : float, default 0.50
        Convex combination weight: 0 <= eta <= 1.
    epsilon : float
        Small positive constant for numerical stability.

    Returns
    -------
    combined_cost : np.ndarray, shape (K, K)
    center_cost : np.ndarray, shape (K, K)
    overlap : np.ndarray, shape (K, K)
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta must be in [0.0, 1.0], got {eta}")

    center_cost = compute_center_cost_matrix(centers_ref, centers_cand, scales_ref, epsilon=epsilon)
    overlap = compute_soft_jaccard_overlap(U_ref, U_cand, epsilon=epsilon)

    combined_cost = eta * center_cost + (1.0 - eta) * (1.0 - overlap)
    return combined_cost, center_cost, overlap
