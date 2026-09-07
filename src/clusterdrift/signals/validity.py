"""Conventional Internal Clustering Validity Controls.

Evaluated on CURRENT probe bank A_t^C under DEPLOYED SOURCE model M_0:
    - FPC: Fuzzy Partition Coefficient in [1/K, 1]
    - PE: Partition Entropy
    - PE_norm: Normalized Partition Entropy in [0, 1]
    - XB_soft_m2: Soft Xie-Beni index with control exponent m=2
    - Silhouette: Standard silhouette on crisp argmax(U_0^C)
"""

from typing import Any, Dict, Tuple
import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import entr
from sklearn.metrics import silhouette_score

from clusterdrift.signals.divergence import check_probability_simplex


def compute_fpc(U: np.ndarray) -> float:
    """Compute Fuzzy Partition Coefficient (FPC) in [1/K, 1]."""
    U_arr = np.asarray(U, dtype=np.float64)
    B = U_arr.shape[0]
    if B == 0:
        return float("nan")
    return float(np.sum(U_arr ** 2) / float(B))


def compute_pe(U: np.ndarray) -> Tuple[float, float]:
    """Compute Partition Entropy (PE) and normalized PE_norm in [0, 1]."""
    U_arr = np.asarray(U, dtype=np.float64)
    B = U_arr.shape[0]
    K = U_arr.shape[1]

    if B == 0 or K == 0:
        return float("nan"), float("nan")

    # entr(x) computes -x * ln(x)
    total_entropy = float(np.sum(entr(U_arr)))
    pe = total_entropy / float(B)

    if K > 1:
        pe_norm = pe / np.log(float(K))
        pe_norm_clamped = float(np.clip(pe_norm, 0.0, 1.0))
    else:
        pe_norm_clamped = 0.0

    return pe, pe_norm_clamped


def compute_xb_soft_m2(
    X: np.ndarray,
    centers: np.ndarray,
    U: np.ndarray,
    min_dist_tol: float = 1e-12,
) -> Tuple[float, str]:
    """Compute soft Xie-Beni index with control exponent m=2.

    XB_soft_m2 = sum_{i,k} (u_{ik}^2 * ||x_i - v_k||^2) / (B * min_{j != k} ||v_j - v_k||^2)
    """
    X_arr = np.asarray(X, dtype=np.float64)
    V_arr = np.asarray(centers, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)

    B = X_arr.shape[0]
    K = V_arr.shape[0]

    if B == 0 or K <= 1:
        return float("nan"), "NOT_DEFINED"

    # Pairwise center distances
    c_dists = pdist(V_arr, metric="euclidean")
    if len(c_dists) == 0:
        return float("nan"), "NOT_DEFINED"

    min_center_dist = float(np.min(c_dists))
    if min_center_dist <= min_dist_tol:
        return float("nan"), "NOT_DEFINED"

    min_dist2 = min_center_dist ** 2

    # Vectorized numerator: sum_{i=1}^B sum_{k=1}^K u_ik^2 * ||x_i - v_k||^2
    # ||x_i - v_k||^2 = ||x_i||^2 + ||v_k||^2 - 2 x_i^T v_k
    x_sq = np.sum(X_arr ** 2, axis=1, keepdims=True)  # (B, 1)
    v_sq = np.sum(V_arr ** 2, axis=1, keepdims=True).T  # (1, K)
    xv = np.dot(X_arr, V_arr.T)  # (B, K)
    d2 = np.maximum(x_sq + v_sq - 2.0 * xv, 0.0)  # (B, K)

    weighted_d2 = (U_arr ** 2) * d2
    numerator = float(np.sum(weighted_d2))

    xb = numerator / (float(B) * min_dist2)
    return float(xb), "SUCCESS"


def compute_silhouette_control(
    X: np.ndarray,
    U: np.ndarray,
) -> Tuple[float, str]:
    """Compute silhouette score on crisp argmax(U) assignments.

    Returns (nan, 'NOT_DEFINED') if fewer than 2 distinct clusters are predicted.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)

    B = len(X_arr)
    if B < 2:
        return float("nan"), "NOT_DEFINED"

    labels = np.argmax(U_arr, axis=1)
    unique_labels = np.unique(labels)
    n_unique = len(unique_labels)

    if n_unique < 2 or n_unique >= B:
        return float("nan"), "NOT_DEFINED"

    try:
        score = float(silhouette_score(X_arr, labels))
        return score, "SUCCESS"
    except Exception:
        return float("nan"), "NOT_DEFINED"


def compute_validity_controls(
    X_current: np.ndarray,
    centers_source: np.ndarray,
    U_source_current: np.ndarray,
    validate_simplex: bool = True,
    simplex_tol: float = 1e-5,
) -> Dict[str, Any]:
    """Compute all conventional validity controls on current bank under source model."""
    if validate_simplex:
        check_probability_simplex(U_source_current, tol=simplex_tol, name="U_source_current")

    fpc = compute_fpc(U_source_current)
    pe, pe_norm = compute_pe(U_source_current)
    xb_soft, xb_status = compute_xb_soft_m2(X_current, centers_source, U_source_current)
    sil, sil_status = compute_silhouette_control(X_current, U_source_current)

    return {
        "FPC": round(fpc, 7) if np.isfinite(fpc) else fpc,
        "PE": round(pe, 7) if np.isfinite(pe) else pe,
        "PE_norm": round(pe_norm, 7) if np.isfinite(pe_norm) else pe_norm,
        "XB_soft_m2": round(xb_soft, 7) if np.isfinite(xb_soft) else xb_soft,
        "XB_status": xb_status,
        "silhouette": round(sil, 7) if np.isfinite(sil) else sil,
        "silhouette_status": sil_status,
    }
