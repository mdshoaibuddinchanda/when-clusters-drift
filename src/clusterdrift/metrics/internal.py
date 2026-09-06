"""Internal unsupervised evaluation metrics for clustering and fuzzy partitions."""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from clusterdrift.methods.utils import ensure_feature_array


def silhouette_metric(
    X: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
) -> Optional[float]:
    """Compute mean Silhouette Coefficient. Returns None if < 2 unique clusters."""
    arr_X = ensure_feature_array(X)
    arr_labels = np.asarray(labels).ravel()
    unique_labels = np.unique(arr_labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(arr_labels):
        return None
    return float(silhouette_score(arr_X, arr_labels))


def fuzzy_partition_coefficient(U: np.ndarray) -> float:
    """Compute Fuzzy Partition Coefficient (FPC).

    FPC = (1 / n) * sum_{i=1}^n sum_{k=1}^K u_{ik}^2
    Range: [1/K, 1.0]. Higher values indicate cleaner, less ambiguous partitions.
    """
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError(f"U must be 2D, got shape {U.shape}")
    N = U.shape[0]
    if N == 0:
        return 0.0
    return float(np.sum(U ** 2) / N)


def partition_entropy(U: np.ndarray, eps: float = 1e-15) -> float:
    """Compute Partition Entropy (PE).

    PE = - (1 / n) * sum_{i=1}^n sum_{k=1}^K u_{ik} * ln(u_{ik})
    Range: [0, ln(K)]. Lower values indicate crisper partitions.
    """
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError(f"U must be 2D, got shape {U.shape}")
    N = U.shape[0]
    if N == 0:
        return 0.0
    # Guard zero entries
    safe_U = np.maximum(U, eps)
    entropy_terms = U * np.log(safe_U)
    return float(-np.sum(entropy_terms) / N)


def xie_beni_index(
    X: np.ndarray | pd.DataFrame,
    U: np.ndarray,
    centers: np.ndarray,
    m: float = 2.0,
) -> Optional[float]:
    """Compute Xie-Beni (XB) index.

    XB = sum_{i=1}^n sum_{k=1}^K u_{ik}^m * ||x_i - v_k||^2 / (n * min_{j != k} ||v_j - v_k||^2)
    Lower values indicate more compact, better-separated clusters.
    """
    arr_X = ensure_feature_array(X)
    U = np.asarray(U, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    N, d = arr_X.shape
    K = centers.shape[0]

    if K < 2:
        return None

    # Compute minimum squared distance between distinct centers
    diff_centers = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (K, K, d)
    dist_sq_centers = np.sum(diff_centers ** 2, axis=-1)  # (K, K)
    # Mask diagonal
    np.fill_diagonal(dist_sq_centers, np.inf)
    min_center_dist_sq = float(np.min(dist_sq_centers))

    if min_center_dist_sq <= 1e-15:
        return None

    # Compute numerator: sum_i sum_k u_ik^m * ||x_i - v_k||^2
    diff_X = arr_X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, K, d)
    dist_sq_X = np.sum(diff_X ** 2, axis=-1)  # (N, K)
    numerator = np.sum((U ** m) * dist_sq_X)

    denominator = N * min_center_dist_sq
    return float(numerator / denominator)
