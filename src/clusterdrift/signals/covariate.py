"""Covariate Shift Signal (D_X = MMD_b^2) and Source-Only Bandwidth Derivation.

Mathematical definitions:
    Kernel: RBF k(x, y) = exp(- ||x - y||^2 / (2 * sigma^2))
    Estimator: Biased non-negative MMD_b^2(X, Y)
    MMD_b^2 = (1 / n^2) * sum_{i,j} k(x_i, x_j)
            + (1 / m^2) * sum_{i,j} k(y_i, y_j)
            - (2 / (n * m)) * sum_{i,j} k(x_i, y_j)

Bandwidth policy:
    sigma is derived ONCE per (dataset, outer_fold) from reference bank A^R only.
    Uses up to 1024 points deterministically sampled with global_signal_seed.
    sigma = median({||x_i - x_j||_2 : i < j, ||x_i - x_j|| > 0}).
    If no positive distance exists: sigma = 1.0, status = FALLBACK.
"""

import hashlib
from typing import Optional, Tuple
import numpy as np
from scipy.spatial.distance import pdist


def derive_source_median_bandwidth(
    A_R: np.ndarray,
    dataset_id: str,
    outer_fold: int,
    ref_bank_sha256: str,
    global_signal_seed: int = 2026090706,
    max_points: int = 1024,
) -> Tuple[float, str]:
    """Derive invariant RBF bandwidth sigma from reference probe bank A^R.

    Parameters
    ----------
    A_R : np.ndarray
        Transformed reference probe bank matrix, shape (B_R, D).
    dataset_id : str
        Canonical dataset identifier.
    outer_fold : int
        Outer fold index.
    ref_bank_sha256 : str
        Hash of reference probe bank descriptor.
    global_signal_seed : int
        Global seed from config.
    max_points : int
        Maximum number of points used for pairwise distances.

    Returns
    -------
    sigma : float
        Median positive pairwise Euclidean distance or fallback 1.0.
    status : str
        "SUCCESS" or "FALLBACK".
    """
    n_samples = len(A_R)
    if n_samples == 0:
        return 1.0, "FALLBACK"

    if n_samples > max_points:
        seed_str = f"mmd_bandwidth:{global_signal_seed}:{dataset_id}:{outer_fold}:{ref_bank_sha256}"
        derived_seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(derived_seed)
        sampled_indices = np.sort(rng.choice(n_samples, size=max_points, replace=False))
        points = A_R[sampled_indices]
    else:
        points = A_R

    dists = pdist(points, metric="euclidean")
    pos_dists = dists[dists > 1e-12]

    if len(pos_dists) > 0:
        sigma = float(np.median(pos_dists))
        return sigma, "SUCCESS"
    return 1.0, "FALLBACK"


def compute_chunked_rbf_kernel_sum(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float,
    chunk_size: int = 512,
) -> float:
    """Compute exact sum_{i,j} exp(-||x_i - y_j||^2 / (2 * sigma^2)) in float64 blocks.

    Uses BLAS GEMM formulation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)

    n = X_arr.shape[0]
    m = Y_arr.shape[0]

    gamma = 1.0 / (2.0 * sigma * sigma)

    # Precompute squared L2 row norms
    x_norms2 = np.sum(X_arr ** 2, axis=1, keepdims=True)
    y_norms2 = np.sum(Y_arr ** 2, axis=1, keepdims=True).T

    total_sum = 0.0

    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        X_chunk = X_arr[i:i_end]
        x_norm_chunk = x_norms2[i:i_end]

        for j in range(0, m, chunk_size):
            j_end = min(j + chunk_size, m)
            Y_chunk = Y_arr[j:j_end]
            y_norm_chunk = y_norms2[:, j:j_end]

            # BLAS dot product
            dots = np.dot(X_chunk, Y_chunk.T)
            dist2 = np.maximum(x_norm_chunk + y_norm_chunk - 2.0 * dots, 0.0)
            k_block = np.exp(-gamma * dist2)
            total_sum += float(np.sum(k_block, dtype=np.float64))

    return total_sum


def compute_mmd_b2(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float,
    chunk_size: int = 512,
    negative_tolerance: float = 1e-10,
    cached_sum_xx: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute biased MMD_b^2 between X and Y using RBF kernel.

    Parameters
    ----------
    X : np.ndarray
        First sample bank (e.g. Reference A^R), shape (n, D).
    Y : np.ndarray
        Second sample bank (e.g. Current A_t^C), shape (m, D).
    sigma : float
        RBF bandwidth parameter.
    chunk_size : int
        Block size for chunked BLAS GEMM.
    negative_tolerance : float
        Tolerance for tiny floating negative values.
    cached_sum_xx : Optional[float]
        Optional precomputed sum_{i,j} k(x_i, x_j) for reuse.

    Returns
    -------
    mmd_b2 : float
        Non-negative biased MMD^2 estimate.
    sum_xx : float
        Computed or passed sum_xx for caching.
    """
    n = len(X)
    m = len(Y)

    if n == 0 or m == 0:
        raise ValueError(f"Cannot compute MMD with empty sample sets: n={n}, m={m}")

    if cached_sum_xx is not None:
        sum_xx = cached_sum_xx
    else:
        sum_xx = compute_chunked_rbf_kernel_sum(X, X, sigma=sigma, chunk_size=chunk_size)

    sum_yy = compute_chunked_rbf_kernel_sum(Y, Y, sigma=sigma, chunk_size=chunk_size)
    sum_xy = compute_chunked_rbf_kernel_sum(X, Y, sigma=sigma, chunk_size=chunk_size)

    term_xx = sum_xx / (float(n) * float(n))
    term_yy = sum_yy / (float(m) * float(m))
    term_xy = (2.0 * sum_xy) / (float(n) * float(m))

    raw_mmd2 = term_xx + term_yy - term_xy

    if -negative_tolerance <= raw_mmd2 < 0.0:
        mmd_b2 = 0.0
    elif raw_mmd2 < -negative_tolerance:
        raise ValueError(
            f"Numerical failure in MMD_b^2: value {raw_mmd2:.3e} is less than -negative_tolerance ({-negative_tolerance:.3e})"
        )
    else:
        mmd_b2 = float(raw_mmd2)

    return mmd_b2, sum_xx


def compute_mmd_b2_torch(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float,
    chunk_size: int = 512,
    device: str = "cuda",
) -> float:
    """Optional PyTorch CUDA implementation for benchmarking numerical equivalence."""
    import torch

    X_t = torch.as_tensor(X, dtype=torch.float64, device=device)
    Y_t = torch.as_tensor(Y, dtype=torch.float64, device=device)

    n = X_t.shape[0]
    m = Y_t.shape[0]
    gamma = 1.0 / (2.0 * sigma * sigma)

    x_norms2 = torch.sum(X_t ** 2, dim=1, keepdim=True)
    y_norms2 = torch.sum(Y_t ** 2, dim=1, keepdim=True).t()

    def _block_sum(A, B, a_norms, b_norms, n_a, n_b):
        tot = torch.tensor(0.0, dtype=torch.float64, device=device)
        for i in range(0, n_a, chunk_size):
            i_end = min(i + chunk_size, n_a)
            A_chunk = A[i:i_end]
            a_n = a_norms[i:i_end]
            for j in range(0, n_b, chunk_size):
                j_end = min(j + chunk_size, n_b)
                B_chunk = B[j:j_end]
                b_n = b_norms[:, j:j_end]
                dist2 = torch.clamp(a_n + b_n - 2.0 * torch.matmul(A_chunk, B_chunk.t()), min=0.0)
                tot += torch.sum(torch.exp(-gamma * dist2))
        return tot

    sum_xx = _block_sum(X_t, X_t, x_norms2, x_norms2.t(), n, n)
    sum_yy = _block_sum(Y_t, Y_t, y_norms2.t(), y_norms2, m, m)
    sum_xy = _block_sum(X_t, Y_t, x_norms2, y_norms2, n, m)

    val = (sum_xx / (n * n)) + (sum_yy / (m * m)) - (2.0 * sum_xy / (n * m))
    val_clamped = torch.clamp(val, min=0.0).item()
    return float(val_clamped)
