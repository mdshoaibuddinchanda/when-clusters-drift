"""Numerical utilities and validation helpers for clustering methods."""

from typing import Tuple
import numpy as np
import pandas as pd


def ensure_feature_array(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Validate and convert input feature matrix to a contiguous float64 2D numpy array.

    Raises
    ------
    TypeError
        If X is neither a pandas DataFrame nor a numpy array.
    ValueError
        If X is not 2D, has zero rows or columns, or contains NaNs or Infs.
    """
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=np.float64, copy=False)
    elif isinstance(X, np.ndarray):
        arr = np.asarray(X, dtype=np.float64)
    else:
        raise TypeError(f"Expected pandas DataFrame or numpy array, got {type(X)}")

    if arr.ndim != 2:
        raise ValueError(f"Feature matrix must be 2-dimensional, got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Feature matrix must not be empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Feature matrix contains non-finite values (NaN, +inf, or -inf).")

    return np.ascontiguousarray(arr)


def check_simplex_constraint(U: np.ndarray, tol: float = 1e-4) -> Tuple[bool, float]:
    """Verify that soft membership matrix U satisfies simplex constraints:
    u_ik >= 0 and sum_k u_ik == 1 for every row.

    Returns
    -------
    Tuple[bool, float]
        (is_valid, max_row_sum_deviation)
    """
    if U.ndim != 2:
        return False, float("inf")
    if (U < -tol).any():
        return False, float(np.max(np.abs(np.minimum(U, 0.0))))
    row_sums = np.sum(U, axis=1)
    max_dev = float(np.max(np.abs(row_sums - 1.0)))
    is_valid = bool(max_dev <= tol and (U >= -tol).all())
    return is_valid, max_dev


def regularize_covariance_matrix(
    cov: np.ndarray,
    min_eig: float = 1e-6,
    max_cond: float = 1e8,
    ridge_factor: float = 1e-5,
) -> Tuple[np.ndarray, float, bool]:
    """Regularize a symmetric covariance matrix to guarantee positive-definiteness and invertibility.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of shape (d, d).
    min_eig : float
        Minimum allowable eigenvalue.
    max_cond : float
        Maximum allowable condition number.
    ridge_factor : float
        Ridge regularization multiplier applied to trace if ill-conditioned.

    Returns
    -------
    Tuple[np.ndarray, float, bool]
        (regularized_cov, condition_number, was_regularized)
    """
    d = cov.shape[0]
    # Symmetrize
    cov_sym = 0.5 * (cov + cov.T)

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    eigvals = np.real(eigvals)

    min_val = eigvals[0]
    max_val = eigvals[-1]
    cond = max_val / max(min_val, 1e-15) if min_val > 0 else float("inf")

    was_regularized = False

    if min_val < min_eig or cond > max_cond:
        was_regularized = True
        trace_val = np.maximum(np.trace(cov_sym) / d, 1e-4)
        ridge = ridge_factor * trace_val
        # Floor eigenvalues or apply ridge
        clamped_eigvals = np.maximum(eigvals, max(min_eig, ridge))
        cov_sym = eigvecs @ np.diag(clamped_eigvals) @ eigvecs.T
        cov_sym = 0.5 * (cov_sym + cov_sym.T)
        new_eigvals = np.linalg.eigvalsh(cov_sym)
        cond = float(new_eigvals[-1] / max(new_eigvals[0], 1e-15))

    return cov_sym, cond, was_regularized


def compute_log_det_root(cov: np.ndarray) -> float:
    """Compute [det(cov)]^(1/d) stably in log-space."""
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return 1e-8
    return float(np.exp(logdet / d))


def compute_pairwise_mahalanobis_sq(
    X: np.ndarray,
    center: np.ndarray,
    metric_A: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis squared distance (x_i - v)^T A (x_i - v) for all points in X.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    center : np.ndarray, shape (d,)
    metric_A : np.ndarray, shape (d, d)

    Returns
    -------
    np.ndarray, shape (n,)
        Squared Mahalanobis distance for each point.
    """
    diff = X - center  # (n, d)
    diff_A = diff @ metric_A  # (n, d)
    d_sq = np.sum(diff_A * diff, axis=1)  # (n,)
    return np.maximum(d_sq, 0.0)


def compute_methods_config_sha256(config: dict) -> str:
    """Compute deterministic SHA-256 digest of methods configuration dictionary."""
    import hashlib
    import json

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

