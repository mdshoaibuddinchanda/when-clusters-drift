"""Vectorized Normalized Jensen-Shannon Divergence and Simplex Operations.

Mathematical definition:
    For probability distributions p, q in Simplex^{K-1}:
    m = 0.5 * (p + q)
    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    JS_N(p, q) = JS(p, q) / ln(2) in [0, 1]

Zero convention:
    0 * ln(0 / m) = 0. Stably evaluated via scipy.special.rel_entr.
"""

from typing import Tuple, Union
import numpy as np
from scipy.special import rel_entr


def check_probability_simplex(
    P: np.ndarray,
    tol: float = 1e-5,
    name: str = "probability matrix",
) -> None:
    """Validate that rows of P lie on the probability simplex."""
    if not np.all(np.isfinite(P)):
        raise ValueError(f"{name} contains non-finite (NaN or Inf) values")
    if np.any(P < -tol):
        min_val = float(np.min(P))
        raise ValueError(f"{name} contains negative values: min = {min_val}")
    row_sums = np.sum(P, axis=-1)
    if np.any(np.abs(row_sums - 1.0) > tol):
        max_err = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"{name} rows do not sum to 1: max absolute deviation = {max_err}")


def compute_normalized_js_divergence(
    P: np.ndarray,
    Q: np.ndarray,
    validate: bool = True,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute normalized Jensen-Shannon divergence JS_N(P, Q) in [0, 1].

    Parameters
    ----------
    P : np.ndarray
        Array of shape (K,) or (B, K) representing probability distributions.
    Q : np.ndarray
        Array of shape (K,) or (B, K) matching P.
    validate : bool
        Whether to check simplex constraints.
    tol : float
        Tolerance for simplex validation.

    Returns
    -------
    js_n : np.ndarray
        Normalized JS divergence: float if 1D input, (B,) if 2D input.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    Q_arr = np.asarray(Q, dtype=np.float64)

    if P_arr.shape != Q_arr.shape:
        raise ValueError(f"Shape mismatch: P {P_arr.shape} vs Q {Q_arr.shape}")

    if validate:
        check_probability_simplex(P_arr, tol=tol, name="P")
        check_probability_simplex(Q_arr, tol=tol, name="Q")

    # Midpoint distribution
    M = 0.5 * (P_arr + Q_arr)

    # KL(P || M) and KL(Q || M) via rel_entr (handles 0 * log(0) = 0)
    kl_p_m = np.sum(rel_entr(P_arr, M), axis=-1)
    kl_q_m = np.sum(rel_entr(Q_arr, M), axis=-1)

    js = 0.5 * (kl_p_m + kl_q_m)
    js_n = js / np.log(2.0)

    # Numerical safeguard: clamp to theoretical [0, 1] range
    js_n_clamped = np.clip(js_n, 0.0, 1.0)

    if P_arr.ndim == 1:
        return float(js_n_clamped)
    return js_n_clamped
