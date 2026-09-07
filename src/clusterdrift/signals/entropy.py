"""Normalized Entropy and Entropy Shift Signal (D_H).

Mathematical definitions:
    For membership vector u_i in Simplex^{K-1}:
    h(u_i) = - sum_k(u_ik * ln(u_ik)) / ln(K) in [0, 1] (for K > 1, 0 for K = 1)
    
    Current bank evaluations:
    H_bar_0^C = (1 / B) * sum_i h(u_0,i^C)
    H_bar_t^C = (1 / B) * sum_i h(u_t,i^C)
    
    Signed change:
    Delta_H = H_bar_t^C - H_bar_0^C
    
    Primary signal:
    D_H = |Delta_H| in [0, 1]
"""

from typing import Dict, Tuple, Union
import numpy as np
from scipy.special import entr

from clusterdrift.signals.divergence import check_probability_simplex


def compute_normalized_entropy(
    U: np.ndarray,
    validate: bool = True,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute normalized Shannon entropy h(u) in [0, 1] for membership matrix U.

    Parameters
    ----------
    U : np.ndarray
        Shape (K,) or (B, K) representing soft memberships.
    validate : bool
        Whether to check simplex constraints.
    tol : float
        Simplex tolerance.

    Returns
    -------
    h : np.ndarray
        Normalized entropy in [0, 1]. Float if 1D, (B,) if 2D.
    """
    U_arr = np.asarray(U, dtype=np.float64)
    if validate:
        check_probability_simplex(U_arr, tol=tol, name="U")

    K = U_arr.shape[-1]
    if K <= 1:
        if U_arr.ndim == 1:
            return 0.0
        return np.zeros(U_arr.shape[0], dtype=np.float64)

    # entr(x) computes -x * ln(x) with entr(0) = 0
    shannon = np.sum(entr(U_arr), axis=-1)
    h = shannon / np.log(float(K))
    h_clamped = np.clip(h, 0.0, 1.0)

    if U_arr.ndim == 1:
        return float(h_clamped)
    return h_clamped


def compute_entropy_shift(
    U_source_current: np.ndarray,
    U_candidate_current: np.ndarray,
    validate: bool = True,
    tol: float = 1e-5,
) -> Dict[str, float]:
    """Compute entropy shift statistics between deployed source and candidate models on current bank.

    Parameters
    ----------
    U_source_current : np.ndarray
        Shape (B, K), memberships of deployed source model on current probe bank A_t^C.
    U_candidate_current : np.ndarray
        Shape (B, K), memberships of aligned candidate model on current probe bank A_t^C.
    validate : bool
        Whether to check simplex constraints.
    tol : float
        Tolerance for simplex checks.

    Returns
    -------
    result : Dict[str, float]
        Contains:
        - entropy_source_current: float
        - entropy_candidate_current: float
        - entropy_signed_change: float (candidate - source)
        - D_H: float (|candidate - source|)
    """
    U0 = np.asarray(U_source_current, dtype=np.float64)
    Ut = np.asarray(U_candidate_current, dtype=np.float64)

    if U0.shape != Ut.shape:
        raise ValueError(f"Shape mismatch: source {U0.shape} vs candidate {Ut.shape}")

    h0 = compute_normalized_entropy(U0, validate=validate, tol=tol)
    ht = compute_normalized_entropy(Ut, validate=validate, tol=tol)

    mean_h0 = float(np.mean(h0))
    mean_ht = float(np.mean(ht))
    delta_h = mean_ht - mean_h0
    d_h = abs(delta_h)

    return {
        "entropy_source_current": round(mean_h0, 7),
        "entropy_candidate_current": round(mean_ht, 7),
        "entropy_signed_change": round(delta_h, 7),
        "D_H": round(d_h, 7),
    }
