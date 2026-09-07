"""Probe Membership Drift Signals: Historical (D_U_R) and Current (D_U_C).

Mathematical definitions:
    For reference probe bank A^R:
    D_U^R = (1 / |A^R|) * sum_{i in A^R} JS_N(u_0,i^R, u_tilde_t,i^R)
    
    For current probe bank A_t^C:
    D_U^C = (1 / |A_t^C|) * sum_{i in A_t^C} JS_N(u_0,i^C, u_tilde_t,i^C)
    
    where u_tilde_t is aligned using the reference-derived Hungarian permutation pi_t.
"""

from typing import Dict
import numpy as np

from clusterdrift.signals.divergence import compute_normalized_js_divergence


def compute_pointwise_membership_drift(
    U_source: np.ndarray,
    U_candidate_aligned: np.ndarray,
    validate: bool = True,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute pointwise normalized JS divergence for each probe sample."""
    U0 = np.asarray(U_source, dtype=np.float64)
    Ut = np.asarray(U_candidate_aligned, dtype=np.float64)

    if U0.shape != Ut.shape:
        raise ValueError(f"Membership shape mismatch: source {U0.shape} vs candidate {Ut.shape}")

    return compute_normalized_js_divergence(U0, Ut, validate=validate, tol=tol)


def compute_historical_membership_drift(
    U_source_reference: np.ndarray,
    U_candidate_reference_aligned: np.ndarray,
    validate: bool = True,
    tol: float = 1e-5,
) -> Dict[str, float]:
    """Compute historical probe membership drift D_U_R on reference bank A^R.

    Parameters
    ----------
    U_source_reference : np.ndarray
        Shape (B_R, K), source model memberships on reference bank A^R.
    U_candidate_reference_aligned : np.ndarray
        Shape (B_R, K), aligned candidate model memberships on reference bank A^R.
    validate : bool
        Whether to check simplex constraints.
    tol : float
        Tolerance for simplex checks.

    Returns
    -------
    result : Dict[str, float]
        Contains D_U_R, D_U_R_median, D_U_R_p95.
    """
    pw = compute_pointwise_membership_drift(
        U_source_reference,
        U_candidate_reference_aligned,
        validate=validate,
        tol=tol,
    )
    return {
        "D_U_R": round(float(np.mean(pw)), 7),
        "D_U_R_median": round(float(np.median(pw)), 7),
        "D_U_R_p95": round(float(np.percentile(pw, 95)), 7),
    }


def compute_current_membership_drift(
    U_source_current: np.ndarray,
    U_candidate_current_aligned: np.ndarray,
    validate: bool = True,
    tol: float = 1e-5,
) -> Dict[str, float]:
    """Compute current probe membership drift D_U_C on current bank A_t^C.

    Parameters
    ----------
    U_source_current : np.ndarray
        Shape (B_C, K), source model memberships on current bank A_t^C.
    U_candidate_current_aligned : np.ndarray
        Shape (B_C, K), aligned candidate model memberships on current bank A_t^C.
    validate : bool
        Whether to check simplex constraints.
    tol : float
        Tolerance for simplex checks.

    Returns
    -------
    result : Dict[str, float]
        Contains D_U_C, D_U_C_median, D_U_C_p95.
    """
    pw = compute_pointwise_membership_drift(
        U_source_current,
        U_candidate_current_aligned,
        validate=validate,
        tol=tol,
    )
    return {
        "D_U_C": round(float(np.mean(pw)), 7),
        "D_U_C_median": round(float(np.median(pw)), 7),
        "D_U_C_p95": round(float(np.percentile(pw, 95)), 7),
    }
