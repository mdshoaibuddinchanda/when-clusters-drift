"""Fuzzy Cluster-Mass Redistribution Signal (D_M).

Mathematical definitions:
    Evaluated on the SAME current probe bank A_t^C:
    p_k^0 = (1 / B) * sum_i u_{0, ik}^C
    p_k^t = (1 / B) * sum_i u_{t, ik}^C (aligned)
    
    Both p^0, p^t lie on Simplex^{K-1}.
    
    Primary signal:
    D_M = JS_N(p^0, p^t) in [0, 1]
"""

from typing import Any, Dict, List
import numpy as np

from clusterdrift.signals.divergence import check_probability_simplex, compute_normalized_js_divergence


def compute_cluster_mass_distribution(
    U: np.ndarray,
    validate: bool = True,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute average fuzzy cluster mass vector p in Simplex^{K-1}."""
    U_arr = np.asarray(U, dtype=np.float64)
    if validate:
        check_probability_simplex(U_arr, tol=tol, name="U")

    p = np.mean(U_arr, axis=0)
    # Ensure exact sum to 1 against floating drift
    p = p / np.sum(p)
    return p


def compute_cluster_mass_shift(
    U_source_current: np.ndarray,
    U_candidate_current: np.ndarray,
    validate: bool = True,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """Compute fuzzy cluster-mass redistribution signal D_M and mass profiles.

    Parameters
    ----------
    U_source_current : np.ndarray
        Shape (B, K), memberships of deployed source model on current probe bank A_t^C.
    U_candidate_current : np.ndarray
        Shape (B, K), memberships of aligned candidate model on current probe bank A_t^C.
    validate : bool
        Whether to validate simplex properties.
    tol : float
        Tolerance for simplex check.

    Returns
    -------
    result : Dict[str, Any]
        Contains:
        - mass_source_current: List[float]
        - mass_candidate_current: List[float]
        - D_M: float in [0, 1]
    """
    U0 = np.asarray(U_source_current, dtype=np.float64)
    Ut = np.asarray(U_candidate_current, dtype=np.float64)

    if U0.shape != Ut.shape:
        raise ValueError(f"Shape mismatch: source {U0.shape} vs candidate {Ut.shape}")

    p0 = compute_cluster_mass_distribution(U0, validate=validate, tol=tol)
    pt = compute_cluster_mass_distribution(Ut, validate=validate, tol=tol)

    d_m = compute_normalized_js_divergence(p0, pt, validate=True, tol=tol)

    return {
        "mass_source_current": [round(float(x), 7) for x in p0],
        "mass_candidate_current": [round(float(x), 7) for x in pt],
        "D_M": round(float(d_m), 7),
    }
