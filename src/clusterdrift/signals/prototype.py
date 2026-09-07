"""Prototype Movement Signal (D_V).

Mathematical definition:
    Using Phase-5 reference cluster scales s_k^R and aligned candidate centers v_tilde_t,k:
    delta_k = ||v_{0, k} - v_tilde_t, k||_2 / (s_k^R + epsilon)
    
    Primary signal:
    D_V = (1 / K) * sum_k delta_k >= 0
"""

from typing import Dict
import numpy as np


def compute_prototype_movement(
    centers_source: np.ndarray,
    centers_candidate_aligned: np.ndarray,
    reference_scales: np.ndarray,
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """Compute reference-normalized prototype movement D_V.

    Parameters
    ----------
    centers_source : np.ndarray
        Shape (K, D), cluster centers of deployed source model.
    centers_candidate_aligned : np.ndarray
        Shape (K, D), aligned cluster centers of candidate model.
    reference_scales : np.ndarray
        Shape (K,), frozen reference cluster scales from source model / source data.
    epsilon : float
        Numerical stabilization constant for cluster radius denominator.

    Returns
    -------
    result : Dict[str, float]
        Contains D_V (mean), D_V_median, D_V_max.
    """
    V0 = np.asarray(centers_source, dtype=np.float64)
    Vt = np.asarray(centers_candidate_aligned, dtype=np.float64)
    scales = np.asarray(reference_scales, dtype=np.float64)

    if V0.shape != Vt.shape:
        raise ValueError(f"Prototype center shapes mismatch: source {V0.shape} vs candidate {Vt.shape}")
    if len(scales) != V0.shape[0]:
        raise ValueError(f"Scales length {len(scales)} does not match cluster count {V0.shape[0]}")

    diff = V0 - Vt
    eucl_distances = np.linalg.norm(diff, axis=1)
    normalized_displacements = eucl_distances / (scales + epsilon)

    mean_dv = float(np.mean(normalized_displacements))
    med_dv = float(np.median(normalized_displacements))
    max_dv = float(np.max(normalized_displacements))

    return {
        "D_V": round(mean_dv, 7),
        "D_V_median": round(med_dv, 7),
        "D_V_max": round(max_dv, 7),
    }
