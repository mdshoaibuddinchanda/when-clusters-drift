"""Aggregate Structural Drift Signals.

Assembles the 5 structural signals:
    - D_U^R : Historical probe membership drift
    - D_U^C : Current probe membership drift
    - D_V   : Normalized prototype movement
    - D_H   : Current entropy shift
    - D_M   : Fuzzy cluster-mass redistribution
"""

from typing import Any, Dict
import numpy as np

from clusterdrift.signals.entropy import compute_entropy_shift
from clusterdrift.signals.mass import compute_cluster_mass_shift
from clusterdrift.signals.membership import (
    compute_current_membership_drift,
    compute_historical_membership_drift,
)
from clusterdrift.signals.prototype import compute_prototype_movement


def compute_all_structural_signals(
    U_source_reference: np.ndarray,
    U_candidate_reference_aligned: np.ndarray,
    U_source_current: np.ndarray,
    U_candidate_current_aligned: np.ndarray,
    centers_source: np.ndarray,
    centers_candidate_aligned: np.ndarray,
    reference_scales: np.ndarray,
    epsilon_v: float = 1e-12,
    validate_simplex: bool = True,
    simplex_tol: float = 1e-5,
) -> Dict[str, Any]:
    """Compute all five structural signals across reference and current probe banks.

    Returns dictionary containing all primary and auxiliary structural metrics.
    """
    # 1. D_U^R
    dur_stats = compute_historical_membership_drift(
        U_source_reference,
        U_candidate_reference_aligned,
        validate=validate_simplex,
        tol=simplex_tol,
    )

    # 2. D_U^C
    duc_stats = compute_current_membership_drift(
        U_source_current,
        U_candidate_current_aligned,
        validate=validate_simplex,
        tol=simplex_tol,
    )

    # 3. D_V
    dv_stats = compute_prototype_movement(
        centers_source,
        centers_candidate_aligned,
        reference_scales,
        epsilon=epsilon_v,
    )

    # 4. D_H
    dh_stats = compute_entropy_shift(
        U_source_current,
        U_candidate_current_aligned,
        validate=validate_simplex,
        tol=simplex_tol,
    )

    # 5. D_M
    dm_stats = compute_cluster_mass_shift(
        U_source_current,
        U_candidate_current_aligned,
        validate=validate_simplex,
        tol=simplex_tol,
    )

    result = {}
    result.update(dur_stats)
    result.update(duc_stats)
    result.update(dv_stats)
    result.update(dh_stats)
    result.update(dm_stats)
    return result
