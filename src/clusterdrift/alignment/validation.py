"""Input validation and integrity checks for cluster alignment."""

from typing import Optional
import numpy as np

from clusterdrift.alignment.state import ClusterState


def validate_cluster_states(state_ref: ClusterState, state_cand: ClusterState) -> None:
    """Validate compatibility between reference and candidate cluster states."""
    if state_ref.n_clusters != state_cand.n_clusters:
        raise ValueError(
            f"Cluster count mismatch: reference has {state_ref.n_clusters}, "
            f"candidate has {state_cand.n_clusters}. Alignment requires equal cluster count."
        )
    if state_ref.n_features != state_cand.n_features:
        raise ValueError(
            f"Feature dimension mismatch: reference has {state_ref.n_features}, "
            f"candidate has {state_cand.n_features}."
        )


def validate_membership_matrix(
    U: np.ndarray,
    expected_n_samples: Optional[int] = None,
    expected_n_clusters: Optional[int] = None,
    check_simplex: bool = True,
    tol: float = 1e-4,
) -> None:
    """Validate numerical integrity and optional simplex properties of a membership matrix."""
    if not isinstance(U, np.ndarray):
        raise TypeError(f"Membership matrix must be np.ndarray, got {type(U)}")
    if U.ndim != 2:
        raise ValueError(f"Membership matrix must be 2D, got shape {U.shape}")
    if not np.all(np.isfinite(U)):
        raise ValueError("Membership matrix contains non-finite values (NaN/Inf)")

    B, K = U.shape
    if expected_n_samples is not None and B != expected_n_samples:
        raise ValueError(f"Expected {expected_n_samples} samples, got {B}")
    if expected_n_clusters is not None and K != expected_n_clusters:
        raise ValueError(f"Expected {expected_n_clusters} clusters, got {K}")

    if np.any(U < -tol):
        raise ValueError(f"Membership values cannot be negative, found min {np.min(U)}")

    if check_simplex:
        row_sums = np.sum(U, axis=1)
        if not np.allclose(row_sums, 1.0, atol=tol):
            max_dev = float(np.max(np.abs(row_sums - 1.0)))
            raise ValueError(f"Membership rows must sum to 1.0; maximum deviation is {max_dev}")
