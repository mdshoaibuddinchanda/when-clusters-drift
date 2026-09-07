"""Probe evaluation containers and identity guards.

Enforces the Core Invariant:
    U_a(X_a) - U_b(X_b) is strictly forbidden when X_a != X_b.
All membership comparisons across models must evaluate on the EXACT SAME probe bank.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


class ProbeIdentityMismatchError(ValueError):
    """Raised when an operation attempts to compare evaluations across different probe banks."""
    pass


@dataclass(frozen=True)
class ProbeEvaluation:
    """Immutable record of a clustering model evaluated on a specific probe bank."""
    bank_sha256: str
    bank_type: str  # "reference" or "current"
    dataset_id: str
    outer_fold: int
    condition: str  # "reference" for reference banks, or scenario condition name
    model_fingerprint: str
    memberships: np.ndarray  # Shape: (B, K)
    n_samples: int
    n_clusters: int

    def __post_init__(self):
        if self.memberships.ndim != 2:
            raise ValueError(f"memberships must be 2D, got shape {self.memberships.shape}")
        if self.memberships.shape[0] != self.n_samples:
            raise ValueError(
                f"memberships rows {self.memberships.shape[0]} does not match n_samples {self.n_samples}"
            )
        if self.memberships.shape[1] != self.n_clusters:
            raise ValueError(
                f"memberships columns {self.memberships.shape[1]} does not match n_clusters {self.n_clusters}"
            )
        if not np.all(np.isfinite(self.memberships)):
            raise ValueError("memberships contains non-finite values (NaN/Inf)")


def verify_probe_identity(eval_a: ProbeEvaluation, eval_b: ProbeEvaluation) -> None:
    """Verify that two probe evaluations share identical sample representations.

    Raises
    ------
    ProbeIdentityMismatchError
        If evaluations were performed on different probe banks (bank_sha256 mismatch).
    """
    if eval_a.bank_sha256 != eval_b.bank_sha256:
        raise ProbeIdentityMismatchError(
            f"Core Invariant Violation: Attempted to compare memberships from different probe banks!\n"
            f"  Evaluation A: dataset={eval_a.dataset_id}, fold={eval_a.outer_fold}, "
            f"type={eval_a.bank_type}, bank_hash={eval_a.bank_sha256[:16]}...\n"
            f"  Evaluation B: dataset={eval_b.dataset_id}, fold={eval_b.outer_fold}, "
            f"type={eval_b.bank_type}, bank_hash={eval_b.bank_sha256[:16]}...\n"
            "All model comparisons must evaluate on the exact same probe points."
        )
