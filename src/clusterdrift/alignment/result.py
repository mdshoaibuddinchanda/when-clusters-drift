"""Data structures for cluster alignment results.

Guarantees non-mutation: encapsulates reordered prototypes and memberships without
altering the original fitted models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    """Immutable result of cluster bipartite alignment via Hungarian algorithm."""
    permutation: np.ndarray  # Shape: (K,), permutation[k_ref] = j_cand
    inverse_permutation: np.ndarray  # Shape: (K,), inverse_permutation[j_cand] = k_ref
    aligned_centers: np.ndarray  # Shape: (K, D), candidate centers reordered
    aligned_memberships: Optional[np.ndarray]  # Shape: (B, K), candidate memberships reordered
    center_cost_matrix: np.ndarray  # Shape: (K, K)
    overlap_matrix: np.ndarray  # Shape: (K, K)
    combined_cost_matrix: np.ndarray  # Shape: (K, K)
    assignment_cost: float
    identity_assignment_cost: float
    cluster_scales: np.ndarray  # Shape: (K,)
    ambiguous: bool
    minimum_assignment_margin: float
    ambiguity_details: Dict[str, Any] = field(default_factory=dict)
    eta: float = 0.50
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_identity(self) -> bool:
        """Check whether alignment is the identity permutation."""
        return np.array_equal(self.permutation, np.arange(len(self.permutation)))
