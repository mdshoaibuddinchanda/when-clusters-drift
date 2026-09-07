"""Cluster state representations and model fingerprinting.

Represents clustering models (FCM, GMM, KMeans) as immutable cluster states
containing prototype centers and membership semantics without label dependencies.
"""

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Optional
import numpy as np


@dataclass(frozen=True)
class ClusterState:
    """Immutable representation of a clustering model's state."""
    centers: np.ndarray  # Shape: (K, D)
    n_clusters: int
    n_features: int
    membership_semantics: str  # e.g. "fuzzy", "probability", "one_hot"
    model_fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.centers.ndim != 2:
            raise ValueError(f"centers must be 2D array, got shape {self.centers.shape}")
        if self.centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"centers row count {self.centers.shape[0]} does not match n_clusters {self.n_clusters}"
            )
        if self.centers.shape[1] != self.n_features:
            raise ValueError(
                f"centers feature count {self.centers.shape[1]} does not match n_features {self.n_features}"
            )
        if not np.all(np.isfinite(self.centers)):
            raise ValueError("centers must contain only finite numerical values (no NaN/Inf)")


def compute_model_fingerprint(
    method_name: str,
    config: Dict[str, Any],
    seed: int,
    centers: np.ndarray,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute deterministic SHA-256 fingerprint for a fitted clustering model state."""
    hasher = hashlib.sha256()
    payload = {
        "method_name": method_name,
        "config": config,
        "seed": seed,
        "n_clusters": int(centers.shape[0]),
        "n_features": int(centers.shape[1]),
    }
    if extra_payload:
        payload["extra"] = extra_payload

    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    hasher.update(canonical_json.encode("utf-8"))
    # Bind exact prototype center bytes
    hasher.update(np.ascontiguousarray(centers, dtype=np.float64).tobytes())
    return hasher.hexdigest()
