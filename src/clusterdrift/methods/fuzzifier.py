"""Fuzzifier policy module for fuzzy clustering algorithms.

Implements fixed and literature-grounded dimension-adaptive fuzzifier policies,
numerical floor management, and structured policy metadata resolution.

References
----------
Winkler, R., Klawonn, F., & Kruse, R. (2010). Problems of Fuzzy c-Means Clustering
and Similar Algorithms with High Dimensional Data Sets. In: Foundations of Reasoning
under Uncertainty, pp. 79-96.
Winkler, R., Klawonn, F., & Kruse, R. (2011). Fuzzy c-means in high dimensional spaces.
International Journal of Fuzzy Systems, 13(1), 1-6.
Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms.
Plenum Press, New York.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class FuzzifierResolution:
    """Structured metadata describing the resolved fuzzifier exponent."""

    policy: str
    requested_m: Optional[float]
    effective_m: float
    dimension: int
    clipped: bool
    formula: str
    reference: str

    def to_dict(self) -> dict[str, Any]:
        """Convert resolution metadata to a serializable dictionary."""
        return {
            "policy": self.policy,
            "requested_m": self.requested_m,
            "effective_m": self.effective_m,
            "dimension": self.dimension,
            "clipped": self.clipped,
            "formula": self.formula,
            "reference": self.reference,
        }


def resolve_fuzzifier(
    policy: str = "fixed",
    value: Optional[float] = 2.0,
    n_features: Optional[int] = None,
    **kwargs: Any,
) -> FuzzifierResolution:
    """Resolve fuzzy weighting exponent m with explicit policy and numerical guardrails.

    Parameters
    ----------
    policy : str, default="fixed"
        Fuzzifier policy:
        - "fixed": Uses specified constant `value`.
        - "dimension_adaptive" or "winkler_dimension_rule":
          Sets m_D = max(1.01, 1 + 2 / D), where D = n_features.
    value : float or None, default=2.0
        Requested m value when policy="fixed". Must be > 1.0.
    n_features : int or None, default=None
        Dimensionality D of the preprocessed feature space X. Required for dimension-adaptive policies.
    **kwargs : Any
        Forbidden additional arguments to strictly prevent label leakage.

    Returns
    -------
    FuzzifierResolution
        Structured resolution object containing effective m and metadata.

    Raises
    ------
    TypeError
        If any label or target arguments are passed.
    ValueError
        If policy is unknown, m <= 1.0, or n_features <= 0.
    """
    if kwargs:
        raise TypeError(
            f"resolve_fuzzifier accepts no extra keyword arguments, got {list(kwargs.keys())}. "
            "Supervised labels or target arguments are strictly prohibited."
        )

    norm_policy = policy.lower().strip()

    if norm_policy == "fixed":
        if value is None:
            raise ValueError("value must be specified when policy='fixed'")
        val_float = float(value)
        if val_float <= 1.0:
            raise ValueError(f"Fuzzifier exponent m must be > 1.0, got {val_float}")

        D = int(n_features) if n_features is not None else 0
        return FuzzifierResolution(
            policy="fixed",
            requested_m=val_float,
            effective_m=val_float,
            dimension=D,
            clipped=False,
            formula="m = const",
            reference="Standard Fuzzy C-Means (Bezdek 1981)",
        )

    elif norm_policy in ("dimension_adaptive", "winkler_dimension_rule"):
        if n_features is None:
            raise ValueError(
                f"n_features (dimension D) is required for fuzzifier policy '{policy}'"
            )
        D = int(n_features)
        if D <= 0:
            raise ValueError(f"n_features must be positive integer, got {D}")

        raw_m = 1.0 + (2.0 / float(D))
        # Documented numerical floor of 1.01 for floating-point log-domain safety
        if raw_m < 1.01:
            effective_m = 1.01
            clipped = True
        else:
            effective_m = raw_m
            clipped = False

        return FuzzifierResolution(
            policy="dimension_adaptive",
            requested_m=raw_m,
            effective_m=effective_m,
            dimension=D,
            clipped=clipped,
            formula="m = max(1.01, 1 + 2 / D)",
            reference="Winkler, Klawonn & Kruse (2010, 2011), 'Problems of Fuzzy c-Means Clustering and Similar Algorithms with High Dimensional Data Sets'",
        )

    else:
        raise ValueError(
            f"Unknown fuzzifier policy '{policy}'. Supported policies: 'fixed', 'dimension_adaptive', 'winkler_dimension_rule'."
        )
