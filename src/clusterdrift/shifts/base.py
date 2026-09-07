"""Base data structures, source statistics extraction, and protocol constants for Phase 4 shifts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


SHIFT_FAMILIES = [
    "location",
    "scale",
    "mcar",
    "outliers",
    "measurement_noise",
    "class_prevalence",
    "local_overlap",
]

SEVERITIES = ["mild", "severe"]

ALL_CONDITIONS = ["clean"] + [
    f"{fam}_{sev}" for fam in SHIFT_FAMILIES for sev in SEVERITIES
]


@dataclass
class ShiftResult:
    """Standard container for the outcome of applying a controlled shift intervention.
    
    CRITICAL ARCHITECTURAL RULE:
    ShiftResult intentionally does NOT contain target labels `y`.
    Learner-facing clustering, preprocessing, and signal code only receive `X_shifted`.
    Offline evaluation code tracks row correspondence exclusively via `row_index_map`.
    """
    X_shifted: pd.DataFrame
    row_index_map: np.ndarray  # Shape (N_shifted,), maps shifted row index to original target row index
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "APPLICABLE"  # "APPLICABLE" or "NOT_APPLICABLE"
    reason: Optional[str] = None


@dataclass
class SourceStatistics:
    """Source-only numeric baseline statistics computed strictly on the outer-source partition.
    
    All perturbation scaling, location offsets, scale references, and outlier bounds
    must be computed from these source statistics alone, never from target data.
    """
    numeric_cols: List[str]
    means: Dict[str, float]
    sds: Dict[str, float]
    medians: Dict[str, float]
    zero_variance_cols: List[str]
    non_zero_variance_cols: List[str]
    counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric_cols": self.numeric_cols,
            "means": self.means,
            "sds": self.sds,
            "medians": self.medians,
            "zero_variance_cols": self.zero_variance_cols,
            "non_zero_variance_cols": self.non_zero_variance_cols,
            "counts": self.counts,
        }


def compute_source_statistics(
    X_source: pd.DataFrame,
    roles: Dict[str, str],
    tol: float = 1e-12,
) -> SourceStatistics:
    """Compute float64 source numeric statistics strictly on raw outer-source data.
    
    Parameters
    ----------
    X_source : pd.DataFrame
        Raw outer-source features before any preprocessing or target exposure.
    roles : Dict[str, str]
        Feature roles mapping column names to "numeric", "categorical", etc.
    tol : float, default 1e-12
        Tolerance below which standard deviation is treated as zero variance.
        
    Returns
    -------
    SourceStatistics
        Populated source statistics container.
    """
    numeric_cols = [c for c, r in roles.items() if r == "numeric" and c in X_source.columns]
    
    means: Dict[str, float] = {}
    sds: Dict[str, float] = {}
    medians: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    zero_variance: List[str] = []
    non_zero_variance: List[str] = []

    for col in numeric_cols:
        series = pd.to_numeric(X_source[col], errors="coerce").dropna()
        n_obs = len(series)
        counts[col] = n_obs
        
        if n_obs == 0:
            means[col] = 0.0
            sds[col] = 0.0
            medians[col] = 0.0
            zero_variance.append(col)
        elif n_obs == 1:
            val = float(series.iloc[0])
            means[col] = val
            sds[col] = 0.0
            medians[col] = val
            zero_variance.append(col)
        else:
            arr = series.to_numpy(dtype=np.float64)
            m = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1))
            med = float(np.median(arr))
            means[col] = m
            sds[col] = sd
            medians[col] = med
            if sd <= tol or np.isnan(sd):
                zero_variance.append(col)
            else:
                non_zero_variance.append(col)

    return SourceStatistics(
        numeric_cols=numeric_cols,
        means=means,
        sds=sds,
        medians=medians,
        zero_variance_cols=zero_variance,
        non_zero_variance_cols=non_zero_variance,
        counts=counts,
    )
