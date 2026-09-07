"""Dataset Ingestion, Join Boundary, and Feature Isolation for Phase 7."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from clusterdrift.falsification.protocol import CONDITION_TO_FAMILY
from clusterdrift.shifts.hashing import atomic_write_csv


FORBIDDEN_PREDICTOR_FEATURES = {
    "y",
    "label",
    "labels",
    "ari_clean",
    "ari_condition",
    "delta_ari",
    "nmi_condition",
    "ami_condition",
    "dataset_id",
    "shift_family",
    "severity",
    "condition",
    "outer_fold",
    "method",
    "seed",
    "K",
    "N",
    "D",
}


def get_dataset_dimensions(project_root: Path, datasets: List[str]) -> Dict[str, int]:
    """Inspect dataset dimension D (number of features) from canonical features.parquet."""
    dims = {}
    for ds in datasets:
        feat_p = project_root / "data" / "canonical" / "controlled" / ds / "features.parquet"
        if not feat_p.exists():
            raise FileNotFoundError(f"Canonical features not found for {ds}: {feat_p}")
        df = pd.read_parquet(feat_p)
        dims[ds] = df.shape[1]
    return dims


def assign_dimension_stratum(D: int) -> str:
    """Pre-registered dimensional grouping: low <= 20, mid 21-100, high > 100."""
    if D <= 20:
        return "low_D"
    elif D <= 100:
        return "mid_D"
    else:
        return "high_D"


def join_signals_and_quality(
    signals_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    dataset_dims: Dict[str, int],
) -> pd.DataFrame:
    """Join signals and evaluation-only quality targets on the exact 5-coordinate key.

    Strictly validates 1:1 join without duplicates or dropped rows.
    """
    key_cols = ["dataset_id", "outer_fold", "condition", "method", "seed"]

    # Validate keys uniqueness
    if signals_df.duplicated(subset=key_cols).any():
        raise ValueError("Duplicate scenario keys found in signals dataframe")
    if quality_df.duplicated(subset=key_cols).any():
        raise ValueError("Duplicate scenario keys found in quality dataframe")

    if len(signals_df) != len(quality_df):
        raise ValueError(
            f"Row count mismatch between signals ({len(signals_df)}) and quality ({len(quality_df)})"
        )

    # Required quality columns to join
    qual_cols_to_join = [
        "dataset_id",
        "outer_fold",
        "condition",
        "method",
        "seed",
        "ari_clean",
        "ari_condition",
        "delta_ari",
        "nmi_condition",
        "ami_condition",
        "n_evaluation_rows",
    ]
    sub_qual = quality_df[qual_cols_to_join].copy()

    joined = pd.merge(signals_df, sub_qual, on=key_cols, how="inner")
    if len(joined) != len(signals_df):
        raise ValueError(
            f"Joined table row count ({len(joined)}) does not match signals count ({len(signals_df)})"
        )

    # Add deterministic shift family and severity
    families = []
    severities = []
    for cond in joined["condition"]:
        if cond not in CONDITION_TO_FAMILY:
            raise KeyError(f"Unknown condition: {cond}")
        fam, sev = CONDITION_TO_FAMILY[cond]
        families.append(fam)
        severities.append(sev)

    joined["shift_family"] = families
    joined["severity"] = severities

    # Add dimensionality and dimension stratum
    dims = [dataset_dims.get(ds, 0) for ds in joined["dataset_id"]]
    strata = [assign_dimension_stratum(d) for d in dims]
    joined["dimension_D"] = dims
    joined["dimension_stratum"] = strata

    # Sort deterministically
    joined.sort_values(by=key_cols, inplace=True)
    joined.reset_index(drop=True, inplace=True)
    return joined


def validate_feature_block_isolation(features: List[str]) -> None:
    """Verify that feature block contains ONLY allowed numerical signals/controls."""
    for feat in features:
        if feat in FORBIDDEN_PREDICTOR_FEATURES:
            raise ValueError(
                f"Forbidden metadata or ground-truth feature '{feat}' found in predictor block!"
            )
