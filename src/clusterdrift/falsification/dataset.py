"""Dataset Ingestion, Join Boundary, and Feature Isolation for Phase 7."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from clusterdrift.falsification.protocol import CONDITION_TO_FAMILY
from clusterdrift.shifts.hashing import atomic_write_csv, compute_file_sha256


FORBIDDEN_PREDICTOR_FEATURES = {
    "y",
    "label",
    "labels",
    "quality labels",
    "ari_clean",
    "ari_condition",
    "delta_ari",
    "nmi_condition",
    "ami_condition",
    "quality_target",
    "quality_record_sha256",
    "dataset_id",
    "shift_family",
    "severity",
    "condition",
    "outer_fold",
    "method",
    "seed",
    "k",
    "n",
    "d",
    "dimension_d",
    "dimension_stratum",
    "source_model_fingerprint",
    "source_model_fingerprint_file_sha256",
    "shift_spec_sha256",
    "shift_replay_sha256",
    "shift_spec_file_sha256",
    "probe_spec_sha256",
    "probe_spec_file_sha256",
    "alignment_doc_sha256",
    "alignment_doc_file_sha256",
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

    Strictly validates 1:1 join without duplicates, missing, or unexpected keys.
    Validates exact numerical identity for all quality fields against quality_df.
    """
    key_cols = ["dataset_id", "outer_fold", "condition", "method", "seed"]

    # Validate keys uniqueness
    if signals_df.duplicated(subset=key_cols).any():
        dups = signals_df[signals_df.duplicated(subset=key_cols, keep=False)]
        raise ValueError(f"Duplicate scenario keys found in signals dataframe: {len(dups)} rows")
    if quality_df.duplicated(subset=key_cols).any():
        dups = quality_df[quality_df.duplicated(subset=key_cols, keep=False)]
        raise ValueError(f"Duplicate scenario keys found in quality dataframe: {len(dups)} rows")

    # Validate key set equality
    sig_keys = set(zip(
        signals_df["dataset_id"],
        signals_df["outer_fold"].astype(int),
        signals_df["condition"],
        signals_df["method"],
        signals_df["seed"].astype(int),
    ))
    qual_keys = set(zip(
        quality_df["dataset_id"],
        quality_df["outer_fold"].astype(int),
        quality_df["condition"],
        quality_df["method"],
        quality_df["seed"].astype(int),
    ))

    if len(sig_keys) != 4500:
        raise ValueError(f"Expected 4500 unique signals keys, got {len(sig_keys)}")
    if len(qual_keys) != 4500:
        raise ValueError(f"Expected 4500 unique quality keys, got {len(qual_keys)}")
    if sig_keys != qual_keys:
        missing = qual_keys - sig_keys
        unexpected = sig_keys - qual_keys
        raise ValueError(f"Key universe mismatch between signals and quality: missing={len(missing)}, unexpected={len(unexpected)}")

    # Sort quality table deterministically
    qual_sorted = quality_df.sort_values(by=key_cols).reset_index(drop=True)

    # Required quality columns to join
    qual_cols_to_join = [
        "dataset_id",
        "outer_fold",
        "condition",
        "method",
        "seed",
        "quality_target",
        "ari_clean",
        "ari_condition",
        "delta_ari",
        "nmi_condition",
        "ami_condition",
        "n_evaluation_rows",
        "quality_record_sha256",
    ]
    sub_qual = qual_sorted[qual_cols_to_join].copy()

    joined = pd.merge(signals_df, sub_qual, on=key_cols, how="inner")
    if len(joined) != 4500:
        raise ValueError(
            f"Joined table row count ({len(joined)}) does not equal 4500"
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

    # Verify numerical identity against quality_df
    for col in ["ari_clean", "ari_condition", "delta_ari", "nmi_condition", "ami_condition", "n_evaluation_rows"]:
        max_diff = np.max(np.abs(joined[col].values - qual_sorted[col].values))
        if max_diff > 1e-12:
            raise ValueError(f"Joined quality column {col} deviates from Pass B quality table by {max_diff}")

    return joined


def validate_feature_block_isolation(features: List[str]) -> None:
    """Verify that feature block contains ONLY allowed numerical signals/controls."""
    for feat in features:
        feat_clean = str(feat).strip().lower()
        if feat_clean in FORBIDDEN_PREDICTOR_FEATURES or feat in FORBIDDEN_PREDICTOR_FEATURES:
            raise ValueError(
                f"Forbidden metadata or ground-truth feature '{feat}' found in predictor block!"
            )


def build_and_persist_joined_table(project_root: Path, cfg: Optional[Dict[str, Any]] = None) -> Tuple[Path, str]:
    """Execute FORM 1.3 Exact Join Gate and persist results/falsification/joined_evaluation_table.csv."""
    root = Path(project_root).resolve()
    sig_p = root / "results" / "falsification" / "signals_label_free.csv"
    qual_p = root / "results" / "falsification" / "quality_evaluation_only.csv"
    out_p = root / "results" / "falsification" / "joined_evaluation_table.csv"

    if not sig_p.exists():
        raise FileNotFoundError(f"Pass A signals CSV not found: {sig_p}")
    if not qual_p.exists():
        raise FileNotFoundError(f"Pass B quality CSV not found: {qual_p}")

    df_sig = pd.read_csv(sig_p)
    df_qual = pd.read_csv(qual_p)

    if cfg is None:
        import yaml
        with open(root / "configs" / "falsification.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    dims = get_dataset_dimensions(root, cfg["datasets"])
    joined_df = join_signals_and_quality(df_sig, df_qual, dims)

    atomic_write_csv(out_p, joined_df)
    joined_sha = compute_file_sha256(out_p)
    return out_p, joined_sha


def build_joined_evaluation_table(
    signals_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    dataset_dims: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Build joined evaluation table with default dummy dimensions if not provided."""
    if dataset_dims is None:
        dataset_dims = {ds: 0 for ds in signals_df["dataset_id"].unique()}
    return join_signals_and_quality(signals_df, quality_df, dataset_dims)


def get_primary_evaluation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return primary falsification evaluation dataset (excluding condition 'clean')."""
    return df[df["condition"] != "clean"].copy().reset_index(drop=True)


def get_secondary_evaluation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return secondary evaluation dataset (including condition 'clean')."""
    return df.copy().reset_index(drop=True)


def get_feature_matrix_and_target(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: str = "delta_ari",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and target after verifying feature block isolation."""
    validate_feature_block_isolation(feature_names)
    X = df[feature_names].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    return X, y
