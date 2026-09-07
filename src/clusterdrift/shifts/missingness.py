"""Missing Completely At Random (MCAR) shift generator with nested severity masks."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from clusterdrift.shifts.base import ShiftResult
from clusterdrift.shifts.hashing import compute_bytes_sha256


def apply_mcar_shift(
    X_target: pd.DataFrame,
    clustering_cols: List[str],
    cfg: Dict[str, Any],
    severity: str,
    cell_seed: int,
) -> ShiftResult:
    """Apply Missing Completely At Random (MCAR) shift on raw target features before preprocessing.
    
    Eligible cells: currently observed cells across all clustering columns.
    Nested design: mild masks first 10% of observed cells, severe masks first 30%.
    Mask_mild is a strict subset of Mask_severe.
    Already-missing cells are not counted as newly masked.
    """
    n_rows = len(X_target)
    row_map = np.arange(n_rows, dtype=np.int64)
    cols = [c for c in clustering_cols if c in X_target.columns]

    if not cols:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=row_map,
            metadata={"reason": "No clustering feature columns found"},
            status="NOT_APPLICABLE",
            reason="No clustering feature columns found",
        )

    # 1. Identify all observed cell coordinates (row_idx, col_idx)
    observed_cells: List[tuple] = []
    for c_idx, col in enumerate(cols):
        series = X_target[col]
        # Check non-null
        notna_mask = pd.notna(series).to_numpy()
        for r_idx in np.where(notna_mask)[0]:
            observed_cells.append((int(r_idx), int(c_idx)))

    n_obs = len(observed_cells)
    if n_obs == 0:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=row_map,
            metadata={"reason": "Target has zero observed cells in clustering columns"},
            status="NOT_APPLICABLE",
            reason="Target has zero observed cells in clustering columns",
        )

    mcar_cfg = cfg.get("mcar", {})
    severities_cfg = mcar_cfg.get("severities", {"mild": 0.10, "severe": 0.30})
    frac = float(severities_cfg.get(severity, 0.10 if severity == "mild" else 0.30))

    # 2. Deterministic permutation of observed cell coordinates
    rng = np.random.default_rng(cell_seed)
    perm = rng.permutation(n_obs)

    n_mild = max(1, round(severities_cfg.get("mild", 0.10) * n_obs))
    n_severe = max(n_mild, round(severities_cfg.get("severe", 0.30) * n_obs))
    n_mask = n_mild if severity == "mild" else n_severe

    chosen_indices = perm[:n_mask]
    
    # Construct binary mask (N, len(cols)) for hashing and application
    mask_array = np.zeros((n_rows, len(cols)), dtype=bool)
    per_feature_counts: Dict[str, int] = {c: 0 for c in cols}

    for idx in chosen_indices:
        r, c = observed_cells[idx]
        mask_array[r, c] = True
        per_feature_counts[cols[c]] += 1

    mask_hash = compute_bytes_sha256(mask_array.tobytes())

    # 3. Apply mask to target dataframe
    X_shifted = X_target.copy()
    for c_idx, col in enumerate(cols):
        col_mask = mask_array[:, c_idx]
        if np.any(col_mask):
            # Check column dtype to assign appropriate missing marker
            if pd.api.types.is_float_dtype(X_shifted[col]):
                X_shifted.loc[col_mask, col] = np.nan
            elif pd.api.types.is_integer_dtype(X_shifted[col]):
                # Convert integer to float or nullable Int
                X_shifted[col] = X_shifted[col].astype(np.float64)
                X_shifted.loc[col_mask, col] = np.nan
            elif pd.api.types.is_bool_dtype(X_shifted[col]):
                X_shifted[col] = X_shifted[col].astype(object)
                X_shifted.loc[col_mask, col] = np.nan
            else:
                # Object / categorical / string
                X_shifted.loc[col_mask, col] = np.nan

    realized_fraction = float(n_mask / n_obs)

    return ShiftResult(
        X_shifted=X_shifted,
        row_index_map=row_map,
        metadata={
            "family": "mcar",
            "severity": severity,
            "target_missing_fraction": frac,
            "eligible_cell_count": n_obs,
            "newly_masked_cell_count": n_mask,
            "realized_new_missing_fraction": realized_fraction,
            "per_feature_added_missingness": per_feature_counts,
            "mask_sha256": mask_hash,
        },
        status="APPLICABLE",
    )
