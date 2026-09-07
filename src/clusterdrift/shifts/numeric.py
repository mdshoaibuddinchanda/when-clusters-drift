"""Numeric controlled distribution shifts: Location, Scale, Measurement Noise, and Outliers."""

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from clusterdrift.shifts.base import ShiftResult, SourceStatistics
from clusterdrift.shifts.backend import apply_affine_transform, apply_additive_noise, should_use_gpu
from clusterdrift.shifts.hashing import compute_bytes_sha256


def apply_location_shift(
    X_target: pd.DataFrame,
    source_stats: SourceStatistics,
    cfg: Dict[str, Any],
    severity: str,
    feature_seed: int,
    backend: str = "auto",
) -> ShiftResult:
    """Apply controlled Location Shift to raw target features.
    
    Selects q = max(1, ceil(0.30 * D_eligible)) numeric features with non-zero source variance.
    Uses the exact same features for mild and severe conditions.
    Shifts selected features by delta * sigma_source.
    Existing missing values remain missing.
    """
    n_rows = len(X_target)
    row_map = np.arange(n_rows, dtype=np.int64)
    eligible_cols = sorted(source_stats.non_zero_variance_cols)
    
    if not eligible_cols:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=row_map,
            metadata={"reason": "No eligible non-zero-variance numeric features"},
            status="NOT_APPLICABLE",
            reason="No eligible non-zero-variance numeric features",
        )

    loc_cfg = cfg.get("location", {})
    feature_fraction = loc_cfg.get("feature_fraction", 0.30)
    severities_cfg = loc_cfg.get("severities", {"mild": 0.5, "severe": 1.0})
    delta = float(severities_cfg.get(severity, 0.5 if severity == "mild" else 1.0))

    q = max(1, math.ceil(feature_fraction * len(eligible_cols)))
    rng = np.random.default_rng(feature_seed)
    perm = rng.permutation(len(eligible_cols))
    selected_cols = [eligible_cols[idx] for idx in perm[:q]]

    X_shifted = X_target.copy()
    use_gpu = should_use_gpu(n_rows, len(selected_cols), backend=backend, min_cells=cfg.get("execution", {}).get("gpu_min_numeric_cells", 2000000))

    sub_arr = X_shifted[selected_cols].to_numpy(dtype=np.float64)
    sds_arr = np.array([source_stats.sds[c] for c in selected_cols], dtype=np.float64)
    offsets = delta * sds_arr

    transformed_sub = apply_affine_transform(
        sub_arr,
        scale=1.0,
        shift=offsets,
        use_gpu=use_gpu,
        chunk_rows=cfg.get("execution", {}).get("chunk_rows", 65536),
    )

    for idx, col in enumerate(selected_cols):
        X_shifted[col] = transformed_sub[:, idx]

    return ShiftResult(
        X_shifted=X_shifted,
        row_index_map=row_map,
        metadata={
            "family": "location",
            "severity": severity,
            "delta": delta,
            "feature_fraction": feature_fraction,
            "eligible_feature_count": len(eligible_cols),
            "selected_feature_count": len(selected_cols),
            "selected_features": selected_cols,
            "source_sds": {c: source_stats.sds[c] for c in selected_cols},
            "realized_standardized_displacement": delta,
        },
        status="APPLICABLE",
    )


def apply_scale_shift(
    X_target: pd.DataFrame,
    source_stats: SourceStatistics,
    cfg: Dict[str, Any],
    severity: str,
    backend: str = "auto",
) -> ShiftResult:
    """Apply controlled Scale Shift to raw target features.
    
    Scales all eligible numeric features around their outer-source mean:
        x' = mu_source + a * (x - mu_source)
    Zero-variance columns remain untouched. Missing values remain missing.
    """
    n_rows = len(X_target)
    row_map = np.arange(n_rows, dtype=np.int64)
    eligible_cols = sorted(source_stats.non_zero_variance_cols)

    if not eligible_cols:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=row_map,
            metadata={"reason": "No eligible non-zero-variance numeric features"},
            status="NOT_APPLICABLE",
            reason="No eligible non-zero-variance numeric features",
        )

    scale_cfg = cfg.get("scale", {})
    severities_cfg = scale_cfg.get("severities", {"mild": 1.25, "severe": 1.75})
    a = float(severities_cfg.get(severity, 1.25 if severity == "mild" else 1.75))

    X_shifted = X_target.copy()
    use_gpu = should_use_gpu(n_rows, len(eligible_cols), backend=backend, min_cells=cfg.get("execution", {}).get("gpu_min_numeric_cells", 2000000))

    sub_arr = X_shifted[eligible_cols].to_numpy(dtype=np.float64)
    means_arr = np.array([source_stats.means[c] for c in eligible_cols], dtype=np.float64)
    # x' = a * x + (1 - a) * mu_source
    scale = a
    shift = (1.0 - a) * means_arr

    transformed_sub = apply_affine_transform(
        sub_arr,
        scale=scale,
        shift=shift,
        use_gpu=use_gpu,
        chunk_rows=cfg.get("execution", {}).get("chunk_rows", 65536),
    )

    for idx, col in enumerate(eligible_cols):
        X_shifted[col] = transformed_sub[:, idx]

    return ShiftResult(
        X_shifted=X_shifted,
        row_index_map=row_map,
        metadata={
            "family": "scale",
            "severity": severity,
            "scale_factor": a,
            "eligible_feature_count": len(eligible_cols),
            "eligible_features": eligible_cols,
            "source_means": {c: source_stats.means[c] for c in eligible_cols},
            "realized_scale_measure": a,
        },
        status="APPLICABLE",
    )


def apply_measurement_noise(
    X_target: pd.DataFrame,
    source_stats: SourceStatistics,
    cfg: Dict[str, Any],
    severity: str,
    tensor_seed: int,
    backend: str = "auto",
) -> ShiftResult:
    """Apply Gaussian measurement noise scaled by outer-source standard deviation.
    
    Generates one standard normal epsilon ~ N(0, 1) tensor on CPU deterministically.
    Applies x' = x + s * sigma_source * epsilon.
    Same epsilon tensor is used for mild (s=0.10) and severe (s=0.30).
    """
    n_rows = len(X_target)
    row_map = np.arange(n_rows, dtype=np.int64)
    eligible_cols = sorted(source_stats.non_zero_variance_cols)

    if not eligible_cols:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=row_map,
            metadata={"reason": "No eligible non-zero-variance numeric features"},
            status="NOT_APPLICABLE",
            reason="No eligible non-zero-variance numeric features",
        )

    noise_cfg = cfg.get("measurement_noise", {})
    severities_cfg = noise_cfg.get("severities", {"mild": 0.10, "severe": 0.30})
    s = float(severities_cfg.get(severity, 0.10 if severity == "mild" else 0.30))

    # Canonical randomness rule: generate Gaussian tensor deterministically on CPU
    rng = np.random.default_rng(tensor_seed)
    epsilon = rng.standard_normal(size=(n_rows, len(eligible_cols))).astype(np.float64)
    epsilon_hash = compute_bytes_sha256(epsilon.tobytes())

    X_shifted = X_target.copy()
    use_gpu = should_use_gpu(n_rows, len(eligible_cols), backend=backend, min_cells=cfg.get("execution", {}).get("gpu_min_numeric_cells", 2000000))

    sub_arr = X_shifted[eligible_cols].to_numpy(dtype=np.float64)
    sds_arr = np.array([source_stats.sds[c] for c in eligible_cols], dtype=np.float64)
    scale_vec = s * sds_arr

    transformed_sub = apply_additive_noise(
        sub_arr,
        noise_tensor=epsilon,
        scale=scale_vec,
        use_gpu=use_gpu,
        chunk_rows=cfg.get("execution", {}).get("chunk_rows", 65536),
    )

    for idx, col in enumerate(eligible_cols):
        X_shifted[col] = transformed_sub[:, idx]

    # Compute realized RMS standardized perturbation on observed cells
    diff = (transformed_sub - sub_arr) / sds_arr
    valid_mask = np.isfinite(diff)
    rms = float(np.sqrt(np.mean(diff[valid_mask] ** 2))) if np.any(valid_mask) else s

    return ShiftResult(
        X_shifted=X_shifted,
        row_index_map=row_map,
        metadata={
            "family": "measurement_noise",
            "severity": severity,
            "noise_multiplier": s,
            "noise_seed": tensor_seed,
            "noise_sha256": epsilon_hash,
            "eligible_feature_count": len(eligible_cols),
            "rms_standardized_perturbation": rms,
        },
        status="APPLICABLE",
    )


def apply_outlier_shift(
    X_target: pd.DataFrame,
    source_stats: SourceStatistics,
    cfg: Dict[str, Any],
    severity: str,
    row_seed: int,
    tensor_seed: int,
    backend: str = "auto",
) -> ShiftResult:
    """Apply heavy-tailed Student-t outliers to nested target row subsets.
    
    Mild affects first 5% of target rows; severe affects first 15% (strict subset).
    Perturbation: x' = x + 3.0 * sigma_source * t_3.
    """
    n_rows = len(X_target)
    row_map = np.arange(n_rows, dtype=np.int64)
    eligible_cols = sorted(source_stats.non_zero_variance_cols)

    if not eligible_cols:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=row_map,
            metadata={"reason": "No eligible non-zero-variance numeric features"},
            status="NOT_APPLICABLE",
            reason="No eligible non-zero-variance numeric features",
        )

    out_cfg = cfg.get("outliers", {})
    row_fracs = out_cfg.get("row_fraction", {"mild": 0.05, "severe": 0.15})
    frac = float(row_fracs.get(severity, 0.05 if severity == "mild" else 0.15))
    df_t = int(out_cfg.get("df", 3))
    shock_scale = float(out_cfg.get("shock_scale_source_sd", 3.0))

    # 1. Deterministic row selection: mild is strict subset of severe
    rng_row = np.random.default_rng(row_seed)
    row_perm = rng_row.permutation(n_rows)
    
    n_mild = max(1, round(row_fracs.get("mild", 0.05) * n_rows))
    n_severe = max(n_mild, round(row_fracs.get("severe", 0.15) * n_rows))
    n_target_rows = n_mild if severity == "mild" else n_severe
    selected_rows = row_perm[:n_target_rows]
    row_sel_hash = compute_bytes_sha256(selected_rows.tobytes())

    # 2. Deterministic Student-t noise tensor on CPU
    rng_tensor = np.random.default_rng(tensor_seed)
    t_tensor = rng_tensor.standard_t(df=df_t, size=(n_rows, len(eligible_cols))).astype(np.float64)
    tensor_hash = compute_bytes_sha256(t_tensor.tobytes())

    X_shifted = X_target.copy()
    use_gpu = should_use_gpu(len(selected_rows), len(eligible_cols), backend=backend, min_cells=cfg.get("execution", {}).get("gpu_min_numeric_cells", 2000000))

    sub_arr = X_shifted[eligible_cols].to_numpy(dtype=np.float64)
    sds_arr = np.array([source_stats.sds[c] for c in eligible_cols], dtype=np.float64)
    shock_vec = shock_scale * sds_arr

    transformed_sub = apply_additive_noise(
        sub_arr,
        noise_tensor=t_tensor,
        scale=shock_vec,
        row_indices=selected_rows,
        use_gpu=use_gpu,
        chunk_rows=cfg.get("execution", {}).get("chunk_rows", 65536),
    )

    for idx, col in enumerate(eligible_cols):
        X_shifted[col] = transformed_sub[:, idx]

    # Calculate realized standardized displacement on affected rows
    diff = (transformed_sub[selected_rows] - sub_arr[selected_rows]) / sds_arr
    valid_mask = np.isfinite(diff)
    mean_disp = float(np.mean(np.abs(diff[valid_mask]))) if np.any(valid_mask) else shock_scale

    return ShiftResult(
        X_shifted=X_shifted,
        row_index_map=row_map,
        metadata={
            "family": "outliers",
            "severity": severity,
            "row_fraction": frac,
            "selected_row_count": len(selected_rows),
            "student_t_df": df_t,
            "shock_scale": shock_scale,
            "realized_standardized_displacement": mean_disp,
            "row_selection_sha256": row_sel_hash,
            "noise_sha256": tensor_hash,
            "eligible_feature_count": len(eligible_cols),
        },
        status="APPLICABLE",
    )
