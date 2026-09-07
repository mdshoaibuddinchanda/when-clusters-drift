"""Offline supervised distribution shifts: Class Prevalence and Local Structural Overlap.

CRITICAL ARCHITECTURAL BOUNDARY:
Labels (y_source, y_target) are strictly isolated within this module to construct offline
experimental interventions. They are NEVER exposed in ShiftResult or passed to learner-facing
clustering, preprocessing, signal, or adaptation code.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from clusterdrift.shifts.base import ShiftResult, SourceStatistics
from clusterdrift.shifts.backend import apply_convex_interpolation, should_use_gpu
from clusterdrift.shifts.hashing import compute_bytes_sha256


def apply_class_prevalence_shift(
    X_target: pd.DataFrame,
    y_target: np.ndarray,
    y_source: np.ndarray,
    cfg: Dict[str, Any],
    severity: str,
    resample_seed: int,
) -> ShiftResult:
    """Apply controlled Class Prevalence shift via weighted target bootstrap.
    
    Anchor class is the rarest class in the outer source partition.
    Anchor rows receive 2x weight (mild) or 4x weight (severe); other classes receive 1x weight.
    Generates a resampled target partition of exactly N_target rows.
    
    ShiftResult contains:
        X_shifted : resampled feature rows
        row_index_map : indices of selected rows (for offline evaluation label mapping)
        labels : NOT RETURNED
    """
    n_rows = len(X_target)
    default_row_map = np.arange(n_rows, dtype=np.int64)

    # 1. Validate classes
    classes_src, counts_src = np.unique(y_source, return_counts=True)
    classes_tgt, counts_tgt = np.unique(y_target, return_counts=True)

    if len(classes_tgt) < 2:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=default_row_map,
            metadata={"reason": "Fewer than 2 classes represented in target partition"},
            status="NOT_APPLICABLE",
            reason="Fewer than 2 classes represented in target partition",
        )

    # 2. Determine anchor class: rarest in outer source present in target
    # Sort classes deterministically for tie-breaking
    sorted_src_indices = np.argsort(counts_src, kind="stable")
    anchor_class = None
    for idx in sorted_src_indices:
        cand = classes_src[idx]
        if cand in classes_tgt:
            anchor_class = cand
            break

    if anchor_class is None:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=default_row_map,
            metadata={"reason": "No source class is present in target partition"},
            status="NOT_APPLICABLE",
            reason="No source class is present in target partition",
        )

    prev_cfg = cfg.get("class_prevalence", {})
    mult_cfg = prev_cfg.get("multiplier", {"mild": 2.0, "severe": 4.0})
    multiplier = float(mult_cfg.get(severity, 2.0 if severity == "mild" else 4.0))

    # 3. Compute sampling weights
    weights = np.ones(n_rows, dtype=np.float64)
    anchor_mask = (y_target == anchor_class)
    weights[anchor_mask] = multiplier
    probs = weights / np.sum(weights)

    # 4. Weighted bootstrap resampling of exactly n_rows
    rng = np.random.default_rng(resample_seed)
    chosen_indices = rng.choice(n_rows, size=n_rows, replace=True, p=probs).astype(np.int64)
    resample_hash = compute_bytes_sha256(chosen_indices.tobytes())

    X_shifted = X_target.iloc[chosen_indices].reset_index(drop=True)

    # 5. Offline metadata tracking (labels used ONLY to record distribution drift metrics)
    y_shifted = y_target[chosen_indices]
    unique_classes = np.unique(np.concatenate([classes_tgt, classes_src]))
    
    orig_props = {str(c): float(np.mean(y_target == c)) for c in unique_classes}
    shifted_props = {str(c): float(np.mean(y_shifted == c)) for c in unique_classes}
    tv_distance = float(0.5 * sum(abs(orig_props[str(c)] - shifted_props[str(c)]) for c in unique_classes))
    unique_rows_count = int(len(np.unique(chosen_indices)))

    return ShiftResult(
        X_shifted=X_shifted,
        row_index_map=chosen_indices,
        metadata={
            "family": "class_prevalence",
            "severity": severity,
            "anchor_class": str(anchor_class),
            "multiplier": multiplier,
            "original_class_proportions": orig_props,
            "shifted_class_proportions": shifted_props,
            "total_variation_distance": tv_distance,
            "unique_target_rows": unique_rows_count,
            "unique_row_fraction": float(unique_rows_count / n_rows),
            "resampling_index_sha256": resample_hash,
        },
        status="APPLICABLE",
    )


def apply_local_overlap_shift(
    X_target: pd.DataFrame,
    y_target: np.ndarray,
    X_source: pd.DataFrame,
    y_source: np.ndarray,
    source_stats: SourceStatistics,
    cfg: Dict[str, Any],
    severity: str,
    backend: str = "auto",
) -> ShiftResult:
    """Apply Local Structural Overlap shift via boundary point contraction toward competing centroids.
    
    Outer-source class centroids are computed in source-standardized numeric space.
    For each target point, identify nearest competing source centroid.
    Select top 30% of boundary-adjacent rows within each class (same rows for mild & severe).
    Perturb observed numeric coordinates: x' = (1 - alpha) * x + alpha * competing_centroid.
    """
    n_rows = len(X_target)
    row_map = np.arange(n_rows, dtype=np.int64)
    eligible_cols = sorted(source_stats.non_zero_variance_cols)

    if len(eligible_cols) < 2:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=row_map,
            metadata={"reason": "Fewer than 2 eligible numeric features for local overlap"},
            status="NOT_APPLICABLE",
            reason="Fewer than 2 eligible numeric features for local overlap",
        )

    classes_src = np.unique(y_source)
    classes_tgt = np.unique(y_target)

    if len(classes_src) < 2 or len(classes_tgt) < 2:
        return ShiftResult(
            X_shifted=X_target.copy(),
            row_index_map=row_map,
            metadata={"reason": "Fewer than 2 classes represented in source or target"},
            status="NOT_APPLICABLE",
            reason="Fewer than 2 classes represented in source or target",
        )

    overlap_cfg = cfg.get("local_overlap", {})
    boundary_fraction = float(overlap_cfg.get("boundary_fraction", 0.30))
    alpha_cfg = overlap_cfg.get("alpha", {"mild": 0.25, "severe": 0.50})
    alpha = float(alpha_cfg.get(severity, 0.25 if severity == "mild" else 0.50))

    # 1. Compute source class centroids in source-standardized space
    means_vec = np.array([source_stats.means[c] for c in eligible_cols], dtype=np.float64)
    sds_vec = np.array([source_stats.sds[c] for c in eligible_cols], dtype=np.float64)
    medians_vec = np.array([source_stats.medians[c] for c in eligible_cols], dtype=np.float64)

    # Geometry-only temporary imputation on source for centroid calculation
    X_src_sub = X_source[eligible_cols].to_numpy(dtype=np.float64)
    nan_mask_src = np.isnan(X_src_sub)
    if np.any(nan_mask_src):
        X_src_sub = np.where(nan_mask_src, medians_vec, X_src_sub)
    Z_src = (X_src_sub - means_vec) / sds_vec

    source_centroids: Dict[Any, np.ndarray] = {}
    for c in classes_src:
        mask = (y_source == c)
        if np.any(mask):
            source_centroids[c] = np.mean(Z_src[mask], axis=0)

    # 2. Map target points into source-standardized space and find nearest competing centroid
    X_tgt_sub = X_target[eligible_cols].to_numpy(dtype=np.float64)
    nan_mask_tgt = np.isnan(X_tgt_sub)
    # Geometry-only temporary imputation using source feature medians (never standardized zero / source mean)
    X_tgt_geom = np.where(nan_mask_tgt, medians_vec, X_tgt_sub)
    Z_tgt = (X_tgt_geom - means_vec) / sds_vec

    nearest_competing_centroids_std = np.zeros_like(Z_tgt)
    competing_class_per_row: List[Any] = []
    competing_dist_before = np.zeros(n_rows, dtype=np.float64)

    for i in range(n_rows):
        curr_class = y_target[i]
        competing_classes = [c for c in classes_src if c != curr_class and c in source_centroids]
        if not competing_classes:
            # Fallback to any other class
            competing_classes = [c for c in source_centroids if c != curr_class]
        
        if not competing_classes:
            # Cannot find competing class
            nearest_competing_centroids_std[i] = Z_tgt[i]
            competing_class_per_row.append(curr_class)
            competing_dist_before[i] = 0.0
            continue

        best_c = competing_classes[0]
        best_d = np.linalg.norm(Z_tgt[i] - source_centroids[best_c])
        for c in competing_classes[1:]:
            d = np.linalg.norm(Z_tgt[i] - source_centroids[c])
            if d < best_d:
                best_d = d
                best_c = c
        
        nearest_competing_centroids_std[i] = source_centroids[best_c]
        competing_class_per_row.append(best_c)
        competing_dist_before[i] = best_d

    # 3. Select boundary-adjacent rows (top 30% closest to competing centroid within each target class)
    selected_row_indices: List[int] = []
    for c in classes_tgt:
        c_mask = np.where(y_target == c)[0]
        n_c = len(c_mask)
        if n_c == 0:
            continue
        n_boundary = max(1, round(boundary_fraction * n_c))
        # Sort by competing distance ascending
        sorted_order = np.argsort(competing_dist_before[c_mask], kind="stable")
        selected_row_indices.extend(c_mask[sorted_order[:n_boundary]])

    selected_rows = np.array(sorted(selected_row_indices), dtype=np.int64)

    # 4. Perturb selected rows toward competing centroids
    # Convert competing centroid back to original unstandardized coordinates
    target_unstd = means_vec + nearest_competing_centroids_std[selected_rows] * sds_vec

    X_shifted = X_target.copy()
    use_gpu = should_use_gpu(len(selected_rows), len(eligible_cols), backend=backend, min_cells=cfg.get("execution", {}).get("gpu_min_numeric_cells", 2000000))

    # Apply convex combination: (1 - alpha) * x + alpha * target_vector
    # Missing coordinates in original target remain NaN
    sub_orig = X_tgt_sub.copy()
    sub_perturbed = apply_convex_interpolation(
        sub_orig,
        target_vectors=target_unstd,
        alpha=alpha,
        row_indices=selected_rows,
        use_gpu=use_gpu,
    )
    # Restore exact NaN mask
    sub_perturbed[nan_mask_tgt] = np.nan

    for idx, col in enumerate(eligible_cols):
        X_shifted[col] = sub_perturbed[:, idx]

    # Metrics on affected rows
    X_perturbed_geom = np.where(nan_mask_tgt, medians_vec, sub_perturbed)
    Z_perturbed = (X_perturbed_geom - means_vec) / sds_vec
    competing_dist_after = np.linalg.norm(
        Z_perturbed[selected_rows] - nearest_competing_centroids_std[selected_rows],
        axis=1,
    )
    mean_before = float(np.mean(competing_dist_before[selected_rows]))
    mean_after = float(np.mean(competing_dist_after))
    mean_movement = float(np.mean(np.linalg.norm(Z_perturbed[selected_rows] - Z_tgt[selected_rows], axis=1)))

    return ShiftResult(
        X_shifted=X_shifted,
        row_index_map=row_map,
        metadata={
            "family": "local_overlap",
            "severity": severity,
            "alpha": alpha,
            "boundary_fraction": boundary_fraction,
            "selected_rows_count": len(selected_rows),
            "mean_source_normalized_movement": mean_movement,
            "mean_competing_centroid_dist_before": mean_before,
            "mean_competing_centroid_dist_after": mean_after,
            "overlap_strength_metric": mean_before - mean_after,
            "selected_rows_sha256": compute_bytes_sha256(selected_rows.tobytes()),
        },
        status="APPLICABLE",
    )
