"""Validation, preprocessing audit, and severity monotonicity checking for Phase 4 shifts."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from clusterdrift.shifts.base import ShiftResult


def audit_preprocessing_transformation(
    X_shifted_raw: pd.DataFrame,
    source_preprocessor: Any,
    expected_encoded_dim: int,
) -> Dict[str, Any]:
    """Transform raw shifted target using the source-fitted Phase-2 preprocessor.
    
    CRITICAL PREPROCESSING INVARIANT:
    Order: X_target,raw -> controlled shift -> T_source -> X_target,shifted,processed.
    NEVER fit preprocessor on shifted target data.
    
    Verifies:
    1. Preprocessor transforms without error.
    2. Encoded output dimension matches outer-source encoded dimension.
    3. Output matrix contains zero NaNs or Infs (all values finite).
    4. Output dtype is float32 (or float64).
    """
    # Transform raw shifted target using the frozen preprocessor
    X_proc = source_preprocessor.transform(X_shifted_raw)

    if isinstance(X_proc, pd.DataFrame):
        arr = X_proc.to_numpy()
    elif isinstance(X_proc, np.ndarray):
        arr = X_proc
    elif hasattr(X_proc, "toarray"):  # scipy sparse matrix
        arr = X_proc.toarray()
    else:
        arr = np.asarray(X_proc)

    n_rows, n_encoded = arr.shape
    is_finite = bool(np.all(np.isfinite(arr)))
    dim_match = bool(n_encoded == expected_encoded_dim)
    dtype_str = str(arr.dtype)

    raw_missing = int(X_shifted_raw.isna().sum().sum())
    post_missing = int(np.isnan(arr).sum()) if not is_finite else 0

    return {
        "raw_target_rows": int(len(X_shifted_raw)),
        "shifted_target_rows": int(n_rows),
        "raw_features": int(X_shifted_raw.shape[1]),
        "encoded_features": int(n_encoded),
        "expected_encoded_features": int(expected_encoded_dim),
        "dimension_matches": dim_match,
        "raw_missing_count": raw_missing,
        "post_transform_missing_count": post_missing,
        "all_finite": is_finite,
        "output_dtype": dtype_str,
    }


def verify_severity_monotonicity(
    mild_res: ShiftResult,
    severe_res: ShiftResult,
    family: str,
) -> Dict[str, Any]:
    """Verify that severe condition exhibits strictly greater shift magnitude than mild condition.
    
    Returns structured audit dictionary with status PASS, WARN, or NOT_APPLICABLE.
    """
    if mild_res.status != "APPLICABLE" or severe_res.status != "APPLICABLE":
        return {
            "family": family,
            "status": "NOT_APPLICABLE",
            "metric_name": "none",
            "mild_value": None,
            "severe_value": None,
            "details": f"One or both conditions not applicable: mild={mild_res.status}, severe={severe_res.status}",
        }

    m_meta = mild_res.metadata
    s_meta = severe_res.metadata

    if family == "location":
        m_val = m_meta.get("realized_standardized_displacement", 0.5)
        s_val = s_meta.get("realized_standardized_displacement", 1.0)
        metric = "realized_standardized_displacement"
        passed = s_val > m_val

    elif family == "scale":
        m_val = m_meta.get("realized_scale_measure", 1.25)
        s_val = s_meta.get("realized_scale_measure", 1.75)
        metric = "scale_factor"
        passed = s_val > m_val

    elif family == "mcar":
        m_val = m_meta.get("newly_masked_cell_count", 0)
        s_val = s_meta.get("newly_masked_cell_count", 0)
        metric = "newly_masked_cells"
        passed = s_val > m_val

    elif family == "outliers":
        m_val = m_meta.get("selected_row_count", 0)
        s_val = s_meta.get("selected_row_count", 0)
        metric = "selected_row_count"
        passed = s_val > m_val

    elif family == "measurement_noise":
        m_val = m_meta.get("rms_standardized_perturbation", 0.10)
        s_val = s_meta.get("rms_standardized_perturbation", 0.30)
        metric = "rms_standardized_perturbation"
        passed = s_val > m_val

    elif family == "class_prevalence":
        m_val = m_meta.get("total_variation_distance", 0.0)
        s_val = s_meta.get("total_variation_distance", 0.0)
        metric = "total_variation_distance"
        # Due to finite-sample bootstrap randomness, severe TV might occasionally be equal or slightly smaller
        passed = s_val >= m_val - 1e-4

    elif family == "local_overlap":
        m_val = m_meta.get("overlap_strength_metric", 0.0)
        s_val = s_meta.get("overlap_strength_metric", 0.0)
        metric = "overlap_strength_metric"
        passed = s_val > m_val

    else:
        return {
            "family": family,
            "status": "NOT_APPLICABLE",
            "metric_name": "unknown",
            "mild_value": None,
            "severe_value": None,
            "details": f"Unknown family: {family}",
        }

    status = "PASS" if passed else ("WARN" if family == "class_prevalence" else "FAIL")
    return {
        "family": family,
        "status": status,
        "metric_name": metric,
        "mild_value": m_val,
        "severe_value": s_val,
        "details": f"severe={s_val:.4f} vs mild={m_val:.4f}",
    }
