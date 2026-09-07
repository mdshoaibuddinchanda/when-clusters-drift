"""Probe matrix reconstruction routines from persistent descriptors.

Transforms selected probe rows using source-fitted Phase-2 preprocessing.
Strictly adheres to the label-free boundary: never loads or accesses target ground truth.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def load_raw_source_features(
    dataset_id: str,
    outer_fold: int,
    project_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load raw source features and metadata without touching labels."""
    ds_dir = project_root / "data" / "canonical" / "controlled" / dataset_id
    features_path = ds_dir / "features.parquet"
    meta_path = ds_dir / "metadata.json"

    df_X = pd.read_parquet(features_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fold_path = project_root / "data" / "splits" / "controlled" / dataset_id / f"fold_{outer_fold}.npz"
    with np.load(fold_path) as npz:
        src_idx = npz["source_indices"]

    X_src_raw = df_X.iloc[src_idx].copy().reset_index(drop=True)
    return X_src_raw, meta


def load_reference_probe_matrix(
    dataset_id: str,
    outer_fold: int,
    canonical_source_indices: np.ndarray,
    preprocessor: Any,
    project_root: Path,
) -> np.ndarray:
    """Reconstruct transformed float32 finite reference probe matrix A^R.

    Parameters
    ----------
    dataset_id : str
        Controlled dataset ID.
    outer_fold : int
        Outer fold index [0..4].
    canonical_source_indices : np.ndarray
        Indices into the outer source partition.
    preprocessor : Any
        Pre-fitted Phase-2 preprocessor (fitted ONLY on outer source).
    project_root : Path
        Repository root.

    Returns
    -------
    probe_matrix : np.ndarray, shape (B, D), dtype float32
        Transformed finite feature matrix.
    """
    X_src_raw, _ = load_raw_source_features(dataset_id, outer_fold, project_root)
    X_probe_raw = X_src_raw.iloc[canonical_source_indices].copy()

    transformed = preprocessor.transform(X_probe_raw)
    matrix = np.ascontiguousarray(transformed, dtype=np.float32)

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Transformed reference probe matrix for {dataset_id} fold {outer_fold} contains non-finite values."
        )
    return matrix


def load_current_probe_matrix(
    X_shifted: pd.DataFrame,
    selected_positions: np.ndarray,
    preprocessor: Any,
) -> np.ndarray:
    """Reconstruct transformed float32 finite current probe matrix A_t^C.

    Parameters
    ----------
    X_shifted : pd.DataFrame
        Observed shifted target DataFrame from Phase-4 engine.
    selected_positions : np.ndarray
        Positions [0..n_target-1] selected for the current probe bank.
    preprocessor : Any
        Source-fitted Phase-2 preprocessor (the SAME preprocessor fitted on outer source).

    Returns
    -------
    probe_matrix : np.ndarray, shape (B, D), dtype float32
        Transformed finite feature matrix.
    """
    X_probe_raw = X_shifted.iloc[selected_positions].copy()
    transformed = preprocessor.transform(X_probe_raw)
    matrix = np.ascontiguousarray(transformed, dtype=np.float32)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Transformed current probe matrix contains non-finite values.")
    return matrix
