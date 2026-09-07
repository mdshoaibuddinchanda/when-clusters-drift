"""Targeted test verifying evaluation-only label remapping for class-prevalence shift."""

import json
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_class_prevalence_label_remapping():
    """Verify class prevalence row_index_map correctly indexes labels for evaluation."""
    # Find an existing class prevalence shift spec and companion npz
    spec_path = PROJECT_ROOT / "data" / "shifts" / "specs" / "iris" / "fold_0" / "class_prevalence_severe.json"
    npz_path = PROJECT_ROOT / "data" / "shifts" / "specs" / "iris" / "fold_0" / "class_prevalence_severe.npz"
    if not spec_path.exists() or not npz_path.exists():
        pytest.skip("Iris class prevalence shift spec not found")

    with np.load(npz_path) as npz:
        assert "row_index_map" in npz
        row_map = npz["row_index_map"]

    # Load target labels for iris fold 0
    fold_p = PROJECT_ROOT / "data" / "splits" / "controlled" / "iris" / "fold_0.npz"
    with np.load(fold_p) as npz:
        target_indices = npz["target_indices"]

    import pandas as pd
    df_y = pd.read_parquet(PROJECT_ROOT / "data" / "canonical" / "controlled" / "iris" / "labels.parquet")
    y_tgt = df_y.iloc[target_indices].to_numpy().ravel()

    # Remap labels
    y_eval = y_tgt[row_map]
    assert len(y_eval) == len(row_map)
    assert np.all(y_eval == y_tgt[row_map])
