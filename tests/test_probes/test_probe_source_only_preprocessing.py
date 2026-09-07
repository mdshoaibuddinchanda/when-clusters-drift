"""Test that probe matrices are transformed strictly with source-fitted preprocessing."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from clusterdrift.data.preprocess import build_preprocessor
from clusterdrift.probes.matrix import load_current_probe_matrix, load_reference_probe_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_source_only_preprocessing_isolation():
    """Verify that probe matrix transformation applies source-fitted statistics, not target statistics."""
    # Synthetic dataset with source having mean 0 and target having mean 100
    df_src = pd.DataFrame({"feat": [0.0, 1.0, 2.0, -1.0, -2.0]})
    df_tgt_shifted = pd.DataFrame({"feat": [98.0, 99.0, 100.0, 101.0, 102.0]})

    prep_cfg = {
        "numeric": {
            "scaling": "standard",
            "imputation": "median",
            "clipping": {"enabled": False},
        },
        "categorical": {"imputation": "most_frequent", "encoding": "onehot"},
        "ordinal": {"policy": "onehot_unless_explicit_order", "all_missing_sentinel": "__MISSING_SOURCE__"},
        "high_cardinality": {"max_categories": 50, "rare_threshold": 0.01},
        "feature_selection": {"zero_variance_removal": False},
    }

    src_prep = build_preprocessor(feature_roles={"feat": "numeric"}, config=prep_cfg)
    src_prep.fit(df_src)

    # Transform target using source-fitted preprocessor
    selected_pos = np.array([0, 1, 2, 3, 4])
    matrix = load_current_probe_matrix(df_tgt_shifted, selected_pos, src_prep)

    # Since source mean is 0 and scale is ~1.58, target values around 100 will have standardized values around ~63
    # If target had been fitted on its own preprocessor, values would be around 0
    assert np.all(matrix > 50.0), "Probe matrix must reflect source-standardized scale, not target-standardized"
