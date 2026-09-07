"""Unit tests for Location Shift."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_location_shift_feature_matching_and_displacement():
    """Verify mild and severe location shifts select identical features and scale by 2x."""
    df_src = pd.DataFrame({
        "x1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "x2": [100.0, 200.0, 300.0, 400.0, 500.0],
        "zero_var": [5.0, 5.0, 5.0, 5.0, 5.0],
    })
    df_tgt = pd.DataFrame({
        "x1": [15.0, np.nan, 35.0, 45.0, 55.0],
        "x2": [150.0, 250.0, 350.0, 450.0, np.nan],
        "zero_var": [5.0, 5.0, 5.0, 5.0, 5.0],
    })
    roles = {"x1": "numeric", "x2": "numeric", "zero_var": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res_mild = engine.generate_shift("test_ds", 0, "location_mild", df_tgt, df_src, roles)
    res_severe = engine.generate_shift("test_ds", 0, "location_severe", df_tgt, df_src, roles)

    assert res_mild.status == "APPLICABLE"
    assert res_severe.status == "APPLICABLE"

    # 1. Exact same selected features
    assert res_mild.metadata["selected_features"] == res_severe.metadata["selected_features"]
    sel_cols = res_mild.metadata["selected_features"]
    assert "zero_var" not in sel_cols

    # 2. Severe displacement is exactly twice mild
    for c in sel_cols:
        diff_mild = res_mild.X_shifted[c] - df_tgt[c]
        diff_severe = res_severe.X_shifted[c] - df_tgt[c]
        valid = df_tgt[c].notna()
        np.testing.assert_allclose(diff_severe[valid].to_numpy(), 2.0 * diff_mild[valid].to_numpy())

    # 3. Existing NaNs remain NaN
    assert res_mild.X_shifted["x1"].isna().iloc[1]
    assert res_severe.X_shifted["x2"].isna().iloc[4]

    # 4. Zero-variance column unchanged
    np.testing.assert_array_equal(res_mild.X_shifted["zero_var"], df_tgt["zero_var"])
