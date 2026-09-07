"""Unit tests for Outliers Shift."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_outlier_shift_nesting_and_heavy_tails():
    """Verify mild outlier rows are a strict subset of severe outlier rows."""
    rng = np.random.default_rng(42)
    n = 100
    df_src = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df_tgt = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    roles = {"x1": "numeric", "x2": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res_mild = engine.generate_shift("test_ds", 0, "outliers_mild", df_tgt, df_src, roles)
    res_severe = engine.generate_shift("test_ds", 0, "outliers_severe", df_tgt, df_src, roles)

    assert res_mild.status == "APPLICABLE"
    assert res_severe.status == "APPLICABLE"

    diff_mild = (res_mild.X_shifted - df_tgt).abs().sum(axis=1) > 1e-9
    diff_severe = (res_severe.X_shifted - df_tgt).abs().sum(axis=1) > 1e-9

    mild_indices = set(np.where(diff_mild)[0])
    severe_indices = set(np.where(diff_severe)[0])

    # 1. Mild rows are a strict subset of severe rows
    assert mild_indices.issubset(severe_indices)
    assert len(severe_indices) > len(mild_indices)

    # 2. Unselected rows remain identical
    unselected = set(range(n)) - severe_indices
    for idx in unselected:
        np.testing.assert_array_equal(res_mild.X_shifted.iloc[idx], df_tgt.iloc[idx])
        np.testing.assert_array_equal(res_severe.X_shifted.iloc[idx], df_tgt.iloc[idx])
