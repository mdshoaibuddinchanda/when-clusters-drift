"""Unit tests for Scale Shift."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_scale_shift_mean_preservation_and_factors():
    """Verify scale shift preserves source mean and scales standard deviations accurately."""
    df_src = pd.DataFrame({
        "x": [0.0, 10.0, 20.0],  # mu = 10.0, sigma = 10.0
    })
    # Target values: at mean (10), at mean + sigma (20), at mean - sigma (0)
    df_tgt = pd.DataFrame({
        "x": [10.0, 20.0, 0.0, np.nan],
    })
    roles = {"x": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res_mild = engine.generate_shift("test_ds", 0, "scale_mild", df_tgt, df_src, roles)
    res_severe = engine.generate_shift("test_ds", 0, "scale_severe", df_tgt, df_src, roles)

    assert res_mild.status == "APPLICABLE"
    assert res_severe.status == "APPLICABLE"

    # 1. Source mean remains unchanged
    assert res_mild.X_shifted["x"].iloc[0] == 10.0
    assert res_severe.X_shifted["x"].iloc[0] == 10.0

    # 2. mu + sigma -> mu + 1.25 sigma (mild), mu + 1.75 sigma (severe)
    np.testing.assert_allclose(res_mild.X_shifted["x"].iloc[1], 10.0 + 1.25 * 10.0)
    np.testing.assert_allclose(res_severe.X_shifted["x"].iloc[1], 10.0 + 1.75 * 10.0)

    # 3. mu - sigma -> mu - 1.25 sigma (mild), mu - 1.75 sigma (severe)
    np.testing.assert_allclose(res_mild.X_shifted["x"].iloc[2], 10.0 - 1.25 * 10.0)
    np.testing.assert_allclose(res_severe.X_shifted["x"].iloc[2], 10.0 - 1.75 * 10.0)

    # 4. Existing NaN remains NaN
    assert np.isnan(res_mild.X_shifted["x"].iloc[3])
    assert np.isnan(res_severe.X_shifted["x"].iloc[3])
