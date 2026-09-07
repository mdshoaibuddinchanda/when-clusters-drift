"""Unit tests verifying strict severity nesting across shift families."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_severity_nesting_invariants():
    """Verify strict nesting properties across location, outliers, noise, and mcar."""
    rng = np.random.default_rng(999)
    n = 100
    df_src = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df_tgt = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    roles = {"x1": "numeric", "x2": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    # 1. Location: same features
    loc_m = engine.generate_shift("ds", 0, "location_mild", df_tgt, df_src, roles)
    loc_s = engine.generate_shift("ds", 0, "location_severe", df_tgt, df_src, roles)
    assert loc_m.metadata["selected_features"] == loc_s.metadata["selected_features"]

    # 2. Outliers: nested rows
    out_m = engine.generate_shift("ds", 0, "outliers_mild", df_tgt, df_src, roles)
    out_s = engine.generate_shift("ds", 0, "outliers_severe", df_tgt, df_src, roles)
    diff_m = set(np.where((out_m.X_shifted - df_tgt).abs().sum(axis=1) > 1e-9)[0])
    diff_s = set(np.where((out_s.X_shifted - df_tgt).abs().sum(axis=1) > 1e-9)[0])
    assert diff_m.issubset(diff_s)

    # 3. Noise: exact 3x scaling
    noi_m = engine.generate_shift("ds", 0, "measurement_noise_mild", df_tgt, df_src, roles)
    noi_s = engine.generate_shift("ds", 0, "measurement_noise_severe", df_tgt, df_src, roles)
    np.testing.assert_allclose(
        (noi_s.X_shifted - df_tgt).to_numpy(),
        3.0 * (noi_m.X_shifted - df_tgt).to_numpy(),
        atol=1e-12,
    )

    # 4. MCAR: nested cell masks
    mcar_m = engine.generate_shift("ds", 0, "mcar_mild", df_tgt, df_src, roles)
    mcar_s = engine.generate_shift("ds", 0, "mcar_severe", df_tgt, df_src, roles)
    mask_m = mcar_m.X_shifted.isna().to_numpy()
    mask_s = mcar_s.X_shifted.isna().to_numpy()
    assert np.all(mask_s[mask_m])
