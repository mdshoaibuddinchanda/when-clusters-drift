"""Unit tests proving shifts use outer-source statistics only."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_shift_independent_of_target_scale():
    """Verify perturbation magnitude depends strictly on source SD, not target SD."""
    # Source with SD = 10.0
    df_src = pd.DataFrame({"x": [0.0, 10.0, 20.0]})
    # Target 1 with small SD (1.0), Target 2 with massive SD (1000.0)
    df_tgt1 = pd.DataFrame({"x": [10.0, 11.0, 12.0]})
    df_tgt2 = pd.DataFrame({"x": [0.0, 1000.0, 2000.0]})

    roles = {"x": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    # Location severe (delta = 1.0) should shift by 1.0 * sigma_source = 10.0 for BOTH targets
    res1 = engine.generate_shift("ds", 0, "location_severe", df_tgt1, df_src, roles)
    res2 = engine.generate_shift("ds", 0, "location_severe", df_tgt2, df_src, roles)

    diff1 = (res1.X_shifted["x"] - df_tgt1["x"]).iloc[0]
    diff2 = (res2.X_shifted["x"] - df_tgt2["x"]).iloc[0]

    np.testing.assert_allclose(diff1, 10.0)
    np.testing.assert_allclose(diff2, 10.0)
    np.testing.assert_allclose(diff1, diff2)
