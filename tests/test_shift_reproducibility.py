"""Unit tests for shift reproducibility, determinism, and order independence."""

from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_order_independence_and_determinism():
    """Verify order of evaluation does not alter generated shifts."""
    df_src = pd.DataFrame({"x": np.linspace(0, 10, 20)})
    df_tgt_A = pd.DataFrame({"x": np.linspace(1, 9, 20)})
    df_tgt_B = pd.DataFrame({"x": np.linspace(2, 8, 20)})
    roles = {"x": "numeric"}

    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    # Order 1: A then B
    res_A1 = engine.generate_shift("ds_A", 0, "location_mild", df_tgt_A, df_src, roles)
    res_B1 = engine.generate_shift("ds_B", 0, "location_mild", df_tgt_B, df_src, roles)

    # Order 2: B then A
    res_B2 = engine.generate_shift("ds_B", 0, "location_mild", df_tgt_B, df_src, roles)
    res_A2 = engine.generate_shift("ds_A", 0, "location_mild", df_tgt_A, df_src, roles)

    assert_frame_equal(res_A1.X_shifted, res_A2.X_shifted)
    assert_frame_equal(res_B1.X_shifted, res_B2.X_shifted)


def test_repeated_evaluation_identical():
    """Verify evaluating the exact same scenario twice yields identical results."""
    df_src = pd.DataFrame({"x": np.linspace(-5, 5, 50), "y": np.linspace(0, 10, 50)})
    df_tgt = pd.DataFrame({"x": np.linspace(-4, 4, 50), "y": np.linspace(1, 9, 50)})
    roles = {"x": "numeric", "y": "numeric"}

    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res1 = engine.generate_shift("test_ds", 1, "outliers_mild", df_tgt, df_src, roles)
    res2 = engine.generate_shift("test_ds", 1, "outliers_mild", df_tgt, df_src, roles)

    assert_frame_equal(res1.X_shifted, res2.X_shifted)
    np.testing.assert_array_equal(res1.row_index_map, res2.row_index_map)
    assert res1.metadata == res2.metadata
