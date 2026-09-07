"""Unit tests for exact identity preservation in clean condition."""

from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_clean_identity_exact():
    """Verify clean condition produces exact byte/value identity without modification."""
    df = pd.DataFrame({
        "num": [1.0, 2.5, np.nan, 4.2],
        "cat": ["a", "b", "c", None],
        "int_col": [10, 20, 30, 40],
    })
    X_src = df.copy()
    roles = {"num": "numeric", "cat": "categorical", "int_col": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res = engine.generate_shift("iris", 0, "clean", df, X_src, roles)

    assert res.status == "APPLICABLE"
    assert_frame_equal(res.X_shifted, df)
    np.testing.assert_array_equal(res.row_index_map, np.arange(len(df)))
    assert res.metadata["family"] == "clean"
    assert res.metadata["severity"] == "none"
