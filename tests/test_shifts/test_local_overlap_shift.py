"""Unit tests for Local Structural Overlap Shift."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_local_overlap_boundary_monotonicity():
    """Verify distance to competing centroid shrinks monotonically for affected boundary points."""
    # Two well-separated clusters with non-zero variance along both x1 and x2
    df_src = pd.DataFrame({
        "x1": [0.0, 0.5, 1.0, 10.0, 10.5, 11.0],
        "x2": [1.0, 2.0, 1.5, 5.0, 6.0, 5.5],
    })
    y_src = np.array([0, 0, 0, 1, 1, 1])

    df_tgt = pd.DataFrame({
        "x1": [0.8, 0.2, 0.1, 9.2, 10.8, 10.9],
        "x2": [1.2, 1.8, 1.4, 5.2, 5.8, 5.4],
    })
    y_tgt = np.array([0, 0, 0, 1, 1, 1])

    roles = {"x1": "numeric", "x2": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res_mild = engine.generate_shift("test_ds", 0, "local_overlap_mild", df_tgt, df_src, roles, y_tgt, y_src)
    res_severe = engine.generate_shift("test_ds", 0, "local_overlap_severe", df_tgt, df_src, roles, y_tgt, y_src)

    assert res_mild.status == "APPLICABLE"
    assert res_severe.status == "APPLICABLE"

    # 1. Selected boundary rows match between mild and severe
    assert res_mild.metadata["selected_rows_sha256"] == res_severe.metadata["selected_rows_sha256"]

    # 2. Monotonic contraction toward competing centroid
    d_before = res_mild.metadata["mean_competing_centroid_dist_before"]
    d_mild = res_mild.metadata["mean_competing_centroid_dist_after"]
    d_severe = res_severe.metadata["mean_competing_centroid_dist_after"]

    assert d_severe < d_mild < d_before
    assert res_severe.metadata["overlap_strength_metric"] > res_mild.metadata["overlap_strength_metric"]
