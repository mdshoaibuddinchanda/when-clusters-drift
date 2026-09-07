"""Unit tests proving zero label leakage in shift generation."""

from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.parametrize("cond", [
    "clean",
    "location_mild",
    "scale_severe",
    "mcar_mild",
    "outliers_mild",
    "measurement_noise_severe",
])
def test_unsupervised_shifts_invariant_to_label_permutation(cond):
    """Verify that permuting or altering labels has zero effect on unsupervised shifts."""
    df_src = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df_tgt = pd.DataFrame({"x": [2.0, 3.0, 4.0, 5.0, 6.0]})
    roles = {"x": "numeric"}

    y1 = np.array([0, 1, 0, 1, 0])
    y2 = np.array([1, 0, 1, 0, 1])  # Inverted

    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res1 = engine.generate_shift("test_ds", 0, cond, df_tgt, df_src, roles, y_target=y1, y_source=y1)
    res2 = engine.generate_shift("test_ds", 0, cond, df_tgt, df_src, roles, y_target=y2, y_source=y2)

    assert_frame_equal(res1.X_shifted, res2.X_shifted)
    assert not hasattr(res1, "labels")
    assert "labels" not in res1.metadata


def test_learner_result_does_not_contain_labels():
    """Verify generic ShiftResult never leaks target labels."""
    df = pd.DataFrame({"x": [1.0, 2.0]})
    roles = {"x": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res = engine.generate_shift(
        "test_ds", 0, "class_prevalence_mild", df, df, roles,
        y_target=np.array([0, 1]), y_source=np.array([0, 1]),
    )
    assert not hasattr(res, "labels")
    assert not hasattr(res, "y")
    assert not hasattr(res, "y_shifted")
    assert "labels" not in res.metadata
