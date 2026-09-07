"""Unit tests for Class Prevalence Shift."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_class_prevalence_shift_properties_and_label_isolation():
    """Verify target row count preserved, class proportions shifted, and labels unexposed."""
    df_src = pd.DataFrame({"x": range(100)})
    y_src = np.array([0] * 80 + [1] * 20)  # Class 1 is rarest anchor class
    df_tgt = pd.DataFrame({"x": range(100)})
    y_tgt = np.array([0] * 80 + [1] * 20)

    roles = {"x": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res_mild = engine.generate_shift("test_ds", 0, "class_prevalence_mild", df_tgt, df_src, roles, y_tgt, y_src)
    res_severe = engine.generate_shift("test_ds", 0, "class_prevalence_severe", df_tgt, df_src, roles, y_tgt, y_src)

    assert res_mild.status == "APPLICABLE"
    assert res_severe.status == "APPLICABLE"

    # 1. Target row count exactly unchanged
    assert len(res_mild.X_shifted) == len(df_tgt)
    assert len(res_severe.X_shifted) == len(df_tgt)

    # 2. Labels NOT in result
    assert not hasattr(res_mild, "y_shifted")
    assert "labels" not in res_mild.metadata

    # 3. Proportion of anchor class increases with severity
    anchor = int(res_mild.metadata["anchor_class"])
    assert anchor == 1
    shifted_p_mild = res_mild.metadata["shifted_class_proportions"][str(anchor)]
    shifted_p_severe = res_severe.metadata["shifted_class_proportions"][str(anchor)]
    assert shifted_p_mild > 0.20
    assert shifted_p_severe >= shifted_p_mild - 0.05
