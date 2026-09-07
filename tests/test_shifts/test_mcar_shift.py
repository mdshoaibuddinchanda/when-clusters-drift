"""Unit tests for MCAR Missingness Shift."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_mcar_shift_nesting_and_accounting():
    """Verify MCAR nesting and accurate masking accounting without double-counting."""
    df_src = pd.DataFrame({"f1": [1.0] * 100, "f2": ["cat"] * 100})
    df_tgt = pd.DataFrame({
        "f1": [float(i) if i % 10 != 0 else np.nan for i in range(100)],
        "f2": [f"val_{i}" if i % 5 != 0 else None for i in range(100)],
    })
    roles = {"f1": "numeric", "f2": "categorical"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res_mild = engine.generate_shift("test_ds", 0, "mcar_mild", df_tgt, df_src, roles)
    res_severe = engine.generate_shift("test_ds", 0, "mcar_severe", df_tgt, df_src, roles)

    assert res_mild.status == "APPLICABLE"
    assert res_severe.status == "APPLICABLE"

    # Pre-existing missing cells
    orig_f1_na = df_tgt["f1"].isna()
    orig_f2_na = df_tgt["f2"].isna()

    # Newly masked cells
    new_mild_f1 = orig_f1_na ^ res_mild.X_shifted["f1"].isna()
    new_severe_f1 = orig_f1_na ^ res_severe.X_shifted["f1"].isna()
    new_mild_f2 = orig_f2_na ^ res_mild.X_shifted["f2"].isna()
    new_severe_f2 = orig_f2_na ^ res_severe.X_shifted["f2"].isna()

    # 1. Mask_mild is strict subset of Mask_severe
    assert (new_mild_f1 & ~new_severe_f1).sum() == 0
    assert (new_mild_f2 & ~new_severe_f2).sum() == 0

    # 2. Severe introduces strictly more missingness than mild
    total_mild = res_mild.metadata["newly_masked_cell_count"]
    total_severe = res_severe.metadata["newly_masked_cell_count"]
    assert total_severe > total_mild

    # 3. Eligible cell count accurately excludes pre-existing missingness
    eligible = res_mild.metadata["eligible_cell_count"]
    assert eligible == (df_tgt["f1"].notna().sum() + df_tgt["f2"].notna().sum())
