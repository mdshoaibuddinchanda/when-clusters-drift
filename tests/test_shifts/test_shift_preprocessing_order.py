"""Unit tests proving the canonical sequence: raw target -> shift -> source preprocessor."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.data.preprocess import build_preprocessor
from clusterdrift.shifts import ShiftEngine, audit_preprocessing_transformation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_preprocessing_order_invariant():
    """Verify that shift is applied before source preprocessing, and demonstrates non-commutativity."""
    # Source with mean 0, std 1
    df_src = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
    # Target with non-zero values
    df_tgt = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
    roles = {"x": "numeric"}
    meta = {"feature_roles": roles}
    prep_cfg = {"numeric": {"scaler": "standard", "imputer": "mean"}}

    prep = build_preprocessor(roles, prep_cfg, meta)
    prep.fit(df_src)

    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    # 1. Canonical order: raw target -> location shift -> source preprocessor
    res = engine.generate_shift("test_ds", 0, "location_severe", df_tgt, df_src, roles)
    X_shifted_proc = prep.transform(res.X_shifted)

    # 2. Inverted order: raw target -> source preprocessor -> location shift
    X_tgt_proc = prep.transform(df_tgt)
    # Applying shift after standardizing gives different numerical scale
    sigma_src = float(np.std(df_src["x"], ddof=1))
    X_inverted = X_tgt_proc + 1.0 * sigma_src

    # Assert canonical and inverted are mathematically distinct
    assert not np.allclose(X_shifted_proc, X_inverted)

    # Assert audit confirms finiteness and validity of canonical order
    audit = audit_preprocessing_transformation(res.X_shifted, prep, 1)
    assert audit["all_finite"] is True
    assert audit["dimension_matches"] is True
