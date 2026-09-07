"""Unit tests for Measurement Noise Shift."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_measurement_noise_exact_threefold_scaling():
    """Verify measurement noise uses identical Gaussian tensors and exactly 3x scaling."""
    rng = np.random.default_rng(123)
    df_src = pd.DataFrame({"x1": rng.normal(loc=5.0, scale=2.0, size=50)})
    df_tgt = pd.DataFrame({"x1": rng.normal(loc=5.0, scale=2.0, size=50)})
    roles = {"x1": "numeric"}
    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res_mild = engine.generate_shift("test_ds", 0, "measurement_noise_mild", df_tgt, df_src, roles)
    res_severe = engine.generate_shift("test_ds", 0, "measurement_noise_severe", df_tgt, df_src, roles)

    assert res_mild.status == "APPLICABLE"
    assert res_severe.status == "APPLICABLE"

    # Perturbations
    pert_mild = res_mild.X_shifted["x1"] - df_tgt["x1"]
    pert_severe = res_severe.X_shifted["x1"] - df_tgt["x1"]

    # Severe perturbation is exactly 3.0x mild (0.30 / 0.10)
    np.testing.assert_allclose(pert_severe.to_numpy(), 3.0 * pert_mild.to_numpy(), atol=1e-12)
    assert res_mild.metadata["noise_sha256"] == res_severe.metadata["noise_sha256"]
