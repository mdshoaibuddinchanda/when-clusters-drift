"""Unit tests verifying CPU and GPU numerical equivalence for Phase 4 shifts."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.shifts import ShiftEngine
from clusterdrift.shifts.backend import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.skipif(
    not (TORCH_AVAILABLE and torch.cuda.is_available()),
    reason="PyTorch CUDA GPU not available",
)
@pytest.mark.parametrize("cond", [
    "location_mild",
    "scale_severe",
    "measurement_noise_mild",
    "outliers_mild",
])
def test_cpu_gpu_numerical_equivalence(cond):
    """Verify NumPy CPU and PyTorch CUDA backends yield identical numerical outputs."""
    rng = np.random.default_rng(42)
    df_src = pd.DataFrame({"x1": rng.normal(size=100), "x2": rng.normal(size=100)})
    df_tgt = pd.DataFrame({"x1": rng.normal(size=100), "x2": rng.normal(size=100)})
    roles = {"x1": "numeric", "x2": "numeric"}

    engine = ShiftEngine(PROJECT_ROOT / "configs" / "shifts.yaml", project_root=PROJECT_ROOT)

    res_cpu = engine.generate_shift("test_ds", 0, cond, df_tgt, df_src, roles, backend="numpy")
    res_gpu = engine.generate_shift("test_ds", 0, cond, df_tgt, df_src, roles, backend="torch")

    assert res_cpu.status == "APPLICABLE"
    assert res_gpu.status == "APPLICABLE"

    arr_cpu = res_cpu.X_shifted.to_numpy(dtype=np.float64)
    arr_gpu = res_gpu.X_shifted.to_numpy(dtype=np.float64)

    valid = np.isfinite(arr_cpu) & np.isfinite(arr_gpu)
    np.testing.assert_allclose(arr_cpu[valid], arr_gpu[valid], rtol=1e-6, atol=1e-6)
