"""Parallel execution determinism tests (jobs=1 vs jobs=2)."""

import importlib.util
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import yaml

from clusterdrift.signals.engine import SignalCache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "canonical" / "controlled" / "iris" / "features.parquet").exists(),
    reason="Canonical iris dataset not present locally",
)

_spec = importlib.util.spec_from_file_location("compute_signals_script", PROJECT_ROOT / "scripts" / "06_compute_signals.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_signals_pass_a = _mod.run_signals_pass_a


def test_parallel_execution_determinism():
    """Verify that Pass A produces identical scientific signals and record hashes under jobs=1 and jobs=2."""
    with open(PROJECT_ROOT / "configs" / "signals.yaml") as f:
        sig_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "alignment.yaml") as f:
        alignment_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "methods.yaml") as f:
        methods_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "preprocessing.yaml") as f:
        prep_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "shifts.yaml") as f:
        shift_cfg = yaml.safe_load(f)

    reduced_sig_cfg = dict(sig_cfg)
    reduced_sig_cfg["validation"] = {
        "datasets": ["iris", "glass"],
        "outer_fold": 0,
        "conditions": ["clean", "location_severe"],
        "methods": ["fcm_adaptive"],
        "algorithm_seeds": [1],
    }
    dataset_classes = {"iris": 3, "glass": 6}

    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        out1 = Path(tmpdir1)
        out2 = Path(tmpdir2)

        # Run with jobs = 1
        cache1 = SignalCache()
        recs_j1 = run_signals_pass_a(
            project_root=PROJECT_ROOT,
            sig_cfg=reduced_sig_cfg,
            alignment_cfg=alignment_cfg,
            methods_cfg=methods_cfg,
            prep_cfg=prep_cfg,
            shift_cfg=shift_cfg,
            dataset_classes=dataset_classes,
            cache=cache1,
            out_dir=out1,
            jobs=1,
        )

        # Run with jobs = 2
        cache2 = SignalCache()
        recs_j2 = run_signals_pass_a(
            project_root=PROJECT_ROOT,
            sig_cfg=reduced_sig_cfg,
            alignment_cfg=alignment_cfg,
            methods_cfg=methods_cfg,
            prep_cfg=prep_cfg,
            shift_cfg=shift_cfg,
            dataset_classes=dataset_classes,
            cache=cache2,
            out_dir=out2,
            jobs=2,
        )

        assert len(recs_j1) == len(recs_j2) == 4

        # Compare scientific fields
        scientific_cols = [
            "D_U_R", "D_U_C", "D_V", "D_H", "D_M", "D_X", "mmd_sigma",
            "FPC", "PE", "PE_norm", "XB_soft_m2", "silhouette",
            "alignment_permutation", "alignment_global_margin",
            "source_model_fingerprint", "candidate_model_fingerprint",
            "shift_spec_sha256", "signal_record_sha256",
        ]

        df1 = pd.DataFrame(recs_j1).sort_values(by=["dataset_id", "condition"]).reset_index(drop=True)
        df2 = pd.DataFrame(recs_j2).sort_values(by=["dataset_id", "condition"]).reset_index(drop=True)

        for col in scientific_cols:
            if col in ["alignment_permutation", "source_model_fingerprint", "candidate_model_fingerprint", "shift_spec_sha256", "signal_record_sha256"]:
                assert (df1[col] == df2[col]).all(), f"Mismatch in {col} between jobs=1 and jobs=2"
            else:
                np.testing.assert_allclose(
                    df1[col].to_numpy(dtype=float),
                    df2[col].to_numpy(dtype=float),
                    atol=1e-7,
                    err_msg=f"Numeric mismatch in {col} between jobs=1 and jobs=2",
                )
