"""Tests for D_X scenario invariance and source-only sigma consistency."""

import importlib.util
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import yaml

from clusterdrift.signals.engine import SignalCache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_spec = importlib.util.spec_from_file_location("compute_signals_script", PROJECT_ROOT / "scripts" / "06_compute_signals.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_signals_pass_a = _mod.run_signals_pass_a


def test_dx_and_sigma_invariance_reduced_panel():
    """Verify that D_X is invariant across methods and seeds for the same scenario,

    and sigma is invariant across conditions for the same dataset and fold.
    """
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
        "datasets": ["iris"],
        "outer_fold": 0,
        "conditions": ["clean", "location_severe"],
        "methods": ["fcm_adaptive", "gmm"],
        "algorithm_seeds": [1, 2],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir)
        cache = SignalCache()
        records = run_signals_pass_a(
            project_root=PROJECT_ROOT,
            sig_cfg=reduced_sig_cfg,
            alignment_cfg=alignment_cfg,
            methods_cfg=methods_cfg,
            prep_cfg=prep_cfg,
            shift_cfg=shift_cfg,
            dataset_classes={"iris": 3},
            cache=cache,
            out_dir=tmp_out,
            jobs=1,
        )

        df = pd.DataFrame(records)
        # 1. D_X invariance across (method, seed) for each condition
        for cond, group in df.groupby("condition"):
            dx_vals = group["D_X"].to_numpy()
            assert len(dx_vals) == 4  # 2 methods * 2 seeds
            np.testing.assert_allclose(
                dx_vals, dx_vals[0], atol=1e-7,
                err_msg=f"D_X varies across method/seed for condition {cond}",
            )

        # 2. Sigma invariance across all conditions for the same dataset/fold
        sig_vals = df["mmd_sigma"].to_numpy()
        assert len(sig_vals) == 8  # 2 conditions * 2 methods * 2 seeds
        np.testing.assert_allclose(
            sig_vals, sig_vals[0], atol=1e-7,
            err_msg="MMD sigma varies across conditions for the same dataset",
        )
