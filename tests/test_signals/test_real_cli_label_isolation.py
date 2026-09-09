"""Real CLI orchestration label-isolation test for Pass A."""

import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import yaml

from clusterdrift.shifts.replay import build_local_overlap_replay_descriptor
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


def test_real_cli_label_isolation_orchestration(monkeypatch):
    """Verify that run_signals_pass_a executes on iris with class_prevalence and local_overlap

    with labels.parquet completely blocked and inaccessible.
    """
    # 1. Ensure local-overlap replay descriptor exists for iris fold 0
    ds = "iris"
    fold = 0
    cond = "local_overlap_severe"
    spec_p = PROJECT_ROOT / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"
    desc_dir = PROJECT_ROOT / "data" / "signals" / "offline_shift_replay" / ds / f"fold_{fold}"
    desc_p = desc_dir / f"{cond}.json"

    if not desc_p.exists():
        desc_dir.mkdir(parents=True, exist_ok=True)
        ds_dir = PROJECT_ROOT / "data" / "canonical" / "controlled" / ds
        df_X = pd.read_parquet(ds_dir / "features.parquet")
        df_y = pd.read_parquet(ds_dir / "labels.parquet")
        fold_p = PROJECT_ROOT / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
        with np.load(fold_p) as npz:
            X_src = df_X.iloc[npz["source_indices"]].reset_index(drop=True)
            X_tgt = df_X.iloc[npz["target_indices"]].reset_index(drop=True)
            y_src = df_y.iloc[npz["source_indices"]].to_numpy().ravel()
            y_tgt = df_y.iloc[npz["target_indices"]].to_numpy().ravel()
        with open(ds_dir / "metadata.json") as f:
            meta = json.load(f)
        roles = meta.get("feature_roles", {c: "numeric" for c in X_src.columns})
        desc = build_local_overlap_replay_descriptor(
            X_target_raw=X_tgt,
            y_target_raw=y_tgt,
            X_source_raw=X_src,
            y_source_raw=y_src,
            roles=roles,
            dataset_id=ds,
            outer_fold=fold,
            condition=cond,
            phase4_spec_path=spec_p,
            project_root=PROJECT_ROOT,
        )
        with open(desc_p, "w", encoding="utf-8") as f:
            json.dump(desc, f, indent=2)

    # 2. Block all access to labels.parquet via monkeypatch
    orig_read_parquet = pd.read_parquet

    def guarded_read_parquet(path, *args, **kwargs):
        path_str = str(path).replace("\\", "/")
        if "labels.parquet" in path_str:
            raise PermissionError(f"[LABEL ACCESS FORBIDDEN] Attempted to read {path} during label-free Pass A!")
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read_parquet)

    # 3. Configure reduced validation panel
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

    # Reduced config: iris with class_prevalence_severe and local_overlap_severe
    test_sig_cfg = dict(sig_cfg)
    test_sig_cfg["validation"] = {
        "datasets": ["iris"],
        "outer_fold": 0,
        "conditions": ["class_prevalence_severe", "local_overlap_severe"],
        "methods": ["fcm_adaptive"],
        "algorithm_seeds": [1],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir)
        cache = SignalCache()
        records = run_signals_pass_a(
            project_root=PROJECT_ROOT,
            sig_cfg=test_sig_cfg,
            alignment_cfg=alignment_cfg,
            methods_cfg=methods_cfg,
            prep_cfg=prep_cfg,
            shift_cfg=shift_cfg,
            dataset_classes={"iris": 3},
            cache=cache,
            out_dir=tmp_out,
            jobs=1,
        )

        assert len(records) == 2
        for r in records:
            assert r["usable"] is True
            assert 0.0 <= r["D_U_R"] <= 1.0
            assert 0.0 <= r["D_U_C"] <= 1.0
            assert r["shift_spec_sha256"] is not None
            assert r["shift_spec_file_sha256"] is not None
            if r["condition"].startswith("local_overlap"):
                assert len(r.get("shift_replay_sha256", "")) > 0
