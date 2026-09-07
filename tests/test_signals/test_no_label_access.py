"""Static AST and Runtime Monkeypatch Tests for Absolute Label Isolation."""

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.signals.engine import SignalCache, SignalEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_static_ast_no_label_imports():
    """Static AST audit ensuring src/clusterdrift/signals never imports label or external metric modules."""
    signals_dir = PROJECT_ROOT / "src" / "clusterdrift" / "signals"
    forbidden_modules = [
        "clusterdrift.metrics.external",
        "sklearn.metrics.adjusted_rand_score",
        "sklearn.metrics.normalized_mutual_info_score",
        "sklearn.metrics.adjusted_mutual_info_score",
    ]
    forbidden_tokens = ["labels.parquet", "y_true", "y_pred", "adjusted_rand_index"]

    for py_file in signals_dir.glob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(py_file))

        # Check import nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forb in forbidden_modules:
                        assert not alias.name.startswith(forb), f"Forbidden import '{alias.name}' in {py_file.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for forb in forbidden_modules:
                    assert not mod.startswith(forb), f"Forbidden import from '{mod}' in {py_file.name}"

        # Check forbidden tokens in source
        for tok in forbidden_tokens:
            assert tok not in code, f"Forbidden token '{tok}' found in {py_file.name}"


def test_runtime_monkeypatch_no_label_access(monkeypatch):
    """Runtime test: monkeypatch parquet reader to blow up if labels.parquet is read during signal computation."""
    orig_read_parquet = pd.read_parquet

    def guarded_read_parquet(path, *args, **kwargs):
        path_str = str(path).replace("\\", "/")
        if "labels.parquet" in path_str:
            raise PermissionError(f"[LABEL LEAKAGE VIOLATION] Attempted to read labels in label-free pass: {path}")
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read_parquet)

    # Construct synthetic inputs for SignalEngine
    K = 2
    D = 3
    B = 20
    rng = np.random.RandomState(42)
    X_src = rng.randn(B, D)
    X_tgt = rng.randn(B, D) + 1.0
    A_R = rng.randn(10, D)
    A_C = rng.randn(10, D) + 1.0

    sig_cfg = {
        "protocol_version": 1,
        "signal_definition_version": 1,
        "global_signal_seed": 2026090706,
        "js": {"log_base": "natural", "normalize_by_ln2": True},
        "prototype_movement": {"normalization": "reference_cluster_rms_radius", "epsilon": 1e-12},
        "entropy": {"normalize_by_lnK": True},
        "cluster_mass": {"bank": "current"},
        "mmd": {"kernel": "rbf", "estimator": "biased", "bandwidth_max_points": 1024, "chunk_size": 512, "negative_tolerance": 1e-10},
        "validity": {"bank": "current", "model": "deployed_source"},
    }

    engine = SignalEngine(signals_cfg=sig_cfg)

    # This MUST succeed without touching labels.parquet
    res = engine.compute_signals(
        dataset_id="test_ds",
        outer_fold=0,
        condition="clean",
        method="fcm_adaptive",
        seed=1,
        K=K,
        X_source_trans=X_src,
        X_target_shifted_trans=X_tgt,
        A_R=A_R,
        A_C=A_C,
        ref_bank_sha256="ref_hash_123",
        cur_bank_sha256="cur_hash_123",
        shift_spec_sha256="shift_hash_123",
    )

    assert res.usable is True
    assert 0.0 <= res.D_U_R <= 1.0
    assert 0.0 <= res.D_U_C <= 1.0
    assert res.D_V >= 0.0
