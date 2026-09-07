"""Tests for independent quality pass in a fresh process with cold cache."""

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
run_quality_pass_b = _mod.run_quality_pass_b


def test_fresh_process_quality_with_empty_cache():
    """Verify run_quality_pass_b fits missing source models in cold cache and completes successfully."""
    # Use existing signal records if available or dummy record for iris
    sig_csv = PROJECT_ROOT / "results" / "signal_validation" / "signals_label_free.csv"
    if sig_csv.exists():
        df_sig = pd.read_csv(sig_csv)
        # Select subset of records (e.g. iris records)
        iris_recs = df_sig[df_sig["dataset_id"] == "iris"].to_dict(orient="records")
    else:
        iris_recs = [{
            "dataset_id": "iris",
            "outer_fold": 0,
            "condition": "clean",
            "method": "fcm_adaptive",
            "seed": 1,
        }]

    with open(PROJECT_ROOT / "configs" / "signals.yaml") as f:
        sig_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "preprocessing.yaml") as f:
        prep_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "shifts.yaml") as f:
        shift_cfg = yaml.safe_load(f)

    # COLD CACHE: completely empty, no source models pre-populated!
    cold_cache = SignalCache()
    assert cold_cache.stats()["cached_source_models"] == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir)
        qual_records = run_quality_pass_b(
            project_root=PROJECT_ROOT,
            signal_records=iris_recs,
            sig_cfg=sig_cfg,
            prep_cfg=prep_cfg,
            shift_cfg=shift_cfg,
            cache=cold_cache,
            out_dir=tmp_out,
        )

        assert len(qual_records) == len(iris_recs)
        # Source models must now be populated in the cache
        assert cold_cache.stats()["cached_source_models"] > 0
        for qr in qual_records:
            assert "ari_clean" in qr
            assert "ari_condition" in qr
            assert "delta_ari" in qr
            assert "quality_record_sha256" in qr


def test_k_provenance_independent_of_labels(monkeypatch):
    """Verify that K is loaded from signal record/manifest and NEVER from len(np.unique(y_src))."""
    with open(PROJECT_ROOT / "configs" / "signals.yaml") as f:
        sig_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "preprocessing.yaml") as f:
        prep_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "shifts.yaml") as f:
        shift_cfg = yaml.safe_load(f)

    # Monkeypatch read_parquet for labels to return only 2 unique classes
    orig_read_parquet = pd.read_parquet

    def mock_read_parquet(path, *args, **kwargs):
        df = orig_read_parquet(path, *args, **kwargs)
        if "labels.parquet" in str(path).replace("\\", "/"):
            # Collapse all labels to 2 classes (0 and 1)
            return pd.DataFrame({"label": np.zeros(len(df), dtype=int)})
        return df

    monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)

    # Signal record has K=3 (canonical for iris)
    iris_recs = [{
        "dataset_id": "iris",
        "outer_fold": 0,
        "condition": "clean",
        "method": "fcm_adaptive",
        "seed": 1,
        "K": 3,
    }]

    cache = SignalCache()
    with tempfile.TemporaryDirectory() as tmpdir:
        qual_records = run_quality_pass_b(
            project_root=PROJECT_ROOT,
            signal_records=iris_recs,
            sig_cfg=sig_cfg,
            prep_cfg=prep_cfg,
            shift_cfg=shift_cfg,
            cache=cache,
            out_dir=Path(tmpdir),
            dataset_classes={"iris": 3},
        )
        assert len(qual_records) == 1
        m_src = cache.get_source_model("iris", 0, "fcm_adaptive", 1)
        # Model must have been fitted with K=3, NOT K=1 or K=2 from tampered labels!
        assert m_src.cluster_centers_.shape[0] == 3


def test_inconsistent_k_in_records_rejected():
    """Verify that inconsistent K values in signal records for same method/seed raise ValueError."""
    with open(PROJECT_ROOT / "configs" / "signals.yaml") as f:
        sig_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "preprocessing.yaml") as f:
        prep_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "shifts.yaml") as f:
        shift_cfg = yaml.safe_load(f)

    inconsistent_recs = [
        {"dataset_id": "iris", "outer_fold": 0, "condition": "clean", "method": "fcm_adaptive", "seed": 1, "K": 3},
        {"dataset_id": "iris", "outer_fold": 0, "condition": "scale_severe", "method": "fcm_adaptive", "seed": 1, "K": 4},
    ]
    cache = SignalCache()
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Inconsistent K values"):
            run_quality_pass_b(
                project_root=PROJECT_ROOT,
                signal_records=inconsistent_recs,
                sig_cfg=sig_cfg,
                prep_cfg=prep_cfg,
                shift_cfg=shift_cfg,
                cache=cache,
                out_dir=Path(tmpdir),
                dataset_classes={"iris": 3},
            )
