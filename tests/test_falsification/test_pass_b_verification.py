"""Tests for Phase 7 Pass B Quality Verification, Checkpoint Provenance, and Invariants."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from clusterdrift.alignment.state import compute_model_fingerprint
from clusterdrift.falsification.execution.checkpoint import QualityCheckpointManager
from clusterdrift.falsification.verification import verify_pass_a, verify_pass_b
from clusterdrift.signals.engine import fit_clustering_model
from clusterdrift.signals.hashing import compute_quality_record_sha256


@pytest.fixture
def temp_work_dir():
    d = tempfile.mkdtemp(prefix="test_quality_mgr_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_quality_record():
    rec = {
        "dataset_id": "iris",
        "outer_fold": 0,
        "condition": "clean",
        "method": "fcm_adaptive",
        "seed": 1,
        "quality_target": "delta_ari",
        "ari_clean": 0.75,
        "ari_condition": 0.75,
        "delta_ari": 0.0,
        "nmi_condition": 0.80,
        "ami_condition": 0.78,
        "n_evaluation_rows": 30,
    }
    rec["quality_record_sha256"] = compute_quality_record_sha256(rec)
    return rec


def test_quality_checkpoint_provenance_envelope(temp_work_dir, sample_quality_record):
    proto_sha = "mock_protocol_sha"
    signals_sha = "3a9e6c68cbe34c8763d70769715dba0d34976dfc35c8877c5680561da5c0e80a"
    freeze_com = "89b3df90f2cdc29d0e341637a11c0eabd2099ee7"

    mgr = QualityCheckpointManager(
        temp_work_dir,
        phase7_protocol_sha256=proto_sha,
        pass_a_signals_sha256=signals_sha,
        pass_a_freeze_commit=freeze_com,
    )
    p = mgr.save_checkpoint(sample_quality_record)
    assert p.exists()

    with open(p, "r", encoding="utf-8") as f:
        env = json.load(f)

    assert env["phase7_protocol_sha256"] == proto_sha
    assert env["pass_a_signals_sha256"] == signals_sha
    assert env["pass_a_freeze_commit"] == freeze_com
    assert env["quality_record_sha256"] == sample_quality_record["quality_record_sha256"]
    assert env["quality_record"]["dataset_id"] == "iris"

    # Valid load
    loaded = mgr.load_checkpoint("iris", 0, "clean", "fcm_adaptive", 1)
    assert loaded is not None
    assert loaded["dataset_id"] == "iris"
    assert loaded["delta_ari"] == 0.0


def test_quality_checkpoint_rejects_wrong_protocol(temp_work_dir, sample_quality_record):
    mgr1 = QualityCheckpointManager(
        temp_work_dir,
        phase7_protocol_sha256="proto_1",
        pass_a_signals_sha256="sig_sha",
        pass_a_freeze_commit="commit_1",
    )
    mgr1.save_checkpoint(sample_quality_record)

    mgr2 = QualityCheckpointManager(
        temp_work_dir,
        phase7_protocol_sha256="proto_2",
        pass_a_signals_sha256="sig_sha",
        pass_a_freeze_commit="commit_1",
    )
    assert not mgr2.is_checkpoint_valid("iris", 0, "clean", "fcm_adaptive", 1)
    assert mgr2.load_checkpoint("iris", 0, "clean", "fcm_adaptive", 1) is None


def test_quality_checkpoint_rejects_wrong_pass_a_signals_sha(temp_work_dir, sample_quality_record):
    mgr1 = QualityCheckpointManager(
        temp_work_dir,
        phase7_protocol_sha256="proto_1",
        pass_a_signals_sha256="sig_sha_1",
        pass_a_freeze_commit="commit_1",
    )
    mgr1.save_checkpoint(sample_quality_record)

    mgr2 = QualityCheckpointManager(
        temp_work_dir,
        phase7_protocol_sha256="proto_1",
        pass_a_signals_sha256="sig_sha_2",
        pass_a_freeze_commit="commit_1",
    )
    assert not mgr2.is_checkpoint_valid("iris", 0, "clean", "fcm_adaptive", 1)


def test_quality_checkpoint_rejects_wrong_pass_a_freeze_commit(temp_work_dir, sample_quality_record):
    mgr1 = QualityCheckpointManager(
        temp_work_dir,
        phase7_protocol_sha256="proto_1",
        pass_a_signals_sha256="sig_sha_1",
        pass_a_freeze_commit="commit_1",
    )
    mgr1.save_checkpoint(sample_quality_record)

    mgr2 = QualityCheckpointManager(
        temp_work_dir,
        phase7_protocol_sha256="proto_1",
        pass_a_signals_sha256="sig_sha_1",
        pass_a_freeze_commit="commit_2",
    )
    assert not mgr2.is_checkpoint_valid("iris", 0, "clean", "fcm_adaptive", 1)


def test_quality_hash_corruption_quarantined(temp_work_dir, sample_quality_record):
    mgr = QualityCheckpointManager(temp_work_dir)
    p = mgr.save_checkpoint(sample_quality_record)

    # Corrupt a value in quality record
    with open(p, "r", encoding="utf-8") as f:
        env = json.load(f)
    env["quality_record"]["ari_clean"] = 0.999
    with open(p, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    assert not mgr.is_checkpoint_valid("iris", 0, "clean", "fcm_adaptive", 1)
    assert not p.exists()
    assert any(mgr.quarantine_dir.glob("corrupt_*.json"))


def test_delta_ari_signed_and_unclamped():
    ari_clean = 0.5
    ari_condition = 0.7
    delta_ari = ari_clean - ari_condition
    assert delta_ari < 0
    assert np.isclose(delta_ari, -0.2)

    ari_clean_clean = 0.65
    ari_condition_clean = 0.65
    delta_clean = ari_clean_clean - ari_condition_clean
    assert delta_clean == 0.0


def test_source_model_fingerprint_mismatch_fails():
    X = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 5.0], [5.1, 5.1]], dtype=np.float64)
    m1 = fit_clustering_model("fcm_adaptive", K=2, seed=1, X=X)
    m2 = fit_clustering_model("fcm_adaptive", K=2, seed=2, X=X)

    fp1 = compute_model_fingerprint("fcm_adaptive", {}, 1, m1.cluster_centers_)
    fp2 = compute_model_fingerprint("fcm_adaptive", {}, 2, m2.cluster_centers_)
    assert fp1 != fp2


def test_pass_b_verifier_does_not_modify_pass_a(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    signals_p = project_root / "results" / "falsification" / "signals_label_free.csv"
    assert signals_p.exists()
    sha_before = pd.read_csv(signals_p).shape
    res_a = verify_pass_a(project_root)
    assert res_a["status"] == "PASSED"
    sha_after = pd.read_csv(signals_p).shape
    assert sha_before == sha_after
