import json
import pytest
from pathlib import Path
from clusterdrift.falsification.execution.checkpoint import SignalCheckpointManager
from clusterdrift.signals.hashing import compute_signal_record_sha256

def test_checkpoint_validation_and_quarantine(tmp_path):
    root = Path(__file__).resolve().parents[2]
    mgr = SignalCheckpointManager(work_dir=tmp_path, project_root=root)

    valid_rec = {
        "dataset_id": "iris",
        "outer_fold": 0,
        "condition": "clean",
        "method": "fcm_adaptive",
        "seed": 1,
        "signal_protocol_sha256": "proto123",
        "reference_bank_sha256": "ref123",
        "current_bank_sha256": "cur123",
        "shift_spec_sha256": "spec123",
        "shift_spec_file_sha256": "file123",
        "shift_replay_sha256": "",
        "source_model_fingerprint": "src_fp",
        "candidate_model_fingerprint": "cand_fp",
        "alignment_permutation": [0, 1, 2],
        "alignment_global_margin": 0.5,
        "D_U_R": 0.1,
        "D_U_C": 0.1,
        "D_V": 0.1,
        "D_H": 0.0,
        "D_M": 0.0,
        "D_X": 0.0,
        "FPC": 0.8,
        "PE_norm": 0.2,
        "XB_soft_m2": 0.3,
        "silhouette": 0.6,
        "usable": True,
    }
    valid_rec["signal_record_sha256"] = compute_signal_record_sha256(valid_rec)

    # 1. Save valid checkpoint
    mgr.save_checkpoint(valid_rec)
    assert mgr.is_checkpoint_valid("iris", 0, "clean", "fcm_adaptive", 1, "proto123") is True

    # 2. Corrupt hash
    corrupt_rec = dict(valid_rec)
    corrupt_rec["signal_record_sha256"] = "wrong_hash"
    mgr.save_checkpoint(corrupt_rec)
    # Validation should fail and quarantine file
    assert mgr.is_checkpoint_valid("iris", 0, "clean", "fcm_adaptive", 1, "proto123") is False
    assert len(list(mgr.quarantine_dir.glob("*"))) > 0

    # 3. Truncated JSON
    cp_path = mgr._get_checkpoint_path("iris", 0, "clean", 1)
    cp_path.write_text("{ incomplete json", encoding="utf-8")
    assert mgr.is_checkpoint_valid("iris", 0, "clean", "fcm_adaptive", 1, "proto123") is False

    # 4. Wrong protocol SHA
    mgr.save_checkpoint(valid_rec)
    assert mgr.is_checkpoint_valid("iris", 0, "clean", "fcm_adaptive", 1, "different_proto") is False
