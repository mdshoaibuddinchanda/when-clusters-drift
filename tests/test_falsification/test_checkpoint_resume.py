import json
import pytest
from pathlib import Path
from clusterdrift.falsification.execution.checkpoint import SignalCheckpointManager
from clusterdrift.signals.hashing import compute_signal_record_sha256

def test_checkpoint_interruption_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[2]
    mgr = SignalCheckpointManager(work_dir=tmp_path, project_root=root)

    expected_keys = [
        ("iris", 0, "clean", "fcm_adaptive", 1),
        ("iris", 0, "clean", "fcm_adaptive", 2),
        ("iris", 0, "location_mild", "fcm_adaptive", 1),
        ("iris", 0, "location_mild", "fcm_adaptive", 2),
    ]

    # First run: only 2 completed before interrupt
    for ds, fold, cond, meth, seed in expected_keys[:2]:
        rec = {
            "dataset_id": ds,
            "outer_fold": fold,
            "condition": cond,
            "method": meth,
            "seed": seed,
            "signal_protocol_sha256": "p1",
            "reference_bank_sha256": "r1",
            "current_bank_sha256": "c1",
            "shift_spec_sha256": "s1",
            "shift_spec_file_sha256": "sf1",
            "shift_replay_sha256": "",
            "source_model_fingerprint": "src_fp",
            "candidate_model_fingerprint": "cand_fp",
            "alignment_permutation": [0, 1],
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
        rec["signal_record_sha256"] = compute_signal_record_sha256(rec)
        mgr.save_checkpoint(rec)

    completed = mgr.get_completed_checkpoint_keys()
    assert len(completed) == 2

    # Second run (resume): only remaining 2 are computed
    recomputed_count = 0
    newly_computed_count = 0
    for k in expected_keys:
        ds, fold, cond, meth, seed = k
        if mgr.is_checkpoint_valid(ds, fold, cond, meth, seed):
            # Skipped!
            pass
        else:
            newly_computed_count += 1
            rec = {
                "dataset_id": ds,
                "outer_fold": fold,
                "condition": cond,
                "method": meth,
                "seed": seed,
                "signal_protocol_sha256": "p1",
                "reference_bank_sha256": "r1",
                "current_bank_sha256": "c1",
                "shift_spec_sha256": "s1",
                "shift_spec_file_sha256": "sf1",
                "shift_replay_sha256": "",
                "source_model_fingerprint": "src_fp",
                "candidate_model_fingerprint": "cand_fp",
                "alignment_permutation": [0, 1],
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
            rec["signal_record_sha256"] = compute_signal_record_sha256(rec)
            mgr.save_checkpoint(rec)

    assert newly_computed_count == 2
    assert len(mgr.get_completed_checkpoint_keys()) == 4

    out_csv = tmp_path / "assembled.csv"
    df = mgr.assemble_signals_csv(out_csv, expected_keys)
    assert len(df) == 4
