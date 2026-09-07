"""Targeted unit tests for Signal Protocol Hashing, Record Hashing, and Invariance."""

import copy
import numpy as np
import pytest

from clusterdrift.signals.hashing import (
    compute_canonical_dict_sha256,
    compute_quality_record_sha256,
    compute_signal_protocol_sha256,
    compute_signal_record_sha256,
)


def test_signal_protocol_sha256_stability_and_sensitivity():
    """Test protocol SHA is deterministic and sensitive to protocol modifications."""
    base_cfg = {
        "protocol_version": 1,
        "signal_definition_version": 1,
        "global_signal_seed": 2026090706,
        "js": {"log_base": "natural", "normalize_by_ln2": True},
        "prototype_movement": {"normalization": "reference_cluster_rms_radius"},
        "entropy": {"normalize_by_lnK": True},
        "cluster_mass": {"bank": "current"},
        "mmd": {"kernel": "rbf", "estimator": "biased"},
        "validity": {"bank": "current"},
    }

    sha1 = compute_signal_protocol_sha256(base_cfg)
    sha2 = compute_signal_protocol_sha256(base_cfg)
    assert sha1 == sha2

    # Modify a key
    mod_cfg = copy.deepcopy(base_cfg)
    mod_cfg["mmd"]["estimator"] = "unbiased"
    sha_mod = compute_signal_protocol_sha256(mod_cfg)
    assert sha_mod != sha1


def test_signal_record_sha256_binding():
    """Test signal record SHA binds all primary signals and alignment fields."""
    rec = {
        "signal_protocol_sha256": "proto_123",
        "dataset_id": "iris",
        "outer_fold": 0,
        "condition": "location_severe",
        "method": "fcm_adaptive",
        "seed": 1,
        "reference_bank_sha256": "ref_sha",
        "current_bank_sha256": "cur_sha",
        "shift_spec_sha256": "spec_sha",
        "source_model_fingerprint": "fp_src",
        "candidate_model_fingerprint": "fp_cand",
        "alignment_permutation": [0, 1, 2],
        "alignment_global_margin": 0.15,
        "D_U_R": 0.123,
        "D_U_C": 0.456,
        "D_V": 0.789,
        "D_H": 0.050,
        "D_M": 0.100,
        "D_X": 0.350,
        "FPC": 0.85,
        "PE_norm": 0.15,
        "XB_soft_m2": 0.42,
        "silhouette": 0.65,
        "usable": True,
    }
    sha_rec = compute_signal_record_sha256(rec)
    assert len(sha_rec) == 64

    # Changing any primary signal alters hash
    rec_tampered = dict(rec)
    rec_tampered["D_V"] = 0.790
    assert compute_signal_record_sha256(rec_tampered) != sha_rec

    # Changing shift_replay_sha256 alters hash
    rec_replay = dict(rec)
    rec_replay["shift_replay_sha256"] = "replay_sha_abc123"
    assert compute_signal_record_sha256(rec_replay) != sha_rec


def test_quality_record_sha256_binding():
    """Test quality record SHA binds delta_ari and quality target."""
    q_rec = {
        "dataset_id": "iris",
        "outer_fold": 0,
        "condition": "clean",
        "method": "gmm",
        "seed": 3,
        "quality_target": "deployed_source_model",
        "ari_clean": 0.95,
        "ari_condition": 0.95,
        "delta_ari": 0.0,
        "nmi_condition": 0.92,
        "ami_condition": 0.91,
        "n_evaluation_rows": 30,
    }
    sha_q = compute_quality_record_sha256(q_rec)
    assert len(sha_q) == 64

    # Tampering alters hash
    q_tampered = dict(q_rec)
    q_tampered["delta_ari"] = 0.05
    assert compute_quality_record_sha256(q_tampered) != sha_q
