"""Targeted tests for nested offline shift replay discovery, schema, and Phase 4 bindings."""

import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pytest

from clusterdrift.falsification.protocol import load_falsification_config
from clusterdrift.falsification.verification import verify_offline_shift_replay


def test_nested_replay_discovery():
    """Verify that all 120 replay descriptors and NPZs are discovered in nested structure."""
    root = Path(__file__).resolve().parents[2]
    res = verify_offline_shift_replay(root)
    assert res["status"] == "VALID"
    assert res["verified_scenarios"] == 120
    assert res["json_count"] == 120
    assert res["npz_count"] == 120


def test_missing_replay_json_rejected():
    """Verify that a missing replay JSON file raises FileNotFoundError."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        replay_src = root / "data" / "falsification" / "offline_shift_replay"
        replay_dst = tmp_root / "data" / "falsification" / "offline_shift_replay"
        shutil.copytree(replay_src, replay_dst)

        # Delete one json
        del_target = replay_dst / "iris" / "fold_0" / "local_overlap_mild.json"
        del_target.unlink()

        with pytest.raises(ValueError, match="Expected exactly 120 replay JSON files"):
            verify_offline_shift_replay(tmp_root, cfg=cfg)


def test_missing_replay_npz_rejected():
    """Verify that a missing companion NPZ file raises FileNotFoundError."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        replay_src = root / "data" / "falsification" / "offline_shift_replay"
        replay_dst = tmp_root / "data" / "falsification" / "offline_shift_replay"
        shutil.copytree(replay_src, replay_dst)

        # Delete one npz
        del_target = replay_dst / "iris" / "fold_0" / "local_overlap_mild.npz"
        del_target.unlink()

        with pytest.raises(ValueError, match="Expected exactly 120 replay NPZ files"):
            verify_offline_shift_replay(tmp_root, cfg=cfg)


def test_wrong_replay_descriptor_sha_rejected():
    """Verify that tampering with descriptor fields fails self-hash verification."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        replay_src = root / "data" / "falsification" / "offline_shift_replay"
        replay_dst = tmp_root / "data" / "falsification" / "offline_shift_replay"
        shutil.copytree(replay_src, replay_dst)

        target_json = replay_dst / "iris" / "fold_0" / "local_overlap_mild.json"
        with open(target_json, "r") as f:
            data = json.load(f)

        data["affected_numeric_columns"] = ["tampered_column"]
        with open(target_json, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Replay descriptor self-hash mismatch"):
            verify_offline_shift_replay(tmp_root, cfg=cfg)


def test_wrong_replay_npz_sha_rejected():
    """Verify that tampering with the NPZ file fails replay_npz_sha256 check."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        replay_src = root / "data" / "falsification" / "offline_shift_replay"
        replay_dst = tmp_root / "data" / "falsification" / "offline_shift_replay"
        shutil.copytree(replay_src, replay_dst)

        target_npz = replay_dst / "iris" / "fold_0" / "local_overlap_mild.npz"
        # Overwrite with dummy bytes
        target_npz.write_bytes(b"corrupt_bytes_for_testing")

        with pytest.raises(ValueError, match="Replay NPZ file hash mismatch"):
            verify_offline_shift_replay(tmp_root, cfg=cfg)


def test_wrong_component_array_sha_rejected():
    """Verify that mismatch between NPZ array content and descriptor array SHA is caught."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        replay_src = root / "data" / "falsification" / "offline_shift_replay"
        replay_dst = tmp_root / "data" / "falsification" / "offline_shift_replay"
        shutil.copytree(replay_src, replay_dst)

        target_json = replay_dst / "iris" / "fold_0" / "local_overlap_mild.json"

        with open(target_json, "r") as f:
            data = json.load(f)

        # Alter selected_target_positions_sha256 in descriptor and update self-hash
        data["selected_target_positions_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        from clusterdrift.shifts.hashing import compute_canonical_json_sha256
        data_no_sha = {k: v for k, v in data.items() if k != "replay_descriptor_sha256"}
        data["replay_descriptor_sha256"] = compute_canonical_json_sha256(data_no_sha)

        with open(target_json, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="selected_target_positions_sha256 mismatch"):
            verify_offline_shift_replay(tmp_root, cfg=cfg)


def test_wrong_phase4_scenario_binding_rejected():
    """Verify that a mismatch with Phase-4 shift spec SHA is rejected."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_falsification_config(root / "configs" / "falsification.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        replay_src = root / "data" / "falsification" / "offline_shift_replay"
        replay_dst = tmp_root / "data" / "falsification" / "offline_shift_replay"
        shutil.copytree(replay_src, replay_dst)

        # Copy shifts specs
        shifts_src = root / "data" / "shifts" / "specs"
        shifts_dst = tmp_root / "data" / "shifts" / "specs"
        shutil.copytree(shifts_src, shifts_dst)

        target_json = replay_dst / "iris" / "fold_0" / "local_overlap_mild.json"
        with open(target_json, "r") as f:
            data = json.load(f)

        data["phase4_shift_spec_sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        from clusterdrift.shifts.hashing import compute_canonical_json_sha256
        data_no_sha = {k: v for k, v in data.items() if k != "replay_descriptor_sha256"}
        data["replay_descriptor_sha256"] = compute_canonical_json_sha256(data_no_sha)

        with open(target_json, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Phase 4 shift spec SHA mismatch"):
            verify_offline_shift_replay(tmp_root, cfg=cfg)
