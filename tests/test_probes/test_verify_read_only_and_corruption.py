"""Test corruption detection and read-only verification behavior."""

import json
import os
from pathlib import Path
import numpy as np
import pytest

from clusterdrift.probes.bank import save_reference_probe_descriptor
from clusterdrift.probes.hashing import compute_file_sha256, derive_reference_probe_seed
from clusterdrift.probes.validation import validate_saved_reference_descriptor
from clusterdrift.shifts.hashing import atomic_write_npz


def test_corruption_of_array_in_npz_detected(tmp_path: Path):
    """Corrupting array data in companion NPZ is caught by validation."""
    pos = np.arange(50, dtype=np.int64)
    can = np.arange(50, dtype=np.int64)
    seed = derive_reference_probe_seed(2026090705, "iris", 0, "split_sha")

    spec_path = tmp_path / "reference" / "iris" / "fold_0.json"
    metadata = {
        "dataset_id": "iris",
        "outer_fold": 0,
        "canonical_bundle_sha256": "bundle_sha",
        "split_sha256": "split_sha",
        "preprocessing_config_sha256": "prep_sha",
        "selection_policy": "uniform_without_replacement",
        "global_probe_seed": 2026090705,
        "derived_seed": seed,
        "available_rows": 100,
        "probe_protocol_sha256": "proto_sha",
    }

    desc = save_reference_probe_descriptor(spec_path, metadata, pos, can)

    # Validate passes initially
    is_val, err, _ = validate_saved_reference_descriptor(
        spec_path=spec_path,
        expected_canonical_bundle_sha256="bundle_sha",
        expected_split_sha256="split_sha",
        expected_preprocessing_config_sha256="prep_sha",
        expected_probe_protocol_sha256="proto_sha",
        global_probe_seed=2026090705,
        dataset_id="iris",
        outer_fold=0,
    )
    assert is_val is True
    assert err is None

    # Corrupt array in NPZ without changing file corruption error (save altered array)
    npz_path = spec_path.with_suffix(".npz")
    corrupted_pos = np.arange(10, 60, dtype=np.int64)
    atomic_write_npz(npz_path, selected_source_positions=corrupted_pos, canonical_row_indices=can)

    is_val_corrupt, err_corrupt, _ = validate_saved_reference_descriptor(
        spec_path=spec_path,
        expected_canonical_bundle_sha256="bundle_sha",
        expected_split_sha256="split_sha",
        expected_preprocessing_config_sha256="prep_sha",
        expected_probe_protocol_sha256="proto_sha",
        global_probe_seed=2026090705,
        dataset_id="iris",
        outer_fold=0,
    )
    assert is_val_corrupt is False
    assert "mismatch" in str(err_corrupt).lower()


def test_corruption_of_json_hash_detected(tmp_path: Path):
    """Altering hash in JSON descriptor is caught by validation."""
    pos = np.arange(50, dtype=np.int64)
    can = np.arange(50, dtype=np.int64)
    seed = derive_reference_probe_seed(2026090705, "iris", 0, "split_sha")

    spec_path = tmp_path / "reference" / "iris" / "fold_0.json"
    metadata = {
        "dataset_id": "iris",
        "outer_fold": 0,
        "canonical_bundle_sha256": "bundle_sha",
        "split_sha256": "split_sha",
        "preprocessing_config_sha256": "prep_sha",
        "selection_policy": "uniform_without_replacement",
        "global_probe_seed": 2026090705,
        "derived_seed": seed,
        "available_rows": 100,
        "probe_protocol_sha256": "proto_sha",
    }

    desc = save_reference_probe_descriptor(spec_path, metadata, pos, can)

    # Tamper with JSON probe_bank_sha256
    with open(spec_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    doc["probe_bank_sha256"] = "bad" + doc["probe_bank_sha256"][3:]
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    is_val, err, _ = validate_saved_reference_descriptor(
        spec_path=spec_path,
        expected_canonical_bundle_sha256="bundle_sha",
        expected_split_sha256="split_sha",
        expected_preprocessing_config_sha256="prep_sha",
        expected_probe_protocol_sha256="proto_sha",
        global_probe_seed=2026090705,
        dataset_id="iris",
        outer_fold=0,
    )
    assert is_val is False
    assert "probe bank sha-256 mismatch" in str(err).lower()


# ---------------------------------------------------------------------------
# Phase 5.1a Verification Failure Regression Tests (Exercising Real Verify Path)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from clusterdrift.probes.verification import verify_phase5_integrity
import clusterdrift.probes.verification as probe_verif


def test_verify_fails_on_datasets_manifest_hash_change(monkeypatch):
    """1. Verify fails when Phase-1 datasets manifest hash changes."""
    real_compute = probe_verif.compute_file_sha256
    def fake_sha(p):
        if str(p).endswith("datasets.json"):
            return "tampered_datasets_sha256"
        return real_compute(p)
    monkeypatch.setattr(probe_verif, "compute_file_sha256", fake_sha)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("phase1_datasets_manifest_sha256 mismatch" in e for e in errs)


def test_verify_fails_on_split_manifest_hash_change(monkeypatch):
    """2. Verify fails when Phase-2 split manifest hash changes."""
    real_compute = probe_verif.compute_file_sha256
    def fake_sha(p):
        if str(p).endswith("split_manifest.json"):
            return "tampered_split_sha256"
        return real_compute(p)
    monkeypatch.setattr(probe_verif, "compute_file_sha256", fake_sha)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("phase2_split_manifest_sha256 mismatch" in e for e in errs)


def test_verify_fails_on_methods_config_change(monkeypatch):
    """3. Verify fails when methods config changes."""
    real_compute = probe_verif.compute_file_sha256
    def fake_sha(p):
        if str(p).endswith("methods.yaml"):
            return "tampered_methods_sha256"
        return real_compute(p)
    monkeypatch.setattr(probe_verif, "compute_file_sha256", fake_sha)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("methods_config_sha256 mismatch" in e for e in errs)


def test_verify_fails_on_probes_config_file_change(monkeypatch):
    """4. Verify fails when probes config file changes."""
    real_compute = probe_verif.compute_file_sha256
    def fake_sha(p):
        if str(p).endswith("probes.yaml"):
            return "tampered_probes_sha256"
        return real_compute(p)
    monkeypatch.setattr(probe_verif, "compute_file_sha256", fake_sha)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("probe_config_sha256 mismatch" in e for e in errs)


def test_verify_fails_on_alignment_config_file_change(monkeypatch):
    """5. Verify fails when alignment config file changes."""
    real_compute = probe_verif.compute_file_sha256
    def fake_sha(p):
        if str(p).endswith("alignment.yaml"):
            return "tampered_alignment_sha256"
        return real_compute(p)
    monkeypatch.setattr(probe_verif, "compute_file_sha256", fake_sha)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("alignment_config_sha256 mismatch" in e for e in errs)


def test_verify_fails_on_wrong_generated_from_commit():
    """6. Verify fails when generated_from_commit is wrong."""
    errs = verify_phase5_integrity(PROJECT_ROOT, expected_producer_commit="bad_producer_commit_hash")
    assert any("generated_from_commit mismatch in lock" in e for e in errs)


def test_verify_fails_on_semantic_reference_row_mapping_wrong(monkeypatch):
    """7. Verify fails when reference canonical-row mapping is self-consistently but semantically wrong."""
    real_load = np.load

    class MockReferenceNpz:
        def __init__(self, npz):
            self.npz = npz
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.npz.close()
        def get(self, k, default=None):
            val = self.npz.get(k, default)
            if k == "canonical_row_indices":
                # Shift row indices to be semantically invalid vs source_indices[positions]
                return val + 1000
            return val
        def __getitem__(self, k):
            val = self.npz[k]
            if k == "canonical_row_indices":
                return val + 1000
            return val
        def __contains__(self, k):
            return k in self.npz

    def fake_load(p, *args, **kwargs):
        npz = real_load(p, *args, **kwargs)
        if "reference" in str(p) and "iris" in str(p) and "fold_0.npz" in str(p):
            return MockReferenceNpz(npz)
        return npz

    monkeypatch.setattr(np, "load", fake_load)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("canonical_row_indices != source_indices[selected_source_positions]" in e for e in errs)


def test_verify_fails_on_semantic_current_target_position_mapping_wrong(monkeypatch):
    """8. Verify fails when current target-position mapping is wrong."""
    real_load = np.load

    class MockCurrentNpz:
        def __init__(self, npz):
            self.npz = npz
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.npz.close()
        def get(self, k, default=None):
            val = self.npz.get(k, default)
            if k == "target_partition_positions":
                return np.roll(val, 1)
            return val
        def __getitem__(self, k):
            val = self.npz[k]
            if k == "target_partition_positions":
                return np.roll(val, 1)
            return val
        def __contains__(self, k):
            return k in self.npz

    def fake_load(p, *args, **kwargs):
        npz = real_load(p, *args, **kwargs)
        if "current" in str(p) and "iris" in str(p) and "clean.npz" in str(p):
            return MockCurrentNpz(npz)
        return npz

    monkeypatch.setattr(np, "load", fake_load)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("target_partition_positions != row_index_map[selected_current_positions]" in e for e in errs)


def test_verify_fails_on_semantic_current_canonical_row_mapping_wrong(monkeypatch):
    """9. Verify fails when current canonical-row mapping is wrong."""
    real_load = np.load

    class MockCurrentNpz:
        def __init__(self, npz):
            self.npz = npz
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.npz.close()
        def get(self, k, default=None):
            val = self.npz.get(k, default)
            if k == "canonical_row_indices":
                return val + 1
            return val
        def __getitem__(self, k):
            val = self.npz[k]
            if k == "canonical_row_indices":
                return val + 1
            return val
        def __contains__(self, k):
            return k in self.npz

    def fake_load(p, *args, **kwargs):
        npz = real_load(p, *args, **kwargs)
        if "current" in str(p) and "iris" in str(p) and "clean.npz" in str(p):
            return MockCurrentNpz(npz)
        return npz

    monkeypatch.setattr(np, "load", fake_load)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("canonical_row_indices != target_indices[target_partition_positions]" in e for e in errs)


def test_verify_fails_on_unexpected_manifest_key(monkeypatch):
    """10. Verify fails when one expected manifest key is replaced with an unexpected key."""
    import builtins
    import io
    real_open = builtins.open

    def fake_open(fpath, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if str(fpath).endswith("probe_manifest.json") and "r" in mode:
            with real_open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["probes"][0]["dataset_id"] = "unexpected_tampered_dataset_id"
            return io.StringIO(json.dumps(data))
        return real_open(fpath, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    errs = verify_phase5_integrity(PROJECT_ROOT)
    assert any("Manifest missing" in e or "Manifest has" in e for e in errs)

