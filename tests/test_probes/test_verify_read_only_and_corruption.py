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
