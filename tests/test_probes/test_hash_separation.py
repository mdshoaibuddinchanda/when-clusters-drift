"""Test separation between array byte hashes and companion NPZ file byte hash."""

from pathlib import Path
import numpy as np
import pytest

from clusterdrift.probes.bank import save_reference_probe_descriptor
from clusterdrift.probes.hashing import (
    compute_array_sha256,
    compute_file_sha256,
)


def test_array_hash_vs_npz_hash_separation(tmp_path: Path):
    """Array SHA-256 is computed directly on array buffer, not the NPZ zip container."""
    arr1 = np.arange(100, dtype=np.int64)
    arr2 = np.arange(100, dtype=np.int64) * 2

    hash1 = compute_array_sha256(arr1)
    hash2 = compute_array_sha256(arr2)
    assert hash1 != hash2
    assert len(hash1) == 64
    assert len(hash2) == 64

    # Saving reference probe
    spec_path = tmp_path / "reference" / "iris" / "fold_0.json"
    metadata = {
        "dataset_id": "iris",
        "outer_fold": 0,
        "canonical_bundle_sha256": "bundle_sha",
        "split_sha256": "split_sha",
        "preprocessing_config_sha256": "prep_sha",
        "selection_policy": "uniform_without_replacement",
        "global_probe_seed": 2026090705,
        "derived_seed": 12345,
        "available_rows": 120,
        "probe_protocol_sha256": "proto_sha",
    }

    desc = save_reference_probe_descriptor(
        spec_path=spec_path,
        metadata=metadata,
        selected_source_positions=arr1,
        canonical_row_indices=arr2,
    )

    npz_path = spec_path.with_suffix(".npz")
    actual_npz_file_sha = compute_file_sha256(npz_path)

    # NPZ file hash is strictly different from raw array hash
    assert desc.npz_sha256 == actual_npz_file_sha
    assert desc.selected_source_positions_sha256 == hash1
    assert desc.canonical_row_indices_sha256 == hash2
    assert desc.selected_source_positions_sha256 != desc.npz_sha256
    assert desc.canonical_row_indices_sha256 != desc.npz_sha256
