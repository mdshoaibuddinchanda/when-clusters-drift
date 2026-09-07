"""Test staleness and corruption detection for probe descriptors."""

from pathlib import Path
import numpy as np
import pytest

from clusterdrift.probes.bank import (
    ReferenceProbeDescriptor,
    save_reference_probe_descriptor,
)
from clusterdrift.probes.hashing import (
    compute_file_sha256,
    compute_reference_bank_sha256,
    derive_reference_probe_seed,
)
from clusterdrift.probes.validation import validate_saved_reference_descriptor


def test_reference_descriptor_staleness_detection(tmp_path):
    """Corrupting NPZ or altering upstream hash invalidates descriptor."""
    dataset_id = "iris"
    fold = 0
    bundle_sha = "bundle_hash_1"
    split_sha = "split_hash_1"
    prep_sha = "prep_hash_1"
    proto_sha = "proto_hash_1"
    global_seed = 2026090705

    derived_seed = derive_reference_probe_seed(global_seed, dataset_id, fold, split_sha)
    indices = np.arange(100, dtype=np.int64)

    spec_path = tmp_path / "reference" / dataset_id / f"fold_{fold}.json"
    npz_path = spec_path.with_suffix(".npz")

    metadata = {
        "dataset_id": dataset_id,
        "outer_fold": fold,
        "canonical_bundle_sha256": bundle_sha,
        "split_sha256": split_sha,
        "preprocessing_config_sha256": prep_sha,
        "selection_policy": "uniform_without_replacement",
        "global_probe_seed": global_seed,
        "derived_seed": derived_seed,
        "available_rows": 100,
        "probe_protocol_sha256": proto_sha,
    }
    desc = save_reference_probe_descriptor(spec_path, metadata, indices, indices)

    # Validate passes
    is_val, err, _ = validate_saved_reference_descriptor(
        spec_path, bundle_sha, split_sha, prep_sha, proto_sha, global_seed, dataset_id, fold
    )
    assert is_val is True
    assert err is None

    # Corrupt NPZ file
    with open(npz_path, "wb") as f:
        f.write(b"corrupted_bytes")

    is_val_corrupt, err_corrupt, _ = validate_saved_reference_descriptor(
        spec_path, bundle_sha, split_sha, prep_sha, proto_sha, global_seed, dataset_id, fold
    )
    assert is_val_corrupt is False

    # Upstream split hash changed
    is_val_split, err_split, _ = validate_saved_reference_descriptor(
        spec_path, bundle_sha, "new_split_hash", prep_sha, proto_sha, global_seed, dataset_id, fold
    )
    assert is_val_split is False
