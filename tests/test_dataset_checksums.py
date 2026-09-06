"""Test cryptographic hash calculation and mutation invalidation for canonical bundles."""

import hashlib
import json
import re
from pathlib import Path
import pytest

from clusterdrift.data.manifest import (
    compute_file_sha256,
    compute_canonical_bundle_sha256,
    compute_synthetic_bundle_sha256,
    compute_real_bundle_hashes,
    compute_synthetic_bundle_hashes,
    get_git_commit,
)


def _setup_mock_real_dataset(target_dir: Path, with_groups: bool = False, with_domains: bool = False):
    """Helper to populate a directory with mock canonical real dataset files."""
    (target_dir / "features.parquet").write_bytes(b"mock_features_parquet_data")
    (target_dir / "labels.parquet").write_bytes(b"mock_labels_parquet_data")
    (target_dir / "metadata.json").write_text(json.dumps({"name": "mock_dataset", "samples": 100}))
    if with_groups:
        (target_dir / "groups.parquet").write_bytes(b"mock_groups_parquet_data")
    if with_domains:
        (target_dir / "domains.parquet").write_bytes(b"mock_domains_parquet_data")


def _setup_mock_synthetic_dataset(target_dir: Path):
    """Helper to populate a directory with mock canonical synthetic dataset files."""
    (target_dir / "features.parquet").write_bytes(b"mock_synthetic_features")
    (target_dir / "hard_labels.parquet").write_bytes(b"mock_synthetic_hard_labels")
    (target_dir / "soft_memberships.npy").write_bytes(b"mock_synthetic_soft_memberships")
    (target_dir / "parameters.json").write_text(json.dumps({"drift_scenario": "sudden"}))
    (target_dir / "metadata.json").write_text(json.dumps({"family": "gaussian_mixture"}))


def test_checksum_mutation_detection(tmp_path):
    """Verify single-file SHA256 detection upon mutation."""
    test_file = tmp_path / "sample.parquet"
    test_file.write_bytes(b"initial_unmodified_bytes_dataset_payload")

    h1 = compute_file_sha256(test_file)
    assert isinstance(h1, str) and len(h1) == 64

    # Identical content yields identical hash
    h2 = compute_file_sha256(test_file)
    assert h1 == h2

    # Modified content alters hash
    test_file.write_bytes(b"mutated_corrupted_dataset_payload")
    h3 = compute_file_sha256(test_file)
    assert h1 != h3, "Hash did not change upon file modification."


def test_feature_mutation(tmp_path):
    """1. Modifying features.parquet alters canonical bundle SHA."""
    ds_dir = tmp_path / "real_ds"
    ds_dir.mkdir()
    _setup_mock_real_dataset(ds_dir)

    initial_hashes = compute_real_bundle_hashes(ds_dir)
    initial_bundle = initial_hashes["canonical_bundle_sha256"]
    assert initial_bundle is not None and len(initial_bundle) == 64

    # Mutate features.parquet
    (ds_dir / "features.parquet").write_bytes(b"mutated_feature_content")
    mutated_hashes = compute_real_bundle_hashes(ds_dir)

    assert mutated_hashes["features_sha256"] != initial_hashes["features_sha256"]
    assert mutated_hashes["canonical_bundle_sha256"] != initial_bundle


def test_label_mutation(tmp_path):
    """2. Modifying labels.parquet alters canonical bundle SHA."""
    ds_dir = tmp_path / "real_ds"
    ds_dir.mkdir()
    _setup_mock_real_dataset(ds_dir)

    initial_hashes = compute_real_bundle_hashes(ds_dir)
    initial_bundle = initial_hashes["canonical_bundle_sha256"]

    # Mutate labels.parquet
    (ds_dir / "labels.parquet").write_bytes(b"mutated_labels_content")
    mutated_hashes = compute_real_bundle_hashes(ds_dir)

    assert mutated_hashes["labels_sha256"] != initial_hashes["labels_sha256"]
    assert mutated_hashes["canonical_bundle_sha256"] != initial_bundle


def test_group_mutation(tmp_path):
    """3. Modifying groups.parquet alters canonical bundle SHA."""
    ds_dir = tmp_path / "grouped_ds"
    ds_dir.mkdir()
    _setup_mock_real_dataset(ds_dir, with_groups=True)

    initial_hashes = compute_real_bundle_hashes(ds_dir)
    initial_bundle = initial_hashes["canonical_bundle_sha256"]
    assert initial_hashes["groups_sha256"] is not None

    # Mutate groups.parquet
    (ds_dir / "groups.parquet").write_bytes(b"mutated_group_clustering_assignments")
    mutated_hashes = compute_real_bundle_hashes(ds_dir)

    assert mutated_hashes["groups_sha256"] != initial_hashes["groups_sha256"]
    assert mutated_hashes["canonical_bundle_sha256"] != initial_bundle


def test_domain_mutation(tmp_path):
    """4. Modifying domains.parquet alters canonical bundle SHA."""
    ds_dir = tmp_path / "domain_ds"
    ds_dir.mkdir()
    _setup_mock_real_dataset(ds_dir, with_domains=True)

    initial_hashes = compute_real_bundle_hashes(ds_dir)
    initial_bundle = initial_hashes["canonical_bundle_sha256"]
    assert initial_hashes["domains_sha256"] is not None

    # Mutate domains.parquet
    (ds_dir / "domains.parquet").write_bytes(b"mutated_shift_domain_assignments")
    mutated_hashes = compute_real_bundle_hashes(ds_dir)

    assert mutated_hashes["domains_sha256"] != initial_hashes["domains_sha256"]
    assert mutated_hashes["canonical_bundle_sha256"] != initial_bundle


def test_metadata_mutation(tmp_path):
    """5. Modifying metadata.json alters canonical bundle SHA."""
    ds_dir = tmp_path / "real_ds"
    ds_dir.mkdir()
    _setup_mock_real_dataset(ds_dir)

    initial_hashes = compute_real_bundle_hashes(ds_dir)
    initial_bundle = initial_hashes["canonical_bundle_sha256"]

    # Mutate metadata.json
    (ds_dir / "metadata.json").write_text(json.dumps({"name": "mock_dataset", "samples": 999}))
    mutated_hashes = compute_real_bundle_hashes(ds_dir)

    assert mutated_hashes["metadata_sha256"] != initial_hashes["metadata_sha256"]
    assert mutated_hashes["canonical_bundle_sha256"] != initial_bundle


def test_optional_artifacts():
    """6. Verify dataset without groups/domains generates deterministic bundle hash ('groups:NONE', 'domains:NONE')."""
    f_sha = "1" * 64
    l_sha = "2" * 64
    m_sha = "5" * 64

    # Direct bundle calculation with None for optional artifacts
    bundle_hash = compute_canonical_bundle_sha256(
        features_sha256=f_sha,
        labels_sha256=l_sha,
        groups_sha256=None,
        domains_sha256=None,
        metadata_sha256=m_sha,
    )

    # Expected raw payload
    expected_payload = (
        f"features:{f_sha}\n"
        f"labels:{l_sha}\n"
        f"groups:NONE\n"
        f"domains:NONE\n"
        f"metadata:{m_sha}"
    )
    expected_sha = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()

    assert bundle_hash == expected_sha
    assert len(bundle_hash) == 64

    # Determinism: multiple calls yield exact same result
    assert compute_canonical_bundle_sha256(f_sha, l_sha, None, None, m_sha) == bundle_hash


def test_synthetic_soft_truth_mutation(tmp_path):
    """7. Modifying soft_memberships.npy alters synthetic canonical bundle SHA."""
    sdir = tmp_path / "synthetic_ds"
    sdir.mkdir()
    _setup_mock_synthetic_dataset(sdir)

    initial_hashes = compute_synthetic_bundle_hashes(sdir)
    initial_bundle = initial_hashes["canonical_bundle_sha256"]
    assert initial_hashes["soft_memberships_sha256"] is not None

    # Mutate soft_memberships.npy
    (sdir / "soft_memberships.npy").write_bytes(b"mutated_ground_truth_soft_memberships")
    mutated_hashes = compute_synthetic_bundle_hashes(sdir)

    assert mutated_hashes["soft_memberships_sha256"] != initial_hashes["soft_memberships_sha256"]
    assert mutated_hashes["canonical_bundle_sha256"] != initial_bundle


def test_provenance_semantics():
    """8. Verify generated_from_commit is a 40-character hexadecimal SHA."""
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) == 40
    assert re.match(r"^[0-9a-fA-F]{40}$", commit) is not None
