"""Test cryptographic hash calculation and mutation invalidation."""

from pathlib import Path
from clusterdrift.data.manifest import compute_file_sha256


def test_checksum_mutation_detection(tmp_path):
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
