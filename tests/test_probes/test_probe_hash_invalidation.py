"""Test probe bank hash invalidation on upstream changes."""

import numpy as np
from clusterdrift.probes.hashing import (
    compute_current_bank_sha256,
    compute_reference_bank_sha256,
)


def test_reference_bank_hash_invalidation():
    """Modifying bundle, split, or protocol changes reference bank hash."""
    indices = np.arange(100)
    b_hash1 = compute_reference_bank_sha256("iris", 0, indices, "bundle_a", "split_a", "prep_a", "proto_a")
    b_hash2 = compute_reference_bank_sha256("iris", 0, indices, "bundle_b", "split_a", "prep_a", "proto_a")
    b_hash3 = compute_reference_bank_sha256("iris", 0, indices, "bundle_a", "split_b", "prep_a", "proto_a")
    b_hash4 = compute_reference_bank_sha256("iris", 0, indices, "bundle_a", "split_a", "prep_b", "proto_a")
    b_hash5 = compute_reference_bank_sha256("iris", 0, indices, "bundle_a", "split_a", "prep_a", "proto_b")

    all_hashes = [b_hash1, b_hash2, b_hash3, b_hash4, b_hash5]
    assert len(set(all_hashes)) == 5


def test_current_bank_hash_invalidation():
    """Modifying shift spec, shift protocol, or probe protocol changes current bank hash."""
    pos = np.arange(100)
    rows = np.arange(100) * 3
    c_hash1 = compute_current_bank_sha256("iris", 0, "clean", pos, rows, "spec_a", "proto_s_a", "prep_a", "proto_p_a")
    c_hash2 = compute_current_bank_sha256("iris", 0, "clean", pos, rows, "spec_b", "proto_s_a", "prep_a", "proto_p_a")
    c_hash3 = compute_current_bank_sha256("iris", 0, "clean", pos, rows, "spec_a", "proto_s_b", "prep_a", "proto_p_a")
    c_hash4 = compute_current_bank_sha256("iris", 0, "clean", pos, rows, "spec_a", "proto_s_a", "prep_a", "proto_p_b")

    all_hashes = [c_hash1, c_hash2, c_hash3, c_hash4]
    assert len(set(all_hashes)) == 4
