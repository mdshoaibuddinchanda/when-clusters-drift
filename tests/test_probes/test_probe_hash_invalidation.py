"""Test probe bank hash invalidation on upstream changes."""

import numpy as np
from clusterdrift.probes.hashing import (
    compute_current_bank_sha256,
    compute_reference_bank_sha256,
)


def test_reference_bank_hash_invalidation():
    """Modifying bundle, split, positions, canonical rows, or protocol changes reference bank hash."""
    pos = np.arange(100, dtype=np.int64)
    can = np.arange(100, dtype=np.int64) * 2
    b_hash1 = compute_reference_bank_sha256("iris", 0, pos, can, "bundle_a", "split_a", "prep_a", "proto_a")
    b_hash2 = compute_reference_bank_sha256("iris", 0, pos, can, "bundle_b", "split_a", "prep_a", "proto_a")
    b_hash3 = compute_reference_bank_sha256("iris", 0, pos, can, "bundle_a", "split_b", "prep_a", "proto_a")
    b_hash4 = compute_reference_bank_sha256("iris", 0, pos, can, "bundle_a", "split_a", "prep_b", "proto_a")
    b_hash5 = compute_reference_bank_sha256("iris", 0, pos, can, "bundle_a", "split_a", "prep_a", "proto_b")
    b_hash6 = compute_reference_bank_sha256("iris", 0, pos + 1, can, "bundle_a", "split_a", "prep_a", "proto_a")
    b_hash7 = compute_reference_bank_sha256("iris", 0, pos, can + 1, "bundle_a", "split_a", "prep_a", "proto_a")

    all_hashes = [b_hash1, b_hash2, b_hash3, b_hash4, b_hash5, b_hash6, b_hash7]
    assert len(set(all_hashes)) == 7


def test_current_bank_hash_invalidation():
    """Modifying shift spec, shift protocol, or probe protocol changes current bank hash."""
    pos = np.arange(100, dtype=np.int64)
    tgt = np.arange(100, dtype=np.int64) * 2
    rows = np.arange(100, dtype=np.int64) * 3
    c_hash1 = compute_current_bank_sha256("iris", 0, "clean", pos, tgt, rows, "spec_a", "proto_s_a", "prep_a", "proto_p_a")
    c_hash2 = compute_current_bank_sha256("iris", 0, "clean", pos, tgt, rows, "spec_b", "proto_s_a", "prep_a", "proto_p_a")
    c_hash3 = compute_current_bank_sha256("iris", 0, "clean", pos, tgt, rows, "spec_a", "proto_s_b", "prep_a", "proto_p_a")
    c_hash4 = compute_current_bank_sha256("iris", 0, "clean", pos, tgt, rows, "spec_a", "proto_s_a", "prep_b", "proto_p_a")
    c_hash5 = compute_current_bank_sha256("iris", 0, "clean", pos, tgt, rows, "spec_a", "proto_s_a", "prep_a", "proto_p_b")
    c_hash6 = compute_current_bank_sha256("iris", 0, "clean", pos + 1, tgt, rows, "spec_a", "proto_s_a", "prep_a", "proto_p_a")
    c_hash7 = compute_current_bank_sha256("iris", 0, "clean", pos, tgt + 1, rows, "spec_a", "proto_s_a", "prep_a", "proto_p_a")
    c_hash8 = compute_current_bank_sha256("iris", 0, "clean", pos, tgt, rows + 1, "spec_a", "proto_s_a", "prep_a", "proto_p_a")

    all_hashes = [c_hash1, c_hash2, c_hash3, c_hash4, c_hash5, c_hash6, c_hash7, c_hash8]
    assert len(set(all_hashes)) == 8
