"""Unit tests for cryptographic shift hashing and invalidation."""

from pathlib import Path
import pytest

from clusterdrift.shifts.hashing import (
    compute_scenario_input_sha256,
    compute_shift_protocol_sha256,
    compute_shift_spec_sha256,
    derive_integer_seed,
)


def test_hashing_invalidation():
    """Verify changes to inputs or protocol invalidate cryptographic hashes."""
    proto_hash1 = compute_shift_protocol_sha256({"v": 1, "seed": 42})
    proto_hash2 = compute_shift_protocol_sha256({"v": 1, "seed": 43})
    assert proto_hash1 != proto_hash2

    scen_hash1 = compute_scenario_input_sha256("bundle_a", "split_0", "iris", 0)
    scen_hash2 = compute_scenario_input_sha256("bundle_b", "split_0", "iris", 0)
    assert scen_hash1 != scen_hash2

    spec_hash1 = compute_shift_spec_sha256(scen_hash1, proto_hash1, "clean", "clean", "none", 100, {"m": 1})
    spec_hash2 = compute_shift_spec_sha256(scen_hash1, proto_hash1, "clean", "clean", "none", 101, {"m": 1})
    assert spec_hash1 != spec_hash2
