"""Test determinism of probe selection across processes and calls."""

import numpy as np
from clusterdrift.probes.hashing import derive_current_probe_seed, derive_reference_probe_seed
from clusterdrift.probes.selection import select_current_probe_positions, select_reference_probe_indices


def test_reference_selection_determinism():
    """Derived seed guarantees identical selection on repeated calls."""
    seed1 = derive_reference_probe_seed(2026090705, "iris", 0, "fake_split_sha")
    seed2 = derive_reference_probe_seed(2026090705, "iris", 0, "fake_split_sha")
    assert seed1 == seed2

    idx1 = select_reference_probe_indices(5000, 2048, seed1)
    idx2 = select_reference_probe_indices(5000, 2048, seed2)
    np.testing.assert_array_equal(idx1, idx2)


def test_current_selection_determinism():
    """Current probe selection is deterministic given identical payload."""
    seed1 = derive_current_probe_seed(2026090705, "sonar", 1, "location_mild", "fake_spec_sha")
    seed2 = derive_current_probe_seed(2026090705, "sonar", 1, "location_mild", "fake_spec_sha")
    assert seed1 == seed2

    row_map = np.arange(4000)
    pos1, rows1 = select_current_probe_positions(4000, row_map, 2048, seed1)
    pos2, rows2 = select_current_probe_positions(4000, row_map, 2048, seed2)
    np.testing.assert_array_equal(pos1, pos2)
    np.testing.assert_array_equal(rows1, rows2)
