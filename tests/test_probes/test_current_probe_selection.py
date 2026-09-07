"""Unit tests for current probe bank selection."""

import numpy as np
import pytest

from clusterdrift.probes.selection import select_current_probe_positions


def test_current_probe_selection_sub_2048():
    """When N_target <= 2048, all target positions must be selected."""
    N = 300
    row_map = np.arange(1000, 1000 + N, dtype=np.int64)
    positions, canonical_rows = select_current_probe_positions(
        n_target=N, row_index_map=row_map, max_size=2048, seed=42
    )
    assert len(positions) == N
    assert len(canonical_rows) == N
    np.testing.assert_array_equal(positions, np.arange(N))
    np.testing.assert_array_equal(canonical_rows, row_map)


def test_current_probe_selection_over_2048():
    """When N_target > 2048, exactly 2048 positions are sampled without replacement."""
    N = 5000
    row_map = np.arange(N, dtype=np.int64) * 2
    positions, canonical_rows = select_current_probe_positions(
        n_target=N, row_index_map=row_map, max_size=2048, seed=999
    )
    assert len(positions) == 2048
    assert len(canonical_rows) == 2048
    assert len(np.unique(positions)) == 2048
    np.testing.assert_array_equal(canonical_rows, row_map[positions])
