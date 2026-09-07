"""Unit tests for reference probe bank selection."""

import numpy as np
import pytest

from clusterdrift.probes.selection import select_reference_probe_indices


def test_reference_probe_selection_sub_2048():
    """When N_source <= 2048, all source rows must be selected in identity order."""
    N = 150
    indices = select_reference_probe_indices(n_source=N, max_size=2048, seed=42)
    assert len(indices) == N
    np.testing.assert_array_equal(indices, np.arange(N))


def test_reference_probe_selection_over_2048():
    """When N_source > 2048, exactly 2048 distinct sorted rows must be selected."""
    N = 10000
    indices = select_reference_probe_indices(n_source=N, max_size=2048, seed=123)
    assert len(indices) == 2048
    # All indices in range [0..N-1]
    assert np.all(indices >= 0)
    assert np.all(indices < N)
    # Distinct (no duplicates)
    assert len(np.unique(indices)) == 2048
    # Sorted
    assert np.all(np.diff(indices) > 0)
