"""Test probe bank size boundary conditions."""

import pytest
from clusterdrift.probes.selection import select_reference_probe_indices


@pytest.mark.parametrize("N", [10, 100, 2047, 2048, 2049, 5000])
def test_probe_bank_size_bound(N):
    """Probe bank size is strictly min(N, 2048)."""
    indices = select_reference_probe_indices(N, max_size=2048, seed=42)
    expected_size = min(N, 2048)
    assert len(indices) == expected_size
    assert len(set(indices)) == expected_size
