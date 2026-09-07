"""Test that bootstrap duplicates in class_prevalence shifts are strictly preserved in current probes."""

import numpy as np
from clusterdrift.probes.selection import select_current_probe_positions


def test_class_prevalence_bootstrap_duplicates_preserved():
    """Verify that duplicate rows in row_index_map are preserved in canonical row indices."""
    # Simulate a bootstrap resampled target where row 42 appears multiple times
    n_target = 100
    row_map = np.array([42] * 20 + list(range(100, 180)), dtype=np.int64)

    positions, canonical_rows = select_current_probe_positions(
        n_target=n_target,
        row_index_map=row_map,
        max_size=2048,
        seed=12345,
    )

    # Length matches
    assert len(positions) == n_target
    assert len(canonical_rows) == n_target

    # Duplicates are preserved
    count_42 = np.sum(canonical_rows == 42)
    assert count_42 == 20
    assert len(np.unique(canonical_rows)) < len(canonical_rows)
