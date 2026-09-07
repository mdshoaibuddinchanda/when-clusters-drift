"""Test explicit probe provenance chains and duplicate preservation."""

from pathlib import Path
import numpy as np
import pytest

from clusterdrift.probes.bank import (
    load_current_probe_descriptor,
    load_reference_probe_descriptor,
    save_current_probe_descriptor,
    save_reference_probe_descriptor,
)
from clusterdrift.probes.selection import (
    select_current_probe_positions,
    select_reference_probe_indices,
)


def test_reference_probe_provenance_chain(tmp_path: Path):
    """Reference probe NPZ stores explicit selected_source_positions and canonical_row_indices."""
    n_source = 100
    phase2_source_indices = np.arange(500, 600, dtype=np.int64)

    positions, canonical_rows = select_reference_probe_indices(
        n_source=n_source,
        max_size=50,
        seed=101,
        phase2_source_indices=phase2_source_indices,
    )

    assert len(positions) == 50
    assert len(canonical_rows) == 50
    np.testing.assert_array_equal(canonical_rows, phase2_source_indices[positions])

    spec_path = tmp_path / "reference" / "mock_ds" / "fold_0.json"
    metadata = {
        "dataset_id": "mock_ds",
        "outer_fold": 0,
        "canonical_bundle_sha256": "bundle_sha",
        "split_sha256": "split_sha",
        "preprocessing_config_sha256": "prep_sha",
        "selection_policy": "uniform_without_replacement",
        "global_probe_seed": 2026090705,
        "derived_seed": 101,
        "available_rows": n_source,
        "probe_protocol_sha256": "proto_sha",
    }

    desc = save_reference_probe_descriptor(
        spec_path=spec_path,
        metadata=metadata,
        selected_source_positions=positions,
        canonical_row_indices=canonical_rows,
    )

    # Inspect companion NPZ directly
    npz_path = spec_path.with_suffix(".npz")
    assert npz_path.exists()
    with np.load(npz_path) as npz:
        assert "selected_source_positions" in npz
        assert "canonical_row_indices" in npz
        np.testing.assert_array_equal(npz["selected_source_positions"], positions)
        np.testing.assert_array_equal(npz["canonical_row_indices"], canonical_rows)

    # Load via descriptor loader
    loaded_desc, loaded_pos, loaded_can = load_reference_probe_descriptor(spec_path)
    assert loaded_desc.selected_rows == 50
    np.testing.assert_array_equal(loaded_pos, positions)
    np.testing.assert_array_equal(loaded_can, canonical_rows)


def test_current_probe_provenance_chain_and_duplicates(tmp_path: Path):
    """Current probe stores 3-tier chain and preserves bootstrap duplicates across all levels."""
    # 20 duplicated rows in target partition
    n_target = 60
    row_map = np.array([5] * 20 + list(range(10, 50)), dtype=np.int64)
    phase2_target_indices = np.arange(1000, 1100, dtype=np.int64)

    positions, target_positions, canonical_rows = select_current_probe_positions(
        n_target=n_target,
        row_index_map=row_map,
        max_size=30,
        seed=202,
        phase2_target_indices=phase2_target_indices,
    )

    assert len(positions) == 30
    assert len(target_positions) == 30
    assert len(canonical_rows) == 30

    # Verify chain: positions -> target_positions -> canonical_rows
    np.testing.assert_array_equal(target_positions, row_map[positions])
    np.testing.assert_array_equal(canonical_rows, phase2_target_indices[target_positions])

    # In class prevalence bootstrap, duplicates are strictly preserved without deduplication
    count_5_in_target = np.sum(target_positions == 5)
    assert count_5_in_target > 1  # Multiple occurrences of target partition position 5
    expected_can_val = phase2_target_indices[5]
    count_can = np.sum(canonical_rows == expected_can_val)
    assert count_can == count_5_in_target

    spec_path = tmp_path / "current" / "mock_ds" / "fold_0" / "class_prevalence_severe.json"
    metadata = {
        "dataset_id": "mock_ds",
        "outer_fold": 0,
        "condition": "class_prevalence_severe",
        "shift_spec_sha256": "spec_sha",
        "shift_protocol_sha256": "shift_proto_sha",
        "preprocessing_config_sha256": "prep_sha",
        "selection_policy": "uniform_without_replacement",
        "global_probe_seed": 2026090705,
        "derived_seed": 202,
        "available_current_rows": n_target,
        "scenario_status": "APPLICABLE",
        "probe_protocol_sha256": "probe_proto_sha",
    }

    desc = save_current_probe_descriptor(
        spec_path=spec_path,
        metadata=metadata,
        selected_current_positions=positions,
        target_partition_positions=target_positions,
        canonical_row_indices=canonical_rows,
    )

    # Inspect companion NPZ directly
    npz_path = spec_path.with_suffix(".npz")
    assert npz_path.exists()
    with np.load(npz_path) as npz:
        assert "selected_current_positions" in npz
        assert "target_partition_positions" in npz
        assert "canonical_row_indices" in npz
        np.testing.assert_array_equal(npz["selected_current_positions"], positions)
        np.testing.assert_array_equal(npz["target_partition_positions"], target_positions)
        np.testing.assert_array_equal(npz["canonical_row_indices"], canonical_rows)

    # Load via descriptor loader
    loaded_desc, l_pos, l_tgt, l_can = load_current_probe_descriptor(spec_path)
    assert loaded_desc.selected_current_rows == 30
    np.testing.assert_array_equal(l_pos, positions)
    np.testing.assert_array_equal(l_tgt, target_positions)
    np.testing.assert_array_equal(l_can, canonical_rows)
