"""Unit tests verifying Phase 4 shift scenario grid completeness."""

from pathlib import Path
import json
import pytest

from clusterdrift.shifts import ALL_CONDITIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_controlled_dataset_grid_dimensions():
    """Verify nominal grid dimensions: 30 controlled datasets x 5 folds x 15 conditions = 2250."""
    splits_manifest = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"
    assert splits_manifest.exists()
    with open(splits_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    controlled_splits = [s for s in manifest.get("splits", []) if "controlled" in s.get("npz_path", "")]
    datasets = sorted(list(set(s["dataset_id"] for s in controlled_splits)))

    assert len(datasets) == 30, f"Expected 30 controlled datasets, found {len(datasets)}"
    assert len(ALL_CONDITIONS) == 15, f"Expected 15 conditions, found {len(ALL_CONDITIONS)}"

    total_nominal = len(datasets) * 5 * len(ALL_CONDITIONS)
    assert total_nominal == 2250, f"Expected 2250 nominal scenarios, got {total_nominal}"
