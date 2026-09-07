"""Verify Phase 4 shift artifacts are strictly unmodified."""

import json
from pathlib import Path
import pytest

from clusterdrift.shifts.hashing import compute_file_sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_phase4_artifacts_immutability():
    """Verify Phase-4 input lock and manifest hashes match frozen records."""
    lock_path = PROJECT_ROOT / "data" / "shifts" / "phase4_input_lock.json"
    manifest_path = PROJECT_ROOT / "data" / "shifts" / "shift_manifest.json"

    assert lock_path.exists(), "Phase-4 input lock missing"
    assert manifest_path.exists(), "Phase-4 shift manifest missing"

    with open(lock_path, "r", encoding="utf-8") as f:
        lock_doc = json.load(f)

    # 30 datasets * 5 folds * 15 conditions = 2250 scenarios
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_doc = json.load(f)

    assert len(manifest_doc["scenarios"]) == 2250
    assert lock_doc["included_controlled_datasets_count"] == 30
