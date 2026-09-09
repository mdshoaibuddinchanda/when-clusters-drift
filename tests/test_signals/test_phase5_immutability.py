"""Immutability verification tests for Phase 1-5 frozen assets."""

from pathlib import Path
import subprocess
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FROZEN_PATHS = [
    "data/canonical/",
    "data/synthetic/",
    "data/splits/",
    "data/shifts/",
    "data/probes/",
    "results/baseline_validation/",
    "results/shift_validation/",
    "results/alignment_validation/",
    "configs/preprocessing.yaml",
    "configs/shifts.yaml",
    "configs/probes.yaml",
    "configs/alignment.yaml",
    "configs/methods.yaml",
]


def test_frozen_paths_unmodified():
    """Verify that no frozen directories or config files have git modifications."""
    if not (PROJECT_ROOT / ".git").exists():
        pytest.skip("Not inside a git worktree")
    cmd = ["git", "status", "--porcelain"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    lines = res.stdout.strip().splitlines()

    for line in lines:
        status = line[:2]
        path_str = line[3:].replace("\\", "/")
        for frozen in FROZEN_PATHS:
            assert not path_str.startswith(frozen), (
                f"Frozen Phase 1-5 asset modified: {path_str} (status: {status})"
            )
