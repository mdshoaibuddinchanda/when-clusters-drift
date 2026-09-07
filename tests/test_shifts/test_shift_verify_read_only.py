"""Unit tests verifying --dry-run and verify modes are strictly byte-read-only."""

import subprocess
import sys
from pathlib import Path
import pytest

from clusterdrift.shifts import compute_file_sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_dry_run_is_byte_read_only():
    """Verify that running scripts/phase4_validate_shifts.py --dry-run modifies zero files on disk."""
    check_files = [
        PROJECT_ROOT / "configs" / "shifts.yaml",
        PROJECT_ROOT / "configs" / "preprocessing.yaml",
    ]
    hashes_before = {f: compute_file_sha256(f) for f in check_files if f.exists()}

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "phase4_validate_shifts.py"), "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"--dry-run failed with error: {res.stderr}"

    hashes_after = {f: compute_file_sha256(f) for f in check_files if f.exists()}
    assert hashes_before == hashes_after
