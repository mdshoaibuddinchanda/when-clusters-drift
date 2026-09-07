"""Unit tests verifying Phase 1 and Phase 2 immutability under Phase 4 execution."""

import subprocess
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_phase1_phase2_frozen_data_untouched():
    """Verify git status shows zero modifications to data/canonical, data/synthetic, or data/splits."""
    cmd = ["git", "status", "--porcelain", "data/canonical", "data/synthetic", "data/splits"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)
    assert res.returncode == 0
    assert res.stdout.strip() == "", f"Found forbidden modifications to frozen data partitions:\n{res.stdout}"
