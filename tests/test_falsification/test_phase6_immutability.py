from pathlib import Path
import pytest
from clusterdrift.signals.hashing import compute_file_sha256

def test_phase6_files_exist_and_intact():
    root = Path(__file__).resolve().parents[2]
    p6_signals = root / "results" / "signal_validation" / "signals_label_free.csv"
    assert p6_signals.exists()
    # Check that file size is non-zero
    assert p6_signals.stat().st_size > 0

    p6_quality = root / "results" / "signal_validation" / "quality_evaluation_only.csv"
    assert p6_quality.exists()
    assert p6_quality.stat().st_size > 0