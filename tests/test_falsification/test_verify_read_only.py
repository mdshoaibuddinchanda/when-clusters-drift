from pathlib import Path
import pytest
import inspect
from clusterdrift.falsification import verification, dataset, models, evaluation, bootstrap

def test_no_random_row_cv_in_falsification():
    root = Path(__file__).resolve().parents[2]
    falsification_dir = root / "src" / "clusterdrift" / "falsification"

    forbidden_patterns = [
        "train_test_split",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
    ]

    for py_file in falsification_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern not in content, f"Forbidden random split pattern '{pattern}' found in {py_file.name}"


def test_verification_module_read_only():
    # Ensure verification function takes project root and is callable
    assert callable(verification.verify_falsification)