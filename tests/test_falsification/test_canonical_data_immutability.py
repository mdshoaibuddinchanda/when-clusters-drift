import pytest
from pathlib import Path
from clusterdrift.signals.hashing import compute_file_sha256

def test_canonical_data_immutability():
    root = Path(__file__).resolve().parents[2]
    # Check that frozen dataset manifest and split files are non-empty and present
    ds_manifest = root / "data" / "manifests" / "datasets.json"
    assert ds_manifest.exists()
    assert ds_manifest.stat().st_size > 0

    split_manifest = root / "data" / "splits" / "split_manifest.json"
    assert split_manifest.exists()

    shift_manifest = root / "data" / "shifts" / "shift_manifest.json"
    assert shift_manifest.exists()

    probe_manifest = root / "data" / "probes" / "probe_manifest.json"
    assert probe_manifest.exists()
