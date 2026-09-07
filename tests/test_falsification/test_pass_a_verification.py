"""Targeted unit tests for Phase 7 Pass A verification, provenance, and isolation."""

import io
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from clusterdrift.falsification.verification import (
    verify_input_lock,
    verify_pass_a,
    verify_pass_a_signals,
)


def test_verify_pass_a_clean():
    """Verify that verify_pass_a passes cleanly on authoritative repo artifacts."""
    root = Path(__file__).resolve().parents[2]
    res = verify_pass_a(root)
    assert res["status"] == "PASSED"
    assert res["pass_a_status"] == "FROZEN"
    assert res["signals"]["rows"] == 4500
    assert res["signals"]["sha256"] == "3a9e6c68cbe34c8763d70769715dba0d34976dfc35c8877c5680561da5c0e80a"


def test_verify_pass_a_byte_read_only():
    """Verify that verify_pass_a performs zero modifications to any files."""
    root = Path(__file__).resolve().parents[2]
    signals_p = root / "results" / "falsification" / "signals_label_free.csv"
    lock_p = root / "data" / "falsification" / "phase7_input_lock.json"

    mtime_sig_before = signals_p.stat().st_mtime_ns
    mtime_lock_before = lock_p.stat().st_mtime_ns

    res = verify_pass_a(root)
    assert res["status"] == "PASSED"

    assert signals_p.stat().st_mtime_ns == mtime_sig_before
    assert lock_p.stat().st_mtime_ns == mtime_lock_before


def test_verify_pass_a_reads_no_labels(monkeypatch):
    """Verify that verify_pass_a never reads or accesses labels.parquet."""
    root = Path(__file__).resolve().parents[2]

    orig_read_parquet = pd.read_parquet

    def guarded_read_parquet(path, *args, **kwargs):
        if "labels" in str(path).lower():
            raise PermissionError(f"CRITICAL: verify_pass_a must not access labels: {path}")
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read_parquet)
    res = verify_pass_a(root)
    assert res["status"] == "PASSED"


def test_pass_a_exact_4500_key_universe():
    """Verify exact 4,500 Cartesian product keys, zero missing, zero duplicates."""
    root = Path(__file__).resolve().parents[2]
    sig_res = verify_pass_a_signals(root)
    assert sig_res["rows"] == 4500
    assert sig_res["status"] == "VALID"


def test_wrong_phase6_freeze_sha_rejected(monkeypatch):
    """Verify that tampering with Phase 6 commit hashes in input lock is rejected."""
    root = Path(__file__).resolve().parents[2]
    lock_p = root / "data" / "falsification" / "phase7_input_lock.json"

    with open(lock_p, "r", encoding="utf-8") as f:
        lock = json.load(f)

    tampered_lock = dict(lock)
    tampered_lock["phase6_final_producer_commit"] = "0000000000000000000000000000000000000000"

    import builtins
    orig_open = builtins.open

    def mocked_open(file, *args, **kwargs):
        if str(file).endswith("phase7_input_lock.json"):
            return io.StringIO(json.dumps(tampered_lock))
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", mocked_open)
    with pytest.raises(ValueError, match="Phase 6 producer commit mismatch"):
        verify_input_lock(root)


def test_obsolete_phase6_freeze_commit_rejected(monkeypatch):
    """Verify that obsolete phase6_freeze_commit in input lock is rejected."""
    root = Path(__file__).resolve().parents[2]
    lock_p = root / "data" / "falsification" / "phase7_input_lock.json"

    with open(lock_p, "r", encoding="utf-8") as f:
        lock = json.load(f)

    tampered_lock = dict(lock)
    tampered_lock["phase6_freeze_commit"] = "a2726e6423405c93d9319eebbe0cbdfdfc7ecbb4"

    import builtins
    orig_open = builtins.open

    def mocked_open(file, *args, **kwargs):
        if str(file).endswith("phase7_input_lock.json"):
            return io.StringIO(json.dumps(tampered_lock))
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", mocked_open)
    with pytest.raises(ValueError, match="Obsolete 'phase6_freeze_commit' found"):
        verify_input_lock(root)


def test_phase6_input_lock_mutation_rejected(monkeypatch):
    """Verify that tampering with phase6_input_lock_sha256 is rejected."""
    root = Path(__file__).resolve().parents[2]
    lock_p = root / "data" / "falsification" / "phase7_input_lock.json"

    with open(lock_p, "r", encoding="utf-8") as f:
        lock = json.load(f)

    tampered_lock = dict(lock)
    tampered_lock["phase6_input_lock_sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

    import builtins
    orig_open = builtins.open

    def mocked_open(file, *args, **kwargs):
        if str(file).endswith("phase7_input_lock.json"):
            return io.StringIO(json.dumps(tampered_lock))
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", mocked_open)
    with pytest.raises(ValueError, match="SHA256 mismatch for phase6_input_lock_sha256"):
        verify_input_lock(root)
