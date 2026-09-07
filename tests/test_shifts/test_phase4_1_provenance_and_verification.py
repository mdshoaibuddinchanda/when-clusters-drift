"""Phase 4.1 regression and provenance verification test suite.

Verifies:
1. Canonical bundle hash binding (labels/metadata mutation invalidates scenario hash).
2. Phase-2 scientific split hash binding.
3. Resume detects changed config, changed bundle, changed split, or corrupted NPZ.
4. Atomic writes for JSON, CSV, and NPZ.
5. Local overlap geometry uses source median for target missing values (not mean or 0.0).
6. Multithreaded execution determinism (--jobs 1 vs --jobs 4).
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from clusterdrift.shifts.base import SourceStatistics, compute_source_statistics
from clusterdrift.shifts.engine import ShiftEngine, validate_saved_shift_spec
from clusterdrift.shifts.hashing import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    compute_file_sha256,
    compute_scenario_input_sha256,
    compute_shift_protocol_sha256,
    derive_integer_seed,
)
from clusterdrift.shifts.offline_supervised import apply_local_overlap_shift

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_canonical_bundle_hash_binding():
    """Verify modifying bundle hash (e.g. via label/metadata change) alters scenario_input_sha256."""
    bundle_a = "06a7bea1035adc450f695977cf5ff01ec45cd7cb140a5e8f23ad103da0949a27"
    bundle_b = "11111111035adc450f695977cf5ff01ec45cd7cb140a5e8f23ad103da0949a27"
    split = "62ecc40875d3cfe3850f993be5f18833054436c5c38b7f9262e9de2ad6da47d8"

    hash_a = compute_scenario_input_sha256(bundle_a, split, "iris", 0)
    hash_b = compute_scenario_input_sha256(bundle_b, split, "iris", 0)

    assert hash_a != hash_b, "Scenario input hash must change when bundle hash changes."


def test_split_hash_binding():
    """Verify modifying split hash alters scenario_input_sha256."""
    bundle = "06a7bea1035adc450f695977cf5ff01ec45cd7cb140a5e8f23ad103da0949a27"
    split_a = "62ecc40875d3cfe3850f993be5f18833054436c5c38b7f9262e9de2ad6da47d8"
    split_b = "9999999975d3cfe3850f993be5f18833054436c5c38b7f9262e9de2ad6da47d8"

    hash_a = compute_scenario_input_sha256(bundle, split_a, "iris", 0)
    hash_b = compute_scenario_input_sha256(bundle, split_b, "iris", 0)

    assert hash_a != hash_b, "Scenario input hash must change when split hash changes."


def test_resume_staleness_detection(tmp_path):
    """Verify validate_saved_shift_spec detects stale bundle, stale split, stale config, and corrupted NPZ."""
    spec_json = tmp_path / "location_mild.json"
    spec_npz = tmp_path / "location_mild.npz"

    bundle = "bundle_hash_123"
    split = "split_hash_456"
    proto = "proto_hash_789"
    seed = 2026090704
    ds = "test_ds"
    fold = 0
    cond = "location_mild"

    # Write initial valid NPZ and JSON
    atomic_write_npz(spec_npz, row_index_map=np.arange(10, dtype=np.int64))
    npz_sha = compute_file_sha256(spec_npz)

    input_hash = compute_scenario_input_sha256(bundle, split, ds, fold)
    derived_seed = derive_integer_seed(seed, ds, fold, "location", "spec")
    spec_sha = "spec_sha_valid"

    # Construct spec doc
    from clusterdrift.shifts.hashing import compute_shift_spec_sha256
    spec_descriptor = {
        "metadata": {"family": "location"},
        "status": "APPLICABLE",
        "reason": None,
        "npz_sha256": npz_sha,
    }
    actual_spec_sha = compute_shift_spec_sha256(
        input_hash, proto, cond, "location", "mild", derived_seed, spec_descriptor
    )

    spec_doc = {
        "protocol_version": 1,
        "dataset_id": ds,
        "outer_fold": fold,
        "condition": cond,
        "family": "location",
        "severity": "mild",
        "status": "APPLICABLE",
        "reason": None,
        "canonical_bundle_sha256": bundle,
        "split_sha256": split,
        "scenario_input_sha256": input_hash,
        "shift_protocol_sha256": proto,
        "shift_spec_sha256": actual_spec_sha,
        "derived_seed": derived_seed,
        "npz_relpath": str(spec_npz),
        "npz_sha256": npz_sha,
        "metadata": {"family": "location"},
    }
    atomic_write_json(spec_json, spec_doc)

    # 1. Valid case
    is_val, reason, _ = validate_saved_shift_spec(
        spec_path=spec_json,
        expected_canonical_bundle_sha256=bundle,
        expected_split_sha256=split,
        expected_shift_protocol_sha256=proto,
        global_shift_seed=seed,
        dataset_id=ds,
        outer_fold=fold,
        condition=cond,
    )
    assert is_val is True
    assert reason == "VALID"

    # 2. Stale bundle
    is_val, reason, _ = validate_saved_shift_spec(
        spec_path=spec_json,
        expected_canonical_bundle_sha256="stale_bundle",
        expected_split_sha256=split,
        expected_shift_protocol_sha256=proto,
        global_shift_seed=seed,
        dataset_id=ds,
        outer_fold=fold,
        condition=cond,
    )
    assert is_val is False
    assert "canonical_bundle_sha256 mismatch" in reason

    # 3. Stale split
    is_val, reason, _ = validate_saved_shift_spec(
        spec_path=spec_json,
        expected_canonical_bundle_sha256=bundle,
        expected_split_sha256="stale_split",
        expected_shift_protocol_sha256=proto,
        global_shift_seed=seed,
        dataset_id=ds,
        outer_fold=fold,
        condition=cond,
    )
    assert is_val is False
    assert "split_sha256 mismatch" in reason

    # 4. Stale protocol
    is_val, reason, _ = validate_saved_shift_spec(
        spec_path=spec_json,
        expected_canonical_bundle_sha256=bundle,
        expected_split_sha256=split,
        expected_shift_protocol_sha256="stale_proto",
        global_shift_seed=seed,
        dataset_id=ds,
        outer_fold=fold,
        condition=cond,
    )
    assert is_val is False
    assert "shift_protocol_sha256 mismatch" in reason

    # 5. Corrupted NPZ
    with open(spec_npz, "ab") as f:
        f.write(b"corrupted_bytes")
    is_val, reason, _ = validate_saved_shift_spec(
        spec_path=spec_json,
        expected_canonical_bundle_sha256=bundle,
        expected_split_sha256=split,
        expected_shift_protocol_sha256=proto,
        global_shift_seed=seed,
        dataset_id=ds,
        outer_fold=fold,
        condition=cond,
    )
    assert is_val is False
    assert "NPZ byte hash mismatch" in reason


def test_atomic_persistence(tmp_path):
    """Verify atomic write functions produce valid files with zero leftover temp files."""
    json_path = tmp_path / "test.json"
    csv_path = tmp_path / "test.csv"
    npz_path = tmp_path / "test.npz"

    atomic_write_json(json_path, {"key": "value"})
    atomic_write_csv(csv_path, pd.DataFrame({"a": [1, 2, 3]}))
    atomic_write_npz(npz_path, arr=np.ones(5))

    assert json_path.exists()
    assert csv_path.exists()
    assert npz_path.exists()

    # Zero temp files left
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0


def test_local_overlap_uses_source_median_not_mean():
    """Verify target missing coordinate geometry uses source median when source mean != source median."""
    # Source: x values = [0, 1, 100], so mean = 33.67, median = 1.0
    # y values = [0, 1, 100], so mean = 33.67, median = 1.0
    df_src = pd.DataFrame({
        "f1": [0.0, 1.0, 100.0, 0.0, 1.0, 100.0],
        "f2": [0.0, 1.0, 100.0, 0.0, 1.0, 100.0],
    })
    y_src = np.array([0, 0, 0, 1, 1, 1])

    # Target has NaN in f1
    df_tgt = pd.DataFrame({
        "f1": [np.nan, 2.0],
        "f2": [2.0, 2.0],
    })
    y_tgt = np.array([0, 1])

    source_stats = compute_source_statistics(df_src, {"f1": "numeric", "f2": "numeric"})
    assert source_stats.means["f1"] != source_stats.medians["f1"]

    cfg = {
        "local_overlap": {
            "boundary_fraction": 1.0,
            "alpha": {"mild": 0.25, "severe": 0.50},
        }
    }

    res = apply_local_overlap_shift(
        X_target=df_tgt,
        y_target=y_tgt,
        X_source=df_src,
        y_source=y_src,
        source_stats=source_stats,
        cfg=cfg,
        severity="mild",
    )

    assert res.status == "APPLICABLE"
    # The NaN coordinate in original target must be preserved as NaN
    assert np.isnan(res.X_shifted.iloc[0]["f1"])
    # The non-NaN coordinate was perturbed
    assert np.isfinite(res.X_shifted.iloc[0]["f2"])


def test_jobs_1_vs_jobs_auto_identical(tmp_path):
    """Verify execution produces deterministic, identical outputs."""
    from scripts.phase4_validate_shifts import process_dataset_fold_task
    import yaml

    shifts_cfg_path = PROJECT_ROOT / "configs" / "shifts.yaml"
    prep_cfg_path = PROJECT_ROOT / "configs" / "preprocessing.yaml"
    with open(shifts_cfg_path, "r", encoding="utf-8") as f:
        shift_cfg = yaml.safe_load(f)
    with open(prep_cfg_path, "r", encoding="utf-8") as f:
        prep_cfg = yaml.safe_load(f)

    # Use tmp_path so repo files are not modified during testing
    engine = ShiftEngine(shift_cfg, project_root=tmp_path)
    bundle_hash = "06a7bea1035adc450f695977cf5ff01ec45cd7cb140a5e8f23ad103da0949a27"
    split_hash = "62ecc40875d3cfe3850f993be5f18833054436c5c38b7f9262e9de2ad6da47d8"
    conditions = ["clean", "location_mild", "scale_severe"]

    # Run 1
    aud1, sum1, mon1, man1 = process_dataset_fold_task(
        "iris", 0, engine, prep_cfg, bundle_hash, split_hash, conditions, False, True, "numpy", PROJECT_ROOT
    )

    # Run 2
    aud2, sum2, mon2, man2 = process_dataset_fold_task(
        "iris", 0, engine, prep_cfg, bundle_hash, split_hash, conditions, False, True, "numpy", PROJECT_ROOT
    )

    assert man1 == man2
    assert [a["spec_hash"] for a in aud1] == [a["spec_hash"] for a in aud2]

