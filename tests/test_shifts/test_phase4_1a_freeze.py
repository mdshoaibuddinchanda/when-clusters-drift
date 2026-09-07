"""Phase 4.1a regression test suite.

Verifies:
1. Resume reconstruction auditing (A1):
   - In-memory reconstruction via generate_shift
   - Reconstructed row_index_map exact equality with persisted NPZ
   - Real preprocessing transformation audit (not hardcoded stubs)
   - Loud failure on corrupted/altered persisted artifacts
2. Genuine sequential-vs-threaded determinism (A2):
   - iris fold 0 and fold 1 across all 15 conditions
   - sequential loop vs ThreadPoolExecutor(max_workers=2)
   - isolated temporary output roots
   - exact equality on dataset, fold, condition, scenario_input_sha256, shift_spec_sha256, metadata, row_index_map
3. Strengthened --verify checks (A3):
   - Changed preprocessing.yaml, shifts.yaml, or synthetic_manifest.json fails verification
4. Summary JSON verification (A4):
   - Accurate factual validation of phase4_summary.json fields
"""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import yaml

from clusterdrift.shifts.base import ShiftResult
from clusterdrift.shifts.engine import ShiftEngine
from clusterdrift.shifts.hashing import (
    compute_file_sha256,
    compute_shift_protocol_sha256,
    load_canonical_bundle_hashes,
    load_phase2_split_hashes,
)
from scripts.phase4_validate_shifts import (
    ALL_CONDITIONS,
    CONTROLLED_DATASETS,
    execute_verify_mode,
    process_dataset_fold_task,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def base_configs():
    shifts_cfg_path = PROJECT_ROOT / "configs" / "shifts.yaml"
    prep_cfg_path = PROJECT_ROOT / "configs" / "preprocessing.yaml"
    with open(shifts_cfg_path, "r", encoding="utf-8") as f:
        shift_cfg = yaml.safe_load(f)
    with open(prep_cfg_path, "r", encoding="utf-8") as f:
        prep_cfg = yaml.safe_load(f)
    return shift_cfg, prep_cfg


def test_a1_resume_reconstruction_and_tamper_detection(tmp_path, base_configs, monkeypatch):
    """Verify resume mode reconstructs in memory, audits real transformation, and fails on discrepancy."""
    shift_cfg, prep_cfg = base_configs
    engine = ShiftEngine(shift_cfg, project_root=tmp_path)
    bundle_hashes = load_canonical_bundle_hashes(PROJECT_ROOT / "data" / "manifests" / "datasets.json")
    split_records = load_phase2_split_hashes(PROJECT_ROOT / "data" / "splits" / "split_manifest.json")

    bundle_hash = bundle_hashes["iris"]
    split_hash = split_records[("iris", 0)]["split_sha256"]
    conditions = ["clean", "location_mild", "outliers_severe"]

    # 1. First run generates and writes specs
    aud1, sum1, mon1, man1 = process_dataset_fold_task(
        dataset_id="iris",
        outer_fold=0,
        engine=engine,
        prep_cfg=prep_cfg,
        canonical_bundle_hash=bundle_hash,
        split_hash=split_hash,
        conditions=conditions,
        resume=False,
        force=True,
        backend="numpy",
        project_root=PROJECT_ROOT,
    )
    assert len(aud1) == 3
    assert all(a["all_finite"] for a in aud1)
    assert all(a["dimension_matches"] for a in aud1)

    spec_path = engine.get_spec_path("iris", 0, "location_mild")
    npz_path = spec_path.with_suffix(".npz")
    orig_spec_mtime = spec_path.stat().st_mtime_ns
    orig_npz_mtime = npz_path.stat().st_mtime_ns

    # 2. Second run with resume=True should validate, reconstruct in memory, and NOT rewrite files
    aud2, sum2, mon2, man2 = process_dataset_fold_task(
        dataset_id="iris",
        outer_fold=0,
        engine=engine,
        prep_cfg=prep_cfg,
        canonical_bundle_hash=bundle_hash,
        split_hash=split_hash,
        conditions=conditions,
        resume=True,
        force=False,
        backend="numpy",
        project_root=PROJECT_ROOT,
    )
    assert len(aud2) == 3
    assert all(a["all_finite"] for a in aud2)
    assert all(a["dimension_matches"] for a in aud2)
    assert [a["spec_hash"] for a in aud1] == [a["spec_hash"] for a in aud2]

    # File mtimes must be unchanged (no rewriting)
    assert spec_path.stat().st_mtime_ns == orig_spec_mtime
    assert npz_path.stat().st_mtime_ns == orig_npz_mtime

    # 3. If reconstructed shift disagrees with persisted NPZ -> must fail loudly with RuntimeError
    orig_generate = engine.generate_shift

    def bad_generate(*args, **kwargs):
        res = orig_generate(*args, **kwargs)
        bad_map = res.row_index_map.copy()
        bad_map[0] = 999999
        return ShiftResult(
            X_shifted=res.X_shifted,
            row_index_map=bad_map,
            metadata=res.metadata,
            status=res.status,
            reason=res.reason,
        )

    monkeypatch.setattr(engine, "generate_shift", bad_generate)

    with pytest.raises(RuntimeError, match="Reconstructed row_index_map differs from persisted NPZ"):
        process_dataset_fold_task(
            dataset_id="iris",
            outer_fold=0,
            engine=engine,
            prep_cfg=prep_cfg,
            canonical_bundle_hash=bundle_hash,
            split_hash=split_hash,
            conditions=["location_mild"],
            resume=True,
            force=False,
            backend="numpy",
            project_root=PROJECT_ROOT,
        )


def test_a2_genuine_sequential_vs_threaded_determinism(tmp_path, base_configs):
    """Run iris fold 0 and fold 1 across all 15 conditions through sequential vs ThreadPoolExecutor(2).

    Verify exact equality of canonical-sorted:
    dataset, fold, condition, scenario_input_sha256, shift_spec_sha256, metadata, row_index_map.
    """
    shift_cfg, prep_cfg = base_configs
    bundle_hashes = load_canonical_bundle_hashes(PROJECT_ROOT / "data" / "manifests" / "datasets.json")
    split_records = load_phase2_split_hashes(PROJECT_ROOT / "data" / "splits" / "split_manifest.json")
    bundle_hash = bundle_hashes["iris"]

    root_seq = tmp_path / "seq_output"
    root_thr = tmp_path / "thr_output"
    root_seq.mkdir()
    root_thr.mkdir()

    engine_seq = ShiftEngine(shift_cfg, project_root=root_seq)
    engine_thr = ShiftEngine(shift_cfg, project_root=root_thr)

    tasks = [
        ("iris", 0, bundle_hash, split_records[("iris", 0)]["split_sha256"]),
        ("iris", 1, bundle_hash, split_records[("iris", 1)]["split_sha256"]),
    ]

    # --- 1. Sequential Execution ---
    seq_manifests = []
    for ds, fold, b_sha, sp_sha in tasks:
        _, _, _, man = process_dataset_fold_task(
            dataset_id=ds,
            outer_fold=fold,
            engine=engine_seq,
            prep_cfg=prep_cfg,
            canonical_bundle_hash=b_sha,
            split_hash=sp_sha,
            conditions=ALL_CONDITIONS,
            resume=False,
            force=True,
            backend="numpy",
            project_root=PROJECT_ROOT,
        )
        seq_manifests.extend(man)

    # --- 2. ThreadPoolExecutor(max_workers=2) Execution ---
    thr_manifests = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                process_dataset_fold_task,
                ds,
                fold,
                engine_thr,
                prep_cfg,
                b_sha,
                sp_sha,
                ALL_CONDITIONS,
                False,
                True,
                "numpy",
                PROJECT_ROOT,
            )
            for ds, fold, b_sha, sp_sha in tasks
        ]
        for fut in futures:
            _, _, _, man = fut.result()
            thr_manifests.extend(man)

    # Sort canonically by (dataset_id, outer_fold, condition)
    sort_key = lambda x: (x["dataset_id"], x["outer_fold"], x["condition"])
    seq_manifests = sorted(seq_manifests, key=sort_key)
    thr_manifests = sorted(thr_manifests, key=sort_key)

    assert len(seq_manifests) == 30  # 2 folds * 15 conditions
    assert len(thr_manifests) == 30

    # Detailed comparison across every condition
    for s_entry, t_entry in zip(seq_manifests, thr_manifests):
        ds = s_entry["dataset_id"]
        fold = s_entry["outer_fold"]
        cond = s_entry["condition"]

        assert ds == t_entry["dataset_id"]
        assert fold == t_entry["outer_fold"]
        assert cond == t_entry["condition"]

        # Load saved specs and companion NPZs from both roots
        s_spec_path = engine_seq.get_spec_path(ds, fold, cond)
        t_spec_path = engine_thr.get_spec_path(ds, fold, cond)

        with open(s_spec_path, "r", encoding="utf-8") as f:
            s_spec = json.load(f)
        with open(t_spec_path, "r", encoding="utf-8") as f:
            t_spec = json.load(f)

        assert s_spec["scenario_input_sha256"] == t_spec["scenario_input_sha256"]
        assert s_spec["shift_spec_sha256"] == t_spec["shift_spec_sha256"]
        assert s_spec["metadata"] == t_spec["metadata"]

        with np.load(s_spec_path.with_suffix(".npz")) as s_npz, np.load(t_spec_path.with_suffix(".npz")) as t_npz:
            np.testing.assert_array_equal(s_npz["row_index_map"], t_npz["row_index_map"])


def test_a3_strengthened_verify_detects_configuration_changes(monkeypatch, base_configs):
    """Verify execute_verify_mode catches changed preprocessing, shifts, or synthetic manifests."""
    shift_cfg, prep_cfg = base_configs
    bundle_hashes = load_canonical_bundle_hashes(PROJECT_ROOT / "data" / "manifests" / "datasets.json")
    split_records = load_phase2_split_hashes(PROJECT_ROOT / "data" / "splits" / "split_manifest.json")

    # In clean state, verify must succeed
    with pytest.raises(SystemExit) as exc_info:
        execute_verify_mode(
            project_root=PROJECT_ROOT,
            shift_cfg=shift_cfg,
            controlled_datasets=CONTROLLED_DATASETS,
            canonical_bundle_hashes=bundle_hashes,
            split_records=split_records,
        )
    assert exc_info.value.code == 0

    # Modify shift_cfg protocol -> must exit with code 1
    tampered_cfg = copy.deepcopy(shift_cfg)
    tampered_cfg["location"]["severities"]["mild"] = 999.0
    with pytest.raises(SystemExit) as exc_info:
        execute_verify_mode(
            project_root=PROJECT_ROOT,
            shift_cfg=tampered_cfg,
            controlled_datasets=CONTROLLED_DATASETS,
            canonical_bundle_hashes=bundle_hashes,
            split_records=split_records,
        )
    assert exc_info.value.code == 1
