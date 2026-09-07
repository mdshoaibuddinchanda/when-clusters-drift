"""Strict byte-read-only Phase 5 integrity verification engine.

Verifies:
1. Phase-5 input lock against all configuration, manifest, and upstream artifact hashes.
2. Exact producer commit binding.
3. Complete expected probe-key universe (150 reference + 2250 current = 2400 total).
4. No orphan JSON or companion NPZ files on disk.
5. Deep descriptor validation (distinct array byte hashes, companion NPZ byte hashes, bank hashes).
6. Semantic multi-level provenance chains:
   - Reference: canonical_row_indices == source_indices[selected_source_positions].
   - Current: target_partition_positions == row_index_map[selected_current_positions],
              canonical_row_indices == target_indices[target_partition_positions].
7. Validation result CSV row counts and integrity.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

from clusterdrift.probes.bank import load_current_probe_descriptor, load_reference_probe_descriptor
from clusterdrift.probes.hashing import (
    compute_alignment_protocol_sha256,
    compute_file_sha256,
    compute_probe_protocol_sha256,
)
from clusterdrift.probes.validation import (
    validate_saved_current_descriptor,
    validate_saved_reference_descriptor,
)
from clusterdrift.shifts.hashing import (
    compute_shift_protocol_sha256,
    load_canonical_bundle_hashes,
    load_phase2_split_hashes,
)

FROZEN_PRODUCER_COMMIT = "412e20f6b8d0f54b21557dfece01e88fdac0ab47"

CONTROLLED_DATASETS = [
    "aps_failure", "balance_scale", "bank_marketing", "banknote_authentication",
    "breast_cancer_wisconsin_diagnostic", "dermatology", "ecoli", "electricity",
    "glass", "haberman_survival", "heart_disease", "human_activity_recognition",
    "image_segmentation", "ionosphere", "iris", "isolet", "letter_recognition",
    "madelon", "mice_protein_expression", "optdigits", "pendigits", "pima_diabetes",
    "satimage", "seeds", "sonar", "spambase", "vehicle_silhouettes", "waveform",
    "wine", "yeast"
]

ALL_CONDITIONS = [
    "clean",
    "location_mild", "location_severe",
    "scale_mild", "scale_severe",
    "mcar_mild", "mcar_severe",
    "outliers_mild", "outliers_severe",
    "measurement_noise_mild", "measurement_noise_severe",
    "class_prevalence_mild", "class_prevalence_severe",
    "local_overlap_mild", "local_overlap_severe"
]


def load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_phase5_integrity(
    project_root: Path,
    expected_producer_commit: str = FROZEN_PRODUCER_COMMIT,
) -> List[str]:
    """Perform strictly byte-read-only integrity verification of all Phase 5 artifacts.

    Returns an empty list if all artifacts and provenance chains are valid,
    or a list of error strings describing specific failures.
    """
    errors: List[str] = []
    probes_dir = project_root / "data" / "probes"
    shifts_dir = project_root / "data" / "shifts"
    lock_path = probes_dir / "phase5_input_lock.json"
    manifest_path = probes_dir / "probe_manifest.json"

    if not lock_path.exists():
        errors.append(f"Missing input lock: {lock_path}")
    if not manifest_path.exists():
        errors.append(f"Missing probe manifest: {manifest_path}")

    if errors:
        return errors

    # -------------------------------------------------------------------------
    # 1. Complete Phase-5 Input-Lock Verification
    # -------------------------------------------------------------------------
    with open(lock_path, "r", encoding="utf-8") as f:
        lock_doc = json.load(f)

    # Check upstream freeze commits
    if lock_doc.get("phase4_tooling_freeze_commit") != "f08335e8a4822f2b95e379abf955a80d5c0fb2c8":
        errors.append("phase4_tooling_freeze_commit mismatch in lock")
    if lock_doc.get("phase4_artifact_commit") != "6f8dd88b59a297ae05da035835c4055aab43fb29":
        errors.append("phase4_artifact_commit mismatch in lock")
    if lock_doc.get("phase4_producer_commit") != "21bcaa7bb00b07de5ee4671adfb06932610bf92d":
        errors.append("phase4_producer_commit mismatch in lock")

    # Explicitly compare every applicable current lock field against disk files
    file_checks = [
        ("phase1_datasets_manifest_sha256", project_root / "data" / "manifests" / "datasets.json"),
        ("phase2_split_manifest_sha256", project_root / "data" / "splits" / "split_manifest.json"),
        ("phase2_preprocessing_config_sha256", project_root / "configs" / "preprocessing.yaml"),
        ("methods_config_sha256", project_root / "configs" / "methods.yaml"),
        ("probe_config_sha256", project_root / "configs" / "probes.yaml"),
        ("alignment_config_sha256", project_root / "configs" / "alignment.yaml"),
        ("phase4_input_lock_sha256", shifts_dir / "phase4_input_lock.json"),
        ("phase4_shift_manifest_sha256", shifts_dir / "shift_manifest.json"),
    ]
    for field_name, fpath in file_checks:
        if not fpath.exists():
            errors.append(f"Missing referenced file for lock field {field_name}: {fpath}")
        else:
            computed_sha = compute_file_sha256(fpath)
            if lock_doc.get(field_name) != computed_sha:
                errors.append(
                    f"{field_name} mismatch: lock has {lock_doc.get(field_name)} but {fpath.name} computed {computed_sha}"
                )

    # Continue verifying protocol hashes independently
    probe_cfg_p = project_root / "configs" / "probes.yaml"
    if probe_cfg_p.exists():
        probe_cfg = load_yaml(probe_cfg_p)
        probe_protocol_sha = compute_probe_protocol_sha256(probe_cfg)
        if lock_doc.get("probe_protocol_sha256") != probe_protocol_sha:
            errors.append("probe_protocol_sha256 mismatch")
    else:
        errors.append("Missing configs/probes.yaml")
        probe_cfg = {}
        probe_protocol_sha = ""

    align_cfg_p = project_root / "configs" / "alignment.yaml"
    if align_cfg_p.exists():
        align_cfg = load_yaml(align_cfg_p)
        align_protocol_sha = compute_alignment_protocol_sha256(align_cfg)
        if lock_doc.get("alignment_protocol_sha256") != align_protocol_sha:
            errors.append("alignment_protocol_sha256 mismatch")
    else:
        errors.append("Missing configs/alignment.yaml")

    # Bind producer commit exactly
    gen_commit = lock_doc.get("generated_from_commit", "")
    if gen_commit != expected_producer_commit:
        errors.append(
            f"generated_from_commit mismatch in lock: expected '{expected_producer_commit}', found '{gen_commit}'"
        )

    # -------------------------------------------------------------------------
    # 2. Exact Expected Probe-Key Universe and Manifest Checks
    # -------------------------------------------------------------------------
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_doc = json.load(f)

    probes = manifest_doc.get("probes", [])
    if len(probes) != 2400:
        errors.append(f"Probe manifest count is {len(probes)}, expected exactly 2400")

    ref_count = sum(1 for p in probes if p["bank_type"] == "reference")
    cur_count = sum(1 for p in probes if p["bank_type"] == "current")

    if ref_count != 150:
        errors.append(f"Reference probe count is {ref_count}, expected 150")
    if cur_count != 2250:
        errors.append(f"Current probe count is {cur_count}, expected 2250")

    # Construct expected key universe programmatically
    expected_keys = set()
    for ds in CONTROLLED_DATASETS:
        for f_idx in range(5):
            expected_keys.add(("reference", ds, f_idx))
            for c in ALL_CONDITIONS:
                expected_keys.add(("current", ds, f_idx, c))

    manifest_keys = set()
    manifest_relpaths = set()
    for p in probes:
        b_type = p.get("bank_type")
        ds = p.get("dataset_id")
        f_idx = p.get("outer_fold")
        if b_type == "reference":
            k = ("reference", ds, f_idx)
        else:
            k = ("current", ds, f_idx, p.get("condition"))
        if k in manifest_keys:
            errors.append(f"Duplicate probe manifest entry: {k}")
        manifest_keys.add(k)
        manifest_relpaths.add(p["spec_relpath"])

    if manifest_keys != expected_keys:
        missing_keys = expected_keys - manifest_keys
        unexpected_keys = manifest_keys - expected_keys
        if missing_keys:
            errors.append(f"Manifest missing {len(missing_keys)} expected keys (sample: {list(missing_keys)[:3]})")
        if unexpected_keys:
            errors.append(f"Manifest has {len(unexpected_keys)} unexpected keys (sample: {list(unexpected_keys)[:3]})")

    # -------------------------------------------------------------------------
    # 3. Orphan File Verification
    # -------------------------------------------------------------------------
    disk_json_files = [
        str(f.relative_to(project_root)).replace("\\", "/")
        for f in probes_dir.rglob("*.json")
        if f.name not in ("probe_manifest.json", "phase5_input_lock.json")
    ]
    orphan_jsons = set(disk_json_files) - manifest_relpaths
    if orphan_jsons:
        errors.append(f"Found {len(orphan_jsons)} orphan descriptor JSON files not in manifest")

    disk_npz_files = [
        str(f.relative_to(project_root)).replace("\\", "/")
        for f in probes_dir.rglob("*.npz")
    ]
    expected_npz = {p.replace(".json", ".npz") for p in manifest_relpaths}
    orphan_npzs = set(disk_npz_files) - expected_npz
    if orphan_npzs:
        errors.append(f"Found {len(orphan_npzs)} orphan NPZ files not in manifest")

    # -------------------------------------------------------------------------
    # 4. Deep Descriptor, Companion NPZ, and Semantic Provenance Verification
    # -------------------------------------------------------------------------
    datasets_manifest_p = project_root / "data" / "manifests" / "datasets.json"
    splits_manifest_p = project_root / "data" / "splits" / "split_manifest.json"
    bundle_hashes = load_canonical_bundle_hashes(datasets_manifest_p) if datasets_manifest_p.exists() else {}
    split_records = load_phase2_split_hashes(splits_manifest_p) if splits_manifest_p.exists() else {}
    global_seed = probe_cfg.get("global_probe_seed", 2026090705)
    prep_config_sha = (
        compute_file_sha256(project_root / "configs" / "preprocessing.yaml")
        if (project_root / "configs" / "preprocessing.yaml").exists()
        else ""
    )

    shifts_cfg_p = project_root / "configs" / "shifts.yaml"
    shift_protocol_sha = (
        compute_shift_protocol_sha256(load_yaml(shifts_cfg_p))
        if shifts_cfg_p.exists()
        else ""
    )

    for p in probes:
        spec_p = project_root / p["spec_relpath"]
        if not spec_p.exists():
            errors.append(f"Missing probe spec: {spec_p}")
            continue
        npz_p = spec_p.with_suffix(".npz")
        if not npz_p.exists():
            errors.append(f"Missing probe companion NPZ: {npz_p}")
            continue

        ds = p["dataset_id"]
        f_idx = p["outer_fold"]
        b_type = p["bank_type"]
        fold_split_p = project_root / "data" / "splits" / "controlled" / ds / f"fold_{f_idx}.npz"

        if b_type == "reference":
            b_sha = bundle_hashes.get(ds, "")
            sp_sha = split_records.get((ds, f_idx), {}).get("split_sha256", "")
            is_val, err, _ = validate_saved_reference_descriptor(
                spec_path=spec_p,
                expected_canonical_bundle_sha256=b_sha,
                expected_split_sha256=sp_sha,
                expected_preprocessing_config_sha256=prep_config_sha,
                expected_probe_protocol_sha256=probe_protocol_sha,
                global_probe_seed=global_seed,
                dataset_id=ds,
                outer_fold=f_idx,
            )
            if not is_val:
                errors.append(f"Reference descriptor validation error on {p['spec_relpath']}: {err}")

            # Semantic reference provenance verification
            if fold_split_p.exists():
                with np.load(npz_p) as npz:
                    sel_pos = npz.get("selected_source_positions")
                    can_rows = npz.get("canonical_row_indices")
                with np.load(fold_split_p) as sp_npz:
                    src_indices = sp_npz["source_indices"]

                if sel_pos is None or can_rows is None:
                    errors.append(f"Missing array keys in reference NPZ: {npz_p}")
                else:
                    if sel_pos.ndim != 1 or not np.issubdtype(sel_pos.dtype, np.integer):
                        errors.append(f"selected_source_positions not 1-D integer array in {npz_p}")
                    if len(sel_pos) > 0 and (np.min(sel_pos) < 0 or np.max(sel_pos) >= len(src_indices)):
                        errors.append(f"selected_source_positions out of bounds [0, {len(src_indices)}) in {npz_p}")
                    if not np.array_equal(can_rows, src_indices[sel_pos]):
                        errors.append(
                            f"Semantic provenance mismatch in reference bank {spec_p}: canonical_row_indices != source_indices[selected_source_positions]"
                        )
        else:
            cond = p["condition"]
            shift_spec_path = shifts_dir / "specs" / ds / f"fold_{f_idx}" / f"{cond}.json"
            shift_npz_path = shift_spec_path.with_suffix(".npz")
            if not shift_spec_path.exists():
                errors.append(f"Missing upstream shift spec: {shift_spec_path}")
                continue
            with open(shift_spec_path, "r", encoding="utf-8") as sf:
                sdoc = json.load(sf)
            shift_spec_sha = sdoc["shift_spec_sha256"]

            is_val, err, _ = validate_saved_current_descriptor(
                spec_path=spec_p,
                expected_shift_spec_sha256=shift_spec_sha,
                expected_shift_protocol_sha256=shift_protocol_sha,
                expected_preprocessing_config_sha256=prep_config_sha,
                expected_probe_protocol_sha256=probe_protocol_sha,
                global_probe_seed=global_seed,
                dataset_id=ds,
                outer_fold=f_idx,
                condition=cond,
            )
            if not is_val:
                errors.append(f"Current descriptor validation error on {p['spec_relpath']}: {err}")

            # Semantic current provenance verification
            if shift_npz_path.exists() and fold_split_p.exists():
                with np.load(npz_p) as npz:
                    sel_cur_pos = npz.get("selected_current_positions")
                    tgt_part_pos = npz.get("target_partition_positions")
                    can_rows = npz.get("canonical_row_indices")
                with np.load(shift_npz_path) as snpz:
                    row_index_map = snpz["row_index_map"]
                with np.load(fold_split_p) as sp_npz:
                    tgt_indices = sp_npz["target_indices"]

                if sel_cur_pos is None or tgt_part_pos is None or can_rows is None:
                    errors.append(f"Missing array keys in current NPZ: {npz_p}")
                else:
                    if sel_cur_pos.ndim != 1 or not np.issubdtype(sel_cur_pos.dtype, np.integer):
                        errors.append(f"selected_current_positions not 1-D integer array in {npz_p}")
                    elif len(sel_cur_pos) > 0 and (np.min(sel_cur_pos) < 0 or np.max(sel_cur_pos) >= len(row_index_map)):
                        errors.append(f"selected_current_positions out of bounds [0, {len(row_index_map)}) in {npz_p}")
                    elif not np.array_equal(tgt_part_pos, row_index_map[sel_cur_pos]):
                        errors.append(
                            f"Semantic provenance mismatch in current bank {spec_p}: target_partition_positions != row_index_map[selected_current_positions]"
                        )

                    if tgt_part_pos.ndim != 1 or not np.issubdtype(tgt_part_pos.dtype, np.integer):
                        errors.append(f"target_partition_positions not 1-D integer array in {npz_p}")
                    elif len(tgt_part_pos) > 0 and (np.min(tgt_part_pos) < 0 or np.max(tgt_part_pos) >= len(tgt_indices)):
                        errors.append(f"target_partition_positions out of bounds [0, {len(tgt_indices)}) in {npz_p}")
                    elif not np.array_equal(can_rows, tgt_indices[tgt_part_pos]):
                        errors.append(
                            f"Semantic provenance mismatch in current bank {spec_p}: canonical_row_indices != target_indices[target_partition_positions]"
                        )

    # -------------------------------------------------------------------------
    # 5. Result CSV Integrity Verification
    # -------------------------------------------------------------------------
    probe_results_dir = project_root / "results" / "probe_validation"
    align_results_dir = project_root / "results" / "alignment_validation"

    expected_csvs = [
        (probe_results_dir / "reference_probe_audit.csv", 150),
        (probe_results_dir / "current_probe_audit.csv", 2250),
        (probe_results_dir / "probe_manifest_summary.csv", 3),
        (align_results_dir / "alignment_runs.csv", 144),
        (align_results_dir / "alignment_summary.csv", 10),
        (probe_results_dir / "probe_size_stability.csv", 12),
        (align_results_dir / "eta_sensitivity.csv", 720),
        (align_results_dir / "performance_benchmark.csv", 1),
    ]

    for csv_path, exp_min in expected_csvs:
        if not csv_path.exists():
            errors.append(f"Missing result CSV: {csv_path}")
        else:
            df = pd.read_csv(csv_path)
            if len(df) < exp_min:
                errors.append(f"{csv_path.name} has {len(df)} rows, expected at least {exp_min}")

    amb_csv = align_results_dir / "ambiguity_cases.csv"
    if not amb_csv.exists():
        errors.append(f"Missing ambiguity cases CSV: {amb_csv}")

    return errors
