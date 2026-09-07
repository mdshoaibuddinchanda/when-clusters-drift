#!/usr/bin/env python
"""Phase 4 Controlled Distribution-Shift Engine Validation and Execution CLI.

Executes, audits, benchmarks, and verifies the 2250 controlled distribution shift scenarios:
30 controlled real datasets x 5 outer folds x 15 conditions (1 clean + 7 families x 2 severities).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clusterdrift.data.preprocess import build_preprocessor
from clusterdrift.shifts import (
    ALL_CONDITIONS,
    SEVERITIES,
    SHIFT_FAMILIES,
    ShiftEngine,
    ShiftResult,
    audit_preprocessing_transformation,
    compute_file_sha256,
    detect_hardware,
    verify_severity_monotonicity,
)
from clusterdrift.shifts.backend import should_use_gpu, TORCH_AVAILABLE
from clusterdrift.shifts.hashing import compute_canonical_json_sha256

if TORCH_AVAILABLE:
    import torch

CONTROLLED_DATASETS = [
    "aps_failure",
    "balance_scale",
    "bank_marketing",
    "banknote_authentication",
    "breast_cancer_wisconsin_diagnostic",
    "dermatology",
    "ecoli",
    "electricity",
    "glass",
    "haberman_survival",
    "heart_disease",
    "human_activity_recognition",
    "image_segmentation",
    "ionosphere",
    "iris",
    "isolet",
    "letter_recognition",
    "madelon",
    "mice_protein_expression",
    "optdigits",
    "pendigits",
    "pima_diabetes",
    "satimage",
    "seeds",
    "sonar",
    "spambase",
    "vehicle_silhouettes",
    "waveform",
    "wine",
    "yeast",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_fold_raw(
    dataset_id: str,
    outer_fold: int,
    project_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any], str, str]:
    """Load raw source and target partitions along with metadata and split hashes."""
    ds_dir = project_root / "data" / "canonical" / "controlled" / dataset_id
    features_path = ds_dir / "features.parquet"
    labels_path = ds_dir / "labels.parquet"
    meta_path = ds_dir / "metadata.json"

    df_X = pd.read_parquet(features_path)
    df_y = pd.read_parquet(labels_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fold_path = project_root / "data" / "splits" / "controlled" / dataset_id / f"fold_{outer_fold}.npz"
    fold_json = project_root / "data" / "splits" / "controlled" / dataset_id / f"fold_{outer_fold}.json"

    with np.load(fold_path) as npz:
        src_idx = npz["source_indices"]
        tgt_idx = npz["target_indices"]

    X_source_raw = df_X.iloc[src_idx].copy()
    X_target_raw = df_X.iloc[tgt_idx].copy()
    y_source = df_y.iloc[src_idx].to_numpy().ravel()
    y_target = df_y.iloc[tgt_idx].to_numpy().ravel()

    bundle_hash = compute_file_sha256(features_path)
    split_hash = compute_file_sha256(fold_json) if fold_json.exists() else compute_file_sha256(fold_path)

    return X_source_raw, X_target_raw, y_source, y_target, meta, bundle_hash, split_hash


def build_phase4_input_lock(
    project_root: Path,
    shift_cfg: Dict[str, Any],
    controlled_datasets: List[str],
) -> Dict[str, Any]:
    """Construct phase4_input_lock.json metadata."""
    splits_manifest_path = project_root / "data" / "splits" / "split_manifest.json"
    prep_cfg_path = project_root / "configs" / "preprocessing.yaml"
    shifts_cfg_path = project_root / "configs" / "shifts.yaml"

    splits_sha = compute_file_sha256(splits_manifest_path) if splits_manifest_path.exists() else None
    prep_sha = compute_file_sha256(prep_cfg_path) if prep_cfg_path.exists() else None
    shift_sha = compute_file_sha256(shifts_cfg_path) if shifts_cfg_path.exists() else None

    # Collect split hashes per dataset
    split_hashes: Dict[str, List[str]] = {}
    for ds in controlled_datasets:
        fold_hashes = []
        for f in range(5):
            fj = project_root / "data" / "splits" / "controlled" / ds / f"fold_{f}.json"
            if fj.exists():
                fold_hashes.append(compute_file_sha256(fj))
            else:
                fn = project_root / "data" / "splits" / "controlled" / ds / f"fold_{f}.npz"
                fold_hashes.append(compute_file_sha256(fn))
        split_hashes[ds] = fold_hashes

    lock_doc = {
        "protocol_version": shift_cfg.get("protocol_version", 1),
        "phase3_freeze_commit": "8a471625af6301fe8941090748c8057e08324326",
        "created_from_git_commit": "8a471625af6301fe8941090748c8057e08324326",
        "global_shift_seed": shift_cfg.get("global_shift_seed", 2026090704),
        "phase2_split_manifest_sha256": splits_sha,
        "phase2_preprocessing_config_sha256": prep_sha,
        "shift_config_sha256": shift_sha,
        "included_controlled_datasets_count": len(controlled_datasets),
        "included_controlled_datasets": controlled_datasets,
        "included_split_hashes": split_hashes,
        "excluded_natural_datasets_and_reason": {
            "tableshift": "Natural-shift benchmark suite serves as external real-world distribution shift controls and must not receive synthetic controlled perturbations",
            "whyshift": "Natural-shift benchmark suite serves as external real-world distribution shift controls and must not receive synthetic controlled perturbations",
        },
    }
    return lock_doc


def run_benchmark_and_equivalence(
    engine: ShiftEngine,
    prep_cfg: Dict[str, Any],
    project_root: Path,
    out_dir: Path,
) -> None:
    """Benchmark NumPy vs PyTorch CPU and CUDA, and record numerical equivalence."""
    print("\n--- Running Backend Performance Benchmark & Equivalence ---")
    benchmark_datasets = ["iris", "wine", "human_activity_recognition"]
    
    eq_records = []
    perf_records = []

    for ds in benchmark_datasets:
        X_src, X_tgt, y_src, y_tgt, meta, _, _ = load_dataset_fold_raw(ds, 0, project_root)
        roles = meta.get("feature_roles", {col: "numeric" for col in X_tgt.columns})
        n_rows, n_cols = X_tgt.shape

        for fam in ["location", "scale", "measurement_noise", "outliers"]:
            cond = f"{fam}_mild"
            
            # 1. NumPy run
            t0 = time.perf_counter()
            res_numpy = engine.generate_shift(ds, 0, cond, X_tgt, X_src, roles, y_tgt, y_src, backend="numpy")
            t_numpy = time.perf_counter() - t0

            # 2. PyTorch CUDA run (if available)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                t0 = time.perf_counter()
                res_torch = engine.generate_shift(ds, 0, cond, X_tgt, X_src, roles, y_tgt, y_src, backend="torch")
                t_torch = time.perf_counter() - t0

                # Compute maximum absolute deviation on numeric columns
                num_cols = [c for c in X_tgt.columns if roles.get(c) == "numeric"]
                arr_np = res_numpy.X_shifted[num_cols].to_numpy(dtype=np.float64)
                arr_th = res_torch.X_shifted[num_cols].to_numpy(dtype=np.float64)
                valid = np.isfinite(arr_np) & np.isfinite(arr_th)
                max_dev = float(np.max(np.abs(arr_np[valid] - arr_th[valid]))) if np.any(valid) else 0.0

                eq_records.append({
                    "dataset": ds,
                    "rows": n_rows,
                    "features": n_cols,
                    "condition": cond,
                    "max_abs_deviation": max_dev,
                    "max_rel_deviation": max_dev,
                    "equivalent_within_1e6": bool(max_dev <= 1e-6),
                })

                speedup = round(t_numpy / max(t_torch, 1e-6), 2)
                perf_records.append({
                    "dataset": ds,
                    "rows": n_rows,
                    "features": n_cols,
                    "family": fam,
                    "backend": "torch_cuda",
                    "wall_seconds": round(t_torch, 5),
                    "speedup_vs_numpy": speedup,
                    "max_cpu_gpu_deviation": max_dev,
                })

            perf_records.append({
                "dataset": ds,
                "rows": n_rows,
                "features": n_cols,
                "family": fam,
                "backend": "numpy_cpu",
                "wall_seconds": round(t_numpy, 5),
                "speedup_vs_numpy": 1.0,
                "max_cpu_gpu_deviation": 0.0,
            })

    if eq_records:
        df_eq = pd.DataFrame(eq_records)
        df_eq.to_csv(out_dir / "backend_equivalence.csv", index=False)
        print(f"Saved: {out_dir / 'backend_equivalence.csv'}")

    if perf_records:
        df_perf = pd.DataFrame(perf_records)
        df_perf.to_csv(out_dir / "performance_benchmark.csv", index=False)
        print(f"Saved: {out_dir / 'performance_benchmark.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Controlled Shift Engine Runner")
    parser.add_argument("--all", action="store_true", help="Process all 30 controlled real datasets")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset ID")
    parser.add_argument("--fold", type=int, default=None, help="Filter by outer fold (0-4)")
    parser.add_argument("--family", type=str, default=None, help="Filter by shift family")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "numpy", "torch"])
    parser.add_argument("--jobs", type=str, default="auto", help="Number of CPU jobs or 'auto'")
    parser.add_argument("--resume", action="store_true", help="Skip already existing valid specs")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing specs")
    parser.add_argument("--verify", action="store_true", help="Strictly byte-read-only verification")
    parser.add_argument("--dry-run", action="store_true", help="Report planned scenario counts without writing")
    args = parser.parse_args()

    print("=" * 75)
    print("PHASE 4: CONTROLLED DISTRIBUTION-SHIFT ENGINE AUDIT & GENERATION")
    print("=" * 75)

    shifts_cfg_path = PROJECT_ROOT / "configs" / "shifts.yaml"
    prep_cfg_path = PROJECT_ROOT / "configs" / "preprocessing.yaml"
    shift_cfg = load_yaml(shifts_cfg_path)
    prep_cfg = load_yaml(prep_cfg_path)

    engine = ShiftEngine(shift_cfg, project_root=PROJECT_ROOT)
    hw = detect_hardware()
    print(f"Hardware: {hw['processor']} | Logical CPUs: {hw['logical_cpus']} | RAM: {hw['total_ram_gb']} GB")
    print(f"CUDA Available: {hw['cuda_available']} | GPU: {hw['gpu_model']} ({hw['gpu_vram_gb']} GB VRAM)")
    print(f"Protocol SHA-256: {engine.protocol_sha256}")

    target_datasets = [args.dataset] if args.dataset else (CONTROLLED_DATASETS if (args.all or not args.dataset) else [CONTROLLED_DATASETS[0]])
    target_folds = [args.fold] if args.fold is not None else list(range(5))
    target_conditions = ALL_CONDITIONS if not args.family else ([c for c in ALL_CONDITIONS if c.startswith(args.family)] or ALL_CONDITIONS)

    total_nominal_scenarios = len(target_datasets) * len(target_folds) * len(target_conditions)
    print(f"Planned scope: {len(target_datasets)} datasets x {len(target_folds)} folds x {len(target_conditions)} conditions = {total_nominal_scenarios} scenarios")

    # 1. DRY RUN
    if args.dry_run:
        print("\n--- DRY RUN MODE (Zero files modified) ---")
        applicable_count = 0
        not_applicable_count = 0
        per_family_counts: Dict[str, int] = {}

        for ds in target_datasets:
            for fold in target_folds:
                meta_path = PROJECT_ROOT / "data" / "canonical" / "controlled" / ds / "metadata.json"
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                roles = meta.get("feature_roles", {})
                num_cols = [c for c, r in roles.items() if r == "numeric"]
                n_classes = int(meta.get("n_classes", 2))

                for cond in target_conditions:
                    fam = "clean" if cond == "clean" else cond.rsplit("_", 1)[0]
                    per_family_counts[fam] = per_family_counts.get(fam, 0) + 1

                    # Check applicability heuristics
                    if fam in ["location", "scale", "outliers", "measurement_noise"] and len(num_cols) == 0:
                        not_applicable_count += 1
                    elif fam == "local_overlap" and (len(num_cols) < 2 or n_classes < 2):
                        not_applicable_count += 1
                    elif fam == "class_prevalence" and n_classes < 2:
                        not_applicable_count += 1
                    else:
                        applicable_count += 1

        print(f"Dry Run Total Nominal Scenarios: {total_nominal_scenarios}")
        print(f"Estimated APPLICABLE: {applicable_count}")
        print(f"Estimated NOT_APPLICABLE: {not_applicable_count}")
        print(f"Per-family condition counts: {per_family_counts}")
        print("\nDry run successfully finished.")
        return

    # 2. VERIFY-ONLY MODE
    if args.verify:
        print("\n--- VERIFY-ONLY MODE (Strictly byte-read-only) ---")
        specs_dir = PROJECT_ROOT / "data" / "shifts" / "specs"
        manifest_path = PROJECT_ROOT / "data" / "shifts" / "shift_manifest.json"
        lock_path = PROJECT_ROOT / "data" / "shifts" / "phase4_input_lock.json"

        if not manifest_path.exists():
            print(f"ERROR: Manifest not found at {manifest_path}. Run full generation first.")
            sys.exit(1)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest_entries = manifest.get("scenarios", [])
        print(f"Loaded manifest with {len(manifest_entries)} scenario entries.")
        assert len(manifest_entries) == 2250, f"Expected exactly 2250 manifest entries, got {len(manifest_entries)}"

        verified_specs = 0
        for entry in manifest_entries:
            spec_file = PROJECT_ROOT / entry["spec_relpath"]
            if not spec_file.exists():
                print(f"ERROR: Missing spec file: {spec_file}")
                sys.exit(1)
            # Verify file hash matches manifest
            file_hash = compute_file_sha256(spec_file)
            if file_hash != entry["shift_spec_file_sha256"]:
                print(f"ERROR: Spec hash mismatch for {spec_file}")
                sys.exit(1)
            verified_specs += 1

        print(f"All {verified_specs} / 2250 shift specifications verified byte-identically on disk!")
        print("Verify mode passed: Zero files modified.")
        return

    # 3. FULL SHIFT GENERATION AND AUDIT
    out_dir = PROJECT_ROOT / "results" / "shift_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    shifts_data_dir = PROJECT_ROOT / "data" / "shifts"
    shifts_data_dir.mkdir(parents=True, exist_ok=True)

    # Write input lock
    lock_doc = build_phase4_input_lock(PROJECT_ROOT, shift_cfg, CONTROLLED_DATASETS)
    lock_path = shifts_data_dir / "phase4_input_lock.json"
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_doc, f, indent=2, sort_keys=True)
    print(f"Saved input lock: {lock_path} (SHA-256: {compute_file_sha256(lock_path)})")

    audit_records = []
    summary_records = []
    monotonicity_records = []
    manifest_entries = []
    failures: Dict[str, Any] = {}

    start_time = time.perf_counter()
    scenario_idx = 0

    for ds_idx, ds in enumerate(target_datasets):
        print(f"[{ds_idx+1}/{len(target_datasets)}] Processing dataset: {ds}")
        
        for fold in target_folds:
            t_fold_start = time.perf_counter()
            X_src, X_tgt, y_src, y_tgt, meta, bundle_hash, split_hash = load_dataset_fold_raw(ds, fold, PROJECT_ROOT)
            roles = meta.get("feature_roles", {col: "numeric" for col in X_tgt.columns})

            # Fit Phase-2 preprocessor ONCE on raw outer source
            prep = build_preprocessor(feature_roles=roles, config=prep_cfg, metadata=meta)
            prep.fit(X_src)
            X_src_proc = prep.transform(X_src)
            D_encoded = X_src_proc.shape[1]

            fold_results: Dict[str, ShiftResult] = {}

            for cond in target_conditions:
                scenario_idx += 1
                t0_cond = time.perf_counter()
                
                spec_path = engine.get_spec_path(ds, fold, cond)
                # Check resume
                if args.resume and not args.force and spec_path.exists():
                    try:
                        with open(spec_path, "r", encoding="utf-8") as f:
                            saved_spec = json.load(f)
                        # Reconstruct basic result for audit
                        # We still run preprocessor transformation to verify finiteness
                        res = engine.generate_shift(ds, fold, cond, X_tgt, X_src, roles, y_tgt, y_src, backend=args.backend)
                        spec_sha = saved_spec["shift_spec_sha256"]
                    except Exception:
                        res = engine.generate_shift(ds, fold, cond, X_tgt, X_src, roles, y_tgt, y_src, backend=args.backend)
                        spec_path = engine.save_spec_atomic(ds, fold, cond, res, bundle_hash, split_hash)
                        spec_sha = compute_file_sha256(spec_path)
                else:
                    res = engine.generate_shift(ds, fold, cond, X_tgt, X_src, roles, y_tgt, y_src, backend=args.backend)
                    spec_path = engine.save_spec_atomic(ds, fold, cond, res, bundle_hash, split_hash)
                    spec_sha = compute_file_sha256(spec_path)

                cond_time = time.perf_counter() - t0_cond
                fold_results[cond] = res

                # Audit preprocessing transformation
                proc_audit = audit_preprocessing_transformation(res.X_shifted, prep, D_encoded)

                audit_records.append({
                    "dataset": ds,
                    "fold": fold,
                    "condition": cond,
                    "raw_target_rows": proc_audit["raw_target_rows"],
                    "shifted_target_rows": proc_audit["shifted_target_rows"],
                    "raw_features": proc_audit["raw_features"],
                    "encoded_features": proc_audit["encoded_features"],
                    "raw_missing_count": proc_audit["raw_missing_count"],
                    "post_transform_missing_count": proc_audit["post_transform_missing_count"],
                    "all_finite": proc_audit["all_finite"],
                    "output_dtype": proc_audit["output_dtype"],
                    "preprocessing_config_hash": compute_canonical_json_sha256(prep_cfg),
                    "split_hash": split_hash,
                    "shift_spec_hash": spec_sha,
                })

                fam = "clean" if cond == "clean" else cond.rsplit("_", 1)[0]
                sev = "none" if cond == "clean" else cond.rsplit("_", 1)[1]
                summary_records.append({
                    "dataset": ds,
                    "fold": fold,
                    "condition": cond,
                    "family": fam,
                    "severity": sev,
                    "status": res.status,
                    "reason": res.reason,
                    "runtime_seconds": round(cond_time, 5),
                    "all_finite": proc_audit["all_finite"],
                    "dimension_matches": proc_audit["dimension_matches"],
                })

                manifest_entries.append({
                    "dataset_id": ds,
                    "outer_fold": fold,
                    "condition": cond,
                    "family": fam,
                    "severity": sev,
                    "status": res.status,
                    "spec_relpath": str(spec_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "shift_spec_file_sha256": spec_sha,
                })

            # Verify severity monotonicity for all 7 families
            for fam in SHIFT_FAMILIES:
                mild_key = f"{fam}_mild"
                severe_key = f"{fam}_severe"
                if mild_key in fold_results and severe_key in fold_results:
                    mono = verify_severity_monotonicity(fold_results[mild_key], fold_results[severe_key], fam)
                    monotonicity_records.append({
                        "dataset": ds,
                        "fold": fold,
                        "family": fam,
                        "status": mono["status"],
                        "metric_name": mono["metric_name"],
                        "mild_value": mono["mild_value"],
                        "severe_value": mono["severe_value"],
                        "details": mono["details"],
                    })

    total_time = time.perf_counter() - start_time
    print(f"\nGenerated {len(manifest_entries)} scenario entries across {len(target_datasets)} datasets in {total_time:.2f}s")

    # Save manifest
    manifest_doc = {
        "protocol_version": shift_cfg.get("protocol_version", 1),
        "total_scenarios": len(manifest_entries),
        "scenarios": manifest_entries,
    }
    manifest_path = shifts_data_dir / "shift_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2, sort_keys=True)
    print(f"Saved manifest: {manifest_path} (SHA-256: {compute_file_sha256(manifest_path)})")

    # Save audit CSVs
    df_audit = pd.DataFrame(audit_records)
    df_audit.to_csv(out_dir / "shift_audit.csv", index=False)
    print(f"Saved: {out_dir / 'shift_audit.csv'}")

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(out_dir / "scenario_summary.csv", index=False)
    print(f"Saved: {out_dir / 'scenario_summary.csv'}")

    df_mono = pd.DataFrame(monotonicity_records)
    df_mono.to_csv(out_dir / "severity_monotonicity.csv", index=False)
    print(f"Saved: {out_dir / 'severity_monotonicity.csv'}")

    failures_path = out_dir / "failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    print(f"Saved: {failures_path}")

    # Run benchmarks
    run_benchmark_and_equivalence(engine, prep_cfg, PROJECT_ROOT, out_dir)


if __name__ == "__main__":
    main()
