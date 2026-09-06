#!/usr/bin/env python
"""Phase 3 Baseline Validation Runner.

Executes and numerically validates 5 clustering baselines across a deterministic
validation panel of real and synthetic datasets for 5 algorithm seeds.
Outputs engineering audit artifacts:
- results/baseline_validation/validation_runs.csv
- results/baseline_validation/validation_summary.csv
- results/baseline_validation/failures.json
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clusterdrift.data.preprocess import build_preprocessor
from clusterdrift.methods.fcm import FCM
from clusterdrift.methods.gmm import GMM
from clusterdrift.methods.gustafson_kessel import GustafsonKessel
from clusterdrift.methods.kmeans import KMeans
from clusterdrift.methods.pfcm import PFCM
from clusterdrift.methods.utils import compute_methods_config_sha256
from clusterdrift.metrics.external import (
    adjusted_mutual_info,
    adjusted_rand_index,
    normalized_mutual_info,
)
from clusterdrift.metrics.internal import (
    fuzzy_partition_coefficient,
    partition_entropy,
    silhouette_metric,
    xie_beni_index,
)


REAL_VALIDATION_DATASETS = [
    "iris",
    "glass",
    "sonar",
    "madelon",
    "human_activity_recognition",
    "mice_protein_expression",
]

SYNTHETIC_VALIDATION_DATASETS = [
    "s01_balanced_gmm",
    "s02_overlap_gmm",
    "s04_heteroscedastic_gmm",
    "s06_high_dimensional_gmm",
    "s08_student_t_mixture",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_real_dataset_fold(
    dataset_id: str,
    project_root: Path,
    prep_cfg: Dict[str, Any],
    fold: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load canonical real dataset and fold 0, returning preprocessed source/target arrays and labels."""
    ds_dir = project_root / "data" / "canonical" / "controlled" / dataset_id
    features_path = ds_dir / "features.parquet"
    labels_path = ds_dir / "labels.parquet"
    meta_path = ds_dir / "metadata.json"

    df_X = pd.read_parquet(features_path)
    df_y = pd.read_parquet(labels_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    K = int(meta["n_classes"])
    feature_roles = meta.get("feature_roles", {col: "numeric" for col in df_X.columns})

    fold_path = project_root / "data" / "splits" / "controlled" / dataset_id / f"fold_{fold}.npz"
    with np.load(fold_path) as npz:
        source_idx = npz["source_indices"]
        target_idx = npz["target_indices"]

    X_source_raw = df_X.iloc[source_idx].copy()
    X_target_raw = df_X.iloc[target_idx].copy()
    y_source = df_y.iloc[source_idx].to_numpy().ravel()
    y_target = df_y.iloc[target_idx].to_numpy().ravel()

    # Preprocess strictly on source
    prep = build_preprocessor(feature_roles=feature_roles, config=prep_cfg, metadata=meta)
    prep.fit(X_source_raw)
    X_source = prep.transform(X_source_raw)
    X_target = prep.transform(X_target_raw)

    return X_source, y_source, X_target, y_target, K


def load_synthetic_dataset_fold(
    dataset_id: str,
    project_root: Path,
    prep_cfg: Dict[str, Any],
    fold: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load synthetic dataset, returning preprocessed source/target arrays and labels."""
    ds_dir = project_root / "data" / "synthetic" / dataset_id
    features_path = ds_dir / "features.parquet"
    labels_path = ds_dir / "hard_labels.parquet"
    meta_path = ds_dir / "metadata.json"

    df_X = pd.read_parquet(features_path)
    df_y = pd.read_parquet(labels_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    K = int(meta["K"])
    feature_roles = {col: "numeric" for col in df_X.columns}

    N = len(df_X)
    # Deterministic 80/20 train/test partition for synthetic validation
    rng = np.random.default_rng(20260907 + fold)
    perm = rng.permutation(N)
    n_source = int(0.8 * N)
    source_idx = perm[:n_source]
    target_idx = perm[n_source:]

    X_source_raw = df_X.iloc[source_idx].copy()
    X_target_raw = df_X.iloc[target_idx].copy()
    y_source = df_y.iloc[source_idx].to_numpy().ravel()
    y_target = df_y.iloc[target_idx].to_numpy().ravel()

    prep = build_preprocessor(feature_roles=feature_roles, config=prep_cfg, metadata=meta)
    prep.fit(X_source_raw)
    X_source = prep.transform(X_source_raw)
    X_target = prep.transform(X_target_raw)

    return X_source, y_source, X_target, y_target, K


def instantiate_method(
    method_name: str,
    K: int,
    seed: int,
    methods_cfg: Dict[str, Any],
) -> Any:
    """Instantiate clustering method with locked configuration."""
    if method_name == "kmeans":
        c = methods_cfg.get("kmeans", {})
        return KMeans(
            n_clusters=K,
            random_state=seed,
            n_init=c.get("n_init", 10),
            max_iter=c.get("max_iter", 300),
            tol=float(c.get("tol", 1e-4)),
            algorithm=c.get("algorithm", "lloyd"),
        )
    elif method_name == "fcm":
        c = methods_cfg.get("fcm", {})
        return FCM(
            n_clusters=K,
            m=float(c.get("m", 2.0)),
            random_state=seed,
            max_iter=c.get("max_iter", 150),
            tol=float(c.get("tol", 1e-5)),
        )
    elif method_name == "gmm":
        c = methods_cfg.get("gmm", {})
        return GMM(
            n_clusters=K,
            random_state=seed,
            covariance_type=c.get("covariance_type", "full"),
            n_init=c.get("n_init", 1),
            reg_covar=float(c.get("reg_covar", 1e-6)),
            max_iter=c.get("max_iter", 100),
            tol=float(c.get("tol", 1e-3)),
        )
    elif method_name == "pfcm":
        c = methods_cfg.get("pfcm", {})
        return PFCM(
            n_clusters=K,
            a=float(c.get("a", 1.0)),
            b=float(c.get("b", 1.0)),
            m=float(c.get("m", 2.0)),
            eta=float(c.get("eta", 2.0)),
            k_scale=float(c.get("k_scale", 1.0)),
            random_state=seed,
            max_iter=c.get("max_iter", 150),
            tol=float(c.get("tol", 1e-5)),
        )
    elif method_name == "gustafson_kessel":
        c = methods_cfg.get("gustafson_kessel", {})
        return GustafsonKessel(
            n_clusters=K,
            m=float(c.get("m", 2.0)),
            random_state=seed,
            max_iter=c.get("max_iter", 150),
            tol=float(c.get("tol", 1e-5)),
            min_eig=float(c.get("min_eig", 1e-6)),
            max_cond=float(c.get("max_cond", 1e8)),
            ridge_factor=float(c.get("ridge_factor", 1e-5)),
        )
    else:
        raise ValueError(f"Unknown method name: {method_name}")


def main() -> None:
    print("=" * 70)
    print("PHASE 3: BASELINE REPRODUCTION AND NUMERICAL VALIDATION")
    print("=" * 70)

    methods_cfg_path = PROJECT_ROOT / "configs" / "methods.yaml"
    prep_cfg_path = PROJECT_ROOT / "configs" / "preprocessing.yaml"

    methods_cfg = load_yaml(methods_cfg_path)
    prep_cfg = load_yaml(prep_cfg_path)

    methods_config_sha = compute_methods_config_sha256(methods_cfg)
    print(f"Methods Config SHA-256: {methods_config_sha}")

    algorithm_seeds: List[int] = methods_cfg.get("algorithm_seeds", [1, 2, 3, 4, 5])
    methods_list = ["kmeans", "fcm", "gmm", "pfcm", "gustafson_kessel"]

    out_dir = PROJECT_ROOT / "results" / "baseline_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_records: List[Dict[str, Any]] = []
    failures_records: List[Dict[str, Any]] = []

    total_combinations = (len(REAL_VALIDATION_DATASETS) + len(SYNTHETIC_VALIDATION_DATASETS)) * len(methods_list) * len(algorithm_seeds)
    print(f"Total Validation Runs to Execute: {total_combinations}")

    run_idx = 0
    t0_all = time.perf_counter()

    all_datasets = [(d, "real") for d in REAL_VALIDATION_DATASETS] + [(d, "synthetic") for d in SYNTHETIC_VALIDATION_DATASETS]

    for ds_id, ds_type in all_datasets:
        print(f"\n--- Loading Dataset: {ds_id} ({ds_type}) ---")
        if ds_type == "real":
            X_source, y_source, X_target, y_target, K = load_real_dataset_fold(
                ds_id, PROJECT_ROOT, prep_cfg, fold=0
            )
        else:
            X_source, y_source, X_target, y_target, K = load_synthetic_dataset_fold(
                ds_id, PROJECT_ROOT, prep_cfg, fold=0
            )

        print(f"  Samples: N_source={len(X_source)}, N_target={len(X_target)}, Features={X_source.shape[1]}, K={K}")

        for method_name in methods_list:
            for seed in algorithm_seeds:
                run_idx += 1
                model = instantiate_method(method_name, K, seed, methods_cfg)

                t0 = time.perf_counter()
                status = "SUCCESS"
                warnings = []
                try:
                    # Model fit strictly unsupervised on X_source
                    model.fit(X_source)
                    status = model.status_
                    warnings = list(model.warnings_)
                except Exception as exc:
                    status = "NUMERICAL_FAILURE"
                    warnings.append(str(exc))
                    failures_records.append({
                        "dataset": ds_id,
                        "fold": 0,
                        "method": method_name,
                        "seed": seed,
                        "error": str(exc),
                    })

                runtime_sec = time.perf_counter() - t0

                # Predictions & Metrics computed post-fit
                if status in ("SUCCESS", "MAX_ITER_REACHED"):
                    pred_labels = model.predict(X_source)
                    ari = adjusted_rand_index(y_source, pred_labels)
                    nmi = normalized_mutual_info(y_source, pred_labels)
                    ami = adjusted_mutual_info(y_source, pred_labels)
                    sil = silhouette_metric(X_source, pred_labels)

                    # Fuzzy / Soft metrics
                    if method_name in ("fcm", "gmm", "pfcm", "gustafson_kessel"):
                        U = model.membership_
                        fpc = fuzzy_partition_coefficient(U)
                        pe = partition_entropy(U)
                    else:
                        fpc = None
                        pe = None

                    if method_name in ("fcm", "pfcm", "gustafson_kessel"):
                        xb = xie_beni_index(X_source, model.membership_, model.cluster_centers_)
                    else:
                        xb = None

                    obj_final = model.objective_history_[-1] if model.objective_history_ else None
                    converged = bool(model.converged_)
                    n_iter = int(model.n_iter_)
                else:
                    ari = None
                    nmi = None
                    ami = None
                    sil = None
                    fpc = None
                    pe = None
                    xb = None
                    obj_final = None
                    converged = False
                    n_iter = 0

                record = {
                    "dataset": ds_id,
                    "fold": 0,
                    "method": method_name,
                    "algorithm_seed": seed,
                    "K": K,
                    "status": status,
                    "converged": converged,
                    "iterations": n_iter,
                    "runtime_seconds": round(runtime_sec, 4),
                    "objective_final": round(obj_final, 6) if obj_final is not None else None,
                    "ARI": round(ari, 4) if ari is not None else None,
                    "NMI": round(nmi, 4) if nmi is not None else None,
                    "AMI": round(ami, 4) if ami is not None else None,
                    "silhouette": round(sil, 4) if sil is not None else None,
                    "FPC": round(fpc, 4) if fpc is not None else None,
                    "PE": round(pe, 4) if pe is not None else None,
                    "XB": round(xb, 4) if xb is not None else None,
                    "warnings": "; ".join(warnings) if warnings else "",
                }
                runs_records.append(record)

                if run_idx % 25 == 0 or run_idx == total_combinations:
                    print(f"  [{run_idx}/{total_combinations}] {ds_id} - {method_name} - seed {seed} => {status} (ARI={record['ARI']}, time={runtime_sec:.3f}s)")

    total_time = time.perf_counter() - t0_all
    print(f"\nCompleted {len(runs_records)} runs in {total_time:.2f}s")

    # Save validation runs
    df_runs = pd.DataFrame(runs_records)
    runs_path = out_dir / "validation_runs.csv"
    df_runs.to_csv(runs_path, index=False)
    print(f"Saved: {runs_path}")

    # Build summary
    summary_rows = []
    for method_name in methods_list:
        sub = df_runs[df_runs["method"] == method_name]
        total_m = len(sub)
        conv_m = int(sub["converged"].sum())
        succ_m = int((sub["status"] == "SUCCESS").sum())
        mean_ari = float(sub["ARI"].dropna().mean()) if not sub["ARI"].dropna().empty else 0.0
        mean_time = float(sub["runtime_seconds"].mean())

        summary_rows.append({
            "method": method_name,
            "total_runs": total_m,
            "success_count": succ_m,
            "convergence_rate": round(conv_m / total_m, 4) if total_m > 0 else 0.0,
            "mean_ARI": round(mean_ari, 4),
            "mean_runtime_seconds": round(mean_time, 4),
            "methods_config_sha256": methods_config_sha,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "validation_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Save failures
    failures_path = out_dir / "failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures_records, f, indent=2)
    print(f"Saved: {failures_path} (Total failures: {len(failures_records)})")

    print("\nSummary Table:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
