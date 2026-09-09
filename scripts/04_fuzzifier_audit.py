#!/usr/bin/env python
"""Phase 3.2 Fuzzifier Policy Audit and Validity Gate Runner.

Produces the required scientific audit artifacts:
1. results/baseline_validation/fuzzifier_objective_landscape.csv:
   Exact objective values J_m at separated initialization, post-convergence,
   and uniform grand centroid solution across m in [1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0] for s01.
2. results/baseline_validation/fuzzifier_m_grid_audit.csv:
   Exploratory source-only grid across [1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00, 2.25, 2.50]
   plus m_D = max(1.01, 1 + 2/D) on the Phase-3 validation panel.
3. results/baseline_validation/fuzzifier_policy_comparison.csv:
   Direct paired comparison of fixed m=2 vs dimension-adaptive m across all 11 validation datasets
   and 5 seeds (all runs in denominator).
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import yaml

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clusterdrift.shifts.hashing import atomic_write_csv

from clusterdrift.methods.fcm import FCM
from clusterdrift.methods.fuzzifier import resolve_fuzzifier
from clusterdrift.metrics.external import adjusted_rand_index
from clusterdrift.metrics.internal import (
    fuzzy_partition_coefficient,
    partition_entropy,
)

import importlib
vb = importlib.import_module("03_validate_baselines")

REAL_VALIDATION_DATASETS = vb.REAL_VALIDATION_DATASETS
SYNTHETIC_VALIDATION_DATASETS = vb.SYNTHETIC_VALIDATION_DATASETS
load_real_dataset_fold = vb.load_real_dataset_fold
load_synthetic_dataset_fold = vb.load_synthetic_dataset_fold
load_yaml = vb.load_yaml


def compute_grand_centroid_objective(X: np.ndarray, K: int, m: float) -> float:
    """Compute exact FCM objective J_m at the uniform grand centroid solution."""
    N, d = X.shape
    x_bar = np.mean(X, axis=0)
    centers_grand = np.tile(x_bar, (K, 1))
    diff = X[:, np.newaxis, :] - centers_grand[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    U_grand = np.full((N, K), 1.0 / K)
    return float(np.sum((U_grand ** m) * dist_sq))


def audit_s01_objective_landscape(out_dir: Path, prep_cfg: Dict[str, Any]) -> None:
    """Audit the objective landscape of s01_balanced_gmm to demonstrate fuzzifier collapse mechanism."""
    print("\n--- Auditing s01_balanced_gmm Objective Landscape ---")
    X_source, y_source, _, _, K = load_synthetic_dataset_fold(
        "s01_balanced_gmm", PROJECT_ROOT, prep_cfg, fold=0
    )
    N, D = X_source.shape

    m_values = [1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0]
    records = []

    for m_val in m_values:
        # 1. Fit FCM with kmeans++ initialization
        fcm = FCM(n_clusters=K, m=m_val, random_state=42, initialization="kmeans++", fuzzifier_policy="fixed")
        fcm.fit(X_source)

        J_init = fcm.objective_history_[0] if fcm.objective_history_ else None
        J_converged = fcm.objective_history_[-1] if fcm.objective_history_ else None
        J_grand = compute_grand_centroid_objective(X_source, K, m_val)

        delta_j = (J_converged - J_grand) if (J_converged is not None) else None
        grand_lower_than_converged = bool(delta_j > 1e-4) if (delta_j is not None) else False
        grand_lower_than_init = bool(J_grand < J_init) if J_init is not None else False
        converged_is_grand = bool(abs(J_converged - J_grand) / max(abs(J_grand), 1e-9) < 1e-4)

        diag = fcm.diagnostics_
        records.append({
            "dataset": "s01_balanced_gmm",
            "N": N,
            "D": D,
            "K": K,
            "m": m_val,
            "J_init_separated": round(J_init, 2) if J_init is not None else None,
            "J_converged": round(J_converged, 2) if J_converged is not None else None,
            "J_grand_centroid": round(J_grand, 2),
            "converged_minus_grand_objective": round(delta_j, 2) if delta_j is not None else None,
            "grand_centroid_lower_than_converged": grand_lower_than_converged,
            "grand_centroid_lower_than_init": grand_lower_than_init,
            "converged_is_grand_centroid": converged_is_grand,
            "converged_is_degenerate": fcm.degenerate_solution_,
            "FPC": round(diag.get("fpc", 0.0), 4),
            "FPC_floor_gap": round(diag.get("fpc_floor_gap", 0.0), 6),
            "min_center_distance": round(diag.get("min_center_distance", 0.0), 6),
            "normalized_min_center_distance": round(diag.get("normalized_min_center_distance", 0.0), 6),
            "effective_distinct_prototypes": diag.get("effective_distinct_prototypes", 0),
        })
        print(f"  m={m_val:4.2f} | J_init={J_init:9.2f} | J_conv={J_converged:9.2f} | J_grand={J_grand:9.2f} | degen={fcm.degenerate_solution_}")

    df = pd.DataFrame(records)
    out_path = out_dir / "fuzzifier_objective_landscape.csv"
    atomic_write_csv(out_path, df, index=False)
    print(f"Saved: {out_path}")


def audit_fuzzifier_policy_comparison(out_dir: Path, prep_cfg: Dict[str, Any], seeds: List[int]) -> None:
    """Generate paired comparison of fixed m=2 vs dimension-adaptive m across all validation datasets."""
    print("\n--- Running Fuzzifier Policy Comparison (Fixed m=2 vs Dimension-Adaptive) ---")
    all_datasets = [(d, "real") for d in REAL_VALIDATION_DATASETS] + [
        (d, "synthetic") for d in SYNTHETIC_VALIDATION_DATASETS
    ]

    records = []
    total_runs = len(all_datasets) * len(seeds)
    run_idx = 0

    for ds_id, ds_type in all_datasets:
        if ds_type == "real":
            X_source, y_source, _, _, K = load_real_dataset_fold(ds_id, PROJECT_ROOT, prep_cfg, fold=0)
        else:
            X_source, y_source, _, _, K = load_synthetic_dataset_fold(ds_id, PROJECT_ROOT, prep_cfg, fold=0)

        N, D = X_source.shape

        for seed in seeds:
            run_idx += 1
            # 1. Fixed m=2 (baseline control)
            fcm_fixed = FCM(n_clusters=K, m=2.0, random_state=seed, initialization="kmeans++", fuzzifier_policy="fixed")
            fcm_fixed.fit(X_source)
            fixed_diag = fcm_fixed.diagnostics_
            fixed_pred = fcm_fixed.predict(X_source)
            ari_fixed = float(adjusted_rand_index(y_source, fixed_pred))

            # 2. Dimension-adaptive m
            fcm_adapt = FCM(n_clusters=K, random_state=seed, initialization="kmeans++", fuzzifier_policy="dimension_adaptive")
            fcm_adapt.fit(X_source)
            adapt_diag = fcm_adapt.diagnostics_
            adapt_pred = fcm_adapt.predict(X_source)
            ari_adapt = float(adjusted_rand_index(y_source, adapt_pred))

            records.append({
                "dataset": ds_id,
                "D": D,
                "N": N,
                "K": K,
                "seed": seed,
                "fixed_m": fcm_fixed.effective_m_,
                "fixed_status": fcm_fixed.status_,
                "fixed_degenerate": fcm_fixed.degenerate_solution_,
                "fixed_FPC": round(fixed_diag.get("fpc", 0.0), 4),
                "fixed_PE": round(fixed_diag.get("pe", 0.0), 4),
                "fixed_center_separation": round(fixed_diag.get("normalized_min_center_distance", 0.0), 6),
                "fixed_objective": round(fcm_fixed.objective_history_[-1], 4) if fcm_fixed.objective_history_ else None,
                "adaptive_m": round(fcm_adapt.effective_m_, 4),
                "adaptive_status": fcm_adapt.status_,
                "adaptive_degenerate": fcm_adapt.degenerate_solution_,
                "adaptive_FPC": round(adapt_diag.get("fpc", 0.0), 4),
                "adaptive_PE": round(adapt_diag.get("pe", 0.0), 4),
                "adaptive_center_separation": round(adapt_diag.get("normalized_min_center_distance", 0.0), 6),
                "adaptive_objective": round(fcm_adapt.objective_history_[-1], 4) if fcm_adapt.objective_history_ else None,
                "ARI_fixed_evaluation_only": round(ari_fixed, 4),
                "ARI_adaptive_evaluation_only": round(ari_adapt, 4),
            })

            if run_idx % 11 == 0 or run_idx == total_runs:
                print(f"  [{run_idx}/{total_runs}] {ds_id} (D={D}, seed={seed}) => Fixed: {fcm_fixed.status_} (ARI={ari_fixed:.3f}) | Adaptive (m={fcm_adapt.effective_m_:.2f}): {fcm_adapt.status_} (ARI={ari_adapt:.3f})")

    df = pd.DataFrame(records)
    out_path = out_dir / "fuzzifier_policy_comparison.csv"
    atomic_write_csv(out_path, df, index=False)
    print(f"Saved: {out_path}")


def audit_fuzzifier_m_grid(out_dir: Path, prep_cfg: Dict[str, Any], seeds: List[int]) -> None:
    """Run exploratory source-only m grid on Phase-3 validation panel."""
    print("\n--- Running Exploratory Fuzzifier m Grid Audit ---")
    all_datasets = [(d, "real") for d in REAL_VALIDATION_DATASETS] + [
        (d, "synthetic") for d in SYNTHETIC_VALIDATION_DATASETS
    ]

    fixed_grid = [1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00, 2.25, 2.50]
    records = []

    # Run exploratory audit across seed 1
    seed = seeds[0]

    for ds_id, ds_type in all_datasets:
        if ds_type == "real":
            X_source, y_source, _, _, K = load_real_dataset_fold(ds_id, PROJECT_ROOT, prep_cfg, fold=0)
        else:
            X_source, y_source, _, _, K = load_synthetic_dataset_fold(ds_id, PROJECT_ROOT, prep_cfg, fold=0)

        N, D = X_source.shape
        m_dim_res = resolve_fuzzifier(policy="dimension_adaptive", n_features=D)
        m_dim = m_dim_res.effective_m

        candidates = list(fixed_grid)
        if m_dim not in candidates:
            candidates.append(m_dim)
        candidates.sort()

        for m_val in candidates:
            fcm = FCM(n_clusters=K, m=m_val, random_state=seed, initialization="kmeans++", fuzzifier_policy="fixed")
            fcm.fit(X_source)
            diag = fcm.diagnostics_
            pred = fcm.predict(X_source)
            ari = float(adjusted_rand_index(y_source, pred))

            records.append({
                "dataset": ds_id,
                "seed": seed,
                "D": D,
                "N": N,
                "K": K,
                "m": round(m_val, 4),
                "is_dimension_rule": bool(abs(m_val - m_dim) < 1e-5),
                "objective": round(fcm.objective_history_[-1], 4) if fcm.objective_history_ else None,
                "FPC": round(diag.get("fpc", 0.0), 4),
                "PE": round(diag.get("pe", 0.0), 4),
                "min_center_distance": round(diag.get("min_center_distance", 0.0), 6),
                "normalized_min_center_distance": round(diag.get("normalized_min_center_distance", 0.0), 6),
                "degenerate_solution": fcm.degenerate_solution_,
                "converged": fcm.converged_,
                "effective_clusters": diag.get("effective_distinct_prototypes", 0),
                "ARI_evaluation_only": round(ari, 4),
            })

        print(f"  Audit grid completed for {ds_id} (D={D}, {len(candidates)} candidates evaluated)")

    df = pd.DataFrame(records)
    out_path = out_dir / "fuzzifier_m_grid_audit.csv"
    atomic_write_csv(out_path, df, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fuzzifier policies and objective landscapes.")
    parser.add_argument("--landscape-only", action="store_true", help="Only audit and regenerate s01 objective landscape.")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3.2: FUZZIFIER POLICY FREEZE AND SOFT-MODEL VALIDITY GATE")
    print("=" * 70)

    prep_cfg = load_yaml(PROJECT_ROOT / "configs" / "preprocessing.yaml")
    methods_cfg = load_yaml(PROJECT_ROOT / "configs" / "methods.yaml")
    seeds = methods_cfg.get("algorithm_seeds", [1, 2, 3, 4, 5])

    out_dir = PROJECT_ROOT / "results" / "baseline_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    audit_s01_objective_landscape(out_dir, prep_cfg)
    if not args.landscape_only:
        audit_fuzzifier_policy_comparison(out_dir, prep_cfg, seeds)
        audit_fuzzifier_m_grid(out_dir, prep_cfg, seeds)
    total_time = time.perf_counter() - t0
    print(f"\nPhase 3.2 audit successfully finished in {total_time:.2f}s")


if __name__ == "__main__":
    main()
