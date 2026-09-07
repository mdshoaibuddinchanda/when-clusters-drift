"""Paired Bootstrap Analysis and Pre-Registered Mechanical Verdict for Phase 7."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_paired_unit_bootstrap(
    deltas: np.ndarray,
    n_repetitions: int = 10000,
    seed: int = 2026090707,
) -> Dict[str, float]:
    """Compute paired bootstrap on independent units (datasets or families).

    Returns 95% percentile intervals for both mean and median paired improvement.
    """
    arr = np.asarray(deltas, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot bootstrap empty array")

    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n, size=(n_repetitions, n))
    resamples = arr[indices]

    boot_means = np.mean(resamples, axis=1)
    boot_medians = np.median(resamples, axis=1)

    mean_est = float(np.mean(arr))
    median_est = float(np.median(arr))

    mean_ci_lower = float(np.percentile(boot_means, 2.5))
    mean_ci_upper = float(np.percentile(boot_means, 97.5))

    median_ci_lower = float(np.percentile(boot_medians, 2.5))
    median_ci_upper = float(np.percentile(boot_medians, 97.5))

    return {
        "n_units": n,
        "sample_mean": mean_est,
        "sample_median": median_est,
        "mean_ci_lower": mean_ci_lower,
        "mean_ci_upper": mean_ci_upper,
        "median_ci_lower": median_ci_lower,
        "median_ci_upper": median_ci_upper,
    }


def evaluate_falsification_verdict(
    mae_p0: float,
    mae_p3: float,
    mae_p4: float,
    dataset_deltas_04: np.ndarray,
    dataset_deltas_34: np.ndarray,
    family_deltas_04: np.ndarray,
    family_deltas_34: np.ndarray,
    bootstrap_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Mechanically evaluate the pre-registered falsification rules.

    Rules for SURVIVES_STRONGLY:
    LODO:
    1. MAE(P4) < MAE(P0)
    2. Either R_04 >= 0.05 OR 95% bootstrap CI lower bound of Delta_04 > 0
    3. MAE(P4) < MAE(P3)
    4. Either R_34 >= 0.02 OR 95% bootstrap CI lower bound of Delta_34 > 0
    5. median_d(Delta_04^(d)) > 0
    6. P4 beats P0 on >= 7/12 held-out datasets (Delta_04^(d) > 0)
    SHIFT-FAMILY:
    7. P4 beats P0 on >= 4/7 held-out shift families (Delta_04^(f) > 0)
    8. P4 beats P3 on >= 4/7 held-out shift families (Delta_34^(f) > 0)

    If all 8 hold: SURVIVES_STRONGLY
    If MAE(P4) < MAE(P0) and MAE(P4) < MAE(P3) but any of 2, 4, 5, 6, 7, 8 fail: BORDERLINE
    If MAE(P4) >= MAE(P0) or MAE(P4) >= MAE(P3): FAILS_PRIMARY_FALSIFICATION
    """
    # Relative improvements
    r_04 = (mae_p0 - mae_p4) / mae_p0 if mae_p0 > 0 else 0.0
    r_34 = (mae_p3 - mae_p4) / mae_p3 if mae_p3 > 0 else 0.0

    # Dataset-level counts
    n_datasets = len(dataset_deltas_04)
    dataset_wins_04 = int(np.sum(dataset_deltas_04 > 0))
    dataset_wins_34 = int(np.sum(dataset_deltas_34 > 0))
    dataset_median_04 = float(np.median(dataset_deltas_04))

    # Family-level counts
    n_families = len(family_deltas_04)
    family_wins_04 = int(np.sum(family_deltas_04 > 0))
    family_wins_34 = int(np.sum(family_deltas_34 > 0))

    # Bootstrap CI lower bounds (mean paired MAE improvement)
    boot_lodo_04_mean_lower = bootstrap_results.get("lodo_delta_04", {}).get("mean_ci_lower", -999.0)
    boot_lodo_34_mean_lower = bootstrap_results.get("lodo_delta_34", {}).get("mean_ci_lower", -999.0)

    # 8 Criteria
    c1 = bool(mae_p4 < mae_p0)
    c2 = bool((r_04 >= 0.05) or (boot_lodo_04_mean_lower > 0))
    c3 = bool(mae_p4 < mae_p3)
    c4 = bool((r_34 >= 0.02) or (boot_lodo_34_mean_lower > 0))
    c5 = bool(dataset_median_04 > 0)
    c6 = bool(dataset_wins_04 >= 7)  # 7 out of 12
    c7 = bool(family_wins_04 >= 4)   # 4 out of 7
    c8 = bool(family_wins_34 >= 4)   # 4 out of 7

    criteria = {
        "rule_1_lodo_mae_p4_lt_p0": c1,
        "rule_2_r04_ge_005_or_ci_gt_0": c2,
        "rule_3_lodo_mae_p4_lt_p3": c3,
        "rule_4_r34_ge_002_or_ci_gt_0": c4,
        "rule_5_median_dataset_delta_04_gt_0": c5,
        "rule_6_dataset_wins_04_ge_7_of_12": c6,
        "rule_7_family_wins_04_ge_4_of_7": c7,
        "rule_8_family_wins_34_ge_4_of_7": c8,
    }

    all_criteria_met = all([c1, c2, c3, c4, c5, c6, c7, c8])
    severe_family_generalization_failure = bool(family_wins_04 < 4)

    if all_criteria_met:
        verdict = "SURVIVES_STRONGLY"
    elif c1 and c3:
        verdict = "BORDERLINE"
    else:
        verdict = "FAILS_PRIMARY_FALSIFICATION"

    return {
        "verdict": verdict,
        "severe_family_generalization_failure_flag": severe_family_generalization_failure,
        "metrics": {
            "lodo_mae_p0": float(mae_p0),
            "lodo_mae_p3": float(mae_p3),
            "lodo_mae_p4": float(mae_p4),
            "r_04": float(r_04),
            "r_34": float(r_34),
            "dataset_median_delta_04": dataset_median_04,
            "dataset_wins_04": f"{dataset_wins_04}/{n_datasets}",
            "dataset_wins_34": f"{dataset_wins_34}/{n_datasets}",
            "family_wins_04": f"{family_wins_04}/{n_families}",
            "family_wins_34": f"{family_wins_34}/{n_families}",
            "bootstrap_lodo_04_mean_ci_lower": boot_lodo_04_mean_lower,
            "bootstrap_lodo_34_mean_ci_lower": boot_lodo_34_mean_lower,
        },
        "criteria": criteria,
    }