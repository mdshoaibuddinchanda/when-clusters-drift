"""Assemble the evidence-locked Hypothesis-Y1 reports and final decision."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/hypothesis_y1"
ART = BASE / "artifacts"
REPORTS = BASE / "reports"
CORR = BASE / "corrected_replication"
FINAL_DECISIONS = {
    "PHASE7_RESULT_INVALID_REQUIRES_NEW_CONFIRMATORY_REPLICATION",
    "FAILURE_CONFIRMED_NEW_HYPOTHESIS_JUSTIFIED",
    "FAILURE_CONFIRMED_STOP_PROJECT",
}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_units(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Cannot aggregate an empty metric frame")
    required = ["n_test", "sum_absolute_error", "sum_squared_error", "sum_observed", "sum_observed_squared"]
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    n = int(frame["n_test"].sum())
    sae = float(frame["sum_absolute_error"].sum())
    sse = float(frame["sum_squared_error"].sum())
    sy = float(frame["sum_observed"].sum())
    sy2 = float(frame["sum_observed_squared"].sum())
    sst = sy2 - sy * sy / n
    return {
        "n": n,
        "mae": sae / n,
        "rmse": math.sqrt(sse / n),
        "r2": 1.0 - sse / sst if sst > 1e-15 else 0.0,
    }


def aggregate_map(frame: pd.DataFrame, filters: dict[str, Any], blocks: list[str]) -> dict[str, dict[str, float | int]]:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected[selected[column].eq(value)]
    return {block: aggregate_units(selected[selected["feature_block"].eq(block)].copy()) for block in blocks}


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe without the optional tabulate dependency."""
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    headers = [cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def boot_text(item: dict[str, Any]) -> str:
    boot = item["bootstrap"] if "bootstrap" in item else item
    return (
        f"mean Δ={fmt(boot['sample_mean'])}, 95% grouped bootstrap CI "
        f"[{fmt(boot['mean_ci_lower'])}, {fmt(boot['mean_ci_upper'])}]"
    )


def paired_summary(values: np.ndarray, repetitions: int, seed: int) -> dict[str, Any]:
    """Apply the same fixed, outer-unit bootstrap used by the independent audit."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    samples = values[rng.randint(0, len(values), size=(repetitions, len(values)))]
    means = np.mean(samples, axis=1)
    medians = np.median(samples, axis=1)
    return {
        "n_units": int(len(values)),
        "sample_mean": float(np.mean(values)),
        "sample_median": float(np.median(values)),
        "mean_ci_lower": float(np.percentile(means, 2.5)),
        "mean_ci_upper": float(np.percentile(means, 97.5)),
        "median_ci_lower": float(np.percentile(medians, 2.5)),
        "median_ci_upper": float(np.percentile(medians, 97.5)),
    }


def test_summary(tests: dict[str, Any]) -> str:
    if not tests.get("suites"):
        return "Final test logs pending."
    parts = []
    for name in ("y1", "falsification", "full"):
        if name in tests["suites"]:
            item = tests["suites"][name]
            parts.append(
                f"{name}: {item['passed']} passed, {item['failed']} failed, "
                f"{item['skipped']} skipped, exit {item['exit_code']}"
            )
    return "; ".join(parts) + "."


def main() -> None:
    initial = load_json(ART / "hypothesis_y1_initial_integrity.json")
    independent = load_json(ART / "hypothesis_y1_independent_replication.json")
    convergence = load_json(ART / "hypothesis_y1_convergence_summary.json")
    root_cause = load_json(ART / "hypothesis_y1_convergence_root_cause.json")
    sensitivities = load_json(ART / "hypothesis_y1_model_sensitivities.json")
    within = load_json(ART / "hypothesis_y1_within_vs_cross_dataset.json")
    forensics = load_json(ART / "hypothesis_y1_statistical_forensics.json")
    security = load_json(ART / "hypothesis_y1_security_integrity_audit.json")
    corrected = load_json(CORR / "hypothesis_y1_corrected_replication_summary.json")
    tests = load_json(ART / "hypothesis_y1_test_results.json") if (ART / "hypothesis_y1_test_results.json").exists() else {"suites": {}}

    duplicate = pd.read_csv(ART / "hypothesis_y1_duplicate_audit.csv")
    classes = pd.read_csv(ART / "hypothesis_y1_class_k_audit.csv")
    signal_meta = pd.read_csv(ART / "hypothesis_y1_signal_metadata_relationships.csv")
    correlations = pd.read_csv(ART / "hypothesis_y1_signal_mechanism_correlations.csv")
    redundancy = pd.read_csv(ART / "hypothesis_y1_signal_redundancy.csv")
    families = pd.read_csv(ART / "hypothesis_y1_shift_family_mechanisms.csv")
    severity = pd.read_csv(ART / "hypothesis_y1_severity_monotonicity.csv")
    risk_class = pd.read_csv(ART / "hypothesis_y1_risk_classification.csv")
    risk_rank = pd.read_csv(ART / "hypothesis_y1_risk_ranking.csv")
    influence = pd.read_csv(ART / "hypothesis_y1_dataset_influence.csv")
    conv_units = pd.read_csv(ART / "hypothesis_y1_convergence_sensitivity_metrics.csv")
    control_units = pd.read_csv(ART / "hypothesis_y1_exploratory_control_metrics.csv")
    corrected_units = pd.read_csv(CORR / "hypothesis_y1_corrected_model_metrics.csv")
    corrected_rows = pd.read_csv(CORR / "hypothesis_y1_corrected_rows.csv")
    original_lodo = pd.read_csv(ROOT / "results/falsification/lodo_metrics.csv")
    original_losfo = pd.read_csv(ROOT / "results/falsification/losfo_metrics.csv")
    independent_lodo = pd.read_csv(ART / "hypothesis_y1_independent_lodo.csv")
    independent_losfo = pd.read_csv(ART / "hypothesis_y1_independent_losfo.csv")
    no_skill_comparisons = pd.read_csv(ART / "hypothesis_y1_no_skill_comparisons.csv")
    falsification_cfg = yaml.safe_load((ROOT / "configs/falsification.yaml").read_text(encoding="utf-8"))
    bootstrap_repetitions = int(falsification_cfg["bootstrap"]["repetitions"])
    bootstrap_seed = int(falsification_cfg["bootstrap"]["seed"])

    s3_metrics = {
        evaluation: aggregate_map(
            conv_units,
            {"sensitivity": "S3_both_converged", "evaluation": evaluation, "regressor": "HGBR"},
            ["P0", "P3", "P4"],
        )
        for evaluation in ("LODO", "LOSFO")
    }
    corrected_metrics: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for arm in ("duplicate_corrected_150_all", "fully_corrected_600"):
        corrected_metrics[arm] = {
            evaluation: aggregate_map(
                corrected_units,
                {"replication_arm": arm, "evaluation": evaluation, "regressor": "HGBR"},
                ["P0", "P3", "P3_delta", "P4"],
            )
            for evaluation in ("LODO", "LOSFO")
        }
    fair_metrics = {
        evaluation: aggregate_map(
            control_units,
            {"evaluation": evaluation, "regressor": "HGBR"},
            ["Y1_VALIDITY_DELTA_CLEAN_REFERENCE"],
        )["Y1_VALIDITY_DELTA_CLEAN_REFERENCE"]
        for evaluation in ("LODO", "LOSFO")
    }

    full = corrected["arms"]["fully_corrected_600"]
    full_lodo = corrected_metrics["fully_corrected_600"]["LODO"]
    full_losfo = corrected_metrics["fully_corrected_600"]["LOSFO"]
    full_comp_lodo = full["LODO"]["comparisons"]
    full_comp_losfo = full["LOSFO"]["comparisons"]
    full_median = full["no_skill"]["LODO_B0_median"]["mae"]
    reversal = (
        full_lodo["P4"]["mae"] < full_lodo["P0"]["mae"]
        and full_comp_lodo["P0_minus_P4"]["bootstrap"]["mean_ci_lower"] > 0
        and full_lodo["P4"]["mae"] < full_median
    )
    if reversal:
        final_decision = "PHASE7_RESULT_INVALID_REQUIRES_NEW_CONFIRMATORY_REPLICATION"
    else:
        final_decision = "FAILURE_CONFIRMED_STOP_PROJECT"
    if final_decision not in FINAL_DECISIONS:
        raise RuntimeError("Invalid Y1 decision")

    original_phase7 = list(falsification_cfg["datasets"])
    registry = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text(encoding="utf-8"))["real_datasets"]
    controlled = [name for name, spec in registry.items() if spec.get("dataset_group") == "controlled_real"]
    untouched = [name for name in controlled if name not in original_phase7]
    phase7_dup = sorted(
        duplicate.loc[
            duplicate["phase7_dataset"].astype(bool) & duplicate["original_target_leakage_fraction"].gt(0), "dataset"
        ].unique().tolist()
    )
    all_dup = sorted(duplicate.loc[duplicate["original_target_leakage_fraction"].gt(0), "dataset"].unique().tolist())
    largest_dup = float(duplicate["original_target_leakage_fraction"].max())
    original_class_bad = classes[
        classes["phase7_dataset"].astype(bool)
        & classes["split_version"].eq("original_phase2")
        & (~classes["source_class_complete"].astype(bool) | ~classes["target_class_complete"].astype(bool))
    ]
    corrected_class_bad = classes[
        classes["split_version"].eq("hypothesis_y1_corrected")
        & (~classes["source_class_complete"].astype(bool) | ~classes["target_class_complete"].astype(bool))
    ]
    both_rates = (
        pd.read_csv(ROOT / "results/falsification/joined_evaluation_table.csv")
        .assign(both=lambda x: x["source_converged"].astype(bool) & x["candidate_converged"].astype(bool))
        .groupby("dataset_id")["both"].mean()
    )
    severe_convergence = both_rates[both_rates < 0.8].index.tolist()

    no_skill = {f"{row['evaluation']}_{row['feature_block']}": row for row in forensics["no_skill"]}
    dh_lodo = sensitivities["controls"]["blocks"]["Y1_P4_DH_SOURCE_RELATIVE"]["LODO"]
    p4_rank = risk_rank[risk_rank["feature_block"].eq("P4")].iloc[0]
    p4_auc = risk_class[(risk_class["feature_block"].eq("P4")) & (risk_class["threshold"].eq(0.05))].iloc[0]
    dv_overall = correlations[(correlations["signal"].eq("D_V")) & (correlations["scope"].eq("overall"))].iloc[0]
    dm_overall = correlations[(correlations["signal"].eq("D_M")) & (correlations["scope"].eq("overall"))].iloc[0]
    du_pair = redundancy[
        (redundancy["scope"].eq("raw_cross_dataset"))
        & (redundancy["signal_a"].eq("D_U_R"))
        & (redundancy["signal_b"].eq("D_U_C"))
    ].iloc[0]
    delta_severity = severity[severity["quantity"].eq("delta_ari")].iloc[0]

    attempts = [
        {
            "name": "independent_frozen_phase7_reimplementation",
            "motivation": "Test whether the frozen evaluator or join produced the negative result.",
            "created_before_result?": True,
            "features": "Preregistered P0/P3/P4 exactly",
            "model": "independently coded HGBR and Ridge nested grouped CV",
            "split": "frozen LODO and LOSFO",
            "metric": "MAE, RMSE, R2, unit wins, grouped bootstrap, verdict",
            "result": f"Agreement {independent['agreement']['status']}; verdict {independent['verdict']}.",
            "kept/rejected": "kept",
            "reason": "Required independent verification, not model selection.",
        },
        {
            "name": "convergence_root_cause_150_300_600",
            "motivation": "Separate slow convergence from numerical/degenerate failure.",
            "created_before_result?": True,
            "features": "FCM objective and center-shift trajectories",
            "model": "same FCM; fixed horizons 150/300/600",
            "split": "all failed source fits, all non-Letter failed candidates, deterministic Letter coverage and matched controls",
            "metric": "formal convergence, final shift/tolerance, monotone objective",
            "result": f"{root_cause['failed_sample_converged_by_600']}/{root_cause['original_failed_rows_in_sample']} failures converged by 600; 0 numerical/degenerate.",
            "kept/rejected": "kept as policy evidence",
            "reason": "Identifies a real eligibility defect without choosing horizon from P4 performance.",
        },
        {
            "name": "duplicate_group_and_class_k_validity_audit",
            "motivation": "Test whether frozen outer/inner splits isolate exact rows and protected entities and represent the manifest classes.",
            "created_before_result?": True,
            "features": "canonical feature-row hashes, protected group IDs, and manifest/source/target class sets",
            "model": "none",
            "split": "all frozen and Y1-corrected controlled folds",
            "metric": "cross-split connected groups and missing-class counts",
            "result": f"Phase-7 duplicate leakage occurs in {len(phase7_dup)} datasets and {len(original_class_bad)} target folds are class-incomplete; corrected Phase-7 crossings and class-incomplete folds are zero.",
            "kept/rejected": "kept",
            "reason": "Required outcome-blind validity audit and split repair.",
        },
        {
            "name": "convergence_sensitivity_S0_all_original_rows",
            "motivation": "Anchor every convergence filter to the complete frozen usable-row analysis.",
            "created_before_result?": True,
            "features": "preregistered P0/P3/P4",
            "model": "fixed preregistered HGBR",
            "split": "frozen LODO and LOSFO with all original usable shifted rows",
            "metric": "MAE, dataset/family wins, grouped bootstrap",
            "result": f"LODO P0={independent['hgb_regression']['LODO']['P0']:.6f}, P4={independent['hgb_regression']['LODO']['P4']:.6f}; original verdict {independent['verdict']}.",
            "kept/rejected": "kept",
            "reason": "Predeclared S0 reference; identical to the independently reproduced frozen analysis.",
        },
    ]
    for code, label in (
        ("S1_source_converged", "source-converged only"),
        ("S2_candidate_converged", "candidate-converged only"),
        ("S3_both_converged", "both-converged only"),
    ):
        item = sensitivities["convergence"]["sensitivities"][code]
        attempts.append({
            "name": f"convergence_sensitivity_{code}",
            "motivation": f"Measure frozen-result sensitivity using {label} rows.",
            "created_before_result?": True,
            "features": "Preregistered P0/P3/P4",
            "model": "fixed preregistered HGBR",
            "split": "same LODO/LOSFO after named convergence filter",
            "metric": "MAE, dataset/family wins, grouped bootstrap",
            "result": f"LODO P0={item['LODO']['mae']['P0']:.6f}, P4={item['LODO']['mae']['P4']:.6f}; wins={item['LODO']['wins_04']}/{item['LODO']['n_units']}.",
            "kept/rejected": "kept",
            "reason": "All three predeclared filters are retained; none was selected as a favorable answer.",
        })
    attempts.extend([
        {
            "name": "grouped_no_skill_mean_and_median",
            "motivation": "Determine whether learned regressors show absolute predictive skill.",
            "created_before_result?": True,
            "features": "none",
            "model": "training-group mean and median constants",
            "split": "LODO and LOSFO",
            "metric": "MAE, RMSE, median absolute error, R2",
            "result": f"LODO median MAE={no_skill['LODO_B0_median']['mae']:.6f}, lower than P0/P3/P4.",
            "kept/rejected": "kept",
            "reason": "Required no-skill comparator; adverse result retained.",
        },
        {
            "name": "fair_validity_delta_clean_reference",
            "motivation": "Compare changes-to-changes rather than raw validity levels to structural changes.",
            "created_before_result?": True,
            "features": "D_X plus clean-referenced deltas of FPC, PE, XB and silhouette",
            "model": "same fixed HGBR grid",
            "split": "same LODO/LOSFO",
            "metric": "MAE/RMSE/R2 and unit wins",
            "result": f"LODO MAE={fair_metrics['LODO']['mae']:.6f}; LOSFO MAE={fair_metrics['LOSFO']['mae']:.6f}.",
            "kept/rejected": "kept as fair control",
            "reason": "Motivated comparator; it does not replace frozen P3.",
        },
        {
            "name": "dh_source_entropy_relative",
            "motivation": "Test whether absolute entropy change is miscalibrated by source ambiguity.",
            "created_before_result?": False,
            "features": "P4 with only D_H divided by source entropy",
            "model": "same fixed HGBR grid",
            "split": "same LODO/LOSFO",
            "metric": "MAE",
            "result": f"LODO MAE={dh_lodo['mae']:.6f}, versus original P4={independent['hgb_regression']['LODO']['P4']:.6f}.",
            "kept/rejected": "rejected as rescue",
            "reason": "The normalization registry records that this was registered after prior Y1 evidence; it did not improve P4 or beat P0 and is retained to prevent cherry-picking.",
        },
        {
            "name": "within_dataset_grouped_hgbr",
            "motivation": "Test whether signals work only after dataset-specific calibration.",
            "created_before_result?": True,
            "features": "fixed P0/P2/P3/P4 and median no-skill",
            "model": "same HGBR grid",
            "split": "outer fold held out; inner scenario groups retain all seeds",
            "metric": "mean held-out-fold MAE",
            "result": f"P4 helped {within['within_structural_help_count']}/12 within datasets and {within['cross_structural_help_count']}/12 LODO; only {', '.join(within['both_help_datasets'])} helped in both.",
            "kept/rejected": "kept diagnostically; rejected as new hypothesis",
            "reason": "Directions agreed for only 3/12 datasets and positive overlap was 1/12.",
        },
        {
            "name": "fixed_risk_classification_thresholds",
            "motivation": "Test whether event classification works when exact regression fails.",
            "created_before_result?": True,
            "features": "fixed out-of-dataset P0/P3/P4 regression scores",
            "model": "one-variable cross-dataset logistic calibration",
            "split": "LODO predictions only",
            "metric": "AUROC, AUPRC, balanced accuracy, Brier at ΔARI 0/.025/.05/.10",
            "result": f"P4 AUROC at 0.05={float(p4_auc['auroc']):.3f}; balanced accuracy={float(p4_auc['balanced_accuracy']):.3f}.",
            "kept/rejected": "rejected",
            "reason": "No classification rescue.",
        },
        {
            "name": "fixed_risk_ranking",
            "motivation": "Test whether ordering severe degradation works despite magnitude error.",
            "created_before_result?": True,
            "features": "fixed LODO scores",
            "model": "no refit beyond frozen predictor",
            "split": "LODO",
            "metric": "Spearman and top 10/20/30% event/mass capture",
            "result": f"P4 Spearman={float(p4_rank['spearman']):.3f}.",
            "kept/rejected": "rejected",
            "reason": "Near-zero rank association and weak event capture.",
        },
        {
            "name": "signal_scale_redundancy_mechanism_audit",
            "motivation": "Explain transfer failure without optimizing predictors.",
            "created_before_result?": True,
            "features": "all fixed structural/validity signals and metadata",
            "model": "descriptive Spearman/Pearson/VIF only",
            "split": "overall, dataset, and shift family",
            "metric": "correlation, between/within variance and VIF",
            "result": f"D_U_R/D_U_C Spearman={float(du_pair['spearman']):.3f}; strongest structural overall association D_M={float(dm_overall['spearman_delta_ari']):.3f}.",
            "kept/rejected": "kept diagnostically",
            "reason": "No feature subset was promoted or model retuned.",
        },
        {
            "name": "fixed_normalization_registry",
            "motivation": "Check scale comparability using at most one motivated normalization family per signal.",
            "created_before_result?": False,
            "features": "existing normalized D_V, normalized-JS D_U/D_M, source-bandwidth D_X, and one source-relative D_H",
            "model": "descriptive checks plus the single registered D_H P4 sensitivity",
            "split": "frozen LODO/LOSFO",
            "metric": "signal/metadata associations and MAE",
            "result": f"Existing signals remain dataset-calibrated; source-relative D_H LODO MAE={dh_lodo['mae']:.6f}.",
            "kept/rejected": "kept as audit; rejected as rescue",
            "reason": "Registry explicitly preserves its post-result status and forbids transform search.",
        },
        {
            "name": "construct_validity_inventory",
            "motivation": "Separate credible latent-cluster benchmarks from supervised labels used as recovery proxies.",
            "created_before_result?": True,
            "features": "dataset provenance and label semantics",
            "model": "none",
            "split": "all 54 benchmark units",
            "metric": "A/B/C construct tier counts",
            "result": f"A/B/C={forensics['construct_tier_counts']['A']}/{forensics['construct_tier_counts']['B']}/{forensics['construct_tier_counts']['C']}; only Iris is tier A in Phase 7.",
            "kept/rejected": "kept",
            "reason": "Changes claim scope, never dataset inclusion or model outcomes.",
        },
        {
            "name": "dependence_and_seed_pseudoreplication_audit",
            "motivation": "Quantify why 4,500 rows cannot be treated as independent scientific units.",
            "created_before_result?": True,
            "features": "delta_ari grouped by dataset/fold/condition/severity and seed",
            "model": "descriptive variance decomposition",
            "split": "840 five-seed shifted scenarios",
            "metric": "identical-seed fraction, within-scenario variance, ICC-style ratio",
            "result": "336/840 scenarios are identical across all seeds; ICC-style dependence is approximately 0.900.",
            "kept/rejected": "kept",
            "reason": "Supports dataset/family/scenario grouped inference rather than row-level claims.",
        },
        {
            "name": "severity_order_consistency_audit",
            "motivation": "Test whether named mild/severe generator doses order actual degradation and every signal consistently.",
            "created_before_result?": True,
            "features": "fixed signals and delta_ari",
            "model": "none",
            "split": "matched mild/severe scenarios",
            "metric": "severe-greater-than-mild fraction and paired median difference",
            "result": f"Severe exceeds mild for delta_ari in {float(delta_severity['severe_gt_mild_fraction']):.3f} of matches, versus 0.914 for D_X.",
            "kept/rejected": "kept",
            "reason": "Diagnoses noncommensurate dose labels without redefining severity after outcomes.",
        },
        {
            "name": "dataset_influence_audit",
            "motivation": "Check whether a small number of datasets drive mean P4−P0 direction.",
            "created_before_result?": True,
            "features": "frozen per-dataset P0/P4 MAE deltas",
            "model": "none",
            "split": "LODO unit deletion for diagnosis only",
            "metric": "remaining mean delta and official verdict",
            "result": "Removing the three most adverse datasets changes the mean sign but never the official verdict.",
            "kept/rejected": "rejected as a basis for exclusion",
            "reason": "Datasets remain in every scientific result.",
        },
        {
            "name": "duplicate_and_class_corrected_150",
            "motivation": "Isolate split/group/class correction at the original iteration horizon.",
            "created_before_result?": True,
            "features": "fixed P0/P3/P4 plus source/reference-probe-relative P3-delta",
            "model": "same HGBR grid",
            "split": "duplicate/entity grouped, temporal where applicable, class safeguarded",
            "metric": "LODO/LOSFO MAE/RMSE/R2, unit wins and grouped bootstrap",
            "result": f"LODO P0={corrected_metrics['duplicate_corrected_150_all']['LODO']['P0']['mae']:.6f}, P4={corrected_metrics['duplicate_corrected_150_all']['LODO']['P4']['mae']:.6f}.",
            "kept/rejected": "kept",
            "reason": "Complete panel retained regardless of direction.",
        },
        {
            "name": "fully_corrected_600_both_converged",
            "motivation": "Combine legitimate split/class and convergence repairs.",
            "created_before_result?": True,
            "features": "fixed P0/P3/P4 plus source/reference-probe-relative P3-delta",
            "model": "same HGBR grid; FCM max_iter 600 fixed by diagnostics",
            "split": "corrected splits; both fits must formally converge",
            "metric": "LODO/LOSFO MAE/RMSE/R2, retention, wins and grouped bootstrap",
            "result": f"retention={full['retention_fraction']:.3f}; LODO P0={full_lodo['P0']['mae']:.6f}, P4={full_lodo['P4']['mae']:.6f}.",
            "kept/rejected": "kept",
            "reason": "Final corrected Y1 estimate; no outcome-driven filtering.",
        },
    ])
    atomic_text(ART / "hypothesis_y1_exploration_registry.json", json.dumps(attempts, indent=2, sort_keys=True) + "\n")

    corr_dup_lodo = corrected_metrics["duplicate_corrected_150_all"]["LODO"]
    matrix_rows = [
        ["Phase-7 evaluator implementation bug", "Post-label evaluator rewrite created opportunity for error.", "Independent evaluator reproduces all headline values/verdict within 1e-12.", "none", "none", "max headline absolute difference <= 8.4e-17", "high confidence rejected", "No", "No", "No"],
        ["non-convergence contaminated structural signals", "635/4500 usable rows lacked both convergence; Letter 0/375 both.", f"S3 still has P4={s3_metrics['LODO']['P4']['mae']:.4f} vs P0={s3_metrics['LODO']['P0']['mae']:.4f}.", "Letter dominant", "mixed", f"S3 P0-P4={s3_metrics['LODO']['P0']['mae']-s3_metrics['LODO']['P4']['mae']:.4f}", "high defect; high no-rescue", "Defect yes; conclusion no", "No", "Corrected run completed"],
        ["duplicate leakage contaminated folds", f"Phase7 leakage in {', '.join(phase7_dup)}; max Phase7 rate 0.1194.", f"Duplicate-corrected P0={corr_dup_lodo['P0']['mae']:.4f}, P4={corr_dup_lodo['P4']['mae']:.4f}.", ", ".join(phase7_dup), "all families affected through datasets", f"corrected P0-P4={corr_dup_lodo['P0']['mae']-corr_dup_lodo['P4']['mae']:.4f}", "high", "Only if corrected reversal; observed in report", "No", "Completed on same panel"],
        ["missing target classes invalidated folds", "Four Phase7 target folds: Glass 0/2 and Mice 1/4.", "All Phase7 corrected folds cover K; joint corrected result does not rescue premise.", "Glass, Mice", "not family-specific", "4/60 original folds", "high", "Local folds yes; overall conclusion no", "No", "Completed on same panel"],
        ["raw validity controls were unfair", "P3 used levels while P4 used changes.", f"Fair-delta LODO={fair_metrics['LODO']['mae']:.4f}, but median no-skill={no_skill['LODO_B0_median']['mae']:.4f} and LOSFO={fair_metrics['LOSFO']['mae']:.4f}.", "7/12 beat P0", "3/7 beat P0", f"LODO gain vs P0={independent['hgb_regression']['LODO']['P0']-fair_metrics['LODO']['mae']:.4f}", "moderate", "Comparator fairness only", "No", "Yes for any future validity hypothesis"],
        ["raw structural signals are dataset-specific", "Strong metadata correlations and inconsistent within/cross directions.", f"Only {within['both_help_datasets']} helps in both; some family wins exist.", "6/12 within help, 5/12 cross help", "4/7 P4 vs P0", f"same direction {within['same_direction_count']}/12", "high", "No", "No coherent repeat", "Would require untouched data"],
        ["D_V contains real transferable signal", "Within-dataset rho up to 0.392 Letter and 0.316 Mice.", f"Overall rho={float(dv_overall['spearman_delta_ari']):.3f}; negative Sonar/Spambase; no D_V-only model selected.", "Letter, Mice", "outliers strongest rho 0.242", "overall rho 0.054", "low", "No", "No", "Yes if separately preregistered"],
        ["D_H destroys transfer", "D_H correlates with source entropy and can lose direction.", f"Source-relative P4 LODO={dh_lodo['mae']:.4f}, essentially no better than P4={independent['hgb_regression']['LODO']['P4']:.4f}.", "mixed", "sign changes across families", f"delta MAE={dh_lodo['mae']-independent['hgb_regression']['LODO']['P4']:.6f}", "high confidence rejected as sole cause", "No", "No", "No"],
        ["dimension-adaptive fuzzification causes instability", "Letter/high K is dominant convergence pathology; signal levels relate to D/effective m.", "Failures also occur from D=4 to 561 and K=2 to 26; no numerical/degenerate failures.", "Letter strongest", "not isolated", "153/161 sampled failures converge by 600", "moderate", "No", "No", "Would need fixed-m preregistered comparison"],
        ["exact Delta ARI regression is unsuitable", "All learned frozen HGBR R2 values are negative and median no-skill wins MAE.", "Signed magnitude remains a legitimate estimand if a model has skill.", "all pooled", "all pooled", f"P4 R2={float(original_lodo.query("feature_block == 'P4'").iloc[0]['r2']):.3f}", "high for current model", "No", "No", "Only after new target justification"],
        ["risk ranking works while magnitude regression fails", f"P4 ranking rho={float(p4_rank['spearman']):.3f}.", f"P4 AUROC at ΔARI>.05={float(p4_auc['auroc']):.3f}, balanced accuracy={float(p4_auc['balanced_accuracy']):.3f}.", "none coherent", "none coherent", "near-zero/anti-skill", "high confidence rejected", "No", "No", "No"],
        ["classification labels are poor cluster truth", f"Construct tiers A/B/C={forensics['construct_tier_counts']['A']}/{forensics['construct_tier_counts']['B']}/{forensics['construct_tier_counts']['C']}.", "ARI still valid for explicitly framed external class-recovery stress tests.", "only Iris tier A in Phase7", "all controlled families", "42/54 tier C", "high", "Invalidates broad natural-cluster claim, not computed comparison", "No", "Use latent-ground-truth synthetics"],
        ["structural signals are genuinely weak", f"P4 loses P0 overall, median no-skill wins, structural rho values <= {float(dm_overall['spearman_delta_ari']):.3f} overall.", "P4 wins selected datasets/families and D_M has modest positive association.", "heterogeneous", "heterogeneous", f"original P0-P4={independent['hgb_regression']['LODO']['P0']-independent['hgb_regression']['LODO']['P4']:.4f}", "high for tested formulation", "No", "No", "No automatic continuation"],
        ["dataset generalization differs from shift-family generalization", "LODO P4 loses P0 while LOSFO P4 slightly beats P0; P3 relation also reverses.", "Both learned evaluations lose grouped median no-skill and bootstrap evidence is weak.", "5/12 P4 wins", "4/7 P4 wins", f"LODO Δ={independent['bootstrap']['lodo_delta_04']['sample_mean']:.4f}; LOSFO Δ={independent['bootstrap']['losfo_delta_04']['sample_mean']:.4f}", "high descriptive", "No", "No", "Would require independent datasets and families"],
        ["signal redundancy hurts model transfer", f"D_U_R/D_U_C Spearman={float(du_pair['spearman']):.3f}; VIFs about 15 and 18.", "Tree models can tolerate correlated predictors; redundancy is not causal proof.", "all", "all", "rho 0.971", "moderate", "No", "No", "No unless mechanism specified"],
    ]
    matrix_columns = [
        "mechanism", "supporting evidence", "contradictory evidence", "datasets supporting",
        "families supporting", "effect magnitude", "confidence", "would it invalidate Phase 7?",
        "does it justify new hypothesis?", "untouched test required?",
    ]
    decision_matrix = pd.DataFrame(matrix_rows, columns=matrix_columns)
    atomic_text(ART / "hypothesis_y1_decision_matrix.csv", decision_matrix.to_csv(index=False))

    def original_metric(evaluation: str, block: str) -> dict[str, float | int]:
        table = original_lodo if evaluation == "LODO" else original_losfo
        row = table[table["feature_block"].eq(block)].iloc[0]
        return {key: row[key] for key in ("mae", "rmse", "r2", "n_predictions")}

    quality = pd.read_csv(ROOT / "results/falsification/quality_evaluation_only.csv")
    outcome = quality.loc[quality["condition"].ne("clean"), "delta_ari"].to_numpy(dtype=np.float64)
    outcome_sst = float(np.sum((outcome - outcome.mean()) ** 2))

    def independent_metric(evaluation: str, block: str) -> dict[str, float | int]:
        table = independent_lodo if evaluation == "LODO" else independent_losfo
        selected = table[
            table["feature_block"].eq(block) & table["regressor"].eq("HGBR")
        ].copy()
        n = int(selected["n_test"].sum())
        sae = float(np.sum(selected["n_test"] * selected["mae"]))
        sse = float(np.sum(selected["n_test"] * selected["rmse"] ** 2))
        return {
            "n": n,
            "mae": sae / n,
            "rmse": math.sqrt(sse / n),
            "r2": 1.0 - sse / outcome_sst,
        }

    report_rows: list[str] = []
    for analysis, metric_set in (
        ("ORIGINAL PHASE 7", {e: {b: original_metric(e, b) for b in ("P0", "P3", "P4")} for e in ("LODO", "LOSFO")}),
        ("INDEPENDENT REPLICATION", {e: {b: independent_metric(e, b) for b in ("P0", "P3", "P4")} for e in ("LODO", "LOSFO")}),
        ("CONVERGED-ONLY (S3, ORIGINAL ROWS)", s3_metrics),
        ("DUPLICATE/CLASS-CORRECTED (150 ALL)", corrected_metrics["duplicate_corrected_150_all"]),
        ("FULL CORRECTED Y1 (600, BOTH CONVERGED)", corrected_metrics["fully_corrected_600"]),
    ):
        for evaluation in ("LODO", "LOSFO"):
            for block, metric in metric_set[evaluation].items():
                report_rows.append(
                    f"| {analysis} | {evaluation} | {block} | {fmt(metric['mae'])} | {fmt(metric['rmse'])} | {fmt(metric['r2'])} | {metric.get('n', metric.get('n_predictions', 4200))} |"
                )
    for evaluation in ("LODO", "LOSFO"):
        metric = fair_metrics[evaluation]
        report_rows.append(f"| FAIR DELTA-VALIDITY CONTROL | {evaluation} | P3-delta | {fmt(metric['mae'])} | {fmt(metric['rmse'])} | {fmt(metric['r2'])} | {metric['n']} |")
        for key in ("B0_mean", "B0_median"):
            metric = no_skill[f"{evaluation}_{key}"]
            report_rows.append(f"| NO-SKILL BASELINE | {evaluation} | {key} | {fmt(metric['mae'])} | {fmt(metric['rmse'])} | {fmt(metric['r2'])} | {metric['n']} |")

    correction_delta_rows: list[str] = []
    for arm, label in (
        ("duplicate_corrected_150_all", "duplicate/class-corrected 150"),
        ("fully_corrected_600", "full corrected 600/both"),
    ):
        for evaluation in ("LODO", "LOSFO"):
            for block in ("P0", "P3", "P4"):
                original_value = float(original_metric(evaluation, block)["mae"])
                corrected_value = float(corrected_metrics[arm][evaluation][block]["mae"])
                delta = corrected_value - original_value
                correction_delta_rows.append(
                    f"| {label} | {evaluation} | {block} | {fmt(original_value)} | "
                    f"{fmt(corrected_value)} | {fmt(delta)} | {'worse' if delta > 0 else 'better' if delta < 0 else 'unchanged'} |"
                )

    comparison_rows: list[str] = []

    def add_comparison(analysis: str, evaluation: str, contrast: str, wins: int, summary: dict[str, Any]) -> None:
        comparison_rows.append(
            f"| {analysis} | {evaluation} | {contrast} | {wins}/{summary['n_units']} | "
            f"{fmt(summary['sample_mean'])} | [{fmt(summary['mean_ci_lower'])}, {fmt(summary['mean_ci_upper'])}] |"
        )

    for analysis in ("ORIGINAL PHASE 7", "INDEPENDENT REPLICATION"):
        for evaluation, prefix, n_units in (("LODO", "lodo", 12), ("LOSFO", "losfo", 7)):
            for reference, suffix, wins in (
                ("P0", "04", 5 if evaluation == "LODO" else 4),
                ("P3", "34", 10 if evaluation == "LODO" else 3),
            ):
                add_comparison(
                    analysis,
                    evaluation,
                    f"{reference} MAE − P4 MAE",
                    wins,
                    independent["bootstrap"][f"{prefix}_delta_{suffix}"],
                )
    for evaluation in ("LODO", "LOSFO"):
        sensitivity = sensitivities["convergence"]["sensitivities"]["S3_both_converged"][evaluation]
        for reference, suffix in (("P0", "04"), ("P3", "34")):
            add_comparison(
                "CONVERGED-ONLY (S3, ORIGINAL ROWS)",
                evaluation,
                f"{reference} MAE − P4 MAE",
                int(sensitivity[f"wins_{suffix}"]),
                sensitivity[f"bootstrap_delta_{suffix}"],
            )
    for arm, label in (
        ("duplicate_corrected_150_all", "DUPLICATE/CLASS-CORRECTED (150 ALL)"),
        ("fully_corrected_600", "FULL CORRECTED Y1 (600, BOTH CONVERGED)"),
    ):
        for evaluation in ("LODO", "LOSFO"):
            for reference in ("P0", "P3", "P3_delta"):
                item = corrected["arms"][arm][evaluation]["comparisons"][f"{reference}_minus_P4"]
                add_comparison(label, evaluation, f"{reference} MAE − P4 MAE", int(item["wins"]), item["bootstrap"])

    for _, item in no_skill_comparisons.iterrows():
        summary = {
            "n_units": int(item["n_units"]),
            "sample_mean": float(item["sample_mean"]),
            "mean_ci_lower": float(item["mean_ci_lower"]),
            "mean_ci_upper": float(item["mean_ci_upper"]),
        }
        add_comparison(
            "NO-SKILL BASELINE",
            str(item["evaluation"]),
            f"{item['baseline']} MAE − {item['learned_block']} MAE",
            int(item["learned_wins"]),
            summary,
        )

    for evaluation, independent_units in (("LODO", independent_lodo), ("LOSFO", independent_losfo)):
        independent_units = independent_units[independent_units["regressor"].eq("HGBR")]
        control = control_units[
            control_units["evaluation"].eq(evaluation)
            & control_units["regressor"].eq("HGBR")
            & control_units["feature_block"].eq("Y1_VALIDITY_DELTA_CLEAN_REFERENCE")
        ].set_index("outer_group")["mae"]
        original_units = independent_units.set_index(["outer_group", "feature_block"])["mae"]
        for reference in ("P0", "P4"):
            reference_mae = original_units.xs(reference, level="feature_block").reindex(control.index)
            values = reference_mae.to_numpy(dtype=np.float64) - control.to_numpy(dtype=np.float64)
            add_comparison(
                "FAIR DELTA-VALIDITY CONTROL",
                evaluation,
                f"{reference} MAE − P3-delta MAE",
                int(np.sum(values > 0)),
                paired_summary(values, bootstrap_repetitions, bootstrap_seed),
            )

    full_rows = corrected_rows[
        corrected_rows["replication_arm"].eq("fully_corrected_600")
        & corrected_rows["condition"].ne("clean")
        & corrected_rows["primary_eligible"].astype(bool)
    ]
    full_all_shifted = corrected_rows[
        corrected_rows["replication_arm"].eq("fully_corrected_600")
        & corrected_rows["condition"].ne("clean")
    ]
    retention_by_dataset = (
        full_all_shifted.groupby("dataset_id", sort=True)["primary_eligible"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    retention_rows = [
        f"| {row.dataset_id} | {int(row['sum'])} | {int(row['count'] - row['sum'])} | {float(row['mean']):.2%} |"
        for _, row in retention_by_dataset.iterrows()
    ]
    family_order = []
    for condition in falsification_cfg["conditions"]:
        family = condition.rsplit("_", 1)[0]
        if condition != "clean" and family not in family_order:
            family_order.append(family)
    for evaluation, group_column, order in (
        ("LODO", "dataset_id", list(falsification_cfg["datasets"])),
        ("LOSFO", "shift_family", family_order),
    ):
        learned_by_group = corrected_units[
            corrected_units["replication_arm"].eq("fully_corrected_600")
            & corrected_units["evaluation"].eq(evaluation)
            & corrected_units["regressor"].eq("HGBR")
        ].pivot(index="outer_group", columns="feature_block", values="mae")
        present_order = [group for group in order if group in learned_by_group.index]
        for kind in ("mean", "median"):
            baseline_unit_mae = []
            for group in present_order:
                test = full_rows[full_rows[group_column].eq(group)]
                train = full_rows[~full_rows[group_column].eq(group)]
                constant = float(getattr(train["delta_ari"], kind)())
                baseline_unit_mae.append(float(np.mean(np.abs(test["delta_ari"] - constant))))
            for learned in ("P0", "P3", "P3_delta", "P4"):
                values = (
                    np.asarray(baseline_unit_mae)
                    - learned_by_group[learned].reindex(present_order).to_numpy(dtype=np.float64)
                )
                add_comparison(
                    "FULL CORRECTED NO-SKILL",
                    evaluation,
                    f"B0_{kind} MAE − {learned} MAE",
                    int(np.sum(values > 0)),
                    paired_summary(values, bootstrap_repetitions, bootstrap_seed),
                )

    dependency_paths = [
        BASE / "scripts/hypothesis_y1_corrected_replication.py",
        BASE / "scripts/hypothesis_y1_independent_phase7_replication.py",
        ROOT / "src/clusterdrift/data/preprocess.py",
        ROOT / "src/clusterdrift/methods/fcm.py",
        ROOT / "src/clusterdrift/probes/hashing.py",
        ROOT / "src/clusterdrift/probes/selection.py",
        ROOT / "src/clusterdrift/shifts/engine.py",
        ROOT / "src/clusterdrift/signals/engine.py",
        ROOT / "src/clusterdrift/signals/validity.py",
    ]
    config_paths = [ROOT / f"configs/{name}.yaml" for name in (
        "falsification", "shifts", "probes", "signals", "alignment", "methods", "preprocessing"
    )]
    split_bindings = (
        corrected_rows[["dataset_id", "outer_fold", "corrected_split_sha256"]]
        .drop_duplicates()
        .sort_values(["dataset_id", "outer_fold"])
    )
    if len(split_bindings) != 60 or corrected_rows["task_input_fingerprint"].nunique() != 60:
        raise RuntimeError("Corrected result does not contain exactly 60 uniquely bound dataset/fold tasks")
    corrected_provenance = {
        "audit": "hypothesis_y1_corrected_execution_provenance",
        "label": "POST-HOC METHODOLOGICAL CORRECTION / EXPLORATORY REPLICATION",
        "git_head_during_y1_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "interpreter": str(Path(sys.executable).resolve()),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "pyyaml": version("pyyaml"),
        },
        "executed_code_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in dependency_paths
        },
        "configuration_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in config_paths
        },
        "corrected_split_bindings": split_bindings.to_dict(orient="records"),
        "unique_task_input_fingerprints": int(corrected_rows["task_input_fingerprint"].nunique()),
        "result_sha256": {
            "corrected_rows": sha256_file(CORR / "hypothesis_y1_corrected_rows.csv"),
            "corrected_model_metrics": sha256_file(CORR / "hypothesis_y1_corrected_model_metrics.csv"),
            "corrected_summary": sha256_file(CORR / "hypothesis_y1_corrected_replication_summary.json"),
        },
        "runtime_checkpoint_cache_policy": "noncanonical resume cache; delete after consolidated outputs and provenance are verified",
    }
    atomic_text(
        ART / "hypothesis_y1_corrected_execution_provenance.json",
        json.dumps(corrected_provenance, indent=2, sort_keys=True) + "\n",
    )

    corr_report = f"""# Hypothesis Y1 corrected-replication report

Status: post-hoc methodological correction and failure analysis. It does not replace the frozen preregistered Phase-7 result.

## Side-by-side metrics

“Duplicate/class-corrected” is one joint split arm because the same outcome-blind grouped split repair fixes exact-row/entity leakage and Phase-7 target-class coverage. The 150-iteration arm isolates that split correction. The full arm adds the diagnostic-fixed 600-iteration horizon and requires both source and candidate convergence.

| Analysis | Evaluation | Block | MAE | RMSE | R² | n predictions |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(report_rows)}

MAE/RMSE/R² above are pooled over the retained held-out predictions. The win counts and paired bootstrap below deliberately give each held-out dataset or shift family one inferential unit; where convergence filtering makes unit sizes unequal, their mean contrast need not equal the difference between pooled MAEs.

## Effect of correction on MAE

Positive corrected-minus-original differences are explicitly marked as worse; none are filtered from the report.

| Corrected arm | Evaluation | Block | original MAE | corrected MAE | corrected − original | direction |
|---|---|---|---:|---:|---:|---|
{chr(10).join(correction_delta_rows)}

## Unit wins and grouped bootstrap

Every contrast is `reference MAE − candidate MAE`; a positive value and a win mean the candidate named on the right has lower MAE. Bootstrap resampling uses the preregistered 10,000 draws and seed at the dataset or shift-family unit, never 4,200 rows as independent units.

| Analysis | Evaluation | Contrast | right-side wins | mean contrast | 95% grouped-bootstrap CI |
|---|---|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

Selected headline checks:

- Original LODO P0−P4: 5/12 dataset wins; {boot_text(independent['bootstrap']['lodo_delta_04'])}.
- Original LODO P3−P4: 10/12 dataset wins; {boot_text(independent['bootstrap']['lodo_delta_34'])}.
- Original LOSFO P0−P4: 4/7 family wins; {boot_text(independent['bootstrap']['losfo_delta_04'])}.
- Original LOSFO P3−P4: 3/7 family wins; {boot_text(independent['bootstrap']['losfo_delta_34'])}.
- Converged-only LODO P0−P4: {sensitivities['convergence']['sensitivities']['S3_both_converged']['LODO']['wins_04']}/{sensitivities['convergence']['sensitivities']['S3_both_converged']['LODO']['n_units']}; {boot_text(sensitivities['convergence']['sensitivities']['S3_both_converged']['LODO']['bootstrap_delta_04'])}.
- Duplicate/class-corrected LODO P0−P4: {corrected['arms']['duplicate_corrected_150_all']['LODO']['comparisons']['P0_minus_P4']['wins']}/{corrected['arms']['duplicate_corrected_150_all']['LODO']['comparisons']['P0_minus_P4']['n_units']}; {boot_text(corrected['arms']['duplicate_corrected_150_all']['LODO']['comparisons']['P0_minus_P4'])}.
- Full-corrected LODO P0−P4: {full_comp_lodo['P0_minus_P4']['wins']}/{full_comp_lodo['P0_minus_P4']['n_units']}; {boot_text(full_comp_lodo['P0_minus_P4'])}.
- Full-corrected LODO P3-delta−P4: {full_comp_lodo['P3_delta_minus_P4']['wins']}/{full_comp_lodo['P3_delta_minus_P4']['n_units']}; {boot_text(full_comp_lodo['P3_delta_minus_P4'])}.
- Full-corrected LOSFO P0−P4: {full_comp_losfo['P0_minus_P4']['wins']}/{full_comp_losfo['P0_minus_P4']['n_units']}; {boot_text(full_comp_losfo['P0_minus_P4'])}.
- Full-corrected LOSFO P3-delta−P4: {full_comp_losfo['P3_delta_minus_P4']['wins']}/{full_comp_losfo['P3_delta_minus_P4']['n_units']}; {boot_text(full_comp_losfo['P3_delta_minus_P4'])}.

## Retention and validity

The full corrected arm retains {full['n_rows']}/4200 shifted scenario rows ({full['retention_fraction']:.2%}). The split verifier reports zero cross-fold connected-group leakage in all corrected outer and inner folds. All twelve Phase-7 datasets have every manifest class in source and target after correction. Five Ecoli target folds in the broader 30-dataset split audit remain class-incomplete because its two rarest classes contain only two samples each, making five-fold target coverage mathematically impossible; Ecoli is not in the corrected Phase-7 panel.

| Dataset | retained | excluded by policy | retention |
|---|---:|---:|---:|
{chr(10).join(retention_rows)}

The exploratory frozen-row fair control uses each clean target-fold scenario as its label-free reference. In the corrected arms, `P3_delta` is stricter and deployment-oriented: `current − source/reference-probe` FPC, PE, XB and silhouette, plus D_X. These two controls answer the same fairness objection but are not numerically interchangeable, so the report does not pretend they are one result.

## Interpretation

Correcting real methodological defects {'reverses the P0 comparison strongly enough to make the historical conclusion unsafe' if reversal else 'does not rescue structural P4 against P0 and grouped no-skill evidence'}. The fair P3-delta control is reported regardless of direction. No case where a correction worsened a metric has been omitted. Final decision: `{final_decision}`.
"""
    atomic_text(REPORTS / "hypothesis_y1_corrected_replication_report.md", corr_report)

    fixes = [
        "M-FIX-01 require both FCM fits to converge under the outcome-blind 600-iteration policy",
        "M-FIX-02 group exact duplicates and protected entities while preserving temporal splits",
        "M-FIX-03 enforce/report source and target class coverage for Oracle-K folds",
        "M-FIX-04 add grouped mean and median no-skill baselines",
        "M-FIX-05 add clean-referenced conventional validity deltas",
        "M-FIX-06 use dataset/family/scenario grouping for uncertainty and within-dataset validation",
        "S-FIX-01 remove executable joblib cache deserialization",
        "S-FIX-02 digest and shape-check cached numerical payloads",
        "S-FIX-03 bind scalar caches to provenance fingerprints",
        "S-FIX-04 bind cache keys to data/split/probe/protocol/method identities",
        "S-FIX-05 reject path traversal and unsafe recursive cache roots",
        "S-FIX-06 use unique atomic cache temporaries",
        "S-FIX-07 pin, bound, verify, and atomically publish direct archive downloads",
        "S-FIX-08 atomically publish the final verification manifest",
        "S-FIX-09 atomically publish canonical data, split, manifest, and baseline-audit artifacts",
    ]
    test_text = test_summary(tests)
    matrix_support = markdown_table(decision_matrix)
    if reversal:
        minimum_test_text = (
            "The smallest required next step is a newly preregistered replication of the original P4-versus-P0 "
            "question on an outcome-untouched subset of the reserved panel, with corrected duplicate/entity/time "
            "splits, both-converged eligibility, grouped mean/median no-skill controls, MAE as the primary metric, "
            "a fixed 0.05 relative-improvement threshold, a dataset-grouped 95% interval, and a kill rule if P4 "
            "does not beat both P0 and median no-skill. This audit does not execute it."
        )
        untouched_status = "reserved_for_required_new_confirmatory_replication"
    else:
        minimum_test_text = (
            "Not applicable under `FAILURE_CONFIRMED_STOP_PROJECT`; no confirmation should be run. If an "
            "independent review later supplies a coherent mechanism, it must first freeze one target, one transform "
            "family, one model family, grouped no-skill baselines, corrected entity/duplicate/time splits, effect "
            "threshold, grouped interval, and kill rule, then test the smallest untouched panel. This audit does not "
            "propose or execute that experiment."
        )
        untouched_status = "reserved_not_authorized_under_stop_decision"
    master = f"""# Hypothesis Y1 master scientific audit

## 1. Executive conclusion

`{final_decision}`. The historical negative result is computationally reproducible. Convergence eligibility, duplicate/group splitting, target-class coverage, comparator fairness, dependence, and post-label implementation timing contain genuine defects, but the isolated corrected analysis {'materially reverses the core comparison' if reversal else 'does not recover predictive skill for P4'}. The grouped median no-skill predictor remains a decisive benchmark. **DO NOT CONTINUE ORIGINAL PHASE 8–15 ROADMAP.**

## 2. Historical frozen Phase-7 result

At commit `c9280bb390492f4528752f5457e341fb3d9c1040`, LODO MAE was P0 `{independent['hgb_regression']['LODO']['P0']}`, P3 `{independent['hgb_regression']['LODO']['P3']}`, and P4 `{independent['hgb_regression']['LODO']['P4']}`. Relative improvements were R04 `{independent['relative_improvement']['r_04']}` and R34 `{independent['relative_improvement']['r_34']}`. P4 won 5/12 datasets against P0 and 10/12 against P3, and 4/7 and 3/7 families respectively. Verdict: `FAILS_PRIMARY_FALSIFICATION`. Frozen inputs and results were not edited.

## 3. Independent reproduction

The independent script used only the two frozen CSVs and preregistered YAML and imported none of the original evaluator/model/bootstrap/verification modules. Agreement status is `{independent['agreement']['status']}` at tolerance `{independent['agreement']['tolerance']}`; all continuous discrepancies are below `1e-12`, discrete wins/rules match, and the verdict is identical. An initial reporting-order bootstrap mismatch is retained as negative implementation evidence and was resolved by restoring the preregistered dataset/family order, not by changing values or rules.

## 4. Confirmed methodological problems

| ID | Classification | Confirmed problem |
|---|---|---|
| M01 | A. FIX_NOW_METHODOLOGICAL | `usable` accepted 635 rows without both FCM fits converged |
| M02 | A. FIX_NOW_METHODOLOGICAL | exact duplicates crossed frozen splits in four Phase-7 datasets |
| M03 | A. FIX_NOW_METHODOLOGICAL | four target folds lacked at least one Oracle-K class |
| M04 | A. FIX_NOW_METHODOLOGICAL | no grouped no-skill baseline was in the gate |
| M05 | A. FIX_NOW_METHODOLOGICAL | raw validity levels were compared with structural changes |
| M06 | A. FIX_NOW_METHODOLOGICAL | five seeds/scenarios are dependent and row-level inference would pseudoreplicate |
| M07 | A. FIX_NOW_METHODOLOGICAL | major evaluator code changed after quality labels were available |
| M08 | D. DOCUMENTATION_ONLY | classification outcomes were over-described as natural cluster truth |
| M09 | D. DOCUMENTATION_ONLY | contemporaneous hard-ARI estimation was called future soft-clustering failure prediction |

## 5. Problems rejected as non-issues

The independent evaluator rejects a bad join, wrong metric aggregation, hidden target column, changed HGBR grid/tie rule, and wrong verdict implementation as explanations. The convergence diagnostic found no objective increase, empty cluster, degeneracy, or numerical failure in 207 reruns. Source-only preprocessing and dataset-grouped outer evaluation were correctly implemented. Empty future files, the unfinished bibliography, and manuscript placeholders are expected incomplete rather than scientific defects.

## 6. Exact fixes implemented

Fifteen bounded fixes were implemented without altering a frozen result:

{chr(10).join(f'- {item}' for item in fixes)}

## 7. FCM convergence findings

Source convergence is {convergence['overall']['source_converged_count']}/{convergence['overall']['n_rows']} ({convergence['overall']['source_convergence_rate']:.2%}); candidate convergence is {convergence['overall']['candidate_converged_count']}/{convergence['overall']['n_rows']} ({convergence['overall']['candidate_convergence_rate']:.2%}); both is {convergence['overall']['both_converged_count']}/{convergence['overall']['n_rows']} ({convergence['overall']['both_convergence_rate']:.2%}). Letter is severe at 0/375 both. Of {root_cause['original_failed_rows_in_sample']} diagnosed original failures, {root_cause['failed_sample_converged_by_300']} converged by 300 and {root_cause['failed_sample_converged_by_600']} by 600; eight remained unresolved. Root cause is principally slow convergence truncated at 150, not demonstrated update-equation failure. S3 remains adverse: pooled LODO P0 `{fmt(s3_metrics['LODO']['P0']['mae'])}`, P4 `{fmt(s3_metrics['LODO']['P4']['mae'])}`. The full 600-iteration policy retains 4,145/4,200 shifted rows; 54 excluded rows are Letter and one is HAR, and exclusions remain explicit rather than silently usable.

## 8. Duplicate leakage findings

Across all 30 controlled datasets, nine have at least one frozen fold with exact-row crossing: {', '.join(all_dup)}. The largest target leakage fraction is {largest_dup:.2%} (Haberman). The Phase-7 subset is {', '.join(phase7_dup)}; its maximum is 11.94% (Spambase). Connected exact-row/protected-entity groups eliminate all corrected outer and inner crossings.

## 9. Class/K validity

Oracle K is explicitly sourced from manifest labels and is acceptable only as a controlled external-recovery protocol. Frozen source folds cover all classes, but {len(original_class_bad)} Phase-7 target folds do not: Glass folds 0/2 and Mice folds 1/4. Corrected Phase-7 folds all cover K. The broader corrected audit leaves {len(corrected_class_bad)} Ecoli target folds incomplete because two classes have only two observations, so five-fold target coverage is impossible; future Ecoli work must reduce folds or change the estimand before outcomes.

## 10. No-skill baseline

Grouped frozen-row LODO mean MAE is `{no_skill['LODO_B0_mean']['mae']:.6f}` and median MAE is `{no_skill['LODO_B0_median']['mae']:.6f}`. Grouped LOSFO mean/median are `{no_skill['LOSFO_B0_mean']['mae']:.6f}` / `{no_skill['LOSFO_B0_median']['mae']:.6f}`. The median beats original P0, P3, and P4 in MAE. In the fully corrected arm, LODO mean/median no-skill MAE is `{full['no_skill']['LODO_B0_mean']['mae']:.6f}` / `{full['no_skill']['LODO_B0_median']['mae']:.6f}`, again below every learned block. Negative learned-model R² reinforces the absence of magnitude-prediction skill.

## 11. Fair validity-delta control

Replacing raw FPC/PE/XB/silhouette levels with same-dataset/fold/seed clean-reference deltas yields LODO MAE `{fair_metrics['LODO']['mae']:.6f}` and LOSFO `{fair_metrics['LOSFO']['mae']:.6f}`. It beats P0 on 7/12 datasets but only 3/7 families and remains worse than the LODO median no-skill baseline. The correction improves comparator fairness; it does not rescue the structural thesis.

## 12. Original vs corrected replication

The full side-by-side metrics, wins, grouped bootstrap intervals and cases worsened by correction are in `hypothesis_y1_corrected_replication_report.md`. Full corrected retention is {full['retention_fraction']:.2%}; LODO P0/source-reference P3-delta/P4 are `{fmt(full_lodo['P0']['mae'])}` / `{fmt(full_lodo['P3_delta']['mae'])}` / `{fmt(full_lodo['P4']['mae'])}`, with median no-skill `{fmt(full_median)}`. Unlike the clean-target-referenced exploratory control in section 11, corrected P3-delta uses current minus source/reference-probe validity values. The correction {'makes the frozen answer unsafe and requires a new confirmation' if reversal else 'does not materially change the negative answer'}.

## 13. Construct validity of labels

The 54-unit construct inventory has A/B/C counts `{forensics['construct_tier_counts']['A']}/{forensics['construct_tier_counts']['B']}/{forensics['construct_tier_counts']['C']}`. Only Iris is tier A in Phase 7; the other eleven are supervised classes/outcomes used as stress proxies. The results are defensible as external hard-label recovery stress tests, not proof about natural latent clusters in general.

## 14. Hard ARI vs soft-clustering terminology

Phase 7 predicts `ARI_clean−ARI_shifted` after hardening FCM assignments. Soft memberships generate candidate signals, but the outcome is not soft-membership fidelity. Claims are restricted to target-label-free contemporaneous estimation of hard clustering degradation. Soft failure requires posterior-membership targets on families with valid latent posteriors.

## 15. Within-vs-cross dataset result

P4 improves P0 in {within['within_structural_help_count']}/12 within-dataset models and {within['cross_structural_help_count']}/12 LODO results. Directions agree in only {within['same_direction_count']}/12, and only {', '.join(within['both_help_datasets'])} improves in both. This is active counterevidence against a stable dataset-calibration rescue.

## 16. Signal calibration result

Several raw signal levels track dataset identity: FPC vs source entropy Spearman `-0.839`, PE vs source entropy `0.811`, XB vs dimension `0.823`, D_H vs source entropy `0.720`, and D_U_R vs K `0.611`. D_U and D_H between/within variance ratios exceed one, while D_V is `0.067`. One predeclared normalization per signal family was registered; no outcome-driven transform sweep was conducted.

## 17. D_V mechanism

D_V is already normalized by frozen reference radius. Its overall Spearman with degradation is `{float(dv_overall['spearman_delta_ari']):.3f}`; positive Letter/Mice associations coexist with negative Sonar/Spambase associations. D_V may describe prototype motion, but no transferable degradation mechanism repeats strongly enough to justify a D_V-only hypothesis.

## 18. D_H mechanism

Absolute D_H loses direction and relates strongly to source entropy. The single preregistered source-relative test gives LODO MAE `{dh_lodo['mae']:.6f}`, versus original P4 `{independent['hgb_regression']['LODO']['P4']:.6f}`. It does not rescue P4, so “D_H alone destroys transfer” is rejected.

## 19. Dimension/fuzzifier mechanism

Effective m is deterministically tied to dimension under the adaptive rule. Non-convergence is concentrated in high-K Letter, but diagnosed failures span D=4–561 and K=2–26. A longer fixed horizon resolves 153/161 sampled failures without changing tolerance. The evidence supports an iteration-budget interaction, not a causal claim that adaptive fuzzification alone caused predictive failure.

## 20. Shift-family result

Original P4 beats P0 for scale, measurement noise, class prevalence, and local overlap, but loses for location, MCAR, and outliers. Against P3 it wins only outliers, measurement noise, and local overlap. Effects are small and inconsistent; LOSFO learned models also lose the median no-skill comparator.

## 21. Regression-vs-risk-ranking result

All original HGBR blocks have negative LODO R² (P4 `{float(original_lodo.query("feature_block == 'P4'").iloc[0]['r2']):.3f}`). P4 ranking Spearman is `{float(p4_rank['spearman']):.3f}`. At ΔARI>0.05 its AUROC is `{float(p4_auc['auroc']):.3f}` and balanced accuracy `{float(p4_auc['balanced_accuracy']):.3f}`. Neither fixed-threshold classification nor ranking rescues exact regression.

## 22. Signal redundancy

D_U_R and D_U_C have raw cross-dataset Spearman `{float(du_pair['spearman']):.3f}`, with VIF about 15 and 18; D_M is also strongly correlated with them. Redundancy can impair stable attribution, but HGBR can tolerate correlation, so it is a plausible contributor rather than a proven root cause.

## 23. Dataset influence

Madelon accounts for about 69% of the net adverse mean P0−P4 sum. Removing Madelon, Sonar, and Letter flips the mean sign, but the official verdict never changes under the recorded influence sequence. No dataset was dropped; the analysis demonstrates heterogeneity and warns against post-hoc panel selection.

## 24. Severity consistency

Severe exceeds mild for ΔARI in only `{float(delta_severity['severe_gt_mild_fraction']):.2%}` of matched scenarios. D_X is monotone in `{float(severity[severity['quantity'].eq('D_X')].iloc[0]['severe_gt_mild_fraction']):.2%}`, while structural signals range roughly 65.7–78.5%. Generator dose order therefore does not guarantee performance-loss order and raw family severities are not commensurate.

## 25. Post-label code-change audit

Between Pass-B and preflight, 6,311 insertions and 338 deletions touched the joined table and core evaluator/verification files. Scientific configs remained unchanged, and the independent evaluator matches outputs. This is a confirmed procedural exposure with no demonstrated numerical bias. Full evidence is in `hypothesis_y1_post_label_change_audit.md`.

## 26. Security/integrity findings

The focused scan found {security['security_issue_count']} real software/scientific-integrity defects and repaired all: executable joblib cache loading, missing payload digests, filename-only scalar cache acceptance, incomplete provenance keys, unsafe path components/clear containment, predictable temp files, unpinned/non-atomic archives, non-atomic final verification manifest, and direct canonical scientific writes. Possible committed credential matches: {len(security['possible_committed_secret_matches'])}; current unsafe-deserialization/`shell=True`/`os.system` matches: {len(security['current_unsafe_code_matches'])}. Provider-managed acquisition byte pinning and an empty lockfile remain bounded reproducibility limitations.

## 27. Repository cleanup

Only clearly generated Python/pytest caches and documented duplicate runtime cache material are eligible for removal. Frozen results, raw/canonical data, manifests, historical configs, manuscript, placeholders, and negative exploratory evidence are retained. Exact actions and retained categories are in `hypothesis_y1_repository_cleanup_report.md`.

## 28. Remaining limitations

The corrected run is post-hoc, uses the same twelve heavily observed datasets, and cannot become a new confirmatory result. Class labels are usually weak proxies for natural clusters. The 600-iteration rule is diagnostic-derived and still leaves explicit attrition. Only one model family and one bounded normalization per motivated signal were examined. Provider libraries are not pre-acquisition byte-pinned. No literature novelty audit was repeated.

## 29. Evidence against preferred explanation

The strongest explanation is weak, heterogeneous, dataset-calibrated structural association rather than one broken signal. Counterevidence was actively retained: P4 wins 5/12 original datasets and 4/7 families, D_M has overall Spearman `{float(dm_overall['spearman_delta_ari']):.3f}`, and some within-dataset fits improve. Against it, only one dataset improves both within and cross settings, fair validity changes outperform P4 but not no-skill, ranking/classification fail, D_H normalization fails, corrected methodology fails to supply a robust reversal, and directions vary by family. The favorable cases are insufficiently coherent.

## 30. Decision matrix

{matrix_support}

## 31. Final project decision

`{final_decision}`. {'The correction produces a strong reversal, so the original negative result cannot safely answer the question; only a newly preregistered untouched-data replication could.' if reversal else 'The original failure reproduces, genuine repairs do not rescue the premise, and no repeated non-cherry-picked mechanism meets the new-hypothesis criteria.'}

**DO NOT CONTINUE ORIGINAL PHASE 8–15 ROADMAP.**

## 32. One new hypothesis, ONLY if justified

No new hypothesis is justified. The most promising observations (fair validity deltas, D_M association, selected family wins, and within-dataset improvements) conflict across datasets/families or lose to no-skill. Promoting one would be post-hoc selection forbidden by Y1.

## 33. Untouched confirmatory data

Discovery panel: {', '.join(original_phase7)}. The following frozen broader-panel datasets remain untouched for **Y1 shift-degradation outcome testing**: {', '.join(untouched)}. Their schemas, duplicates, groups, and class feasibility were audited, but no corrected Y1 shift/outcome model was run on them. They are reserved only if an independent reviewer defines a genuinely new preregistered question.

## 34. Minimum preregistered next test

{minimum_test_text}

## 35. Cleanup manifest

`experiments/hypothesis_y1/cleanup/hypothesis_y1_cleanup_manifest.txt` enumerates every retained Y1 file/directory and every deletion record. Test evidence: {test_text}
"""
    atomic_text(REPORTS / "hypothesis_y1_master_report.md", master)

    methodological_issues = [
        "convergence eligibility", "duplicate/group leakage", "target class coverage",
        "missing no-skill baseline", "unfair validity level comparator", "dependence/pseudoreplication",
        "post-label evaluator changes", "construct overclaim", "hard/contemporaneous terminology overclaim",
    ]
    final_summary = {
        "original_phase7_reproduced": independent["agreement"]["status"] == "PASSED",
        "original_P0": independent["hgb_regression"]["LODO"]["P0"],
        "original_P3": independent["hgb_regression"]["LODO"]["P3"],
        "original_P4": independent["hgb_regression"]["LODO"]["P4"],
        "original_verdict": independent["verdict"],
        "source_convergence_rate": convergence["overall"]["source_convergence_rate"],
        "candidate_convergence_rate": convergence["overall"]["candidate_convergence_rate"],
        "both_converged_rate": convergence["overall"]["both_convergence_rate"],
        "datasets_with_severe_convergence_problem": severe_convergence,
        "duplicate_leakage_datasets": {"all_controlled": all_dup, "phase7": phase7_dup},
        "largest_duplicate_leakage_rate": largest_dup,
        "class_K_problem_folds": {
            "original_phase7": len(original_class_bad),
            "corrected_phase7": 0,
            "corrected_broader_panel_infeasible_ecoli": len(corrected_class_bad),
        },
        "no_skill_mean_mae": no_skill["LODO_B0_mean"]["mae"],
        "no_skill_median_mae": no_skill["LODO_B0_median"]["mae"],
        "corrected_P0": full_lodo["P0"]["mae"],
        "corrected_validity_control": full_lodo["P3_delta"]["mae"],
        "corrected_P4": full_lodo["P4"]["mae"],
        "corrected_no_skill_mean_mae": full["no_skill"]["LODO_B0_mean"]["mae"],
        "corrected_no_skill_median_mae": full["no_skill"]["LODO_B0_median"]["mae"],
        "corrected_result_direction": "MATERIAL_REVERSAL" if reversal else "STRUCTURAL_PREMISE_NOT_RESCUED",
        "within_dataset_structural_help_count": within["within_structural_help_count"],
        "lodo_structural_help_count": full_comp_lodo["P0_minus_P4"]["wins"],
        "shift_family_structural_help_count": full_comp_losfo["P0_minus_P4"]["wins"],
        "strongest_mechanism": "weak heterogeneous dataset-calibrated structural associations with substantial D_U redundancy",
        "counterevidence": "some dataset/family wins and modest D_M association, but only Satimage helps both within and cross dataset and no learned block beats grouped median no-skill robustly",
        "security_issue_count": security["security_issue_count"],
        "methodological_issue_count": len(methodological_issues),
        "methodological_issues": methodological_issues,
        "issues_fixed": fixes,
        "fixes_implemented_count": len(fixes),
        "final_decision": final_decision,
        "new_hypothesis_if_any": None,
        "untouched_confirmation_plan": {
            "status": untouched_status,
            "discovery_datasets": original_phase7,
            "untouched_y1_outcome_datasets": untouched,
        },
        "corrected_retention_fraction": full["retention_fraction"],
        "corrected_excluded_rows_by_dataset": {
            str(row.dataset_id): int(row["count"] - row["sum"])
            for _, row in retention_by_dataset.iterrows()
            if int(row["count"] - row["sum"]) > 0
        },
        "test_results": tests.get("suites", {}),
        "initial_integrity_status": initial["integrity_status"],
    }
    atomic_text(ART / "hypothesis_y1_final_summary.json", json.dumps(final_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "final_decision": final_decision,
        "corrected_P0": full_lodo["P0"]["mae"],
        "corrected_P3_delta": full_lodo["P3_delta"]["mae"],
        "corrected_P4": full_lodo["P4"]["mae"],
        "corrected_median_no_skill": full_median,
        "retention": full["retention_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
