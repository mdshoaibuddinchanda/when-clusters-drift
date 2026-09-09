"""Frozen-result statistical forensics for Hypothesis Y1.

This script does not alter or reinterpret the preregistered verdict.  It derives
explicitly post-hoc diagnostics from the frozen Phase-7 tables and writes only
to ``experiments/hypothesis_y1/artifacts``.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/hypothesis_y1/artifacts"
KEYS = ["dataset_id", "outer_fold", "condition", "method", "seed"]
SIGNALS = ["D_X", "D_U_R", "D_U_C", "D_V", "D_H", "D_M", "FPC", "PE", "XB_soft_m2", "silhouette"]
STRUCTURAL = ["D_U_R", "D_U_C", "D_V", "D_H", "D_M"]
THRESHOLDS = [0.0, 0.025, 0.05, 0.10]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "median_absolute_error": float(median_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def paired_bootstrap(values: np.ndarray, order_seed: int, repetitions: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(order_seed)
    idx = rng.randint(0, len(values), size=(repetitions, len(values)))
    means = values[idx].mean(axis=1)
    return {
        "n_units": int(len(values)),
        "sample_mean": float(values.mean()),
        "sample_median": float(np.median(values)),
        "mean_ci_lower": float(np.percentile(means, 2.5)),
        "mean_ci_upper": float(np.percentile(means, 97.5)),
    }


def no_skill(primary: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    unit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    family_order: list[str] = []
    for condition in cfg["conditions"]:
        family = condition.rsplit("_", 1)[0]
        if condition != "clean" and family not in family_order:
            family_order.append(family)
    for evaluation, column, order in [
        ("LODO", "dataset_id", cfg["datasets"]),
        ("LOSFO", "shift_family", family_order),
    ]:
        for outer in order:
            train = primary.loc[primary[column].ne(outer)]
            test = primary.loc[primary[column].eq(outer)]
            if test.empty:
                continue
            for baseline, value in [
                ("B0_mean", float(train["delta_ari"].mean())),
                ("B0_median", float(train["delta_ari"].median())),
            ]:
                part = test[KEYS + ["shift_family", "severity", "delta_ari"]].copy()
                part["evaluation"] = evaluation
                part["outer_group"] = outer
                part["feature_block"] = baseline
                part["predicted_delta_ari"] = value
                prediction_rows.append(part)
                unit_rows.append({
                    "evaluation": evaluation,
                    "outer_group": outer,
                    "feature_block": baseline,
                    "n_test": len(test),
                    **metrics(test["delta_ari"].to_numpy(), np.full(len(test), value)),
                })
    predictions = pd.concat(prediction_rows, ignore_index=True)
    units = pd.DataFrame(unit_rows)
    for (evaluation, block), part in predictions.groupby(["evaluation", "feature_block"], sort=True):
        metric_rows.append({"evaluation": evaluation, "feature_block": block, "n": len(part), **metrics(
            part["delta_ari"].to_numpy(), part["predicted_delta_ari"].to_numpy()
        )})
    return predictions, units, pd.DataFrame(metric_rows)


def baseline_comparisons(
    no_skill_units: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed = int(cfg["bootstrap"]["seed"])
    reps = int(cfg["bootstrap"]["repetitions"])
    for evaluation, frozen_path, unit_column in [
        ("LODO", ROOT / "results/falsification/lodo_dataset_metrics.csv", "dataset_id"),
        ("LOSFO", ROOT / "results/falsification/losfo_family_metrics.csv", "shift_family"),
    ]:
        frozen = pd.read_csv(frozen_path)
        for baseline in ["B0_mean", "B0_median"]:
            b = no_skill_units[(no_skill_units["evaluation"] == evaluation) & (no_skill_units["feature_block"] == baseline)]
            b = b.set_index("outer_group")["mae"]
            for block in ["P0", "P3", "P4"]:
                if "feature_block" in frozen.columns:
                    learned = frozen[frozen["feature_block"].eq(block)].set_index(unit_column)["mae"]
                else:
                    learned = frozen.set_index(unit_column)[f"mae_{block}"]
                order = [name for name in b.index if name in learned.index]
                # Positive means the learned block beats the no-skill baseline.
                deltas = (b.loc[order] - learned.loc[order]).to_numpy()
                boot = paired_bootstrap(deltas, seed, reps)
                rows.append({
                    "evaluation": evaluation,
                    "baseline": baseline,
                    "learned_block": block,
                    "learned_wins": int((deltas > 0).sum()),
                    "n_units": len(deltas),
                    **boot,
                })
    return pd.DataFrame(rows)


def crossfit_risk_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate fixed regression scores without exposing a held-out dataset.

    Each dataset's scores are transformed to probabilities by a one-variable
    logistic calibration fitted only on the other datasets' already-out-of-
    dataset predictions.  No hyperparameter or threshold search is performed.
    """
    metric_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for block, frame in predictions.groupby("feature_block", sort=True):
        frame = frame.reset_index(drop=True)
        for threshold in THRESHOLDS:
            target = (frame["observed_delta_ari"].to_numpy() > threshold).astype(int)
            prob = np.empty(len(frame), dtype=np.float64)
            for dataset in sorted(frame["dataset_id"].unique()):
                test = frame["dataset_id"].eq(dataset).to_numpy()
                train = ~test
                y_train = target[train]
                if np.unique(y_train).size < 2:
                    prob[test] = float(y_train.mean())
                else:
                    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=20260907)
                    model.fit(frame.loc[train, ["predicted_delta_ari"]].to_numpy(), y_train)
                    prob[test] = model.predict_proba(frame.loc[test, ["predicted_delta_ari"]].to_numpy())[:, 1]
            row = {
                "feature_block": block,
                "threshold": threshold,
                "positive_rate": float(target.mean()),
                "calibration": "cross-dataset one-variable logistic calibration of fixed LODO regression score",
                "auroc": float(roc_auc_score(target, prob)) if np.unique(target).size == 2 else np.nan,
                "auprc": float(average_precision_score(target, prob)) if target.sum() else np.nan,
                "balanced_accuracy": float(balanced_accuracy_score(target, prob >= 0.5)),
                "brier": float(brier_score_loss(target, prob)),
            }
            metric_rows.append(row)

        y = frame["observed_delta_ari"].to_numpy(dtype=float)
        score = frame["predicted_delta_ari"].to_numpy(dtype=float)
        rho = spearmanr(y, score, nan_policy="omit").statistic
        positive_mass = np.maximum(y, 0.0)
        for fraction in [0.10, 0.20, 0.30]:
            n_top = max(1, int(np.ceil(len(frame) * fraction)))
            selected = np.argsort(score)[-n_top:]
            ranking_rows.append({
                "feature_block": block,
                "top_fraction": fraction,
                "spearman": float(rho),
                "positive_degradation_mass_capture": float(positive_mass[selected].sum() / positive_mass.sum()) if positive_mass.sum() else np.nan,
                "event_capture_delta_gt_005": float((y[selected] > 0.05).sum() / (y > 0.05).sum()) if (y > 0.05).sum() else np.nan,
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(ranking_rows)


def source_metadata(primary: pd.DataFrame) -> pd.DataFrame:
    manifests = {r["dataset_id"]: r for r in json.loads((ROOT / "data/manifests/datasets.json").read_text(encoding="utf-8"))}
    cache_base = Path(os.environ["LOCALAPPDATA"]) / "clusterdrift/when-clusters-drift/phase7/phase7"
    cache_roots = sorted(p.parent for p in cache_base.glob("*/.clusterdrift_phase7_cache"))
    cache = cache_roots[0] if len(cache_roots) == 1 else None
    rows: list[dict[str, Any]] = []
    for dataset, part in primary.groupby("dataset_id", sort=True):
        manifest = manifests[dataset]
        scales: list[float] = []
        if cache is not None:
            for path in (cache / "memberships").glob(f"{dataset}_fold_*_fcm_adaptive_seed_*_scales.npy"):
                values = np.load(path, allow_pickle=False)
                scales.extend(np.asarray(values, dtype=float).ravel().tolist())
        rows.append({
            "dataset_id": dataset,
            "N": manifest["n_rows"],
            "D": int(part["dimension_D"].iloc[0]),
            "K": int(part["K"].iloc[0]),
            "clean_ari": float(pd.read_csv(ROOT / "results/falsification/quality_evaluation_only.csv").query(
                "dataset_id == @dataset and condition == 'clean'"
            )["ari_clean"].mean()),
            "source_entropy": float(part["entropy_source_current"].mean()),
            "prototype_scale": float(np.mean(scales)) if scales else np.nan,
            "effective_m": float(1.0 + (1418.0 / float(part["dimension_D"].iloc[0]) + 22.05) ** -2.0),
        })
    return pd.DataFrame(rows)


def signal_scale(primary: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    for signal in SIGNALS:
        valid = primary[["dataset_id", signal]].dropna()
        for dataset, part in valid.groupby("dataset_id", sort=True):
            x = part[signal].to_numpy(dtype=float)
            rows.append({
                "dataset_id": dataset,
                "signal": signal,
                "n": len(x),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "std": float(np.std(x, ddof=1)),
                "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
                "q05": float(np.percentile(x, 5)),
                "q95": float(np.percentile(x, 95)),
            })
        means = valid.groupby("dataset_id")[signal].mean()
        within = valid.groupby("dataset_id")[signal].var(ddof=1)
        between_var = float(means.var(ddof=1))
        within_var = float(within.mean())
        variance_rows.append({
            "signal": signal,
            "between_dataset_variance": between_var,
            "mean_within_dataset_variance": within_var,
            "between_within_ratio": between_var / within_var if within_var > 0 else np.inf,
        })
        ds_mean = means.rename("signal_mean").reset_index().merge(metadata, on="dataset_id", how="left")
        for covariate in ["N", "D", "K", "clean_ari", "effective_m", "source_entropy", "prototype_scale"]:
            pair = ds_mean[["signal_mean", covariate]].dropna()
            stat = spearmanr(pair["signal_mean"], pair[covariate]).statistic if len(pair) >= 3 else np.nan
            relationship_rows.append({"signal": signal, "covariate": covariate, "n_datasets": len(pair), "spearman": float(stat)})
    return pd.DataFrame(rows), pd.DataFrame(variance_rows), pd.DataFrame(relationship_rows)


def mechanism_correlations(primary: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = primary.merge(metadata, on="dataset_id", how="left", suffixes=("", "_meta"))
    rows: list[dict[str, Any]] = []
    for signal in STRUCTURAL:
        for scope, group_col in [("overall", None), ("dataset", "dataset_id"), ("shift_family", "shift_family")]:
            groups: Iterable[tuple[str, pd.DataFrame]] = [("ALL", enriched)] if group_col is None else enriched.groupby(group_col, sort=True)
            for group, part in groups:
                pair = part[[signal, "delta_ari"]].dropna()
                rho = spearmanr(pair[signal], pair["delta_ari"]).statistic if len(pair) >= 3 else np.nan
                rows.append({"signal": signal, "scope": scope, "group": group, "n": len(pair), "spearman_delta_ari": float(rho)})
        if signal in ["D_V", "D_H"]:
            for covariate in ["D", "K", "clean_ari", "effective_m", "source_entropy", "prototype_scale"]:
                pair = enriched[[signal, covariate]].dropna()
                rows.append({
                    "signal": signal,
                    "scope": "metadata",
                    "group": covariate,
                    "n": len(pair),
                    "spearman_delta_ari": float(spearmanr(pair[signal], pair[covariate]).statistic),
                })

    centered = primary.copy()
    centered[STRUCTURAL] = centered[STRUCTURAL] - centered.groupby("dataset_id")[STRUCTURAL].transform("mean")
    redundancy: list[dict[str, Any]] = []
    for scope, frame in [("raw_cross_dataset", primary), ("within_dataset_centered", centered)]:
        for a, b in itertools.combinations(STRUCTURAL, 2):
            pair = frame[[a, b]].dropna()
            redundancy.append({
                "scope": scope, "signal_a": a, "signal_b": b,
                "pearson": float(pair[a].corr(pair[b], method="pearson")),
                "spearman": float(pair[a].corr(pair[b], method="spearman")),
            })
        x = frame[STRUCTURAL].replace([np.inf, -np.inf], np.nan).dropna()
        for feature in STRUCTURAL:
            others = [c for c in STRUCTURAL if c != feature]
            r2 = LinearRegression().fit(x[others], x[feature]).score(x[others], x[feature])
            redundancy.append({
                "scope": scope, "signal_a": feature, "signal_b": "VIF",
                "pearson": float(1.0 / max(1.0 - r2, 1e-12)), "spearman": np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(redundancy)


def severity_audit(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_keys = ["dataset_id", "outer_fold", "shift_family", "method", "seed"]
    values = SIGNALS + ["delta_ari"]
    wide = primary.pivot_table(index=pair_keys, columns="severity", values=values, aggfunc="first")
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for value in values:
        mild = wide[(value, "mild")]
        severe = wide[(value, "severe")]
        valid = mild.notna() & severe.notna()
        diff = severe[valid] - mild[valid]
        rows.append({
            "quantity": value,
            "n_pairs": len(diff),
            "severe_gt_mild_fraction": float((diff > 0).mean()),
            "violation_fraction": float((diff <= 0).mean()),
            "median_severe_minus_mild": float(diff.median()),
        })
        for index, delta in diff.items():
            details.append(dict(zip(pair_keys, index)) | {"quantity": value, "severe_minus_mild": float(delta), "violation": bool(delta <= 0)})
    return pd.DataFrame(rows), pd.DataFrame(details)


def family_summary(primary: pd.DataFrame, no_skill_predictions: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_csv(ROOT / "results/falsification/losfo_family_metrics.csv")
    if "feature_block" in frozen:
        learned = frozen.pivot(index="shift_family", columns="feature_block", values="mae")
    else:
        learned = frozen.set_index("shift_family")[["mae_P0", "mae_P3", "mae_P4"]].rename(columns=lambda c: c.replace("mae_", ""))
    baseline = no_skill_predictions[no_skill_predictions["evaluation"].eq("LOSFO")].groupby(
        ["outer_group", "feature_block"]
    ).apply(lambda x: mean_absolute_error(x["delta_ari"], x["predicted_delta_ari"]), include_groups=False).unstack()
    outcome = primary.groupby(["shift_family", "severity"])["delta_ari"].agg(["mean", "median", "std"]).unstack()
    sig = primary.groupby(["shift_family", "severity"])[STRUCTURAL + ["D_X"]].mean().unstack()
    outcome.columns = ["delta_ari_" + "_".join(map(str, c)) for c in outcome.columns]
    sig.columns = ["signal_" + "_".join(map(str, c)) for c in sig.columns]
    rows = learned.join(baseline).join(outcome).join(sig)
    rows.columns = ["_".join(map(str, c)).strip("_") if isinstance(c, tuple) else str(c) for c in rows.columns]
    rows = rows.reset_index().rename(columns={"index": "shift_family"})
    rows["delta04"] = rows["P0"] - rows["P4"]
    rows["delta34"] = rows["P3"] - rows["P4"]
    return rows


def dataset_influence() -> pd.DataFrame:
    deltas = pd.read_csv(ROOT / "results/falsification/paired_dataset_deltas.csv").set_index("dataset_id")["delta_04"]
    total = float(deltas.sum())
    rows: list[dict[str, Any]] = []
    for n_remove in [0, 1, 2, 3]:
        combinations = [()] if n_remove == 0 else itertools.combinations(deltas.index, n_remove)
        candidates = []
        for removed in combinations:
            remaining = deltas.drop(list(removed))
            candidates.append((float(remaining.mean()), tuple(removed)))
        # Largest remaining mean identifies the subset contributing the most negative loss.
        remaining_mean, removed = max(candidates, key=lambda item: item[0])
        removed_sum = float(deltas.loc[list(removed)].sum()) if removed else 0.0
        rows.append({
            "n_removed": n_remove,
            "removed_datasets": json.dumps(list(removed)),
            "remaining_mean_delta04": remaining_mean,
            "removed_delta04_sum": removed_sum,
            "fraction_of_net_negative_sum_attributable": removed_sum / total if total != 0 else np.nan,
            "official_verdict_changed": False,
        })
    return pd.DataFrame(rows)


def construct_validity() -> pd.DataFrame:
    real = json.loads((ROOT / "data/manifests/datasets.json").read_text(encoding="utf-8"))
    synthetic = json.loads((ROOT / "data/manifests/synthetic_manifest.json").read_text(encoding="utf-8"))
    tier_a = {"iris", "wine", "seeds", "image_segmentation"}
    phase7 = set(yaml.safe_load((ROOT / "configs/falsification.yaml").read_text(encoding="utf-8"))["datasets"])
    rows: list[dict[str, Any]] = []
    for rec in real:
        dataset = rec["dataset_id"]
        tier = "A" if dataset in tier_a else "C"
        rows.append({
            "dataset": dataset,
            "tier": tier,
            "label_semantics": "domain/species/material category with plausible geometric grouping" if tier == "A" else "supervised class or domain outcome used as external clustering proxy",
            "why_label_may_or_may_not_represent_clusters": "plausible latent grouping, but still not proof of the FCM geometry" if tier == "A" else "class label is not guaranteed to be a natural geometric cluster and may be non-convex, overlapping, or outcome-defined",
            "appropriate_use": "external hard-partition recovery benchmark" if tier == "A" else "stress test of label recovery; do not infer latent-cluster fidelity",
            "headline_or_stress_only": "headline eligible with construct caveat" if tier == "A" else "stress-only",
            "phase7_discovery_dataset": dataset in phase7,
        })
    for rec in synthetic:
        dataset = rec["family_id"]
        rows.append({
            "dataset": dataset,
            "tier": "B",
            "label_semantics": f"known latent membership from {rec['generator']}",
            "why_label_may_or_may_not_represent_clusters": "generator supplies latent component identity and stored soft truth; posterior formula must match the generator family",
            "appropriate_use": "hard and soft membership recovery with generator-correct posterior",
            "headline_or_stress_only": "headline eligible for synthetic construct only",
            "phase7_discovery_dataset": False,
        })
    return pd.DataFrame(rows)


def normalization_registry() -> dict[str, Any]:
    return {
        "status": "POST_HOC_EXPLORATORY_REGISTRY_DO_NOT_REPLACE_P4",
        "constraints": {"max_one_family_per_signal_type": True, "target_labels_used": False},
        "attempts": [
            {"name": "D_V_reference_radius", "signal": "D_V", "formula": "mean_k ||v0_k-vt_pi(k)||/(s_k^R+eps)", "motivation": "dimensionless prototype motion", "created_before_result": True, "status": "already_the_frozen_definition", "kept_or_rejected": "kept_as_existing_signal"},
            {"name": "D_H_source_entropy_relative", "signal": "D_H", "formula": "D_H/(H_source_current+eps)", "motivation": "same absolute entropy change may differ by source ambiguity", "created_before_result": False, "status": "registered_for_one_exploratory model", "kept_or_rejected": "pending_model_result"},
            {"name": "D_U_normalized_JS", "signal": "D_U_R,D_U_C,D_M", "formula": "retain normalized Jensen-Shannon definitions", "motivation": "these signals are already bounded/source-comparable", "created_before_result": True, "status": "no additional transform", "kept_or_rejected": "avoid redundant formula search"},
            {"name": "D_X_source_bandwidth", "signal": "D_X", "formula": "retain MMD with source-reference median bandwidth", "motivation": "bandwidth already derives only from source", "created_before_result": True, "status": "no additional transform", "kept_or_rejected": "avoid redundant formula search"},
        ],
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/falsification.yaml").read_text(encoding="utf-8"))
    joined = pd.read_csv(ROOT / "results/falsification/joined_evaluation_table.csv")
    primary = joined[joined["condition"].ne("clean")].copy()
    no_pred, no_units, no_metrics = no_skill(primary, cfg)
    no_comparisons = baseline_comparisons(no_units, cfg)
    atomic_csv(OUT / "hypothesis_y1_no_skill_predictions.csv", no_pred)
    atomic_csv(OUT / "hypothesis_y1_no_skill_unit_metrics.csv", no_units)
    atomic_csv(OUT / "hypothesis_y1_no_skill_metrics.csv", no_metrics)
    atomic_csv(OUT / "hypothesis_y1_no_skill_comparisons.csv", no_comparisons)

    frozen_pred = pd.read_csv(ROOT / "results/falsification/lodo_predictions.csv")
    frozen_pred = frozen_pred[frozen_pred["feature_block"].isin(["P0", "P3", "P4"])][
        ["dataset_id", "outer_fold", "condition", "method", "seed", "feature_block", "observed_delta_ari", "predicted_delta_ari"]
    ]
    baseline_pred = no_pred[no_pred["evaluation"].eq("LODO")].rename(columns={"delta_ari": "observed_delta_ari"})
    risk_input = pd.concat([
        frozen_pred,
        baseline_pred[["dataset_id", "outer_fold", "condition", "method", "seed", "feature_block", "observed_delta_ari", "predicted_delta_ari"]],
    ], ignore_index=True)
    risk, ranking = crossfit_risk_metrics(risk_input)
    atomic_csv(OUT / "hypothesis_y1_risk_classification.csv", risk)
    atomic_csv(OUT / "hypothesis_y1_risk_ranking.csv", ranking)

    meta = source_metadata(primary)
    stats, variance, relationships = signal_scale(primary, meta)
    mechanisms, redundancy = mechanism_correlations(primary, meta)
    severity, severity_detail = severity_audit(primary)
    family = family_summary(primary, no_pred)
    influence = dataset_influence()
    construct = construct_validity()
    atomic_csv(OUT / "hypothesis_y1_dataset_metadata.csv", meta)
    atomic_csv(OUT / "hypothesis_y1_signal_scale_by_dataset.csv", stats)
    atomic_csv(OUT / "hypothesis_y1_signal_scale_variance.csv", variance)
    atomic_csv(OUT / "hypothesis_y1_signal_metadata_relationships.csv", relationships)
    atomic_csv(OUT / "hypothesis_y1_signal_mechanism_correlations.csv", mechanisms)
    atomic_csv(OUT / "hypothesis_y1_signal_redundancy.csv", redundancy)
    atomic_csv(OUT / "hypothesis_y1_severity_monotonicity.csv", severity)
    atomic_csv(OUT / "hypothesis_y1_severity_violations.csv", severity_detail)
    atomic_csv(OUT / "hypothesis_y1_shift_family_mechanisms.csv", family)
    atomic_csv(OUT / "hypothesis_y1_dataset_influence.csv", influence)
    atomic_csv(OUT / "hypothesis_y1_benchmark_construct_validity.csv", construct)
    atomic_json(OUT / "hypothesis_y1_normalization_registry.json", normalization_registry())

    summary = {
        "label": "POST-HOC FAILURE FORENSICS; NOT THE PREREGISTERED RESULT",
        "no_skill": no_metrics.to_dict(orient="records"),
        "risk": risk.to_dict(orient="records"),
        "ranking": ranking.to_dict(orient="records"),
        "signal_between_within": variance.to_dict(orient="records"),
        "severity": severity.to_dict(orient="records"),
        "dataset_influence": influence.to_dict(orient="records"),
        "construct_tier_counts": construct["tier"].value_counts().sort_index().to_dict(),
    }
    atomic_json(OUT / "hypothesis_y1_statistical_forensics.json", summary)
    print(json.dumps({
        "no_skill": no_metrics.to_dict(orient="records"),
        "risk_rows": len(risk),
        "construct_rows": len(construct),
        "outputs": 19,
    }, indent=2))


if __name__ == "__main__":
    main()
