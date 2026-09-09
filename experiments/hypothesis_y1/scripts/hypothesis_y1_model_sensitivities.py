"""Post-hoc grouped model sensitivities for Hypothesis Y1.

Runs the preregistered HGBR procedure on convergence subsets, two narrowly
motivated control blocks, and grouped within-dataset folds.  These results are
diagnostic and never overwrite the frozen Phase-7 verdict.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/hypothesis_y1"
OUT = BASE / "artifacts"
INDEPENDENT = BASE / "scripts/hypothesis_y1_independent_phase7_replication.py"


def load_independent():
    spec = importlib.util.spec_from_file_location("hypothesis_y1_independent", INDEPENDENT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent Y1 evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def ordered_families(cfg: dict[str, Any], present: set[str]) -> list[str]:
    result: list[str] = []
    for condition in cfg["conditions"]:
        family = condition.rsplit("_", 1)[0]
        if condition != "clean" and family in present and family not in result:
            result.append(family)
    return result


def summarize(mod, rows: pd.DataFrame, cfg: dict[str, Any], evaluation: str) -> tuple[dict[str, Any], pd.DataFrame]:
    if evaluation == "LODO":
        present = set(rows["outer_group"])
        order = [x for x in cfg["datasets"] if x in present]
    else:
        order = ordered_families(cfg, set(rows["outer_group"]))
    aggregate, wide = mod.summarize_outer(rows, "HGBR", order)
    seed = int(cfg["bootstrap"]["seed"])
    reps = int(cfg["bootstrap"]["repetitions"])
    boot04 = mod.paired_bootstrap(wide["delta_04"].to_numpy(), reps, seed)
    boot34 = mod.paired_bootstrap(wide["delta_34"].to_numpy(), reps, seed)
    return {
        "evaluation": evaluation,
        "n_units": len(wide),
        "mae": aggregate,
        "wins_04": int((wide["delta_04"] > 0).sum()),
        "wins_34": int((wide["delta_34"] > 0).sum()),
        "bootstrap_delta_04": boot04,
        "bootstrap_delta_34": boot34,
    }, wide


def convergence_sensitivities(mod, primary: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    definitions = {
        "S0_all_original_rows": np.ones(len(primary), dtype=bool),
        "S1_source_converged": primary["source_converged"].astype(bool).to_numpy(),
        "S2_candidate_converged": primary["candidate_converged"].astype(bool).to_numpy(),
        "S3_both_converged": (primary["source_converged"].astype(bool) & primary["candidate_converged"].astype(bool)).to_numpy(),
    }
    all_rows: list[pd.DataFrame] = []
    summary: dict[str, Any] = {"label": "POST-HOC CONVERGENCE SENSITIVITY", "sensitivities": {}}
    for name, mask in definitions.items():
        subset = primary.loc[mask].copy()
        if name == "S0_all_original_rows":
            lodo = pd.read_csv(OUT / "hypothesis_y1_independent_lodo.csv")
            losfo = pd.read_csv(OUT / "hypothesis_y1_independent_losfo.csv")
            lodo = lodo[lodo["regressor"].eq("HGBR")].copy()
            losfo = losfo[losfo["regressor"].eq("HGBR")].copy()
        else:
            lodo = mod.run_outer(subset, cfg, "LODO", workers=6, regressors=("HGBR",))
            losfo = mod.run_outer(subset, cfg, "LOSFO", workers=6, regressors=("HGBR",))
        lodo["sensitivity"] = name
        losfo["sensitivity"] = name
        all_rows.extend([lodo, losfo])
        lodo_summary, _ = summarize(mod, lodo, cfg, "LODO")
        losfo_summary, _ = summarize(mod, losfo, cfg, "LOSFO")
        summary["sensitivities"][name] = {
            "n_rows": len(subset),
            "n_datasets": int(subset["dataset_id"].nunique()),
            "excluded_datasets": sorted(set(cfg["datasets"]) - set(subset["dataset_id"])),
            "LODO": lodo_summary,
            "LOSFO": losfo_summary,
        }
        atomic_json(OUT / "hypothesis_y1_convergence_sensitivity_partial.json", summary)
    return pd.concat(all_rows, ignore_index=True), summary


def clean_reference_deltas(joined: pd.DataFrame) -> pd.DataFrame:
    controls = ["FPC", "PE", "XB_soft_m2", "silhouette"]
    clean = joined[joined["condition"].eq("clean")][
        ["dataset_id", "outer_fold", "method", "seed"] + controls
    ].rename(columns={name: f"{name}_clean_reference" for name in controls})
    shifted = joined[joined["condition"].ne("clean")].merge(
        clean, on=["dataset_id", "outer_fold", "method", "seed"], validate="many_to_one"
    )
    for name in controls:
        shifted[f"Delta_{name}"] = shifted[name] - shifted[f"{name}_clean_reference"]
    return shifted


def exploratory_blocks(mod, primary: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = clean_reference_deltas(pd.read_csv(ROOT / "results/falsification/joined_evaluation_table.csv"))
    frame["D_H_source_relative"] = frame["D_H"] / (frame["entropy_source_current"].abs() + 1e-12)
    blocks = {
        "Y1_VALIDITY_DELTA_CLEAN_REFERENCE": ["D_X", "Delta_FPC", "Delta_PE", "Delta_XB_soft_m2", "Delta_silhouette"],
        "Y1_P4_DH_SOURCE_RELATIVE": ["D_X", "D_U_R", "D_U_C", "D_V", "D_H_source_relative", "D_M"],
    }
    all_rows: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {
        "label": "POST-HOC EXPLORATORY CONTROLS",
        "reference_definition": "same dataset/fold/seed unshifted target fold; label-free and train-independent, but not the source A_R probe",
        "blocks": {},
    }
    for name, features in blocks.items():
        for evaluation in ["LODO", "LOSFO"]:
            rows = mod.run_outer(frame, cfg, evaluation, workers=6, blocks={name: features}, regressors=("HGBR",))
            rows["analysis_block"] = name
            all_rows.append(rows)
            # A one-block summary cannot use P0/P3/P4 helper.
            if evaluation == "LODO":
                order = [x for x in cfg["datasets"] if x in set(rows["outer_group"])]
            else:
                order = ordered_families(cfg, set(rows["outer_group"]))
            units = rows.set_index("outer_group").loc[order]
            summaries["blocks"].setdefault(name, {})[evaluation] = {
                "mae": float(np.average(units["mae"], weights=units["n_test"])),
                "rmse": float(np.sqrt(np.average(units["rmse"] ** 2, weights=units["n_test"]))),
                "median_unit_mae": float(units["mae"].median()),
                "n_units": len(units),
            }
    return pd.concat(all_rows, ignore_index=True), summaries


def one_within_job(
    mod,
    data: pd.DataFrame,
    cfg: dict[str, Any],
    dataset: str,
    outer_fold: int,
    block: str,
    features: list[str] | None,
) -> dict[str, Any]:
    test = data[(data["dataset_id"] == dataset) & (data["outer_fold"] == outer_fold)]
    train = data[(data["dataset_id"] == dataset) & (data["outer_fold"] != outer_fold)]
    y_train = train["delta_ari"].to_numpy(float)
    y_test = test["delta_ari"].to_numpy(float)
    if features is None:
        value = float(np.median(y_train))
        pred = np.full(len(test), value)
        selected = {"constant": value}
    else:
        scenario_groups = (train["outer_fold"].astype(str) + "|" + train["condition"].astype(str)).to_numpy()
        model, prep, selected = mod.fit_hgbr(
            train[features].to_numpy(float), y_train, scenario_groups, cfg
        )
        pred = model.predict(prep.transform(test[features].to_numpy(float)))
    return {
        "dataset": dataset,
        "heldout_outer_fold": outer_fold,
        "feature_block": block,
        "n_train": len(train),
        "n_test": len(test),
        "grouping": "outer fold held out; inner GroupKFold on outer_fold|condition keeps all seeds of a scenario together",
        "selected_params": json.dumps(selected, sort_keys=True),
        **mod.metrics(y_test, pred),
    }


def within_dataset(mod, primary: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks: dict[str, list[str] | None] = {
        "P0": cfg["feature_blocks"]["P0"],
        "P2": cfg["feature_blocks"]["P2"],
        "P3": cfg["feature_blocks"]["P3"],
        "P4": cfg["feature_blocks"]["P4"],
        "B0_median": None,
    }
    tasks = [
        (dataset, fold, block, features)
        for dataset in cfg["datasets"]
        for fold in cfg["outer_folds"]
        for block, features in blocks.items()
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one_within_job, mod, primary, cfg, *task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    detail = pd.DataFrame(rows).sort_values(["dataset", "feature_block", "heldout_outer_fold"]).reset_index(drop=True)
    overall = detail.groupby(["dataset", "feature_block"], sort=True).agg(
        within_mae=("mae", "mean"),
        within_rmse=("rmse", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
        within_r2=("r2", "mean"),
    ).reset_index().pivot(index="dataset", columns="feature_block", values="within_mae")
    cross = pd.read_csv(ROOT / "results/falsification/lodo_dataset_metrics.csv").set_index("dataset_id")
    comparison = pd.DataFrame(index=overall.index)
    for block in blocks:
        comparison[f"within_{block}"] = overall[block]
    comparison["within_delta04"] = comparison["within_P0"] - comparison["within_P4"]
    comparison["LODO_P0"] = cross["mae_P0"]
    comparison["LODO_P4"] = cross["mae_P4"]
    comparison["LODO_delta04"] = comparison["LODO_P0"] - comparison["LODO_P4"]
    return detail, comparison.reset_index()


def main() -> None:
    started = time.perf_counter()
    mod = load_independent()
    cfg = yaml.safe_load((ROOT / "configs/falsification.yaml").read_text(encoding="utf-8"))
    joined = pd.read_csv(ROOT / "results/falsification/joined_evaluation_table.csv")
    primary = joined[joined["condition"].ne("clean")].copy()

    sensitivity_rows, sensitivity_summary = convergence_sensitivities(mod, primary, cfg)
    atomic_csv(OUT / "hypothesis_y1_convergence_sensitivity_metrics.csv", sensitivity_rows)
    atomic_json(OUT / "hypothesis_y1_convergence_sensitivity.json", sensitivity_summary)

    control_rows, control_summary = exploratory_blocks(mod, primary, cfg)
    atomic_csv(OUT / "hypothesis_y1_exploratory_control_metrics.csv", control_rows)
    atomic_json(OUT / "hypothesis_y1_exploratory_controls.json", control_summary)

    within_detail, within_comparison = within_dataset(mod, primary, cfg)
    atomic_csv(OUT / "hypothesis_y1_within_dataset_metrics.csv", within_detail)
    atomic_csv(OUT / "hypothesis_y1_within_vs_cross_dataset.csv", within_comparison)
    summary = {
        "label": "POST-HOC GROUPED MODEL SENSITIVITIES; NOT CONFIRMATORY",
        "runtime_seconds": time.perf_counter() - started,
        "convergence": sensitivity_summary,
        "controls": control_summary,
        "within_structural_help_count": int((within_comparison["within_delta04"] > 0).sum()),
        "cross_structural_help_count": int((within_comparison["LODO_delta04"] > 0).sum()),
    }
    atomic_json(OUT / "hypothesis_y1_model_sensitivities.json", summary)
    print(json.dumps({
        "runtime_seconds": summary["runtime_seconds"],
        "within_structural_help_count": summary["within_structural_help_count"],
        "cross_structural_help_count": summary["cross_structural_help_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
