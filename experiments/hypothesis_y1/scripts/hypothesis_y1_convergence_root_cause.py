"""Characterize FCM non-convergence without altering the official optimizer.

The frozen table supplies status labels. This script reruns all unique failed
source fits, all non-Letter failed candidate fits, a deterministic stratified
sample of failed Letter fits, and matched successful controls at max_iter=600.
The longer run is diagnostic only; it is never selected using predictive P4.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from clusterdrift.methods.fcm import FCM


ARTIFACTS = ROOT / "experiments/hypothesis_y1/artifacts"
TOL = 1e-5
MAX_ITER = 600


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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def find_cache_root() -> Path:
    base = Path(os.environ["LOCALAPPDATA"]) / "clusterdrift/when-clusters-drift/phase7/phase7"
    candidates = sorted(p.parent for p in base.glob("*/.clusterdrift_phase7_cache"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Phase-7 protocol cache, found {len(candidates)}")
    return candidates[0]


def select_tasks(joined: pd.DataFrame) -> pd.DataFrame:
    task_columns = ["fit_role", "dataset_id", "outer_fold", "condition", "seed", "K", "original_converged"]
    source = joined[["dataset_id", "outer_fold", "seed", "K", "source_converged"]].drop_duplicates()
    source = source.rename(columns={"source_converged": "original_converged"})
    source["fit_role"] = "source"
    source["condition"] = "source"
    source_failed = source.loc[~source["original_converged"].astype(bool)]
    source_control = source.loc[source["original_converged"].astype(bool)].groupby("dataset_id", sort=True).head(2)

    candidate = joined[["dataset_id", "outer_fold", "condition", "seed", "K", "candidate_converged"]].drop_duplicates()
    candidate = candidate.rename(columns={"candidate_converged": "original_converged"})
    candidate["fit_role"] = "candidate"
    failed = candidate.loc[~candidate["original_converged"].astype(bool)]
    failed_other = failed.loc[failed["dataset_id"].ne("letter_recognition")]
    failed_letter = failed.loc[failed["dataset_id"].eq("letter_recognition")]
    # Cover every Letter condition and every fold/initialization combination.
    letter_condition = failed_letter.sort_values(["condition", "outer_fold", "seed"]).groupby("condition", sort=True).head(1)
    letter_fold_seed = failed_letter.sort_values(["outer_fold", "seed", "condition"]).groupby(["outer_fold", "seed"], sort=True).head(1)
    failed_sample = pd.concat([failed_other, letter_condition, letter_fold_seed]).drop_duplicates(
        ["dataset_id", "outer_fold", "condition", "seed"]
    )
    candidate_control = candidate.loc[candidate["original_converged"].astype(bool)].groupby("dataset_id", sort=True).head(2)
    tasks = pd.concat([source_failed, source_control, failed_sample, candidate_control], ignore_index=True)
    return tasks[task_columns].sort_values(["fit_role", "dataset_id", "outer_fold", "condition", "seed"]).reset_index(drop=True)


def status_at(history: list[float], iteration: int) -> str:
    if len(history) < iteration:
        return "SUCCESS_CONVERGED"
    shift = history[iteration - 1]
    if not np.isfinite(shift):
        return "NUMERICAL_FAILURE"
    if shift < TOL:
        return "SUCCESS_CONVERGED"
    if shift <= 10.0 * TOL:
        return "MAX_ITER_NEAR_CONVERGED"
    return "MAX_ITER_NONCONVERGED"


def one_fit(row: dict[str, Any], cache: Path) -> dict[str, Any]:
    ds = str(row["dataset_id"])
    fold = int(row["outer_fold"])
    seed = int(row["seed"])
    role = str(row["fit_role"])
    if role == "source":
        x_path = cache / "datasets" / f"{ds}_fold_{fold}_X_src.npy"
    else:
        condition = str(row["condition"])
        x_path = cache / "scenarios" / f"{ds}_fold_{fold}_{condition}_X_tgt.npy"
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    model = FCM(
        n_clusters=int(row["K"]),
        fuzzifier_policy="dimension_adaptive",
        random_state=seed,
        max_iter=MAX_ITER,
        tol=TOL,
        initialization="kmeans++",
    ).fit(x)
    shifts = np.asarray(model.center_shift_history_, dtype=np.float64)
    objectives = np.asarray(model.objective_history_, dtype=np.float64)
    objective_diff = np.diff(objectives)
    u_m = np.asarray(model.membership_, dtype=np.float64) ** float(model.effective_m_)
    fuzzy_mass = np.sum(u_m, axis=0)
    result = dict(row)
    result.update({
        "N": int(x.shape[0]),
        "D": int(x.shape[1]),
        "effective_m": float(model.effective_m_),
        "rerun_converged_by_600": bool(model.converged_),
        "rerun_n_iter": int(model.n_iter_),
        "rerun_status": str(model.status_),
        "status_at_150": status_at(model.center_shift_history_, 150),
        "status_at_300": status_at(model.center_shift_history_, 300),
        "status_at_600": status_at(model.center_shift_history_, 600),
        "shift_at_150": float(shifts[149]) if len(shifts) >= 150 else np.nan,
        "shift_at_300": float(shifts[299]) if len(shifts) >= 300 else np.nan,
        "final_center_shift": float(shifts[-1]),
        "final_shift_over_tol": float(shifts[-1] / TOL),
        "objective_initial": float(objectives[0]),
        "objective_final": float(objectives[-1]),
        "objective_relative_reduction": float((objectives[0] - objectives[-1]) / max(abs(objectives[0]), 1e-15)),
        "objective_increase_count": int(np.sum(objective_diff > np.maximum(1e-10, 1e-10 * np.abs(objectives[:-1])))),
        "tail_shift_increase_fraction": float(np.mean(np.diff(shifts[-20:]) > 0)) if len(shifts) >= 20 else np.nan,
        "minimum_fuzzy_mass": float(np.min(fuzzy_mass)),
        "maximum_fuzzy_mass": float(np.max(fuzzy_mass)),
        "mean_max_membership": float(np.mean(np.max(model.membership_, axis=1))),
        "degenerate": bool(model.degenerate_solution_),
        "diagnostic_input_path": x_path.relative_to(cache).as_posix(),
    })
    return result


def main() -> None:
    joined = pd.read_csv(ROOT / "results/falsification/joined_evaluation_table.csv")
    tasks = select_tasks(joined)
    cache = find_cache_root()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one_fit, row._asdict(), cache) for row in tasks.itertuples(index=False)]
        for future in as_completed(futures):
            rows.append(future.result())
    result = pd.DataFrame(rows).sort_values(
        ["fit_role", "dataset_id", "outer_fold", "condition", "seed"]
    ).reset_index(drop=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    atomic_csv(ARTIFACTS / "hypothesis_y1_convergence_root_cause_runs.csv", result)
    summary = {
        "scope": {
            "all_unique_original_failed_source_fits": True,
            "all_non_letter_failed_candidate_fits": True,
            "letter_failed_candidate_sampling": "one per condition plus one per fold-seed combination",
            "matched_controls_per_dataset": 2,
            "n_diagnostic_fits": len(result),
        },
        "original_failed_rows_in_sample": int((~result["original_converged"].astype(bool)).sum()),
        "failed_sample_converged_by_300": int(((~result["original_converged"].astype(bool)) & result["status_at_300"].eq("SUCCESS_CONVERGED")).sum()),
        "failed_sample_converged_by_600": int(((~result["original_converged"].astype(bool)) & result["status_at_600"].eq("SUCCESS_CONVERGED")).sum()),
        "failed_sample_near_at_150": int(((~result["original_converged"].astype(bool)) & result["status_at_150"].eq("MAX_ITER_NEAR_CONVERGED")).sum()),
        "objective_increase_runs": int((result["objective_increase_count"] > 0).sum()),
        "numerical_or_degenerate_runs": int((result["degenerate"].astype(bool) | result["rerun_status"].isin(["EMPTY_CLUSTER", "NUMERICAL_FAILURE"])).sum()),
        "by_dataset": result.groupby(["fit_role", "dataset_id"], sort=True).agg(
            runs=("dataset_id", "size"),
            original_failures=("original_converged", lambda x: int((~x.astype(bool)).sum())),
            median_final_shift_over_tol=("final_shift_over_tol", "median"),
            converged_by_300=("status_at_300", lambda x: int((x == "SUCCESS_CONVERGED").sum())),
            converged_by_600=("status_at_600", lambda x: int((x == "SUCCESS_CONVERGED").sum())),
        ).reset_index().to_dict(orient="records"),
    }
    atomic_text(
        ARTIFACTS / "hypothesis_y1_convergence_root_cause.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
