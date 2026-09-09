"""Independent Phase-7 replication for Hypothesis Y1.

Core calculations intentionally do not import any clusterdrift falsification
module. Inputs are limited to the two frozen Pass-A/Pass-B CSVs and the frozen
falsification YAML. Outputs live only under experiments/hypothesis_y1.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "hypothesis_y1" / "artifacts"
KEYS = ["dataset_id", "outer_fold", "condition", "method", "seed"]

EXPECTED = {
    "P0": 0.04670466347898161,
    "P3": 0.06392587674262015,
    "P4": 0.050894704583937404,
    "r_04": -0.08971354877316323,
    "r_34": 0.20384815700141518,
    "dataset_wins_04": 5,
    "dataset_wins_34": 10,
    "family_wins_04": 4,
    "family_wins_34": 3,
    "bootstrap_lodo_04": [-0.0118544034051598, 0.002436548707505495],
    "bootstrap_lodo_34": [0.002950967679446092, 0.02330558734454633],
    "verdict": "FAILS_PRIMARY_FALSIFICATION",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


class MedianPreprocessor:
    def __init__(self, scale: bool):
        self.scale = scale
        self.median: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MedianPreprocessor":
        x = np.asarray(x, dtype=np.float64)
        self.median = np.zeros(x.shape[1], dtype=np.float64)
        for j in range(x.shape[1]):
            valid = x[:, j][~np.isnan(x[:, j])]
            self.median[j] = np.median(valid) if valid.size else 0.0
        if self.scale:
            imputed = self.transform(x, apply_scale=False)
            self.mean = np.mean(imputed, axis=0)
            self.std = np.std(imputed, axis=0)
            self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray, apply_scale: bool = True) -> np.ndarray:
        if self.median is None:
            raise RuntimeError("preprocessor not fitted")
        out = np.asarray(x, dtype=np.float64).copy()
        for j in range(out.shape[1]):
            missing = np.isnan(out[:, j])
            out[missing, j] = self.median[j]
        if self.scale and apply_scale:
            if self.mean is None or self.std is None:
                raise RuntimeError("scaler not fitted")
            out = (out - self.mean) / self.std
        return out


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = y - pred
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_absolute_error": float(np.median(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
    }


def hgbr_candidates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = cfg["primary_regressor"]["grid"]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def fit_hgbr(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: dict[str, Any]
) -> tuple[HistGradientBoostingRegressor, MedianPreprocessor, dict[str, Any]]:
    cv = GroupKFold(n_splits=cfg["inner_cv"]["n_splits"])
    candidates: list[tuple[float, dict[str, Any]]] = []
    for params in hgbr_candidates(cfg):
        fold_scores = []
        for train_idx, val_idx in cv.split(x, y, groups):
            prep = MedianPreprocessor(scale=False).fit(x[train_idx])
            model = HistGradientBoostingRegressor(
                loss=cfg["primary_regressor"]["loss"],
                random_state=cfg["primary_regressor"]["random_state"],
                early_stopping=cfg["primary_regressor"]["early_stopping"],
                min_samples_leaf=cfg["primary_regressor"]["min_samples_leaf"],
                **params,
            )
            model.fit(prep.transform(x[train_idx]), y[train_idx])
            fold_scores.append(mae(y[val_idx], model.predict(prep.transform(x[val_idx]))))
        candidates.append((float(np.mean(fold_scores)), params))
    best = min(score for score, _ in candidates)
    tied = [item for item in candidates if abs(item[0] - best) <= 1e-7]
    tied.sort(
        key=lambda item: (
            item[1].get("max_leaf_nodes", 31),
            item[1].get("max_iter", 100),
            -item[1].get("l2_regularization", 0.0),
            item[1].get("learning_rate", 0.1),
        )
    )
    selected = tied[0][1]
    prep = MedianPreprocessor(scale=False).fit(x)
    model = HistGradientBoostingRegressor(
        loss=cfg["primary_regressor"]["loss"],
        random_state=cfg["primary_regressor"]["random_state"],
        early_stopping=cfg["primary_regressor"]["early_stopping"],
        min_samples_leaf=cfg["primary_regressor"]["min_samples_leaf"],
        **selected,
    )
    model.fit(prep.transform(x), y)
    return model, prep, selected


def fit_ridge(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: dict[str, Any]
) -> tuple[Ridge, MedianPreprocessor, dict[str, Any]]:
    cv = GroupKFold(n_splits=cfg["inner_cv"]["n_splits"])
    candidates: list[tuple[float, float]] = []
    for alpha in sorted(cfg["linear_control"]["alphas"]):
        fold_scores = []
        for train_idx, val_idx in cv.split(x, y, groups):
            prep = MedianPreprocessor(scale=True).fit(x[train_idx])
            model = Ridge(alpha=alpha)
            model.fit(prep.transform(x[train_idx]), y[train_idx])
            fold_scores.append(mae(y[val_idx], model.predict(prep.transform(x[val_idx]))))
        candidates.append((float(np.mean(fold_scores)), float(alpha)))
    best = min(score for score, _ in candidates)
    tied = [item for item in candidates if abs(item[0] - best) <= 1e-7]
    tied.sort(key=lambda item: -item[1])
    selected = tied[0][1]
    prep = MedianPreprocessor(scale=True).fit(x)
    model = Ridge(alpha=selected)
    model.fit(prep.transform(x), y)
    return model, prep, {"alpha": selected}


def one_outer_job(
    data: pd.DataFrame,
    cfg: dict[str, Any],
    eval_type: str,
    outer_group: str,
    block: str,
    features: list[str],
    regressor: str,
) -> dict[str, Any]:
    if eval_type == "LODO":
        test_mask = data["dataset_id"].eq(outer_group)
    else:
        test_mask = data["shift_family"].eq(outer_group)
    train = data.loc[~test_mask]
    test = data.loc[test_mask]
    x_train = train[features].to_numpy(dtype=np.float64)
    y_train = train["delta_ari"].to_numpy(dtype=np.float64)
    groups = train["dataset_id"].to_numpy()
    x_test = test[features].to_numpy(dtype=np.float64)
    y_test = test["delta_ari"].to_numpy(dtype=np.float64)
    if regressor == "HGBR":
        model, prep, selected = fit_hgbr(x_train, y_train, groups, cfg)
    else:
        model, prep, selected = fit_ridge(x_train, y_train, groups, cfg)
    pred = model.predict(prep.transform(x_test))
    residual = y_test - pred
    result = {
        "evaluation": eval_type,
        "outer_group": outer_group,
        "regressor": regressor,
        "feature_block": block,
        "n_test": int(len(test)),
        "selected_params": json.dumps(selected, sort_keys=True),
        "sum_absolute_error": float(np.sum(np.abs(residual))),
        "sum_squared_error": float(np.sum(residual**2)),
        "sum_observed": float(np.sum(y_test)),
        "sum_observed_squared": float(np.sum(y_test**2)),
        **metrics(y_test, pred),
    }
    return result


def run_outer(
    data: pd.DataFrame,
    cfg: dict[str, Any],
    eval_type: str,
    workers: int = 4,
    blocks: dict[str, list[str]] | None = None,
    regressors: tuple[str, ...] = ("HGBR", "Ridge"),
) -> pd.DataFrame:
    if blocks is None:
        blocks = {name: cfg["feature_blocks"][name] for name in ["P0", "P3", "P4"]}
    if eval_type == "LODO":
        present = set(data["dataset_id"].unique())
        groups = [name for name in cfg["datasets"] if name in present]
    else:
        present = set(data["shift_family"].unique())
        preregistered = []
        for condition in cfg["conditions"]:
            family = condition.rsplit("_", 1)[0]
            if condition != "clean" and family not in preregistered:
                preregistered.append(family)
        groups = [name for name in preregistered if name in present]
    tasks = [
        (group, name, features, regressor)
        for group in groups
        for name, features in blocks.items()
        for regressor in regressors
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(one_outer_job, data, cfg, eval_type, *task): task
            for task in tasks
        }
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows).sort_values(
        ["regressor", "feature_block", "outer_group"]
    ).reset_index(drop=True)


def paired_bootstrap(values: np.ndarray, repetitions: int, seed: int) -> dict[str, float]:
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


def summarize_outer(
    rows: pd.DataFrame,
    regressor: str,
    order: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    sub = rows[rows["regressor"].eq(regressor)]
    # Every outer unit contains the same number of rows in this frozen design, so
    # its overall MAE is the arithmetic mean of unit MAEs.
    overall = {
        block: float(sub.loc[sub["feature_block"].eq(block), "mae"].mean())
        for block in ["P0", "P3", "P4"]
    }
    wide = sub.pivot(index="outer_group", columns="feature_block", values="mae").loc[order].reset_index()
    wide["delta_04"] = wide["P0"] - wide["P4"]
    wide["delta_34"] = wide["P3"] - wide["P4"]
    return overall, wide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Recompute aggregation/bootstrap checks from this script's completed independent model fits.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    signals_path = ROOT / "results/falsification/signals_label_free.csv"
    quality_path = ROOT / "results/falsification/quality_evaluation_only.csv"
    config_path = ROOT / "configs/falsification.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    signals = pd.read_csv(signals_path)
    quality = pd.read_csv(quality_path)
    if signals.duplicated(KEYS).any() or quality.duplicated(KEYS).any():
        raise RuntimeError("duplicate frozen join key")
    if set(map(tuple, signals[KEYS].to_numpy())) != set(map(tuple, quality[KEYS].to_numpy())):
        raise RuntimeError("Pass-A/Pass-B key universes differ")
    quality_columns = KEYS + ["delta_ari"]
    joined = signals.merge(quality[quality_columns], on=KEYS, validate="one_to_one")
    joined = joined.sort_values(KEYS).reset_index(drop=True)
    joined["shift_family"] = joined["condition"].str.rsplit("_", n=1).str[0]
    primary = joined.loc[joined["condition"].ne("clean")].copy()
    if len(joined) != 4500 or len(primary) != 4200:
        raise RuntimeError(f"unexpected row counts: joined={len(joined)}, primary={len(primary)}")

    prior_summary: dict[str, Any] = {}
    summary_path = OUT / "hypothesis_y1_independent_replication.json"
    if args.summarize_existing:
        if not summary_path.exists():
            raise RuntimeError("--summarize-existing requires a completed independent replication JSON")
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        failure_path = OUT / "hypothesis_y1_independent_replication_initial_order_failure.json"
        if prior_summary.get("agreement", {}).get("status") == "FAILED" and not failure_path.exists():
            atomic_text(failure_path, json.dumps(prior_summary, indent=2, sort_keys=True) + "\n")
        core_columns = [
            "evaluation", "outer_group", "regressor", "feature_block", "n_test",
            "selected_params", "mae", "rmse", "median_absolute_error", "r2",
        ]
        lodo = pd.read_csv(OUT / "hypothesis_y1_independent_lodo.csv")[core_columns]
        losfo = pd.read_csv(OUT / "hypothesis_y1_independent_losfo.csv")[core_columns]
    else:
        lodo = run_outer(primary, cfg, "LODO")
        losfo = run_outer(primary, cfg, "LOSFO")
    dataset_order = [name for name in cfg["datasets"] if name in set(primary["dataset_id"])]
    family_order: list[str] = []
    for condition in cfg["conditions"]:
        family = condition.rsplit("_", 1)[0]
        if condition != "clean" and family in set(primary["shift_family"]) and family not in family_order:
            family_order.append(family)
    h_lodo, lodo_wide = summarize_outer(lodo, "HGBR", dataset_order)
    h_losfo, losfo_wide = summarize_outer(losfo, "HGBR", family_order)
    r_lodo, _ = summarize_outer(lodo, "Ridge", dataset_order)
    r_losfo, _ = summarize_outer(losfo, "Ridge", family_order)

    reps = int(cfg["bootstrap"]["repetitions"])
    seed = int(cfg["bootstrap"]["seed"])
    boot_rows = []
    boots: dict[str, dict[str, float]] = {}
    for name, values in {
        "lodo_delta_04": lodo_wide["delta_04"].to_numpy(),
        "lodo_delta_34": lodo_wide["delta_34"].to_numpy(),
        "losfo_delta_04": losfo_wide["delta_04"].to_numpy(),
        "losfo_delta_34": losfo_wide["delta_34"].to_numpy(),
    }.items():
        result = paired_bootstrap(values, reps, seed)
        boots[name] = result
        boot_rows.append({"comparison": name, **result})

    r04 = (h_lodo["P0"] - h_lodo["P4"]) / h_lodo["P0"]
    r34 = (h_lodo["P3"] - h_lodo["P4"]) / h_lodo["P3"]
    rules = {
        "rule_1_lodo_mae_p4_lt_p0": h_lodo["P4"] < h_lodo["P0"],
        "rule_2_r04_ge_005_or_ci_gt_0": r04 >= 0.05 or boots["lodo_delta_04"]["mean_ci_lower"] > 0,
        "rule_3_lodo_mae_p4_lt_p3": h_lodo["P4"] < h_lodo["P3"],
        "rule_4_r34_ge_002_or_ci_gt_0": r34 >= 0.02 or boots["lodo_delta_34"]["mean_ci_lower"] > 0,
        "rule_5_median_dataset_delta_04_gt_0": float(lodo_wide["delta_04"].median()) > 0,
        "rule_6_dataset_wins_04_ge_7_of_12": int((lodo_wide["delta_04"] > 0).sum()) >= 7,
        "rule_7_family_wins_04_ge_4_of_7": int((losfo_wide["delta_04"] > 0).sum()) >= 4,
        "rule_8_family_wins_34_ge_4_of_7": int((losfo_wide["delta_34"] > 0).sum()) >= 4,
    }
    if all(rules.values()):
        verdict = "SURVIVES_STRONGLY"
    elif rules["rule_1_lodo_mae_p4_lt_p0"] and rules["rule_3_lodo_mae_p4_lt_p3"]:
        verdict = "BORDERLINE"
    else:
        verdict = "FAILS_PRIMARY_FALSIFICATION"

    comparisons = {
        "P0": abs(h_lodo["P0"] - EXPECTED["P0"]),
        "P3": abs(h_lodo["P3"] - EXPECTED["P3"]),
        "P4": abs(h_lodo["P4"] - EXPECTED["P4"]),
        "r_04": abs(r04 - EXPECTED["r_04"]),
        "r_34": abs(r34 - EXPECTED["r_34"]),
        "bootstrap_lodo_04_lower": abs(boots["lodo_delta_04"]["mean_ci_lower"] - EXPECTED["bootstrap_lodo_04"][0]),
        "bootstrap_lodo_04_upper": abs(boots["lodo_delta_04"]["mean_ci_upper"] - EXPECTED["bootstrap_lodo_04"][1]),
        "bootstrap_lodo_34_lower": abs(boots["lodo_delta_34"]["mean_ci_lower"] - EXPECTED["bootstrap_lodo_34"][0]),
        "bootstrap_lodo_34_upper": abs(boots["lodo_delta_34"]["mean_ci_upper"] - EXPECTED["bootstrap_lodo_34"][1]),
    }
    discrete_match = {
        "dataset_wins_04": int((lodo_wide["delta_04"] > 0).sum()) == EXPECTED["dataset_wins_04"],
        "dataset_wins_34": int((lodo_wide["delta_34"] > 0).sum()) == EXPECTED["dataset_wins_34"],
        "family_wins_04": int((losfo_wide["delta_04"] > 0).sum()) == EXPECTED["family_wins_04"],
        "family_wins_34": int((losfo_wide["delta_34"] > 0).sum()) == EXPECTED["family_wins_34"],
        "verdict": verdict == EXPECTED["verdict"],
    }
    tolerance = 1e-12
    agreement = max(comparisons.values()) <= tolerance and all(discrete_match.values())

    OUT.mkdir(parents=True, exist_ok=True)
    lodo_out = lodo.merge(lodo_wide, on="outer_group", how="left", suffixes=("", "_hgbr_unit"))
    losfo_out = losfo.merge(losfo_wide, on="outer_group", how="left", suffixes=("", "_hgbr_unit"))
    atomic_csv(OUT / "hypothesis_y1_independent_lodo.csv", lodo_out)
    atomic_csv(OUT / "hypothesis_y1_independent_losfo.csv", losfo_out)
    atomic_csv(OUT / "hypothesis_y1_independent_bootstrap.csv", pd.DataFrame(boot_rows))
    elapsed = time.perf_counter() - started
    summary = {
        "audit": "hypothesis_y1_independent_phase7_replication",
        "input_sha256": {
            "signals_label_free.csv": sha256(signals_path),
            "quality_evaluation_only.csv": sha256(quality_path),
            "falsification.yaml": sha256(config_path),
        },
        "input_rows": {"signals": len(signals), "quality": len(quality), "joined": len(joined), "primary": len(primary)},
        "hgb_regression": {"LODO": h_lodo, "LOSFO": h_losfo},
        "ridge_regression": {"LODO": r_lodo, "LOSFO": r_losfo},
        "lodo_dataset_deltas": lodo_wide.to_dict(orient="records"),
        "losfo_family_deltas": losfo_wide.to_dict(orient="records"),
        "bootstrap": boots,
        "relative_improvement": {"r_04": r04, "r_34": r34},
        "rules": rules,
        "verdict": verdict,
        "agreement": {
            "status": "PASSED" if agreement else "FAILED",
            "absolute_differences": comparisons,
            "discrete_matches": discrete_match,
            "tolerance": tolerance,
        },
        "runtime": {
            "seconds": elapsed,
            "model_fit_seconds": prior_summary.get("runtime", {}).get("seconds", elapsed),
            "aggregation_recovery": bool(args.summarize_existing),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    atomic_text(
        OUT / "hypothesis_y1_independent_replication.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({"agreement": summary["agreement"]["status"], "verdict": verdict, "LODO": h_lodo, "runtime_seconds": summary["runtime"]["seconds"]}, indent=2))
    if not agreement:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
