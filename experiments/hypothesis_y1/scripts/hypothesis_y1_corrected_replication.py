"""Post-hoc corrected methodological replication for Hypothesis Y1.

This is explicitly not a new confirmatory Phase 7.  It regenerates the complete
12-dataset controlled-shift panel on duplicate-grouped/class-safeguarded Y1
folds.  A 150-iteration arm isolates split correction; a 600-iteration arm
implements the outcome-blind convergence policy and requires both fits to
converge.  Per-fold checkpoints make the long computation safely resumable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from clusterdrift.data.preprocess import build_preprocessor
from clusterdrift.methods.fcm import FCM
from clusterdrift.probes.hashing import derive_current_probe_seed, derive_reference_probe_seed
from clusterdrift.probes.selection import select_current_probe_positions, select_reference_probe_positions
from clusterdrift.shifts.engine import ShiftEngine
from clusterdrift.signals.engine import SignalCache, SignalEngine
from clusterdrift.signals.validity import compute_validity_controls


BASE = ROOT / "experiments/hypothesis_y1"
WORK = BASE / "corrected_replication"
CACHE = WORK / "cache"
ARTIFACTS = BASE / "artifacts"
INDEPENDENT = BASE / "scripts/hypothesis_y1_independent_phase7_replication.py"
KEYS = ["dataset_id", "outer_fold", "condition", "method", "seed"]


class PrefitSignalEngine(SignalEngine):
    """Signal engine whose next candidate model is supplied by the Y1 policy."""

    next_candidate: FCM | None = None

    def _fit_model(self, method: str, K: int, seed: int, X: np.ndarray) -> Any:
        if self.next_candidate is None:
            raise RuntimeError("Y1 candidate model was not supplied")
        model = self.next_candidate
        self.next_candidate = None
        return model


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def matrix_hash(matrix: np.ndarray, metadata: dict[str, Any]) -> str:
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    h.update(np.ascontiguousarray(matrix, dtype=np.float64).tobytes())
    return h.hexdigest()


def frame_hash(frame: pd.DataFrame, row_map: np.ndarray, metadata: dict[str, Any]) -> str:
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(np.uint64).tobytes())
    h.update(np.ascontiguousarray(row_map, dtype=np.int64).tobytes())
    return h.hexdigest()


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


def fit_fcm(x: np.ndarray, K: int, seed: int, max_iter: int) -> FCM:
    return FCM(
        n_clusters=K,
        random_state=seed,
        fuzzifier_policy="dimension_adaptive",
        initialization="kmeans++",
        max_iter=max_iter,
        tol=1e-5,
    ).fit(x)


def status(model: FCM, horizon: int) -> str:
    centers = np.asarray(model.cluster_centers_)
    shifts = np.asarray(model.center_shift_history_, dtype=float)
    if not np.isfinite(centers).all() or not np.isfinite(shifts).all():
        return "NUMERICAL_FAILURE"
    if bool(model.degenerate_solution_):
        return "DEGENERATE"
    if str(model.status_) == "EMPTY_CLUSTER":
        return "EMPTY_CLUSTER"
    if bool(model.converged_) and len(shifts) and shifts[-1] < 1e-5:
        return "SUCCESS_CONVERGED"
    if len(shifts) >= horizon and shifts[-1] <= 1e-4:
        return "MAX_ITER_NEAR_CONVERGED"
    return "MAX_ITER_NONCONVERGED"


def pair_models(x: np.ndarray, K: int, seed: int) -> tuple[FCM, FCM]:
    model150 = fit_fcm(x, K, seed, 150)
    model600 = model150 if status(model150, 150) == "SUCCESS_CONVERGED" else fit_fcm(x, K, seed, 600)
    return model150, model600


def load_configs() -> dict[str, Any]:
    names = ["falsification", "shifts", "probes", "signals", "alignment", "methods", "preprocessing"]
    return {name: yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text(encoding="utf-8")) for name in names}


def task_fingerprint(dataset: str, fold: int, configs: dict[str, Any]) -> str:
    return canonical_hash({
        "policy": "hypothesis_y1_corrected_replication_v1",
        "dataset": dataset,
        "fold": fold,
        "features": sha256(ROOT / f"data/canonical/controlled/{dataset}/features.parquet"),
        "labels": sha256(ROOT / f"data/canonical/controlled/{dataset}/labels.parquet"),
        "metadata": sha256(ROOT / f"data/canonical/controlled/{dataset}/metadata.json"),
        "split": sha256(BASE / f"corrected_splits/{dataset}/hypothesis_y1_fold_{fold}.npz"),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "configs": {name: canonical_hash(value) for name, value in configs.items()},
        "convergence_policy": sha256(BASE / "configs/hypothesis_y1_convergence_policy.yaml"),
    })


def run_fold(dataset: str, fold: int) -> dict[str, Any]:
    configs = load_configs()
    fingerprint = task_fingerprint(dataset, fold, configs)
    csv_path = CACHE / dataset / f"hypothesis_y1_fold_{fold}_rows.csv"
    json_path = CACHE / dataset / f"hypothesis_y1_fold_{fold}_checkpoint.json"
    if csv_path.exists() and json_path.exists():
        checkpoint = json.loads(json_path.read_text(encoding="utf-8"))
        if checkpoint.get("input_fingerprint") == fingerprint and checkpoint.get("csv_sha256") == sha256(csv_path):
            return checkpoint

    ds_dir = ROOT / f"data/canonical/controlled/{dataset}"
    features = pd.read_parquet(ds_dir / "features.parquet")
    labels = pd.read_parquet(ds_dir / "labels.parquet").iloc[:, 0].to_numpy()
    metadata = json.loads((ds_dir / "metadata.json").read_text(encoding="utf-8"))
    roles = metadata.get("feature_roles", {column: "numeric" for column in features.columns})
    K = int(metadata["n_classes"])
    split_path = BASE / f"corrected_splits/{dataset}/hypothesis_y1_fold_{fold}.npz"
    with np.load(split_path, allow_pickle=False) as split:
        source_indices = split["source_indices"].astype(np.int64)
        target_indices = split["target_indices"].astype(np.int64)
    all_classes = set(np.unique(labels))
    class_valid = set(np.unique(labels[source_indices])) == all_classes and set(np.unique(labels[target_indices])) == all_classes
    if not class_valid:
        raise RuntimeError(f"Y1 corrected Phase-7 split lacks a class: {dataset} fold {fold}")

    x_source_raw = features.iloc[source_indices].copy().reset_index(drop=True)
    x_target_raw = features.iloc[target_indices].copy().reset_index(drop=True)
    y_source = labels[source_indices]
    y_target = labels[target_indices]
    preprocessor = build_preprocessor(feature_roles=roles, config=configs["preprocessing"], metadata=metadata)
    preprocessor.fit(x_source_raw)
    x_source = np.asarray(preprocessor.transform(x_source_raw), dtype=np.float64)
    x_target_clean = np.asarray(preprocessor.transform(x_target_raw), dtype=np.float64)

    split_digest = sha256(split_path)
    probe_seed = derive_reference_probe_seed(
        int(configs["probes"]["global_probe_seed"]), dataset, fold, split_digest
    )
    ref_positions = select_reference_probe_positions(
        len(x_source), int(configs["probes"]["reference"]["max_size"]), probe_seed
    )
    a_r = np.ascontiguousarray(x_source[ref_positions], dtype=np.float32)
    ref_hash = matrix_hash(a_r, {"dataset": dataset, "fold": fold, "split": split_digest, "type": "reference"})

    shift_engine = ShiftEngine(configs["shifts"], project_root=BASE)
    cache150, cache600 = SignalCache(), SignalCache()
    engine150 = PrefitSignalEngine(configs["signals"], configs["alignment"].get("alignment", {}), configs["methods"], cache150)
    engine600 = PrefitSignalEngine(configs["signals"], configs["alignment"].get("alignment", {}), configs["methods"], cache600)
    seeds = [int(x) for x in configs["falsification"]["algorithm_seeds"]]
    source_models: dict[tuple[str, int], FCM] = {}
    reference_controls: dict[tuple[str, int], dict[str, Any]] = {}
    clean_ari: dict[tuple[str, int], float] = {}
    for seed in seeds:
        model150, model600 = pair_models(x_source, K, seed)
        for arm, model, cache in [("duplicate_corrected_150_all", model150, cache150), ("fully_corrected_600", model600, cache600)]:
            source_models[(arm, seed)] = model
            cache.put_source_model(dataset, fold, "fcm_adaptive", seed, model)
            u_ref = model.predict_membership(a_r)
            reference_controls[(arm, seed)] = compute_validity_controls(a_r, model.cluster_centers_, u_ref)
            clean_ari[(arm, seed)] = float(adjusted_rand_score(y_target, np.argmax(model.predict_membership(x_target_clean), axis=1)))

    rows: list[dict[str, Any]] = []
    for condition in configs["falsification"]["conditions"]:
        shift = shift_engine.generate_shift(
            dataset_id=dataset,
            outer_fold=fold,
            condition=condition,
            X_target=x_target_raw,
            X_source=x_source_raw,
            roles=roles,
            y_target=y_target,
            y_source=y_source,
            backend="numpy",
        )
        if shift.status != "APPLICABLE":
            raise RuntimeError(f"corrected shift not applicable: {dataset} fold {fold} {condition}: {shift.reason}")
        shifted_raw = shift.X_shifted.reset_index(drop=True)
        row_map = np.asarray(shift.row_index_map, dtype=np.int64)
        y_shifted = y_target[row_map]
        x_shifted = np.asarray(preprocessor.transform(shifted_raw), dtype=np.float64)
        shift_hash = frame_hash(shifted_raw, row_map, {
            "dataset": dataset, "fold": fold, "condition": condition,
            "split": split_digest, "shift_config": canonical_hash(configs["shifts"]),
        })
        current_seed = derive_current_probe_seed(
            int(configs["probes"]["global_probe_seed"]), dataset, fold, condition, shift_hash
        )
        current_positions, _, _ = select_current_probe_positions(
            len(shifted_raw), row_map,
            int(configs["probes"]["current"]["max_size"]), current_seed, target_indices,
        )
        a_c = np.ascontiguousarray(x_shifted[current_positions], dtype=np.float32)
        current_hash = matrix_hash(a_c, {"dataset": dataset, "fold": fold, "condition": condition, "shift": shift_hash})

        for seed in seeds:
            candidate150, candidate600 = pair_models(x_shifted, K, seed)
            for arm, candidate, engine in [
                ("duplicate_corrected_150_all", candidate150, engine150),
                ("fully_corrected_600", candidate600, engine600),
            ]:
                source = source_models[(arm, seed)]
                engine.next_candidate = candidate
                signal_result = engine.compute_signals(
                    dataset_id=dataset,
                    outer_fold=fold,
                    condition=condition,
                    method="fcm_adaptive",
                    seed=seed,
                    K=K,
                    X_source_trans=x_source,
                    X_target_shifted_trans=x_shifted,
                    A_R=a_r,
                    A_C=a_c,
                    ref_bank_sha256=ref_hash,
                    cur_bank_sha256=current_hash,
                    shift_spec_sha256=shift_hash,
                    shift_spec_file_sha256="hypothesis_y1_in_memory_shift",
                    shift_replay_sha256="hypothesis_y1_in_memory_shift",
                )
                current = signal_result.to_dict()
                predicted = np.argmax(source.predict_membership(x_shifted), axis=1)
                ari_condition = float(adjusted_rand_score(y_shifted, predicted))
                delta_ari = clean_ari[(arm, seed)] - ari_condition
                reference = reference_controls[(arm, seed)]
                source_status = status(source, 150 if arm.startswith("duplicate") else 600)
                candidate_status = status(candidate, 150 if arm.startswith("duplicate") else 600)
                primary_eligible = bool(
                    current["usable"] and class_valid and source_status == "SUCCESS_CONVERGED"
                    and candidate_status == "SUCCESS_CONVERGED"
                )
                family, severity = ("clean", "none") if condition == "clean" else condition.rsplit("_", 1)
                current.update({
                    "replication_arm": arm,
                    "shift_family": family,
                    "severity": severity,
                    "delta_ari": float(delta_ari),
                    "ari_clean": clean_ari[(arm, seed)],
                    "ari_condition": ari_condition,
                    "source_policy_status": source_status,
                    "candidate_policy_status": candidate_status,
                    "source_final_shift": float(source.center_shift_history_[-1]),
                    "candidate_final_shift": float(candidate.center_shift_history_[-1]),
                    "source_n_iter": int(source.n_iter_),
                    "candidate_n_iter": int(candidate.n_iter_),
                    "class_valid": class_valid,
                    "duplicate_group_isolation": True,
                    "primary_eligible": primary_eligible,
                    "Delta_FPC": float(current["FPC"] - reference["FPC"]),
                    "Delta_PE": float(current["PE"] - reference["PE"]),
                    "Delta_XB_soft_m2": float(current["XB_soft_m2"] - reference["XB_soft_m2"]),
                    "Delta_silhouette": float(current["silhouette"] - reference["silhouette"]),
                    "reference_FPC": reference["FPC"],
                    "reference_PE": reference["PE"],
                    "reference_XB_soft_m2": reference["XB_soft_m2"],
                    "reference_silhouette": reference["silhouette"],
                    "corrected_split_sha256": split_digest,
                    "task_input_fingerprint": fingerprint,
                })
                rows.append(current)

    result = pd.DataFrame(rows).sort_values(["replication_arm", "dataset_id", "outer_fold", "condition", "seed"])
    atomic_csv(csv_path, result)
    checkpoint = {
        "dataset": dataset,
        "outer_fold": fold,
        "input_fingerprint": fingerprint,
        "rows": len(result),
        "csv_path": csv_path.relative_to(ROOT).as_posix(),
        "csv_sha256": sha256(csv_path),
        "completed": True,
    }
    atomic_json(json_path, checkpoint)
    return checkpoint


def load_independent():
    spec = importlib.util.spec_from_file_location("hypothesis_y1_independent", INDEPENDENT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_corrected(rows: pd.DataFrame, configs: dict[str, Any]) -> dict[str, Any]:
    mod = load_independent()
    cfg = configs["falsification"]
    blocks = {
        "P0": cfg["feature_blocks"]["P0"],
        "P3": cfg["feature_blocks"]["P3"],
        "P3_delta": ["D_X", "Delta_FPC", "Delta_PE", "Delta_XB_soft_m2", "Delta_silhouette"],
        "P4": cfg["feature_blocks"]["P4"],
    }
    metric_rows: list[pd.DataFrame] = []
    summary: dict[str, Any] = {"label": "POST-HOC METHODOLOGICAL CORRECTION / EXPLORATORY REPLICATION", "arms": {}}
    family_order: list[str] = []
    for condition in cfg["conditions"]:
        family = condition.rsplit("_", 1)[0]
        if condition != "clean" and family not in family_order:
            family_order.append(family)
    for arm in ["duplicate_corrected_150_all", "fully_corrected_600"]:
        data = rows[(rows["replication_arm"] == arm) & (rows["condition"] != "clean")].copy()
        if arm == "fully_corrected_600":
            data = data[data["primary_eligible"].astype(bool)].copy()
        arm_summary: dict[str, Any] = {"n_rows": len(data), "retention_fraction": len(data) / 4200.0}
        for evaluation in ["LODO", "LOSFO"]:
            modeled = mod.run_outer(data, cfg, evaluation, workers=6, blocks=blocks, regressors=("HGBR",))
            modeled["replication_arm"] = arm
            metric_rows.append(modeled)
            if evaluation == "LODO":
                order = [x for x in cfg["datasets"] if x in set(modeled["outer_group"])]
            else:
                order = [x for x in family_order if x in set(modeled["outer_group"])]
            wide = modeled.pivot(index="outer_group", columns="feature_block", values="mae").loc[order]
            comparisons: dict[str, Any] = {}
            for reference in ["P0", "P3", "P3_delta"]:
                delta = (wide[reference] - wide["P4"]).to_numpy()
                comparisons[f"{reference}_minus_P4"] = {
                    "wins": int((delta > 0).sum()),
                    "n_units": len(delta),
                    "bootstrap": mod.paired_bootstrap(delta, int(cfg["bootstrap"]["repetitions"]), int(cfg["bootstrap"]["seed"])),
                }
            arm_summary[evaluation] = {
                "metrics": modeled.groupby("feature_block").agg(
                    mae=("sum_absolute_error", lambda x: float(x.sum() / modeled.loc[x.index, "n_test"].sum())),
                    rmse=("sum_squared_error", lambda x: float(np.sqrt(x.sum() / modeled.loc[x.index, "n_test"].sum()))),
                    n=("n_test", "sum"),
                ).to_dict(orient="index"),
                "comparisons": comparisons,
            }
        # Grouped no-skill constants for corrected LODO/LOSFO.
        baseline: dict[str, Any] = {}
        for evaluation, column, order in [("LODO", "dataset_id", cfg["datasets"]), ("LOSFO", "shift_family", family_order)]:
            for kind in ["mean", "median"]:
                observed, predicted = [], []
                unit_mae = []
                for group in order:
                    test = data[data[column] == group]
                    train = data[data[column] != group]
                    if test.empty:
                        continue
                    constant = float(getattr(train["delta_ari"], kind)())
                    observed.extend(test["delta_ari"].tolist())
                    predicted.extend([constant] * len(test))
                    unit_mae.append(float(np.mean(np.abs(test["delta_ari"] - constant))))
                observed_a, predicted_a = np.asarray(observed), np.asarray(predicted)
                baseline[f"{evaluation}_B0_{kind}"] = {
                    "mae": float(np.mean(np.abs(observed_a - predicted_a))),
                    "rmse": float(np.sqrt(np.mean((observed_a - predicted_a) ** 2))),
                    "median_absolute_error": float(np.median(np.abs(observed_a - predicted_a))),
                    "r2": float(1.0 - np.sum((observed_a - predicted_a) ** 2) / np.sum((observed_a - observed_a.mean()) ** 2)),
                    "unit_mae": unit_mae,
                }
        arm_summary["no_skill"] = baseline
        summary["arms"][arm] = arm_summary
    all_metrics = pd.concat(metric_rows, ignore_index=True)
    atomic_csv(WORK / "hypothesis_y1_corrected_model_metrics.csv", all_metrics)
    atomic_json(WORK / "hypothesis_y1_corrected_replication_summary.json", summary)
    return summary


def main() -> None:
    started = time.perf_counter()
    configs = load_configs()
    datasets = list(configs["falsification"]["datasets"])
    folds = [int(x) for x in configs["falsification"]["outer_folds"]]
    tasks = [(dataset, fold) for dataset in datasets for fold in folds]
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_fold, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            completed.append(result)
            print(f"[Y1 corrected] {task[0]} fold {task[1]} complete ({len(completed)}/{len(tasks)})", flush=True)
    frames = [pd.read_csv(ROOT / item["csv_path"]) for item in sorted(completed, key=lambda x: (x["dataset"], x["outer_fold"]))]
    rows = pd.concat(frames, ignore_index=True).sort_values(
        ["replication_arm", "dataset_id", "outer_fold", "condition", "seed"]
    ).reset_index(drop=True)
    atomic_csv(WORK / "hypothesis_y1_corrected_rows.csv", rows)
    summary = evaluate_corrected(rows, configs)
    summary["runtime_seconds"] = time.perf_counter() - started
    summary["row_file_sha256"] = sha256(WORK / "hypothesis_y1_corrected_rows.csv")
    atomic_json(WORK / "hypothesis_y1_corrected_replication_summary.json", summary)
    print(json.dumps({"runtime_seconds": summary["runtime_seconds"], "arms": summary["arms"]}, indent=2))


if __name__ == "__main__":
    main()
