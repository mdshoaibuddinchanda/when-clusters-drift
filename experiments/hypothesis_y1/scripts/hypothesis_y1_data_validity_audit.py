"""Convergence, duplicate, class-representation, split, and dependence audit.

All outputs are isolated under experiments/hypothesis_y1. Original Phase-2
splits and frozen Phase-7 results are read-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments" / "hypothesis_y1"
ARTIFACTS = BASE / "artifacts"
CORRECTED = BASE / "corrected_splits"
SPLIT_SEED = 20260907
PHASE7_DATASETS = {
    "iris", "glass", "sonar", "breast_cancer_wisconsin_diagnostic",
    "spambase", "waveform", "satimage", "pendigits", "letter_recognition",
    "madelon", "human_activity_recognition", "mice_protein_expression",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        np.savez_compressed(tmp, **arrays)
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
        with tmp.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def canonical_row_hashes(frame: pd.DataFrame) -> np.ndarray:
    """Return stable dtype-aware hashes for exact row identity.

    Pandas' canonical scalar hashing treats equal values and equal missing values
    identically. Collision risk is checked below against exact duplicate counts.
    """
    return pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(np.uint64)


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def combined_groups(row_hashes: np.ndarray, protected_groups: np.ndarray | None) -> np.ndarray:
    uf = UnionFind(len(row_hashes))
    for values in [row_hashes, protected_groups]:
        if values is None:
            continue
        first: dict[Any, int] = {}
        for idx, value in enumerate(values):
            key = value.item() if isinstance(value, np.generic) else value
            if key in first:
                uf.union(first[key], idx)
            else:
                first[key] = idx
    roots = np.array([uf.find(i) for i in range(len(row_hashes))], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def safe_group_splits(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    dummy = np.zeros((len(y), 1), dtype=np.float64)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(dummy, y, groups))
        strategy = "StratifiedGroupKFold"
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(dummy, y, groups))
        strategy = "GroupKFold_fallback"
    return splits, strategy


def verify_group_isolation(groups: np.ndarray, left: np.ndarray, right: np.ndarray) -> bool:
    return set(groups[left]).isdisjoint(set(groups[right]))


def duplicate_and_class_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    duplicate_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {"policy_version": 1, "datasets": {}}
    canonical_root = ROOT / "data/canonical/controlled"

    for ds_dir in sorted(p for p in canonical_root.iterdir() if p.is_dir()):
        dataset = ds_dir.name
        features = pd.read_parquet(ds_dir / "features.parquet")
        labels = pd.read_parquet(ds_dir / "labels.parquet").iloc[:, 0].to_numpy()
        metadata = json.loads((ds_dir / "metadata.json").read_text(encoding="utf-8"))
        row_hash = canonical_row_hashes(features)
        value_counts = pd.Series(row_hash).value_counts()
        duplicated_hashes = value_counts[value_counts > 1]
        exact_unique = int(len(features.drop_duplicates()))
        hash_unique = int(len(np.unique(row_hash)))
        if exact_unique != hash_unique:
            raise RuntimeError(f"row-hash collision or canonical mismatch in {dataset}")

        protected_groups = None
        if metadata.get("group_column") and (ds_dir / "groups.parquet").exists():
            protected_groups = pd.read_parquet(ds_dir / "groups.parquet").iloc[:, 0].to_numpy()
        groups = combined_groups(row_hash, protected_groups)
        original_strategy = str(metadata.get("split_strategy", "kfold"))

        outer_splits: list[tuple[np.ndarray, np.ndarray]] = []
        outer_strategy = ""
        if original_strategy == "temporal_block":
            for fold in range(5):
                with np.load(
                    ROOT / f"data/splits/controlled/{dataset}/fold_{fold}.npz",
                    allow_pickle=False,
                ) as saved:
                    outer_splits.append((saved["source_indices"], saved["target_indices"]))
            outer_strategy = "preserved_temporal_block"
        else:
            outer_splits, outer_strategy = safe_group_splits(labels, groups, 5, SPLIT_SEED)

        ds_out = CORRECTED / dataset
        split_summary["datasets"][dataset] = {
            "original_strategy": original_strategy,
            "corrected_outer_strategy": outer_strategy,
            "protected_group_column": metadata.get("group_column"),
            "n_connected_groups": int(len(np.unique(groups))),
            "folds": [],
        }

        all_classes = set(map(str, np.unique(labels)))
        for fold, (source_idx, target_idx) in enumerate(outer_splits):
            source_idx = np.asarray(source_idx, dtype=np.int64)
            target_idx = np.asarray(target_idx, dtype=np.int64)
            original_path = ROOT / f"data/splits/controlled/{dataset}/fold_{fold}.npz"
            with np.load(original_path, allow_pickle=False) as original:
                orig_source = original["source_indices"]
                orig_target = original["target_indices"]
            target_leak = np.isin(row_hash[orig_target], np.unique(row_hash[orig_source]))
            duplicate_rows.append({
                "dataset": dataset,
                "N": len(features),
                "unique_rows": exact_unique,
                "duplicate_rows_excess": int(len(features) - exact_unique),
                "rows_in_duplicate_groups": int(duplicated_hashes.sum()),
                "duplicate_groups": int(len(duplicated_hashes)),
                "largest_duplicate_group": int(duplicated_hashes.max()) if len(duplicated_hashes) else 1,
                "outer_fold": fold,
                "original_target_rows": int(len(orig_target)),
                "original_target_rows_with_source_duplicate": int(target_leak.sum()),
                "original_target_leakage_fraction": float(target_leak.mean()),
                "phase7_dataset": dataset in PHASE7_DATASETS,
            })

            source_classes = set(map(str, np.unique(labels[orig_source])))
            target_classes = set(map(str, np.unique(labels[orig_target])))
            class_rows.append({
                "dataset": dataset,
                "outer_fold": fold,
                "K_manifest": int(metadata["n_classes"]),
                "source_classes_represented": len(source_classes),
                "target_classes_represented": len(target_classes),
                "missing_source_classes": json.dumps(sorted(all_classes - source_classes)),
                "missing_target_classes": json.dumps(sorted(all_classes - target_classes)),
                "source_class_complete": source_classes == all_classes,
                "target_class_complete": target_classes == all_classes,
                "split_version": "original_phase2",
                "phase7_dataset": dataset in PHASE7_DATASETS,
            })

            if not verify_group_isolation(groups, source_idx, target_idx):
                raise RuntimeError(f"corrected outer duplicate/group leakage: {dataset} fold {fold}")
            inner_splits, inner_strategy = safe_group_splits(
                labels[source_idx], groups[source_idx], 3, SPLIT_SEED + fold + 1
            )
            arrays: dict[str, np.ndarray] = {
                "source_indices": source_idx,
                "target_indices": target_idx,
            }
            inner_verified = True
            for inner_fold, (inner_train_pos, inner_val_pos) in enumerate(inner_splits):
                inner_train = source_idx[inner_train_pos]
                inner_val = source_idx[inner_val_pos]
                arrays[f"inner_{inner_fold}_train"] = inner_train
                arrays[f"inner_{inner_fold}_val"] = inner_val
                inner_verified &= verify_group_isolation(groups, inner_train, inner_val)
            if not inner_verified:
                raise RuntimeError(f"corrected inner duplicate/group leakage: {dataset} fold {fold}")

            corrected_source_classes = set(map(str, np.unique(labels[source_idx])))
            corrected_target_classes = set(map(str, np.unique(labels[target_idx])))
            class_rows.append({
                "dataset": dataset,
                "outer_fold": fold,
                "K_manifest": int(metadata["n_classes"]),
                "source_classes_represented": len(corrected_source_classes),
                "target_classes_represented": len(corrected_target_classes),
                "missing_source_classes": json.dumps(sorted(all_classes - corrected_source_classes)),
                "missing_target_classes": json.dumps(sorted(all_classes - corrected_target_classes)),
                "source_class_complete": corrected_source_classes == all_classes,
                "target_class_complete": corrected_target_classes == all_classes,
                "split_version": "hypothesis_y1_corrected",
                "phase7_dataset": dataset in PHASE7_DATASETS,
            })

            npz_path = ds_out / f"hypothesis_y1_fold_{fold}.npz"
            atomic_npz(npz_path, arrays)
            record = {
                "dataset": dataset,
                "outer_fold": fold,
                "policy": "connected protected-group and exact-row-hash components",
                "outer_strategy": outer_strategy,
                "inner_strategy": inner_strategy,
                "source_rows": int(len(source_idx)),
                "target_rows": int(len(target_idx)),
                "outer_group_leakage": False,
                "inner_group_leakage": False,
                "source_classes": sorted(corrected_source_classes),
                "target_classes": sorted(corrected_target_classes),
                "npz_path": npz_path.relative_to(ROOT).as_posix(),
                "npz_sha256": sha256(npz_path),
            }
            atomic_json(ds_out / f"hypothesis_y1_fold_{fold}.json", record)
            split_summary["datasets"][dataset]["folds"].append(record)

    return pd.DataFrame(duplicate_rows), pd.DataFrame(class_rows), split_summary


def convergence_audit(joined: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groupings = {
        "overall": [],
        "dataset": ["dataset_id"],
        "fold": ["outer_fold"],
        "condition": ["condition"],
        "severity": ["severity"],
        "seed": ["seed"],
        "dataset_fold": ["dataset_id", "outer_fold"],
        "dataset_condition": ["dataset_id", "condition"],
    }
    for level, columns in groupings.items():
        groups: Iterable[tuple[Any, pd.DataFrame]]
        if columns:
            groups = joined.groupby(columns, dropna=False, sort=True)
        else:
            groups = [("ALL", joined)]
        for key, part in groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            record = {"grouping_level": level, "n_rows": len(part)}
            for col, value in zip(columns, key_tuple):
                record[col] = value
            src = part["source_converged"].astype(bool)
            cand = part["candidate_converged"].astype(bool)
            record.update(
                source_converged_count=int(src.sum()),
                source_convergence_rate=float(src.mean()),
                candidate_converged_count=int(cand.sum()),
                candidate_convergence_rate=float(cand.mean()),
                both_converged_count=int((src & cand).sum()),
                both_convergence_rate=float((src & cand).mean()),
                usable_count=int(part["usable"].astype(bool).sum()),
                usable_nonconverged_count=int((part["usable"].astype(bool) & ~(src & cand)).sum()),
            )
            rows.append(record)
    overall = next(row for row in rows if row["grouping_level"] == "overall")
    accepted_bad = joined["usable"].astype(bool) & (
        ~joined["source_converged"].astype(bool) | ~joined["candidate_converged"].astype(bool)
    )
    summary = {
        "overall": overall,
        "usable_accepts_nonconvergence": bool(accepted_bad.any()),
        "usable_nonconverged_rows": int(accepted_bad.sum()),
        "letter_recognition": next(
            row for row in rows
            if row["grouping_level"] == "dataset" and row.get("dataset_id") == "letter_recognition"
        ),
    }
    return pd.DataFrame(rows), summary


def dependence_audit(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_cols = ["dataset_id", "outer_fold", "condition"]
    scenario = primary.groupby(scenario_cols, sort=True)["delta_ari"].agg(
        n_seeds="size", mean="mean", variance=lambda x: float(np.var(x, ddof=0)),
        minimum="min", maximum="max", unique_values="nunique",
    ).reset_index()
    between = float(np.var(scenario["mean"], ddof=0))
    within = float(np.mean(scenario["variance"]))
    icc = between / (between + within) if between + within > 0 else 0.0
    summary = {
        "n_rows": len(primary),
        "n_scenarios": len(scenario),
        "identical_across_seed_count": int((scenario["unique_values"] == 1).sum()),
        "identical_across_seed_fraction": float((scenario["unique_values"] == 1).mean()),
        "mean_within_scenario_seed_variance": within,
        "between_scenario_mean_variance": between,
        "icc_style_ratio": icc,
        "scientific_inference_unit": "dataset and shift-family, not rows or seeds",
    }
    return scenario, summary


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joined = pd.read_csv(ROOT / "results/falsification/joined_evaluation_table.csv")
    joined["severity"] = joined["condition"].str.rsplit("_", n=1).str[-1]
    joined.loc[joined["condition"].eq("clean"), "severity"] = "none"
    convergence, convergence_summary = convergence_audit(joined)
    atomic_csv(ARTIFACTS / "hypothesis_y1_convergence_diagnostics.csv", convergence)
    atomic_json(ARTIFACTS / "hypothesis_y1_convergence_summary.json", convergence_summary)

    duplicates, classes, split_summary = duplicate_and_class_audit()
    atomic_csv(ARTIFACTS / "hypothesis_y1_duplicate_audit.csv", duplicates)
    atomic_csv(ARTIFACTS / "hypothesis_y1_class_k_audit.csv", classes)
    atomic_json(ARTIFACTS / "hypothesis_y1_corrected_split_summary.json", split_summary)

    primary = joined.loc[joined["condition"].ne("clean")].copy()
    seed_scenarios, dependence = dependence_audit(primary)
    atomic_csv(ARTIFACTS / "hypothesis_y1_seed_dependence.csv", seed_scenarios)
    atomic_json(ARTIFACTS / "hypothesis_y1_dependence_summary.json", dependence)
    print(json.dumps({
        "convergence": convergence_summary,
        "phase7_duplicate_leakage_datasets": sorted(
            duplicates.loc[
                duplicates["phase7_dataset"] & duplicates["original_target_rows_with_source_duplicate"].gt(0),
                "dataset",
            ].unique().tolist()
        ),
        "phase7_original_class_problem_folds": int(
            classes.loc[
                classes["phase7_dataset"] & classes["split_version"].eq("original_phase2"),
                ["source_class_complete", "target_class_complete"],
            ].all(axis=1).eq(False).sum()
        ),
        "dependence": dependence,
    }, indent=2))


if __name__ == "__main__":
    main()
