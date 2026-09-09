"""Deterministic split generation module for Phase 2 data partitioning and inner validation."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from clusterdrift.shifts.hashing import atomic_write_json, atomic_write_npz


@dataclass
class InnerFold:
    inner_fold: int
    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass
class FoldSplit:
    outer_fold: int
    source_indices: np.ndarray
    target_indices: np.ndarray
    inner_folds: List[InnerFold]
    split_strategy: str
    metadata: Dict[str, Any]


def compute_split_protocol_sha256(config: Dict[str, Any]) -> str:
    """Compute deterministic cryptographic SHA256 of the Phase-2 splitting protocol via canonical JSON."""
    split_cfg = config.get("splitting", {})
    payload_dict = {
        "protocol_version": config.get("protocol_version", 1),
        "outer_folds": int(split_cfg.get("outer_folds", 5)),
        "inner_folds": int(split_cfg.get("inner_folds", 3)),
        "split_seed": int(split_cfg.get("split_seed", 20260907)),
        "temporal": split_cfg.get("temporal", {
            "order_columns": ["date", "period"],
            "policy": "expanding_window_tie_safe",
        }),
        "natural_resampling": split_cfg.get("natural_resampling", {
            "policy": "source_resampling_fixed_target",
        }),
    }
    canonical_json = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_preprocessing_config_sha256(config: Dict[str, Any]) -> str:
    """Compute deterministic cryptographic SHA256 of preprocessing transformation settings via canonical JSON."""
    prep_cfg = config.get("preprocessing", config)
    payload_dict = {
        "output_dtype": str(prep_cfg.get("output_dtype", "float32")),
        "numeric": prep_cfg.get("numeric", {}),
        "categorical": prep_cfg.get("categorical", {}),
        "ordinal": prep_cfg.get("ordinal", {}),
        "boolean": prep_cfg.get("boolean", {}),
    }
    canonical_json = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_scenario_input_sha256(source_bundle_sha: str, target_bundle_sha: str) -> str:
    """Compute a valid 64-character SHA256 digest binding source and target canonical bundles."""
    payload = f"source:{source_bundle_sha}\ntarget:{target_bundle_sha}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_split_hash(
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    inner_folds: List[InnerFold],
    split_strategy: str,
    split_seed: int,
    canonical_bundle_sha256: str,
    split_protocol_sha256: Optional[str] = None,
) -> str:
    """Compute deterministic cryptographic SHA256 digest over partition structure, indices, and protocol."""
    source_digest = hashlib.sha256(np.ascontiguousarray(source_indices).tobytes()).hexdigest()
    target_digest = hashlib.sha256(np.ascontiguousarray(target_indices).tobytes()).hexdigest()

    components = [
        f"strategy:{split_strategy}",
        f"seed:{split_seed}",
        f"bundle:{canonical_bundle_sha256}",
        f"protocol:{split_protocol_sha256 or 'NONE'}",
        f"source:{source_digest}",
        f"target:{target_digest}",
    ]
    for inf in sorted(inner_folds, key=lambda x: x.inner_fold):
        tr_dig = hashlib.sha256(np.ascontiguousarray(inf.train_indices).tobytes()).hexdigest()
        val_dig = hashlib.sha256(np.ascontiguousarray(inf.val_indices).tobytes()).hexdigest()
        components.append(f"inner_{inf.inner_fold}_train:{tr_dig}")
        components.append(f"inner_{inf.inner_fold}_val:{val_dig}")

    payload = "\n".join(components)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_kfold_splits(
    n_samples: int,
    n_outer: int = 5,
    n_inner: int = 3,
    split_seed: int = 20260907,
) -> List[FoldSplit]:
    """Generate 5-fold outer cross-validation splits and 3-fold inner validation folds."""
    indices = np.arange(n_samples)
    kf = KFold(n_splits=n_outer, shuffle=True, random_state=split_seed)
    splits: List[FoldSplit] = []

    for r, (source_idx, target_idx) in enumerate(kf.split(indices)):
        # Inner splits within outer source only
        inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=split_seed + r * 10)
        inner_folds: List[InnerFold] = []
        for q, (inner_train_sub, inner_val_sub) in enumerate(inner_kf.split(source_idx)):
            in_train = source_idx[inner_train_sub]
            in_val = source_idx[inner_val_sub]
            inner_folds.append(InnerFold(inner_fold=q, train_indices=in_train, val_indices=in_val))

        splits.append(
            FoldSplit(
                outer_fold=r,
                source_indices=source_idx,
                target_indices=target_idx,
                inner_folds=inner_folds,
                split_strategy="kfold",
                metadata={"split_seed": split_seed},
            )
        )
    return splits


def generate_group_kfold_splits(
    groups: np.ndarray,
    n_outer: int = 5,
    n_inner: int = 3,
    group_column_name: str = "group_id",
) -> List[FoldSplit]:
    """Generate 5-fold group-isolated outer splits and 3-fold group-isolated inner folds."""
    n_samples = len(groups)
    indices = np.arange(n_samples)
    gkf = GroupKFold(n_splits=n_outer)
    splits: List[FoldSplit] = []

    for r, (source_idx, target_idx) in enumerate(gkf.split(indices, groups=groups)):
        # Enforce strict group disjointness
        src_groups = set(groups[source_idx])
        tgt_groups = set(groups[target_idx])
        if not src_groups.isdisjoint(tgt_groups):
            raise ValueError(f"Group leakage detected in fold {r}: {src_groups.intersection(tgt_groups)}")

        # Inner group splits within outer source
        inner_gkf = GroupKFold(n_splits=n_inner)
        inner_groups = groups[source_idx]
        inner_folds: List[InnerFold] = []
        for q, (inner_train_sub, inner_val_sub) in enumerate(inner_gkf.split(source_idx, groups=inner_groups)):
            in_train = source_idx[inner_train_sub]
            in_val = source_idx[inner_val_sub]

            # Verify inner group disjointness
            in_tr_groups = set(inner_groups[inner_train_sub])
            in_val_groups = set(inner_groups[inner_val_sub])
            if not in_tr_groups.isdisjoint(in_val_groups):
                raise ValueError(f"Inner group leakage in fold {r}, inner {q}")

            inner_folds.append(InnerFold(inner_fold=q, train_indices=in_train, val_indices=in_val))

        splits.append(
            FoldSplit(
                outer_fold=r,
                source_indices=source_idx,
                target_indices=target_idx,
                inner_folds=inner_folds,
                split_strategy="group_kfold",
                metadata={
                    "group_column": group_column_name,
                    "n_source_groups": len(src_groups),
                    "n_target_groups": len(tgt_groups),
                },
            )
        )
    return splits


def generate_temporal_block_splits(
    features_df: pd.DataFrame,
    time_columns: Optional[List[str]] = None,
    n_outer: int = 5,
    n_inner: int = 3,
) -> List[FoldSplit]:
    """Generate chronological expanding-window splits with strict tuple ordering and tie protection."""
    if time_columns is None:
        time_columns = ["date", "period"] if "period" in features_df.columns else ["date"]

    time_cols = [c for c in time_columns if c in features_df.columns]
    if not time_cols:
        raise ValueError(f"None of the specified time columns {time_columns} exist in features DataFrame.")

    # Sort indices stably by lexicographic tuple order
    sorted_order = features_df.sort_values(by=time_cols, kind="stable").index.values
    df_keys = features_df[time_cols]

    tscv = TimeSeriesSplit(n_splits=n_outer)
    splits: List[FoldSplit] = []

    for r, (train_sub, test_sub) in enumerate(tscv.split(sorted_order)):
        # Tie-breaking boundary adjustment:
        # If identical time keys straddle train and test boundary, adjust so all identical keys remain on one side
        cutoff = len(train_sub)
        while cutoff > 0 and cutoff < len(sorted_order):
            src_last_key = tuple(df_keys.loc[sorted_order[cutoff - 1]])
            tgt_first_key = tuple(df_keys.loc[sorted_order[cutoff]])
            if src_last_key == tgt_first_key:
                cutoff -= 1
            else:
                break

        source_idx = sorted_order[:cutoff]
        target_idx = sorted_order[cutoff:len(train_sub) + len(test_sub)]

        # Verify strict chronological inequality
        src_keys = [tuple(df_keys.loc[i]) for i in source_idx]
        tgt_keys = [tuple(df_keys.loc[i]) for i in target_idx]
        max_source_key = max(src_keys)
        min_target_key = min(tgt_keys)
        if not (max_source_key < min_target_key):
            raise ValueError(
                f"Temporal leakage in fold {r}: max source key {max_source_key} is not strictly less than min target key {min_target_key}"
            )

        # Inner chronological splits within outer source
        inner_tscv = TimeSeriesSplit(n_splits=n_inner)
        inner_folds: List[InnerFold] = []
        for q, (inner_tr_sub, inner_val_sub) in enumerate(inner_tscv.split(source_idx)):
            in_cutoff = len(inner_tr_sub)
            while in_cutoff > 0 and in_cutoff < len(source_idx):
                in_src_key = tuple(df_keys.loc[source_idx[in_cutoff - 1]])
                in_tgt_key = tuple(df_keys.loc[source_idx[in_cutoff]])
                if in_src_key == in_tgt_key:
                    in_cutoff -= 1
                else:
                    break
            in_train = source_idx[:in_cutoff]
            in_val = source_idx[in_cutoff:len(inner_tr_sub) + len(inner_val_sub)]

            in_src_keys = [tuple(df_keys.loc[i]) for i in in_train]
            in_val_keys = [tuple(df_keys.loc[i]) for i in in_val]
            if not (max(in_src_keys) < min(in_val_keys)):
                raise ValueError(f"Inner temporal leakage in fold {r}, inner {q}")

            inner_folds.append(InnerFold(inner_fold=q, train_indices=in_train, val_indices=in_val))

        splits.append(
            FoldSplit(
                outer_fold=r,
                source_indices=source_idx,
                target_indices=target_idx,
                inner_folds=inner_folds,
                split_strategy="temporal_block",
                metadata={
                    "time_columns": time_cols,
                    "max_source_key": list(max_source_key),
                    "min_target_key": list(min_target_key),
                    "notes": "Chronological expanding-window split with strict tuple ordering.",
                },
            )
        )
    return splits


def generate_whyshift_natural_splits(
    n_source: int,
    n_target: int,
    source_domain: str = "CA",
    target_domain: str = "TX",
    n_outer: int = 5,
    n_inner: int = 3,
    split_seed: int = 20260907,
) -> List[FoldSplit]:
    """Generate 5 deterministic source-resampling folds of source domain while preserving complete target domain."""
    source_indices = np.arange(n_source)
    target_indices = np.arange(n_target)
    kf = KFold(n_splits=n_outer, shuffle=True, random_state=split_seed)
    splits: List[FoldSplit] = []

    for r, (source_train_idx, source_excluded_idx) in enumerate(kf.split(source_indices)):
        # Inner folds within source_train
        inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=split_seed + r * 10)
        inner_folds: List[InnerFold] = []
        for q, (inner_train_sub, inner_val_sub) in enumerate(inner_kf.split(source_train_idx)):
            in_train = source_train_idx[inner_train_sub]
            in_val = source_train_idx[inner_val_sub]
            inner_folds.append(InnerFold(inner_fold=q, train_indices=in_train, val_indices=in_val))

        splits.append(
            FoldSplit(
                outer_fold=r,
                source_indices=source_train_idx,
                target_indices=target_indices,
                inner_folds=inner_folds,
                split_strategy="natural_domain",
                metadata={
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "split_seed": split_seed,
                    "source_excluded_rows": len(source_excluded_idx),
                },
            )
        )
    return splits


def generate_tableshift_natural_splits(
    domain_series: np.ndarray,
    candidate_ood_values: List[int],
    n_outer: int = 5,
    n_inner: int = 3,
    split_seed: int = 20260907,
) -> Dict[str, List[FoldSplit]]:
    """Generate leave-one-domain-out natural scenarios with 5 source-resampling folds each."""
    scenarios: Dict[str, List[FoldSplit]] = {}

    for ood_val in candidate_ood_values:
        scenario_id = f"ood_{ood_val}"
        target_mask = (domain_series == ood_val)
        target_indices = np.where(target_mask)[0]

        source_domain_mask = ~target_mask
        source_domain_indices = np.where(source_domain_mask)[0]

        kf = KFold(n_splits=n_outer, shuffle=True, random_state=split_seed)
        splits: List[FoldSplit] = []

        for r, (source_train_sub, source_excluded_sub) in enumerate(kf.split(source_domain_indices)):
            source_train_idx = source_domain_indices[source_train_sub]
            source_excluded_idx = source_domain_indices[source_excluded_sub]

            # Inner folds within source_train_idx
            inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=split_seed + r * 10)
            inner_folds: List[InnerFold] = []
            for q, (inner_train_sub, inner_val_sub) in enumerate(inner_kf.split(source_train_idx)):
                in_train = source_train_idx[inner_train_sub]
                in_val = source_train_idx[inner_val_sub]
                inner_folds.append(InnerFold(inner_fold=q, train_indices=in_train, val_indices=in_val))

            splits.append(
                FoldSplit(
                    outer_fold=r,
                    source_indices=source_train_idx,
                    target_indices=target_indices,
                    inner_folds=inner_folds,
                    split_strategy="natural_domain",
                    metadata={
                        "ood_domain_value": int(ood_val),
                        "target_rows": len(target_indices),
                        "source_excluded_rows": len(source_excluded_idx),
                        "split_seed": split_seed,
                    },
                )
            )

        scenarios[scenario_id] = splits
    return scenarios


def save_fold_split_artifacts(
    split: FoldSplit,
    target_dir: Path,
    file_prefix: str,
    common_metadata: Dict[str, Any],
    canonical_bundle_sha256: str,
    split_config_sha256: str,
    split_protocol_sha256: Optional[str] = None,
    scenario_input_sha256: Optional[str] = None,
) -> Tuple[Path, Path, str]:
    """Persist split indices as compressed NPZ and metadata as JSON."""
    target_dir.mkdir(parents=True, exist_ok=True)
    npz_path = target_dir / f"{file_prefix}.npz"
    json_path = target_dir / f"{file_prefix}.json"

    # Compute deterministic split hash
    split_seed = common_metadata.get("split_seed", split.metadata.get("split_seed", 20260907))
    binding_bundle_sha = scenario_input_sha256 or canonical_bundle_sha256
    split_sha256 = compute_split_hash(
        source_indices=split.source_indices,
        target_indices=split.target_indices,
        inner_folds=split.inner_folds,
        split_strategy=split.split_strategy,
        split_seed=split_seed,
        canonical_bundle_sha256=binding_bundle_sha,
        split_protocol_sha256=split_protocol_sha256,
    )

    # Save NPZ arrays
    npz_dict = {
        "source_indices": np.ascontiguousarray(split.source_indices, dtype=np.int64),
        "target_indices": np.ascontiguousarray(split.target_indices, dtype=np.int64),
    }
    for inf in split.inner_folds:
        npz_dict[f"inner_{inf.inner_fold}_train"] = np.ascontiguousarray(inf.train_indices, dtype=np.int64)
        npz_dict[f"inner_{inf.inner_fold}_val"] = np.ascontiguousarray(inf.val_indices, dtype=np.int64)

    atomic_write_npz(npz_path, **npz_dict)

    # Save JSON metadata
    meta_dict = {
        **common_metadata,
        "split_strategy": split.split_strategy,
        "outer_fold": split.outer_fold,
        "source_rows": len(split.source_indices),
        "target_rows": len(split.target_indices),
        "inner_folds": [
            {
                "inner_fold": inf.inner_fold,
                "train_rows": len(inf.train_indices),
                "val_rows": len(inf.val_indices),
            }
            for inf in split.inner_folds
        ],
        "split_seed": split_seed,
        "canonical_bundle_sha256": canonical_bundle_sha256,
        "scenario_input_sha256": scenario_input_sha256 or canonical_bundle_sha256,
        "split_protocol_sha256": split_protocol_sha256,
        "split_config_sha256": split_config_sha256,
        "split_sha256": split_sha256,
        "extra": split.metadata,
    }

    atomic_write_json(json_path, meta_dict, indent=2, sort_keys=False)

    return npz_path, json_path, split_sha256


def verify_split_artifact(
    npz_path: Path,
    json_path: Path,
    expected_bundle_sha256: Optional[str] = None,
    expected_protocol_sha256: Optional[str] = None,
    expected_split_seed: Optional[int] = None,
    expected_strategy: Optional[str] = None,
) -> bool:
    """Verify cryptographic integrity, provenance, and protocol matching of split artifacts."""
    if not npz_path.exists() or not json_path.exists():
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if expected_bundle_sha256 is not None:
            if (
                meta.get("canonical_bundle_sha256") != expected_bundle_sha256
                and meta.get("scenario_input_sha256") != expected_bundle_sha256
            ):
                return False

        if expected_protocol_sha256 is not None:
            if meta.get("split_protocol_sha256") != expected_protocol_sha256:
                return False

        if expected_split_seed is not None:
            if int(meta.get("split_seed", -1)) != int(expected_split_seed):
                return False

        if expected_strategy is not None:
            if meta.get("split_strategy") != expected_strategy:
                return False

        with np.load(npz_path) as npz:
            source_indices = npz["source_indices"]
            target_indices = npz["target_indices"]

            inner_folds = []
            inner_count = len(meta.get("inner_folds", []))
            for i in range(inner_count):
                tr_key = f"inner_{i}_train"
                val_key = f"inner_{i}_val"
                if tr_key not in npz or val_key not in npz:
                    return False
                inner_folds.append(InnerFold(inner_fold=i, train_indices=npz[tr_key], val_indices=npz[val_key]))

        recomputed_hash = compute_split_hash(
            source_indices=source_indices,
            target_indices=target_indices,
            inner_folds=inner_folds,
            split_strategy=meta["split_strategy"],
            split_seed=meta["split_seed"],
            canonical_bundle_sha256=meta.get("scenario_input_sha256") or meta.get("canonical_bundle_sha256", ""),
            split_protocol_sha256=meta.get("split_protocol_sha256"),
        )

        return recomputed_hash == meta.get("split_sha256")
    except Exception:
        return False
