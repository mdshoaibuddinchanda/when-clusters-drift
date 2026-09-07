"""Frozen Distribution-Shift Replay Layer for Phase 6.

CRITICAL ARCHITECTURAL BOUNDARY:
- replay_frozen_shift(): strictly label-free; accepts NO labels and loads NO labels.
- build_local_overlap_replay_descriptor(): offline experiment preparation ONLY;
  allowed to use labels once to materialize feature-only replay descriptors.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from clusterdrift.shifts.base import ShiftResult
from clusterdrift.shifts.engine import ShiftEngine
from clusterdrift.shifts.hashing import (
    atomic_write_json,
    atomic_write_npz,
    compute_bytes_sha256,
    compute_canonical_json_sha256,
    compute_file_sha256,
)


def build_local_overlap_replay_descriptor(
    X_target_raw: pd.DataFrame,
    y_target_raw: np.ndarray,
    X_source_raw: pd.DataFrame,
    y_source_raw: np.ndarray,
    roles: Dict[str, str],
    dataset_id: str,
    outer_fold: int,
    condition: str,
    phase4_spec_path: Union[str, Path],
    project_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    commit_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a derived, sparse, feature-only local-overlap replay descriptor.

    This function is strictly for OFFLINE EXPERIMENT CONSTRUCTION.
    It runs Phase 4's intervention using labels once, extracts the affected
    feature coordinates and replacement values, verifies exact reproduction,
    persists compressed array coordinates in a companion NPZ, and returns
    a cryptographically bound metadata replay descriptor.

    NO raw labels, target y, class IDs, or raw inline array data are persisted in the descriptor JSON.
    """
    root = project_root or Path(phase4_spec_path).parents[4]
    spec_p = Path(phase4_spec_path)
    if not spec_p.exists():
        raise FileNotFoundError(f"Phase-4 spec file not found: {spec_p}")

    with open(spec_p, "r", encoding="utf-8") as f:
        spec_doc = json.load(f)

    # 1. Load shift configuration and run Phase-4 generator
    shifts_cfg_path = root / "configs" / "shifts.yaml"
    with open(shifts_cfg_path, "r", encoding="utf-8") as f:
        shift_cfg = yaml.safe_load(f)

    engine_shift = ShiftEngine(shift_cfg, project_root=root)
    shift_res = engine_shift.generate_shift(
        dataset_id=dataset_id,
        outer_fold=outer_fold,
        condition=condition,
        X_target=X_target_raw,
        X_source=X_source_raw,
        roles=roles,
        y_target=y_target_raw,
        y_source=y_source_raw,
        backend="numpy",
    )

    # 2. Extract affected numeric columns and selected rows
    eligible_cols = sorted([c for c in roles if roles[c] == "numeric"])
    
    # Identify which rows differ between original and shifted features
    orig_sub = X_target_raw[eligible_cols].to_numpy(dtype=np.float64)
    shifted_sub = shift_res.X_shifted[eligible_cols].to_numpy(dtype=np.float64)
    diff = ~np.isclose(shifted_sub, orig_sub, equal_nan=True).all(axis=1)
    selected_rows = np.where(diff)[0].astype(np.int64)

    # Verify against frozen Phase-4 metadata
    selected_rows_bytes = selected_rows.tobytes()
    selected_rows_sha = compute_bytes_sha256(selected_rows_bytes)
    exp_selected_sha = spec_doc.get("metadata", {}).get("selected_rows_sha256")
    if exp_selected_sha and selected_rows_sha != exp_selected_sha:
        raise ValueError(
            f"Selected rows SHA mismatch for {dataset_id} {outer_fold} {condition}: "
            f"{selected_rows_sha} vs expected {exp_selected_sha}"
        )

    # Extract replacement values for affected columns on selected rows
    replacement_vals = np.ascontiguousarray(shifted_sub[selected_rows, :], dtype=np.float64)
    replacement_vals_bytes = replacement_vals.tobytes()
    replacement_vals_sha = compute_bytes_sha256(replacement_vals_bytes)

    # 3. Test replay equivalence immediately
    X_replayed = X_target_raw.copy()
    for idx, col in enumerate(eligible_cols):
        col_vals = X_replayed[col].to_numpy(dtype=np.float64, copy=True)
        col_vals[selected_rows] = replacement_vals[:, idx]
        X_replayed[col] = col_vals

    pd.testing.assert_frame_equal(shift_res.X_shifted, X_replayed, check_exact=True)

    # 4. Save companion NPZ with compressed arrays
    out_base = output_root or ((project_root or root) / "data" / "signals" / "offline_shift_replay")
    out_dir = Path(out_base) / dataset_id / f"fold_{outer_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{condition}.npz"
    atomic_write_npz(
        npz_path,
        selected_target_positions=selected_rows,
        replacement_values=replacement_vals,
    )
    replay_npz_sha = compute_file_sha256(npz_path)

    # 5. Construct metadata-only descriptor (no inline array coordinates or values)
    phase4_spec_file_sha = compute_file_sha256(spec_p)
    descriptor: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "outer_fold": outer_fold,
        "condition": condition,
        "generated_from_commit": commit_sha or "unknown",
        "phase4_shift_spec_sha256": spec_doc["shift_spec_sha256"],
        "phase4_shift_spec_file_sha256": phase4_spec_file_sha,
        "phase4_npz_sha256": spec_doc["npz_sha256"],
        "phase4_protocol_sha256": spec_doc.get("shift_protocol_sha256", ""),
        "canonical_bundle_sha256": spec_doc.get("canonical_bundle_sha256", ""),
        "split_sha256": spec_doc.get("split_sha256", ""),
        "affected_numeric_columns": eligible_cols,
        "selected_target_positions_sha256": selected_rows_sha,
        "replacement_values_sha256": replacement_vals_sha,
        "replay_npz_sha256": replay_npz_sha,
    }
    descriptor["replay_descriptor_sha256"] = compute_canonical_json_sha256(descriptor)
    return descriptor


def replay_frozen_shift(
    X_target_raw: pd.DataFrame,
    dataset_id: str,
    outer_fold: int,
    condition: str,
    phase4_spec_path: Union[str, Path],
    replay_descriptor: Optional[Union[Dict[str, Any], Path]] = None,
    X_source_raw: Optional[pd.DataFrame] = None,
    roles: Optional[Dict[str, str]] = None,
    project_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> ShiftResult:
    """Deterministically replay a frozen Phase-4 shift without access to labels.

    Parameters
    ----------
    X_target_raw : pd.DataFrame
        Raw held-out target features.
    dataset_id : str
        Dataset identifier.
    outer_fold : int
        Outer fold index.
    condition : str
        Shift condition name.
    phase4_spec_path : Union[str, Path]
        Path to Phase-4 scenario JSON specification.
    replay_descriptor : Optional[Union[Dict[str, Any], Path]]
        Descriptor or path to descriptor for local overlap shifts.
        If None and condition is local overlap, loads from standard path.
    X_source_raw : Optional[pd.DataFrame]
        Raw outer source features (for ordinary shifts).
    roles : Optional[Dict[str, str]]
        Feature roles mapping.
    project_root : Optional[Path]
        Repository root.
    output_root : Optional[Path]
        Root directory for offline shift replay descriptors/NPZs.

    Returns
    -------
    ShiftResult
        The shifted target features and row index map.
    """
    spec_p = Path(phase4_spec_path)
    if not spec_p.exists():
        raise FileNotFoundError(f"Phase-4 spec file not found: {spec_p}")

    root = project_root or spec_p.parents[4]

    with open(spec_p, "r", encoding="utf-8") as f:
        spec_doc = json.load(f)

    # Validate basic scenario coordinates
    if spec_doc.get("dataset_id") != dataset_id:
        raise ValueError(f"dataset_id mismatch: {spec_doc.get('dataset_id')} vs {dataset_id}")
    if spec_doc.get("outer_fold") != outer_fold:
        raise ValueError(f"outer_fold mismatch: {spec_doc.get('outer_fold')} vs {outer_fold}")
    if spec_doc.get("condition") != condition:
        raise ValueError(f"condition mismatch: {spec_doc.get('condition')} vs {condition}")

    n_rows = len(X_target_raw)

    # 1. Clean Condition: exact identity
    if condition == "clean":
        return ShiftResult(
            X_shifted=X_target_raw.copy(),
            row_index_map=np.arange(n_rows, dtype=np.int64),
            metadata={"family": "clean", "severity": "none"},
            status="APPLICABLE",
        )

    # 2. Class Prevalence: replay using frozen row_index_map from NPZ (zero labels)
    if condition.startswith("class_prevalence"):
        npz_path = spec_p.with_suffix(".npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Companion NPZ not found for class prevalence: {npz_path}")

        # Check NPZ file hash
        npz_file_sha = compute_file_sha256(npz_path)
        if npz_file_sha != spec_doc.get("npz_sha256"):
            raise ValueError(
                f"NPZ hash mismatch for {dataset_id} {outer_fold} {condition}: "
                f"{npz_file_sha} vs {spec_doc.get('npz_sha256')}"
            )

        with np.load(npz_path) as npz:
            if "row_index_map" not in npz:
                raise KeyError(f"NPZ missing 'row_index_map': {npz_path}")
            row_map = npz["row_index_map"]

        # Check row map bytes hash
        row_map_bytes = row_map.tobytes()
        row_map_sha = compute_bytes_sha256(row_map_bytes)
        exp_row_sha = spec_doc.get("metadata", {}).get("resampling_index_sha256")
        if exp_row_sha and row_map_sha != exp_row_sha:
            raise ValueError(
                f"Resampling index hash mismatch: {row_map_sha} vs {exp_row_sha}"
            )

        if len(row_map) != n_rows:
            raise ValueError(f"Row map length {len(row_map)} does not match target rows {n_rows}")

        X_shifted = X_target_raw.iloc[row_map].reset_index(drop=True)
        return ShiftResult(
            X_shifted=X_shifted,
            row_index_map=row_map,
            metadata={
                "family": "class_prevalence",
                "severity": condition.rsplit("_", 1)[1],
                "row_index_map_sha256": row_map_sha,
            },
            status="APPLICABLE",
        )

    # 3. Local Overlap: replay using sparse feature-only replay descriptor and companion NPZ
    if condition.startswith("local_overlap"):
        replay_base = Path(output_root) if output_root is not None else (root / "data" / "signals" / "offline_shift_replay")
        if replay_descriptor is None:
            desc_p = (
                replay_base
                / dataset_id
                / f"fold_{outer_fold}"
                / f"{condition}.json"
            )
            if not desc_p.exists():
                raise FileNotFoundError(
                    f"Replay descriptor not found: {desc_p}. Run with --prepare-offline-replay first."
                )
            with open(desc_p, "r", encoding="utf-8") as f:
                desc = json.load(f)
            companion_npz_p = desc_p.with_suffix(".npz")
        elif isinstance(replay_descriptor, (str, Path)):
            desc_p = Path(replay_descriptor)
            if not desc_p.exists():
                raise FileNotFoundError(f"Replay descriptor not found: {desc_p}")
            with open(desc_p, "r", encoding="utf-8") as f:
                desc = json.load(f)
            companion_npz_p = desc_p.with_suffix(".npz")
        else:
            desc = replay_descriptor
            companion_npz_p = (
                replay_base
                / dataset_id
                / f"fold_{outer_fold}"
                / f"{condition}.npz"
            )

        # Validate replay descriptor integrity and binding to Phase 4
        if desc.get("dataset_id") != dataset_id or desc.get("outer_fold") != outer_fold or desc.get("condition") != condition:
            raise ValueError(f"Replay descriptor coordinates mismatch: {desc.get('dataset_id')} vs {dataset_id}")

        if desc.get("phase4_shift_spec_sha256") != spec_doc["shift_spec_sha256"]:
            raise ValueError("Replay descriptor bound to stale/different Phase-4 shift_spec_sha256")

        spec_file_sha = compute_file_sha256(spec_p)
        if desc.get("phase4_shift_spec_file_sha256") != spec_file_sha:
            raise ValueError("Replay descriptor bound to modified Phase-4 spec file")

        # Verify descriptor self-hash
        stored_desc_sha = desc.get("replay_descriptor_sha256")
        desc_copy = dict(desc)
        del desc_copy["replay_descriptor_sha256"]
        recomputed_desc_sha = compute_canonical_json_sha256(desc_copy)
        if stored_desc_sha != recomputed_desc_sha:
            raise ValueError("Replay descriptor corrupted: hash does not recompute")

        # Validate companion NPZ file existence and hash
        if not companion_npz_p.exists():
            raise FileNotFoundError(
                f"Companion NPZ not found for local overlap: {companion_npz_p}"
            )
        npz_file_sha = compute_file_sha256(companion_npz_p)
        if desc.get("replay_npz_sha256") and npz_file_sha != desc["replay_npz_sha256"]:
            raise ValueError(
                f"Companion NPZ SHA mismatch for {dataset_id} {outer_fold} {condition}: "
                f"{npz_file_sha} vs {desc['replay_npz_sha256']}"
            )

        with np.load(companion_npz_p) as npz:
            if "selected_target_positions" not in npz or "replacement_values" not in npz:
                raise KeyError(f"Companion NPZ missing required arrays: {companion_npz_p}")
            selected_rows = npz["selected_target_positions"].astype(np.int64)
            replacement_vals = npz["replacement_values"].astype(np.float64)

        selected_rows_sha = compute_bytes_sha256(selected_rows.tobytes())
        if desc.get("selected_target_positions_sha256") and selected_rows_sha != desc["selected_target_positions_sha256"]:
            raise ValueError(
                f"selected_target_positions hash mismatch: {selected_rows_sha} vs {desc['selected_target_positions_sha256']}"
            )

        replacement_vals_sha = compute_bytes_sha256(np.ascontiguousarray(replacement_vals, dtype=np.float64).tobytes())
        if desc.get("replacement_values_sha256") and replacement_vals_sha != desc["replacement_values_sha256"]:
            raise ValueError(
                f"replacement_values hash mismatch: {replacement_vals_sha} vs {desc['replacement_values_sha256']}"
            )

        affected_cols = desc["affected_numeric_columns"]
        if replacement_vals.shape[1] != len(affected_cols):
            raise ValueError(
                f"Replacement values column count {replacement_vals.shape[1]} does not match affected columns {len(affected_cols)}"
            )

        # Apply replacement values preserving columns and row counts
        X_shifted = X_target_raw.copy()
        for idx, col in enumerate(affected_cols):
            col_vals = X_shifted[col].to_numpy(dtype=np.float64, copy=True)
            col_vals[selected_rows] = replacement_vals[:, idx]
            X_shifted[col] = col_vals

        if len(X_shifted) != n_rows:
            raise ValueError(f"Replayed row count {len(X_shifted)} does not match {n_rows}")

        return ShiftResult(
            X_shifted=X_shifted,
            row_index_map=np.arange(n_rows, dtype=np.int64),
            metadata={
                "family": "local_overlap",
                "severity": condition.rsplit("_", 1)[1],
                "replay_descriptor_sha256": stored_desc_sha,
                "selected_rows_count": len(selected_rows),
            },
            status="APPLICABLE",
        )

    # 4. Ordinary Shifts: location, scale, mcar, outliers, measurement_noise
    # Replay using frozen Phase-4 config, seeds, and source statistics without labels
    shifts_cfg_path = root / "configs" / "shifts.yaml"
    with open(shifts_cfg_path, "r", encoding="utf-8") as f:
        shift_cfg = yaml.safe_load(f)

    if X_source_raw is None:
        from clusterdrift.probes.matrix import load_raw_source_features
        X_source_raw, meta = load_raw_source_features(dataset_id, outer_fold, root)
        if roles is None:
            roles = meta.get("feature_roles", {c: "numeric" for c in X_source_raw.columns})
    elif roles is None:
        roles = {c: "numeric" for c in X_source_raw.columns}

    engine_shift = ShiftEngine(shift_cfg, project_root=root)
    shift_res = engine_shift.generate_shift(
        dataset_id=dataset_id,
        outer_fold=outer_fold,
        condition=condition,
        X_target=X_target_raw,
        X_source=X_source_raw,
        roles=roles,
        y_target=None,
        y_source=None,
        backend="numpy",
    )
    return shift_res
