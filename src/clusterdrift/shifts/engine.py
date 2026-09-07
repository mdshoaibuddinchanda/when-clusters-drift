"""Central distribution shift engine for generating, applying, caching, and validating controlled interventions."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from clusterdrift.shifts.base import (
    ALL_CONDITIONS,
    SEVERITIES,
    SHIFT_FAMILIES,
    ShiftResult,
    SourceStatistics,
    compute_source_statistics,
)
from clusterdrift.shifts.backend import detect_hardware
from clusterdrift.shifts.hashing import (
    atomic_write_json,
    atomic_write_npz,
    compute_canonical_json_sha256,
    compute_file_sha256,
    compute_scenario_input_sha256,
    compute_shift_protocol_sha256,
    compute_shift_spec_sha256,
    derive_integer_seed,
)
from clusterdrift.shifts.missingness import apply_mcar_shift
from clusterdrift.shifts.numeric import (
    apply_location_shift,
    apply_measurement_noise,
    apply_outlier_shift,
    apply_scale_shift,
)
from clusterdrift.shifts.offline_supervised import (
    apply_class_prevalence_shift,
    apply_local_overlap_shift,
)


def validate_saved_shift_spec(
    spec_path: Union[str, Path],
    expected_canonical_bundle_sha256: str,
    expected_split_sha256: str,
    expected_shift_protocol_sha256: str,
    global_shift_seed: int,
    dataset_id: str,
    outer_fold: int,
    condition: str,
    project_root: Optional[Path] = None,
    manifest_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Validate on-disk shift spec and companion NPZ against current inputs.
    
    Used identically by both --resume and --verify.
    Returns (is_valid, reason_str, parsed_spec_or_none).
    """
    p = Path(spec_path)
    if not p.exists():
        return False, f"Spec JSON not found: {p}", None

    npz_path = p.with_suffix(".npz")
    if not npz_path.exists():
        return False, f"Companion NPZ not found: {npz_path}", None

    try:
        with open(p, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        return False, f"Invalid JSON in spec file: {e}", None

    # 1. Verify scenario coordinates
    if spec.get("dataset_id") != dataset_id:
        return False, f"dataset_id mismatch: {spec.get('dataset_id')} vs {dataset_id}", spec
    if spec.get("outer_fold") != outer_fold:
        return False, f"outer_fold mismatch: {spec.get('outer_fold')} vs {outer_fold}", spec
    if spec.get("condition") != condition:
        return False, f"condition mismatch: {spec.get('condition')} vs {condition}", spec

    # 2. Verify canonical bundle hash
    if spec.get("canonical_bundle_sha256") != expected_canonical_bundle_sha256:
        return False, "canonical_bundle_sha256 mismatch (stale bundle)", spec

    # 3. Verify Phase-2 split hash
    if spec.get("split_sha256") != expected_split_sha256:
        return False, "split_sha256 mismatch (stale split)", spec

    # 4. Verify protocol hash
    if spec.get("shift_protocol_sha256") != expected_shift_protocol_sha256:
        return False, "shift_protocol_sha256 mismatch (stale config)", spec

    # 5. Recompute scenario_input_sha256
    expected_input_hash = compute_scenario_input_sha256(
        canonical_bundle_sha256=expected_canonical_bundle_sha256,
        split_sha256=expected_split_sha256,
        dataset_id=dataset_id,
        outer_fold=outer_fold,
    )
    if spec.get("scenario_input_sha256") != expected_input_hash:
        return False, "scenario_input_sha256 mismatch", spec

    # 6. Recompute derived seed
    family = "clean" if condition == "clean" else condition.rsplit("_", 1)[0]
    expected_seed = derive_integer_seed(global_shift_seed, dataset_id, outer_fold, family, "spec")
    if spec.get("derived_seed") != expected_seed:
        return False, f"derived_seed mismatch: {spec.get('derived_seed')} vs {expected_seed}", spec

    # 7. Check NPZ actual bytes hash vs saved npz_sha256
    actual_npz_hash = compute_file_sha256(npz_path)
    if spec.get("npz_sha256") != actual_npz_hash:
        return False, "NPZ byte hash mismatch (corrupted or altered NPZ)", spec

    try:
        with np.load(npz_path) as npz:
            if "row_index_map" not in npz:
                return False, "NPZ missing 'row_index_map' array", spec
    except Exception as e:
        return False, f"Failed to load NPZ: {e}", spec

    # 8. Recompute shift_spec_sha256
    severity = "none" if condition == "clean" else condition.rsplit("_", 1)[1]
    spec_descriptor = {
        "metadata": spec.get("metadata", {}),
        "status": spec.get("status", "APPLICABLE"),
        "reason": spec.get("reason"),
        "npz_sha256": actual_npz_hash,
    }
    expected_spec_hash = compute_shift_spec_sha256(
        scenario_input_sha256=expected_input_hash,
        shift_protocol_sha256=expected_shift_protocol_sha256,
        condition=condition,
        family=family,
        severity=severity,
        derived_seed=expected_seed,
        spec_descriptor=spec_descriptor,
    )
    if spec.get("shift_spec_sha256") != expected_spec_hash:
        return False, "shift_spec_sha256 mismatch (spec descriptor altered)", spec

    # 9. Verify manifest entry agreement if provided
    if manifest_entry is not None:
        if manifest_entry.get("spec_hash") != expected_spec_hash:
            return False, "Manifest spec_hash mismatch", spec
        if manifest_entry.get("status") != spec.get("status"):
            return False, "Manifest status mismatch", spec
        if manifest_entry.get("shift_spec_file_sha256") != compute_file_sha256(p):
            return False, "Manifest spec file SHA mismatch", spec

    return True, "VALID", spec


class ShiftEngine:
    """Deterministic, order-independent controlled distribution shift engine."""

    def __init__(
        self,
        config: Union[Dict[str, Any], Path, str],
        project_root: Optional[Union[Path, str]] = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            cfg_path = Path(config)
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            self.config_path = cfg_path
        else:
            self.config = config
            self.config_path = None

        if project_root is not None:
            self.project_root = Path(project_root)
        elif self.config_path is not None:
            self.project_root = self.config_path.resolve().parent.parent
        else:
            self.project_root = Path.cwd()

        self.protocol_version = self.config.get("protocol_version", 1)
        self.global_shift_seed = self.config.get("global_shift_seed", 2026090704)
        self.protocol_sha256 = compute_shift_protocol_sha256(self.config)

        # In-memory source statistics cache: (dataset_id, outer_fold) -> SourceStatistics
        self._source_stats_cache: Dict[Tuple[str, int], SourceStatistics] = {}

    def get_source_statistics(
        self,
        dataset_id: str,
        outer_fold: int,
        X_source: pd.DataFrame,
        roles: Dict[str, str],
    ) -> SourceStatistics:
        """Retrieve or compute float64 outer-source statistics."""
        key = (dataset_id, outer_fold)
        if key not in self._source_stats_cache:
            self._source_stats_cache[key] = compute_source_statistics(X_source, roles)
        return self._source_stats_cache[key]

    def generate_shift(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        X_target: pd.DataFrame,
        X_source: pd.DataFrame,
        roles: Dict[str, str],
        y_target: Optional[np.ndarray] = None,
        y_source: Optional[np.ndarray] = None,
        backend: str = "auto",
    ) -> ShiftResult:
        """Apply a specific shift condition to raw target data."""
        if condition not in ALL_CONDITIONS:
            raise ValueError(f"Unknown condition '{condition}'. Expected one of: {ALL_CONDITIONS}")

        # 1. Clean Condition: exact identity
        if condition == "clean":
            n_rows = len(X_target)
            return ShiftResult(
                X_shifted=X_target.copy(),
                row_index_map=np.arange(n_rows, dtype=np.int64),
                metadata={"family": "clean", "severity": "none"},
                status="APPLICABLE",
            )

        family, severity = condition.rsplit("_", 1)
        if severity not in SEVERITIES or family not in SHIFT_FAMILIES:
            raise ValueError(f"Invalid condition parsing: family={family}, severity={severity}")

        source_stats = self.get_source_statistics(dataset_id, outer_fold, X_source, roles)

        if family == "location":
            feature_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "location", "feature_selection")
            return apply_location_shift(X_target, source_stats, self.config, severity, feature_seed, backend=backend)

        if family == "scale":
            return apply_scale_shift(X_target, source_stats, self.config, severity, backend=backend)

        if family == "measurement_noise":
            noise_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "measurement_noise", "noise_tensor")
            return apply_measurement_noise(X_target, source_stats, self.config, severity, noise_seed, backend=backend)

        if family == "outliers":
            row_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "outliers", "row_selection")
            tensor_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "outliers", "outlier_tensor")
            return apply_outlier_shift(X_target, source_stats, self.config, severity, row_seed, tensor_seed, backend=backend)

        if family == "mcar":
            cell_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "mcar", "cell_permutation")
            clustering_cols = [c for c in X_target.columns if roles.get(c) in ["numeric", "categorical", "ordinal", "boolean"]]
            return apply_mcar_shift(X_target, clustering_cols, self.config, severity, cell_seed)

        if family == "class_prevalence":
            if y_target is None or y_source is None:
                return ShiftResult(
                    X_shifted=X_target.copy(),
                    row_index_map=np.arange(len(X_target), dtype=np.int64),
                    metadata={"reason": "Labels required for offline class prevalence intervention"},
                    status="NOT_APPLICABLE",
                    reason="Labels required for offline class prevalence intervention",
                )
            resample_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "class_prevalence", "resample")
            return apply_class_prevalence_shift(X_target, y_target, y_source, self.config, severity, resample_seed)

        if family == "local_overlap":
            if y_target is None or y_source is None:
                return ShiftResult(
                    X_shifted=X_target.copy(),
                    row_index_map=np.arange(len(X_target), dtype=np.int64),
                    metadata={"reason": "Labels required for offline local overlap intervention"},
                    status="NOT_APPLICABLE",
                    reason="Labels required for offline local overlap intervention",
                )
            return apply_local_overlap_shift(
                X_target=X_target,
                y_target=y_target,
                X_source=X_source,
                y_source=y_source,
                source_stats=source_stats,
                cfg=self.config,
                severity=severity,
                backend=backend,
            )

        raise NotImplementedError(f"Unhandled shift family '{family}'")

    def get_spec_path(self, dataset_id: str, outer_fold: int, condition: str) -> Path:
        """Standardized relative path for a scenario shift specification JSON."""
        return self.project_root / "data" / "shifts" / "specs" / dataset_id / f"fold_{outer_fold}" / f"{condition}.json"

    def save_spec_atomic(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        shift_res: ShiftResult,
        canonical_bundle_sha256: str,
        split_sha256: str,
    ) -> Path:
        """Save shift specification atomically with companion NPZ array storage."""
        target_json = self.get_spec_path(dataset_id, outer_fold, condition)
        target_npz = target_json.with_suffix(".npz")

        scenario_input_hash = compute_scenario_input_sha256(
            canonical_bundle_sha256=canonical_bundle_sha256,
            split_sha256=split_sha256,
            dataset_id=dataset_id,
            outer_fold=outer_fold,
        )

        family = "clean" if condition == "clean" else condition.rsplit("_", 1)[0]
        severity = "none" if condition == "clean" else condition.rsplit("_", 1)[1]
        derived_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, family, "spec")

        # Atomic write of compressed companion NPZ
        atomic_write_npz(target_npz, row_index_map=shift_res.row_index_map)
        npz_hash = compute_file_sha256(target_npz)

        spec_descriptor = {
            "metadata": shift_res.metadata,
            "status": shift_res.status,
            "reason": shift_res.reason,
            "npz_sha256": npz_hash,
        }

        spec_sha256 = compute_shift_spec_sha256(
            scenario_input_sha256=scenario_input_hash,
            shift_protocol_sha256=self.protocol_sha256,
            condition=condition,
            family=family,
            severity=severity,
            derived_seed=derived_seed,
            spec_descriptor=spec_descriptor,
        )

        spec_doc = {
            "protocol_version": self.protocol_version,
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": condition,
            "family": family,
            "severity": severity,
            "status": shift_res.status,
            "reason": shift_res.reason,
            "canonical_bundle_sha256": canonical_bundle_sha256,
            "split_sha256": split_sha256,
            "scenario_input_sha256": scenario_input_hash,
            "shift_protocol_sha256": self.protocol_sha256,
            "shift_spec_sha256": spec_sha256,
            "derived_seed": derived_seed,
            "npz_relpath": str(target_npz.relative_to(self.project_root)).replace("\\", "/"),
            "npz_sha256": npz_hash,
            "metadata": shift_res.metadata,
        }

        # Atomic write of spec JSON
        atomic_write_json(target_json, spec_doc, indent=2, sort_keys=True)
        return target_json
