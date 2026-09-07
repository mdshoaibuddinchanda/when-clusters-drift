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
        """Apply a specific shift condition to raw target data.
        
        CRITICAL ARCHITECTURAL RULE:
        Operates strictly on raw target features before Phase-2 preprocessing.
        Target labels (if provided for offline supervised interventions) are never returned in ShiftResult.
        """
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

        # Parse family and severity
        family, severity = condition.rsplit("_", 1)
        if severity not in SEVERITIES or family not in SHIFT_FAMILIES:
            raise ValueError(f"Invalid condition parsing: family={family}, severity={severity}")

        source_stats = self.get_source_statistics(dataset_id, outer_fold, X_source, roles)

        # 2. Location Shift
        if family == "location":
            seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "location", "feature_selection")
            return apply_location_shift(X_target, source_stats, self.config, severity, seed, backend=backend)

        # 3. Scale Shift
        elif family == "scale":
            return apply_scale_shift(X_target, source_stats, self.config, severity, backend=backend)

        # 4. Measurement Noise
        elif family == "measurement_noise":
            seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "measurement_noise", "gaussian_tensor")
            return apply_measurement_noise(X_target, source_stats, self.config, severity, seed, backend=backend)

        # 5. Outliers
        elif family == "outliers":
            row_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "outliers", "row_selection")
            tensor_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "outliers", "student_t_noise")
            return apply_outlier_shift(X_target, source_stats, self.config, severity, row_seed, tensor_seed, backend=backend)

        # 6. MCAR Missingness
        elif family == "mcar":
            cell_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, "mcar", "cell_permutation")
            clustering_cols = [c for c in X_target.columns if roles.get(c) in ["numeric", "categorical", "ordinal", "boolean"]]
            return apply_mcar_shift(X_target, clustering_cols, self.config, severity, cell_seed)

        # 7. Class Prevalence
        elif family == "class_prevalence":
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

        # 8. Local Structural Overlap
        elif family == "local_overlap":
            if y_target is None or y_source is None:
                return ShiftResult(
                    X_shifted=X_target.copy(),
                    row_index_map=np.arange(len(X_target), dtype=np.int64),
                    metadata={"reason": "Labels required for offline local overlap intervention"},
                    status="NOT_APPLICABLE",
                    reason="Labels required for offline local overlap intervention",
                )
            return apply_local_overlap_shift(X_target, y_target, X_source, y_source, source_stats, self.config, severity, backend=backend)

        raise RuntimeError(f"Unhandled family: {family}")

    def get_spec_path(self, dataset_id: str, outer_fold: int, condition: str) -> Path:
        """Get path for a shift specification JSON file."""
        specs_dir = self.project_root / "data" / "shifts" / "specs" / dataset_id / f"fold_{outer_fold}"
        return specs_dir / f"{condition}.json"

    def save_spec_atomic(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        shift_res: ShiftResult,
        canonical_bundle_hash: str,
        split_hash: str,
    ) -> Path:
        """Save shift specification atomically with companion NPZ array storage if needed."""
        target_json = self.get_spec_path(dataset_id, outer_fold, condition)
        target_json.parent.mkdir(parents=True, exist_ok=True)
        target_npz = target_json.with_suffix(".npz")

        scenario_input_hash = compute_scenario_input_sha256(
            canonical_bundle_hash=canonical_bundle_hash,
            split_hash=split_hash,
            dataset_id=dataset_id,
            outer_fold=outer_fold,
        )

        family = "clean" if condition == "clean" else condition.rsplit("_", 1)[0]
        severity = "none" if condition == "clean" else condition.rsplit("_", 1)[1]
        derived_seed = derive_integer_seed(self.global_shift_seed, dataset_id, outer_fold, family, "spec")

        # Save large array data (e.g. row_index_map) in compressed NPZ
        np.savez_compressed(target_npz, row_index_map=shift_res.row_index_map)
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
            "scenario_input_sha256": scenario_input_hash,
            "shift_protocol_sha256": self.protocol_sha256,
            "shift_spec_sha256": spec_sha256,
            "derived_seed": derived_seed,
            "npz_relpath": str(target_npz.relative_to(self.project_root)).replace("\\", "/"),
            "npz_sha256": npz_hash,
            "metadata": shift_res.metadata,
        }

        # Atomic write: write temp file then rename
        tmp_path = target_json.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(spec_doc, f, indent=2, sort_keys=True)
        os.replace(tmp_path, target_json)

        return target_json
