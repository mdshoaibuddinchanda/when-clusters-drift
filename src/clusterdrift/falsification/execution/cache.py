"""Persistent, Content-Addressed, Ephemeral Cache for Phase 7 Execution."""

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

from clusterdrift.falsification.execution.environment import get_library_versions
from clusterdrift.falsification.execution.resources import assert_not_frozen_data_path
from clusterdrift.shifts.hashing import (
    atomic_write_json,
    compute_canonical_json_sha256,
    compute_file_sha256,
)


def compute_content_fingerprint(data: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint for cache key dictionary."""
    return compute_canonical_json_sha256(data)


class PersistentPhase7Cache:
    """Robust content-addressed cache for derived runtime matrices, models, and MMD components."""

    def __init__(
        self,
        cache_dir: Path,
        protocol_sha: str,
        project_root: Optional[Path] = None,
    ):
        self.project_root = Path(project_root).resolve() if project_root else None
        self.cache_dir = assert_not_frozen_data_path(cache_dir, self.project_root)
        self.protocol_sha = protocol_sha

        self.root = self.cache_dir / "phase7" / protocol_sha
        self.datasets_dir = self.root / "datasets"
        self.scenarios_dir = self.root / "scenarios"
        self.models_dir = self.root / "models"
        self.memberships_dir = self.root / "memberships"
        self.mmd_dir = self.root / "mmd"

        for d in [self.datasets_dir, self.scenarios_dir, self.models_dir, self.memberships_dir, self.mmd_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.marker_file = self.root / ".clusterdrift_phase7_cache"
        if not self.marker_file.exists():
            marker_data = {
                "marker": "clusterdrift_phase7_cache",
                "protocol_sha": protocol_sha,
                "library_versions": get_library_versions(),
            }
            atomic_write_json(self.marker_file, marker_data, indent=2)

    # -----------------------------------------------------------------------
    # Source Preprocessed Arrays & Reference Probes
    # -----------------------------------------------------------------------

    def get_source_data(
        self,
        dataset_id: str,
        outer_fold: int,
        expected_fingerprint: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached X_source_trans and A_R if fingerprint matches exactly."""
        meta_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}.json"
        x_src_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}_X_src.npy"
        a_r_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}_A_R.npy"

        if not (meta_p.exists() and x_src_p.exists() and a_r_p.exists()):
            return None

        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != expected_fingerprint:
                return None
            X_src = np.load(x_src_p, mmap_mode="r")
            A_R = np.load(a_r_p, mmap_mode="r")
            return X_src, A_R
        except Exception:
            return None

    def put_source_data(
        self,
        dataset_id: str,
        outer_fold: int,
        fingerprint: str,
        X_source_trans: np.ndarray,
        A_R: np.ndarray,
    ) -> None:
        """Persist X_source_trans and A_R arrays with metadata."""
        x_src_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}_X_src.npy"
        a_r_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}_A_R.npy"
        meta_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}.json"

        # Atomic array writes
        tmp_x = x_src_p.with_suffix(".tmp.npy")
        np.save(tmp_x, np.asarray(X_source_trans, dtype=np.float64))
        os.replace(tmp_x, x_src_p)

        tmp_a = a_r_p.with_suffix(".tmp.npy")
        np.save(tmp_a, np.asarray(A_R, dtype=np.float64))
        os.replace(tmp_a, a_r_p)

        meta = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "fingerprint": fingerprint,
            "x_src_shape": list(X_source_trans.shape),
            "a_r_shape": list(A_R.shape),
            "dtype": "float64",
        }
        atomic_write_json(meta_p, meta, indent=2)

    # -----------------------------------------------------------------------
    # Scenario Shifted Arrays & Current Probes
    # -----------------------------------------------------------------------

    def get_scenario_data(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        expected_fingerprint: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached X_target_shifted_trans and A_C."""
        base_name = f"{dataset_id}_fold_{outer_fold}_{condition}"
        meta_p = self.scenarios_dir / f"{base_name}.json"
        x_tgt_p = self.scenarios_dir / f"{base_name}_X_tgt.npy"
        a_c_p = self.scenarios_dir / f"{base_name}_A_C.npy"

        if not (meta_p.exists() and x_tgt_p.exists() and a_c_p.exists()):
            return None

        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != expected_fingerprint:
                return None
            X_tgt = np.load(x_tgt_p, mmap_mode="r")
            A_C = np.load(a_c_p, mmap_mode="r")
            return X_tgt, A_C
        except Exception:
            return None

    def put_scenario_data(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        fingerprint: str,
        X_target_shifted_trans: np.ndarray,
        A_C: np.ndarray,
    ) -> None:
        """Persist shifted target matrix and current probe matrix."""
        base_name = f"{dataset_id}_fold_{outer_fold}_{condition}"
        x_tgt_p = self.scenarios_dir / f"{base_name}_X_tgt.npy"
        a_c_p = self.scenarios_dir / f"{base_name}_A_C.npy"
        meta_p = self.scenarios_dir / f"{base_name}.json"

        tmp_x = x_tgt_p.with_suffix(".tmp.npy")
        np.save(tmp_x, np.asarray(X_target_shifted_trans, dtype=np.float64))
        os.replace(tmp_x, x_tgt_p)

        tmp_a = a_c_p.with_suffix(".tmp.npy")
        np.save(tmp_a, np.asarray(A_C, dtype=np.float64))
        os.replace(tmp_a, a_c_p)

        meta = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": condition,
            "fingerprint": fingerprint,
            "x_tgt_shape": list(X_target_shifted_trans.shape),
            "a_c_shape": list(A_C.shape),
            "dtype": "float64",
        }
        atomic_write_json(meta_p, meta, indent=2)

    # -----------------------------------------------------------------------
    # Source Models
    # -----------------------------------------------------------------------

    def get_source_model(
        self,
        dataset_id: str,
        outer_fold: int,
        method: str,
        seed: int,
        expected_fingerprint: str,
    ) -> Optional[Any]:
        """Retrieve serialized source clustering model after verifying fingerprint & integrity."""
        base_name = f"{dataset_id}_fold_{outer_fold}_{method}_seed_{seed}"
        meta_p = self.models_dir / f"{base_name}.json"
        model_p = self.models_dir / f"{base_name}.joblib"

        if not (meta_p.exists() and model_p.exists()):
            return None

        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != expected_fingerprint:
                return None

            model = joblib.load(model_p)

            # Verification of model integrity
            centers = getattr(model, "cluster_centers_", None)
            if centers is None or not np.isfinite(centers).all():
                return None
            if getattr(model, "n_clusters", None) != meta.get("K"):
                return None

            return model
        except Exception:
            return None

    def put_source_model(
        self,
        dataset_id: str,
        outer_fold: int,
        method: str,
        seed: int,
        K: int,
        fingerprint: str,
        model: Any,
    ) -> None:
        """Persist fitted source clustering model."""
        base_name = f"{dataset_id}_fold_{outer_fold}_{method}_seed_{seed}"
        model_p = self.models_dir / f"{base_name}.joblib"
        meta_p = self.models_dir / f"{base_name}.json"

        tmp_model = model_p.with_suffix(".tmp.joblib")
        joblib.dump(model, tmp_model)
        os.replace(tmp_model, model_p)

        fp_str = getattr(model, "fingerprint_", "") or getattr(model, "model_fingerprint", "")
        meta = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "method": method,
            "seed": seed,
            "K": K,
            "fingerprint": fingerprint,
            "model_fingerprint": str(fp_str),
        }
        atomic_write_json(meta_p, meta, indent=2)

    # -----------------------------------------------------------------------
    # Source Reference Memberships & Scales
    # -----------------------------------------------------------------------

    def get_source_membership_scales(
        self,
        dataset_id: str,
        outer_fold: int,
        method: str,
        seed: int,
        expected_fingerprint: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        base_name = f"{dataset_id}_fold_{outer_fold}_{method}_seed_{seed}"
        meta_p = self.memberships_dir / f"{base_name}.json"
        u_p = self.memberships_dir / f"{base_name}_U.npy"
        s_p = self.memberships_dir / f"{base_name}_scales.npy"

        if not (meta_p.exists() and u_p.exists() and s_p.exists()):
            return None

        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != expected_fingerprint:
                return None
            U_0_R = np.load(u_p, mmap_mode="r")
            scales = np.load(s_p)
            return U_0_R, scales
        except Exception:
            return None

    def put_source_membership_scales(
        self,
        dataset_id: str,
        outer_fold: int,
        method: str,
        seed: int,
        fingerprint: str,
        U_0_R: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        base_name = f"{dataset_id}_fold_{outer_fold}_{method}_seed_{seed}"
        u_p = self.memberships_dir / f"{base_name}_U.npy"
        s_p = self.memberships_dir / f"{base_name}_scales.npy"
        meta_p = self.memberships_dir / f"{base_name}.json"

        tmp_u = u_p.with_suffix(".tmp.npy")
        np.save(tmp_u, np.asarray(U_0_R, dtype=np.float64))
        os.replace(tmp_u, u_p)

        tmp_s = s_p.with_suffix(".tmp.npy")
        np.save(tmp_s, np.asarray(scales, dtype=np.float64))
        os.replace(tmp_s, s_p)

        meta = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "method": method,
            "seed": seed,
            "fingerprint": fingerprint,
        }
        atomic_write_json(meta_p, meta, indent=2)

    # -----------------------------------------------------------------------
    # MMD Components & D_X
    # -----------------------------------------------------------------------

    def get_mmd_sigma(self, dataset_id: str, outer_fold: int) -> Optional[Tuple[float, str, float]]:
        p = self.mmd_dir / f"{dataset_id}_fold_{outer_fold}_sigma.json"
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                doc = json.load(f)
            return float(doc["sigma"]), str(doc["status"]), float(doc["sum_xx"])
        except Exception:
            return None

    def put_mmd_sigma(self, dataset_id: str, outer_fold: int, sigma: float, status: str, sum_xx: float) -> None:
        p = self.mmd_dir / f"{dataset_id}_fold_{outer_fold}_sigma.json"
        doc = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "sigma": float(sigma),
            "status": str(status),
            "sum_xx": float(sum_xx),
        }
        atomic_write_json(p, doc, indent=2)

    def get_dx(self, dataset_id: str, outer_fold: int, condition: str) -> Optional[float]:
        p = self.mmd_dir / f"{dataset_id}_fold_{outer_fold}_{condition}_dx.json"
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                doc = json.load(f)
            return float(doc["D_X"])
        except Exception:
            return None

    def put_dx(self, dataset_id: str, outer_fold: int, condition: str, dx: float) -> None:
        p = self.mmd_dir / f"{dataset_id}_fold_{outer_fold}_{condition}_dx.json"
        doc = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": condition,
            "D_X": float(dx),
        }
        atomic_write_json(p, doc, indent=2)

    # -----------------------------------------------------------------------
    # Cache Size, Validation & Safe Clear
    # -----------------------------------------------------------------------

    def cache_size_report(self) -> Dict[str, Any]:
        """Compute disk usage across cache categories in GB."""
        def dir_size_gb(d: Path) -> float:
            if not d.exists():
                return 0.0
            total_bytes = sum(f.stat().st_size for f in d.glob("**/*") if f.is_file())
            return round(total_bytes / (1024 ** 3), 4)

        ds_gb = dir_size_gb(self.datasets_dir)
        sc_gb = dir_size_gb(self.scenarios_dir)
        md_gb = dir_size_gb(self.models_dir)
        mb_gb = dir_size_gb(self.memberships_dir)
        mmd_gb = dir_size_gb(self.mmd_dir)

        return {
            "source_arrays_gb": ds_gb,
            "scenario_arrays_gb": sc_gb,
            "model_cache_gb": md_gb,
            "membership_scale_gb": mb_gb,
            "mmd_cache_gb": mmd_gb,
            "total_cache_gb": round(ds_gb + sc_gb + md_gb + mb_gb + mmd_gb, 4),
        }

    def validate_cache(self) -> Dict[str, Any]:
        """Validate integrity of all stored cache items."""
        valid_items = 0
        corrupt_items = 0

        for npy_file in self.root.glob("**/*.npy"):
            try:
                arr = np.load(npy_file, mmap_mode="r")
                if arr.size > 0 and arr.dtype == np.float64:
                    valid_items += 1
                else:
                    corrupt_items += 1
            except Exception:
                corrupt_items += 1

        for joblib_file in self.models_dir.glob("*.joblib"):
            try:
                m = joblib.load(joblib_file)
                if hasattr(m, "cluster_centers_"):
                    valid_items += 1
                else:
                    corrupt_items += 1
            except Exception:
                corrupt_items += 1

        return {
            "valid_items": valid_items,
            "corrupt_items": corrupt_items,
            "size_report": self.cache_size_report(),
        }

    def clear_cache(self, confirmed: bool = False) -> None:
        """Safely delete cache after verifying marker file."""
        if not confirmed:
            raise PermissionError("Cache clear requires explicit confirmation")
        if not self.marker_file.exists():
            raise FileNotFoundError(f"Safety violation: marker file missing: {self.marker_file}")
        with open(self.marker_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("marker") != "clusterdrift_phase7_cache":
            raise ValueError("Safety violation: invalid marker file")

        shutil.rmtree(self.root)