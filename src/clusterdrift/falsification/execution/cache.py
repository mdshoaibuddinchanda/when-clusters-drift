"""Persistent, Content-Addressed, Ephemeral Cache for Phase 7 Execution."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from clusterdrift.falsification.execution.environment import get_library_versions
from clusterdrift.falsification.execution.resources import assert_not_frozen_data_path
from clusterdrift.methods.fcm import FCM
from clusterdrift.shifts.hashing import (
    atomic_write_json,
    compute_canonical_json_sha256,
    compute_file_sha256,
)


def compute_content_fingerprint(data: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint for cache key dictionary."""
    return compute_canonical_json_sha256(data)


def compute_source_cache_fingerprint(
    project_root: Path, dataset_id: str, outer_fold: int, prep_config_sha: str
) -> str:
    """Bind source-derived cache state to data, split, preprocessing, and probe."""
    root = Path(project_root)
    ds_dir = root / "data/canonical/controlled" / _safe_component(dataset_id, "dataset_id")
    split = root / "data/splits/controlled" / dataset_id / f"fold_{int(outer_fold)}.npz"
    probe = root / "data/probes/reference" / dataset_id / f"fold_{int(outer_fold)}.json"
    return compute_content_fingerprint({
        "kind": "phase7_source_data_v2",
        "dataset_id": dataset_id,
        "outer_fold": int(outer_fold),
        "features_sha256": compute_file_sha256(ds_dir / "features.parquet"),
        "metadata_sha256": compute_file_sha256(ds_dir / "metadata.json"),
        "split_sha256": compute_file_sha256(split),
        "reference_probe_sha256": compute_file_sha256(probe),
        "preprocessing_sha256": prep_config_sha,
    })


def compute_model_cache_fingerprint(
    project_root: Path,
    source_fingerprint: str,
    method: str,
    seed: int,
    K: int,
) -> str:
    """Bind a fitted source model to source state and method configuration."""
    return compute_content_fingerprint({
        "kind": "phase7_source_model_v2",
        "source_fingerprint": source_fingerprint,
        "method": _safe_component(method, "method"),
        "seed": int(seed),
        "K": int(K),
        "methods_config_sha256": compute_file_sha256(Path(project_root) / "configs/methods.yaml"),
    })


def _safe_component(value: Union[str, int], field: str) -> str:
    """Reject path separators and ambiguous filename components."""
    text = str(value)
    if not text or text in {".", ".."} or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in text):
        raise ValueError(f"Unsafe cache {field}: {value!r}")
    return text


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        np.save(tmp, np.asarray(value, dtype=np.float64), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        np.savez(tmp, **{key: np.asarray(value) for key, value in arrays.items()})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _digest_matches(path: Path, expected: Any) -> bool:
    return isinstance(expected, str) and len(expected) == 64 and compute_file_sha256(path) == expected


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
        dataset_id = _safe_component(dataset_id, "dataset_id")
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
            if not _digest_matches(x_src_p, meta.get("x_src_sha256")):
                return None
            if not _digest_matches(a_r_p, meta.get("a_r_sha256")):
                return None
            X_src = np.load(x_src_p, mmap_mode="r", allow_pickle=False)
            A_R = np.load(a_r_p, mmap_mode="r", allow_pickle=False)
            if list(X_src.shape) != meta.get("x_src_shape") or list(A_R.shape) != meta.get("a_r_shape"):
                return None
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
        dataset_id = _safe_component(dataset_id, "dataset_id")
        x_src_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}_X_src.npy"
        a_r_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}_A_R.npy"
        meta_p = self.datasets_dir / f"{dataset_id}_fold_{outer_fold}.json"

        _atomic_save_npy(x_src_p, X_source_trans)
        _atomic_save_npy(a_r_p, A_R)

        meta = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "fingerprint": fingerprint,
            "x_src_shape": list(X_source_trans.shape),
            "a_r_shape": list(A_R.shape),
            "dtype": "float64",
            "x_src_sha256": compute_file_sha256(x_src_p),
            "a_r_sha256": compute_file_sha256(a_r_p),
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
        dataset_id = _safe_component(dataset_id, "dataset_id")
        condition = _safe_component(condition, "condition")
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
            if not _digest_matches(x_tgt_p, meta.get("x_tgt_sha256")):
                return None
            if not _digest_matches(a_c_p, meta.get("a_c_sha256")):
                return None
            X_tgt = np.load(x_tgt_p, mmap_mode="r", allow_pickle=False)
            A_C = np.load(a_c_p, mmap_mode="r", allow_pickle=False)
            if list(X_tgt.shape) != meta.get("x_tgt_shape") or list(A_C.shape) != meta.get("a_c_shape"):
                return None
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
        dataset_id = _safe_component(dataset_id, "dataset_id")
        condition = _safe_component(condition, "condition")
        base_name = f"{dataset_id}_fold_{outer_fold}_{condition}"
        x_tgt_p = self.scenarios_dir / f"{base_name}_X_tgt.npy"
        a_c_p = self.scenarios_dir / f"{base_name}_A_C.npy"
        meta_p = self.scenarios_dir / f"{base_name}.json"

        _atomic_save_npy(x_tgt_p, X_target_shifted_trans)
        _atomic_save_npy(a_c_p, A_C)

        meta = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": condition,
            "fingerprint": fingerprint,
            "x_tgt_shape": list(X_target_shifted_trans.shape),
            "a_c_shape": list(A_C.shape),
            "dtype": "float64",
            "x_tgt_sha256": compute_file_sha256(x_tgt_p),
            "a_c_sha256": compute_file_sha256(a_c_p),
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
        """Retrieve a safely serialized FCM model after digest verification.

        Joblib/pickle is deliberately not accepted: deserialization can execute
        code before post-load integrity checks.  Legacy joblib entries are cache
        misses and are recomputed.
        """
        dataset_id = _safe_component(dataset_id, "dataset_id")
        method = _safe_component(method, "method")
        base_name = f"{dataset_id}_fold_{outer_fold}_{method}_seed_{seed}"
        meta_p = self.models_dir / f"{base_name}.json"
        model_p = self.models_dir / f"{base_name}.npz"

        if not (meta_p.exists() and model_p.exists()):
            return None

        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != expected_fingerprint:
                return None
            if meta.get("serialization") != "fcm_numeric_npz_v1":
                return None
            if not _digest_matches(model_p, meta.get("model_sha256")):
                return None
            if any(meta.get(key) != expected for key, expected in {
                "dataset_id": dataset_id, "outer_fold": int(outer_fold),
                "method": method, "seed": int(seed),
            }.items()):
                return None
            with np.load(model_p, allow_pickle=False) as payload:
                centers = np.asarray(payload["cluster_centers"], dtype=np.float64)
            if centers.ndim != 2 or centers.shape[0] != int(meta["K"]) or not np.isfinite(centers).all():
                return None
            model = FCM(
                n_clusters=int(meta["K"]),
                m=float(meta.get("m", 2.0)),
                random_state=int(seed),
                max_iter=int(meta.get("max_iter", 150)),
                tol=float(meta.get("tol", 1e-5)),
                initialization=str(meta.get("initialization", "kmeans++")),
                fuzzifier_policy=str(meta.get("fuzzifier_policy", "dimension_adaptive")),
            )
            model.cluster_centers_ = centers
            model.effective_m_ = float(meta["effective_m"])
            model.effective_dimension_ = int(meta.get("effective_dimension", centers.shape[1]))
            model.fuzzifier_policy_ = str(meta.get("fuzzifier_policy_resolved", meta.get("fuzzifier_policy", "dimension_adaptive")))
            model.fuzzifier_clipped_ = bool(meta.get("fuzzifier_clipped", False))
            model.converged_ = bool(meta.get("converged", False))
            model.n_iter_ = int(meta.get("n_iter", 0))
            model.status_ = str(meta.get("status", "INVALID_INPUT"))
            model.degenerate_solution_ = bool(meta.get("degenerate", False))
            model.initialization_method_ = str(meta.get("initialization", "kmeans++"))
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
        """Persist fitted FCM inference state without executable serialization."""
        dataset_id = _safe_component(dataset_id, "dataset_id")
        method = _safe_component(method, "method")
        if not isinstance(model, FCM):
            raise TypeError("Persistent Phase-7 model cache supports safe FCM numeric state only")
        base_name = f"{dataset_id}_fold_{outer_fold}_{method}_seed_{seed}"
        model_p = self.models_dir / f"{base_name}.npz"
        meta_p = self.models_dir / f"{base_name}.json"
        centers = np.asarray(getattr(model, "cluster_centers_", None), dtype=np.float64)
        if centers.ndim != 2 or centers.shape[0] != int(K) or not np.isfinite(centers).all():
            raise ValueError("Cannot cache an unfitted or non-finite FCM model")
        _atomic_save_npz(model_p, cluster_centers=centers)

        fp_str = getattr(model, "fingerprint_", "") or getattr(model, "model_fingerprint", "")
        meta = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "method": method,
            "seed": seed,
            "K": K,
            "fingerprint": fingerprint,
            "model_fingerprint": str(fp_str),
            "serialization": "fcm_numeric_npz_v1",
            "model_sha256": compute_file_sha256(model_p),
            "m": float(model.m),
            "max_iter": int(model.max_iter),
            "tol": float(model.tol),
            "initialization": str(model.initialization),
            "fuzzifier_policy": str(model.fuzzifier_policy),
            "fuzzifier_policy_resolved": str(model.fuzzifier_policy_),
            "effective_m": float(model.effective_m_),
            "effective_dimension": int(model.effective_dimension_),
            "fuzzifier_clipped": bool(model.fuzzifier_clipped_),
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "status": str(model.status_),
            "degenerate": bool(model.degenerate_solution_),
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
        dataset_id = _safe_component(dataset_id, "dataset_id")
        method = _safe_component(method, "method")
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
            if not _digest_matches(u_p, meta.get("u_sha256")) or not _digest_matches(s_p, meta.get("scales_sha256")):
                return None
            U_0_R = np.load(u_p, mmap_mode="r", allow_pickle=False)
            scales = np.load(s_p, allow_pickle=False)
            if list(U_0_R.shape) != meta.get("u_shape") or list(scales.shape) != meta.get("scales_shape"):
                return None
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
        dataset_id = _safe_component(dataset_id, "dataset_id")
        method = _safe_component(method, "method")
        base_name = f"{dataset_id}_fold_{outer_fold}_{method}_seed_{seed}"
        u_p = self.memberships_dir / f"{base_name}_U.npy"
        s_p = self.memberships_dir / f"{base_name}_scales.npy"
        meta_p = self.memberships_dir / f"{base_name}.json"

        _atomic_save_npy(u_p, U_0_R)
        _atomic_save_npy(s_p, scales)

        meta = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "method": method,
            "seed": seed,
            "fingerprint": fingerprint,
            "u_shape": list(U_0_R.shape),
            "scales_shape": list(scales.shape),
            "u_sha256": compute_file_sha256(u_p),
            "scales_sha256": compute_file_sha256(s_p),
        }
        atomic_write_json(meta_p, meta, indent=2)

    # -----------------------------------------------------------------------
    # MMD Components & D_X
    # -----------------------------------------------------------------------

    def get_mmd_sigma(
        self, dataset_id: str, outer_fold: int, expected_fingerprint: str
    ) -> Optional[Tuple[float, str, float]]:
        dataset_id = _safe_component(dataset_id, "dataset_id")
        p = self.mmd_dir / f"{dataset_id}_fold_{outer_fold}_sigma.json"
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("fingerprint") != expected_fingerprint:
                return None
            return float(doc["sigma"]), str(doc["status"]), float(doc["sum_xx"])
        except Exception:
            return None

    def put_mmd_sigma(
        self, dataset_id: str, outer_fold: int, fingerprint: str,
        sigma: float, status: str, sum_xx: float,
    ) -> None:
        dataset_id = _safe_component(dataset_id, "dataset_id")
        p = self.mmd_dir / f"{dataset_id}_fold_{outer_fold}_sigma.json"
        doc = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "fingerprint": fingerprint,
            "sigma": float(sigma),
            "status": str(status),
            "sum_xx": float(sum_xx),
        }
        atomic_write_json(p, doc, indent=2)

    def get_dx(
        self, dataset_id: str, outer_fold: int, condition: str, expected_fingerprint: str
    ) -> Optional[float]:
        dataset_id = _safe_component(dataset_id, "dataset_id")
        condition = _safe_component(condition, "condition")
        p = self.mmd_dir / f"{dataset_id}_fold_{outer_fold}_{condition}_dx.json"
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("fingerprint") != expected_fingerprint:
                return None
            return float(doc["D_X"])
        except Exception:
            return None

    def put_dx(
        self, dataset_id: str, outer_fold: int, condition: str, fingerprint: str, dx: float
    ) -> None:
        dataset_id = _safe_component(dataset_id, "dataset_id")
        condition = _safe_component(condition, "condition")
        p = self.mmd_dir / f"{dataset_id}_fold_{outer_fold}_{condition}_dx.json"
        doc = {
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": condition,
            "fingerprint": fingerprint,
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
        """Validate non-executable cache payloads without loading pickle/joblib."""
        valid_items = 0
        corrupt_items = 0

        for npy_file in self.root.glob("**/*.npy"):
            try:
                arr = np.load(npy_file, mmap_mode="r", allow_pickle=False)
                if arr.size > 0 and arr.dtype == np.float64:
                    valid_items += 1
                else:
                    corrupt_items += 1
            except Exception:
                corrupt_items += 1

        for npz_file in self.models_dir.glob("*.npz"):
            try:
                meta_file = npz_file.with_suffix(".json")
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                with np.load(npz_file, allow_pickle=False) as payload:
                    centers = payload["cluster_centers"]
                if _digest_matches(npz_file, meta.get("model_sha256")) and centers.ndim == 2 and np.isfinite(centers).all():
                    valid_items += 1
                else:
                    corrupt_items += 1
            except Exception:
                corrupt_items += 1

        # Legacy executable serialization is never inspected; it is invalidated.
        corrupt_items += sum(1 for _ in self.models_dir.glob("*.joblib"))

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
        resolved_root = self.root.resolve()
        resolved_cache_dir = self.cache_dir.resolve()
        if self.root.is_symlink() or resolved_cache_dir not in resolved_root.parents:
            raise ValueError(f"Safety violation: cache root escapes configured directory: {resolved_root}")
        shutil.rmtree(self.root)
