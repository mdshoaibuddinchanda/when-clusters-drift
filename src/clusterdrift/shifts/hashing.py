"""Cryptographic hashing, provenance binding, and atomic persistence for Phase 4 shifts."""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of a byte string."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def derive_integer_seed(
    global_seed: int,
    dataset_id: str,
    outer_fold: int,
    family: str,
    purpose: str,
) -> int:
    """Derive a deterministic integer seed from a stable SHA-256 payload.
    
    Guarantees order-independent, cross-platform reproducible randomness.
    Never relies on Python's process-dependent hash().
    """
    payload = f"{global_seed}:{dataset_id}:{outer_fold}:{family}:{purpose}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31 - 1)


def compute_canonical_json_sha256(obj: Any) -> str:
    """Compute SHA-256 hash of a canonical JSON serialization."""
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def compute_shift_protocol_sha256(cfg_dict: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of the frozen shift protocol configuration."""
    return compute_canonical_json_sha256(cfg_dict)


def compute_scenario_input_sha256(
    canonical_bundle_sha256: str = "",
    split_sha256: str = "",
    dataset_id: str = "",
    outer_fold: int = 0,
    **kwargs: Any,
) -> str:
    """Bind canonical bundle hash, scientific split hash, dataset ID, and outer fold.
    
    Guarantees that label changes, metadata/role mutations, or split partition
    alterations strictly invalidate the scenario input hash.
    """
    bundle_hash = canonical_bundle_sha256 or kwargs.get("canonical_bundle_hash", "")
    sp_hash = split_sha256 or kwargs.get("split_hash", "")
    payload = {
        "canonical_bundle_sha256": str(bundle_hash),
        "dataset_id": str(dataset_id),
        "outer_fold": int(outer_fold),
        "split_sha256": str(sp_hash),
    }
    return compute_canonical_json_sha256(payload)


def compute_shift_spec_sha256(
    scenario_input_sha256: str,
    shift_protocol_sha256: str,
    condition: str,
    family: str,
    severity: str,
    derived_seed: int,
    spec_descriptor: Dict[str, Any],
) -> str:
    """Compute deterministic hash of the complete shift specification."""
    payload = {
        "condition": condition,
        "derived_seed": derived_seed,
        "family": family,
        "scenario_input_sha256": scenario_input_sha256,
        "severity": severity,
        "shift_protocol_sha256": shift_protocol_sha256,
        "spec_descriptor": spec_descriptor,
    }
    return compute_canonical_json_sha256(payload)


def load_canonical_bundle_hashes(manifest_path: Union[str, Path]) -> Dict[str, str]:
    """Load canonical bundle hashes from Phase-1 dataset manifest."""
    p = Path(manifest_path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    bundle_hashes: Dict[str, str] = {}
    if isinstance(data, list):
        for entry in data:
            ds_id = entry.get("dataset_id")
            bundle_hash = entry.get("canonical_bundle_sha256") or entry.get("canonical_sha256")
            if ds_id and bundle_hash:
                bundle_hashes[ds_id] = bundle_hash
    elif isinstance(data, dict):
        for ds_id, meta in data.items():
            if isinstance(meta, dict):
                bundle_hash = meta.get("canonical_bundle_sha256") or meta.get("canonical_sha256")
                if bundle_hash:
                    bundle_hashes[ds_id] = bundle_hash
    return bundle_hashes


def load_phase2_split_hashes(split_manifest_path: Union[str, Path]) -> Dict[Tuple[str, int], Dict[str, str]]:
    """Load Phase-2 scientific split hashes and companion file hashes from split manifest."""
    p = Path(split_manifest_path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    split_records: Dict[Tuple[str, int], Dict[str, str]] = {}
    for entry in data.get("splits", []):
        ds_id = entry.get("dataset_id")
        fold = entry.get("outer_fold")
        if ds_id is not None and fold is not None:
            split_records[(ds_id, fold)] = {
                "split_sha256": entry.get("split_sha256", ""),
                "json_sha256": entry.get("json_sha256", ""),
                "npz_sha256": entry.get("npz_sha256", ""),
                "json_path": entry.get("json_path", ""),
                "npz_path": entry.get("npz_path", ""),
            }
    return split_records


def atomic_write_json(
    path: Union[str, Path],
    obj: Any,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    """Write JSON atomically using temporary file and os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_id = hashlib.sha256(f"{time.time_ns()}:{os.getpid()}".encode()).hexdigest()[:8]
    temp_path = target.with_name(f".{target.name}.tmp_{tmp_id}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, sort_keys=sort_keys)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_csv(
    path: Union[str, Path],
    df: Any,
    index: bool = False,
) -> None:
    """Write DataFrame to CSV atomically using temporary file and os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_id = hashlib.sha256(f"{time.time_ns()}:{os.getpid()}".encode()).hexdigest()[:8]
    temp_path = target.with_name(f".{target.name}.tmp_{tmp_id}")
    try:
        df.to_csv(temp_path, index=index)
        with open(temp_path, "a", encoding="utf-8") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_parquet(
    path: Union[str, Path],
    df: Any,
    index: bool = False,
    **kwargs: Any,
) -> None:
    """Write a parquet table atomically in the destination directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_id = hashlib.sha256(f"{time.time_ns()}:{os.getpid()}".encode()).hexdigest()[:8]
    temp_path = target.with_name(f".{target.name}.tmp_{tmp_id}")
    try:
        df.to_parquet(temp_path, index=index, **kwargs)
        with open(temp_path, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_npy(path: Union[str, Path], array: Any) -> None:
    """Write one non-pickle NPY array atomically in the destination directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_id = hashlib.sha256(f"{time.time_ns()}:{os.getpid()}".encode()).hexdigest()[:8]
    temp_path = target.with_name(f".{target.name}.tmp_{tmp_id}")
    try:
        with open(temp_path, "wb") as f:
            np.save(f, np.asarray(array), allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_npz(
    path: Union[str, Path],
    **arrays: Any,
) -> None:
    """Write compressed NPZ array file atomically using temporary file and os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_id = hashlib.sha256(f"{time.time_ns()}:{os.getpid()}".encode()).hexdigest()[:8]
    temp_path = target.with_name(f".{target.name}.tmp_{tmp_id}")
    try:
        with open(temp_path, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
