"""Hashing, seed derivation, and input lock utilities for Phase 5 Dual Probe Banks."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_file_sha256(path: Path) -> str:
    """Compute deterministic SHA-256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def derive_integer_seed(payload: Dict[str, Any]) -> int:
    """Derive deterministic non-negative 31-bit integer seed from arbitrary JSON-serializable payload."""
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_json.encode("utf-8")).digest()
    val = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(val % 2147483647)


def derive_reference_probe_seed(
    global_probe_seed: int,
    dataset_id: str,
    outer_fold: int,
    split_sha256: str,
) -> int:
    """Derive deterministic seed for reference probe bank sampling."""
    payload = {
        "global_probe_seed": int(global_probe_seed),
        "dataset_id": str(dataset_id),
        "outer_fold": int(outer_fold),
        "bank_type": "reference",
        "split_sha256": str(split_sha256),
    }
    return derive_integer_seed(payload)


def derive_current_probe_seed(
    global_probe_seed: int,
    dataset_id: str,
    outer_fold: int,
    condition: str,
    shift_spec_sha256: str,
) -> int:
    """Derive deterministic seed for current probe bank sampling."""
    payload = {
        "global_probe_seed": int(global_probe_seed),
        "dataset_id": str(dataset_id),
        "outer_fold": int(outer_fold),
        "condition": str(condition),
        "shift_spec_sha256": str(shift_spec_sha256),
        "bank_type": "current",
    }
    return derive_integer_seed(payload)


def compute_probe_protocol_sha256(probe_cfg: Dict[str, Any]) -> str:
    """Compute protocol hash for probe configuration."""
    relevant_keys = ["protocol_version", "global_probe_seed", "reference", "current", "selection"]
    filtered = {k: probe_cfg[k] for k in relevant_keys if k in probe_cfg}
    canonical_json = json.dumps(filtered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_alignment_protocol_sha256(alignment_cfg: Dict[str, Any]) -> str:
    """Compute protocol hash for alignment configuration."""
    relevant_keys = ["protocol_version", "alignment"]
    filtered = {k: alignment_cfg[k] for k in relevant_keys if k in alignment_cfg}
    canonical_json = json.dumps(filtered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_reference_bank_sha256(
    dataset_id: str,
    outer_fold: int,
    canonical_row_indices: np.ndarray,
    canonical_bundle_sha256: str,
    split_sha256: str,
    preprocessing_config_sha256: str,
    probe_protocol_sha256: str,
) -> str:
    """Compute deterministic SHA-256 for a reference probe bank."""
    hasher = hashlib.sha256()
    metadata = {
        "dataset_id": dataset_id,
        "outer_fold": int(outer_fold),
        "bank_type": "reference",
        "canonical_bundle_sha256": canonical_bundle_sha256,
        "split_sha256": split_sha256,
        "preprocessing_config_sha256": preprocessing_config_sha256,
        "probe_protocol_sha256": probe_protocol_sha256,
        "num_rows": int(len(canonical_row_indices)),
    }
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(np.ascontiguousarray(canonical_row_indices, dtype=np.int64).tobytes())
    return hasher.hexdigest()


def compute_current_bank_sha256(
    dataset_id: str,
    outer_fold: int,
    condition: str,
    selected_positions: np.ndarray,
    canonical_row_indices: np.ndarray,
    shift_spec_sha256: str,
    shift_protocol_sha256: str,
    preprocessing_config_sha256: str,
    probe_protocol_sha256: str,
) -> str:
    """Compute deterministic SHA-256 for a current probe bank."""
    hasher = hashlib.sha256()
    metadata = {
        "dataset_id": dataset_id,
        "outer_fold": int(outer_fold),
        "condition": condition,
        "bank_type": "current",
        "shift_spec_sha256": shift_spec_sha256,
        "shift_protocol_sha256": shift_protocol_sha256,
        "preprocessing_config_sha256": preprocessing_config_sha256,
        "probe_protocol_sha256": probe_protocol_sha256,
        "num_rows": int(len(selected_positions)),
    }
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(np.ascontiguousarray(selected_positions, dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(canonical_row_indices, dtype=np.int64).tobytes())
    return hasher.hexdigest()


def build_phase5_input_lock(
    project_root: Path,
    probe_cfg: Dict[str, Any],
    alignment_cfg: Dict[str, Any],
    generated_from_commit: str,
) -> Dict[str, Any]:
    """Build Phase-5 input lock binding Phase 1-4 provenance."""
    splits_manifest_path = project_root / "data" / "splits" / "split_manifest.json"
    prep_cfg_path = project_root / "configs" / "preprocessing.yaml"
    datasets_manifest_path = project_root / "data" / "manifests" / "datasets.json"
    methods_cfg_path = project_root / "configs" / "methods.yaml"
    probes_cfg_path = project_root / "configs" / "probes.yaml"
    alignment_cfg_path = project_root / "configs" / "alignment.yaml"

    phase4_input_lock_path = project_root / "data" / "shifts" / "phase4_input_lock.json"
    phase4_manifest_path = project_root / "data" / "shifts" / "shift_manifest.json"
    shifts_cfg_path = project_root / "configs" / "shifts.yaml"

    from clusterdrift.shifts.hashing import compute_shift_protocol_sha256
    with open(shifts_cfg_path, "r", encoding="utf-8") as f:
        shifts_cfg_data = json.load(f) if shifts_cfg_path.suffix == ".json" else yaml_load(shifts_cfg_path)

    phase4_protocol_sha = compute_shift_protocol_sha256(shifts_cfg_data)

    lock_doc = {
        "protocol_version": probe_cfg.get("protocol_version", 1),
        "phase3_freeze_commit": "8a471625af6301fe8941090748c8057e08324326",
        "phase4_producer_commit": "21bcaa7bb00b07de5ee4671adfb06932610bf92d",
        "phase4_artifact_commit": "6f8dd88b59a297ae05da035835c4055aab43fb29",
        "phase4_tooling_freeze_commit": "f08335e8a4822f2b95e379abf955a80d5c0fb2c8",
        "generated_from_commit": generated_from_commit,
        "phase4_input_lock_sha256": compute_file_sha256(phase4_input_lock_path),
        "phase4_shift_manifest_sha256": compute_file_sha256(phase4_manifest_path),
        "phase4_protocol_sha256": phase4_protocol_sha,
        "phase2_split_manifest_sha256": compute_file_sha256(splits_manifest_path),
        "phase2_preprocessing_config_sha256": compute_file_sha256(prep_cfg_path),
        "phase1_datasets_manifest_sha256": compute_file_sha256(datasets_manifest_path),
        "methods_config_sha256": compute_file_sha256(methods_cfg_path),
        "alignment_config_sha256": compute_file_sha256(alignment_cfg_path),
        "probe_config_sha256": compute_file_sha256(probes_cfg_path),
        "probe_protocol_sha256": compute_probe_protocol_sha256(probe_cfg),
        "alignment_protocol_sha256": compute_alignment_protocol_sha256(alignment_cfg),
        "global_probe_seed": probe_cfg.get("global_probe_seed", 2026090705),
    }
    return lock_doc


def yaml_load(p: Path) -> Dict[str, Any]:
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
