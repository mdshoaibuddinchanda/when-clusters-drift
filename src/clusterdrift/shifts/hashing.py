"""Cryptographic hashing and deterministic seed derivation for Phase 4 shifts."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


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
    canonical_bundle_hash: str,
    split_hash: str,
    dataset_id: str,
    outer_fold: int,
) -> str:
    """Bind canonical raw dataset, split, dataset ID, and outer fold into scenario input hash."""
    payload = {
        "canonical_bundle_hash": canonical_bundle_hash,
        "dataset_id": dataset_id,
        "outer_fold": outer_fold,
        "split_hash": split_hash,
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
