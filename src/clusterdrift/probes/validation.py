"""Validation and staleness checking routines for saved probe bank descriptors."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from clusterdrift.probes.bank import load_current_probe_descriptor, load_reference_probe_descriptor
from clusterdrift.probes.hashing import (
    compute_current_bank_sha256,
    compute_file_sha256,
    compute_reference_bank_sha256,
    derive_current_probe_seed,
    derive_reference_probe_seed,
)


def validate_saved_reference_descriptor(
    spec_path: Path,
    expected_canonical_bundle_sha256: str,
    expected_split_sha256: str,
    expected_preprocessing_config_sha256: str,
    expected_probe_protocol_sha256: str,
    global_probe_seed: int,
    dataset_id: str,
    outer_fold: int,
) -> Tuple[bool, Optional[str], Optional[Any]]:
    """Validate saved reference probe descriptor JSON and companion NPZ."""
    if not spec_path.exists():
        return False, f"Descriptor file missing: {spec_path}", None

    npz_path = spec_path.with_suffix(".npz")
    if not npz_path.exists():
        return False, f"Companion NPZ missing: {npz_path}", None

    try:
        desc, indices = load_reference_probe_descriptor(spec_path)
    except Exception as e:
        return False, f"Corrupted descriptor or NPZ: {e}", None

    if desc.dataset_id != dataset_id or desc.outer_fold != outer_fold:
        return False, f"Descriptor metadata mismatch for {spec_path}", None
    if desc.canonical_bundle_sha256 != expected_canonical_bundle_sha256:
        return False, "Canonical bundle hash mismatch", None
    if desc.split_sha256 != expected_split_sha256:
        return False, "Split hash mismatch", None
    if desc.preprocessing_config_sha256 != expected_preprocessing_config_sha256:
        return False, "Preprocessing config hash mismatch", None
    if desc.probe_protocol_sha256 != expected_probe_protocol_sha256:
        return False, "Probe protocol hash mismatch", None

    expected_seed = derive_reference_probe_seed(global_probe_seed, dataset_id, outer_fold, expected_split_sha256)
    if desc.derived_seed != expected_seed:
        return False, f"Derived seed mismatch: got {desc.derived_seed}, expected {expected_seed}", None

    # Recompute NPZ hash
    npz_sha = compute_file_sha256(npz_path)
    if desc.canonical_row_indices_sha256 != npz_sha:
        return False, "Companion NPZ byte SHA-256 mismatch", None

    # Recompute bank hash
    recomputed_bank_sha = compute_reference_bank_sha256(
        dataset_id=dataset_id,
        outer_fold=outer_fold,
        canonical_row_indices=indices,
        canonical_bundle_sha256=expected_canonical_bundle_sha256,
        split_sha256=expected_split_sha256,
        preprocessing_config_sha256=expected_preprocessing_config_sha256,
        probe_protocol_sha256=expected_probe_protocol_sha256,
    )
    if desc.probe_bank_sha256 != recomputed_bank_sha:
        return False, "Probe bank SHA-256 mismatch", None

    return True, None, desc


def validate_saved_current_descriptor(
    spec_path: Path,
    expected_shift_spec_sha256: str,
    expected_shift_protocol_sha256: str,
    expected_preprocessing_config_sha256: str,
    expected_probe_protocol_sha256: str,
    global_probe_seed: int,
    dataset_id: str,
    outer_fold: int,
    condition: str,
) -> Tuple[bool, Optional[str], Optional[Any]]:
    """Validate saved current probe descriptor JSON and companion NPZ."""
    if not spec_path.exists():
        return False, f"Descriptor file missing: {spec_path}", None

    npz_path = spec_path.with_suffix(".npz")
    if not npz_path.exists():
        return False, f"Companion NPZ missing: {npz_path}", None

    try:
        desc, positions, canonical_rows = load_current_probe_descriptor(spec_path)
    except Exception as e:
        return False, f"Corrupted descriptor or NPZ: {e}", None

    if desc.dataset_id != dataset_id or desc.outer_fold != outer_fold or desc.condition != condition:
        return False, f"Descriptor metadata mismatch for {spec_path}", None
    if desc.shift_spec_sha256 != expected_shift_spec_sha256:
        return False, "Shift spec hash mismatch", None
    if desc.shift_protocol_sha256 != expected_shift_protocol_sha256:
        return False, "Shift protocol hash mismatch", None
    if desc.preprocessing_config_sha256 != expected_preprocessing_config_sha256:
        return False, "Preprocessing config hash mismatch", None
    if desc.probe_protocol_sha256 != expected_probe_protocol_sha256:
        return False, "Probe protocol hash mismatch", None

    expected_seed = derive_current_probe_seed(
        global_probe_seed, dataset_id, outer_fold, condition, expected_shift_spec_sha256
    )
    if desc.derived_seed != expected_seed:
        return False, f"Derived seed mismatch: got {desc.derived_seed}, expected {expected_seed}", None

    npz_sha = compute_file_sha256(npz_path)
    if desc.selected_positions_sha256 != npz_sha:
        return False, "Companion NPZ byte SHA-256 mismatch", None

    recomputed_bank_sha = compute_current_bank_sha256(
        dataset_id=dataset_id,
        outer_fold=outer_fold,
        condition=condition,
        selected_positions=positions,
        canonical_row_indices=canonical_rows,
        shift_spec_sha256=expected_shift_spec_sha256,
        shift_protocol_sha256=expected_shift_protocol_sha256,
        preprocessing_config_sha256=expected_preprocessing_config_sha256,
        probe_protocol_sha256=expected_probe_protocol_sha256,
    )
    if desc.probe_bank_sha256 != recomputed_bank_sha:
        return False, "Probe bank SHA-256 mismatch", None

    return True, None, desc
