"""Data descriptors and atomic persistence for reference and current probe banks."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from clusterdrift.probes.hashing import (
    compute_array_sha256,
    compute_current_bank_sha256,
    compute_reference_bank_sha256,
)
from clusterdrift.shifts.hashing import atomic_write_json, atomic_write_npz, compute_file_sha256


@dataclass(frozen=True)
class ReferenceProbeDescriptor:
    """Descriptor for an outer source historical reference probe bank A^R."""
    dataset_id: str
    outer_fold: int
    bank_type: str  # "reference"
    canonical_bundle_sha256: str
    split_sha256: str
    preprocessing_config_sha256: str
    selection_policy: str
    global_probe_seed: int
    derived_seed: int
    available_rows: int
    selected_rows: int
    selected_source_positions_sha256: str
    canonical_row_indices_sha256: str
    npz_sha256: str
    probe_protocol_sha256: str
    probe_bank_sha256: str


@dataclass(frozen=True)
class CurrentProbeDescriptor:
    """Descriptor for an observed shifted target probe bank A_t^C."""
    dataset_id: str
    outer_fold: int
    condition: str
    bank_type: str  # "current"
    shift_spec_sha256: str
    shift_protocol_sha256: str
    preprocessing_config_sha256: str
    selection_policy: str
    global_probe_seed: int
    derived_seed: int
    available_current_rows: int
    selected_current_rows: int
    selected_current_positions_sha256: str
    target_partition_positions_sha256: str
    canonical_row_indices_sha256: str
    npz_sha256: str
    scenario_status: str
    probe_protocol_sha256: str
    probe_bank_sha256: str


def save_reference_probe_descriptor(
    spec_path: Path,
    metadata: Dict[str, Any],
    selected_source_positions: np.ndarray,
    canonical_row_indices: np.ndarray,
) -> ReferenceProbeDescriptor:
    """Atomically save reference probe descriptor JSON and companion NPZ array.

    Refactored execution sequence (Requirement 7):
    1. construct arrays;
    2. atomically write NPZ exactly once;
    3. compute NPZ SHA;
    4. construct final descriptor with distinct array hashes + NPZ SHA;
    5. atomically write JSON exactly once.
    """
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = spec_path.with_suffix(".npz")

    arr_pos = np.ascontiguousarray(selected_source_positions, dtype=np.int64)
    arr_can = np.ascontiguousarray(canonical_row_indices, dtype=np.int64)

    # 1 & 2. Write companion NPZ atomically exactly once
    atomic_write_npz(
        npz_path,
        selected_source_positions=arr_pos,
        canonical_row_indices=arr_can,
    )

    # 3. Compute NPZ file SHA-256
    npz_sha = compute_file_sha256(npz_path)

    # 4. Compute distinct array hashes and probe bank hash
    pos_sha = compute_array_sha256(arr_pos)
    can_sha = compute_array_sha256(arr_can)

    bank_sha = compute_reference_bank_sha256(
        dataset_id=metadata["dataset_id"],
        outer_fold=metadata["outer_fold"],
        selected_source_positions=arr_pos,
        canonical_row_indices=arr_can,
        canonical_bundle_sha256=metadata["canonical_bundle_sha256"],
        split_sha256=metadata["split_sha256"],
        preprocessing_config_sha256=metadata["preprocessing_config_sha256"],
        probe_protocol_sha256=metadata["probe_protocol_sha256"],
    )

    desc = ReferenceProbeDescriptor(
        dataset_id=metadata["dataset_id"],
        outer_fold=metadata["outer_fold"],
        bank_type="reference",
        canonical_bundle_sha256=metadata["canonical_bundle_sha256"],
        split_sha256=metadata["split_sha256"],
        preprocessing_config_sha256=metadata["preprocessing_config_sha256"],
        selection_policy=metadata["selection_policy"],
        global_probe_seed=metadata["global_probe_seed"],
        derived_seed=metadata["derived_seed"],
        available_rows=metadata["available_rows"],
        selected_rows=len(arr_pos),
        selected_source_positions_sha256=pos_sha,
        canonical_row_indices_sha256=can_sha,
        npz_sha256=npz_sha,
        probe_protocol_sha256=metadata["probe_protocol_sha256"],
        probe_bank_sha256=bank_sha,
    )

    # 5. Write descriptor JSON atomically exactly once
    atomic_write_json(spec_path, asdict(desc), indent=2, sort_keys=True)
    return desc


def load_reference_probe_descriptor(
    spec_path: Path,
) -> Tuple[ReferenceProbeDescriptor, np.ndarray, np.ndarray]:
    """Load reference probe descriptor and companion position/canonical row mapping arrays."""
    with open(spec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    desc = ReferenceProbeDescriptor(**data)

    npz_path = spec_path.with_suffix(".npz")
    if npz_path.exists():
        with np.load(npz_path) as npz:
            if "selected_source_positions" in npz:
                positions = npz["selected_source_positions"]
            else:
                positions = npz["canonical_row_indices"]
            canonical_rows = npz["canonical_row_indices"]
    else:
        positions = np.array([], dtype=np.int64)
        canonical_rows = np.array([], dtype=np.int64)

    return desc, positions, canonical_rows


def save_current_probe_descriptor(
    spec_path: Path,
    metadata: Dict[str, Any],
    selected_current_positions: np.ndarray,
    target_partition_positions: np.ndarray,
    canonical_row_indices: np.ndarray,
) -> CurrentProbeDescriptor:
    """Atomically save current probe descriptor JSON and companion NPZ arrays.

    Refactored execution sequence (Requirement 7):
    1. construct arrays;
    2. atomically write NPZ exactly once;
    3. compute NPZ SHA;
    4. construct final descriptor with distinct array hashes + NPZ SHA;
    5. atomically write JSON exactly once.
    """
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = spec_path.with_suffix(".npz")

    arr_cur = np.ascontiguousarray(selected_current_positions, dtype=np.int64)
    arr_tgt = np.ascontiguousarray(target_partition_positions, dtype=np.int64)
    arr_can = np.ascontiguousarray(canonical_row_indices, dtype=np.int64)

    # 1 & 2. Write companion NPZ atomically exactly once
    atomic_write_npz(
        npz_path,
        selected_current_positions=arr_cur,
        target_partition_positions=arr_tgt,
        canonical_row_indices=arr_can,
    )

    # 3. Compute NPZ file SHA-256
    npz_sha = compute_file_sha256(npz_path)

    # 4. Compute distinct array hashes and probe bank hash
    cur_sha = compute_array_sha256(arr_cur)
    tgt_sha = compute_array_sha256(arr_tgt)
    can_sha = compute_array_sha256(arr_can)

    bank_sha = compute_current_bank_sha256(
        dataset_id=metadata["dataset_id"],
        outer_fold=metadata["outer_fold"],
        condition=metadata["condition"],
        selected_current_positions=arr_cur,
        target_partition_positions=arr_tgt,
        canonical_row_indices=arr_can,
        shift_spec_sha256=metadata["shift_spec_sha256"],
        shift_protocol_sha256=metadata["shift_protocol_sha256"],
        preprocessing_config_sha256=metadata["preprocessing_config_sha256"],
        probe_protocol_sha256=metadata["probe_protocol_sha256"],
    )

    desc = CurrentProbeDescriptor(
        dataset_id=metadata["dataset_id"],
        outer_fold=metadata["outer_fold"],
        condition=metadata["condition"],
        bank_type="current",
        shift_spec_sha256=metadata["shift_spec_sha256"],
        shift_protocol_sha256=metadata["shift_protocol_sha256"],
        preprocessing_config_sha256=metadata["preprocessing_config_sha256"],
        selection_policy=metadata["selection_policy"],
        global_probe_seed=metadata["global_probe_seed"],
        derived_seed=metadata["derived_seed"],
        available_current_rows=metadata["available_current_rows"],
        selected_current_rows=len(arr_cur),
        selected_current_positions_sha256=cur_sha,
        target_partition_positions_sha256=tgt_sha,
        canonical_row_indices_sha256=can_sha,
        npz_sha256=npz_sha,
        scenario_status=metadata["scenario_status"],
        probe_protocol_sha256=metadata["probe_protocol_sha256"],
        probe_bank_sha256=bank_sha,
    )

    # 5. Write descriptor JSON atomically exactly once
    atomic_write_json(spec_path, asdict(desc), indent=2, sort_keys=True)
    return desc


def load_current_probe_descriptor(
    spec_path: Path,
) -> Tuple[CurrentProbeDescriptor, np.ndarray, np.ndarray, np.ndarray]:
    """Load current probe descriptor and companion position/row mapping arrays."""
    with open(spec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    desc = CurrentProbeDescriptor(**data)

    npz_path = spec_path.with_suffix(".npz")
    if npz_path.exists():
        with np.load(npz_path) as npz:
            if "selected_current_positions" in npz:
                selected_positions = npz["selected_current_positions"]
            else:
                selected_positions = npz["selected_positions"]

            if "target_partition_positions" in npz:
                target_positions = npz["target_partition_positions"]
            else:
                target_positions = selected_positions

            canonical_rows = npz["canonical_row_indices"]
    else:
        selected_positions = np.array([], dtype=np.int64)
        target_positions = np.array([], dtype=np.int64)
        canonical_rows = np.array([], dtype=np.int64)

    return desc, selected_positions, target_positions, canonical_rows
