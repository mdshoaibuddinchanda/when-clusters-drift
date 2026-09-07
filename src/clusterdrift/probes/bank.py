"""Data descriptors and atomic persistence for reference and current probe banks."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

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
    canonical_row_indices_sha256: str
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
    selected_positions_sha256: str
    canonical_row_map_sha256: str
    scenario_status: str
    probe_protocol_sha256: str
    probe_bank_sha256: str


def save_reference_probe_descriptor(
    spec_path: Path,
    desc: ReferenceProbeDescriptor,
    canonical_row_indices: np.ndarray,
) -> None:
    """Atomically save reference probe descriptor JSON and companion NPZ array."""
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = spec_path.with_suffix(".npz")

    # Atomic write companion NPZ
    atomic_write_npz(npz_path, canonical_row_indices=np.ascontiguousarray(canonical_row_indices, dtype=np.int64))

    # Atomic write descriptor JSON
    doc = asdict(desc)
    atomic_write_json(spec_path, doc, indent=2, sort_keys=True)


def load_reference_probe_descriptor(
    spec_path: Path,
) -> Tuple[ReferenceProbeDescriptor, np.ndarray]:
    """Load reference probe descriptor and companion canonical row indices."""
    with open(spec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    desc = ReferenceProbeDescriptor(**data)

    npz_path = spec_path.with_suffix(".npz")
    with np.load(npz_path) as npz:
        indices = npz["canonical_row_indices"]

    return desc, indices


def save_current_probe_descriptor(
    spec_path: Path,
    desc: CurrentProbeDescriptor,
    selected_positions: np.ndarray,
    canonical_row_indices: np.ndarray,
) -> None:
    """Atomically save current probe descriptor JSON and companion NPZ arrays."""
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = spec_path.with_suffix(".npz")

    atomic_write_npz(
        npz_path,
        selected_positions=np.ascontiguousarray(selected_positions, dtype=np.int64),
        canonical_row_indices=np.ascontiguousarray(canonical_row_indices, dtype=np.int64),
    )

    doc = asdict(desc)
    atomic_write_json(spec_path, doc, indent=2, sort_keys=True)


def load_current_probe_descriptor(
    spec_path: Path,
) -> Tuple[CurrentProbeDescriptor, np.ndarray, np.ndarray]:
    """Load current probe descriptor and companion position/row mapping arrays."""
    with open(spec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    desc = CurrentProbeDescriptor(**data)

    npz_path = spec_path.with_suffix(".npz")
    with np.load(npz_path) as npz:
        positions = npz["selected_positions"]
        canonical_rows = npz["canonical_row_indices"]

    return desc, positions, canonical_rows
