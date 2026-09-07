"""Immutable Result Containers for Phase 6 Signals and Quality Targets.

Ensures strict decoupling:
    - SignalResult: strictly label-free structural signals and conventional validity controls.
    - QualityResult: evaluation-only clustering quality degradations under deployed model.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List


@dataclass(frozen=True)
class SignalResult:
    """Immutable record of label-free signals and conventional validity controls."""
    dataset_id: str
    outer_fold: int
    condition: str
    method: str
    seed: int
    K: int

    source_model_status: str
    candidate_model_status: str
    source_converged: bool
    candidate_converged: bool
    source_degenerate: bool
    candidate_degenerate: bool

    usable: bool

    reference_bank_sha256: str
    current_bank_sha256: str
    shift_spec_sha256: str

    source_model_fingerprint: str
    candidate_model_fingerprint: str

    alignment_permutation: List[int]
    alignment_best_cost: float
    alignment_second_best_cost: float
    alignment_global_margin: float
    alignment_ambiguous: bool

    D_U_R: float
    D_U_R_median: float
    D_U_R_p95: float

    D_U_C: float
    D_U_C_median: float
    D_U_C_p95: float

    D_V: float
    D_V_median: float
    D_V_max: float

    entropy_source_current: float
    entropy_candidate_current: float
    entropy_signed_change: float
    D_H: float

    mass_source_current: List[float]
    mass_candidate_current: List[float]
    D_M: float

    D_X: float
    mmd_sigma: float
    mmd_bandwidth_status: str

    FPC: float
    PE: float
    PE_norm: float
    XB_soft_m2: float
    XB_status: str
    silhouette: float
    silhouette_status: str

    runtime_source_fit: float
    runtime_candidate_fit: float
    runtime_alignment: float
    runtime_signals: float
    runtime_mmd: float
    runtime_total: float

    signal_protocol_sha256: str
    signal_record_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with JSON-serialized list fields for tabular persistence."""
        d = asdict(self)
        d["alignment_permutation"] = json.dumps(self.alignment_permutation)
        d["mass_source_current"] = json.dumps(self.mass_source_current)
        d["mass_candidate_current"] = json.dumps(self.mass_candidate_current)
        return d


@dataclass(frozen=True)
class QualityResult:
    """Immutable record of evaluation-only clustering quality targets under deployed source model."""
    dataset_id: str
    outer_fold: int
    condition: str
    method: str
    seed: int

    quality_target: str
    ari_clean: float
    ari_condition: float
    delta_ari: float

    nmi_condition: float
    ami_condition: float

    n_evaluation_rows: int
    quality_record_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for tabular persistence."""
        return asdict(self)
