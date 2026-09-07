"""Phase 4 Controlled Distribution-Shift Engine."""

from clusterdrift.shifts.base import (
    ALL_CONDITIONS,
    SEVERITIES,
    SHIFT_FAMILIES,
    ShiftResult,
    SourceStatistics,
    compute_source_statistics,
)
from clusterdrift.shifts.backend import (
    apply_additive_noise,
    apply_affine_transform,
    apply_convex_interpolation,
    detect_hardware,
    should_use_gpu,
)
from clusterdrift.shifts.engine import ShiftEngine
from clusterdrift.shifts.hashing import (
    compute_bytes_sha256,
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
from clusterdrift.shifts.validation import (
    audit_preprocessing_transformation,
    verify_severity_monotonicity,
)

__all__ = [
    "ALL_CONDITIONS",
    "SEVERITIES",
    "SHIFT_FAMILIES",
    "ShiftResult",
    "SourceStatistics",
    "compute_source_statistics",
    "ShiftEngine",
    "apply_location_shift",
    "apply_scale_shift",
    "apply_measurement_noise",
    "apply_outlier_shift",
    "apply_mcar_shift",
    "apply_class_prevalence_shift",
    "apply_local_overlap_shift",
    "audit_preprocessing_transformation",
    "verify_severity_monotonicity",
    "detect_hardware",
    "should_use_gpu",
    "compute_bytes_sha256",
    "compute_file_sha256",
    "derive_integer_seed",
    "compute_shift_protocol_sha256",
    "compute_scenario_input_sha256",
    "compute_shift_spec_sha256",
]
