"""Phase 5 Dual Probe Banks Package.

Provides representation probes for model comparisons across deployment conditions:
- Reference probe bank A^R (outer source historical points)
- Current probe bank A_t^C (observed shifted target points)
- Core invariant enforcement: verify_probe_identity
- Preprocessor caching and matrix reconstruction
- Strict label-leakage boundary
"""

from clusterdrift.probes.bank import (
    CurrentProbeDescriptor,
    ReferenceProbeDescriptor,
    load_current_probe_descriptor,
    load_reference_probe_descriptor,
    save_current_probe_descriptor,
    save_reference_probe_descriptor,
)
from clusterdrift.probes.cache import ProbeCache
from clusterdrift.probes.evaluation import (
    ProbeEvaluation,
    ProbeIdentityMismatchError,
    verify_probe_identity,
)
from clusterdrift.probes.hashing import (
    build_phase5_input_lock,
    compute_alignment_protocol_sha256,
    compute_array_sha256,
    compute_current_bank_sha256,
    compute_file_sha256,
    compute_probe_protocol_sha256,
    compute_reference_bank_sha256,
    derive_current_probe_seed,
    derive_integer_seed,
    derive_reference_probe_seed,
)
from clusterdrift.probes.matrix import (
    load_current_probe_matrix,
    load_raw_source_features,
    load_reference_probe_matrix,
)
from clusterdrift.probes.selection import (
    select_current_probe_positions,
    select_reference_probe_indices,
    select_reference_probe_positions,
)
from clusterdrift.probes.validation import (
    validate_saved_current_descriptor,
    validate_saved_reference_descriptor,
)

__all__ = [
    "ReferenceProbeDescriptor",
    "CurrentProbeDescriptor",
    "save_reference_probe_descriptor",
    "load_reference_probe_descriptor",
    "save_current_probe_descriptor",
    "load_current_probe_descriptor",
    "ProbeCache",
    "ProbeEvaluation",
    "ProbeIdentityMismatchError",
    "verify_probe_identity",
    "derive_integer_seed",
    "derive_reference_probe_seed",
    "derive_current_probe_seed",
    "compute_probe_protocol_sha256",
    "compute_alignment_protocol_sha256",
    "compute_reference_bank_sha256",
    "compute_current_bank_sha256",
    "build_phase5_input_lock",
    "compute_file_sha256",
    "load_raw_source_features",
    "load_reference_probe_matrix",
    "load_current_probe_matrix",
    "select_reference_probe_indices",
    "select_current_probe_positions",
    "validate_saved_reference_descriptor",
    "validate_saved_current_descriptor",
]
