"""Phase 6 Structural Signal Engine and Conventional Validity Controls."""

from clusterdrift.signals.covariate import (
    compute_chunked_rbf_kernel_sum,
    compute_mmd_b2,
    compute_mmd_b2_torch,
    derive_source_median_bandwidth,
)
from clusterdrift.signals.divergence import (
    check_probability_simplex,
    compute_normalized_js_divergence,
)
from clusterdrift.signals.engine import SignalCache, SignalEngine
from clusterdrift.signals.entropy import compute_entropy_shift, compute_normalized_entropy
from clusterdrift.signals.hashing import (
    build_phase6_input_lock,
    compute_file_sha256,
    compute_quality_record_sha256,
    compute_signal_protocol_sha256,
    compute_signal_record_sha256,
)
from clusterdrift.signals.mass import (
    compute_cluster_mass_distribution,
    compute_cluster_mass_shift,
)
from clusterdrift.signals.membership import (
    compute_current_membership_drift,
    compute_historical_membership_drift,
    compute_pointwise_membership_drift,
)
from clusterdrift.signals.prototype import compute_prototype_movement
from clusterdrift.signals.result import QualityResult, SignalResult
from clusterdrift.signals.signature import (
    PREDICTOR_BLOCKS,
    PRIMARY_SIGNAL_NAMES,
    extract_predictor_features,
    extract_primary_signal_vector,
)
from clusterdrift.signals.structural import compute_all_structural_signals
from clusterdrift.signals.validity import (
    compute_fpc,
    compute_pe,
    compute_silhouette_control,
    compute_validity_controls,
    compute_xb_soft_m2,
)
from clusterdrift.signals.verification import verify_phase6_integrity

__all__ = [
    # Engine and cache
    "SignalEngine",
    "SignalCache",
    "SignalResult",
    "QualityResult",
    # Divergence
    "check_probability_simplex",
    "compute_normalized_js_divergence",
    # Structural signals
    "compute_normalized_entropy",
    "compute_entropy_shift",
    "compute_cluster_mass_distribution",
    "compute_cluster_mass_shift",
    "compute_pointwise_membership_drift",
    "compute_historical_membership_drift",
    "compute_current_membership_drift",
    "compute_prototype_movement",
    "compute_all_structural_signals",
    # Covariate MMD
    "derive_source_median_bandwidth",
    "compute_chunked_rbf_kernel_sum",
    "compute_mmd_b2",
    "compute_mmd_b2_torch",
    # Validity controls
    "compute_fpc",
    "compute_pe",
    "compute_xb_soft_m2",
    "compute_silhouette_control",
    "compute_validity_controls",
    # Signature and blocks
    "PRIMARY_SIGNAL_NAMES",
    "PREDICTOR_BLOCKS",
    "extract_primary_signal_vector",
    "extract_predictor_features",
    # Hashing and verification
    "compute_signal_protocol_sha256",
    "compute_signal_record_sha256",
    "compute_quality_record_sha256",
    "build_phase6_input_lock",
    "compute_file_sha256",
    "verify_phase6_integrity",
]
