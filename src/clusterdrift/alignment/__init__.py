"""Phase 5 Cluster Alignment Package.

Provides representation alignment between clustering models across deployment conditions:
- ClusterState representations
- Reference cluster RMS radius scales
- Pairwise normalized center costs
- Soft-Jaccard membership overlap on historical probe banks
- Exact Hungarian bipartite assignment
- Ambiguity detection and non-mutation guarantees
"""

from clusterdrift.alignment.costs import (
    compute_center_cost_matrix,
    compute_combined_alignment_cost,
    compute_reference_cluster_scales,
    compute_soft_jaccard_overlap,
)
from clusterdrift.alignment.hungarian import align_clusters
from clusterdrift.alignment.result import AlignmentResult
from clusterdrift.alignment.state import ClusterState, compute_model_fingerprint
from clusterdrift.alignment.validation import (
    validate_cluster_states,
    validate_membership_matrix,
)

__all__ = [
    "ClusterState",
    "compute_model_fingerprint",
    "compute_reference_cluster_scales",
    "compute_soft_jaccard_overlap",
    "compute_center_cost_matrix",
    "compute_combined_alignment_cost",
    "align_clusters",
    "AlignmentResult",
    "validate_cluster_states",
    "validate_membership_matrix",
]
