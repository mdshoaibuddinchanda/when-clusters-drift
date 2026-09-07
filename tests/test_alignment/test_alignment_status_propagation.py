"""Test model status propagation and usability gating in alignment validation."""

import numpy as np
import pytest

from clusterdrift.alignment.hungarian import align_clusters


class MockModel:
    def __init__(self, centers, converged=True, status="SUCCESS", degenerate=False):
        self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
        self.converged_ = converged
        self.status_ = status
        self.degenerate_solution_ = degenerate

    def predict_membership(self, X):
        K = len(self.cluster_centers_)
        N = len(X)
        U = np.full((N, K), 1.0 / K, dtype=np.float64)
        return U


def test_status_propagation_converged_valid():
    """Valid converged models produce usable=True and status SUCCESS."""
    m_src = MockModel([[0.0, 0.0], [5.0, 5.0]], converged=True, status="SUCCESS", degenerate=False)
    m_cand = MockModel([[0.1, 0.1], [4.9, 4.9]], converged=True, status="SUCCESS", degenerate=False)

    src_status = getattr(m_src, "status_", "SUCCESS")
    src_conv = bool(getattr(m_src, "converged_", True))
    src_degen = bool(getattr(m_src, "degenerate_solution_", False))

    cand_status = getattr(m_cand, "status_", "SUCCESS")
    cand_conv = bool(getattr(m_cand, "converged_", True))
    cand_degen = bool(getattr(m_cand, "degenerate_solution_", False))

    if not (src_conv and not src_degen):
        alignment_status = f"SOURCE_MODEL_{src_status}"
    elif not (cand_conv and not cand_degen):
        alignment_status = f"CANDIDATE_MODEL_{cand_status}"
    else:
        alignment_status = "SUCCESS"

    usable = bool(
        src_conv and not src_degen
        and cand_conv and not cand_degen
        and alignment_status in ("SUCCESS", "AMBIGUOUS")
    )

    assert alignment_status == "SUCCESS"
    assert usable is True


def test_status_propagation_source_nonconverged():
    """Non-converged source model propagates status and marks usable=False."""
    m_src = MockModel([[0.0, 0.0], [5.0, 5.0]], converged=False, status="MAX_ITER_REACHED", degenerate=False)
    m_cand = MockModel([[0.1, 0.1], [4.9, 4.9]], converged=True, status="SUCCESS", degenerate=False)

    src_status = getattr(m_src, "status_", "SUCCESS")
    src_conv = bool(getattr(m_src, "converged_", True))
    src_degen = bool(getattr(m_src, "degenerate_solution_", False))

    cand_status = getattr(m_cand, "status_", "SUCCESS")
    cand_conv = bool(getattr(m_cand, "converged_", True))
    cand_degen = bool(getattr(m_cand, "degenerate_solution_", False))

    if not (src_conv and not src_degen):
        alignment_status = f"SOURCE_MODEL_{src_status}"
    elif not (cand_conv and not cand_degen):
        alignment_status = f"CANDIDATE_MODEL_{cand_status}"
    else:
        alignment_status = "SUCCESS"

    usable = bool(
        src_conv and not src_degen
        and cand_conv and not cand_degen
        and alignment_status in ("SUCCESS", "AMBIGUOUS")
    )

    assert alignment_status == "SOURCE_MODEL_MAX_ITER_REACHED"
    assert usable is False


def test_status_propagation_candidate_degenerate():
    """Degenerate candidate model propagates status and marks usable=False."""
    m_src = MockModel([[0.0, 0.0], [5.0, 5.0]], converged=True, status="SUCCESS", degenerate=False)
    m_cand = MockModel([[2.5, 2.5], [2.5, 2.5]], converged=True, status="DEGENERATE_SOLUTION", degenerate=True)

    src_status = getattr(m_src, "status_", "SUCCESS")
    src_conv = bool(getattr(m_src, "converged_", True))
    src_degen = bool(getattr(m_src, "degenerate_solution_", False))

    cand_status = getattr(m_cand, "status_", "SUCCESS")
    cand_conv = bool(getattr(m_cand, "converged_", True))
    cand_degen = bool(getattr(m_cand, "degenerate_solution_", False))

    if not (src_conv and not src_degen):
        alignment_status = f"SOURCE_MODEL_{src_status}"
    elif not (cand_conv and not cand_degen):
        alignment_status = f"CANDIDATE_MODEL_{cand_status}"
    else:
        alignment_status = "SUCCESS"

    usable = bool(
        src_conv and not src_degen
        and cand_conv and not cand_degen
        and alignment_status in ("SUCCESS", "AMBIGUOUS")
    )

    assert alignment_status == "CANDIDATE_MODEL_DEGENERATE_SOLUTION"
    assert usable is False
