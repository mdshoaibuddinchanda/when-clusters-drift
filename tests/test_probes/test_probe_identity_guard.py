"""Test core invariant enforcement via ProbeEvaluation and verify_probe_identity."""

import numpy as np
import pytest

from clusterdrift.probes.evaluation import (
    ProbeEvaluation,
    ProbeIdentityMismatchError,
    verify_probe_identity,
)


def test_probe_identity_guard_matching_succeeds():
    """Identical bank_sha256 succeeds without error."""
    eval_a = ProbeEvaluation(
        bank_sha256="same_bank_hash",
        bank_type="reference",
        dataset_id="iris",
        outer_fold=0,
        condition="reference",
        model_fingerprint="fcm_model",
        memberships=np.array([[0.7, 0.3]]),
        n_samples=1,
        n_clusters=2,
    )
    eval_b = ProbeEvaluation(
        bank_sha256="same_bank_hash",
        bank_type="reference",
        dataset_id="iris",
        outer_fold=0,
        condition="reference",
        model_fingerprint="gmm_model",
        memberships=np.array([[0.6, 0.4]]),
        n_samples=1,
        n_clusters=2,
    )

    # Must not raise
    verify_probe_identity(eval_a, eval_b)


def test_probe_identity_guard_mismatch_raises():
    """Different bank_sha256 raises ProbeIdentityMismatchError explicitly."""
    eval_a = ProbeEvaluation(
        bank_sha256="bank_hash_alpha",
        bank_type="reference",
        dataset_id="iris",
        outer_fold=0,
        condition="reference",
        model_fingerprint="fcm_model",
        memberships=np.array([[0.7, 0.3]]),
        n_samples=1,
        n_clusters=2,
    )
    eval_b = ProbeEvaluation(
        bank_sha256="bank_hash_beta",
        bank_type="current",
        dataset_id="iris",
        outer_fold=0,
        condition="location_mild",
        model_fingerprint="gmm_model",
        memberships=np.array([[0.6, 0.4]]),
        n_samples=1,
        n_clusters=2,
    )

    with pytest.raises(ProbeIdentityMismatchError, match="Core Invariant Violation"):
        verify_probe_identity(eval_a, eval_b)
