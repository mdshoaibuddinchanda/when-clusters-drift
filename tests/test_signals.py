"""Integration smoke tests for Phase 6 Structural Signal Engine."""

import numpy as np
import pytest

from clusterdrift.signals.divergence import compute_normalized_js_divergence
from clusterdrift.signals.entropy import compute_entropy_shift
from clusterdrift.signals.mass import compute_cluster_mass_shift
from clusterdrift.signals.membership import (
    compute_current_membership_drift,
    compute_historical_membership_drift,
)
from clusterdrift.signals.prototype import compute_prototype_movement
from clusterdrift.signals.signature import (
    PREDICTOR_BLOCKS,
    PRIMARY_SIGNAL_NAMES,
    extract_predictor_features,
    extract_primary_signal_vector,
)


def test_signal_engine_exports_and_signature():
    """Verify primary signal vector structure and predictor blocks."""
    assert len(PRIMARY_SIGNAL_NAMES) == 6
    assert PRIMARY_SIGNAL_NAMES == ["D_U_R", "D_U_C", "D_V", "D_H", "D_M", "D_X"]

    sample_record = {
        "D_U_R": 0.1,
        "D_U_C": 0.2,
        "D_V": 0.3,
        "D_H": 0.4,
        "D_M": 0.5,
        "D_X": 0.6,
        "FPC": 0.8,
        "PE_norm": 0.2,
        "XB_soft_m2": 0.5,
        "silhouette": 0.7,
    }

    z_t = extract_primary_signal_vector(sample_record)
    assert len(z_t) == 6
    assert np.allclose(z_t, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    p0 = extract_predictor_features(sample_record, "P0")
    assert len(p0) == 1

    p1 = extract_predictor_features(sample_record, "P1")
    assert len(p1) == 4

    p2 = extract_predictor_features(sample_record, "P2")
    assert len(p2) == 5

    p5 = extract_predictor_features(sample_record, "P5")
    assert len(p5) == 10
