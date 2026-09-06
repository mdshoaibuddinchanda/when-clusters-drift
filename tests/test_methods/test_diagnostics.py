"""Unit tests for fuzzy partition degeneracy diagnostics."""

import numpy as np
import pytest

from clusterdrift.methods.diagnostics import assess_fuzzy_partition_degeneracy


def test_diagnostics_detects_complete_collapse():
    """Verify that an artificial uniform partition with coincident centers is flagged as DEGENERATE."""
    # Data centered at [0, 0]
    X = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ])
    K = 3
    # Coincident prototypes at global mean [0, 0]
    centers = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    # Exact uniform membership 1/K = 1/3
    U = np.full((len(X), K), 1.0 / K)

    diag = assess_fuzzy_partition_degeneracy(X, centers, U, K)

    assert diag["is_degenerate"] is True
    assert diag["fpc"] == pytest.approx(1.0 / K, abs=1e-5)
    assert diag["fpc_floor_gap"] == pytest.approx(0.0, abs=1e-5)
    assert diag["pe"] == pytest.approx(np.log(K), abs=1e-5)
    assert diag["min_center_distance"] == pytest.approx(0.0, abs=1e-6)
    assert diag["normalized_min_center_distance"] == pytest.approx(0.0, abs=1e-6)
    assert diag["effective_distinct_prototypes"] == 1


def test_diagnostics_clears_well_separated_clusters():
    """Verify that well-separated clusters are correctly flagged as NOT degenerate."""
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ])
    K = 2
    centers = np.array([[0.05, 0.05], [10.05, 10.05]])
    U = np.array([
        [0.98, 0.02],
        [0.97, 0.03],
        [0.02, 0.98],
        [0.03, 0.97],
    ])

    diag = assess_fuzzy_partition_degeneracy(X, centers, U, K)

    assert diag["is_degenerate"] is False
    assert diag["fpc_floor_gap"] > 0.3
    assert diag["normalized_min_center_distance"] > 0.5
    assert diag["effective_distinct_prototypes"] == 2


def test_diagnostics_does_not_flag_low_fpc_when_centers_separated():
    """Verify that diffuse overlapping clusters with separated centers are NOT marked degenerate."""
    # Data with separated centers but moderate fuzziness
    X = np.array([
        [-2.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    K = 2
    centers = np.array([[-1.5, 0.0], [1.5, 0.0]])
    # Overlapping fuzzy memberships, FPC ~ 0.53 (close to 0.5 floor)
    U = np.array([
        [0.60, 0.40],
        [0.55, 0.45],
        [0.45, 0.55],
        [0.40, 0.60],
    ])

    diag = assess_fuzzy_partition_degeneracy(X, centers, U, K, tol_fpc=0.02, tol_dist=0.02)

    # Even if FPC is low, centers are separated (normalized_dist >> 0.02), so NOT degenerate
    assert diag["is_degenerate"] is False
    assert diag["effective_distinct_prototypes"] == 2
