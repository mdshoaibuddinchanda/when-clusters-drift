"""Unit tests for reference cluster RMS radius scale computation and flooring."""

import numpy as np
import pytest

from clusterdrift.alignment.costs import compute_reference_cluster_scales


def test_reference_cluster_scale_hand_computable():
    """Verify weighted RMS radius against analytical hand computation."""
    # 3 points on 1D line: x = [0, 2, 4]
    X = np.array([[0.0], [2.0], [4.0]])
    # Prototype center at v_0 = 2.0
    centers = np.array([[2.0]])
    # Uniform weights
    U = np.array([[1.0], [1.0], [1.0]])

    # distances to v_0: (0-2)^2=4, (2-2)^2=0, (4-2)^2=4.
    # sum = 8. weighted var = 8 / 3. RMS radius = sqrt(8/3) = 1.632993...
    expected_scale = np.sqrt(8.0 / 3.0)

    scales, floored, s_global = compute_reference_cluster_scales(X, U, centers, epsilon=0.0)
    assert not floored[0]
    np.testing.assert_allclose(scales[0], expected_scale, atol=1e-10)


def test_reference_cluster_scale_numerical_flooring():
    """Verify that a degenerate cluster with 0 raw variance receives numerical floor rho * s_global."""
    # Cluster 0 is degenerate (all points at (10, 10)), Cluster 1 has spread
    X = np.array([
        [10.0, 10.0],
        [10.0, 10.0],
        [0.0, 0.0],
        [4.0, 4.0],
    ])
    centers = np.array([
        [10.0, 10.0],
        [2.0, 2.0],
    ])
    U = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    scales, floored, s_global = compute_reference_cluster_scales(
        X, U, centers, scale_floor_ratio=1e-4, epsilon=1e-12
    )
    # Cluster 0 had 0 raw distance to its center -> must be floored!
    assert floored[0]
    assert scales[0] == pytest.approx(1e-4 * s_global, rel=1e-5)
    # Cluster 1 has positive spread -> not floored
    assert not floored[1]
    assert scales[1] > 1e-4 * s_global
