"""Test that synthetic generators produce bit-for-bit identical arrays with fixed seeds."""

import numpy as np
from clusterdrift.data.registry import get_synthetic_spec, list_synthetic
from clusterdrift.data.synthetic import generate_synthetic_family


def test_synthetic_reproducibility():
    for fam in list_synthetic():
        spec = get_synthetic_spec(fam)

        # Run 1
        X1, y1, tau1, _ = generate_synthetic_family(spec)

        # Run 2
        X2, y2, tau2, _ = generate_synthetic_family(spec)

        np.testing.assert_array_equal(X1, X2, err_msg=f"Features not bit-identical across runs for {fam}")
        np.testing.assert_array_equal(y1, y2, err_msg=f"Labels not bit-identical across runs for {fam}")
        np.testing.assert_allclose(tau1, tau2, atol=1e-12, err_msg=f"Soft memberships not identical for {fam}")
