"""Test that synthetic true posterior memberships satisfy probability simplex constraints."""

import numpy as np
from clusterdrift.data.registry import get_synthetic_spec, list_synthetic
from clusterdrift.data.synthetic import generate_synthetic_family


def test_synthetic_soft_truth_simplex():
    for fam in list_synthetic():
        spec = get_synthetic_spec(fam)
        X, y, tau, params = generate_synthetic_family(spec)

        # 1. Shape check
        assert tau.shape == (spec.n, spec.K), f"Tau shape {tau.shape} != ({spec.n}, {spec.K}) in {fam}"

        # 2. Non-negativity check: tau_ik >= 0
        assert (tau >= 0.0).all(), f"Negative membership probability in {fam}"

        # 3. Partition of unity: sum_k tau_ik == 1.0
        row_sums = tau.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row sums diverge from 1.0 in {fam}"

        # 4. No NaNs or infs
        assert not np.isnan(tau).any(), f"NaN in tau for {fam}"
        assert not np.isinf(tau).any(), f"Inf in tau for {fam}"
