import numpy as np
import pytest
from clusterdrift.falsification.bootstrap import compute_paired_unit_bootstrap

def test_paired_bootstrap():
    deltas = np.array([0.01, 0.03, 0.02, 0.05, 0.04, 0.02, 0.06, 0.01, 0.03, 0.02, 0.04, 0.05])
    res1 = compute_paired_unit_bootstrap(deltas, n_repetitions=1000, seed=2026090707)
    res2 = compute_paired_unit_bootstrap(deltas, n_repetitions=1000, seed=2026090707)

    assert res1["sample_mean"] == pytest.approx(float(np.mean(deltas)))
    assert res1["sample_median"] == pytest.approx(float(np.median(deltas)))
    assert res1["mean_ci_lower"] == res2["mean_ci_lower"]
    assert res1["mean_ci_upper"] == res2["mean_ci_upper"]
    assert res1["mean_ci_lower"] < res1["sample_mean"] < res1["mean_ci_upper"]