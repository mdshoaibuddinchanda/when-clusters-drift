import numpy as np
import pytest
from clusterdrift.falsification.bootstrap import evaluate_falsification_verdict

def test_verdict_survives_strongly():
    # P4 strictly beats P0 and P3
    mae_p0 = 0.20
    mae_p3 = 0.18
    mae_p4 = 0.16
    # R_04 = (0.20 - 0.16)/0.20 = 0.20 >= 0.05
    # R_34 = (0.18 - 0.16)/0.18 = 0.111 >= 0.02
    dataset_deltas_04 = np.array([0.04] * 12)  # 12/12 wins, median 0.04 > 0
    dataset_deltas_34 = np.array([0.02] * 12)  # 12/12 wins
    family_deltas_04 = np.array([0.04] * 7)    # 7/7 wins
    family_deltas_34 = np.array([0.02] * 7)    # 7/7 wins

    boot_res = {
        "lodo_delta_04": {"mean_ci_lower": 0.01},
        "lodo_delta_34": {"mean_ci_lower": 0.005},
    }

    res = evaluate_falsification_verdict(
        mae_p0, mae_p3, mae_p4,
        dataset_deltas_04, dataset_deltas_34,
        family_deltas_04, family_deltas_34,
        boot_res,
    )
    assert res["verdict"] == "SURVIVES_STRONGLY"
    assert res["severe_family_generalization_failure_flag"] is False


def test_verdict_borderline():
    # Overall MAE(P4) < MAE(P0) and MAE(P4) < MAE(P3)
    mae_p0 = 0.20
    mae_p3 = 0.18
    mae_p4 = 0.179  # very small improvement, R_34 < 0.02, bootstrap CI lower <= 0
    dataset_deltas_04 = np.array([0.02] * 12)
    dataset_deltas_34 = np.array([0.001] * 12)
    family_deltas_04 = np.array([0.02] * 7)
    family_deltas_34 = np.array([0.001] * 7)

    boot_res = {
        "lodo_delta_04": {"mean_ci_lower": 0.001},
        "lodo_delta_34": {"mean_ci_lower": -0.005},  # CI lower bound <= 0
    }

    res = evaluate_falsification_verdict(
        mae_p0, mae_p3, mae_p4,
        dataset_deltas_04, dataset_deltas_34,
        family_deltas_04, family_deltas_34,
        boot_res,
    )
    assert res["verdict"] == "BORDERLINE"


def test_verdict_fails_primary():
    # P4 has worse MAE than P0 or P3
    mae_p0 = 0.20
    mae_p3 = 0.18
    mae_p4 = 0.22  # worse!
    dataset_deltas_04 = np.array([-0.02] * 12)
    dataset_deltas_34 = np.array([-0.04] * 12)
    family_deltas_04 = np.array([-0.02] * 7)
    family_deltas_34 = np.array([-0.04] * 7)

    boot_res = {
        "lodo_delta_04": {"mean_ci_lower": -0.05},
        "lodo_delta_34": {"mean_ci_lower": -0.05},
    }

    res = evaluate_falsification_verdict(
        mae_p0, mae_p3, mae_p4,
        dataset_deltas_04, dataset_deltas_34,
        family_deltas_04, family_deltas_34,
        boot_res,
    )
    assert res["verdict"] == "FAILS_PRIMARY_FALSIFICATION"