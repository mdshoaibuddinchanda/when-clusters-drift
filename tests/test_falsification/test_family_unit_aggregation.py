import numpy as np
import pandas as pd
import pytest
from clusterdrift.falsification.protocol import ALL_SHIFT_FAMILIES

def test_family_unit_aggregation():
    records = []
    for fam in ALL_SHIFT_FAMILIES:
        for _ in range(40):
            records.append({"shift_family": fam, "err_p0": 0.18, "err_p4": 0.12})
    df = pd.DataFrame(records)

    fam_maes = []
    for fam in ALL_SHIFT_FAMILIES:
        sub = df[df["shift_family"] == fam]
        fam_maes.append({
            "shift_family": fam,
            "mae_p0": sub["err_p0"].mean(),
            "mae_p4": sub["err_p4"].mean(),
            "delta_04": sub["err_p0"].mean() - sub["err_p4"].mean(),
        })
    agg_df = pd.DataFrame(fam_maes)

    assert len(agg_df) == 7
    np.testing.assert_allclose(agg_df["delta_04"], 0.06)