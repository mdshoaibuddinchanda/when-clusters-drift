import pandas as pd
import pytest
from clusterdrift.falsification.protocol import ALL_SHIFT_FAMILIES, CONDITION_TO_FAMILY

def test_losfo_split_disjointness():
    rows = []
    for cond, (fam, sev) in CONDITION_TO_FAMILY.items():
        if cond == "clean":
            continue
        rows.append({"condition": cond, "shift_family": fam, "severity": sev, "delta_ari": 0.05})
    df = pd.DataFrame(rows)

    tested_families = []
    for test_fam in ALL_SHIFT_FAMILIES:
        tr_mask = (df["shift_family"] != test_fam)
        te_mask = (df["shift_family"] == test_fam)

        tr_data = df[tr_mask]
        te_data = df[te_mask]

        assert test_fam not in tr_data["shift_family"].values
        assert (te_data["shift_family"] == test_fam).all()
        assert len(te_data) == 2  # mild and severe
        tested_families.append(test_fam)

    assert sorted(tested_families) == sorted(ALL_SHIFT_FAMILIES)