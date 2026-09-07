import numpy as np
import pandas as pd
import pytest

def test_dataset_unit_aggregation():
    # 12 datasets, multiple rows per dataset
    datasets = [f"dataset_{i}" for i in range(12)]
    records = []
    for d in datasets:
        for _ in range(50):
            records.append({"dataset_id": d, "err_p0": 0.20, "err_p4": 0.15})
    df = pd.DataFrame(records)

    # Compute MAE per dataset
    ds_maes = []
    for d in datasets:
        sub = df[df["dataset_id"] == d]
        ds_maes.append({
            "dataset_id": d,
            "mae_p0": sub["err_p0"].mean(),
            "mae_p4": sub["err_p4"].mean(),
            "delta_04": sub["err_p0"].mean() - sub["err_p4"].mean(),
        })
    agg_df = pd.DataFrame(ds_maes)

    assert len(agg_df) == 12
    np.testing.assert_allclose(agg_df["delta_04"], 0.05)