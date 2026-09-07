import numpy as np
import pandas as pd
import pytest

from clusterdrift.falsification.protocol import load_falsification_config
from pathlib import Path

def test_lodo_split_disjointness():
    root = Path(__file__).resolve().parents[2]
    cfg = load_falsification_config(root / "configs" / "falsification.yaml")
    datasets = cfg["datasets"]

    # Mock joined dataframe
    rows = []
    for ds in datasets:
        for fold in range(5):
            for cond in ["location_mild", "scale_mild"]:
                rows.append({"dataset_id": ds, "outer_fold": fold, "condition": cond, "delta_ari": 0.1})
    df = pd.DataFrame(rows)

    tested_datasets = []
    for test_ds in datasets:
        tr_mask = (df["dataset_id"] != test_ds)
        te_mask = (df["dataset_id"] == test_ds)

        tr_data = df[tr_mask]
        te_data = df[te_mask]

        # Ensure no intersection of dataset_id
        assert test_ds not in tr_data["dataset_id"].values
        assert (te_data["dataset_id"] == test_ds).all()
        assert len(set(tr_data.index).intersection(set(te_data.index))) == 0
        tested_datasets.append(test_ds)

    assert sorted(tested_datasets) == sorted(datasets)