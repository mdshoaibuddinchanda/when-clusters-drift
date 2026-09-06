"""Test that canonical datasets strictly separate features and labels and preserve metadata schema."""

import json
from pathlib import Path
import pandas as pd
from clusterdrift.data.downloader import DatasetDownloader
from clusterdrift.data.registry import get_dataset_spec


def test_label_isolation_and_schema(tmp_path):
    downloader = DatasetDownloader(data_root=tmp_path, force=False)
    spec = get_dataset_spec("wine")

    res = downloader.download(spec)
    assert res.status.value in ["ok", "already_exists"]

    target_dir = tmp_path / "canonical" / "controlled" / "wine"
    features_file = target_dir / "features.parquet"
    labels_file = target_dir / "labels.parquet"
    meta_file = target_dir / "metadata.json"

    assert features_file.exists()
    assert labels_file.exists()
    assert meta_file.exists()

    df_X = pd.read_parquet(features_file)
    df_y = pd.read_parquet(labels_file)

    assert spec.target_column not in df_X.columns, f"Target '{spec.target_column}' leaked into features.parquet!"
    assert len(df_X) == len(df_y), "Row count mismatch between features and labels"

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["dataset_id"] == "wine"
    assert meta["n_rows"] == len(df_X)
    assert meta["n_features"] == df_X.shape[1]
    assert "target_column" in meta
