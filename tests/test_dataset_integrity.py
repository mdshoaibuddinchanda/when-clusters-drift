"""Test dataset integrity checks including label leakage, empty data, and column duplication detection."""

import json
from pathlib import Path
import pandas as pd
import pytest

from clusterdrift.data.schemas import DatasetSpec
from clusterdrift.data.validation import DataValidator


def test_validator_detects_empty_dataset(tmp_path):
    validator = DataValidator(data_root=tmp_path)
    empty_dir = tmp_path / "canonical" / "controlled" / "empty_ds"
    empty_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame().to_parquet(empty_dir / "features.parquet")
    pd.DataFrame().to_parquet(empty_dir / "labels.parquet")
    (empty_dir / "metadata.json").write_text(json.dumps({"dataset_id": "empty_ds"}), encoding="utf-8")

    res = validator.validate_canonical_dataset(empty_dir)
    assert not res.is_valid
    assert any("0 rows" in err for err in res.errors)


def test_validator_detects_xy_mismatch(tmp_path):
    validator = DataValidator(data_root=tmp_path)
    mismatch_dir = tmp_path / "canonical" / "controlled" / "mismatch_ds"
    mismatch_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"f1": [1, 2, 3, 4]}).to_parquet(mismatch_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1]}).to_parquet(mismatch_dir / "labels.parquet")
    (mismatch_dir / "metadata.json").write_text(json.dumps({"dataset_id": "mismatch_ds"}), encoding="utf-8")

    res = validator.validate_canonical_dataset(mismatch_dir)
    assert not res.is_valid
    assert any("Row count mismatch" in err for err in res.errors)


def test_validator_detects_label_leakage(tmp_path):
    validator = DataValidator(data_root=tmp_path)
    leakage_dir = tmp_path / "canonical" / "controlled" / "leak_ds"
    leakage_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"f1": [1, 2], "target": [0, 1]}).to_parquet(leakage_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1]}).to_parquet(leakage_dir / "labels.parquet")
    (leakage_dir / "metadata.json").write_text(
        json.dumps({"dataset_id": "leak_ds", "target_column": "target"}),
        encoding="utf-8",
    )

    spec = DatasetSpec(
        id="leak_ds",
        display_name="Leakage Test",
        dataset_group="controlled_real",
        source_provider="test",
        source_type="test",
        source_url=None,
        source_id="test",
        source_version="1",
        download_mode="auto",
        license="MIT",
        citation="",
        target_column="target",
    )

    res = validator.validate_canonical_dataset(leakage_dir, spec)
    assert not res.is_valid
    assert any("LABEL LEAKAGE" in err for err in res.errors)
