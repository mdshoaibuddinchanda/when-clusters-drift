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


def test_validator_rejects_source_name_mismatch(tmp_path):
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "controlled" / "mice_protein_expression"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"p1": [1.0, 2.0]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"class": [0, 1]}).to_parquet(ds_dir / "labels.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "mice_protein_expression",
            "openml_id": 40996,
            "openml_name": "Fashion-MNIST",
            "target_column": "class",
        }),
        encoding="utf-8",
    )

    spec = DatasetSpec(
        id="mice_protein_expression",
        display_name="Mice Protein",
        dataset_group="controlled_real",
        source_provider="openml",
        source_type="api",
        source_url="https://www.openml.org/d/40966",
        source_id="40966",
        source_version="1",
        download_mode="auto",
        license="CC BY 4.0",
        citation="",
        target_column="class",
        expected_source_name="MiceProtein",
        expected_rows=1080,
        expected_features=77,
        expected_classes=8,
    )

    res = validator.validate_canonical_dataset(ds_dir, spec)
    assert not res.is_valid
    assert any("SOURCE IDENTITY MISMATCH" in err for err in res.errors)


def test_validator_rejects_shape_and_class_mismatches(tmp_path):
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "controlled" / "shape_test"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # 10 rows, 5 features, 2 classes
    pd.DataFrame({f"f{i}": range(10) for i in range(5)}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"target": [0] * 5 + [1] * 5}).to_parquet(ds_dir / "labels.parquet")
    (ds_dir / "metadata.json").write_text(json.dumps({"dataset_id": "shape_test"}), encoding="utf-8")

    # Spec expects 100 rows, 20 features, 5 classes
    spec = DatasetSpec(
        id="shape_test",
        display_name="Shape Test",
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
        expected_rows=100,
        expected_features=20,
        expected_classes=5,
    )

    res = validator.validate_canonical_dataset(ds_dir, spec)
    assert not res.is_valid
    assert any("ROW COUNT MISMATCH" in err for err in res.errors)
    assert any("FEATURE COUNT MISMATCH" in err for err in res.errors)
    assert any("CLASS COUNT MISMATCH" in err for err in res.errors)


def test_all_canonical_datasets_pass_strict_validation():
    from clusterdrift.data.registry import load_registry, get_dataset_spec, list_controlled_real
    load_registry(force_reload=True)
    validator = DataValidator()

    for did in list_controlled_real():
        spec = get_dataset_spec(did)
        target_dir = Path("data/canonical/controlled") / did
        assert target_dir.exists(), f"Missing canonical dataset directory: {did}"
        res = validator.validate_canonical_dataset(target_dir, spec)
        assert res.is_valid, f"Dataset '{did}' failed strict validation with errors: {res.errors}"

