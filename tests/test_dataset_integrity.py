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


def test_domain_column_isolation(tmp_path):
    """Negative unit test proving domain-defining feature inside features.parquet causes validation failure."""
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "natural" / "tableshift" / "leak_test"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feat1": [1.0, 2.0], "admission_source_id": [1, 2]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"readmitted": [0, 1]}).to_parquet(ds_dir / "labels.parquet")
    pd.DataFrame({"admission_source_id": [1, 2]}).to_parquet(ds_dir / "domains.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "leak_test",
            "domain_column": "admission_source_id",
            "target_column": "readmitted",
        }),
        encoding="utf-8",
    )

    spec = DatasetSpec(
        id="leak_test",
        display_name="Leak Test",
        dataset_group="natural_shift",
        source_provider="tableshift",
        source_type="test",
        source_url=None,
        source_id="test",
        source_version="1",
        download_mode="auto",
        license="MIT",
        citation="",
        target_column="readmitted",
        domain_column="admission_source_id",
    )

    res = validator.validate_canonical_dataset(ds_dir, spec)
    assert not res.is_valid
    assert any("CRITICAL DOMAIN LEAKAGE" in err for err in res.errors)


def test_domain_alignment(tmp_path):
    """Verify that domain rows must match feature and label rows."""
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "natural" / "alignment_test"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feat1": [1.0, 2.0, 3.0]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1, 0]}).to_parquet(ds_dir / "labels.parquet")
    pd.DataFrame({"domain": [1, 2]}).to_parquet(ds_dir / "domains.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "alignment_test",
            "domain_column": "domain",
            "target_column": "target",
        }),
        encoding="utf-8",
    )

    res = validator.validate_canonical_dataset(ds_dir)
    assert not res.is_valid
    assert any("Domain row count mismatch" in err for err in res.errors)


def test_group_column_isolation(tmp_path):
    """Negative unit test proving group-defining feature inside features.parquet causes validation failure."""
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "controlled" / "group_leak_test"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feat1": [1.0, 2.0], "subject_id": [101, 102]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1]}).to_parquet(ds_dir / "labels.parquet")
    pd.DataFrame({"subject_id": [101, 102]}).to_parquet(ds_dir / "groups.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "group_leak_test",
            "group_column": "subject_id",
            "target_column": "target",
        }),
        encoding="utf-8",
    )

    spec = DatasetSpec(
        id="group_leak_test",
        display_name="Group Leak Test",
        dataset_group="controlled_real",
        source_provider="uci",
        source_type="test",
        source_url=None,
        source_id="test",
        source_version="1",
        download_mode="auto",
        license="MIT",
        citation="",
        target_column="target",
        group_column="subject_id",
    )

    res = validator.validate_canonical_dataset(ds_dir, spec)
    assert not res.is_valid
    assert any("CRITICAL GROUP LEAKAGE" in err for err in res.errors)


def test_group_alignment(tmp_path):
    """Verify that group rows must match feature and label rows."""
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "controlled" / "group_align_test"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feat1": [1.0, 2.0, 3.0]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1, 0]}).to_parquet(ds_dir / "labels.parquet")
    pd.DataFrame({"subject_id": [1, 2]}).to_parquet(ds_dir / "groups.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "group_align_test",
            "group_column": "subject_id",
            "target_column": "target",
        }),
        encoding="utf-8",
    )

    res = validator.validate_canonical_dataset(ds_dir)
    assert not res.is_valid
    assert any("Group row count mismatch" in err for err in res.errors)


def test_har_subject_groups_preserved():
    """Verify HAR has exactly 10,299 group assignments, 30 subjects, and subject_id not in features."""
    ds_dir = Path("data/canonical/controlled/human_activity_recognition")
    assert ds_dir.exists(), "HAR canonical directory missing"
    f = pd.read_parquet(ds_dir / "features.parquet")
    l = pd.read_parquet(ds_dir / "labels.parquet")
    g = pd.read_parquet(ds_dir / "groups.parquet")
    assert len(f) == len(l) == len(g) == 10299
    assert f.shape[1] == 561
    assert "subject_id" not in f.columns
    assert "subject_id" in g.columns
    assert g["subject_id"].nunique() == 30


def test_mice_mouse_groups_preserved():
    """Verify Mice Protein Expression has exactly 1,080 group rows, 72 biological mice, mouse_subject_id not in features."""
    ds_dir = Path("data/canonical/controlled/mice_protein_expression")
    assert ds_dir.exists(), "Mice Protein canonical directory missing"
    f = pd.read_parquet(ds_dir / "features.parquet")
    l = pd.read_parquet(ds_dir / "labels.parquet")
    g = pd.read_parquet(ds_dir / "groups.parquet")
    assert len(f) == len(l) == len(g) == 1080
    assert f.shape[1] == 77
    assert "MouseID" not in f.columns
    assert "mouse_subject_id" not in f.columns
    assert "mouse_subject_id" in g.columns
    assert g["mouse_subject_id"].nunique() == 72


def test_mice_protein_no_multiple_group_identifiers():
    """Verify no biological mouse appears under multiple group identifiers."""
    ds_dir = Path("data/canonical/controlled/mice_protein_expression")
    assert ds_dir.exists(), "Mice Protein canonical directory missing"
    g = pd.read_parquet(ds_dir / "groups.parquet")
    counts = g["mouse_subject_id"].value_counts()
    assert (counts == 15).all(), f"Found inconsistent measurement counts per mouse: {counts.unique()}"
    assert len(counts) == 72


def test_feature_roles_required(tmp_path):
    """Verify that DataValidator fails when feature_roles is missing or empty."""
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "controlled" / "no_roles_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feat1": [1.0, 2.0]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1]}).to_parquet(ds_dir / "labels.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "no_roles_ds",
            "target_column": "target",
            "feature_roles": {},
        }),
        encoding="utf-8",
    )

    res = validator.validate_canonical_dataset(ds_dir)
    assert not res.is_valid
    assert any("FEATURE ROLE ERROR" in err for err in res.errors)


def test_feature_roles_cover_every_feature(tmp_path):
    """Verify that DataValidator fails if any feature is missing a declared role."""
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "controlled" / "missing_role_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1]}).to_parquet(ds_dir / "labels.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "missing_role_ds",
            "target_column": "target",
            "feature_roles": {"feat1": "numeric"},
        }),
        encoding="utf-8",
    )

    res = validator.validate_canonical_dataset(ds_dir)
    assert not res.is_valid
    assert any("FEATURE ROLE INCOMPLETENESS" in err for err in res.errors)


def test_feature_roles_reject_extra_columns(tmp_path):
    """Verify that DataValidator fails if feature_roles contains columns not in features.parquet."""
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "controlled" / "extra_role_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feat1": [1.0, 2.0]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1]}).to_parquet(ds_dir / "labels.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "extra_role_ds",
            "target_column": "target",
            "feature_roles": {"feat1": "numeric", "ghost_feat": "numeric"},
        }),
        encoding="utf-8",
    )

    res = validator.validate_canonical_dataset(ds_dir)
    assert not res.is_valid
    assert any("Extra columns in feature_roles" in err for err in res.errors)


def test_feature_roles_allowed_vocabulary(tmp_path):
    """Verify that DataValidator rejects roles not in allowed vocabulary."""
    validator = DataValidator(data_root=tmp_path)
    ds_dir = tmp_path / "canonical" / "controlled" / "invalid_role_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feat1": [1.0, 2.0]}).to_parquet(ds_dir / "features.parquet")
    pd.DataFrame({"target": [0, 1]}).to_parquet(ds_dir / "labels.parquet")
    (ds_dir / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "invalid_role_ds",
            "target_column": "target",
            "feature_roles": {"feat1": "magic_type"},
        }),
        encoding="utf-8",
    )

    res = validator.validate_canonical_dataset(ds_dir)
    assert not res.is_valid
    assert any("FEATURE ROLE VOCABULARY ERROR" in err for err in res.errors)


def test_no_global_categorical_encoding():
    """Verify TableShift Hospital Readmission categorical columns are not globally factorized into integer codes."""
    ds_dir = Path("data/canonical/natural/tableshift/tableshift_hospital_readmission")
    assert ds_dir.exists(), "TableShift Hospital Readmission directory missing"
    f = pd.read_parquet(ds_dir / "features.parquet")
    l = pd.read_parquet(ds_dir / "labels.parquet")
    d = pd.read_parquet(ds_dir / "domains.parquet")
    assert len(f) == len(l) == len(d) == 99493
    assert f.shape[1] == 46
    assert "admission_source_id" not in f.columns
    assert "admission_source_id" in d.columns
    for cat_col in ["race", "gender", "payer_code", "medical_specialty"]:
        assert not pd.api.types.is_numeric_dtype(f[cat_col]), f"Column {cat_col} was factorized into numeric dtype!"


def test_split_strategy_registry():
    """Verify all 40 registered real datasets have valid split strategy metadata."""
    from clusterdrift.data.registry import load_registry, get_dataset_spec, list_datasets
    load_registry(force_reload=True)
    for did in list_datasets():
        spec = get_dataset_spec(did)
        assert spec.split_strategy in {"kfold", "group_kfold", "temporal_block", "natural_domain"}, f"Invalid strategy for {did}"
        if did in {"human_activity_recognition", "mice_protein_expression"}:
            assert spec.split_strategy == "group_kfold"
            assert spec.group_column is not None
        elif did == "electricity":
            assert spec.split_strategy == "temporal_block"
            assert spec.time_column == "date"
        elif spec.dataset_group == "natural_shift":
            assert spec.split_strategy == "natural_domain"


