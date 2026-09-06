"""Validation module for auditing data integrity, label isolation, and schema correctness."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from clusterdrift.data.manifest import compute_file_sha256
from clusterdrift.data.schemas import DatasetSpec, ValidationResult


class DataValidator:
    """Validates structural, numerical, and label integrity of canonical and synthetic datasets."""

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or (Path.cwd() / "data")
        self.canonical_dir = self.data_root / "canonical"
        self.synthetic_dir = self.data_root / "synthetic"
        self.manifest_dir = self.data_root / "manifests"

    def validate_canonical_dataset(
        self,
        dataset_dir: Path,
        spec: Optional[DatasetSpec] = None,
    ) -> ValidationResult:
        """Thoroughly audit a canonical dataset directory."""
        features_path = dataset_dir / "features.parquet"
        labels_path = dataset_dir / "labels.parquet"
        metadata_path = dataset_dir / "metadata.json"

        checks_passed: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        if not features_path.exists():
            return ValidationResult(
                dataset_id=dataset_dir.name,
                is_valid=False,
                rows=0,
                features=0,
                classes=None,
                missing_count=0,
                missing_fraction=0.0,
                duplicate_rows=0,
                constant_features=[],
                checks_passed=[],
                warnings=[],
                errors=["features.parquet does not exist"],
            )

        # 1. Load data
        X = pd.read_parquet(features_path)
        y = pd.read_parquet(labels_path) if labels_path.exists() else None
        domains_path = dataset_dir / "domains.parquet"
        domains = pd.read_parquet(domains_path) if domains_path.exists() else None
        groups_path = dataset_dir / "groups.parquet"
        groups = pd.read_parquet(groups_path) if groups_path.exists() else None
        meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

        n_rows, n_cols = X.shape

        # 2. Structural Checks
        if n_rows > 0:
            checks_passed.append("rows_positive")
        else:
            errors.append("Dataset has 0 rows.")

        if n_cols > 0:
            checks_passed.append("features_positive")
        else:
            errors.append("Dataset has 0 features.")

        if len(set(X.columns)) == len(X.columns):
            checks_passed.append("unique_feature_names")
        else:
            errors.append("Duplicate feature column names detected.")

        # Strict Label Isolation Check
        target_col = spec.target_column if spec else meta.get("target_column")
        if target_col and target_col in X.columns:
            errors.append(f"CRITICAL LABEL LEAKAGE: target column '{target_col}' found in features.parquet!")
        else:
            checks_passed.append("label_isolation_enforced")

        # Strict Domain Isolation Check
        domain_col = spec.domain_column if spec else meta.get("domain_column")
        if domain_col and domain_col in X.columns:
            errors.append(f"CRITICAL DOMAIN LEAKAGE: domain column '{domain_col}' found in features.parquet!")
        else:
            checks_passed.append("domain_isolation_enforced")

        # Strict Group Isolation Check
        group_col = spec.group_column if spec else meta.get("group_column")
        if group_col and group_col in X.columns:
            errors.append(f"CRITICAL GROUP LEAKAGE: group column '{group_col}' found in features.parquet!")
        else:
            checks_passed.append("group_isolation_enforced")

        # Row count alignment
        if y is not None:
            if len(X) == len(y):
                checks_passed.append("xy_length_match")
            else:
                errors.append(f"Row count mismatch: features={len(X)}, labels={len(y)}.")
            n_classes = int(y.iloc[:, 0].nunique()) if len(y) > 0 else None
        else:
            n_classes = None

        if domains is not None:
            if len(X) == len(domains):
                checks_passed.append("domain_length_match")
            else:
                errors.append(f"Domain row count mismatch: features={len(X)}, domains={len(domains)}.")

        if groups is not None:
            if len(X) == len(groups):
                checks_passed.append("group_length_match")
            else:
                errors.append(f"Group row count mismatch: features={len(X)}, groups={len(groups)}.")

        # Feature role completeness
        feature_roles = meta.get("feature_roles", {})
        if feature_roles:
            missing_roles = [col for col in X.columns if col not in feature_roles]
            if missing_roles:
                errors.append(f"FEATURE ROLE INCOMPLETENESS: Missing declared role for: {missing_roles[:5]}.")
            else:
                checks_passed.append("feature_roles_complete")

        # Forbidden global categorical factorization check
        if meta.get("dataset_id") == "tableshift_hospital_readmission":
            for cat_col in ["race", "gender", "payer_code", "medical_specialty"]:
                if cat_col in X.columns and pd.api.types.is_numeric_dtype(X[cat_col]):
                    errors.append(
                        f"FORBIDDEN ACQUISITION PREPROCESSING: Categorical column '{cat_col}' was factorized into numeric dtype!"
                    )

        # Strict Source Identity & Specification Checks
        if spec is not None:
            # 1. Source Identity check
            if spec.source_provider == "openml":
                actual_id = meta.get("openml_id")
                actual_name = meta.get("openml_name")
                if actual_id is not None and str(actual_id) != str(spec.source_id):
                    errors.append(f"SOURCE ID MISMATCH: Expected OpenML ID {spec.source_id}, but found {actual_id}!")
                if spec.expected_source_name and actual_name:
                    if actual_name.lower() != spec.expected_source_name.lower():
                        errors.append(
                            f"SOURCE IDENTITY MISMATCH: Expected OpenML dataset name '{spec.expected_source_name}', "
                            f"but found '{actual_name}' (ID={actual_id})!"
                        )
            elif spec.source_provider == "uci":
                actual_id = meta.get("uci_id")
                actual_name = meta.get("uci_name")
                if actual_id is not None and str(actual_id) != str(spec.source_id):
                    errors.append(f"SOURCE ID MISMATCH: Expected UCI ID {spec.source_id}, but found {actual_id}!")
                if spec.expected_source_name and actual_name:
                    if actual_name.lower() != spec.expected_source_name.lower():
                        errors.append(
                            f"SOURCE IDENTITY MISMATCH: Expected UCI dataset name '{spec.expected_source_name}', "
                            f"but found '{actual_name}'!"
                        )
            elif spec.source_provider == "tableshift":
                actual_task = meta.get("tableshift_task")
                if spec.expected_source_name and actual_task:
                    if actual_task.lower() != spec.expected_source_name.lower():
                        errors.append(
                            f"SOURCE IDENTITY MISMATCH: Expected TableShift task '{spec.expected_source_name}', "
                            f"but found '{actual_task}'!"
                        )

            # 2. Strict Row Count check
            if spec.expected_rows is not None:
                if n_rows != spec.expected_rows:
                    errors.append(f"ROW COUNT MISMATCH: Expected exactly {spec.expected_rows} rows, but found {n_rows}!")
                else:
                    checks_passed.append("exact_rows_matched")
            elif spec.expected_min_rows:
                if n_rows < spec.expected_min_rows:
                    errors.append(f"ROW COUNT VIOLATION: Expected at least {spec.expected_min_rows} rows, but found {n_rows}!")
                else:
                    checks_passed.append("min_rows_satisfied")

            # 3. Strict Feature Count check
            if spec.expected_features is not None:
                if n_cols != spec.expected_features:
                    errors.append(f"FEATURE COUNT MISMATCH: Expected exactly {spec.expected_features} features, but found {n_cols}!")
                else:
                    checks_passed.append("exact_features_matched")
            elif spec.expected_min_features:
                if n_cols < spec.expected_min_features:
                    errors.append(f"FEATURE COUNT VIOLATION: Expected at least {spec.expected_min_features} features, but found {n_cols}!")
                else:
                    checks_passed.append("min_features_satisfied")

            # 4. Strict Class Count check
            if spec.expected_classes is not None and n_classes is not None:
                if n_classes != spec.expected_classes:
                    errors.append(f"CLASS COUNT MISMATCH: Expected exactly {spec.expected_classes} classes, but found {n_classes}!")
                else:
                    checks_passed.append("exact_classes_matched")

            # 5. Strict Target Column Name check
            if spec.expected_target is not None and y is not None:
                if y.columns[0] != spec.expected_target:
                    errors.append(f"TARGET NAME MISMATCH: Expected target '{spec.expected_target}', but found '{y.columns[0]}'!")
                else:
                    checks_passed.append("target_name_matched")

        # 3. Numerical Audits
        missing_count = int(X.isna().sum().sum())
        missing_fraction = float(missing_count / (n_rows * n_cols)) if n_rows * n_cols > 0 else 0.0
        if missing_count > 0:
            warnings.append(f"Dataset contains {missing_count} missing values ({missing_fraction:.2%}).")
        else:
            checks_passed.append("no_missing_values")

        # Inf checks for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_count = int(np.isinf(X[numeric_cols].values).sum())
            if inf_count > 0:
                warnings.append(f"Dataset contains {inf_count} infinite values.")
            else:
                checks_passed.append("no_infinite_values")

        # Constant features
        constant_features = []
        for col in numeric_cols:
            if X[col].std() == 0 or X[col].nunique() <= 1:
                constant_features.append(str(col))
        if constant_features:
            warnings.append(f"Constant/zero-variance features detected: {constant_features}.")
        else:
            checks_passed.append("no_constant_features")

        # Duplicate rows
        duplicate_rows = int(X.duplicated().sum())
        if duplicate_rows > 0:
            warnings.append(f"{duplicate_rows} duplicate rows detected ({duplicate_rows / n_rows:.2%}).")
        else:
            checks_passed.append("no_duplicate_rows")

        is_valid = len(errors) == 0
        return ValidationResult(
            dataset_id=meta.get("dataset_id", dataset_dir.name),
            is_valid=is_valid,
            rows=n_rows,
            features=n_cols,
            classes=n_classes,
            missing_count=missing_count,
            missing_fraction=missing_fraction,
            duplicate_rows=duplicate_rows,
            constant_features=constant_features,
            checks_passed=checks_passed,
            warnings=warnings,
            errors=errors,
        )

    def generate_summary_csv(self, specs: List[DatasetSpec]) -> Path:
        """Scan all canonical and synthetic datasets and produce dataset_summary.csv."""
        records: List[Dict[str, Any]] = []

        # 1. Controlled real datasets
        for spec in specs:
            if spec.dataset_group == "controlled_real":
                tdir = self.canonical_dir / "controlled" / spec.id
                if tdir.exists() and (tdir / "features.parquet").exists():
                    res = self.validate_canonical_dataset(tdir, spec)
                    fpath = tdir / "features.parquet"
                    sha = compute_file_sha256(fpath)
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    has_group = (tdir / "groups.parquet").exists()
                    has_domain = (tdir / "domains.parquet").exists()
                    n_groups = int(pd.read_parquet(tdir / "groups.parquet").iloc[:, 0].nunique()) if has_group else 0
                    n_domains = int(pd.read_parquet(tdir / "domains.parquet").iloc[:, 0].nunique()) if has_domain else 0

                    records.append({
                        "dataset_id": spec.id,
                        "dataset_group": spec.dataset_group,
                        "provider": spec.source_provider,
                        "status": "available",
                        "rows": res.rows,
                        "features": res.features,
                        "classes": res.classes,
                        "missing_values": res.missing_count,
                        "missing_fraction": round(res.missing_fraction, 4),
                        "duplicate_rows": res.duplicate_rows,
                        "constant_features": len(res.constant_features),
                        "canonical_size_mb": round(size_mb, 4),
                        "split_strategy": spec.split_strategy,
                        "n_groups": n_groups,
                        "n_domains": n_domains,
                        "has_group_artifact": has_group,
                        "has_domain_artifact": has_domain,
                        "license": spec.license,
                        "download_mode": spec.download_mode,
                        "source_version": spec.source_version,
                        "sha256": sha,
                    })
                else:
                    records.append({
                        "dataset_id": spec.id,
                        "dataset_group": spec.dataset_group,
                        "provider": spec.source_provider,
                        "status": "unacquired",
                        "rows": 0,
                        "features": 0,
                        "classes": spec.expected_classes,
                        "missing_values": 0,
                        "missing_fraction": 0.0,
                        "duplicate_rows": 0,
                        "constant_features": 0,
                        "canonical_size_mb": 0.0,
                        "split_strategy": spec.split_strategy,
                        "n_groups": 0,
                        "n_domains": 0,
                        "has_group_artifact": False,
                        "has_domain_artifact": False,
                        "license": spec.license,
                        "download_mode": spec.download_mode,
                        "source_version": spec.source_version,
                        "sha256": "none",
                    })
            elif spec.dataset_group == "natural_shift":
                if spec.source_provider == "whyshift":
                    task = spec.domain_metadata.get("task", spec.source_id)
                    domains = spec.domain_metadata.get("domains", [])
                    task_dir = self.canonical_dir / "natural" / "whyshift" / task
                    for d in domains:
                        ddir = task_dir / d
                        if ddir.exists() and (ddir / "features.parquet").exists():
                            res = self.validate_canonical_dataset(ddir, spec)
                            fpath = ddir / "features.parquet"
                            sha = compute_file_sha256(fpath)
                            size_mb = fpath.stat().st_size / (1024 * 1024)
                            records.append({
                                "dataset_id": f"{spec.id}_{d}",
                                "dataset_group": spec.dataset_group,
                                "provider": spec.source_provider,
                                "status": "available",
                                "rows": res.rows,
                                "features": res.features,
                                "classes": res.classes,
                                "missing_values": res.missing_count,
                                "missing_fraction": round(res.missing_fraction, 4),
                                "duplicate_rows": res.duplicate_rows,
                                "constant_features": len(res.constant_features),
                                "canonical_size_mb": round(size_mb, 4),
                                "split_strategy": spec.split_strategy,
                                "n_groups": 0,
                                "n_domains": 1,
                                "has_group_artifact": False,
                                "has_domain_artifact": True,
                                "license": spec.license,
                                "download_mode": spec.download_mode,
                                "source_version": spec.source_version,
                                "sha256": sha,
                            })
                elif spec.source_provider == "tableshift":
                    task_dir = self.canonical_dir / "natural" / "tableshift" / spec.id
                    if task_dir.exists() and (task_dir / "features.parquet").exists():
                        res = self.validate_canonical_dataset(task_dir, spec)
                        fpath = task_dir / "features.parquet"
                        sha = compute_file_sha256(fpath)
                        size_mb = fpath.stat().st_size / (1024 * 1024)
                        has_group = (task_dir / "groups.parquet").exists()
                        has_domain = (task_dir / "domains.parquet").exists()
                        n_groups = int(pd.read_parquet(task_dir / "groups.parquet").iloc[:, 0].nunique()) if has_group else 0
                        n_domains = int(pd.read_parquet(task_dir / "domains.parquet").iloc[:, 0].nunique()) if has_domain else 0

                        records.append({
                            "dataset_id": spec.id,
                            "dataset_group": spec.dataset_group,
                            "provider": spec.source_provider,
                            "status": "available",
                            "rows": res.rows,
                            "features": res.features,
                            "classes": res.classes,
                            "missing_values": res.missing_count,
                            "missing_fraction": round(res.missing_fraction, 4),
                            "duplicate_rows": res.duplicate_rows,
                            "constant_features": len(res.constant_features),
                            "canonical_size_mb": round(size_mb, 4),
                            "split_strategy": spec.split_strategy,
                            "n_groups": n_groups,
                            "n_domains": n_domains,
                            "has_group_artifact": has_group,
                            "has_domain_artifact": has_domain,
                            "license": spec.license,
                            "download_mode": spec.download_mode,
                            "source_version": spec.source_version,
                            "sha256": sha,
                        })

        # 2. Synthetic datasets
        if self.synthetic_dir.exists():
            for sdir in sorted(self.synthetic_dir.iterdir()):
                if sdir.is_dir() and (sdir / "features.parquet").exists():
                    fpath = sdir / "features.parquet"
                    X = pd.read_parquet(fpath)
                    sha = compute_file_sha256(fpath)
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    records.append({
                        "dataset_id": sdir.name,
                        "dataset_group": "synthetic",
                        "provider": "clusterdrift_synthetic",
                        "status": "available",
                        "rows": len(X),
                        "features": X.shape[1],
                        "classes": None,
                        "missing_values": 0,
                        "missing_fraction": 0.0,
                        "duplicate_rows": int(X.duplicated().sum()),
                        "constant_features": 0,
                        "canonical_size_mb": round(size_mb, 4),
                        "split_strategy": "kfold",
                        "n_groups": 0,
                        "n_domains": 0,
                        "has_group_artifact": False,
                        "has_domain_artifact": False,
                        "license": "Apache-2.0",
                        "download_mode": "generated",
                        "source_version": "1.0",
                        "sha256": sha,
                    })

        df_summary = pd.DataFrame(records)
        out_csv = self.manifest_dir / "dataset_summary.csv"
        df_summary.to_csv(out_csv, index=False)
        return out_csv
