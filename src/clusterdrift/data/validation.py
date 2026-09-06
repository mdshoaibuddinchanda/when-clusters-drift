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

        # Row count alignment
        if y is not None:
            if len(X) == len(y):
                checks_passed.append("xy_length_match")
            else:
                errors.append(f"Row count mismatch: features={len(X)}, labels={len(y)}.")
            n_classes = int(y.iloc[:, 0].nunique()) if len(y) > 0 else None
        else:
            n_classes = None

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
                        "license": "Apache-2.0",
                        "download_mode": "generated",
                        "source_version": "1.0",
                        "sha256": sha,
                    })

        df_summary = pd.DataFrame(records)
        out_csv = self.manifest_dir / "dataset_summary.csv"
        df_summary.to_csv(out_csv, index=False)
        return out_csv
