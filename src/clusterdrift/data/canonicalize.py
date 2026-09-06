"""Canonicalization module to serialize raw datasets into standardized, label-isolated parquet files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from clusterdrift.data.schemas import DatasetBundle, DatasetSpec


def canonicalize_bundle(
    bundle: DatasetBundle,
    spec: DatasetSpec,
    output_dir: Path,
    subdomain: Optional[str] = None,
    domain_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Serialize a DatasetBundle into standardized canonical artifacts:
      - features.parquet (strictly without labels)
      - labels.parquet (target ground truth)
      - metadata.json (feature schema, license, provenance)

    Enforces strict label isolation: target_column must NOT appear in features.parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    X = bundle.X.copy()
    y = bundle.y.copy() if bundle.y is not None else None

    # Verify and enforce label isolation
    target_col = spec.target_column
    if target_col and target_col in X.columns:
        if y is None:
            y = X[target_col].copy()
        X = X.drop(columns=[target_col])

    # Drop explicit id_columns or drop_columns from features
    for col in spec.drop_columns:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Ensure feature names are strings and clean
    X.columns = [str(c) for c in X.columns]
    feature_names = list(X.columns)

    # Save features.parquet
    features_path = output_dir / "features.parquet"
    X.to_parquet(features_path, index=False, engine="pyarrow")

    # Save labels.parquet
    labels_path = output_dir / "labels.parquet"
    if y is not None:
        if isinstance(y, pd.Series):
            y_df = pd.DataFrame({target_col or "target": y.values})
        elif isinstance(y, pd.DataFrame):
            y_df = y
        else:
            y_df = pd.DataFrame({target_col or "target": y})
        y_df.to_parquet(labels_path, index=False, engine="pyarrow")
        n_classes = int(y_df.iloc[:, 0].nunique())
    else:
        # Dummy empty labels if purely unlabeled
        y_df = pd.DataFrame({"target": []})
        y_df.to_parquet(labels_path, index=False, engine="pyarrow")
        n_classes = None

    # Construct metadata
    meta = {
        "dataset_id": spec.id,
        "display_name": spec.display_name,
        "dataset_group": spec.dataset_group,
        "subdomain": subdomain,
        "n_rows": int(len(X)),
        "n_features": int(len(feature_names)),
        "n_classes": n_classes,
        "feature_names": feature_names,
        "target_column": target_col,
        "domain_column": spec.domain_column,
        "protected_from_features": spec.domain_column is not None,
        "source_provider": spec.source_provider,
        "source_id": spec.source_id,
        "source_version": spec.source_version,
        "license": spec.license,
        "citation": spec.citation,
        "has_missing_values": bool(X.isna().any().any()),
        "extra": bundle.metadata,
        "domain_metadata": domain_meta or spec.domain_metadata,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "metadata_path": str(metadata_path),
    }
