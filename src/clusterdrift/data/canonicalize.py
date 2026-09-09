"""Canonicalization module to serialize raw datasets into standardized, label-isolated parquet files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from clusterdrift.data.schemas import DatasetBundle, DatasetSpec
from clusterdrift.shifts.hashing import atomic_write_json, atomic_write_parquet


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

    # Verify and enforce domain isolation
    domain_col = spec.domain_column
    domains = bundle.domains
    if domain_col and domain_col in X.columns:
        if domains is None:
            domains = pd.DataFrame({domain_col: X[domain_col].copy()})
        X = X.drop(columns=[domain_col])

    # Verify and enforce group isolation
    group_col = spec.group_column
    groups = bundle.groups
    if group_col and group_col in X.columns:
        if groups is None:
            groups = pd.DataFrame({group_col: X[group_col].copy()})
        X = X.drop(columns=[group_col])

    # Drop explicit id_columns or drop_columns from features
    for col in list(spec.id_columns) + list(spec.drop_columns):
        if col in X.columns:
            X = X.drop(columns=[col])

    # Ensure feature names are strings and clean
    X.columns = [str(c) for c in X.columns]
    feature_names = list(X.columns)

    # Save features.parquet
    features_path = output_dir / "features.parquet"
    atomic_write_parquet(features_path, X, index=False, engine="pyarrow")

    # Save labels.parquet
    labels_path = output_dir / "labels.parquet"
    if y is not None:
        if isinstance(y, pd.Series):
            y_df = pd.DataFrame({target_col or "target": y.values})
        elif isinstance(y, pd.DataFrame):
            y_df = y
        else:
            y_df = pd.DataFrame({target_col or "target": y})
        atomic_write_parquet(labels_path, y_df, index=False, engine="pyarrow")
        n_classes = int(y_df.iloc[:, 0].nunique())
    else:
        # Dummy empty labels if purely unlabeled
        y_df = pd.DataFrame({"target": []})
        atomic_write_parquet(labels_path, y_df, index=False, engine="pyarrow")
        n_classes = None

    # Save domains.parquet if domain artifact exists
    domains_path = output_dir / "domains.parquet"
    has_domains = domains is not None
    if has_domains:
        if isinstance(domains, pd.DataFrame):
            dom_df = domains.copy()
        elif isinstance(domains, pd.Series):
            dom_df = pd.DataFrame({domain_col or "domain": domains.values})
        else:
            dom_df = pd.DataFrame({domain_col or "domain": domains})
        atomic_write_parquet(domains_path, dom_df, index=False, engine="pyarrow")

    # Save groups.parquet if group artifact exists
    groups_path = output_dir / "groups.parquet"
    has_groups = groups is not None
    if has_groups:
        if isinstance(groups, pd.DataFrame):
            grp_df = groups.copy()
        elif isinstance(groups, pd.Series):
            grp_df = pd.DataFrame({group_col or "group": groups.values})
        else:
            grp_df = pd.DataFrame({group_col or "group": groups})
        atomic_write_parquet(groups_path, grp_df, index=False, engine="pyarrow")

    # Build feature roles ensuring completeness
    feature_roles = dict(spec.feature_roles)
    if bundle.feature_roles:
        feature_roles.update(bundle.feature_roles)
    for col in feature_names:
        if col not in feature_roles:
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_roles[col] = "numeric"
            else:
                feature_roles[col] = "categorical"

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
        "feature_roles": feature_roles,
        "target_column": target_col,
        "domain_column": domain_col,
        "group_column": group_col,
        "time_column": spec.time_column,
        "split_strategy": spec.split_strategy,
        "has_domain_artifact": has_domains,
        "has_group_artifact": has_groups,
        "protected_from_features": domain_col is not None,
        "source_provider": spec.source_provider,
        "source_id": spec.source_id,
        "source_version": spec.source_version,
        "license": spec.license,
        "citation": spec.citation,
        "has_missing_values": bool(X.isna().any().any()),
        "extra": bundle.metadata,
        "domain_metadata": {**(spec.domain_metadata or {}), **(domain_meta or {})},
    }

    metadata_path = output_dir / "metadata.json"
    atomic_write_json(metadata_path, meta, indent=2, sort_keys=False)

    ret = {
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "metadata_path": str(metadata_path),
    }
    if has_domains:
        ret["domains_path"] = str(domains_path)
    if has_groups:
        ret["groups_path"] = str(groups_path)
    return ret
