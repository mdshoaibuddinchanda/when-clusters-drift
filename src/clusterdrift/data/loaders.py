"""Loaders module converting provider-specific outputs into standard DatasetBundle objects."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine

from clusterdrift.data.schemas import DatasetBundle, DatasetSpec


def load_sklearn_dataset(spec: DatasetSpec) -> DatasetBundle:
    """Load scikit-learn built-in datasets."""
    sid = spec.source_id
    if sid == "sklearn_iris":
        bunch = load_iris(as_frame=True)
    elif sid == "sklearn_wine":
        bunch = load_wine(as_frame=True)
    elif sid == "sklearn_breast_cancer":
        bunch = load_breast_cancer(as_frame=True)
    else:
        raise ValueError(f"Unknown sklearn dataset id: {sid}")

    X = bunch.data.copy()
    y = bunch.target.copy()
    feature_names = list(X.columns)

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        metadata={"target_names": bunch.target_names.tolist() if hasattr(bunch, "target_names") else []},
    )


def load_openml_dataset(spec: DatasetSpec, data_home: Optional[Path] = None) -> DatasetBundle:
    """Load dataset from OpenML via official OpenML data_id."""
    data_id = int(spec.source_id)
    bunch = fetch_openml(
        data_id=data_id,
        as_frame=True,
        data_home=str(data_home) if data_home else None,
        parser="auto",
    )

    X = bunch.data.copy()
    y = bunch.target.copy() if bunch.target is not None else None

    # Handle rare cases where X is a numpy array
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    feature_names = list(X.columns)
    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        metadata={"openml_id": data_id, "url": spec.source_url},
    )


def load_whyshift_dataset(
    spec: DatasetSpec,
    domain: str,
    raw_dir: Path,
    year: int = 2018,
) -> DatasetBundle:
    """Load a specific domain split from WhyShift (e.g. ACSIncome for state CA)."""
    import whyshift

    task = spec.domain_metadata.get("task", spec.source_id)
    raw_dir.mkdir(parents=True, exist_ok=True)

    res = whyshift.get_data(
        task=task,
        state=domain,
        need_preprocess=False,
        root_dir=str(raw_dir),
        year=year,
    )

    X_arr, y_arr, fnames = res[0], res[1], res[2]
    if fnames is None or len(fnames) != X_arr.shape[1]:
        fnames = [f"feature_{i}" for i in range(X_arr.shape[1])]

    X_df = pd.DataFrame(X_arr, columns=fnames)
    y_s = pd.Series(y_arr, name="target")

    return DatasetBundle(
        X=X_df,
        y=y_s,
        feature_names=fnames,
        metadata={"domain": domain, "task": task, "year": year},
    )
