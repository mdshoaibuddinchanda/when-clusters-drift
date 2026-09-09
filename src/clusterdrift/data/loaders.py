"""Loaders module converting provider-specific outputs into standard DatasetBundle objects."""

import io
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Tuple
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine

from clusterdrift.data.schemas import DatasetBundle, DatasetSpec


_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_or_download_pinned_archive(url: str, path: Path, expected_sha256: Optional[str]) -> bytes:
    """Read or atomically download an HTTPS archive and enforce its pinned digest."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError(f"A pinned source_sha256 is required for archive acquisition: {path.name}")
    expected_sha256 = expected_sha256.lower()
    if path.exists() and path.stat().st_size > 0:
        content = path.read_bytes()
        actual = _sha256_bytes(content)
        if actual != expected_sha256:
            raise ValueError(f"Cached archive digest mismatch for {path}: expected {expected_sha256}, got {actual}")
        return content
    if not url.lower().startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS dataset download: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "clusterdrift-research/1"})
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".download", dir=path.parent)
    tmp = Path(raw)
    try:
        total = 0
        hasher = hashlib.sha256()
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=60) as response:
            advertised = response.headers.get("Content-Length")
            if advertised and int(advertised) > _MAX_ARCHIVE_BYTES:
                raise ValueError(f"Dataset archive exceeds {_MAX_ARCHIVE_BYTES} bytes: {url}")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_ARCHIVE_BYTES:
                    raise ValueError(f"Dataset archive exceeds {_MAX_ARCHIVE_BYTES} bytes: {url}")
                hasher.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = hasher.hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"Downloaded archive digest mismatch for {url}: expected {expected_sha256}, got {actual}")
        os.replace(tmp, path)
        return path.read_bytes()
    finally:
        if tmp.exists():
            tmp.unlink()


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
    feature_roles = {fn: "numeric" for fn in feature_names}

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        feature_roles=feature_roles,
        metadata={
            "source_id": sid,
            "sklearn_name": spec.id,
            "target_names": bunch.target_names.tolist() if hasattr(bunch, "target_names") else [],
        },
    )


def load_uci_mice_protein_dataset(spec: DatasetSpec, raw_dir: Path) -> DatasetBundle:
    """Load authentic UCI Mice Protein Expression dataset (UCI 342, Data_Cortex_Nuclear.xls)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "mice_protein_expression.zip"

    url = spec.source_url or "https://archive.ics.uci.edu/static/public/342/mice+protein+expression.zip"
    content = _load_or_download_pinned_archive(url, zip_path, spec.source_sha256)

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open("Data_Cortex_Nuclear.xls") as f:
            df_uci = pd.read_excel(f)

    # Biological grouping: 72 biological mice with 15 measurements per mouse (1,080 total)
    # Physical mouse identifier derived deterministically by stripping the measurement index suffix (e.g. '309_1' -> '309')
    raw_mouse_id = df_uci["MouseID"].astype(str)
    mouse_subject_id = raw_mouse_id.str.rsplit("_", n=1).str[0]
    groups_df = pd.DataFrame({"mouse_subject_id": mouse_subject_id})

    # Features: strictly 77 cortical protein expression measurements
    # Explicitly exclude identifiers, grouping columns, design metadata, and target
    non_feature_cols = {"MouseID", "mouse_subject_id", "Genotype", "Treatment", "Behavior", "class"}
    feature_cols = [c for c in df_uci.columns if c not in non_feature_cols]

    X = df_uci[feature_cols].copy()
    y = df_uci["class"].copy().astype(str)

    feature_roles = {col: "numeric" for col in feature_cols}

    meta: Dict[str, Any] = {
        "uci_id": "342",
        "uci_name": "Mice Protein Expression",
        "n_groups": int(groups_df["mouse_subject_id"].nunique()),
        "url": url,
    }

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_cols,
        groups=groups_df,
        feature_roles=feature_roles,
        metadata=meta,
    )


def load_openml_dataset(
    spec: DatasetSpec,
    data_home: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> DatasetBundle:
    """Load dataset from OpenML via official OpenML data_id and capture source dataset name."""
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
    openml_name = bunch.details.get("name") if hasattr(bunch, "details") else None

    feature_roles = {}
    for col in feature_names:
        if pd.api.types.is_numeric_dtype(X[col]):
            feature_roles[col] = "numeric"
        else:
            feature_roles[col] = "categorical"

    meta: Dict[str, Any] = {
        "openml_id": data_id,
        "openml_name": openml_name,
        "url": spec.source_url,
    }

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        feature_roles=feature_roles,
        metadata=meta,
    )


def load_uci_har_dataset(spec: DatasetSpec, raw_dir: Path) -> DatasetBundle:
    """Load authentic UCI Human Activity Recognition Using Smartphones dataset (UCI 240)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "human_activity_recognition_using_smartphones.zip"

    url = spec.source_url or "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    content = _load_or_download_pinned_archive(url, zip_path, spec.source_sha256)

    with zipfile.ZipFile(io.BytesIO(content)) as outer_zip:
        inner_bytes = outer_zip.read("UCI HAR Dataset.zip")

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z:
        # Read feature names
        feats_raw = z.read("UCI HAR Dataset/features.txt").decode("utf-8").strip().split("\n")
        feat_names = [line.strip().split()[1] for line in feats_raw]
        seen = {}
        unique_feat_names = []
        for fn in feat_names:
            if fn in seen:
                seen[fn] += 1
                unique_feat_names.append(f"{fn}_{seen[fn]}")
            else:
                seen[fn] = 0
                unique_feat_names.append(fn)

        # Read train + test splits
        X_train = pd.read_csv(io.BytesIO(z.read("UCI HAR Dataset/train/X_train.txt")), sep=r"\s+", header=None, names=unique_feat_names, dtype=np.float32)
        y_train = pd.read_csv(io.BytesIO(z.read("UCI HAR Dataset/train/y_train.txt")), sep=r"\s+", header=None, names=["activity"])
        sub_train = pd.read_csv(io.BytesIO(z.read("UCI HAR Dataset/train/subject_train.txt")), sep=r"\s+", header=None, names=["subject_id"])

        X_test = pd.read_csv(io.BytesIO(z.read("UCI HAR Dataset/test/X_test.txt")), sep=r"\s+", header=None, names=unique_feat_names, dtype=np.float32)
        y_test = pd.read_csv(io.BytesIO(z.read("UCI HAR Dataset/test/y_test.txt")), sep=r"\s+", header=None, names=["activity"])
        sub_test = pd.read_csv(io.BytesIO(z.read("UCI HAR Dataset/test/subject_test.txt")), sep=r"\s+", header=None, names=["subject_id"])

        X_df = pd.concat([X_train, X_test], ignore_index=True)
        y_df = pd.concat([y_train, y_test], ignore_index=True)["activity"]
        groups_df = pd.concat([sub_train, sub_test], ignore_index=True)

    feature_roles = {fn: "numeric" for fn in unique_feat_names}

    return DatasetBundle(
        X=X_df,
        y=y_df,
        feature_names=unique_feat_names,
        groups=groups_df,
        feature_roles=feature_roles,
        metadata={
            "uci_id": 240,
            "uci_name": "Human Activity Recognition Using Smartphones",
            "url": url,
            "train_observations": len(X_train),
            "test_observations": len(X_test),
            "n_groups": int(groups_df["subject_id"].nunique()),
        },
    )


def load_tableshift_hospital_readmission(spec: DatasetSpec, raw_dir: Path) -> DatasetBundle:
    """Load authentic TableShift Diabetes Hospital Readmission benchmark (UCI 296)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "diabetes_130_us_hospitals.zip"

    url = spec.source_url or "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    content = _load_or_download_pinned_archive(url, zip_path, spec.source_sha256)

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open("diabetic_data.csv") as f:
            df = pd.read_csv(f)

    # Standard TableShift benchmark filter: drop encounters with missing race
    df = df[df["race"] != "?"].copy()

    # Binary target readmission: 1 if readmitted (either <30 or >30), 0 if NO
    y_series = (df["readmitted"] != "NO").astype(int)
    y_series.name = "readmitted"

    # Natural-shift domain variable: admission_source_id isolated into domains.parquet
    domain_series = df["admission_source_id"].astype(int)
    domains_df = pd.DataFrame({"admission_source_id": domain_series})

    # Drop patient/encounter IDs, target column, AND domain column from features
    drop_cols = ["encounter_id", "patient_nbr", "readmitted", "admission_source_id"]
    X_df = df.drop(columns=drop_cols).copy()

    # Preserve semantic data types WITHOUT acquisition-time factorization or one-hot encoding
    ordinal_cols = {"age", "max_glu_serum", "A1Cresult"}
    numeric_cols = {
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    }
    feature_roles = {}
    for col in X_df.columns:
        if col in ordinal_cols:
            feature_roles[col] = "ordinal"
            X_df[col] = X_df[col].astype(str)
        elif col in numeric_cols:
            feature_roles[col] = "numeric"
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce").astype(np.float32)
        else:
            feature_roles[col] = "categorical"
            X_df[col] = X_df[col].astype(str)

    return DatasetBundle(
        X=X_df,
        y=y_series,
        domains=domains_df,
        feature_roles=feature_roles,
        feature_names=list(X_df.columns),
        metadata={
            "tableshift_task": "diabetes_readmission",
            "domain_column": "admission_source_id",
            "domain_values": sorted(list(domain_series.unique().tolist())),
            "n_domains": int(domain_series.nunique()),
            "url": url,
        },
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
