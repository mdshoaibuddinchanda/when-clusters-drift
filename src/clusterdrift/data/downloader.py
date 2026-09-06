"""Downloader module managing download, caching, idempotency, and canonicalization of real datasets."""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from clusterdrift.data.canonicalize import canonicalize_bundle
from clusterdrift.data.loaders import (
    load_openml_dataset,
    load_sklearn_dataset,
    load_tableshift_hospital_readmission,
    load_uci_har_dataset,
    load_whyshift_dataset,
)
from clusterdrift.data.schemas import DatasetSpec, DownloadMode, DownloadResult, DownloadStatus


class DatasetDownloader:
    """Orchestrator for idempotent, credential-aware acquisition of benchmark datasets."""

    def __init__(self, data_root: Optional[Path] = None, force: bool = False, verify: bool = True):
        self.data_root = data_root or (Path.cwd() / "data")
        self.raw_dir = self.data_root / "raw"
        self.canonical_dir = self.data_root / "canonical"
        self.force = force
        self.verify = verify

        # Ensure directory structure exists
        (self.raw_dir / "controlled").mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "natural" / "whyshift").mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "natural" / "tableshift").mkdir(parents=True, exist_ok=True)
        (self.canonical_dir / "controlled").mkdir(parents=True, exist_ok=True)
        (self.canonical_dir / "natural").mkdir(parents=True, exist_ok=True)

    def is_canonical_present(self, spec: DatasetSpec) -> bool:
        """Check if dataset already exists in canonical form and is complete."""
        if spec.dataset_group == "controlled_real":
            target_dir = self.canonical_dir / "controlled" / spec.id
            f_path = target_dir / "features.parquet"
            l_path = target_dir / "labels.parquet"
            m_path = target_dir / "metadata.json"
            return f_path.exists() and l_path.exists() and m_path.exists() and f_path.stat().st_size > 0
        elif spec.dataset_group == "natural_shift":
            if spec.source_provider == "whyshift":
                task = spec.domain_metadata.get("task", spec.source_id)
                task_dir = self.canonical_dir / "natural" / "whyshift" / task
                domains = spec.domain_metadata.get("domains", [])
                if not domains or not task_dir.exists():
                    return False
                return all(
                    (task_dir / d / "features.parquet").exists() and (task_dir / d / "features.parquet").stat().st_size > 0
                    for d in domains
                )
            elif spec.source_provider == "tableshift":
                task_dir = self.canonical_dir / "natural" / "tableshift" / spec.id
                return (task_dir / "features.parquet").exists() and (task_dir / "features.parquet").stat().st_size > 0
        return False

    def download(self, spec: DatasetSpec) -> DownloadResult:
        """Execute acquisition for a single dataset with idempotency and credential checking."""
        t0 = time.time()

        # 1. Check idempotency
        if not self.force and self.is_canonical_present(spec):
            return DownloadResult(
                dataset_id=spec.id,
                status=DownloadStatus.ALREADY_EXISTS,
                message=f"[SKIP VERIFIED] {spec.id} already exists in canonical format.",
                duration_seconds=round(time.time() - t0, 3),
            )

        # 2. Check credentials & download mode
        if spec.download_mode == DownloadMode.AUTH_REQUIRED:
            return self._handle_auth_required(spec, t0)
        elif spec.download_mode == DownloadMode.MANUAL_LICENSE_ACCEPTANCE:
            return self._handle_manual_license(spec, t0)
        elif spec.download_mode == DownloadMode.UNAVAILABLE:
            return DownloadResult(
                dataset_id=spec.id,
                status=DownloadStatus.FAILED,
                message=f"[UNAVAILABLE] {spec.id}: source is marked unavailable.",
                duration_seconds=round(time.time() - t0, 3),
            )

        # 3. Dispatch to appropriate provider adapter
        try:
            if spec.source_provider == "sklearn":
                return self._download_sklearn(spec, t0)
            elif spec.source_provider == "openml":
                return self._download_openml(spec, t0)
            elif spec.source_provider == "whyshift":
                return self._download_whyshift(spec, t0)
            elif spec.source_provider == "tableshift":
                return self._download_tableshift(spec, t0)
            elif spec.source_provider == "uci":
                return self._download_uci(spec, t0)
            else:
                return DownloadResult(
                    dataset_id=spec.id,
                    status=DownloadStatus.FAILED,
                    message=f"Unsupported provider: {spec.source_provider}",
                    duration_seconds=round(time.time() - t0, 3),
                )
        except Exception as e:
            return DownloadResult(
                dataset_id=spec.id,
                status=DownloadStatus.FAILED,
                message=f"Error downloading {spec.id}: {str(e)}",
                duration_seconds=round(time.time() - t0, 3),
            )

    def _handle_auth_required(self, spec: DatasetSpec, t0: float) -> DownloadResult:
        """Handle credential-gated dataset by checking if local archive exists or reporting instructions."""
        # Check if user already manually provided the raw file in data/raw/
        raw_controlled = self.raw_dir / "controlled" / spec.id
        raw_natural = self.raw_dir / "natural" / spec.source_provider / spec.id

        if raw_controlled.exists() or raw_natural.exists():
            # Process local file if provided
            pass

        instructions = (
            f"[AUTH REQUIRED] {spec.id}: Requires credentials or API token. "
            f"Source: {spec.source_url}. "
            f"To enable, configure access or place downloaded raw archive in '{raw_natural}'."
        )
        return DownloadResult(
            dataset_id=spec.id,
            status=DownloadStatus.AUTH_REQUIRED,
            message=instructions,
            duration_seconds=round(time.time() - t0, 3),
        )

    def _handle_manual_license(self, spec: DatasetSpec, t0: float) -> DownloadResult:
        """Handle manual license acceptance required dataset."""
        instructions = (
            f"[MANUAL LICENSE REQUIRED] {spec.id}: Requires manual license acceptance at {spec.source_url}. "
            f"License: {spec.license}. "
            f"Once accepted, place raw data files into 'data/raw/natural/tableshift/{spec.id}/'."
        )
        return DownloadResult(
            dataset_id=spec.id,
            status=DownloadStatus.MANUAL_LICENSE_ACCEPTANCE,
            message=instructions,
            duration_seconds=round(time.time() - t0, 3),
        )

    def _download_sklearn(self, spec: DatasetSpec, t0: float) -> DownloadResult:
        bundle = load_sklearn_dataset(spec)
        out_dir = self.canonical_dir / "controlled" / spec.id
        paths = canonicalize_bundle(bundle, spec, out_dir)
        return DownloadResult(
            dataset_id=spec.id,
            status=DownloadStatus.OK,
            message=f"[OK] {spec.id} acquired from scikit-learn built-in.",
            n_rows=len(bundle.X),
            n_features=len(bundle.feature_names),
            n_classes=int(bundle.y.nunique()) if bundle.y is not None else None,
            canonical_files=list(paths.values()),
            duration_seconds=round(time.time() - t0, 3),
        )

    def _download_openml(self, spec: DatasetSpec, t0: float) -> DownloadResult:
        raw_cache = self.raw_dir / "controlled" / spec.id
        raw_cache.mkdir(parents=True, exist_ok=True)
        bundle = load_openml_dataset(spec, data_home=raw_cache)
        out_dir = self.canonical_dir / "controlled" / spec.id
        paths = canonicalize_bundle(bundle, spec, out_dir)
        return DownloadResult(
            dataset_id=spec.id,
            status=DownloadStatus.OK,
            message=f"[OK] {spec.id} acquired from OpenML (data_id={spec.source_id}).",
            n_rows=len(bundle.X),
            n_features=len(bundle.feature_names),
            n_classes=int(bundle.y.nunique()) if bundle.y is not None else None,
            canonical_files=list(paths.values()),
            duration_seconds=round(time.time() - t0, 3),
        )

    def _download_whyshift(self, spec: DatasetSpec, t0: float) -> DownloadResult:
        task = spec.domain_metadata.get("task", spec.source_id)
        domains = spec.domain_metadata.get("domains", ["CA"])
        task_raw = self.raw_dir / "natural" / "whyshift" / task
        task_canonical = self.canonical_dir / "natural" / "whyshift" / task

        total_rows = 0
        canonical_files = []
        n_features = 0
        n_classes = 2

        for domain in domains:
            domain_raw = task_raw / domain
            domain_out = task_canonical / domain
            bundle = load_whyshift_dataset(spec, domain=domain, raw_dir=domain_raw)
            paths = canonicalize_bundle(
                bundle,
                spec,
                domain_out,
                subdomain=domain,
                domain_meta={"state": domain, "task": task, "year": 2018},
            )
            total_rows += len(bundle.X)
            n_features = len(bundle.feature_names)
            canonical_files.extend(list(paths.values()))

        return DownloadResult(
            dataset_id=spec.id,
            status=DownloadStatus.OK,
            message=f"[OK] {spec.id} acquired across {len(domains)} state domains: {domains}.",
            n_rows=total_rows,
            n_features=n_features,
            n_classes=n_classes,
            canonical_files=canonical_files,
            duration_seconds=round(time.time() - t0, 3),
        )

    def _download_uci(self, spec: DatasetSpec, t0: float) -> DownloadResult:
        """Handle UCI datasets using authentic repository archives."""
        raw_cache = self.raw_dir / "controlled" / spec.id
        raw_cache.mkdir(parents=True, exist_ok=True)
        if spec.id == "human_activity_recognition":
            bundle = load_uci_har_dataset(spec, raw_cache)
        else:
            raise ValueError(f"Unknown UCI dataset: {spec.id}")

        out_dir = self.canonical_dir / "controlled" / spec.id
        paths = canonicalize_bundle(bundle, spec, out_dir)
        return DownloadResult(
            dataset_id=spec.id,
            status=DownloadStatus.OK,
            message=f"[OK] {spec.id} acquired from UCI Repository (id={spec.source_id}).",
            n_rows=len(bundle.X),
            n_features=len(bundle.feature_names),
            n_classes=int(bundle.y.nunique()) if bundle.y is not None else None,
            canonical_files=list(paths.values()),
            duration_seconds=round(time.time() - t0, 3),
        )

    def _download_tableshift(self, spec: DatasetSpec, t0: float) -> DownloadResult:
        """Handle TableShift datasets using official benchmark sources."""
        raw_cache = self.raw_dir / "natural" / "tableshift" / spec.id
        raw_cache.mkdir(parents=True, exist_ok=True)
        if spec.id == "tableshift_hospital_readmission":
            bundle = load_tableshift_hospital_readmission(spec, raw_cache)
            out_dir = self.canonical_dir / "natural" / "tableshift" / spec.id
            domain_meta = {
                "task": "diabetes_readmission",
                "domain_column": "admission_source_id",
            }
            paths = canonicalize_bundle(bundle, spec, out_dir, domain_meta=domain_meta)
            return DownloadResult(
                dataset_id=spec.id,
                status=DownloadStatus.OK,
                message=f"[OK] {spec.id} acquired from TableShift diabetes readmission benchmark.",
                n_rows=len(bundle.X),
                n_features=len(bundle.feature_names),
                n_classes=int(bundle.y.nunique()) if bundle.y is not None else None,
                canonical_files=list(paths.values()),
                duration_seconds=round(time.time() - t0, 3),
            )
        elif spec.source_type == "openml" and spec.source_id:
            bundle = load_openml_dataset(spec, data_home=raw_cache)
            out_dir = self.canonical_dir / "natural" / "tableshift" / spec.id
            paths = canonicalize_bundle(bundle, spec, out_dir)
            return DownloadResult(
                dataset_id=spec.id,
                status=DownloadStatus.OK,
                message=f"[OK] {spec.id} acquired from OpenML data_id={spec.source_id}.",
                n_rows=len(bundle.X),
                n_features=len(bundle.feature_names),
                n_classes=int(bundle.y.nunique()) if bundle.y is not None else None,
                canonical_files=list(paths.values()),
                duration_seconds=round(time.time() - t0, 3),
            )
        else:
            return DownloadResult(
                dataset_id=spec.id,
                status=DownloadStatus.AUTH_REQUIRED if spec.download_mode == DownloadMode.AUTH_REQUIRED else DownloadStatus.FAILED,
                message=f"[NOTICE] {spec.id} requires TableShift source files or manual fetch from {spec.source_url}.",
                duration_seconds=round(time.time() - t0, 3),
            )
