"""Manifest generation and auditing module tracking dataset provenance, checksums, and metadata."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from clusterdrift.data.schemas import DatasetManifestEntry, DatasetSpec


def compute_file_sha256(filepath: Path) -> str:
    """Compute streaming SHA256 checksum for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Retrieve current Git commit hash or fallback."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


class ManifestManager:
    """Manages creation, updates, and serialization of dataset manifests."""

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or (Path.cwd() / "data")
        self.manifest_dir = self.data_root / "manifests"
        self.canonical_dir = self.data_root / "canonical"
        self.synthetic_dir = self.data_root / "synthetic"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def create_real_manifest_entry(
        self,
        spec: DatasetSpec,
        target_dir: Path,
        status: str = "canonicalized",
    ) -> Optional[DatasetManifestEntry]:
        """Generate a DatasetManifestEntry for a canonicalized real dataset."""
        features_path = target_dir / "features.parquet"
        metadata_path = target_dir / "metadata.json"

        if not features_path.exists() or not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        features_sha256 = compute_file_sha256(features_path)
        fnames = meta.get("feature_names", [])
        fnames_hash = hashlib.sha256("::".join(fnames).encode("utf-8")).hexdigest()

        return DatasetManifestEntry(
            dataset_id=spec.id,
            dataset_type=spec.dataset_group,
            source_provider=spec.source_provider,
            source_identifier=str(spec.source_id),
            source_version=str(spec.source_version),
            source_url=spec.source_url,
            retrieved_at=meta.get("retrieved_at", "phase_1_acquisition"),
            raw_sha256=None,
            canonical_sha256=features_sha256,
            n_rows=meta.get("n_rows", 0),
            n_features=meta.get("n_features", 0),
            n_classes=meta.get("n_classes"),
            feature_names_hash=fnames_hash,
            target_name=spec.target_column,
            license=spec.license,
            status=status,
            git_commit=get_git_commit(),
            extra={
                "domain_column": spec.domain_column,
                "subdomain": meta.get("subdomain"),
                "split_strategy": spec.split_strategy,
                "group_column": spec.group_column,
                "time_column": spec.time_column,
                "has_group_artifact": meta.get("has_group_artifact", False),
                "has_domain_artifact": meta.get("has_domain_artifact", False),
            },
        )

    def generate_all_manifests(
        self,
        specs: List[DatasetSpec],
        download_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Scan canonical and synthetic directories and write all manifest artifacts."""
        real_entries: List[Dict[str, Any]] = []
        unavailable_entries: List[Dict[str, Any]] = []

        for spec in specs:
            if spec.dataset_group == "controlled_real":
                tdir = self.canonical_dir / "controlled" / spec.id
                entry = self.create_real_manifest_entry(spec, tdir)
                if entry:
                    real_entries.append(entry.__dict__)
                else:
                    unavailable_entries.append({
                        "dataset_id": spec.id,
                        "group": spec.dataset_group,
                        "download_mode": spec.download_mode,
                        "source_url": spec.source_url,
                        "status": "missing_or_unacquired",
                    })
            elif spec.dataset_group == "natural_shift":
                if spec.source_provider == "whyshift":
                    task = spec.domain_metadata.get("task", spec.source_id)
                    task_dir = self.canonical_dir / "natural" / "whyshift" / task
                    domains = spec.domain_metadata.get("domains", [])
                    found_any = False
                    for d in domains:
                        ddir = task_dir / d
                        entry = self.create_real_manifest_entry(spec, ddir)
                        if entry:
                            found_any = True
                            d_dict = entry.__dict__.copy()
                            d_dict["dataset_id"] = f"{spec.id}_{d}"
                            d_dict["domain"] = d
                            real_entries.append(d_dict)
                    if not found_any:
                        unavailable_entries.append({
                            "dataset_id": spec.id,
                            "group": spec.dataset_group,
                            "download_mode": spec.download_mode,
                            "source_url": spec.source_url,
                            "status": "missing_or_unacquired",
                        })
                elif spec.source_provider == "tableshift":
                    task_dir = self.canonical_dir / "natural" / "tableshift" / spec.id
                    entry = self.create_real_manifest_entry(spec, task_dir)
                    if entry:
                        real_entries.append(entry.__dict__)
                    else:
                        unavailable_entries.append({
                            "dataset_id": spec.id,
                            "group": spec.dataset_group,
                            "download_mode": spec.download_mode,
                            "source_url": spec.source_url,
                            "status": "missing_or_unacquired",
                        })

        # Scan synthetic datasets
        synthetic_entries: List[Dict[str, Any]] = []
        if self.synthetic_dir.exists():
            for sdir in sorted(self.synthetic_dir.iterdir()):
                if sdir.is_dir() and (sdir / "metadata.json").exists():
                    fpath = sdir / "features.parquet"
                    mpath = sdir / "metadata.json"
                    ppath = sdir / "parameters.json"
                    with open(mpath, "r", encoding="utf-8") as f:
                        smeta = json.load(f)
                    f_hash = compute_file_sha256(fpath)
                    p_hash = compute_file_sha256(ppath)
                    synthetic_entries.append({
                        "family_id": sdir.name,
                        "generator": smeta.get("generator"),
                        "generator_seed": smeta.get("generator_seed"),
                        "K": smeta.get("K"),
                        "n_rows": smeta.get("n_rows"),
                        "n_features": smeta.get("n_features"),
                        "canonical_sha256": f_hash,
                        "parameters_sha256": p_hash,
                        "soft_truth_available": True,
                        "git_commit": get_git_commit(),
                    })

        # Write datasets.json
        with open(self.manifest_dir / "datasets.json", "w", encoding="utf-8") as f:
            json.dump(real_entries, f, indent=2)

        # Write synthetic_manifest.json
        with open(self.manifest_dir / "synthetic_manifest.json", "w", encoding="utf-8") as f:
            json.dump(synthetic_entries, f, indent=2)

        # Write unavailable_datasets.json
        with open(self.manifest_dir / "unavailable_datasets.json", "w", encoding="utf-8") as f:
            json.dump(unavailable_entries, f, indent=2)

        # Write download_report.json if provided
        if download_results is not None:
            with open(self.manifest_dir / "download_report.json", "w", encoding="utf-8") as f:
                json.dump(download_results, f, indent=2)

        return {
            "real_entries_count": len(real_entries),
            "synthetic_entries_count": len(synthetic_entries),
            "unavailable_entries_count": len(unavailable_entries),
        }
