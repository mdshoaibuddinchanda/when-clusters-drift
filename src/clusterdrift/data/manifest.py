"""Manifest generation and auditing module tracking dataset provenance, checksums, and metadata."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from clusterdrift.data.schemas import DatasetManifestEntry, DatasetSpec, SyntheticManifestEntry


def compute_file_sha256(filepath: Path) -> str:
    """Compute streaming SHA256 checksum for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_canonical_bundle_sha256(
    features_sha256: str,
    labels_sha256: str,
    groups_sha256: Optional[str],
    domains_sha256: Optional[str],
    metadata_sha256: str,
) -> str:
    """Compute deterministic canonical bundle hash for a real dataset.

    Fixed component order: features, labels, groups, domains, metadata.
    Absent optional artifacts use 'NONE' digest in payload.
    """
    components = [
        ("features", features_sha256),
        ("labels", labels_sha256),
        ("groups", groups_sha256 or "NONE"),
        ("domains", domains_sha256 or "NONE"),
        ("metadata", metadata_sha256),
    ]
    payload = "\n".join(f"{name}:{digest}" for name, digest in components)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_synthetic_bundle_sha256(
    features_sha256: str,
    hard_labels_sha256: str,
    soft_memberships_sha256: str,
    parameters_sha256: str,
    metadata_sha256: str,
) -> str:
    """Compute deterministic canonical bundle hash for a synthetic benchmark family.

    Fixed component order: features, hard_labels, soft_memberships, parameters, metadata.
    """
    components = [
        ("features", features_sha256),
        ("hard_labels", hard_labels_sha256),
        ("soft_memberships", soft_memberships_sha256),
        ("parameters", parameters_sha256),
        ("metadata", metadata_sha256),
    ]
    payload = "\n".join(f"{name}:{digest}" for name, digest in components)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_real_bundle_hashes(target_dir: Path) -> Dict[str, Optional[str]]:
    """Calculate individual artifact hashes and bundle hash for a real canonical directory."""
    fpath = target_dir / "features.parquet"
    lpath = target_dir / "labels.parquet"
    gpath = target_dir / "groups.parquet"
    dpath = target_dir / "domains.parquet"
    mpath = target_dir / "metadata.json"

    f_sha = compute_file_sha256(fpath) if fpath.exists() else None
    l_sha = compute_file_sha256(lpath) if lpath.exists() else None
    g_sha = compute_file_sha256(gpath) if gpath.exists() else None
    d_sha = compute_file_sha256(dpath) if dpath.exists() else None
    m_sha = compute_file_sha256(mpath) if mpath.exists() else None

    bundle_sha = None
    if f_sha and l_sha and m_sha:
        bundle_sha = compute_canonical_bundle_sha256(
            features_sha256=f_sha,
            labels_sha256=l_sha,
            groups_sha256=g_sha,
            domains_sha256=d_sha,
            metadata_sha256=m_sha,
        )

    return {
        "features_sha256": f_sha,
        "labels_sha256": l_sha,
        "groups_sha256": g_sha,
        "domains_sha256": d_sha,
        "metadata_sha256": m_sha,
        "canonical_bundle_sha256": bundle_sha,
    }


def compute_synthetic_bundle_hashes(sdir: Path) -> Dict[str, Optional[str]]:
    """Calculate individual artifact hashes and bundle hash for a synthetic benchmark directory."""
    fpath = sdir / "features.parquet"
    hpath = sdir / "hard_labels.parquet"
    spath = sdir / "soft_memberships.npy"
    ppath = sdir / "parameters.json"
    mpath = sdir / "metadata.json"

    f_sha = compute_file_sha256(fpath) if fpath.exists() else None
    h_sha = compute_file_sha256(hpath) if hpath.exists() else None
    s_sha = compute_file_sha256(spath) if spath.exists() else None
    p_sha = compute_file_sha256(ppath) if ppath.exists() else None
    m_sha = compute_file_sha256(mpath) if mpath.exists() else None

    bundle_sha = None
    if f_sha and h_sha and s_sha and p_sha and m_sha:
        bundle_sha = compute_synthetic_bundle_sha256(
            features_sha256=f_sha,
            hard_labels_sha256=h_sha,
            soft_memberships_sha256=s_sha,
            parameters_sha256=p_sha,
            metadata_sha256=m_sha,
        )

    return {
        "features_sha256": f_sha,
        "hard_labels_sha256": h_sha,
        "soft_memberships_sha256": s_sha,
        "parameters_sha256": p_sha,
        "metadata_sha256": m_sha,
        "canonical_bundle_sha256": bundle_sha,
    }


def get_git_commit() -> str:
    """Retrieve current Git commit hash representing generated_from_commit."""
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
    """Manages creation, updates, and serialization of dataset manifests (Schema v2)."""

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or (Path.cwd() / "data")
        self.raw_dir = self.data_root / "raw"
        self.manifest_dir = self.data_root / "manifests"
        self.canonical_dir = self.data_root / "canonical"
        self.synthetic_dir = self.data_root / "synthetic"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def create_real_manifest_entry(
        self,
        spec: DatasetSpec,
        target_dir: Path,
        status: str = "canonicalized",
        domain: Optional[str] = None,
        source_commit: Optional[str] = None,
    ) -> Optional[DatasetManifestEntry]:
        """Generate a Schema v2 DatasetManifestEntry for a canonicalized real dataset."""
        features_path = target_dir / "features.parquet"
        labels_path = target_dir / "labels.parquet"
        groups_path = target_dir / "groups.parquet"
        domains_path = target_dir / "domains.parquet"
        metadata_path = target_dir / "metadata.json"

        if not features_path.exists() or not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        features_sha256 = compute_file_sha256(features_path)
        labels_sha256 = compute_file_sha256(labels_path) if labels_path.exists() else "NONE"
        groups_sha256 = compute_file_sha256(groups_path) if groups_path.exists() else None
        domains_sha256 = compute_file_sha256(domains_path) if domains_path.exists() else None
        metadata_sha256 = compute_file_sha256(metadata_path)

        bundle_sha256 = compute_canonical_bundle_sha256(
            features_sha256=features_sha256,
            labels_sha256=labels_sha256,
            groups_sha256=groups_sha256,
            domains_sha256=domains_sha256,
            metadata_sha256=metadata_sha256,
        )

        fnames = meta.get("feature_names", [])
        fnames_hash = hashlib.sha256("::".join(fnames).encode("utf-8")).hexdigest()

        # Raw artifact hashing
        raw_sha256 = None
        raw_hash_status = "not_applicable"

        if spec.source_provider == "uci":
            raw_source_dir = self.raw_dir / "controlled" / spec.id
            zips = list(raw_source_dir.glob("*.zip")) if raw_source_dir.exists() else []
            if zips:
                raw_sha256 = compute_file_sha256(zips[0])
                raw_hash_status = "verified"
            else:
                raw_hash_status = "provider_managed"
        elif spec.source_provider == "tableshift":
            raw_source_dir = self.raw_dir / "natural" / "tableshift" / spec.id
            zips = list(raw_source_dir.glob("*.zip")) if raw_source_dir.exists() else []
            if zips:
                raw_sha256 = compute_file_sha256(zips[0])
                raw_hash_status = "verified"
            else:
                raw_hash_status = "provider_managed"
        elif spec.source_provider == "whyshift":
            task = spec.domain_metadata.get("task", spec.source_id)
            dom = domain or meta.get("subdomain") or "CA"
            raw_dom_dir = self.raw_dir / "natural" / "whyshift" / task / dom
            csvs = list(raw_dom_dir.rglob("*.csv")) if raw_dom_dir.exists() else []
            if csvs:
                raw_sha256 = compute_file_sha256(csvs[0])
                raw_hash_status = "verified"
            else:
                raw_hash_status = "provider_managed"
        elif spec.source_provider == "openml":
            raw_sha256 = None
            raw_hash_status = "provider_managed"
        elif spec.source_provider == "sklearn":
            raw_sha256 = None
            raw_hash_status = "not_applicable"
        else:
            raw_sha256 = None
            raw_hash_status = "provider_managed"

        commit_sha = source_commit or get_git_commit()
        did = f"{spec.id}_{domain}" if domain else spec.id

        return DatasetManifestEntry(
            dataset_id=did,
            dataset_type=spec.dataset_group,
            source_provider=spec.source_provider,
            source_identifier=str(spec.source_id),
            source_version=str(spec.source_version),
            source_url=spec.source_url,
            retrieved_at=meta.get("retrieved_at", "phase_1_acquisition"),
            raw_sha256=raw_sha256,
            raw_hash_status=raw_hash_status,
            features_sha256=features_sha256,
            labels_sha256=labels_sha256,
            groups_sha256=groups_sha256,
            domains_sha256=domains_sha256,
            metadata_sha256=metadata_sha256,
            canonical_bundle_sha256=bundle_sha256,
            canonical_sha256=bundle_sha256,
            n_rows=meta.get("n_rows", 0),
            n_features=meta.get("n_features", 0),
            n_classes=meta.get("n_classes"),
            feature_names_hash=fnames_hash,
            target_name=spec.target_column,
            license=spec.license,
            status=status,
            generated_from_commit=commit_sha,
            manifest_schema_version=2,
            git_commit=commit_sha,
            extra={
                "domain_column": spec.domain_column,
                "subdomain": domain or meta.get("subdomain"),
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
        """Scan canonical and synthetic directories and write all Schema v2 manifest artifacts."""
        source_commit = get_git_commit()
        real_entries: List[Dict[str, Any]] = []
        unavailable_entries: List[Dict[str, Any]] = []

        for spec in specs:
            if spec.dataset_group == "controlled_real":
                tdir = self.canonical_dir / "controlled" / spec.id
                entry = self.create_real_manifest_entry(spec, tdir, source_commit=source_commit)
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
                        entry = self.create_real_manifest_entry(spec, ddir, domain=d, source_commit=source_commit)
                        if entry:
                            found_any = True
                            real_entries.append(entry.__dict__)
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
                    entry = self.create_real_manifest_entry(spec, task_dir, source_commit=source_commit)
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
                    hashes = compute_synthetic_bundle_hashes(sdir)
                    mpath = sdir / "metadata.json"
                    with open(mpath, "r", encoding="utf-8") as f:
                        smeta = json.load(f)

                    entry = SyntheticManifestEntry(
                        family_id=sdir.name,
                        generator=smeta.get("generator", ""),
                        generator_seed=smeta.get("generator_seed", 0),
                        K=smeta.get("K", 0),
                        n_rows=smeta.get("n_rows", 0),
                        n_features=smeta.get("n_features", 0),
                        features_sha256=hashes["features_sha256"] or "",
                        hard_labels_sha256=hashes["hard_labels_sha256"] or "",
                        soft_memberships_sha256=hashes["soft_memberships_sha256"] or "",
                        parameters_sha256=hashes["parameters_sha256"] or "",
                        metadata_sha256=hashes["metadata_sha256"] or "",
                        canonical_bundle_sha256=hashes["canonical_bundle_sha256"] or "",
                        canonical_sha256=hashes["canonical_bundle_sha256"] or "",
                        soft_truth_available=True,
                        generated_from_commit=source_commit,
                        manifest_schema_version=2,
                        git_commit=source_commit,
                    )
                    synthetic_entries.append(entry.__dict__)

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
            if isinstance(download_results, dict):
                download_results["manifest_schema_version"] = 2
                download_results["generated_from_commit"] = source_commit
            with open(self.manifest_dir / "download_report.json", "w", encoding="utf-8") as f:
                json.dump(download_results, f, indent=2)

        return {
            "real_entries_count": len(real_entries),
            "synthetic_entries_count": len(synthetic_entries),
            "unavailable_entries_count": len(unavailable_entries),
            "generated_from_commit": source_commit,
            "manifest_schema_version": 2,
        }
