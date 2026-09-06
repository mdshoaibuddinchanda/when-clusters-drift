"""Data schemas and specifications for the dataset acquisition and validation pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd


class DatasetGroup(str, Enum):
    CONTROLLED_REAL = "controlled_real"
    NATURAL_SHIFT = "natural_shift"
    SYNTHETIC = "synthetic"


class DownloadMode(str, Enum):
    AUTO = "auto"
    AUTH_REQUIRED = "auth_required"
    MANUAL_LICENSE_ACCEPTANCE = "manual_license_acceptance"
    UNAVAILABLE = "unavailable"


class DownloadStatus(str, Enum):
    OK = "ok"
    ALREADY_EXISTS = "already_exists"
    AUTH_REQUIRED = "auth_required"
    MANUAL_LICENSE_ACCEPTANCE = "manual_license_acceptance"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable specification for a real-world dataset."""
    id: str
    display_name: str
    dataset_group: str
    source_provider: str
    source_type: str
    source_url: Optional[str]
    source_id: Optional[str]
    source_version: Optional[str]
    download_mode: str
    license: str
    citation: str
    target_column: Optional[str]
    domain_column: Optional[str] = None
    id_columns: List[str] = field(default_factory=list)
    drop_columns: List[str] = field(default_factory=list)
    expected_min_rows: int = 10
    expected_min_features: int = 2
    expected_classes: Optional[int] = None
    has_missing_values: bool = False
    categorical_features: List[str] = field(default_factory=list)
    notes: str = ""
    domain_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    """Standardized representation of an in-memory dataset before serialization."""
    X: pd.DataFrame
    y: Optional[pd.Series] = None
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain_labels: Optional[pd.Series] = None


@dataclass
class DownloadResult:
    """Outcome of attempting to acquire/canonicalize a dataset."""
    dataset_id: str
    status: DownloadStatus
    message: str
    n_rows: int = 0
    n_features: int = 0
    n_classes: Optional[int] = None
    canonical_files: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class SyntheticDatasetSpec:
    """Specification for a synthetic benchmark generator family."""
    family_id: str
    generator: str
    generator_seed: int
    K: int
    n: int
    d: int
    notes: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetManifestEntry:
    """Auditable provenance record for a dataset artifact."""
    dataset_id: str
    dataset_type: str
    source_provider: str
    source_identifier: str
    source_version: str
    source_url: Optional[str]
    retrieved_at: str
    raw_sha256: Optional[str]
    canonical_sha256: str
    n_rows: int
    n_features: int
    n_classes: Optional[int]
    feature_names_hash: str
    target_name: Optional[str]
    license: str
    status: str
    git_commit: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Comprehensive validation report for a dataset."""
    dataset_id: str
    is_valid: bool
    rows: int
    features: int
    classes: Optional[int]
    missing_count: int
    missing_fraction: float
    duplicate_rows: int
    constant_features: List[str]
    checks_passed: List[str]
    warnings: List[str]
    errors: List[str]
