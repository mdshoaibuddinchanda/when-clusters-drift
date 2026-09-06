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


ALLOWED_FEATURE_ROLES = frozenset({"numeric", "categorical", "ordinal", "boolean"})


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
    expected_rows: Optional[int] = None
    expected_features: Optional[int] = None
    expected_classes: Optional[int] = None
    expected_source_name: Optional[str] = None
    expected_target: Optional[str] = None
    has_missing_values: bool = False
    categorical_features: List[str] = field(default_factory=list)
    notes: str = ""
    domain_metadata: Dict[str, Any] = field(default_factory=dict)
    split_strategy: str = "kfold"
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    feature_roles: Dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    """Standardized representation of an in-memory dataset before serialization."""
    X: pd.DataFrame
    y: Optional[pd.Series] = None
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain_labels: Optional[pd.Series] = None
    groups: Optional[pd.DataFrame] = None
    domains: Optional[pd.DataFrame] = None
    feature_roles: Dict[str, str] = field(default_factory=dict)


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
    """Auditable provenance record for a real-world dataset artifact (Schema v2)."""
    dataset_id: str
    dataset_type: str
    source_provider: str
    source_identifier: str
    source_version: str
    source_url: Optional[str]
    retrieved_at: str
    raw_sha256: Optional[str]
    raw_hash_status: str
    features_sha256: str
    labels_sha256: str
    groups_sha256: Optional[str]
    domains_sha256: Optional[str]
    metadata_sha256: str
    canonical_bundle_sha256: str
    canonical_sha256: str  # Backward-compatible alias of canonical_bundle_sha256
    n_rows: int
    n_features: int
    n_classes: Optional[int]
    feature_names_hash: str
    target_name: Optional[str]
    license: str
    status: str
    generated_from_commit: str
    manifest_schema_version: int = 2
    git_commit: Optional[str] = None  # Deprecated backward-compatible alias of generated_from_commit
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.git_commit is None:
            self.git_commit = self.generated_from_commit
        if not self.canonical_sha256:
            self.canonical_sha256 = self.canonical_bundle_sha256


@dataclass
class SyntheticManifestEntry:
    """Auditable provenance record for a synthetic benchmark family (Schema v2)."""
    family_id: str
    generator: str
    generator_seed: int
    K: int
    n_rows: int
    n_features: int
    features_sha256: str
    hard_labels_sha256: str
    soft_memberships_sha256: str
    parameters_sha256: str
    metadata_sha256: str
    canonical_bundle_sha256: str
    canonical_sha256: str  # Backward-compatible alias of canonical_bundle_sha256
    soft_truth_available: bool = True
    generated_from_commit: str = ""
    manifest_schema_version: int = 2
    git_commit: Optional[str] = None  # Deprecated backward-compatible alias of generated_from_commit
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.git_commit is None:
            self.git_commit = self.generated_from_commit
        if not self.canonical_sha256:
            self.canonical_sha256 = self.canonical_bundle_sha256


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
