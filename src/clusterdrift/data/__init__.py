"""ClusterDrift Data Module: Acquisition, Canonicalization, Synthetic Generation, and Validation."""

from clusterdrift.data.canonicalize import canonicalize_bundle
from clusterdrift.data.downloader import DatasetDownloader
from clusterdrift.data.manifest import ManifestManager, compute_file_sha256
from clusterdrift.data.folds import (
    FoldSplit,
    InnerFold,
    compute_preprocessing_config_sha256,
    compute_scenario_input_sha256,
    compute_split_hash,
    compute_split_protocol_sha256,
    generate_group_kfold_splits,
    generate_kfold_splits,
    generate_tableshift_natural_splits,
    generate_temporal_block_splits,
    generate_whyshift_natural_splits,
    save_fold_split_artifacts,
    verify_split_artifact,
)
from clusterdrift.data.preprocess import (
    SourceOnlyPreprocessor,
    build_preprocessor,
)
from clusterdrift.data.registry import (
    get_dataset_spec,
    get_synthetic_spec,
    list_by_provider,
    list_controlled_real,
    list_datasets,
    list_natural_shift,
    list_synthetic,
    load_registry,
)
from clusterdrift.data.schemas import (
    DatasetBundle,
    DatasetGroup,
    DatasetManifestEntry,
    DatasetSpec,
    DownloadMode,
    DownloadResult,
    DownloadStatus,
    SyntheticDatasetSpec,
    ValidationResult,
)
from clusterdrift.data.synthetic import generate_synthetic_family, save_synthetic_dataset
from clusterdrift.data.validation import DataValidator

__all__ = [
    "DatasetSpec",
    "SyntheticDatasetSpec",
    "DatasetBundle",
    "DatasetManifestEntry",
    "DownloadResult",
    "ValidationResult",
    "DatasetGroup",
    "DownloadMode",
    "DownloadStatus",
    "load_registry",
    "get_dataset_spec",
    "get_synthetic_spec",
    "list_datasets",
    "list_controlled_real",
    "list_natural_shift",
    "list_synthetic",
    "list_by_provider",
    "canonicalize_bundle",
    "DatasetDownloader",
    "save_synthetic_dataset",
    "generate_synthetic_family",
    "ManifestManager",
    "DataValidator",
    "compute_file_sha256",
    "FoldSplit",
    "InnerFold",
    "compute_split_hash",
    "compute_split_protocol_sha256",
    "compute_preprocessing_config_sha256",
    "compute_scenario_input_sha256",
    "generate_kfold_splits",
    "generate_group_kfold_splits",
    "generate_temporal_block_splits",
    "generate_whyshift_natural_splits",
    "generate_tableshift_natural_splits",
    "save_fold_split_artifacts",
    "verify_split_artifact",
    "SourceOnlyPreprocessor",
    "build_preprocessor",
]

