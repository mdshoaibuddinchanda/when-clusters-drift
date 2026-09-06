"""Dataset registry module for loading and validating dataset specifications."""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from clusterdrift.data.schemas import DatasetSpec, SyntheticDatasetSpec


_REAL_REGISTRY: Dict[str, DatasetSpec] = {}
_SYNTHETIC_REGISTRY: Dict[str, SyntheticDatasetSpec] = {}
_CONFIG_LOADED: bool = False


def _find_config_path() -> Path:
    """Find the path to datasets.yaml relative to project root."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        cand = parent / "configs" / "datasets.yaml"
        if cand.exists():
            return cand
    # Fallback to working directory
    cand = Path.cwd() / "configs" / "datasets.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError("Could not locate configs/datasets.yaml")


def load_registry(config_path: Optional[Path] = None, force_reload: bool = False) -> None:
    """Load and validate the dataset specifications from YAML config."""
    global _REAL_REGISTRY, _SYNTHETIC_REGISTRY, _CONFIG_LOADED

    if _CONFIG_LOADED and not force_reload:
        return

    path = config_path or _find_config_path()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    real_data = data.get("real_datasets", {})
    synthetic_data = data.get("synthetic_datasets", {})

    real_reg: Dict[str, DatasetSpec] = {}
    for ds_id, spec in real_data.items():
        if ds_id in real_reg:
            raise ValueError(f"Duplicate real dataset slug found: {ds_id}")
        real_reg[ds_id] = DatasetSpec(
            id=spec["id"],
            display_name=spec.get("display_name", ds_id),
            dataset_group=spec["dataset_group"],
            source_provider=spec["source_provider"],
            source_type=spec["source_type"],
            source_url=spec.get("source_url"),
            source_id=spec.get("source_id"),
            source_version=spec.get("source_version"),
            download_mode=spec.get("download_mode", "auto"),
            license=spec.get("license", "unknown"),
            citation=spec.get("citation", ""),
            target_column=spec.get("target_column"),
            domain_column=spec.get("domain_column"),
            id_columns=spec.get("id_columns", []),
            drop_columns=spec.get("drop_columns", []),
            expected_min_rows=spec.get("expected_min_rows", 10),
            expected_min_features=spec.get("expected_min_features", 2),
            expected_rows=spec.get("expected_rows"),
            expected_features=spec.get("expected_features"),
            expected_classes=spec.get("expected_classes"),
            expected_source_name=spec.get("expected_source_name"),
            expected_target=spec.get("expected_target"),
            has_missing_values=spec.get("has_missing_values", False),
            categorical_features=spec.get("categorical_features", []),
            notes=spec.get("notes", ""),
            domain_metadata=spec.get("domain_metadata", {}),
            split_strategy=spec.get("split_strategy", "kfold"),
            group_column=spec.get("group_column"),
            time_column=spec.get("time_column"),
            feature_roles=spec.get("feature_roles", {}),
        )

    syn_reg: Dict[str, SyntheticDatasetSpec] = {}
    for syn_id, spec in synthetic_data.items():
        if syn_id in syn_reg:
            raise ValueError(f"Duplicate synthetic family slug found: {syn_id}")
        syn_reg[syn_id] = SyntheticDatasetSpec(
            family_id=spec["family_id"],
            generator=spec["generator"],
            generator_seed=spec["generator_seed"],
            K=spec["K"],
            n=spec["n"],
            d=spec["d"],
            notes=spec.get("notes", ""),
            parameters=spec.get("parameters", {}),
        )

    _REAL_REGISTRY = real_reg
    _SYNTHETIC_REGISTRY = syn_reg
    _CONFIG_LOADED = True


def get_dataset_spec(name: str) -> DatasetSpec:
    """Retrieve an immutable DatasetSpec by slug."""
    load_registry()
    if name not in _REAL_REGISTRY:
        raise KeyError(f"Dataset '{name}' not found in real datasets registry. Available: {list(_REAL_REGISTRY.keys())}")
    return _REAL_REGISTRY[name]


def get_synthetic_spec(name: str) -> SyntheticDatasetSpec:
    """Retrieve an immutable SyntheticDatasetSpec by family slug."""
    load_registry()
    if name not in _SYNTHETIC_REGISTRY:
        raise KeyError(f"Synthetic dataset '{name}' not found. Available: {list(_SYNTHETIC_REGISTRY.keys())}")
    return _SYNTHETIC_REGISTRY[name]


def list_datasets() -> List[str]:
    """List all registered real dataset slugs (40 datasets)."""
    load_registry()
    return list(_REAL_REGISTRY.keys())


def list_controlled_real() -> List[str]:
    """List all 30 controlled real dataset slugs."""
    load_registry()
    return [k for k, v in _REAL_REGISTRY.items() if v.dataset_group == "controlled_real"]


def list_natural_shift() -> List[str]:
    """List all 10 natural-shift real dataset slugs."""
    load_registry()
    return [k for k, v in _REAL_REGISTRY.items() if v.dataset_group == "natural_shift"]


def list_synthetic() -> List[str]:
    """List all 8 synthetic benchmark family slugs."""
    load_registry()
    return list(_SYNTHETIC_REGISTRY.keys())


def list_by_provider(provider: str) -> List[str]:
    """List real datasets matching a specific source provider."""
    load_registry()
    return [k for k, v in _REAL_REGISTRY.items() if v.source_provider == provider]
