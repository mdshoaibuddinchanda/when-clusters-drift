"""Pre-Registered Protocol Definitions for Phase 7 Structural Falsification Pilot."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

from clusterdrift.signals.hashing import compute_canonical_dict_sha256


CONDITION_TO_FAMILY: Dict[str, Tuple[str, str]] = {
    "clean": ("clean", "none"),
    "location_mild": ("location", "mild"),
    "location_severe": ("location", "severe"),
    "scale_mild": ("scale", "mild"),
    "scale_severe": ("scale", "severe"),
    "mcar_mild": ("mcar", "mild"),
    "mcar_severe": ("mcar", "severe"),
    "outliers_mild": ("outliers", "mild"),
    "outliers_severe": ("outliers", "severe"),
    "measurement_noise_mild": ("measurement_noise", "mild"),
    "measurement_noise_severe": ("measurement_noise", "severe"),
    "class_prevalence_mild": ("class_prevalence", "mild"),
    "class_prevalence_severe": ("class_prevalence", "severe"),
    "local_overlap_mild": ("local_overlap", "mild"),
    "local_overlap_severe": ("local_overlap", "severe"),
}

ALL_SHIFT_FAMILIES: List[str] = [
    "location",
    "scale",
    "mcar",
    "outliers",
    "measurement_noise",
    "class_prevalence",
    "local_overlap",
]

PREDEFINED_FEATURE_BLOCKS: Dict[str, List[str]] = {
    "P0": ["D_X"],
    "P1": ["FPC", "PE_norm", "XB_soft_m2", "silhouette"],
    "P2": ["D_U_R", "D_U_C", "D_V", "D_H", "D_M"],
    "P3": ["D_X", "FPC", "PE_norm", "XB_soft_m2", "silhouette"],
    "P4": ["D_X", "D_U_R", "D_U_C", "D_V", "D_H", "D_M"],
    "P5": [
        "D_X",
        "FPC",
        "PE_norm",
        "XB_soft_m2",
        "silhouette",
        "D_U_R",
        "D_U_C",
        "D_V",
        "D_H",
        "D_M",
    ],
}

# P4 Leave-One-Structural-Signal-Out Ablations
P4_STRUCTURAL_ABLATIONS: Dict[str, List[str]] = {
    "P4_minus_D_U_R": ["D_X", "D_U_C", "D_V", "D_H", "D_M"],
    "P4_minus_D_U_C": ["D_X", "D_U_R", "D_V", "D_H", "D_M"],
    "P4_minus_D_V": ["D_X", "D_U_R", "D_U_C", "D_H", "D_M"],
    "P4_minus_D_H": ["D_X", "D_U_R", "D_U_C", "D_V", "D_M"],
    "P4_minus_D_M": ["D_X", "D_U_R", "D_U_C", "D_V", "D_H"],
}

# Single-Signal Incremental Diagnostics
SINGLE_SIGNAL_INCREMENTAL: Dict[str, List[str]] = {
    "DX_plus_D_U_R": ["D_X", "D_U_R"],
    "DX_plus_D_U_C": ["D_X", "D_U_C"],
    "DX_plus_D_V": ["D_X", "D_V"],
    "DX_plus_D_H": ["D_X", "D_H"],
    "DX_plus_D_M": ["D_X", "D_M"],
}


def load_falsification_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the frozen Phase-7 configuration."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Falsification config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def compute_falsification_protocol_sha256(cfg: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash of the pre-registered falsification protocol."""
    keys_to_bind = [
        "protocol_version",
        "falsification_version",
        "global_seed",
        "datasets",
        "outer_folds",
        "conditions",
        "methods",
        "algorithm_seeds",
        "primary_analysis",
        "secondary_analysis",
        "feature_blocks",
        "primary_regressor",
        "linear_control",
        "inner_cv",
        "bootstrap",
    ]
    sub_dict = {k: cfg[k] for k in keys_to_bind if k in cfg}
    return compute_canonical_dict_sha256(sub_dict)
