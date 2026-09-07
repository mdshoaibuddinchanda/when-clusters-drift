"""Signal Vector Representations and Predictor Block Definitions.

Defines:
    Primary signal vector:
    Z_t = [D_U^R, D_U^C, D_V, D_H, D_M, D_X]
    
    Frozen predictor blocks for Phase 7:
    P0 : Covariate shift baseline [D_X]
    P1 : Conventional validity controls [FPC, PE_norm, XB_soft_m2, silhouette]
    P2 : Structural signals only [D_U_R, D_U_C, D_V, D_H, D_M]
    P3 : Covariate + Conventional (P0 + P1)
    P4 : Covariate + Structural (P0 + P2)
    P5 : Full hybrid model (P0 + P1 + P2)
"""

from typing import Dict, List
import numpy as np

PRIMARY_SIGNAL_NAMES: List[str] = [
    "D_U_R",
    "D_U_C",
    "D_V",
    "D_H",
    "D_M",
    "D_X",
]

PREDICTOR_BLOCKS: Dict[str, List[str]] = {
    "P0": ["D_X"],
    "P1": ["FPC", "PE_norm", "XB_soft_m2", "silhouette"],
    "P2": ["D_U_R", "D_U_C", "D_V", "D_H", "D_M"],
    "P3": ["D_X", "FPC", "PE_norm", "XB_soft_m2", "silhouette"],
    "P4": ["D_X", "D_U_R", "D_U_C", "D_V", "D_H", "D_M"],
    "P5": ["D_X", "FPC", "PE_norm", "XB_soft_m2", "silhouette", "D_U_R", "D_U_C", "D_V", "D_H", "D_M"],
}


def extract_primary_signal_vector(record: Dict[str, float]) -> np.ndarray:
    """Extract primary 6-dimensional signal vector Z_t from record dict."""
    return np.array([float(record[name]) for name in PRIMARY_SIGNAL_NAMES], dtype=np.float64)


def extract_predictor_features(record: Dict[str, float], block_name: str) -> np.ndarray:
    """Extract named feature block for risk prediction modeling."""
    if block_name not in PREDICTOR_BLOCKS:
        raise ValueError(f"Unknown predictor block: {block_name}. Valid: {list(PREDICTOR_BLOCKS.keys())}")
    names = PREDICTOR_BLOCKS[block_name]
    return np.array([float(record[n]) for n in names], dtype=np.float64)
