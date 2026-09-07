from pathlib import Path
import pytest
import yaml

from clusterdrift.falsification.protocol import (
    P4_STRUCTURAL_ABLATIONS,
    PREDEFINED_FEATURE_BLOCKS,
    SINGLE_SIGNAL_INCREMENTAL,
)


def test_predefined_feature_blocks_exact():
    assert PREDEFINED_FEATURE_BLOCKS["P0"] == ["D_X"]
    assert PREDEFINED_FEATURE_BLOCKS["P1"] == ["FPC", "PE_norm", "XB_soft_m2", "silhouette"]
    assert PREDEFINED_FEATURE_BLOCKS["P2"] == ["D_U_R", "D_U_C", "D_V", "D_H", "D_M"]
    assert PREDEFINED_FEATURE_BLOCKS["P3"] == ["D_X", "FPC", "PE_norm", "XB_soft_m2", "silhouette"]
    assert PREDEFINED_FEATURE_BLOCKS["P4"] == ["D_X", "D_U_R", "D_U_C", "D_V", "D_H", "D_M"]
    assert PREDEFINED_FEATURE_BLOCKS["P5"] == [
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
    ]


def test_p4_structural_ablations_exact():
    assert len(P4_STRUCTURAL_ABLATIONS) == 5
    for k, features in P4_STRUCTURAL_ABLATIONS.items():
        removed_sig = k.replace("P4_minus_", "")
        assert removed_sig not in features
        assert len(features) == 5
        assert "D_X" in features


def test_single_signal_incremental_exact():
    assert len(SINGLE_SIGNAL_INCREMENTAL) == 5
    for k, features in SINGLE_SIGNAL_INCREMENTAL.items():
        added_sig = k.replace("DX_plus_", "")
        assert features == ["D_X", added_sig]


def test_config_yaml_matches_protocol():
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "configs" / "falsification.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["feature_blocks"] == PREDEFINED_FEATURE_BLOCKS