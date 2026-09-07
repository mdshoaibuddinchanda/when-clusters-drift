from pathlib import Path
import pytest
import yaml

def test_clean_excluded_primary():
    root = Path(__file__).resolve().parents[2]
    cfg_p = root / "configs" / "falsification.yaml"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["primary_analysis"]["exclude_clean"] is True
    assert cfg["primary_analysis"]["target"] == "delta_ari"
    assert cfg["primary_analysis"]["error_metric"] == "mae"