from pathlib import Path
import pytest
import yaml

def test_hyperparameter_grid_freeze():
    root = Path(__file__).resolve().parents[2]
    cfg_p = root / "configs" / "falsification.yaml"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hgbr_grid = cfg["primary_regressor"]["grid"]
    assert hgbr_grid["learning_rate"] == [0.05, 0.10]
    assert hgbr_grid["max_leaf_nodes"] == [7, 15]
    assert hgbr_grid["max_iter"] == [150, 300]
    assert hgbr_grid["l2_regularization"] == [0.0, 1.0]

    ridge_alphas = cfg["linear_control"]["alphas"]
    assert ridge_alphas == [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]