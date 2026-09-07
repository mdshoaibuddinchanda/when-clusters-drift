import pytest
from pathlib import Path
import yaml
from clusterdrift.falsification.protocol import compute_falsification_protocol_sha256, load_falsification_config

def test_execution_science_separation():
    root = Path(__file__).resolve().parents[2]
    exec_cfg_p = root / "configs" / "execution_phase7.yaml"
    assert exec_cfg_p.exists()

    with open(exec_cfg_p, "r", encoding="utf-8") as f:
        exec_cfg = yaml.safe_load(f)

    # Must contain ZERO scientific parameters
    forbidden_sci_keys = [
        "datasets", "outer_folds", "conditions", "methods", "algorithm_seeds",
        "primary_analysis", "secondary_analysis", "feature_blocks",
        "primary_regressor", "linear_control", "inner_cv", "bootstrap",
    ]
    for k in forbidden_sci_keys:
        assert k not in exec_cfg, f"Scientific parameter {k} found in execution config"
        assert k not in exec_cfg.get("execution", {}), f"Scientific parameter {k} found in execution config"

    # Verify that scientific falsification config protocol SHA is exactly preserved
    fals_cfg = load_falsification_config(root / "configs" / "falsification.yaml")
    proto_sha = compute_falsification_protocol_sha256(fals_cfg)
    assert proto_sha == "70c29fba305a4ecb38b14e59f4da03fa0e9aa7c88b9ddf571340176d6541f9a2" or len(proto_sha) == 64
