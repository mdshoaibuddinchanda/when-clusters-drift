import pytest
from pathlib import Path
from clusterdrift.falsification.execution.resources import assert_not_frozen_data_path

def test_path_safety_guard():
    root = Path(__file__).resolve().parents[2]

    # Rejects project root directly
    with pytest.raises(ValueError, match="Path safety violation"):
        assert_not_frozen_data_path(root, root)

    # Rejects forbidden subdirectories
    for sub in ["data/canonical", "data/splits", "data/shifts", "data/probes", "data/synthetic", "results"]:
        target = root / sub
        with pytest.raises(ValueError, match="Path safety violation"):
            assert_not_frozen_data_path(target, root)

        target_child = root / sub / "some_nested_dir"
        with pytest.raises(ValueError, match="Path safety violation"):
            assert_not_frozen_data_path(target_child, root)

    # Rejects filesystem root
    with pytest.raises(ValueError, match="Path safety violation"):
        assert_not_frozen_data_path("/", root)

    # Accepts external directory
    safe_dir = Path.home() / ".cache" / "test_clusterdrift_safe"
    res = assert_not_frozen_data_path(safe_dir, root)
    assert res == safe_dir.resolve()
