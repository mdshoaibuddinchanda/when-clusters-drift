import pytest
import numpy as np
from pathlib import Path
from clusterdrift.falsification.execution.cache import PersistentPhase7Cache

def test_cache_corruption_detection(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cache = PersistentPhase7Cache(cache_dir=tmp_path, protocol_sha="proto123", project_root=root)

    # 1. Put valid source data
    X_src = np.ones((50, 4), dtype=np.float64)
    A_R = np.ones((10, 4), dtype=np.float64)
    cache.put_source_data("iris", 0, "fp1", X_src, A_R)

    res = cache.get_source_data("iris", 0, "fp1")
    assert res is not None

    # Stale fingerprint
    res_stale = cache.get_source_data("iris", 0, "fp2")
    assert res_stale is None

    del res
    import gc
    gc.collect()

    # 2. Corrupt file on disk
    x_file = cache.datasets_dir / "iris_fold_0_X_src.npy"
    x_file.write_bytes(b"corrupt array data")
    res_corrupt = cache.get_source_data("iris", 0, "fp1")
    assert res_corrupt is None
