from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest

def test_thread_execution_determinism():
    def compute_task(seed: int):
        rng = np.random.RandomState(seed)
        return float(np.sum(rng.randn(100)))

    seeds = [1, 2, 3, 4, 5]
    seq_results = [compute_task(s) for s in seeds]

    with ThreadPoolExecutor(max_workers=2) as ex:
        par_results = list(ex.map(compute_task, seeds))

    assert seq_results == par_results