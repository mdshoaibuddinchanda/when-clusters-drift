"""Test that parallel probe selection is identical to sequential probe selection."""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from clusterdrift.probes.hashing import derive_reference_probe_seed
from clusterdrift.probes.selection import select_reference_probe_indices


def test_parallel_vs_sequential_probe_selection():
    """Independent tasks in ThreadPoolExecutor produce bitwise identical probe banks."""
    global_seed = 2026090705
    datasets = ["iris", "glass", "wine", "sonar", "ecoli"]
    folds = [0, 1, 2, 3, 4]

    tasks = [(ds, f) for ds in datasets for f in folds]

    def run_task(ds, f):
        seed = derive_reference_probe_seed(global_seed, ds, f, f"split_{ds}_{f}")
        return select_reference_probe_indices(3000, 2048, seed)

    # Sequential
    seq_results = [run_task(ds, f) for ds, f in tasks]

    # ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        thr_results = list(executor.map(lambda t: run_task(t[0], t[1]), tasks))

    for s_arr, t_arr in zip(seq_results, thr_results):
        np.testing.assert_array_equal(s_arr, t_arr)
