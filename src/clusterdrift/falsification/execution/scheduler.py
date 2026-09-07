"""Longest-Processing-Time (LPT) Task Scheduler and Dataset-Fold Affinity Management."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd


def compute_task_complexity(dataset_id: str, project_root: Path) -> int:
    """Compute task complexity proxy cost proportional to N x D x K."""
    ds_dir = project_root / "data" / "canonical" / "controlled" / dataset_id
    feat_p = ds_dir / "features.parquet"
    if not feat_p.exists():
        return 1

    df_X = pd.read_parquet(feat_p)
    N, D = df_X.shape

    # Read n_classes from metadata
    meta_p = ds_dir / "metadata.json"
    K = 2
    if meta_p.exists():
        try:
            import json
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            K = meta.get("n_classes", 2)
        except Exception:
            pass

    return int(N * D * K)


def sort_tasks_lpt(
    datasets: List[str],
    folds: List[int],
    project_root: Path,
    measured_runtimes: Dict[str, float] = None,
) -> List[Tuple[str, int]]:
    """Order dataset-fold tasks Longest-Processing-Time (LPT) first."""
    tasks = [(ds, fold) for ds in datasets for fold in folds]

    if measured_runtimes:
        def sort_key(t: Tuple[str, int]):
            return measured_runtimes.get(t[0], 0.0)
    else:
        complexity_cache = {ds: compute_task_complexity(ds, project_root) for ds in datasets}

        def sort_key(t: Tuple[str, int]):
            return complexity_cache.get(t[0], 0)

    # Sort descending: largest / slowest tasks first
    tasks.sort(key=sort_key, reverse=True)
    return tasks