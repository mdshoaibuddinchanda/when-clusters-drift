"""Tests for frozen shift replay layer (clusterdrift.shifts.replay)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from clusterdrift.shifts.engine import ShiftEngine
from clusterdrift.shifts.replay import (
    build_local_overlap_replay_descriptor,
    replay_frozen_shift,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_clean_replay():
    """Verify clean replay returns exact copy with identity row index map."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    spec_p = PROJECT_ROOT / "data" / "shifts" / "specs" / "iris" / "fold_0" / "clean.json"
    res = replay_frozen_shift(
        X_target_raw=df,
        dataset_id="iris",
        outer_fold=0,
        condition="clean",
        phase4_spec_path=spec_p,
        project_root=PROJECT_ROOT,
    )
    pd.testing.assert_frame_equal(res.X_shifted, df)
    assert np.array_equal(res.row_index_map, np.arange(3))


def test_class_prevalence_replay_no_labels():
    """Verify class prevalence replays from frozen NPZ with zero label access."""
    fold_p = PROJECT_ROOT / "data" / "splits" / "controlled" / "iris" / "fold_0.npz"
    with np.load(fold_p) as npz:
        tgt_idx = npz["target_indices"]
    df_X = pd.read_parquet(PROJECT_ROOT / "data" / "canonical" / "controlled" / "iris" / "features.parquet")
    X_tgt = df_X.iloc[tgt_idx].reset_index(drop=True)

    spec_p = PROJECT_ROOT / "data" / "shifts" / "specs" / "iris" / "fold_0" / "class_prevalence_severe.json"
    res = replay_frozen_shift(
        X_target_raw=X_tgt,
        dataset_id="iris",
        outer_fold=0,
        condition="class_prevalence_severe",
        phase4_spec_path=spec_p,
        project_root=PROJECT_ROOT,
    )
    assert len(res.X_shifted) == len(X_tgt)
    assert res.metadata["family"] == "class_prevalence"
    assert len(res.row_index_map) == len(X_tgt)


def test_local_overlap_descriptor_and_replay():
    """Verify local overlap descriptor generation and subsequent label-free replay."""
    ds = "iris"
    fold = 0
    cond = "local_overlap_severe"
    ds_dir = PROJECT_ROOT / "data" / "canonical" / "controlled" / ds
    df_X = pd.read_parquet(ds_dir / "features.parquet")
    df_y = pd.read_parquet(ds_dir / "labels.parquet")
    fold_p = PROJECT_ROOT / "data" / "splits" / "controlled" / ds / f"fold_{fold}.npz"
    with np.load(fold_p) as npz:
        X_src = df_X.iloc[npz["source_indices"]].reset_index(drop=True)
        X_tgt = df_X.iloc[npz["target_indices"]].reset_index(drop=True)
        y_src = df_y.iloc[npz["source_indices"]].to_numpy().ravel()
        y_tgt = df_y.iloc[npz["target_indices"]].to_numpy().ravel()

    with open(ds_dir / "metadata.json") as f:
        meta = json.load(f)
    roles = meta.get("feature_roles", {c: "numeric" for c in X_src.columns})
    spec_p = PROJECT_ROOT / "data" / "shifts" / "specs" / ds / f"fold_{fold}" / f"{cond}.json"

    # 1. Build descriptor offline with labels
    desc = build_local_overlap_replay_descriptor(
        X_target_raw=X_tgt,
        y_target_raw=y_tgt,
        X_source_raw=X_src,
        y_source_raw=y_src,
        roles=roles,
        dataset_id=ds,
        outer_fold=fold,
        condition=cond,
        phase4_spec_path=spec_p,
        project_root=PROJECT_ROOT,
    )
    assert "replay_descriptor_sha256" in desc
    assert len(desc["selected_target_positions"]) > 0

    # 2. Replay using descriptor without labels
    res = replay_frozen_shift(
        X_target_raw=X_tgt,
        dataset_id=ds,
        outer_fold=fold,
        condition=cond,
        phase4_spec_path=spec_p,
        replay_descriptor=desc,
        project_root=PROJECT_ROOT,
    )
    assert len(res.X_shifted) == len(X_tgt)
    assert res.metadata["family"] == "local_overlap"
