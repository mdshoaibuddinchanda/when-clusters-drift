"""Rebuild the Y1 within-dataset versus cross-dataset comparison table.

The expensive grouped within-dataset fits are produced by
``hypothesis_y1_model_sensitivities.py`` and retained at fold level.  This
dedicated script independently aggregates those retained results and joins the
frozen LODO dataset metrics; it never refits or selects features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from clusterdrift.shifts.hashing import atomic_write_csv, atomic_write_json


BASE = ROOT / "experiments/hypothesis_y1"
ARTIFACTS = BASE / "artifacts"
BLOCKS = ["P0", "P2", "P3", "P4", "B0_median"]


def main() -> None:
    detail_path = ARTIFACTS / "hypothesis_y1_within_dataset_metrics.csv"
    if not detail_path.exists():
        raise FileNotFoundError(
            "Run hypothesis_y1_model_sensitivities.py once to generate grouped fold-level fits"
        )
    detail = pd.read_csv(detail_path)
    required = {"dataset", "heldout_outer_fold", "feature_block", "mae"}
    if not required.issubset(detail.columns):
        raise ValueError(f"Within-dataset artifact lacks columns: {sorted(required - set(detail.columns))}")
    duplicated = detail.duplicated(["dataset", "heldout_outer_fold", "feature_block"])
    if duplicated.any():
        raise ValueError(f"Duplicate within-dataset fold/block records: {int(duplicated.sum())}")
    counts = detail.groupby(["dataset", "feature_block"])["heldout_outer_fold"].nunique()
    if not (counts == 5).all():
        raise ValueError("Every dataset/block must contain exactly five held-out outer-fold results")

    means = detail.groupby(["dataset", "feature_block"], sort=False)["mae"].mean().unstack()
    missing = set(BLOCKS) - set(means.columns)
    if missing:
        raise ValueError(f"Missing preregistered/control blocks: {sorted(missing)}")
    comparison = pd.DataFrame({"dataset": means.index})
    for block in BLOCKS:
        comparison[f"within_{block}"] = means[block].to_numpy()
    comparison["within_delta04"] = comparison["within_P0"] - comparison["within_P4"]

    frozen = pd.read_csv(ROOT / "results/falsification/lodo_dataset_metrics.csv")
    frozen = frozen.rename(columns={"dataset_id": "dataset", "mae_P0": "LODO_P0", "mae_P4": "LODO_P4"})
    comparison = comparison.merge(frozen[["dataset", "LODO_P0", "LODO_P4"]], on="dataset", how="left", validate="one_to_one")
    if comparison[["LODO_P0", "LODO_P4"]].isna().any().any():
        raise ValueError("Frozen LODO comparison is incomplete")
    comparison["LODO_delta04"] = comparison["LODO_P0"] - comparison["LODO_P4"]

    output = ARTIFACTS / "hypothesis_y1_within_vs_cross_dataset.csv"
    atomic_write_csv(output, comparison, index=False)
    summary = {
        "label": "POST-HOC WITHIN-VS-CROSS DATASET COMPARISON; NOT CONFIRMATORY",
        "within_structural_help_count": int((comparison["within_delta04"] > 0).sum()),
        "cross_structural_help_count": int((comparison["LODO_delta04"] > 0).sum()),
        "same_direction_count": int(((comparison["within_delta04"] > 0) == (comparison["LODO_delta04"] > 0)).sum()),
        "both_help_datasets": comparison.loc[
            (comparison["within_delta04"] > 0) & (comparison["LODO_delta04"] > 0), "dataset"
        ].tolist(),
        "grouping": "outer fold held out; inner groups outer_fold|condition so all five seeds of a scenario remain together",
    }
    atomic_write_json(ARTIFACTS / "hypothesis_y1_within_vs_cross_dataset.json", summary, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
