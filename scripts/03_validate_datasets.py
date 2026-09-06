#!/usr/bin/env python3
"""Script 03: Validate all acquired canonical datasets and synthetic datasets, generating dataset_summary.csv."""

import sys
import time
from pathlib import Path
import pandas as pd

from clusterdrift.data import (
    DataValidator,
    ManifestManager,
    get_dataset_spec,
    list_controlled_real,
    list_datasets,
    list_natural_shift,
    list_synthetic,
)


def main():
    print("=== Validating All Datasets and Generating Comprehensive Manifests ===")
    t_start = time.time()

    validator = DataValidator()
    all_real_specs = [get_dataset_spec(s) for s in list_datasets()]

    # 1. Generate summary CSV
    print("Auditing canonical and synthetic datasets...")
    summary_csv_path = validator.generate_summary_csv(all_real_specs)
    df_summary = pd.read_csv(summary_csv_path)

    # 2. Update manifests
    manifest_mgr = ManifestManager()
    manifest_report = manifest_mgr.generate_all_manifests(all_real_specs)

    # 3. Print validation report
    available_real = df_summary[(df_summary["dataset_group"] != "synthetic") & (df_summary["status"] == "available")]
    controlled_avail = available_real[available_real["dataset_group"] == "controlled_real"]
    natural_records = available_real[available_real["dataset_group"] == "natural_shift"]

    # Deduplicate natural datasets by mapping back to registered natural dataset IDs
    natural_specs = [s for s in all_real_specs if s.dataset_group == "natural_shift"]
    natural_dataset_ids = set()
    for did in natural_records["dataset_id"]:
        for s in natural_specs:
            if did.startswith(s.id):
                natural_dataset_ids.add(s.id)
                break
        else:
            natural_dataset_ids.add(did)

    synthetic_avail = df_summary[df_summary["dataset_group"] == "synthetic"]
    total_rows = available_real["rows"].sum() + synthetic_avail["rows"].sum()

    print("\n" + "=" * 60)
    print("WHEN CLUSTERS DRIFT — DATASET INTEGRITY & VALIDATION REPORT")
    print("=" * 60)
    print(f"Registered Real Datasets:        40")
    print(f"  - Controlled Real Available:   {len(controlled_avail)} / 30")
    print(f"  - Natural Datasets Available:  {len(natural_dataset_ids)} / 10")
    print(f"  - Natural Domains/Partitions:  {len(natural_records)}")
    print(f"Synthetic Families Available:    {len(synthetic_avail)} / 8")
    print(f"Total Observations Acquired:     {total_rows:,}")
    print(f"Validation Duration:             {round(time.time() - t_start, 2)}s")
    print("-" * 60)
    print(f"Dataset Summary CSV:   {summary_csv_path}")
    print(f"Manifest JSON:         data/manifests/datasets.json")
    print(f"Synthetic Manifest:    data/manifests/synthetic_manifest.json")
    print(f"Unavailable Manifest:  data/manifests/unavailable_datasets.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
