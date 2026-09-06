#!/usr/bin/env python3
"""Script 02: Generate the eight controlled synthetic benchmark families with deterministic soft ground truth."""

import argparse
import sys
import time
from pathlib import Path
from typing import List
import numpy as np

from clusterdrift.data import (
    ManifestManager,
    get_synthetic_spec,
    list_datasets,
    list_synthetic,
    save_synthetic_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark families with exact soft memberships.")
    parser.add_argument("--all", action="store_true", help="Generate all eight synthetic benchmark families.")
    parser.add_argument("--dataset", type=str, default=None, help="Generate a specific synthetic family by slug.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if files already exist.")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify probabilistic constraints.")
    parser.add_argument("--dry-run", action="store_true", help="Preview synthetic families to be generated.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.all or args.dataset):
        print("Error: Specify --all or --dataset <slug>.")
        sys.exit(1)

    targets: List[str] = [args.dataset] if args.dataset else list_synthetic()
    print(f"=== Selected {len(targets)} synthetic benchmark families ===")

    if args.dry_run:
        print("\n[DRY RUN] Planned synthetic generation:")
        for t in targets:
            spec = get_synthetic_spec(t)
            print(f"  - {t:<30} K={spec.K} n={spec.n} d={spec.d} seed={spec.generator_seed}")
        return

    root = Path.cwd() / "data" / "synthetic"
    t_start = time.time()
    generated_count = 0
    skipped_count = 0

    for idx, slug in enumerate(targets, start=1):
        spec = get_synthetic_spec(slug)
        target_dir = root / slug
        features_path = target_dir / "features.parquet"
        tau_path = target_dir / "soft_memberships.npy"

        if not args.force and features_path.exists() and tau_path.exists():
            print(f"[{idx}/{len(targets)}] [SKIP VERIFIED] {slug} already exists.")
            skipped_count += 1
            continue

        print(f"[{idx}/{len(targets)}] Generating '{slug}' (seed={spec.generator_seed}, K={spec.K}, n={spec.n}, d={spec.d})...")
        paths = save_synthetic_dataset(spec, output_root=root)

        if args.verify:
            tau = np.load(paths["soft_memberships"])
            assert tau.shape == (spec.n, spec.K), f"Shape mismatch in {slug}: {tau.shape}"
            assert np.allclose(tau.sum(axis=1), 1.0, atol=1e-5), f"Simplex sum error in {slug}"
            assert (tau >= 0.0).all(), f"Negative probability in {slug}"

        generated_count += 1
        print(f"  --> [OK] Saved {slug} artifacts (features, labels, soft memberships).")

    # Update manifests
    all_real_specs = [clusterdrift.data.get_dataset_spec(s) for s in list_datasets()]
    manifest_mgr = ManifestManager()
    manifest_mgr.generate_all_manifests(all_real_specs)

    print("\n" + "=" * 50)
    print("SYNTHETIC BENCHMARK GENERATION REPORT")
    print("=" * 50)
    print(f"Total synthetic families:  {len(targets)}")
    print(f"Generated:                 {generated_count}")
    print(f"Skipped (already valid):   {skipped_count}")
    print(f"Total time elapsed:        {round(time.time() - t_start, 2)}s")
    print("=" * 50)
    print("Manifest written to data/manifests/synthetic_manifest.json")


if __name__ == "__main__":
    import clusterdrift.data
    main()
