#!/usr/bin/env python3
"""Script 01: Download, acquire, cache, verify, and canonicalize the 40 real-world datasets."""

import argparse
import sys
import time
from pathlib import Path
from typing import List

from clusterdrift.data import (
    DatasetDownloader,
    DownloadStatus,
    ManifestManager,
    get_dataset_spec,
    list_controlled_real,
    list_datasets,
    list_natural_shift,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download and canonicalize real-world benchmark datasets.")
    parser.add_argument("--all", action="store_true", help="Download all 40 registered real datasets.")
    parser.add_argument("--dataset", type=str, default=None, help="Download a specific dataset by slug.")
    parser.add_argument(
        "--group",
        type=str,
        choices=["controlled", "natural", "whyshift", "tableshift"],
        default=None,
        help="Download datasets belonging to a specific group.",
    )
    parser.add_argument("--force", action="store_true", help="Force redownload even if canonical data exists.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted downloads.")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify integrity after download.")
    parser.add_argument("--dry-run", action="store_true", help="Preview planned downloads without executing.")
    parser.add_argument("--workers", type=int, default=1, help="Number of download workers (default 1).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.all or args.dataset or args.group):
        print("Error: Specify --all, --dataset <slug>, or --group <group>.")
        sys.exit(1)

    # 1. Determine targets
    targets: List[str] = []
    if args.dataset:
        targets = [args.dataset]
    elif args.group == "controlled":
        targets = list_controlled_real()
    elif args.group == "natural":
        targets = list_natural_shift()
    elif args.group == "whyshift":
        targets = [s for s in list_natural_shift() if s.startswith("whyshift")]
    elif args.group == "tableshift":
        targets = [s for s in list_natural_shift() if s.startswith("tableshift")]
    elif args.all:
        targets = list_datasets()

    print(f"=== Selected {len(targets)} real-world datasets for acquisition ===")

    if args.dry_run:
        print("\n[DRY RUN] The following datasets would be acquired:")
        for t in targets:
            spec = get_dataset_spec(t)
            print(f"  - {t:<35} [{spec.dataset_group}] provider={spec.source_provider} mode={spec.download_mode}")
        return

    downloader = DatasetDownloader(force=args.force, verify=args.verify)
    results = []

    count_ok = 0
    count_already = 0
    count_auth = 0
    count_manual = 0
    count_failed = 0

    t_start = time.time()
    for idx, slug in enumerate(targets, start=1):
        spec = get_dataset_spec(slug)
        print(f"[{idx}/{len(targets)}] Processing '{slug}' (provider={spec.source_provider}, mode={spec.download_mode})...")
        res = downloader.download(spec)
        results.append(res.__dict__)

        if res.status == DownloadStatus.OK:
            count_ok += 1
            print(f"  --> {res.message}")
        elif res.status == DownloadStatus.ALREADY_EXISTS:
            count_already += 1
            print(f"  --> {res.message}")
        elif res.status == DownloadStatus.AUTH_REQUIRED:
            count_auth += 1
            print(f"  --> {res.message}")
        elif res.status == DownloadStatus.MANUAL_LICENSE_ACCEPTANCE:
            count_manual += 1
            print(f"  --> {res.message}")
        else:
            count_failed += 1
            print(f"  --> [FAILED] {res.message}")

    # Compute group availability counts
    all_specs = [get_dataset_spec(s) for s in list_datasets()]
    controlled_avail = sum(1 for s in list_controlled_real() if downloader.is_canonical_present(get_dataset_spec(s)))
    natural_avail = sum(1 for s in list_natural_shift() if downloader.is_canonical_present(get_dataset_spec(s)))

    # Generate / update manifests
    manifest_mgr = ManifestManager()
    manifest_mgr.generate_all_manifests(
        all_specs,
        download_results={
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_targets": len(targets),
            "downloaded_ok": count_ok,
            "already_verified": count_already,
            "auth_required": count_auth,
            "manual_license_required": count_manual,
            "failed": count_failed,
            "controlled_available": f"{controlled_avail} / 30",
            "natural_shift_available": f"{natural_avail} / 10",
            "duration_seconds": round(time.time() - t_start, 2),
            "details": results,
        },
    )

    print("\n" + "=" * 50)
    print("WHEN CLUSTERS DRIFT — DATA ACQUISITION REPORT")
    print("=" * 50)
    print(f"Registered real datasets:       40")
    print(f"Downloaded successfully:        {count_ok}")
    print(f"Already verified:               {count_already}")
    print(f"Authentication required:        {count_auth}")
    print(f"Manual license action required: {count_manual}")
    print(f"Failed:                         {count_failed}")
    print("")
    print(f"Controlled real available:      {controlled_avail} / 30")
    print(f"Natural-shift available:        {natural_avail} / 10")
    print("=" * 50)
    print("Manifest written to data/manifests/download_report.json")


if __name__ == "__main__":
    main()
