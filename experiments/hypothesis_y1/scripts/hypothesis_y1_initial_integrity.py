"""Verify the frozen Phase-7 record before Hypothesis-Y1 analyses.

This script is deliberately independent of clusterdrift verification helpers.
It reads bytes, computes SHA-256 digests, checks historical commits, and writes
only inside experiments/hypothesis_y1/artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "hypothesis_y1" / "artifacts" / "hypothesis_y1_initial_integrity.json"

COMMITS = {
    "scientific_preregistration": "8dc7bc8056a686f1eb147f9ec5bf211935454da6",
    "pass_a_freeze": "89b3df90f2cdc29d0e341637a11c0eabd2099ee7",
    "pass_b_freeze": "889624d2ade5d15635d9459dd2601c5664a2747b",
    "pass_c_preflight": "ef07a68f973fdc77bae7d7410149214b2a10ee21",
    "phase7_result": "c9280bb390492f4528752f5457e341fb3d9c1040",
    "manifest_finalization": "3da9c6ee6af8f00110e6eab569d1dba009a6598c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if check and proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def verify_file(rel: str, expected: str | None = None) -> dict[str, Any]:
    path = ROOT / rel
    result: dict[str, Any] = {"path": rel, "exists": path.is_file()}
    if path.is_file():
        actual = sha256(path)
        result.update(
            actual_sha256=actual,
            expected_sha256=expected,
            matches_expected=(expected is None or actual == expected),
            size_bytes=path.stat().st_size,
        )
    else:
        result.update(expected_sha256=expected, matches_expected=False)
    return result


def main() -> None:
    manifest_path = ROOT / "results" / "falsification" / "phase7_result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pass_a = json.loads((ROOT / "results/falsification/pass_a_freeze.json").read_text(encoding="utf-8"))
    pass_b = json.loads((ROOT / "results/falsification/pass_b_freeze.json").read_text(encoding="utf-8"))
    input_lock = json.loads((ROOT / "data/falsification/phase7_input_lock.json").read_text(encoding="utf-8"))

    files: dict[str, dict[str, Any]] = {}
    files["pass_a_signals"] = verify_file(
        "results/falsification/signals_label_free.csv", manifest["pass_a_signals_sha256"]
    )
    files["pass_b_quality"] = verify_file(
        "results/falsification/quality_evaluation_only.csv", manifest["pass_b_quality_sha256"]
    )
    files["joined_table"] = verify_file(
        "results/falsification/joined_evaluation_table.csv", manifest["joined_table_sha256"]
    )
    files["pass_a_freeze"] = verify_file(
        "results/falsification/pass_a_freeze.json", pass_b["pass_a_freeze_manifest_sha256"]
    )

    for key, expected in manifest["metric_artifacts"].items():
        stem = key.removesuffix("_sha256")
        files[f"metric:{stem}"] = verify_file(f"results/falsification/{stem}.csv" if stem != "verdict" else "results/falsification/verdict.json", expected)
    for key, record in manifest["prediction_artifacts"].items():
        files[f"prediction:{key}"] = verify_file(
            f"results/falsification/{key}.csv", record["sha256"]
        )

    lock_mapping = {
        "alignment_config_sha256": "configs/alignment.yaml",
        "falsification_config_sha256": "configs/falsification.yaml",
        "methods_config_sha256": "configs/methods.yaml",
        "phase1_dataset_manifest_sha256": "data/manifests/datasets.json",
        "phase2_preprocessing_config_sha256": "configs/preprocessing.yaml",
        "phase2_split_manifest_sha256": "data/splits/split_manifest.json",
        "phase4_shift_manifest_sha256": "data/shifts/shift_manifest.json",
        "phase5_probe_manifest_sha256": "data/probes/probe_manifest.json",
        "phase6_input_lock_sha256": "data/signals/phase6_input_lock.json",
        "probe_config_sha256": "configs/probes.yaml",
        "signals_config_sha256": "configs/signals.yaml",
    }
    config_and_input_files: dict[str, dict[str, Any]] = {}
    for key, rel in lock_mapping.items():
        config_and_input_files[key] = verify_file(rel, input_lock[key])

    cfg = yaml.safe_load((ROOT / "configs/falsification.yaml").read_text(encoding="utf-8"))
    protocol_keys = [
        "protocol_version", "falsification_version", "global_seed", "datasets",
        "outer_folds", "conditions", "methods", "algorithm_seeds",
        "primary_analysis", "secondary_analysis", "feature_blocks",
        "primary_regressor", "linear_control", "inner_cv", "bootstrap",
    ]
    computed_protocol_sha = canonical_sha256({k: cfg[k] for k in protocol_keys})
    protocol_sha_values = set()
    for csv_rel in [
        "results/falsification/lodo_predictions.csv",
        "results/falsification/losfo_predictions.csv",
    ]:
        import pandas as pd

        values = pd.read_csv(ROOT / csv_rel, usecols=["phase7_protocol_sha"])["phase7_protocol_sha"].dropna().unique()
        protocol_sha_values.update(map(str, values))

    scientific_globs = [
        "scripts/07_falsification_pilot.py",
        "src/clusterdrift/falsification/*.py",
        "src/clusterdrift/methods/fcm.py",
        "src/clusterdrift/methods/fuzzifier.py",
        "src/clusterdrift/signals/*.py",
    ]
    scientific_source_sha256: dict[str, str] = {}
    for pattern in scientific_globs:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                scientific_source_sha256[path.relative_to(ROOT).as_posix()] = sha256(path)

    commit_checks = {}
    for name, commit in COMMITS.items():
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
            capture_output=True, check=False,
        ).returncode == 0
        commit_checks[name] = {"sha": commit, "exists": exists}

    frozen_commit_identity = {
        "pass_a_signals_at_freeze": git("show", f"{COMMITS['pass_a_freeze']}:results/falsification/signals_label_free.csv", check=False) != "",
        "pass_b_quality_at_freeze": git("show", f"{COMMITS['pass_b_freeze']}:results/falsification/quality_evaluation_only.csv", check=False) != "",
        "result_manifest_at_result_commit": subprocess.run(
            ["git", "cat-file", "-e", f"{COMMITS['phase7_result']}:results/falsification/verdict.json"],
            cwd=ROOT, capture_output=True, check=False,
        ).returncode == 0,
    }

    all_file_checks = [*files.values(), *config_and_input_files.values()]
    failures = [x for x in all_file_checks if not x.get("matches_expected", False)]
    failures.extend(
        {"commit": name, **check}
        for name, check in commit_checks.items() if not check["exists"]
    )
    failures.extend(
        {"historical_identity": name, "matches_expected": False}
        for name, value in frozen_commit_identity.items() if not value
    )
    if protocol_sha_values != {computed_protocol_sha}:
        failures.append({
            "protocol_sha_mismatch": {
                "computed": computed_protocol_sha,
                "prediction_values": sorted(protocol_sha_values),
            },
            "matches_expected": False,
        })

    result = {
        "audit": "hypothesis_y1_initial_integrity",
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_git_status": git("status", "--short", "--untracked-files=no"),
        "full_git_status": git("status", "--short"),
        "historical_commits": commit_checks,
        "frozen_commit_identity": frozen_commit_identity,
        "files": files,
        "config_and_input_files": config_and_input_files,
        "computed_phase7_protocol_sha256": computed_protocol_sha,
        "protocol_sha_values_in_predictions": sorted(protocol_sha_values),
        "scientific_source_sha256": scientific_source_sha256,
        "result_manifest_sha256": sha256(manifest_path),
        "integrity_status": "PASSED" if not failures else "FAILED",
        "failures": failures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(OUT, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"integrity_status": result["integrity_status"], "failures": len(failures), "output": str(OUT)}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
