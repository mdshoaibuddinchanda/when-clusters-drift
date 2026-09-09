"""Plan and document conservative repository cleanup for Hypothesis Y1.

This script never deletes paths.  ``plan`` writes an explicit allow-list;
the caller must validate and remove those exact paths with the host shell.
``finalize`` verifies the allow-list is gone and records the complete Y1
inventory plus every deletion.  Scientific inputs and frozen evidence are
never candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/hypothesis_y1"
CLEANUP = BASE / "cleanup"
PLAN = CLEANUP / "hypothesis_y1_cleanup_plan.json"
MANIFEST = CLEANUP / "hypothesis_y1_cleanup_manifest.txt"
REPORT = BASE / "reports/hypothesis_y1_repository_cleanup_report.md"
SAFE_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
SAFE_FILE_NAMES = {".coverage", "Thumbs.db", ".DS_Store"}
SAFE_FILE_SUFFIXES = {".pyc", ".pyo"}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: Path) -> str:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"cleanup candidate escapes repository: {resolved}")
    if ".git" in path.relative_to(ROOT).parts:
        raise ValueError(f"cleanup candidate enters .git: {path}")
    return path.relative_to(ROOT).as_posix()


def build_plan() -> dict[str, Any]:
    candidates: list[tuple[Path, str]] = []
    for current, dirnames, filenames in os.walk(ROOT, followlinks=False):
        here = Path(current)
        if here == ROOT / ".git" or ".git" in here.relative_to(ROOT).parts:
            dirnames[:] = []
            continue
        for dirname in list(dirnames):
            path = here / dirname
            if dirname in SAFE_DIRECTORY_NAMES:
                candidates.append((path, "safe generated tool/Python cache"))
                dirnames.remove(dirname)
        for filename in filenames:
            path = here / filename
            if ROOT / "data/raw" in path.parents:
                # Raw-provider bundles are outside automatic cleanup even when
                # they contain platform metadata; preservation wins here.
                continue
            if filename in SAFE_FILE_NAMES or path.suffix.lower() in SAFE_FILE_SUFFIXES:
                candidates.append((path, "safe generated interpreter/tool file"))
            elif BASE in path.parents and path.name.startswith(".") and ".tmp" in path.name:
                candidates.append((path, "orphaned Y1 atomic temporary"))

    runtime_cache = BASE / "corrected_replication/cache"
    if runtime_cache.exists():
        required = [
            BASE / "corrected_replication/hypothesis_y1_corrected_rows.csv",
            BASE / "corrected_replication/hypothesis_y1_corrected_model_metrics.csv",
            BASE / "corrected_replication/hypothesis_y1_corrected_replication_summary.json",
            BASE / "artifacts/hypothesis_y1_corrected_execution_provenance.json",
        ]
        missing = [safe_relative(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"will not plan runtime-cache deletion before consolidation: {missing}")
        summary = json.loads(required[2].read_text(encoding="utf-8"))
        if summary.get("row_file_sha256") != sha256(required[0]):
            raise RuntimeError("corrected consolidated row digest does not match its summary")
        candidates.append((runtime_cache, "duplicate resumable per-fold cache; consolidated rows, metrics, summary and provenance verified"))

    ordered: list[tuple[Path, str]] = []
    for path, reason in sorted(candidates, key=lambda item: (len(item[0].parts), str(item[0]).lower())):
        resolved = path.resolve()
        if any(parent.resolve() in resolved.parents or parent.resolve() == resolved for parent, _ in ordered):
            continue
        safe_relative(path)
        ordered.append((path, reason))
    return {
        "audit": "hypothesis_y1_cleanup_plan",
        "deletion_policy": "exact allow-list; generated caches only; PowerShell performs deletion after containment and reparse-point checks",
        "frozen_scientific_evidence_is_candidate": False,
        "planned_deletions": [
            {
                "absolute_path": str(path.resolve()),
                "relative_path": safe_relative(path),
                "kind": "directory" if path.is_dir() else "file",
                "reason": reason,
            }
            for path, reason in ordered
        ],
    }


def finalize() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    remaining = [item["relative_path"] for item in plan["planned_deletions"] if Path(item["absolute_path"]).exists()]
    if remaining:
        raise RuntimeError(f"planned cleanup paths still exist: {remaining}")

    deleted_lines = [
        f"- `{item['relative_path']}` — {item['kind']}; {item['reason']}"
        for item in plan["planned_deletions"]
    ]
    report = f"""# Hypothesis Y1 repository-cleanup report

## Outcome

Cleanup was deliberately conservative. {len(deleted_lines)} exact paths were removed, all generated interpreter/tool caches, orphaned atomic temporaries, or the resumable per-fold corrected-replication cache after its consolidated rows, model metrics, summary, and execution provenance were verified. No frozen result, input data, canonical bundle, split provenance, historical configuration, manuscript/protocol, intentional placeholder, test log, or negative exploratory result was deleted.

## Classification

- Safe generated junk / Python / pytest caches: deleted only from the explicit plan.
- Duplicate runtime caches: corrected-replication per-fold checkpoints were deleted only after consolidation and digest verification.
- Temporary logs: required real pytest logs retained; no disposable Y1 log selected.
- Obsolete experimental scratch: none deleted; the initial failed bootstrap-order reproduction and partial sensitivity checkpoint are retained as anti-cherry-picking/provenance evidence.
- Dead empty directories: none removed unless they were inside an approved cache tree.
- Sensitive local-only files: none identified by the focused tracked-text scan.
- Canonical scientific artifacts: retained.
- Raw input/provider-bundle contents: retained without exception, including platform metadata.
- Intentional future placeholders and manuscript/protocol files: retained and untouched.

## Exact deletions

{chr(10).join(deleted_lines) if deleted_lines else '- None found.'}

The machine-readable cleanup plan is `experiments/hypothesis_y1/cleanup/hypothesis_y1_cleanup_plan.json`. The complete removable Y1 inventory is `experiments/hypothesis_y1/cleanup/hypothesis_y1_cleanup_manifest.txt`.
"""
    atomic_text(REPORT, report)

    retained_files = sorted(
        path.relative_to(ROOT).as_posix() for path in BASE.rglob("*") if path.is_file() and path != MANIFEST
    )
    retained_directories = sorted(
        path.relative_to(ROOT).as_posix() + "/" for path in BASE.rglob("*") if path.is_dir()
    )
    retained_files.append(MANIFEST.relative_to(ROOT).as_posix())
    retained_files.sort()
    lines = [
        "HYPOTHESIS Y1 CLEANUP / REMOVABILITY MANIFEST",
        "root: experiments/hypothesis_y1/",
        "scope: every retained Y1-only file/directory plus every cleanup deletion",
        "frozen Phase-7 evidence modified: NO",
        "",
        f"RETAINED FILES ({len(retained_files)})",
        *retained_files,
        "",
        f"RETAINED DIRECTORIES ({len(retained_directories)})",
        *retained_directories,
        "",
        f"DELETED PATHS ({len(plan['planned_deletions'])})",
        *[f"{item['relative_path']} | {item['kind']} | {item['reason']}" for item in plan["planned_deletions"]],
        "",
    ]
    atomic_text(MANIFEST, "\n".join(lines))
    print(json.dumps({
        "deleted_paths_verified_absent": len(plan["planned_deletions"]),
        "retained_files": len(retained_files),
        "retained_directories": len(retained_directories),
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "report": REPORT.relative_to(ROOT).as_posix(),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "finalize"))
    args = parser.parse_args()
    if args.mode == "plan":
        plan = build_plan()
        atomic_text(PLAN, json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"planned_deletions": len(plan["planned_deletions"]), "plan": PLAN.relative_to(ROOT).as_posix()}, indent=2))
    else:
        finalize()


if __name__ == "__main__":
    main()
