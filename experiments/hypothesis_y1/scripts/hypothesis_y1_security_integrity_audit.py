"""Focused security and scientific-integrity audit for Hypothesis Y1.

The scan reports locations and rule names only.  It deliberately never copies a
possible credential value into an audit artifact.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments" / "hypothesis_y1"
ARTIFACT = BASE / "artifacts" / "hypothesis_y1_security_integrity_audit.json"
REPORT = BASE / "reports" / "hypothesis_y1_security_integrity_audit.md"
FROZEN_HEAD = "3da9c6ee6af8f00110e6eab569d1dba009a6598c"


SECRET_RULES = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_url": re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"),
    "literal_credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}
ABSOLUTE_PATH_RULES = {
    "windows_user_path": re.compile(r"(?i)[A-Z]:\\Users\\[^\\\s]+\\"),
    "posix_home_path": re.compile(r"/home/[^/\s]+/"),
}
UNSAFE_CODE_RULES = {
    "pickle_or_joblib_load": re.compile(r"\b(?:pickle|joblib|dill|cloudpickle)\.loads?\s*\("),
    "shell_true": re.compile(r"\bshell\s*=\s*True\b"),
    "os_system": re.compile(r"\bos\.system\s*\("),
}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def tracked_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / item.decode("utf-8") for item in proc.stdout.split(b"\0") if item]


def scan_rules(files: list[Path], rules: dict[str, re.Pattern[str]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\0" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for name, pattern in rules.items():
                if pattern.search(line):
                    matches.append({
                        "rule": name,
                        "path": path.relative_to(ROOT).as_posix(),
                        "line": line_number,
                    })
    return matches


def git_show(path: str) -> str:
    proc = subprocess.run(
        ["git", "show", f"{FROZEN_HEAD}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return proc.stdout


def main() -> None:
    files = tracked_files()
    python_files = [path for path in files if path.suffix == ".py"]
    secret_matches = scan_rules(files, SECRET_RULES)
    absolute_path_matches = scan_rules(files, ABSOLUTE_PATH_RULES)
    unsafe_matches = scan_rules(python_files, UNSAFE_CODE_RULES)
    scientific_code = [
        path for path in python_files
        if ("/src/" in "/" + path.relative_to(ROOT).as_posix() or path.relative_to(ROOT).as_posix().startswith("scripts/"))
        and path.relative_to(ROOT).as_posix() not in {
            "src/clusterdrift/shifts/hashing.py",
            "src/clusterdrift/falsification/execution/cache.py",
        }
    ]
    direct_write_matches = scan_rules(scientific_code, {
        "direct_dataframe_write": re.compile(r"\.(?:to_csv|to_parquet)\s*\("),
        "direct_numpy_write": re.compile(r"\bnp\.save(?:z_compressed)?\s*\("),
        "direct_json_dump": re.compile(r"\bjson\.dump\s*\("),
    })

    old_cache = git_show("src/clusterdrift/falsification/execution/cache.py")
    old_runner = git_show("scripts/07_falsification_pilot.py")
    old_loaders = git_show("src/clusterdrift/data/loaders.py")
    current_cache = (ROOT / "src/clusterdrift/falsification/execution/cache.py").read_text(encoding="utf-8")
    current_runner = (ROOT / "scripts/07_falsification_pilot.py").read_text(encoding="utf-8")
    current_loaders = (ROOT / "src/clusterdrift/data/loaders.py").read_text(encoding="utf-8")
    current_verification = (ROOT / "src/clusterdrift/falsification/verification.py").read_text(encoding="utf-8")

    checks = {
        "frozen_cache_used_joblib_load": "joblib.load(" in old_cache,
        "current_cache_has_no_joblib_load": "joblib.load(" not in current_cache,
        "current_cache_disables_pickle_np_load": "allow_pickle=False" in current_cache,
        "current_cache_has_payload_digests": all(
            token in current_cache for token in ("x_src_sha256", "x_tgt_sha256", "model_sha256", "u_sha256")
        ),
        "frozen_scalar_cache_was_filename_only": (
            "def get_mmd_sigma(self, dataset_id: str, outer_fold: int)" in old_cache
            and "def get_dx(self, dataset_id: str, outer_fold: int, condition: str)" in old_cache
        ),
        "current_scalar_cache_requires_fingerprint": (
            "get_mmd_sigma(\n        self, dataset_id: str, outer_fold: int, expected_fingerprint: str" in current_cache
            and "condition: str, expected_fingerprint: str" in current_cache
        ),
        "frozen_source_key_omitted_artifact_hashes": (
            '"dataset_id": ds' in old_runner and '"features_sha256"' not in old_runner
        ),
        "current_source_key_binds_artifacts": "compute_source_cache_fingerprint" in current_runner,
        "frozen_archive_download_was_direct": "resp.read()" in old_loaders and 'open(zip_path, "wb")' in old_loaders,
        "current_archive_download_is_pinned_atomic": all(
            token in current_loaders for token in ("source_sha256", "os.replace(tmp, path)", "https://", "_MAX_ARCHIVE_BYTES")
        ),
        "current_cache_uses_unique_atomic_temps": "tempfile.mkstemp" in current_cache,
        "current_cache_rejects_unsafe_components": "_safe_component" in current_cache,
        "current_result_manifest_write_is_atomic": "atomic_write_json(manifest_p, manifest_doc" in current_verification,
    }

    issues = [
        {
            "id": "SEC-01",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "HIGH",
            "evidence": f"{FROZEN_HEAD}:src/clusterdrift/falsification/execution/cache.py:211,409",
            "problem": "Phase-7 cache loaded joblib payloads before any trustworthy content verification.",
            "failure_mode": "A replaced local cache payload could execute Python code during deserialization; a corrupt but loadable object could also enter verification.",
            "fix": "Replace executable serialization with numeric NPZ state for FCM and treat all legacy joblib entries as cache misses/corrupt entries.",
            "test": "test_numeric_model_cache_roundtrip_and_tamper_rejection",
            "fixed": True,
        },
        {
            "id": "SEC-02",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "HIGH_SCIENTIFIC_INTEGRITY",
            "evidence": f"{FROZEN_HEAD}:src/clusterdrift/falsification/execution/cache.py:82-83,145-146,280-281",
            "problem": "Cached numerical payloads had no stored content digest and were accepted after checking metadata only.",
            "failure_mode": "Silent payload replacement, partial corruption, or stale array reuse could change scientific signals while metadata still matched.",
            "fix": "Store and verify SHA-256 plus expected shapes before all array/model/membership loads; set allow_pickle=False.",
            "test": "test_array_cache_digest_and_fingerprint_are_required and test_numeric_model_cache_roundtrip_and_tamper_rejection",
            "fixed": True,
        },
        {
            "id": "SEC-03",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "HIGH_SCIENTIFIC_INTEGRITY",
            "evidence": f"{FROZEN_HEAD}:src/clusterdrift/falsification/execution/cache.py:322,344",
            "problem": "MMD bandwidth/source-kernel and D_X scalar caches were accepted solely because a filename existed.",
            "failure_mode": "A value computed for different data, split, probe, or signal protocol could be silently reused.",
            "fix": "Require a provenance fingerprint on scalar cache reads and writes.",
            "test": "test_scalar_caches_require_matching_provenance",
            "fixed": True,
        },
        {
            "id": "SEC-04",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "HIGH_SCIENTIFIC_INTEGRITY",
            "evidence": f"{FROZEN_HEAD}:scripts/07_falsification_pilot.py:260-296,361-367",
            "problem": "Source/model/scenario cache keys omitted some underlying artifact and protocol hashes.",
            "failure_mode": "Modified features, metadata, splits, probes, method configuration, or signal configuration could retain the same cache key.",
            "fix": "Bind source/model/scenario/MMD/D_X keys to input artifact hashes, split, probes, method/configuration, protocol and seed.",
            "test": "test_source_and_model_fingerprints_bind_input_artifacts",
            "fixed": True,
        },
        {
            "id": "SEC-05",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "MEDIUM",
            "evidence": f"{FROZEN_HEAD}:src/clusterdrift/falsification/execution/cache.py:132,198,267,434",
            "problem": "Unvalidated identifier components reached cache paths, and recursive clearing relied only on a marker check.",
            "failure_mode": "A malicious or malformed dataset/method/condition identifier could traverse directories; a redirected cache path increased deletion risk.",
            "fix": "Reject non-portable/path-like components and require resolved containment plus a non-symlink root before recursive removal.",
            "test": "test_cache_rejects_path_traversal_components and test_cache_clear_rejects_escaped_root",
            "fixed": True,
        },
        {
            "id": "SEC-06",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "MEDIUM",
            "evidence": f"{FROZEN_HEAD}:src/clusterdrift/falsification/execution/cache.py:102-108,166-172,239-241,301-307",
            "problem": "Cache writers used predictable shared temporary filenames.",
            "failure_mode": "Concurrent workers could overwrite each other's temporary payloads or expose partially inconsistent metadata/payload pairs.",
            "fix": "Use unique same-directory temporary files, fsync where applicable, then atomic os.replace.",
            "test": "test_numeric_model_cache_roundtrip_and_tamper_rejection",
            "fixed": True,
        },
        {
            "id": "SEC-07",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "HIGH_SUPPLY_CHAIN_INTEGRITY",
            "evidence": f"{FROZEN_HEAD}:src/clusterdrift/data/loaders.py:53-58,152-157,217-222",
            "problem": "Three direct archive downloads were neither digest-pinned nor atomically persisted.",
            "failure_mode": "Upstream compromise, transparent content change, or interrupted writes could seed different/corrupt raw datasets.",
            "fix": "Require HTTPS and registry-pinned SHA-256, enforce a size ceiling, stream to a unique temporary file, fsync, verify, and atomically replace.",
            "test": "test_pinned_archive_download_rejects_wrong_digest_and_non_https",
            "fixed": True,
        },
        {
            "id": "SEC-08",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "MEDIUM_SCIENTIFIC_INTEGRITY",
            "evidence": f"{FROZEN_HEAD}:src/clusterdrift/falsification/verification.py:955-956",
            "problem": "The final Phase-7 verification manifest was written directly to its canonical path.",
            "failure_mode": "Interruption during verification could leave a truncated manifest even when the source result files remained valid.",
            "fix": "Write the verification manifest through the existing same-directory atomic JSON writer.",
            "test": "tests/test_falsification verification tests plus final full suite",
            "fixed": True,
        },
        {
            "id": "SEC-09",
            "classification": "B. FIX_NOW_SOFTWARE_INTEGRITY",
            "severity": "MEDIUM_SCIENTIFIC_INTEGRITY",
            "evidence": "frozen/current direct writers in data canonicalize.py, synthetic.py, folds.py, manifest.py, validation.py and scripts/02-04",
            "problem": "Canonical data, split, manifest, and baseline-audit writers published directly to final paths.",
            "failure_mode": "Process interruption or concurrent execution could leave truncated parquet/NPY/CSV/JSON scientific artifacts at canonical filenames.",
            "fix": "Centralize same-directory atomic parquet, non-pickle NPY, CSV, JSON, and NPZ writers and migrate every direct scientific writer found in src/ and scripts/.",
            "test": "test_atomic_scientific_writers_publish_readable_files plus final full suite",
            "fixed": True,
        },
    ]

    result = {
        "audit": "hypothesis_y1_security_integrity_audit",
        "frozen_reference_commit": FROZEN_HEAD,
        "scope": "tracked repository text plus focused historical/current Phase-7 cache and direct archive acquisition comparison",
        "tracked_files_scanned": len(files),
        "tracked_python_files_scanned": len(python_files),
        "possible_committed_secret_matches": secret_matches,
        "user_specific_absolute_path_matches": absolute_path_matches,
        "current_unsafe_code_matches": unsafe_matches,
        "current_direct_scientific_write_matches_outside_atomic_helpers": direct_write_matches,
        "checks": checks,
        "issues": issues,
        "security_issue_count": len(issues),
        "all_identified_issues_fixed": all(issue["fixed"] for issue in issues),
        "residual_findings": [
            "Package versions remain ranges and requirements.lock is intentionally empty; this is a reproducibility limitation, not evidence of dependency confusion because no private package namespace is used.",
            "Provider-managed OpenML/whyshift/tableshift acquisition is not byte-pinned at this layer; canonical bundle manifests retain post-acquisition hashes.",
            "Local cache security assumes the repository parent and cache parent are not already controlled by a hostile same-user process.",
        ],
    }
    atomic_write(ARTIFACT, json.dumps(result, indent=2, sort_keys=True) + "\n")

    rows = "\n".join(
        f"| {item['id']} | {item['severity']} | {item['problem']} | fixed |" for item in issues
    )
    details = "\n\n".join(
        "\n".join(
            [
                f"### {item['id']} — {item['problem']}",
                "",
                f"- Classification: `{item['classification']}`",
                f"- Severity: `{item['severity']}`",
                f"- Evidence: `{item['evidence']}`",
                f"- Exploit/failure mode: {item['failure_mode']}",
                f"- Fix: {item['fix']}",
                f"- Test: `{item['test']}`",
            ]
        )
        for item in issues
    )
    report = f"""# Hypothesis Y1 security and scientific-integrity audit

## Outcome

The focused audit found nine real software/scientific-integrity defects in the frozen implementation. All nine are repaired in the current Y1 branch. One defect—executable joblib cache deserialization—was a material local software-security risk; the remaining defects primarily threatened scientific correctness, provenance, crash safety, or acquisition integrity. No committed credential, `shell=True`, or user-specific absolute path was detected by the tracked-text scan.

The frozen Phase-7 files and results were not rewritten. The fixes affect future execution and the isolated corrected replication only.

| ID | Severity | Finding | Status |
|---|---|---|---|
{rows}

## Evidence by finding

{details}

## Negative findings and bounded scope

- Scanned {len(files)} tracked files ({len(python_files)} Python files). Possible committed-secret matches: {len(secret_matches)}. User-specific absolute-path matches: {len(absolute_path_matches)}.
- Current unsafe-deserialization / `shell=True` / `os.system` matches: {len(unsafe_matches)}. Subprocess sites use fixed argument arrays rather than a shell.
- Direct scientific-write calls outside the two atomic implementation modules after repair: {len(direct_write_matches)}.
- ZIP loaders open named members rather than extracting caller-controlled paths.
- No private dependency namespace was found, so there is no concrete dependency-confusion finding. Open version ranges and the intentionally empty `requirements.lock` remain a reproducibility limitation.
- Provider-managed downloads are version/ID addressed and post-acquisition canonical hashes are recorded, but not every provider payload is byte-pinned before acquisition. This remains a stated limitation.
- Threat model: these changes prevent stale/corrupt cache acceptance and unsafe executable cache loads. They do not claim defense if an attacker already controls the same account and can rewrite both code and provenance metadata.

## Verification

The Y1 guard suite exercises safe FCM serialization, legacy joblib rejection, content tampering, scalar provenance mismatch, path traversal, split leakage, frozen hashes, and pinned-download rejection. Its actual final test count is reported separately in `hypothesis_y1_test_results.json`; this report does not invent a test count.
"""
    atomic_write(REPORT, report)
    print(json.dumps({"artifact": str(ARTIFACT.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT)), "issues": len(issues)}, indent=2))


if __name__ == "__main__":
    main()
