# Hypothesis Y1 security and scientific-integrity audit

## Outcome

The focused audit found nine real software/scientific-integrity defects in the frozen implementation. All nine are repaired in the current Y1 branch. One defect—executable joblib cache deserialization—was a material local software-security risk; the remaining defects primarily threatened scientific correctness, provenance, crash safety, or acquisition integrity. No committed credential, `shell=True`, or user-specific absolute path was detected by the tracked-text scan.

The frozen Phase-7 files and results were not rewritten. The fixes affect future execution and the isolated corrected replication only.

| ID | Severity | Finding | Status |
|---|---|---|---|
| SEC-01 | HIGH | Phase-7 cache loaded joblib payloads before any trustworthy content verification. | fixed |
| SEC-02 | HIGH_SCIENTIFIC_INTEGRITY | Cached numerical payloads had no stored content digest and were accepted after checking metadata only. | fixed |
| SEC-03 | HIGH_SCIENTIFIC_INTEGRITY | MMD bandwidth/source-kernel and D_X scalar caches were accepted solely because a filename existed. | fixed |
| SEC-04 | HIGH_SCIENTIFIC_INTEGRITY | Source/model/scenario cache keys omitted some underlying artifact and protocol hashes. | fixed |
| SEC-05 | MEDIUM | Unvalidated identifier components reached cache paths, and recursive clearing relied only on a marker check. | fixed |
| SEC-06 | MEDIUM | Cache writers used predictable shared temporary filenames. | fixed |
| SEC-07 | HIGH_SUPPLY_CHAIN_INTEGRITY | Three direct archive downloads were neither digest-pinned nor atomically persisted. | fixed |
| SEC-08 | MEDIUM_SCIENTIFIC_INTEGRITY | The final Phase-7 verification manifest was written directly to its canonical path. | fixed |
| SEC-09 | MEDIUM_SCIENTIFIC_INTEGRITY | Canonical data, split, manifest, and baseline-audit writers published directly to final paths. | fixed |

## Evidence by finding

### SEC-01 — Phase-7 cache loaded joblib payloads before any trustworthy content verification.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `HIGH`
- Evidence: `3da9c6ee6af8f00110e6eab569d1dba009a6598c:src/clusterdrift/falsification/execution/cache.py:211,409`
- Exploit/failure mode: A replaced local cache payload could execute Python code during deserialization; a corrupt but loadable object could also enter verification.
- Fix: Replace executable serialization with numeric NPZ state for FCM and treat all legacy joblib entries as cache misses/corrupt entries.
- Test: `test_numeric_model_cache_roundtrip_and_tamper_rejection`

### SEC-02 — Cached numerical payloads had no stored content digest and were accepted after checking metadata only.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `HIGH_SCIENTIFIC_INTEGRITY`
- Evidence: `3da9c6ee6af8f00110e6eab569d1dba009a6598c:src/clusterdrift/falsification/execution/cache.py:82-83,145-146,280-281`
- Exploit/failure mode: Silent payload replacement, partial corruption, or stale array reuse could change scientific signals while metadata still matched.
- Fix: Store and verify SHA-256 plus expected shapes before all array/model/membership loads; set allow_pickle=False.
- Test: `test_array_cache_digest_and_fingerprint_are_required and test_numeric_model_cache_roundtrip_and_tamper_rejection`

### SEC-03 — MMD bandwidth/source-kernel and D_X scalar caches were accepted solely because a filename existed.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `HIGH_SCIENTIFIC_INTEGRITY`
- Evidence: `3da9c6ee6af8f00110e6eab569d1dba009a6598c:src/clusterdrift/falsification/execution/cache.py:322,344`
- Exploit/failure mode: A value computed for different data, split, probe, or signal protocol could be silently reused.
- Fix: Require a provenance fingerprint on scalar cache reads and writes.
- Test: `test_scalar_caches_require_matching_provenance`

### SEC-04 — Source/model/scenario cache keys omitted some underlying artifact and protocol hashes.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `HIGH_SCIENTIFIC_INTEGRITY`
- Evidence: `3da9c6ee6af8f00110e6eab569d1dba009a6598c:scripts/07_falsification_pilot.py:260-296,361-367`
- Exploit/failure mode: Modified features, metadata, splits, probes, method configuration, or signal configuration could retain the same cache key.
- Fix: Bind source/model/scenario/MMD/D_X keys to input artifact hashes, split, probes, method/configuration, protocol and seed.
- Test: `test_source_and_model_fingerprints_bind_input_artifacts`

### SEC-05 — Unvalidated identifier components reached cache paths, and recursive clearing relied only on a marker check.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `MEDIUM`
- Evidence: `3da9c6ee6af8f00110e6eab569d1dba009a6598c:src/clusterdrift/falsification/execution/cache.py:132,198,267,434`
- Exploit/failure mode: A malicious or malformed dataset/method/condition identifier could traverse directories; a redirected cache path increased deletion risk.
- Fix: Reject non-portable/path-like components and require resolved containment plus a non-symlink root before recursive removal.
- Test: `test_cache_rejects_path_traversal_components and test_cache_clear_rejects_escaped_root`

### SEC-06 — Cache writers used predictable shared temporary filenames.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `MEDIUM`
- Evidence: `3da9c6ee6af8f00110e6eab569d1dba009a6598c:src/clusterdrift/falsification/execution/cache.py:102-108,166-172,239-241,301-307`
- Exploit/failure mode: Concurrent workers could overwrite each other's temporary payloads or expose partially inconsistent metadata/payload pairs.
- Fix: Use unique same-directory temporary files, fsync where applicable, then atomic os.replace.
- Test: `test_numeric_model_cache_roundtrip_and_tamper_rejection`

### SEC-07 — Three direct archive downloads were neither digest-pinned nor atomically persisted.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `HIGH_SUPPLY_CHAIN_INTEGRITY`
- Evidence: `3da9c6ee6af8f00110e6eab569d1dba009a6598c:src/clusterdrift/data/loaders.py:53-58,152-157,217-222`
- Exploit/failure mode: Upstream compromise, transparent content change, or interrupted writes could seed different/corrupt raw datasets.
- Fix: Require HTTPS and registry-pinned SHA-256, enforce a size ceiling, stream to a unique temporary file, fsync, verify, and atomically replace.
- Test: `test_pinned_archive_download_rejects_wrong_digest_and_non_https`

### SEC-08 — The final Phase-7 verification manifest was written directly to its canonical path.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `MEDIUM_SCIENTIFIC_INTEGRITY`
- Evidence: `3da9c6ee6af8f00110e6eab569d1dba009a6598c:src/clusterdrift/falsification/verification.py:955-956`
- Exploit/failure mode: Interruption during verification could leave a truncated manifest even when the source result files remained valid.
- Fix: Write the verification manifest through the existing same-directory atomic JSON writer.
- Test: `tests/test_falsification verification tests plus final full suite`

### SEC-09 — Canonical data, split, manifest, and baseline-audit writers published directly to final paths.

- Classification: `B. FIX_NOW_SOFTWARE_INTEGRITY`
- Severity: `MEDIUM_SCIENTIFIC_INTEGRITY`
- Evidence: `frozen/current direct writers in data canonicalize.py, synthetic.py, folds.py, manifest.py, validation.py and scripts/02-04`
- Exploit/failure mode: Process interruption or concurrent execution could leave truncated parquet/NPY/CSV/JSON scientific artifacts at canonical filenames.
- Fix: Centralize same-directory atomic parquet, non-pickle NPY, CSV, JSON, and NPZ writers and migrate every direct scientific writer found in src/ and scripts/.
- Test: `test_atomic_scientific_writers_publish_readable_files plus final full suite`

## Negative findings and bounded scope

- Scanned 7682 tracked files (254 Python files). Possible committed-secret matches: 0. User-specific absolute-path matches: 0.
- Current unsafe-deserialization / `shell=True` / `os.system` matches: 0. Subprocess sites use fixed argument arrays rather than a shell.
- Direct scientific-write calls outside the two atomic implementation modules after repair: 0.
- ZIP loaders open named members rather than extracting caller-controlled paths.
- No private dependency namespace was found, so there is no concrete dependency-confusion finding. Open version ranges and the intentionally empty `requirements.lock` remain a reproducibility limitation.
- Provider-managed downloads are version/ID addressed and post-acquisition canonical hashes are recorded, but not every provider payload is byte-pinned before acquisition. This remains a stated limitation.
- Threat model: these changes prevent stale/corrupt cache acceptance and unsafe executable cache loads. They do not claim defense if an attacker already controls the same account and can rewrite both code and provenance metadata.

## Verification

The Y1 guard suite exercises safe FCM serialization, legacy joblib rejection, content tampering, scalar provenance mismatch, path traversal, split leakage, frozen hashes, and pinned-download rejection. Its actual final test count is reported separately in `hypothesis_y1_test_results.json`; this report does not invent a test count.
