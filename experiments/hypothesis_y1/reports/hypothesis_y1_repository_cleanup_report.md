# Hypothesis Y1 repository-cleanup report

## Outcome

Cleanup was deliberately conservative. 22 exact paths were removed, all generated interpreter/tool caches, orphaned atomic temporaries, or the resumable per-fold corrected-replication cache after its consolidated rows, model metrics, summary, and execution provenance were verified. No frozen result, input data, canonical bundle, split provenance, historical configuration, manuscript/protocol, intentional placeholder, test log, or negative exploratory result was deleted.

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

- `.pytest_cache` — directory; safe generated tool/Python cache
- `scripts/__pycache__` — directory; safe generated tool/Python cache
- `tests/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/__pycache__` — directory; safe generated tool/Python cache
- `tests/test_alignment/__pycache__` — directory; safe generated tool/Python cache
- `tests/test_falsification/__pycache__` — directory; safe generated tool/Python cache
- `tests/test_methods/__pycache__` — directory; safe generated tool/Python cache
- `tests/test_probes/__pycache__` — directory; safe generated tool/Python cache
- `tests/test_shifts/__pycache__` — directory; safe generated tool/Python cache
- `tests/test_signals/__pycache__` — directory; safe generated tool/Python cache
- `experiments/hypothesis_y1/corrected_replication/cache` — directory; duplicate resumable per-fold cache; consolidated rows, metrics, summary and provenance verified
- `experiments/hypothesis_y1/scripts/__pycache__` — directory; safe generated tool/Python cache
- `experiments/hypothesis_y1/tests/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/alignment/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/data/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/falsification/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/methods/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/metrics/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/probes/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/shifts/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/signals/__pycache__` — directory; safe generated tool/Python cache
- `src/clusterdrift/falsification/execution/__pycache__` — directory; safe generated tool/Python cache

The machine-readable cleanup plan is `experiments/hypothesis_y1/cleanup/hypothesis_y1_cleanup_plan.json`. The complete removable Y1 inventory is `experiments/hypothesis_y1/cleanup/hypothesis_y1_cleanup_manifest.txt`.
