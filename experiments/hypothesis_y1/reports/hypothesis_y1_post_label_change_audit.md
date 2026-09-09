# Hypothesis Y1 post-label code-change audit

## Conclusion

There was a material procedural exposure after the Phase-7 quality labels were available: the implementation of the evaluation/preflight pipeline was substantially rewritten between the Pass-B freeze (`889624d`) and the Pass-C preflight (`ef07a68`). This is a confirmed research-integrity design weakness because the implementation was not frozen before outcome visibility.

The evidence does **not** show that the model family, feature blocks, hyperparameter grids, folds, target, primary metric, tie rule, or verdict thresholds were changed to favor structural signals. The independently written Y1 evaluator reproduces the frozen prediction metrics and verdict to numerical tolerance. The appropriate conclusion is therefore “material procedural exposure, no demonstrated outcome-induced result change,” not “fabricated result” and not “no issue.”

Classification: `A. FIX_NOW_METHODOLOGICAL` for future experiments; historical Phase 7 remains immutable.

## Timeline

| Commit | Local time (+05:30) | State |
|---|---:|---|
| `8dc7bc8056a686f1eb147f9ec5bf211935454da6` | 2026-09-08 01:20:02 | Scientific protocol preregistered |
| `89b3df90f2cdc29d0e341637a11c0eabd2099ee7` | 2026-09-08 04:38:49 | Label-free Pass-A signals frozen |
| `889624d2ade5d15635d9459dd2601c5664a2747b` | 2026-09-08 05:21:02 | Evaluation-only Pass-B quality outcomes frozen; labels were now inspectable |
| `ef07a68f973fdc77bae7d7410149214b2a10ee21` | 2026-09-08 05:57:59 | Joined table/preflight plus major evaluation and verification changes |
| `c9280bb390492f4528752f5457e341fb3d9c1040` | 2026-09-08 07:34:20 | Preregistered result artifacts generated |
| `3da9c6ee6af8f00110e6eab569d1dba009a6598c` | 2026-09-08 07:35:13 | Manifest provenance finalized |

Git establishes commit ordering, not the within-commit order of edits or what a person viewed. Consequently it proves opportunity for outcome exposure, not intent.

## Exact change surface after quality-label availability

Diff `889624d..ef07a68` contains 6,311 insertions and 338 deletions across ten files:

| File | Insertions | Deletions | Scientific assessment |
|---|---:|---:|---|
| `results/falsification/joined_evaluation_table.csv` | 4,501 | 0 | First committed signal/quality join; directly exposes target beside predictors |
| `results/falsification/pass_c_calibration.json` | 40 | 0 | Execution calibration/provenance |
| `scripts/07_falsification_pilot.py` | 235 | 25 | Execution, persistence, assertions, and orchestration changed |
| `src/clusterdrift/falsification/dataset.py` | 127 | 16 | Stricter join/key/forbidden-column checks; adds primary/secondary selectors |
| `src/clusterdrift/falsification/evaluation.py` | 571 | 202 | Material rewrite of the evaluator and addition of clean-inclusive secondary evaluations |
| `src/clusterdrift/falsification/execution/checkpoint.py` | 195 | 1 | Crash recovery, quarantine, and provenance changes |
| `src/clusterdrift/falsification/models.py` | 67 | 4 | Adds inner-fold assertions, hashes, and candidate/fold logging |
| `src/clusterdrift/falsification/protocol.py` | 29 | 0 | Adds prediction-record hash |
| `src/clusterdrift/falsification/verification.py` | 259 | 90 | Substantial preflight/result verification expansion |
| `tests/test_falsification/test_pass_c_preflight.py` | 287 | 0 | New preflight tests |

## Did the scientific decision rule change?

No configuration change was found from preregistration (`8dc7bc8`) through the result commit (`c9280bb`) in `configs/falsification.yaml`, `configs/signals.yaml`, `configs/methods.yaml`, or `configs/preprocessing.yaml`. In particular, the following remained preregistered:

- target `delta_ari`, primary metric MAE, and clean-condition exclusion;
- P0–P5 feature definitions;
- HGBR grid and fixed random seed;
- Ridge alpha grid;
- five-fold grouped inner CV;
- LODO and LOSFO grouping;
- 10,000 bootstrap repetitions and decision thresholds.

The `models.py` diff preserves the fitted HGBR/Ridge candidate grids, scoring, final-fit path, and deterministic tie ordering. It adds exact-five-fold assertions and audit logging. The evaluator rewrite factors per-job fitting, adds row-count assertions and secondary clean-inclusive analyses, and persists richer inner-CV evidence. These are plausibly hardening changes, but their timing prevents treating that explanation as an outcome-blind guarantee.

## Changes after the preflight commit

Diff `ef07a68..c9280bb` is overwhelmingly result generation (201,605 inserted artifact rows). The only source change is 11 lines in `verification.py`: round-trip CSV float parsing, imports for metric recomputation, and recording the preflight commit. No model or feature-selection rule changed in this interval.

## Independent check against implementation bias

`hypothesis_y1_independent_phase7_replication.py` does not import the original falsification evaluator, models, bootstrap, or verification modules. Starting only from the two frozen Pass-A/Pass-B CSVs and the frozen YAML rule, it independently reconstructs the join, grouped evaluations, model tuning, metrics, paired unit bootstrap, and verdict. After correcting a reporting-only dataset/family iteration-order mismatch in the independent bootstrap implementation, it reproduces:

- P0 MAE `0.04670466347898161`;
- P3 MAE `0.06392587674262015`;
- P4 MAE `0.050894704583937404`;
- dataset/family wins and bootstrap probabilities;
- verdict `FAILS_PRIMARY_FALSIFICATION`.

This does not erase the exposure. It materially reduces the probability that the post-label refactor changed the frozen numerical conclusion.

## Required future control

For any new confirmatory experiment, executable evaluation code, model grids, tie rules, eligibility rules, bootstrap unit ordering, and reporting schemas must be committed and hash-frozen before any outcome labels are generated or joined. Later changes must be limited to independently reviewable bug fixes, with both pre-fix and post-fix outputs retained. The corrected Y1 work is explicitly post-hoc and cannot retroactively satisfy this rule.
