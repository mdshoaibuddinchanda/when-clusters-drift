# Final Experiment Lock (Phase 7 Falsification Gate)

## Scope of Evaluation

The Phase-7 structural-signal falsification experiment was frozen across a complete factorial grid:

- **12 Benchmark Datasets**:
  - Small / Low-D: *Iris*, *Glass*
  - Medium-D: *Sonar*, *Breast Cancer*, *Spambase*, *Waveform*, *Satimage*, *Pendigits*
  - High-D / Complex: *Letter Recognition*, *Madelon*, *Human Activity Recognition (HAR)*, *Mice Protein Expression*
- **5 Grouped Outer Folds**: Strict source/candidate split per dataset.
- **15 Shift Conditions**:
  - 1 Clean reference ($P_0$)
  - 14 Shift conditions across **7 distinct shift families** at two severities (`mild`, `severe`):
    1. *Scale shift* (variance scaling)
    2. *Location shift* (centroid displacement)
    3. *MCAR shift* (missingness completely at random)
    4. *Outlier shift* (heavy-tailed synthetic contamination)
    5. *Measurement noise* (additive Gaussian perturbation)
    6. *Class prevalence shift* (subpopulation ratio perturbation)
    7. *Local overlap shift* (inter-cluster margin contraction)
- **5 Algorithm Seeds**: Random initializations per condition.
- **Total Scenarios Evaluated**: $12 \times 5 \times 15 \times 5 = 4,500$ rows.

---

## Preregistered Regressors & Validation Protocols

- **Estimator**: `HistGradientBoostingRegressor`
- **Inner Tuning**: 5-fold inner Group-CV over frozen hyperparameter grids (learning rate, max depth, l2 regularization).
- **Outer Validation Schemes**:
  1. **Leave-One-Dataset-Out (LODO)**: Model is evaluated on an entirely held-out dataset, testing transfer across data domains.
  2. **Leave-One-Shift-Family-Out (LOSFO)**: Model is evaluated on an unseen shift family, testing transfer across perturbation mechanisms.

---

## Frozen Cryptographic Artifacts

All artifact hashes are cryptographically verified:

| Artifact | Location | Rows | Checksum (SHA-256) |
|---|---|---|---|
| `signals_label_free.csv` | `results/falsification/` | 4,500 | `3a9e6c68cbe34c8763d70769715dba0d34976dfc35c8877c5680561da5c0e80a` |
| `quality_evaluation_only.csv` | `results/falsification/` | 4,500 | `1b3873bd92233702e761791e7b0f4c3e5f5a9f3f41320641fa4105e11c6a5fa5` |
| `joined_evaluation_table.csv` | `results/falsification/` | 4,500 | Verified combined table |

---

## Gate Verdict

```text
FINAL EVALUATION VERDICT: FAILS_PRIMARY_FALSIFICATION
DOWNSTREAM ACTION: TERMINATE PROJECT BEFORE PHASE 8
```

Detailed reproduction script and audit reports are preserved on branch [`hypothesis-y1-master-audit`](https://github.com/mdshoaibuddinchanda/when-clusters-drift/tree/hypothesis-y1-master-audit).
