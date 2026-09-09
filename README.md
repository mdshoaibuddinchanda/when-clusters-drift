# When Clusters Drift: An Empirical Falsification of Structural-Signal Cluster Degradation Prediction

[![CI](https://github.com/mdshoaibuddinchanda/when-clusters-drift/actions/workflows/ci.yml/badge.svg)](https://github.com/mdshoaibuddinchanda/when-clusters-drift/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Scientific Status](https://img.shields.io/badge/Scientific%20Status-HYPOTHESIS%201%20FALSIFIED-critical.svg)](#verdict-hypothesis-1-falsified)
[![Audit Verdict](https://img.shields.io/badge/Audit%20Verdict-FAILURE__CONFIRMED__STOP__PROJECT-red.svg)](#master-scientific-audit-hypothesis-y1)

> **PRIMARY FALSIFICATION VERDICT: `FAILS_PRIMARY_FALSIFICATION`**  
> **PROJECT STATUS: `FAILURE_CONFIRMED_STOP_PROJECT` (PERMANENTLY STOPPED)**  
>
> In accordance with preregistered open-science protocol and strict stopping rules, this research project has been **formally terminated before Phase 8 (Risk Model)** and **Phases 9–15 (RISA Adaptation)**. Structural soft-clustering signals failed to predict out-of-distribution clustering degradation across datasets, and degraded performance relative to simple data-drift and trivial no-skill baselines.

---

## Executive Summary

The central scientific premise of this project was that **unsupervised, label-free internal structural signals** from soft clustering (membership sharpness/entropy, prototype movement, partition entropy, cluster mass changes) could reliably predict *whether* and *by how much* clustering quality ($\Delta\text{ARI}$) degrades under distribution shift *before* any ground-truth labels arrive.

Through a preregistered, frozen falsification experiment across **12 benchmark datasets, 5 folds, 15 shift conditions (7 families), and 5 seeds ($4,500$ evaluation scenarios)**, this hypothesis was **decisively falsified**:

1. **Structural signals worsened out-of-distribution error**: The full structural feature block ($P4$) achieved a Leave-One-Dataset-Out (LODO) Mean Absolute Error (MAE) of **$0.05089$**, underperforming the baseline Maximum Mean Discrepancy ($D_X$) data-drift block ($P0$, MAE = **$0.04670$**) by **$-8.97\%$** ($R_{04} = -0.0897$).
2. **Win thresholds failed**: $P4$ beat $P0$ on only **5 of 12 datasets** (threshold required $\ge 7$) and **4 of 7 shift families** (threshold required $\ge 5$).
3. **Paired bootstrap confidence interval spans zero**: The 95% bootstrap CI for $\Delta_{04}$ is **$[-0.01185, +0.00244]$**, with a median paired delta of **$-0.000192 \le 0$**.
4. **Grouped median no-skill benchmark outperformed all learned models**: A non-learning baseline that simply predicts the median historical degradation of training datasets achieved an out-of-distribution MAE of **$0.03801$**—beating $P0$, $P3$, and $P4$.
5. **Master audit confirmed negative result**: An independent mathematical re-implementation and a comprehensive methodological audit (fixing non-convergence, duplicate leakage, and Oracle-K class coverage in an isolated corrected replication) confirmed that the hypothesis remains falsified ($P4 = 0.04839$ vs. $P0 = 0.04276$ vs. median no-skill $= 0.03568$).

Consequently, following preregistered decision rules, **the downstream pipeline was stopped**. No downstream risk models or adaptive clustering algorithms were trained. This repository serves as a permanent, cryptographically verified scientific record of the falsification.

---

## The Research Question & Hypothesis

### Research Question
*Under distribution shift $P_t(X) \neq P_0(X)$, can label-free geometric and membership signals internal to Fuzzy C-Means (FCM) predict true partition degradation $\Delta\text{ARI} = \text{ARI}_{\text{clean}} - \text{ARI}_{\text{shifted}}$ on unseen datasets better than raw data-drift measures alone?*

### Formal Hypothesis 1 (Structural Prediction)
$$\mathbb{E}[\widehat{\Delta Q}(D_X, Z_{\text{struct}})] < \mathbb{E}[\widehat{\Delta Q}(D_X)]$$

- $D_X$: Covariate shift measured via Maximum Mean Discrepancy (MMD) with median-heuristic bandwidth computed strictly from source data.
- $Z_{\text{struct}}$: Internal structural soft-clustering signals:
  - $D_U^R$: Aligned membership divergence evaluated on a fixed reference probe.
  - $D_U^C$: Aligned membership divergence evaluated on a contemporaneous current probe.
  - $D_V$: Reference-normalized prototype (cluster center) displacement.
  - $D_H$: Change in normalized partition entropy.
  - $D_M$: Total variation divergence of cluster mass distribution.

### Evaluated Feature Blocks

| Block | Predictors | Description |
|---|---|---|
| **$P0$** | $D_X$ (MMD) | Primary baseline: raw data distribution drift |
| **$P1$** | $D_U^R, D_U^C$ | Membership drift signals only |
| **$P2$** | $D_V, D_H, D_M$ | Prototype and entropy geometry signals only |
| **$P3$** | FPC, PE, XB, Silhouette | Classical internal cluster validity indices |
| **$P4$** | $D_X + Z_{\text{struct}}$ | Full structural signal hypothesis |

Evaluation models were trained using **HistGradientBoostingRegressor** with frozen 5-fold inner Group-CV hyperparameter grids under **Leave-One-Dataset-Out (LODO)** and **Leave-One-Shift-Family-Out (LOSFO)** protocols.

---

## Benchmark Design & Experimental Protocol

The falsification experiment evaluated **4,500 scenarios** across a diverse panel:

- **12 Benchmark Datasets**:
  - Low dimension ($D \le 10$): *Iris* ($D=4$), *Glass* ($D=9$)
  - Medium dimension ($10 < D \le 60$): *Sonar* ($D=60$), *Breast Cancer* ($D=30$), *Spambase* ($D=57$), *Waveform* ($D=40$), *Satimage* ($D=36$), *Pendigits* ($D=16$)
  - High dimension / complex ($D > 60$): *Letter Recognition* ($D=16, K=26$), *Madelon* ($D=500$), *HAR* ($D=561$), *Mice Protein* ($D=77$)
- **5 Outer Folds**: Group-stratified outer folds with strict source-only preprocessing.
- **15 Shift Conditions**:
  - 1 Clean reference ($P_0$)
  - 14 Shifted environments across **7 families** at two severities (mild / severe):
    1. *Scale shift* (variance expansion)
    2. *Location shift* (centroid displacement)
    3. *MCAR shift* (missing completely at random)
    4. *Outliers shift* (heavy-tailed contamination)
    5. *Measurement noise* (additive Gaussian noise)
    6. *Class prevalence shift* (subpopulation imbalance)
    7. *Local overlap shift* (boundary contraction)
- **5 Random Algorithm Seeds**: Per fold and condition ($12 \times 5 \times 15 \times 5 = 4,500$ rows).

---

## Detailed Falsification Findings

### 1. Headline LODO Performance

Under the preregistered Leave-One-Dataset-Out (LODO) evaluation:

| Block | LODO MAE | Relative Improvement vs $P0$ ($R_{0x}$) | Dataset Wins ($\ge 7/12$) | Family Wins ($\ge 5/7$) |
|---|---|---|---|---|
| **$P0$ (MMD Baseline)** | **$0.04670$** | Baseline ($0.0\%$) | — | — |
| **$P1$ (Memberships)** | $0.05193$ | $-11.19\%$ | 4 / 12 | 3 / 7 |
| **$P2$ (Geometry)** | $0.05057$ | $-8.28\%$ | 5 / 12 | 4 / 7 |
| **$P3$ (Validity Indices)**| $0.06393$ | $-36.87\%$ | 2 / 12 | 4 / 7 |
| **$P4$ (Full Structural)** | **$0.05089$** | **$-8.97\%$** | **5 / 12** | **4 / 7** |

```text
Relative Improvement R_04 = -8.97% (Worse than baseline)
Relative Improvement R_34 = +20.38% (Beats classical validity, but loses to MMD)
Paired Bootstrap 95% CI (P0 - P4): [-0.01185, +0.00244]
Median Paired Delta: -0.000192 <= 0
Verdict: FAILS_PRIMARY_FALSIFICATION
```

### 2. Dataset-by-Dataset Breakdown

$P4$ failed to beat $P0$ across the majority of datasets:

| Dataset | $P0$ MAE | $P4$ MAE | $\Delta_{04}$ ($P0 - P4$) | Winner |
|---|---|---|---|---|
| **breast_cancer** | $0.03357$ | $0.02685$ | $+0.00672$ | $P4$ |
| **glass** | $0.03848$ | $0.04250$ | $-0.00402$ | $P0$ |
| **har** | $0.07685$ | $0.06915$ | $+0.00770$ | $P4$ |
| **iris** | $0.02874$ | $0.03195$ | $-0.00321$ | $P0$ |
| **letter** | $0.06979$ | $0.07849$ | $-0.00870$ | $P0$ |
| **madelon** | $0.02058$ | $0.05494$ | $-0.03436$ | $P0$ |
| **mice_protein** | $0.03058$ | $0.03222$ | $-0.00164$ | $P0$ |
| **pendigits** | $0.05342$ | $0.04870$ | $+0.00472$ | $P4$ |
| **satimage** | $0.06013$ | $0.05586$ | $+0.00427$ | $P4$ |
| **sonar** | $0.05510$ | $0.06734$ | $-0.01224$ | $P0$ |
| **spambase** | $0.05436$ | $0.05517$ | $-0.00081$ | $P0$ |
| **waveform** | $0.03886$ | $0.03757$ | $+0.00129$ | $P4$ |
| **Overall Mean** | **$0.04670$** | **$0.05089$** | **$-0.00419$** | **$P0$ Wins (7–5)** |

### 3. Shift-Family Breakdown

$P4$ won only 4 of 7 shift families against $P0$:

| Shift Family | $P0$ MAE | $P4$ MAE | Winner |
|---|---|---|---|
| **Class Prevalence** | $0.04941$ | $0.04620$ | $P4$ |
| **Local Overlap** | $0.04473$ | $0.04018$ | $P4$ |
| **Location** | $0.03923$ | $0.05495$ | $P0$ |
| **MCAR (Missingness)**| $0.04786$ | $0.05607$ | $P0$ |
| **Measurement Noise** | $0.05739$ | $0.04768$ | $P4$ |
| **Outliers** | $0.04603$ | $0.04825$ | $P0$ |
| **Scale** | $0.04229$ | $0.03986$ | $P4$ |

---

## Why Hypothesis 1 Failed: Scientific Root Cause Analysis

A dedicated master audit (`experiments/hypothesis_y1/`) investigated the underlying failure mechanisms:

### 1. Lack of Cross-Dataset Generalizability
While structural signals sometimes correlate with degradation *within* a single dataset (e.g., prototype drift $D_V$ had Spearman $\rho = +0.392$ in *Letter Recognition*), these correlations do not transfer out-of-distribution across datasets. On *Sonar* and *Spambase*, prototype drift had a *negative* correlation with degradation. When tested across unseen datasets under LODO, learned models showed negative out-of-distribution $R^2$ ($P4$ LODO $R^2 = -0.187$), meaning their predictions were worse than a constant mean prediction.

### 2. Signals Track Dataset Geometry Rather Than Degradation Magnitude
Raw structural signal magnitudes are heavily dominated by intrinsic dataset properties:
- Partition entropy change ($D_H$) correlated with source entropy ($\rho = 0.720$).
- Membership divergence ($D_U^R$) scaled with the number of clusters $K$ ($\rho = 0.611$).
- Xie-Beni index ($XB$) scaled with dimensionality $D$ ($\rho = 0.823$).
A tree-based regressor trained on datasets with $D \in [4, 60]$ could not extrapolate degradation magnitude to datasets with $D \in [77, 561]$.

### 3. The Grouped Median No-Skill Baseline Defeated All Learned Models
To test whether *any* learned model possessed genuine predictive skill, the audit implemented a zero-parameter grouped no-skill baseline:
$$\widehat{\Delta\text{ARI}}_{\text{no-skill}} = \text{median}(\Delta\text{ARI}_{\text{train\_datasets}})$$

| Model | LODO MAE | LOSFO MAE |
|---|---|---|
| **Grouped Median No-Skill Baseline** | **$0.03801$** | **$0.03828$** |
| Grouped Mean No-Skill Baseline | $0.04269$ | $0.04316$ |
| Learned $P0$ (MMD Baseline) | $0.04670$ | $0.04840$ |
| Learned $P4$ (Full Structural Signals)| $0.05089$ | $0.04683$ |
| Learned $P3$ (Internal Validity Indices)| $0.06393$ | $0.06212$ |

Because the median historical degradation achieved lower error than all learned regressors, the learned models demonstrated **negative generalization skill**.

### 4. Ranking and Binary Risk Classification Also Failed
To ensure the failure was not merely an artifact of exact regression:
- **Spearman Rank Correlation**: $P4$ ranking correlation across datasets was only **$\rho = 0.042$** (near zero).
- **Binary Failure Detection**: For classifying whether severe degradation occurred ($\Delta\text{ARI} > 0.05$), $P4$ achieved an AUROC of **$0.184$** and balanced accuracy of **$0.500$** (equivalent to random guessing).

---

## Master Scientific Audit (`hypothesis_y1`)

To guarantee absolute scientific integrity, the findings were subjected to an independent master audit:

1. **Independent 1:1 Re-Implementation**: An independent evaluator (`experiments/hypothesis_y1/scripts/hypothesis_y1_independent_phase7_replication.py`) was written without importing any original falsification modules. It reproduced all headline figures, relative improvements, bootstrap intervals, and verdicts to within **$\le 10^{-12}$** numerical tolerance.
2. **Methodological Defect Identification**: The audit audited and isolated three methodological concerns in the pilot execution:
   - *FCM non-convergence*: 635 of 4,500 rows had been accepted without both source and candidate FCM fits converging.
   - *Duplicate leakage*: Exact duplicates crossed outer folds in 4 Phase-7 datasets (*Glass*, *Iris*, *Letter*, *Spambase*).
   - *Oracle-K coverage*: 4 target folds lacked at least one target class.
3. **Isolated Corrected Replication**: A corrected replication was executed with:
   - 600-iteration FCM policy (raising convergence to $98.69\%$),
   - Duplicate-grouped connected-component splitting,
   - Guaranteed target-class coverage,
   - Clean-referenced validity deltas ($P3\text{-delta}$).

**Outcome of the Corrected Replication**:
- Corrected LODO $P0 = 0.04276$
- Corrected LODO $P3\text{-delta} = 0.04378$
- Corrected LODO $P4 = 0.04839$
- Corrected Grouped Median No-Skill $= 0.03568$

Even after fixing every methodological limitation, **$P4$ remained worse than $P0$ and decisively worse than the no-skill baseline**. The negative conclusion was reaffirmed with high confidence:

```text
AUDIT VERDICT: FAILURE_CONFIRMED_STOP_PROJECT
DO NOT CONTINUE ORIGINAL PHASE 8–15 ROADMAP.
```

---

## Project Structure

This repository is a **scientific falsification archive**, not a production machine learning package. Modules corresponding to canceled downstream phases (Phase 8 Risk Model training, Phase 9–15 RISA adaptive clustering) remain inert.

```text
when-clusters-drift/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated CI test suite running on Python 3.12
├── configs/                       # Preregistered scientific configurations
│   ├── alignment.yaml             # Hungarian / Soft Jaccard alignment configuration
│   ├── falsification.yaml         # Preregistered Phase-7 evaluation grid & thresholds
│   ├── methods.yaml               # FCM, GK, PFCM hyperparameters & adaptive fuzzifier
│   ├── preprocessing.yaml         # Source-only scaling & median imputation rules
│   ├── probes.yaml                # Probe bank sizes & selection policies
│   └── shifts.yaml                # 7 shift families and mild/severe severity rules
├── experiments/
│   └── hypothesis_y1/             # Master audit & independent replication
│       ├── artifacts/             # Verified audit metrics, decision matrices, provenance
│       ├── corrected_replication/ # 600-iter, duplicate-corrected replication artifacts
│       ├── reports/               # Complete scientific audit reports:
│       │   ├── hypothesis_y1_master_report.md
│       │   ├── hypothesis_y1_corrected_replication_report.md
│       │   ├── hypothesis_y1_convergence_root_cause.md
│       │   ├── hypothesis_y1_post_label_change_audit.md
│       │   └── hypothesis_y1_security_integrity_audit.md
│       ├── scripts/               # Independent reproduction scripts
│       └── tests/                 # Executable audit guards and invariant tests
├── results/
│   └── falsification/             # Permanently frozen Phase-7 outputs
│       ├── signals_label_free.csv # 4,500 label-free structural signal rows
│       ├── quality_evaluation_only.csv # 4,500 quality evaluation rows (Delta ARI)
│       └── joined_evaluation_table.csv # Joined 4,500-row primary evaluation table
├── src/
│   └── clusterdrift/              # Core scientific algorithms
│       ├── alignment/             # Cluster alignment under permutation
│       ├── data/                  # Dataset loaders & integrity checks
│       ├── falsification/         # Evaluation engines & cache managers
│       ├── methods/               # FCM, Gustafson-Kessel, GMM implementations
│       ├── metrics/               # Hard/soft ARI, NMI, external recovery metrics
│       ├── probes/                # Current & reference probe selection
│       ├── shifts/                # Unsupervised shift injection engine
│       └── signals/               # Structural signal extractors (D_U, D_V, D_H, D_M, MMD)
├── tests/                         # 400+ unit, integration, and immutability tests
├── reproduce.py                   # One-command replication entry point
├── requirements.txt               # Pinned dependencies for Python 3.12
└── pyproject.toml                 # Package definition and build configuration
```

---

## Installation & Replication

### Requirements
- **Python**: `3.10`, `3.11`, or `3.12` (Python 3.12 recommended; tested on Windows and Linux)
- **Conda / Virtualenv**: Recommended

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/mdshoaibuddinchanda/when-clusters-drift.git
cd when-clusters-drift

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install exact dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Fast Falsification Replication (< 1 second)

To instantly verify the frozen Phase-7 results and re-evaluate the falsification verdict:

```bash
python reproduce.py
```

Expected output:
```json
{
  "agreement": "PASSED",
  "verdict": "FAILS_PRIMARY_FALSIFICATION",
  "LODO": {
    "P0": 0.046704663478981584,
    "P3": 0.06392587674262011,
    "P4": 0.05089470458393738
  },
  "runtime_seconds": 0.29
}
```

### 3. Run Automated Tests

To execute the test suite (all tests passing on Python 3.12):

```bash
# Run falsification and audit test suite
pytest tests/test_falsification/ experiments/hypothesis_y1/tests/ -v

# Run core algorithm, alignment, probe, and signal tests
pytest tests/test_alignment/ tests/test_methods/ tests/test_signals/ tests/test_probes/ -v
```

---

## Cryptographic Artifact Lineage

All Phase-7 artifacts are cryptographically bound to prevent retrospective modification or cherry-picking:

| Artifact | SHA-256 Checksum | Rows | Columns |
|---|---|---|---|
| `signals_label_free.csv` | `3a9e6c68cbe34c8763d70769715dba0d34976dfc35c8877c5680561da5c0e80a` | 4,500 | 59 |
| `quality_evaluation_only.csv` | `1b3873bd92233702e761791e7b0f4c3e5f5a9f3f41320641fa4105e11c6a5fa5` | 4,500 | 25 |
| `falsification.yaml` | `f58cb23e20ec4cf7450fa2e3c0420df20ef93fc0c5be65bf2d89fa30fa1432ee` | — | — |

The protocol SHA-256 fingerprint remains:
```text
8dc7bc8056a686f1eb147f9ec5bf211935454da6
```

---

## Open Science Statement

Null and negative results are foundational to empirical scientific progress. Rather than altering hypotheses post-hoc, switching to convenient subsets of datasets or shifts, or p-hacking evaluation thresholds to salvage a positive claim, this study adheres strictly to its preregistered protocol. 

The negative finding that **unlabeled internal soft-clustering signals do not transfer out-of-distribution to predict hard-clustering degradation** is preserved in full fidelity to inform future research in unsupervised distribution shift and test-time adaptation.

---

## License

This repository and all original source code are licensed under the [Apache-2.0 License](LICENSE).
