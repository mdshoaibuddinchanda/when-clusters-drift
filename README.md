# When Clusters Drift: Unsupervised Prediction of Cluster Degradation Under Distribution Shift

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Scientific Status](https://img.shields.io/badge/Scientific%20Status-HYPOTHESIS%201%20FALSIFIED-critical.svg)](#why-stop-why-did-we-stop-the-project)
[![Audit Verdict](https://img.shields.io/badge/Audit%20Verdict-FAILURE__CONFIRMED__STOP__PROJECT-red.svg)](#the-authoritative-audit-branch-hypothesis-y1-master-audit)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

> [!CAUTION]
> ### SCIENTIFIC STATUS: FORMALLY TERMINATED (NEGATIVE RESULT CONFIRMED)
> **Authoritative Audit & Replication Branch:** [`hypothesis-y1-master-audit`](https://github.com/mdshoaibuddinchanda/when-clusters-drift/tree/hypothesis-y1-master-audit)
>
> In accordance with strict preregistered stopping rules, this research project was **permanently terminated at Phase 7**.
> The core scientific hypothesis—that unsupervised, internal structural signals from soft clustering can forecast downstream partition degradation ($\Delta\text{ARI}$) under distribution shift—was **empirically falsified**.
>
> All frozen evaluation matrices ($4,500$ scenarios), independent mathematical replications, root-cause analyses, and reproduction entry points are preserved on the dedicated branch:
>
> 👉 **[Switch to `hypothesis-y1-master-audit` Branch](https://github.com/mdshoaibuddinchanda/when-clusters-drift/tree/hypothesis-y1-master-audit)**

---

## The Main Project Idea

### The Core Problem
In real-world machine learning deployments, data distributions continuously evolve due to environmental drift, sensor aging, demographic shifts, and temporal fluctuations ($P_t(X) \neq P_0(X)$).

While supervised systems eventually receive delayed labels or ground truth feedback to detect performance loss, **unsupervised clustering algorithms (e.g., Fuzzy C-Means, Gustafson-Kessel, GMM, K-Means) operate in complete label-free isolation**. When distributions drift, cluster assignments silently degrade, cluster boundaries collapse, and downstream decision systems fail without any warning signal.

### The Research Question
> **Can label-free internal geometric and soft-clustering structural signals detect out-of-distribution shift and accurately predict clustering degradation ($\Delta\text{ARI} = \text{ARI}_{\text{clean}} - \text{ARI}_{\text{shifted}}$) across unseen datasets and shift mechanisms, better than raw covariate data-drift measures alone?**

### The Proposed Architecture
The project envisioned a 15-phase research lifecycle to develop **RISA (Risk-Informed Shift Adaptation)**:

1. **Dual-Probe Bank Framework**: Contrasting a static reference probe bank ($D_U^R$) against a contemporaneous current probe bank ($D_U^C$) to isolate algorithm-induced artifacts from genuine geometric manifold shifts.
2. **Internal Structural Signals ($Z_{\text{struct}}$)**:
   - **$D_U$ (Membership Drift)**: Hungarian-aligned soft membership divergence across probe points.
   - **$D_V$ (Prototype Displacement)**: Centroid movement normalized by reference cluster scale.
   - **$D_H$ (Entropy Variation)**: Relative change in partition entropy.
   - **$D_M$ (Cluster Mass Shift)**: Total variation divergence in cluster point allocations.
3. **Preregistered Feature Blocks**:
   - **$P0$ (Data Drift Baseline)**: Maximum Mean Discrepancy (MMD, $D_X$) on raw feature space.
   - **$P1$ (Memberships)**: $D_U^R, D_U^C$.
   - **$P2$ (Geometry)**: $D_V, D_H, D_M$.
   - **$P3$ (Classical Validity)**: Partition Coefficient, Partition Entropy, Xie-Beni index, Silhouette.
   - **$P4$ (The Structural Hypothesis)**: $D_X + Z_{\text{struct}}$ (combining data drift and all internal structural signals).
4. **Planned Downstream Pipeline (Canceled)**:
   - *Phase 8*: Downstream Risk Estimator predicting continuous degradation and catastrophic binary failure ($\Delta\text{ARI} > 0.05$).
   - *Phases 9–15*: RISA active adaptation engine triggering adaptive reclustering conditioned on predicted risk bounds.

---

## Why Stop? Why Would We Stop The Project?

A common pitfall in machine learning research is continuing to iterate, modify hyperparameters, or selectively prune datasets until an artificial "positive" result emerges. This project committed from inception to **strict preregistered open science**. 

The project was stopped permanently because:

### 1. Preregistered Stopping Rule
The experimental protocol established a binding hard gate at Phase 7: if the structural feature block ($P4$) failed to achieve statistically significant superior predictive power over the simple raw data-drift baseline ($P0$) under Leave-One-Dataset-Out (LODO) cross-validation, **the hypothesis was falsified and all downstream work had to halt immediately**.

### 2. The Empirical Falsification (Phase 7 Results)
Across a frozen panel of **12 benchmark datasets, 5 outer folds, 15 shift conditions (7 shift families), and 5 algorithm seeds ($4,500$ evaluation scenarios)**:

| Evaluation Metric | Preregistered Success Threshold | Actual Empirical Result | Status |
|---|---|---|---|
| **LODO MAE Relative Improvement ($R_{04}$)** | $> +5.0\%$ improvement over $P0$ | **$-8.97\%$** ($0.04670 \to 0.05089$) | **FALSIFIED** |
| **Dataset Win Ratio** | $P4$ beats $P0$ on $\ge 7 / 12$ datasets | **$5 / 12$ datasets** ($41.7\%$) | **FALSIFIED** |
| **Shift Family Win Ratio** | $P4$ beats $P0$ on $\ge 5 / 7$ families | **$4 / 7$ families** ($57.1\%$) | **FALSIFIED** |
| **Paired Bootstrap 95% CI** | Lower bound strictly $> 0$ | **$[-0.01185, +0.00244]$** | **FALSIFIED** |
| **No-Skill Benchmark Comparison** | Must beat naive baseline | Lost to simple dataset median ($0.03801$) | **FALSIFIED** |
| **Binary Failure Detection AUROC** | $\text{AUROC} \ge 0.70$ | **$\text{AUROC} = 0.184$** (worse than random) | **FALSIFIED** |

```text
EVALUATION VERDICT: FAILS_PRIMARY_FALSIFICATION
DECISION: HALT RESEARCH. DO NOT PROCEED TO PHASE 8.
```

### 3. Why Did Soft Structural Signals Fail? (Root Cause Analysis)
Under real and synthetic distribution shifts (variance scaling, boundary overlap, heavy-tailed outliers, missingness):
- **Centroid Displacement Noise**: Soft-clustering centroids ($V$) shift unpredictably when boundaries deform, introducing high-variance geometric noise that fails to generalize across heterogeneous dataset spaces.
- **Membership Entropy Inflation/Deflation**: In regions of local cluster overlap, membership sharpness flattens uniformly, producing high entropy shifts that do not correlate with partition degradation.
- **Cross-Dataset Scale Invariance Broken**: What constitutes a "large" prototype shift in a 4-dimensional space (*Iris*) is numerically incomparable to a prototype shift in a 500-dimensional space (*Madelon*), causing cross-dataset regression models to overfit spurious signal dynamics.
- **MMD Outperformed Soft Signals**: A simple two-sample kernel distance ($D_X$) on raw data was far more stable and generalizable than complex clustering internals.

### 4. Scientific Integrity vs. P-Hacking / HARKing
Proceeding to train downstream risk models (Phase 8) or adaptive clustering algorithms (Phases 9–15) on top of a non-predictive signal would have built a **house of cards**. Rather than engaging in HARKing (Hypothesizing After Results are Known) or torturing the benchmark to engineer a misleading positive publication, we uphold scientific integrity by publishing and archiving the complete falsification.

---

## The Authoritative Audit Branch: `hypothesis-y1-master-audit`

To provide absolute, verifiable scientific proof, an independent master audit and replication was conducted on the branch [`hypothesis-y1-master-audit`](https://github.com/mdshoaibuddinchanda/when-clusters-drift/tree/hypothesis-y1-master-audit).

This branch contains:
1. **One-Command Reproduction (`reproduce.py`)**: An instant (<0.5 second) evaluation script that recomputes all headline metrics and validates the falsification verdict against frozen arrays.
2. **Independent 1:1 Clean Re-Implementation**: `experiments/hypothesis_y1/scripts/hypothesis_y1_independent_phase7_replication.py` independently reproduces every result to $\le 10^{-12}$ numerical tolerance without importing project internals.
3. **Corrected Replication**: A second, hardened replication that resolved all identified edge cases (raising FCM convergence to $98.69\%$ with 600 iterations, isolating duplicate leakage across folds, and enforcing Oracle-K coverage). The conclusion was unchanged: **$P4$ ($0.04839$) remained inferior to $P0$ ($0.04276$) and lost decisively to the no-skill median baseline ($0.03568$)**.
4. **Complete Audit Reports**:
   - `hypothesis_y1_master_report.md`: Complete audit findings and mathematical proofs.
   - `hypothesis_y1_corrected_replication_report.md`: Independent corrected execution.
   - `hypothesis_y1_convergence_root_cause.md`: Optimization convergence diagnostics.
   - `hypothesis_y1_security_integrity_audit.md`: Cryptographic provenance and security audits.
5. **Continuous Integration**: Complete GitHub Actions CI pipeline running across Ubuntu and Windows on Python 3.12.

---

## Repository Structure (`main` branch)

```text
when-clusters-drift/
├── configs/                       # Preregistered scientific configurations
│   ├── alignment.yaml             # Hungarian & soft Jaccard alignment specs
│   ├── falsification.yaml         # Preregistered Phase-7 evaluation thresholds
│   ├── methods.yaml               # FCM, GK, PFCM hyperparameters & fuzzifier
│   ├── preprocessing.yaml         # Source-only scaling & median imputation rules
│   ├── probes.yaml                # Probe bank sizing & selection rules
│   └── shifts.yaml                # 7 shift families and mild/severe severity rules
├── docs/                          # Core research documentation
│   ├── research_question.md       # Primary and secondary research questions
│   ├── hypotheses.md              # Formal mathematical hypotheses & falsification rules
│   ├── research_protocol.md       # 7-phase experimental lifecycle
│   ├── final_experiment_lock.md   # Factorial design (4,500 scenarios)
│   ├── leakage_policy.md          # Zero-label access and preprocessing isolation
│   ├── novelty_collision_matrix.md# Comparison with existing literature
│   └── literature_matrix.csv      # Foundational literature references
├── src/
│   └── clusterdrift/              # Core algorithm library
│       ├── alignment/             # Cluster permutation alignment
│       ├── data/                  # Dataset loaders & integrity verifiers
│       ├── falsification/         # Evaluation engines & cache verification
│       ├── methods/               # FCM, Gustafson-Kessel, GMM, K-Means
│       ├── metrics/               # Hard/soft ARI, NMI, external metrics
│       ├── probes/                # Reference & current probe selection
│       ├── shifts/                # Unsupervised shift injection engine
│       └── signals/               # Structural signal extractors (D_U, D_V, D_H, D_M)
├── tests/                         # Unit and integration test suite
├── .gitignore                     # Strict exclusion of all binary data files
├── .gitattributes                 # Cross-platform LF/CRLF and binary safeguards
├── requirements.txt               # Pinned dependencies for Python 3.12
└── pyproject.toml                 # Package definition
```

---

## How to Switch to the Audit Branch & Reproduce

To review the full experimental data, run the master audit, and reproduce the falsification:

```bash
# Clone repository
git clone https://github.com/mdshoaibuddinchanda/when-clusters-drift.git
cd when-clusters-drift

# Switch to the authoritative audit branch
git checkout hypothesis-y1-master-audit

# Set up Python 3.12 environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Run instant falsification verification (<0.5 seconds)
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

---

## Open Science & Value of Negative Results

Null and negative results are indispensable to scientific progress. Publishing only positive anomalies skews scientific literature, encourages p-hacking, and wastes research resources pursuing dead ends. 

By preserving and documenting this negative result with cryptographic rigor, this repository:
1. **Prevents duplicated effort**: Demonstrates that internal soft-clustering drift signals do not transfer out-of-distribution to predict partition collapse.
2. **Highlights the resilience of simple baselines**: Establishes that input-space Maximum Mean Discrepancy ($D_X$) and simple historical dataset statistics remain far superior to complex internal structural signals for degradation awareness.
3. **Provides a benchmark suite**: Supplies an open, leakage-free shift-injection framework (7 families, 2 severities) and probe bank methodology for future unsupervised drift research.

---

## License

This repository and all original source code are licensed under the [Apache-2.0 License](LICENSE).
