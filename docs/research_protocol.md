# Research Protocol & Execution Architecture

## 1. Multi-Phase Research Lifecycle

The `when-clusters-drift` investigation was structured into discrete, sequential, and reproducible experimental phases:

```
[Phase 1: Dataset Curation & Integrity Verification]
       ↓
[Phase 2: Split Locking & Preprocessing Audit]
       ↓
[Phase 3: Shift Injection Engine (7 Families, 2 Severities)]
       ↓
[Phase 4: Dual-Probe Bank Generation (Reference vs. Current)]
       ↓
[Phase 5: Structural Signal Extraction (D_U, D_V, D_H, D_M, D_X)]
       ↓
[Phase 6: Cryptographic Result Freezing & Pass-A / Pass-B Verification]
       ↓
[Phase 7: Preregistered Falsification Gate (Pass C)]
       ↓
    [GATE: Pass C Evaluation]
       ├── If PASS: Proceed to Phase 8 (Risk Estimator) & Phases 9-15 (RISA Adaptation)
       └── If FAIL: STOP PROJECT. Halt all downstream phases. Document negative result.
```

---

## 2. Leakage and Blinding Protocols

1. **Strict Zero-Label Access**: Signal extraction modules (`src/clusterdrift/signals/`) and probe bank selectors (`src/clusterdrift/probes/`) operate with absolute zero access to ground truth labels $y$.
2. **Two-Pass Separation**:
   - **Pass A (Signals Only)**: Structural signals $Z_{\text{struct}}$ and data-drift $D_X$ were extracted and cryptographically frozen into `signals_label_free.csv` without inspecting downstream performance.
   - **Pass B (Quality Only)**: Downstream clustering degradation $\Delta\text{ARI}$ was evaluated in a separate, isolated process and frozen into `quality_evaluation_only.csv`.
   - **Pass C (Falsification)**: The tables were unblinded, joined, and evaluated against preregistered thresholds.
3. **Source-Only Preprocessing**: Imputation parameters and scaling statistics (standardization, min-max) were strictly fit on reference source data and transformed out-of-sample onto candidate data.

---

## 3. Stopping Decision & Master Audit

At the conclusion of Phase 7, the preregistered falsification gate produced `FAILS_PRIMARY_FALSIFICATION`.

Per the research protocol:
- **No ad-hoc redesign was permitted.**
- **No feature tinkering or selective dataset exclusion was permitted.**
- **The project halted immediately.**
- An independent master audit was conducted on branch [`hypothesis-y1-master-audit`](https://github.com/mdshoaibuddinchanda/when-clusters-drift/tree/hypothesis-y1-master-audit), confirming that the failure was robust and definitive.
