# Formal Scientific Hypotheses

## Primary Hypothesis: Structural Degradation Prediction ($H_1$)

Let $D_X$ denote the Maximum Mean Discrepancy (MMD) measuring raw covariate distribution shift between reference source data $X_{\text{ref}}$ and candidate target data $X_{\text{tgt}}$. Let $Z_{\text{struct}} = [D_U^R, D_U^C, D_V, D_H, D_M]$ represent the vector of label-free internal soft-clustering structural signals extracted from Fuzzy C-Means (FCM) and cluster probe banks.

Let $\Delta Q = \text{ARI}_{\text{clean}} - \text{ARI}_{\text{shifted}}$ denote the true partition quality degradation measured on unseen evaluation ground truth.

### Hypothesis 1 Formulation
The addition of internal structural clustering signals $Z_{\text{struct}}$ to the raw data-drift feature $D_X$ strictly reduces out-of-distribution prediction error of partition degradation $\Delta Q$ across novel datasets:

$$\mathbb{E}_{(D, S) \sim \mathcal{P}_{\text{test}}} \left[ \mathcal{L}\left(\widehat{\Delta Q}(D_X, Z_{\text{struct}}), \Delta Q\right) \right] < \mathbb{E}_{(D, S) \sim \mathcal{P}_{\text{test}}} \left[ \mathcal{L}\left(\widehat{\Delta Q}(D_X), \Delta Q\right) \right]$$

where $\mathcal{L}$ is Mean Absolute Error (MAE) evaluated under Leave-One-Dataset-Out (LODO) cross-validation.

---

## Feature Block Hierarchy

| Feature Block | Predictors | Description |
|---|---|---|
| **$P0$ (Baseline)** | $D_X$ (MMD) | Raw input space distribution drift |
| **$P1$ (Membership)**| $D_U^R, D_U^C$ | Soft membership divergence on reference and current probes |
| **$P2$ (Geometry)** | $D_V, D_H, D_M$ | Prototype shift, partition entropy change, cluster mass divergence |
| **$P3$ (Classical)** | FPC, PE, XB, Silhouette | Standard internal cluster validity indices |
| **$P4$ (Hypothesis)**| $D_X + Z_{\text{struct}}$ | Full combined structural signal hypothesis ($D_X + P1 + P2$) |

---

## Preregistered Falsification Criteria

Hypothesis 1 is **FALSIFIED** if ANY of the following primary conditions occur:
1. **LODO Relative Improvement**: $R_{04} = \frac{\text{MAE}(P0) - \text{MAE}(P4)}{\text{MAE}(P0)} \le 0.05$ (must exceed 5% improvement).
2. **Dataset Win Ratio**: $P4$ beats $P0$ on fewer than $7$ of the $12$ evaluation datasets ($< 58.3\%$).
3. **Shift Family Win Ratio**: $P4$ beats $P0$ on fewer than $5$ of the $7$ shift families ($< 71.4\%$).
4. **Statistical Significance**: The 95% paired bootstrap confidence interval of $(P0 - P4)$ includes or falls below zero ($\text{CI}_{\text{lower}} \le 0$).

---

## Empirical Outcome & Verdict

| Criterion | Preregistered Gate | Empirical Result | Status |
|---|---|---|---|
| Relative Improvement ($R_{04}$) | $> +5.0\%$ | **$-8.97\%$** ($0.04670 \to 0.05089$) | **FAILED** |
| Dataset Wins | $\ge 7 / 12$ | **$5 / 12$** | **FAILED** |
| Shift Family Wins | $\ge 5 / 7$ | **$4 / 7$** | **FAILED** |
| Bootstrap 95% CI | Strictly $> 0$ | **$[-0.01185, +0.00244]$** | **FAILED** |
| No-Skill Benchmark | Must beat median | Lost to median no-skill ($0.03801$) | **FAILED** |

### Final Verdict: `FAILS_PRIMARY_FALSIFICATION`
Hypothesis 1 was conclusively falsified. Structural signals internal to soft clustering did not generalize across datasets and worsened degradation prediction relative to raw data-drift alone.

For full audit logs, see branch [`hypothesis-y1-master-audit`](https://github.com/mdshoaibuddinchanda/when-clusters-drift/tree/hypothesis-y1-master-audit).
