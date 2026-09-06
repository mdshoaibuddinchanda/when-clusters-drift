# Fuzzifier Policy Freeze and Soft-Model Validity Gate (Phase 3.2)

## 1. Executive Summary & Research Freeze
This document records the formal freeze of the clustering baseline fuzzifier policy prior to beginning Phase 4 (distribution shift experiments and RISA-FCM).

### Research Integrity Notice
The original experimental pre-registration planned:
- Primary baseline: fixed $m = 2.0$
- Sensitivity analysis: $m \in \{1.5, 1.75, 2.0, 2.25, 2.5\}$

This plan was **falsified** during Phase 3 and Phase 3.1 validation audits. Empirical and theoretical analyses established that fixed $m = 2.0$ causes widespread source-model prototype collapse and near-uniform partitions ($FPC \approx 1/K, PE \approx \ln(K)$), including on the nominally well-separated benchmark `s01_balanced_gmm` ($D=10$).

Because no distribution-shift perturbations or RISA adaptation experiments have begun, updating this policy prior to Phase 4 constitutes rigorous research integrity rather than post-hoc benchmark tuning.

---

## 2. Mathematical Foundation: The Curse of Fuzziness
The standard Fuzzy C-Means (FCM) objective is:
$$J_m(U, V) = \sum_{i=1}^N \sum_{k=1}^K u_{ik}^m \|x_i - v_k\|_2^2, \quad \text{subject to } \sum_{k=1}^K u_{ik} = 1, \; u_{ik} \ge 0$$

### The Objective Collapse at $m=2$
As proven by Winkler, Klawonn, and Kruse (2010, 2011), as dimensionality $D$ increases or relative distance variance shrinks:
1. When prototypes coincide at the grand sample centroid $\bar{x} = \frac{1}{N}\sum_i x_i$, the uniform membership partition $u_{ik} = \frac{1}{K}$ yields an objective value:
   $$J_m(\text{grand centroid}) = K^{1-m} \sum_{i=1}^N \|x_i - \bar{x}\|_2^2$$
2. For $K=3, m=2$:
   $$J_2(\text{grand centroid}) = \frac{1}{3} \sum_{i=1}^N \|x_i - \bar{x}\|_2^2 = \frac{1}{3} S_{\text{tot}}$$
3. For $m \ge 1.5$ on `s01_balanced_gmm` ($D=10$), this uniform grand-centroid state achieves a strictly lower objective $J_m$ than any separated prototype configuration. Consequently, gradient descent actively pulls prototypes toward $\bar{x}$ regardless of prototype initialization.

---

## 3. The Literature-Grounded Dimension-Adaptive Rule
To preserve meaningful soft partitions in higher dimensions without tuning on labels, we implement the published dimension-dependent fuzzifier rule:
$$m_D = 1 + \frac{2}{D}$$
where $D$ is the number of features after Phase-2 source-only preprocessing.

### Literature References
- **Winkler, R., Klawonn, F., & Kruse, R. (2010)**. *Problems of Fuzzy c-Means Clustering and Similar Algorithms with High Dimensional Data Sets*. In: Foundations of Reasoning under Uncertainty, pp. 79–96.
- **Winkler, R., Klawonn, F., & Kruse, R. (2011)**. *Fuzzy c-means in high dimensional spaces*. International Journal of Fuzzy Systems, 13(1), 1–6.

### Numerical Floor & Log-Domain Stabilization
As $D \to \infty$, $m_D \to 1.0$, causing the exponent $\beta = \frac{2}{m-1}$ to diverge ($\beta \to \infty$). To guarantee machine precision and avoid floating-point overflow:
1. **Numerical Floor**:
   $$m_{\text{effective}} = \max\left(1.01, 1 + \frac{2}{D}\right)$$
   If $1 + 2/D < 1.01$ (e.g. $D \ge 201$), $m$ is clipped to $1.01$ and metadata flag `fuzzifier_clipped_ = True` is recorded.
2. **Log-Domain Softmax**:
   For non-coincident points ($d_{ik} > 0$):
   $$\log w_{ik} = -\frac{2}{m-1} \log d_{ik}, \quad M_i = \max_{k} \log w_{ik}$$
   $$u_{ik} = \frac{\exp(\log w_{ik} - M_i)}{\sum_{j=1}^K \exp(\log w_{ij} - M_i)}$$
   This guarantees $\sum_k u_{ik} = 1.0 \pm 10^{-14}$ to floating-point precision without numerical overflow.
3. **Exact Coincidence**: If any $d_{ik} = 0$, coincident prototypes receive $1/c$ and non-coincident receive $0.0$.

---

## 4. Policy Comparison & Empirical Validation

### s01_balanced_gmm Objective Landscape
Evaluated on `s01_balanced_gmm` ($N=8000, D=10, K=3$):

| $m$ | $J_{\text{init}}$ (Separated) | $J_{\text{converged}}$ | $J_{\text{grand}}$ (Centroid) | Grand Centroid Lower? | Partition Status | Final FPC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.10** | 83,849.19 | 61,913.70 | 71,676.68 | False | Non-degenerate | 0.9392 |
| **1.20** ($m_D$) | 79,815.41 | 59,842.24 | 64,219.32 | False | Non-degenerate | 0.7795 |
| **1.30** | 74,484.21 | 56,064.91 | 57,537.85 | False | Non-degenerate | 0.6017 |
| **1.40** | 68,653.14 | 51,343.96 | 51,551.52 | False | Non-degenerate | 0.4485 |
| **1.50** | 62,782.68 | 46,188.02 | 46,188.02 | **True** | **DEGENERATE** | 0.3333 |
| **1.75** | 49,256.85 | 35,095.31 | 35,095.31 | **True** | **DEGENERATE** | 0.3333 |
| **2.00** | 38,111.39 | 26,666.67 | 26,666.67 | **True** | **DEGENERATE** | 0.3333 |

### Panel Summary (11 Datasets $\times$ 5 Seeds = 55 Runs)
- **Fixed $m=2.0$ Control**:
  - Degenerate rate: **72.73% (40 / 55 runs collapsed)**
  - Mean ARI across all runs: **0.2803**
- **Dimension-Adaptive $m_D = 1 + 2/D$**:
  - Degenerate rate: **0.00% (0 / 55 runs collapsed)**
  - Mean ARI across all runs: **0.4650**

---

## 5. Scope & Fair Method Comparisons
1. **FCM**: The dimension-adaptive rule is literature-grounded and will serve as the primary source model for structural drift signals and RISA-FCM.
2. **Fixed $m=2.0$ Control (`fcm_fixed_m2`)**: Preserved in experimental metadata as an explicit baseline control condition.
3. **Gustafson-Kessel (GK)**: Evaluated as a controlled parameter policy using the same exponent structure.
4. **Pal et al. PFCM**: Distinct from polynomial FCM. Warm-starts with matching effective $m$.
5. **Phase 4 Consistency**: The primary RISA-FCM comparison must use the exact same dimension-adaptive source fuzzifier policy as the FCM baseline.
