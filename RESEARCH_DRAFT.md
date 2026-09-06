# When Clusters Drift: Label-Free Failure Prediction and Risk-Guided Adaptation for Soft Clustering

**Working Title**: *When Clusters Drift: Label-Free Failure Prediction and Risk-Guided Adaptation for Soft Clustering*  
**Repository**: `when-clusters-drift`  
**Python Package**: `clusterdrift`  
**Core Algorithm**: `RISA-FCM` (*Risk-Informed Selective Adaptation Fuzzy C-Means*)  
**Target Venue Standard**: Pattern Recognition (PR) / Top-Tier ML Benchmarking Standard  

---

## 1. Paper Identity & Core Narrative

### Title Decomposition

The paper title conveys the complete technical and methodological narrative:

- **When Clusters Drift**: The central problem domain—unsupervised distribution shift where ground-truth labels are absent.
- **Label-Free**: The real-world deployment operational constraint—zero target-domain labels are available for evaluation or model selection.
- **Failure Prediction**: The fundamental research gap—predicting *whether* and *by how much* clustering degrades before/without labels.
- **Risk-Guided Adaptation**: The algorithmic contribution—using predicted degradation risk to continuously steer the stability–plasticity trade-off.
- **Soft Clustering**: Broader framing than a single fuzzy heuristic; establishes general applicability across soft/probabilistic partition representations.

### Framing Rationale

- Terminology deliberately adopts **risk-guided adaptation** rather than "selective adaptation" to distinguish from test-time adaptation (TTA) in supervised vision/NLP literature (e.g., ICML 2025).
- Focuses specifically on soft clustering (memberships, entropy, overlap) where geometric and distributional uncertainty provide diagnostic failure signals impossible in hard clustering.

---

## 2. Research Context, Gap & Surviving Scope

Recent literature establishes several neighboring domains as closed or saturated:

1. **Feature-weighted fuzzy clustering**: Saturated (e.g., 2025 Neurocomputing review benchmarked 26 algorithms with Friedman/Holm statistics).
2. **Soft clustering evaluation frameworks**: Established (e.g., 2023 optimal transport and distributional representation metrics).
3. **Dynamic FCM**: Established (e.g., 2025 I-DFCM with 6 dynamic validity indices for evolving cluster counts).
4. **Tabular distribution shift benchmarks**: Covered in supervised learning by TableShift (15 real-world shift tasks).
5. **Supervised performance prediction under shift**: Demonstrated that raw distribution metrics (e.g., Maximum Mean Discrepancy, MMD) do not reliably track downstream performance loss (Guillory et al., ICCV 2021).
6. **Theoretical foundations of fuzzy clustering**: Maturing rapidly (e.g., 2026 WFCM preprint establishes consistency, asymptotic normality, likelihood-ratio testing, and bootstrap uncertainty).

### The Surviving Core Research Questions

```
+-----------------------------------------------------------------------------------+
|  Can unlabeled internal soft-clustering changes predict actual future clustering  |
|  degradation?                                                                     |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|  Can predicted degradation determine how strongly the clustering system should    |
|  adapt?                                                                           |
+-----------------------------------------------------------------------------------+
```

### Research Thesis

At training time $t=0$:
$$X_0 \sim P_0(X) \implies \text{learn soft clustering } \Theta_0 = (U_0, V_0)$$

At deployment time $t$:
$$X_t \sim P_t(X) \quad \text{with } P_t(X) \neq P_0(X)$$

Crucial Observation:
$$P_t(X) \neq P_0(X) \not\implies \text{clustering is failing}$$

- A large benign rigid translation drastically shifts input distribution $P(X)$ while preserving cluster partition geometry and separability.
- A small localized shift can destroy cluster boundaries, collapse margins, and corrupt memberships.
- Therefore, conventional shift detection triggers:
  $$\text{shift detected} \implies \text{adapt (naive)}$$
- Instead, this project establishes:
  $$\text{shift} \implies \text{structural signature} \implies \text{predicted degradation risk} \implies \text{choose adaptation strength}$$

---

## 3. Formal Research Questions & Hypotheses

### Research Questions (RQs)

| ID | Question |
| :--- | :--- |
| **RQ1** | Which unlabeled internal soft-clustering signals predict actual clustering degradation? |
| **RQ2** | Do these structural signals provide information beyond ordinary input distribution-shift measures? |
| **RQ3** | Does this predictive relationship generalize to completely unseen datasets? |
| **RQ4** | Does it generalize to unseen shift families? |
| **RQ5** | Can predicted degradation accurately select how much FCM should adapt? |
| **RQ6** | Can risk-guided adaptation outperform static, always-adapt, and detector-triggered strategies? |
| **RQ7** | Can risk-guided adaptation significantly reduce harmful adaptations (Negative Adaptation Rate, NAR)? |
| **RQ8** | Can it recover original representations after reversible $A \to B \to A$ drift trajectories? |

### Hypotheses (Hs)

- **H1 (Structural Prediction)**:
  $$\mathbb{E}[\widehat{\Delta Q}(D_X, Z_{\text{struct}})] < \mathbb{E}[\widehat{\Delta Q}(D_X)]$$
  *Membership and prototype structural signals reduce degradation prediction error compared to raw distribution distance alone.*
- **H2 (Unseen-Dataset Generalization)**: A risk predictor trained without dataset $D_j$ remains significantly predictive on $D_j$ (evaluated under Leave-One-Dataset-Out).
- **H3 (Unseen-Shift Generalization)**: A risk predictor trained without shift family $S_j$ remains predictive when tested under $S_j$ (evaluated under Leave-One-Shift-Family-Out).
- **H4 (Adaptation Superiority)**:
  $$Q_{\mathrm{RISA}} > Q_{\mathrm{freeze}} \quad \text{and} \quad Q_{\mathrm{RISA}} > Q_{\mathrm{always\text{-}adapt}}$$
  under distribution shifts with structural impact.
- **H5 (Clean Non-Inferiority)**:
  $$Q_{\mathrm{RISA,clean}} \ge Q_{\mathrm{FCM,clean}} - \delta$$
  *Under zero shift, RISA-FCM does not destabilize clean clustering performance.*
- **H6 (Reduced Harmful Adaptation)**:
  $$\mathrm{NAR}_{\mathrm{RISA}} < \mathrm{NAR}_{\mathrm{detector\text{-}triggered}}$$
  *Risk guidance suppresses catastrophic updates when shifts are benign or unrecoverable.*

---

## 4. Complete Method Formula Sheet

Every formula is assigned a unique reference ID for strict tracking across codebase and manuscript.

### 4.1 Data Preprocessing & Classical Soft Clustering

- **F01 (Source-Only Standardization)**:
  $$\tilde{x}_{ij} = \frac{x_{ij} - \mu_j^{(src)}}{\sigma_j^{(src)} + \epsilon}$$
  *Rule: Parameters $\mu^{(src)}, \sigma^{(src)}$ are calculated exclusively on the source reference fold. Target data is never used for scaler computation.*
- **F02 (Classical FCM Objective)**:
  $$J_m(U, V) = \sum_{i=1}^{n} \sum_{k=1}^{K} u_{ik}^{m} \|x_i - v_k\|_2^2 \quad \text{s.t.} \quad u_{ik} \in [0, 1], \; \sum_{k=1}^K u_{ik} = 1$$
- **F03 (FCM Centroid Update)**:
  $$v_k = \frac{\sum_{i=1}^n u_{ik}^{m} x_i}{\sum_{i=1}^n u_{ik}^{m}}$$
- **F04 (FCM Membership Update)**:
  $$u_{ik} = \left[ \sum_{j=1}^{K} \left( \frac{\|x_i - v_k\|_2}{\|x_i - v_j\|_2} \right)^{\frac{2}{m-1}} \right]^{-1}$$
  *Primary parameter: $m = 2.0$. Sensitivity grid: $m \in \{1.5, 1.75, 2.0, 2.25, 2.5\}$.*

---

### 4.2 Proximal FCM Adaptation Mechanism

- **F05 (Proximal FCM Objective)**:
  $$J_t(U, V; \lambda_t) = \sum_{i=1}^{n_t} \sum_{k=1}^K u_{ik}^m \|x_{i,t} - v_{k,t}\|_2^2 + \lambda_t \sum_{k=1}^K \|v_{k,t} - \tilde{v}_{k,t-1}\|_2^2$$
  *$\lambda_t \to \infty$: frozen (maximum stability). $\lambda_t \to 0$: standard refit (maximum plasticity).*
- **F06 (Proximal Centroid Update)**:
  Let $S_{k,t} = \sum_{i=1}^{n_t} u_{ik,t}^m$.
  $$v_{k,t} = \frac{\sum_{i=1}^{n_t} u_{ik,t}^m x_{i,t} + \lambda_t \tilde{v}_{k,t-1}}{S_{k,t} + \lambda_t}$$
- **F07 (Stability–Plasticity Analytical Relation)**:
  Let $\bar{v}_{k,t} = \frac{\sum_i u_{ik,t}^m x_{i,t}}{S_{k,t}}$ be the unconstrained target centroid.
  $$v_{k,t} - v_{k,t-1} = \frac{S_{k,t}}{S_{k,t} + \lambda_t} (\bar{v}_{k,t} - v_{k,t-1})$$
  $$\|v_{k,t} - v_{k,t-1}\|_2 = \frac{S_{k,t}}{S_{k,t} + \lambda_t} \|\bar{v}_{k,t} - v_{k,t-1}\|_2$$
  *Proof that prototype movement is strictly monotonically decreasing in $\lambda_t$.*

---

### 4.3 Permutation Invariance & Cluster Alignment

- **F08 (Alignment Cost Matrix)**:
  $$C_{kj} = \eta \frac{\|v_{k,t-1} - v_{j,t}\|_2}{s_{k,t-1} + \epsilon} + (1 - \eta)(1 - O_{kj})$$
  where normalized cluster radius is $s_{k,t-1}$, $\eta = 0.5$, and membership overlap on reference probes is:
  $$O_{kj} = \frac{u_k^{old} \cdot u_j^{new}}{\|u_k^{old}\|_2 \|u_j^{new}\|_2 + \epsilon}$$
- **F09 (Optimal Hungarian Assignment)**:
  $$P_t^\star = \arg\min_{P \in \Pi_K} \langle C, P \rangle = \arg\min_{P \in \Pi_K} \sum_{k,j} C_{kj} P_{kj}$$
  *Solved via `scipy.optimize.linear_sum_assignment`.*

---

### 4.4 Membership Uncertainty & Dual Probe Architecture

- **F10 (Normalized Membership Entropy)**:
  $$H_i = -\frac{1}{\log K} \sum_{k=1}^K u_{ik} \log(u_{ik} + \epsilon), \quad H_i \in [0, 1]$$
- **Dual Probe Banks**:
  - **Historical Reference Bank ($A^R \subset X_0$)**: Fixed samples frozen from source distribution. For each cluster $k$, stratified selection of core points ($H_i \approx 0$) and boundary points ($H_i \approx 1$). Never updated over time. Detects *historical forgetting*.
  - **Current Deployment Bank ($A_t^C \subset X_t$)**: Sampled from current deployment stream. Stratified across current cluster assignments. Detects *current-distribution model disagreement*.
- **F11 (Jensen–Shannon Divergence)**:
  $$JS(p, q) = \frac{1}{2} KL\left(p \,\Big\|\, \frac{p+q}{2}\right) + \frac{1}{2} KL\left(q \,\Big\|\, \frac{p+q}{2}\right)$$
  $$KL(p \| q) = \sum_{k=1}^K p_k \log \frac{p_k + \epsilon}{q_k + \epsilon}$$

---

### 4.5 Structural Drift Signals & Signatures

- **F12 (Historical Membership Drift)**:
  $$D_U^R(t) = \frac{1}{|A^R|} \sum_{x \in A^R} JS\left(U_{t-1}(x), P_t^\top U_t(x)\right)$$
- **F13 (Current Disagreement Drift)**:
  $$D_U^C(t) = \frac{1}{|A_t^C|} \sum_{x \in A_t^C} JS\left(U_{old}(x), P_t^\top U_{candidate}(x)\right)$$
- **F14 (Prototype Drift)**:
  $$D_V(t) = \frac{1}{K} \sum_{k=1}^K \frac{\|v_{k,t} - v_{k,t-1}\|_2}{s_{k,t-1} + \epsilon}$$
- **F15 (Entropy Shift)**:
  $$\bar{H}_t = \frac{1}{n_t} \sum_{i=1}^{n_t} H_{i,t}, \quad D_H(t) = |\bar{H}_t - \bar{H}_{t-1}|$$
- **F16 (Soft Cluster Mass Drift)**:
  $$\pi_{k,t} = \frac{1}{n_t} \sum_{i=1}^{n_t} u_{ik,t}, \quad D_M(t) = JS(\pi_{t-1}, \pi_t)$$
- **F17 (Maximum Mean Discrepancy - MMD Baseline)**:
  $$\operatorname{MMD}^2(X_0, X_t) = \frac{1}{n_0^2}\sum_{i,j} k(x_i, x_j) + \frac{1}{n_t^2}\sum_{i,j} k(y_i, y_j) - \frac{2}{n_0 n_t}\sum_{i,j} k(x_i, y_j)$$
  with RBF kernel $k(x, y) = \exp(-\frac{\|x-y\|^2}{2\sigma^2})$ (median heuristic bandwidth).
- **F18 (Full Structural Signature Vector)**:
  $$Z_t = \big[ D_U^R, \; D_U^C, \; D_V, \; D_H, \; D_M, \; D_X \big]^\top$$

---

### 4.6 Classical Fuzzy Validity Controls

Alternative benchmark predictors to verify structural signals add information beyond established indices:

- **F19 (Fuzzy Partition Coefficient - FPC)**:
  $$\operatorname{FPC} = \frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K u_{ik}^2$$
- **F20 (Partition Entropy - PE)**:
  $$\operatorname{PE} = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K u_{ik} \log(u_{ik} + \epsilon)$$
- **F21 (Xie–Beni Index - XB)**:
  $$\operatorname{XB} = \frac{\sum_{i=1}^n \sum_{k=1}^K u_{ik}^m \|x_i - v_k\|_2^2}{n \cdot \min_{j \neq k} \|v_j - v_k\|_2^2}$$
- **Alternative Validity Feature Vector**:
  $$Z_{\text{validity}} = [\operatorname{FPC}, \operatorname{PE}, \operatorname{XB}, \text{Silhouette}]^\top$$

---

### 4.7 Degradation Target & Risk Estimation

- **F22 (Ground-Truth Performance Degradation)**:
  $$\Delta Q_t = Q_0 - Q_t = \operatorname{ARI}_0 - \operatorname{ARI}_t$$
  *(Used strictly during benchmark meta-training/evaluation; never enters deployed RISA-FCM).*
- **F23 (Degradation Risk Estimator)**:
  $$\widehat{\Delta Q}_t = g_\phi(Z_t)$$
  Primary model: `HistGradientBoostingRegressor`.  
  Controls: `ElasticNet`, `RandomForestRegressor`, `LinearRegression`.
- **F24 (Risk Uncertainty & Conservative Risk Score)**:
  Grouped bootstrap ensemble over source datasets $b = 1, \dots, B$:
  $$\hat{\mu}_t = \frac{1}{B} \sum_{b=1}^B g_{\phi_b}(Z_t), \quad \hat{\sigma}_t = \sqrt{\frac{1}{B-1}\sum_{b=1}^B (g_{\phi_b}(Z_t) - \hat{\mu}_t)^2}$$
  $$R_t = \hat{\mu}_t + \kappa \hat{\sigma}_t \quad (\text{Primary: } \kappa = 1.0; \; \text{Sensitivity: } \kappa \in \{0, 0.5, 1.0, 2.0\})$$

---

### 4.8 Risk-Guided Adaptation Selector & Theoretical Guarantee

- **Candidate Grid $\Lambda$**:
  $$\Lambda = \{\text{freeze } (\lambda=\infty), \; \text{very\_stable } (10.0), \; \text{stable } (2.0), \; \text{balanced } (0.5), \; \text{plastic } (0.1), \; \text{full\_refit } (0.0)\}$$
- **F25 (Adaptation Cost Penalty)**:
  $$C_t(\lambda) = \frac{\|V_t^{(\lambda)} - V_{t-1}\|_F^2}{\|V_{t-1}\|_F^2 + \epsilon}$$
- **F26 (RISA-FCM Selection Rule)**:
  $$\lambda_t^\star = \arg\min_{\lambda \in \Lambda} \left[ R_t^{(\lambda)} + \gamma C_t(\lambda) \right]$$
  *(Constraint: `selector.py` must never import or observe external label metrics).*
- **Theoretical Bound (Selection-Regret Theorem)**:
  Suppose $|\hat{R}(\lambda) - R(\lambda)| \le \epsilon$ for all candidates $\lambda \in \Lambda$.  
  Let oracle $\lambda^\star = \arg\min_\lambda R(\lambda)$ and predicted $\hat{\lambda} = \arg\min_\lambda \hat{R}(\lambda)$. Then:
  $$R(\hat{\lambda}) - R(\lambda^\star) \le 2\epsilon$$
  *Directly links risk-estimation uniform error bound to downstream adaptation regret.*

---

### 4.9 Synthetic Soft Ground Truth

- **F27 (True Posterior Gaussian Mixture Membership)**:
  $$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k) \implies \tau_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
- **F28 (Soft Membership Error Metric)**:
  $$E_U = \frac{1}{n} \sum_{i=1}^n JS(\tau_i, P^\star \top u_i)$$
  *Enables evaluating true soft boundary fidelity where discrete labels cannot.*

---

### 4.10 Evaluation & Operational Metrics

- **F29 (Adjusted Rand Index - ARI)**:
  Primary hard external clustering metric (chance-corrected pair agreement).
- **F30 (Adjusted Mutual Information - AMI)**:
  Secondary external metric for non-uniform cluster distributions.
- **F31 (Reversible Shift Recovery Metrics - $A \to B \to A$)**:
  $$\operatorname{RecoveryLoss}_Q = Q_{\text{initial}} - Q_{\text{return}}$$
  $$\operatorname{RecoveryLoss}_U = \frac{1}{|A^R|} \sum_{x \in A^R} JS(U_{\text{initial}}(x), P^\star \top U_{\text{return}}(x))$$
- **F32 (Oracle Adaptation Regret)**:
  $$\lambda_t^{\text{oracle}} = \arg\max_{\lambda \in \Lambda} Q_t^{(\lambda)}, \quad \operatorname{Regret}_t = Q_t^{\text{oracle}} - Q_t^{\mathrm{RISA}}$$
- **F33 (Negative Adaptation Rate - NAR)**:
  Headline operational risk metric:
  $$\mathrm{NAR} = \frac{\#\{t : Q_t^{\text{adapt}} < Q_t^{\text{freeze}}\}}{\#\{t : \text{adaptation occurred}\}}$$
- **F34 (Area Under Degradation Curve - AUDC)**:
  $$\operatorname{AUDC} = \int_0^{s_{\max}} [Q(0) - Q(s)] \, ds \quad (\text{Trapezoidal numerical integration})$$

---

### 4.11 Statistical Omnibus & Post-Hoc Tests

- **Friedman Omnibus Test**:
  For $N$ datasets and $K$ algorithms:
  $$\chi_F^2 = \frac{12 N}{K(K+1)} \sum_{j=1}^K \bar{R}_j^2 - 3N(K+1)$$
- **Post-Hoc Pairwise**:
  Two-sided Wilcoxon signed-rank test across dataset-level paired performance with Holm–Bonferroni family-wise error rate control.

---

## 5. Benchmark Dataset Protocol (52 Units)

### Tier A: Core Real-World Clustering Datasets (30 Datasets)

Standard benchmark covering varied dimensions, sample sizes, and cluster geometry:

1. **Iris** (Fisher, 1936): 4D, 3 classes, baseline sanity check.
2. **Wine**: 13D, 3 classes, chemical continuous features.
3. **Seeds**: 7D, 3 classes, geometric grain measurements.
4. **Glass Identification**: 9D, 6 classes, high overlap.
5. **Ecoli**: 7D, 8 classes, severe class imbalance.
6. **Vehicle Silhouettes**: 18D, 4 classes, overlapping geometric profiles.
7. **Breast Cancer Wisconsin (Diagnostic)**: 30D, 2 classes, clean medical clustering.
8. **Ionosphere**: 33D, 2 classes, radar signal structure.
9. **Sonar**: 60D, 2 classes, small $n$, high $d$ stress.
10. **Banknote Authentication**: 4D, 2 classes, continuous wavelet features.
11. **Pima Indians Diabetes**: 8D, 2 classes, medical boundary ambiguity.
12. **Heart Disease (Cleveland)**: 13D, 2 classes, mixed continuous/discrete.
13. **Haberman Survival**: 3D, 2 classes, low-dimensional high-overlap.
14. **Dermatology**: 33D, 6 classes, clinical evaluation with missingness.
15. **Balance Scale**: 4D, 3 classes, non-linear balance dynamics.
16. **Image Segmentation**: 19D, 7 classes, visual feature descriptors.
17. **Satimage (Statlog)**: 36D, 6 classes, multi-spectral satellite pixels.
18. **Pendigits**: 16D, 10 classes, pen-based digit trajectory coordinates.
19. **Optdigits**: 64D, 10 classes, normalized pixel matrices.
20. **Letter Recognition**: 16D, 26 classes, 20K observations large scale.
21. **Waveform (Version 1)**: 21D, 3 classes, noisy synthetic continuous.
22. **Spambase**: 57D, 2 classes, word/char frequencies.
23. **Mice Protein Expression**: 77D, 8 classes, biological expression with missingness.
24. **Human Activity Recognition (HAR)**: 561D, 6 classes, high-dimensional sensor timeseries features.
25. **Isolet**: 617D, 26 classes, spoken letter acoustic structure.
26. **Madelon**: 500D, 2 classes, redundant and noisy synthetic features.
27. **Digits (scikit-learn)**: 64D, 10 classes, standard optical digits.
28. **USPS**: 256D, 10 classes, handwritten digit pixel matrices.
29. **COIL-20**: 1024D, 20 classes, object reflectance manifolds.
30. **Olivetti Faces**: 4096D, 40 classes, extreme high-dimensional low-$n$ challenge.

### Tier B: Prior Robustness Line Stress Datasets (5 Datasets)

High-volume real-world tabular data connecting to prior publication lineage:
31. **Adult Income**: Census income shift benchmark.
32. **Credit Card Default**: Financial risk tabular structure.
33. **Electricity (Australian Market)**: Real time-series tabular pricing shift.
34. **APS Failure at Scania Trucks**: Industrial sensor dataset with severe class imbalance and missingness.
35. **Santander Customer Satisfaction**: High-dimensional sparse industrial financial dataset.

### Tier C: Natural-Distribution-Shift Datasets (5 TableShift Benchmarks)

Real natural distribution shifts with officially partitioned domains:
36. **Voting / ANES**: Geographic-region shift.
37. **HELOC**: Third-party risk-level shift.
38. **Childhood Lead**: County poverty-level shift.
39. **Hospital Readmission**: Admission-source clinical shift.
40. **College Scorecard**: Public vs. private institution-type shift.

### Tier D: Synthetic Benchmark Families (12 Generators with Soft Ground Truth $\tau$)

 1. **S01**: Balanced, well-separated Gaussian mixture ($K=3$).
 2. **S02**: Moderate overlap Gaussian mixture.
 3. **S03**: Severe overlap Gaussian mixture.
 4. **S04**: Unequal cluster prior weights ($\pi = [0.7, 0.2, 0.1]$).
 5. **S05**: Heteroscedastic unequal covariance structures.
 6. **S06**: Highly anisotropic non-spherical clusters.
 7. **S07**: High-dimensional Gaussian mixture ($d=100$).
 8. **S08**: High-dimensional Gaussian mixture with 80% uninformative noise features.
 9. **S09**: Heavy-tailed mixture with Student-$t$ distribution ($\nu=3$).
10. **S10**: Localized single-cluster movement (benign to other clusters).
11. **S11**: Gradual continuous trajectory ($A \to B$).
12. **S12**: Reversible trajectory ($A \to B \to A$).

---

## 6. Controlled Shift Generator Suite

Eight distinct controlled shift families across three calibrated severity levels:

| Shift Family | Mechanism | Low (Severity 1) | Medium (Severity 2) | High (Severity 3) |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian Corruption** | Additive $\mathcal{N}(0, \sigma^2 \mathbf{I})$ | $\sigma = 0.05 \cdot \sigma^{(src)}$ | $\sigma = 0.15 \cdot \sigma^{(src)}$ | $\sigma = 0.30 \cdot \sigma^{(src)}$ |
| **Location / Mean Shift** | Translation along principal drift axis | $\Delta\mu = 0.25 \cdot \sigma^{(src)}$ | $\Delta\mu = 0.50 \cdot \sigma^{(src)}$ | $\Delta\mu = 1.00 \cdot \sigma^{(src)}$ |
| **Covariance Scaling** | Scale variance around cluster centers | $\times 1.25$ | $\times 1.50$ | $\times 2.00$ |
| **Missingness (MCAR)** | Missing completely at random entry mask | $5\%$ missing | $15\%$ missing | $30\%$ missing |
| **Outlier Contamination** | Uniform hypersphere background noise | $5\%$ outliers | $10\%$ outliers | $20\%$ outliers |
| **Cluster Proportion Shift** | Resample class priors Dirichlet $(\alpha)$ | Mild skew ($\alpha=5$) | Moderate skew ($\alpha=2$) | Severe skew ($\alpha=0.5$) |
| **Cluster Overlap Increase** | Contract cluster centers toward global mean | $10\%$ center contraction | $25\%$ center contraction | $50\%$ center contraction |
| **Feature-Subset Shift** | Shift applied to random feature subset | $10\%$ of features | $25\%$ of features | $50\%$ of features |

Total conditions per dataset: 1 clean source reference + 8 shift families $\times$ 3 severities = **25 conditions**.

---

## 7. Experimental Scale & Computational Budget

- **Core Real-World Benchmark**: 30 datasets $\times$ 8 methods $\times$ 5 fold rotations $\times$ 25 conditions = **30,000 units**.
- **Synthetic Suite**: 12 generators $\times$ 8 methods $\times$ 5 fold rotations $\times$ 25 conditions = **12,000 units**.
- **Stress Datasets (Tier B)**: 5 datasets $\times$ 4 methods $\times$ 5 fold rotations $\times$ 10 conditions = **1,000 units**.
- **Natural Shift Benchmarks (Tier C)**: 5 datasets $\times$ 6 methods $\times$ 5 fold rotations $\times$ 2 domains = **300 units**.
- **Final Confirmation Experiment (15 Seeds)**: 12 representative datasets $\times$ 6 methods $\times$ 10 conditions $\times$ 15 seeds = **10,800 units**.
- **Ablation Studies & Sensitivity Sweeps**: $\approx$ **5,000 – 10,000 units**.
- **Total Experiment Units**: $\approx$ **60,000 – 65,000 units**.

### Evaluation Protocol

- **Five-Fold Source–Target Rotation**: Randomly partition dataset into 5 equal splits: $D = F_1 \cup \dots \cup F_5$ without using labels. In rotation $r$, $D_{\text{source}} = D \setminus F_r$ and $D_{\text{target}} = F_r$. Shifts are applied exclusively to $D_{\text{target}}$.
- **Internal Optimization Seeds**: $n\_init = 5$ random restarts (seeds 1 to 5) for every clustering fit, retaining the minimal objective solution.
- **Confirmation Seeds**: Seeds 1 to 15 executed strictly after freezing algorithm code and hyperparameter configurations.

---

## 8. Baselines & Benchmark Comparison

### Static & Alternative Soft Clustering Baselines

1. **Standard K-Means** (Hard clustering baseline).
2. **Standard FCM** (Classical soft baseline, $m=2.0$).
3. **Gaussian Mixture Models (GMM)** (Probabilistic soft baseline with full covariance).
4. **Possibilistic C-Means (PCM)** (Relaxed constraint clustering).
5. **Possibilistic Fuzzy C-Means (PFCM)** (Hybrid possibilistic-fuzzy baseline).
6. **Gustafson–Kessel (GK)** (Adaptive distance norm soft clustering).
7. **Robust FCM** (Noise/outlier resistant soft clustering).
8. **Feature-Weighted FCM (FWSCA)** (Representative algorithm from feature-weighting literature).

### Adaptation Strategies for Comparison

1. **Static FCM (Freeze)**: Retain $(U_0, V_0)$, evaluate directly on $X_t$.
2. **Always-Refit FCM**: Discard previous state, fit fresh FCM on $X_t$.
3. **Sliding-Window FCM**: Online update using windowed batch data.
4. **Fixed-Memory Proximal FCM**: Proximal FCM with fixed static $\lambda = 1.0$.
5. **Drift-Detector-Triggered FCM (D3 / MMD-Triggered)**: Freeze until MMD test detects shift ($p < 0.05$), then full refit.
6. **RISA-FCM (Ours)**: Risk-informed candidate selection via $\lambda^\star = \arg\min_{\lambda \in \Lambda} [R_t^{(\lambda)} + \gamma C_t(\lambda)]$.
7. **Oracle Adaptation (Upper Bound)**: Ground-truth target labels select $\arg\max_{\lambda \in \Lambda} \operatorname{ARI}_t(\lambda)$.

---

## 9. Critical Risk-Model Ablation Comparison

Evaluated under strict Leave-One-Dataset-Out (LODO) cross-validation:

| Predictor Configuration | Input Feature Set | Hypothesis Tested | Target Outcome |
| :--- | :--- | :--- | :--- |
| **P0 (Raw Shift Baseline)** | $D_X$ (MMD) | Can raw distribution distance predict failure? | Baseline error |
| **P1 (Classical Validity)** | $Z_{\text{validity}} = [\operatorname{FPC}, \operatorname{PE}, \operatorname{XB}, \text{Silhouette}]$ | Do existing validity indices suffice? | Control comparison |
| **P2 (Structural Signals Only)** | $Z_{\text{struct}} = [D_U^R, D_U^C, D_V, D_H, D_M]$ | Are internal soft signals predictive alone? | $\text{Error}(P2) < \text{Error}(P0)$ |
| **P3 (Raw Shift + Validity)** | $[D_X, Z_{\text{validity}}]$ | Combined external & standard cluster metrics | Intermediate benchmark |
| **P4 (Raw Shift + Structural)** | $[D_X, Z_{\text{struct}}]$ | Primary paper thesis: shift + soft signature | $\text{Error}(P4) < \text{Error}(P0)$ & $\text{Error}(P4) < \text{Error}(P3)$ |
| **P5 (Full Combined Signature)** | $[D_X, Z_{\text{struct}}, Z_{\text{validity}}]$ | Exhaustive information signature | Saturated performance |

---

## 10. Research Integrity & Leakage Policy

Six mandatory rules enforced via automated unit tests:

1. **Rule 1 (Label Isolation)**: Ground-truth labels $y$ may only be imported in:
   - `shifts/proportion.py` (controlled experimental intervention only)
   - `shifts/overlap.py` (controlled experimental intervention only)
   - `risk/targets.py` (meta-training target generation only)
   - `metrics/external.py` (evaluation benchmark reporting only)  
   *Enforced: `tests/test_no_label_leakage.py` scans AST of all other modules.*
2. **Rule 2 (Source-Only Preprocessing)**: Preprocessors (scalers, imputers) must only call `.fit(X_source)` and never `.fit(X_source \cup X_target)` or `.fit(X_target)`.
3. **Rule 3 (Target Independence)**: Target split data must never determine $K$, $m$, feature scalers, imputer parameters, probe banks, risk hyperparameters, or candidate $\lambda$ grids.
4. **Rule 4 (Oracle $K$ Protocol)**: Controlled benchmark operates under an explicit Oracle-$K$ protocol ($K = \text{true reference classes}$) to isolate adaptation capability. Sensitivity is separately tested on $K \in \{K_{\text{true}}-1, K_{\text{true}}, K_{\text{true}}+1\}$. No Auto-$K$ heuristics in RISA-FCM.
5. **Rule 5 (Natural-Shift Variable Masking)**: Domain split indicators in natural shift datasets (e.g., TableShift region/institution IDs) must never be passed to clustering algorithms as input features.
6. **Rule 6 (Strict Grouped Validation)**: Grouped cross-validation must group by source dataset ID. Shift variants derived from dataset $D_j$ must never appear in both meta-train and meta-test partitions.

---

## 11. Planned Figures & Tables

### Figures

- **Figure 1**: Conceptual system architecture: $X_t \to \text{Structural Signature } Z_t \to \text{Risk Model } R_t \to \text{Adaptation Selection } \lambda_t^\star$.
- **Figure 2**: Shift magnitude ($D_X$) versus actual performance degradation ($\Delta\operatorname{ARI}$), demonstrating non-monotonicity and decoupling.
- **Figure 3**: Structural signals ($D_U^R, D_U^C, D_V, D_H, D_M$) versus true degradation across datasets.
- **Figure 4**: Case study: Equal MMD distance producing radically different structural failure vs. benign translation.
- **Figure 5**: Risk prediction scatter and calibration curves on completely unseen datasets (Leave-One-Dataset-Out).
- **Figure 6**: Degradation profiles across 8 shift families and 3 severity levels.
- **Figure 7**: Comparative trajectories: Freeze vs. Always-Refit vs. Detector-Triggered vs. RISA-FCM.
- **Figure 8**: Pareto adaptation-budget curve (Performance vs. Prototype Movement Cost $C_t$).
- **Figure 9**: Negative Adaptation Rate ($\mathrm{NAR}$) distribution across methods.
- **Figure 10**: Reversible shift recovery dynamics ($A \to B \to A$) tracking membership recovery loss.
- **Figure 11**: Selected candidate adaptation strength $\lambda^\star$ vs. Oracle optimal $\lambda^{\text{oracle}}$.
- **Figure 12**: Systematic ablation of individual structural signals within $Z_t$.

### Tables

- **Table 1**: Benchmark dataset taxonomy (52 units: dimensions, samples, cluster count, domain).
- **Table 2**: Literature & novelty collision matrix distinguishing RISA-FCM from dynamic FCM, TTA, and feature weighting.
- **Table 3**: Risk prediction model ablation results (P0 to P5 comparison: MAE, RMSE, Spearman $\rho$).
- **Table 4**: Primary clustering benchmark performance ($\operatorname{ARI}$ and $\operatorname{AMI}$) across 30 Tier-A datasets.
- **Table 5**: Area Under Degradation Curve ($\operatorname{AUDC}$) across shift families.
- **Table 6**: Operational risk metrics: Negative Adaptation Rate ($\mathrm{NAR}$) and Adaptation Cost ($C_t$).
- **Table 7**: Oracle adaptation regret comparison across adaptation baselines.
- **Table 8**: Reversible trajectory recovery losses ($\operatorname{RecoveryLoss}_Q$ and $\operatorname{RecoveryLoss}_U$).
- **Table 9**: Algorithmic sensitivity analysis: Fuzzifier $m \in [1.5, 2.5]$, uncertainty trade-off $\kappa$, cost weight $\gamma$.
- **Table 10**: Non-parametric statistical tests: Friedman omnibus $\chi_F^2$ rank test and post-hoc Holm-adjusted Wilcoxon signed-rank $p$-values.

---

## 12. 16-Phase Execution Roadmap

- **Phase 0 (Specification & Protocols)**: Freeze research questions, hypotheses, novelty collision matrix, leakage policy.
- **Phase 1 (Repository Foundation)**: Configs, schemas, logging, reproducible seeds, cryptographic run hashing.
- **Phase 2 (Dataset Pipeline)**: Registry, loader, preprocessors, 5-fold rotations, datasets manifest with SHA256 checksums.
- **Phase 3 (Baseline Implementation)**: FCM, K-Means, GMM, PCM, PFCM, GK, FWSCA baseline validations.
- **Phase 4 (Controlled Shift Engine)**: 8 parameterized shift generators + 3 severity tiers + trajectory generators.
- **Phase 5 (Alignment & Probes)**: Hungarian cluster alignment, reference core/boundary probe banks, deployment probe sampler.
- **Phase 6 (Structural Signals)**: $D_U^R, D_U^C, D_V, D_H, D_M, D_X$, and classical fuzzy validity metrics.
- **Phase 7 (Falsification Pilot)**: Pilot experiment verifying structural signals improve degradation prediction over $D_X$ on held-out datasets.
- **Phase 8 (Risk Dataset Generation)**: Build offline risk meta-dataset separating `risk_features.parquet` and `risk_targets.parquet`.
- **Phase 9 (Risk Model Optimization)**: Grouped cross-validation comparing Linear, ElasticNet, RF, and HistGradientBoosting regressors.
- **Phase 10 (RISA-FCM Implementation)**: Proximal FCM objective, candidate evaluation pipeline, and cost-regularized risk selector.
- **Phase 11 (Adaptation Pilot)**: Head-to-head comparison of adaptation strategies, regret analysis against oracle.
- **Phase 12 (Full Benchmark Execution)**: Lock `final_locked.yaml` SHA256; execute 60,000+ experimental units across compute cluster.
- **Phase 13 (Statistical Testing)**: Friedman omnibus test, pairwise Wilcoxon with Holm adjustment, bootstrap effect sizes.
- **Phase 14 (Drift Trajectory Experiments)**: Sequential $A \to B$ gradual drift and $A \to B \to A$ reversible drift recovery evaluations.
- **Phase 15 (Final Integrity Audit)**: Verification script `12_verify_research_integrity.py` validates complete artifact lineage and zero leakage.

---

## 13. Locked Decisions & Invariants Summary

| Item | Frozen Decision | Invariant Rationale |
| :--- | :--- | :--- |
| **Paper Title** | *When Clusters Drift: Label-Free Failure Prediction and Risk-Guided Adaptation for Soft Clustering* | Precise, captures gap, method, and setting; passes collision checks |
| **Algorithm Name** | `RISA-FCM` | Risk-Informed Selective Adaptation Fuzzy C-Means |
| **Repository Name** | `when-clusters-drift` | Standardized GitHub repo identifier |
| **Python Package** | `clusterdrift` | Clean modular import namespace |
| **Core Novelty** | Unlabeled structural prediction of clustering failure | Decouples distribution shift from clustering degradation |
| **Adaptation Core** | Proximal FCM objective with risk-selected $\lambda^\star$ | Analytically grounded stability-plasticity control without ad-hoc heuristics |
| **Fuzzifier Default** | $m = 2.0$ | Classical standard, evaluated with sensitivity $m \in \{1.5, 1.75, 2.0, 2.25, 2.5\}$ |
| **Primary Metric** | Adjusted Rand Index ($\operatorname{ARI}$) | Chance-corrected standard external metric |
| **Secondary Metric** | Adjusted Mutual Information ($\operatorname{AMI}$) | Information-theoretic complement |
| **Soft Ground Truth** | Gaussian mixture posterior $\tau$ and Membership JSD | Ground-truth fuzziness evaluation |
| **Operational Metric** | Negative Adaptation Rate ($\mathrm{NAR}$) & Oracle Regret | Directly measures frequency and severity of harmful adaptations |
| **Cross-Validation** | 5-Fold Source-Target Rotations + LODO Grouping | Eliminates dataset identity leakage in risk models |
| **Statistical Test** | Friedman $\to$ Post-hoc Wilcoxon with Holm correction | Adheres to high statistical standards in clustering literature |
| **Strict Bounds** | No Auto-$K$, No online feature-weighting, No neural components | Keeps method theoretically tractable, modular, and defensible |
