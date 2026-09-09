# Zero-Label-Leakage & Data Preprocessing Policy

## 1. Zero-Label Access Rule
In an unsupervised clustering regime, inference-time labels do not exist. To avoid optimistic bias:
- **Algorithms**: Implementations of FCM, Gustafson-Kessel, GMM, and K-Means must not accept, inspect, or reference ground truth array `y`.
- **Signal Extraction**: All metrics in `src/clusterdrift/signals/` ($D_X, D_U, D_V, D_H, D_M$, MMD) compute distances purely between unlabeled coordinate arrays $X_{\text{source}}, X_{\text{target}}$ and soft membership matrices $U_{\text{source}}, U_{\text{target}}$.
- **Quality Evaluation**: Ground truth labels are quarantined in a dedicated, isolated evaluation harness (`src/clusterdrift/falsification/evaluation.py`) that computes $\Delta\text{ARI}$ and $\Delta\text{AMI}$ in Pass B only.

---

## 2. Preprocessing & Imputation Isolation
- **Source-Only Fitting**: Scalers (e.g. `StandardScaler`) and imputers (e.g. median imputer) are strictly fit on the source training fold $X_{\text{source}}$.
- **Out-of-Sample Transform**: Candidate target sets $X_{\text{target}}$ are transformed using source statistics. Target sets are never used to compute means, variances, or medians.
- **Probe Bank Independence**: Reference probe banks are drawn exclusively from source distributions and never updated with shifted target samples.
