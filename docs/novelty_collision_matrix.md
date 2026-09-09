# Novelty & Collision Matrix

| Dimension | Existing Literature | `when-clusters-drift` Approach | Scientific Outcome |
|---|---|---|---|
| **Covariate Shift Detection** | Tests whether $P(X)$ changed via two-sample testing (MMD, energy distance, classifier two-sample test). | Evaluates whether $D_X$ (MMD) directly predicts degradation $\Delta\text{ARI}$ of clustering models. | $D_X$ alone ($P0$) is a moderate predictor (MAE = 0.04670), outperforming complex soft-clustering features. |
| **Internal Cluster Validity** | Static indices (Silhouette, Davies-Bouldin, Xie-Beni, Partition Coefficient) evaluate single clustering quality. | Compares static validity deltas against differential drift signals across shifted environments. | Classical static validity ($P3$) proved worst (LODO MAE = 0.06393). |
| **Soft Structural Signals** | FCM membership entropy and prototype displacement used primarily for convergence monitoring or cluster selection. | Hypothesized that membership divergence ($D_U$) and prototype shift ($D_V$) predict degradation under shift. | **Falsified**: Structural signals added noise ($P4$ MAE = 0.05089), degrading performance relative to $P0$. |
| **Test-Time Adaptation (RISA)** | Adaptive clustering algorithms (e.g. streaming K-Means, adaptive FCM) update centroids without risk bounds. | Planned a risk-informed adaptation architecture (RISA) conditioned on degradation predictions. | **Halted**: Canceled before Phase 8 because predictor failed primary falsification gate. |
