# Research Question: Predicting Cluster Degradation Under Distribution Shift

## Core Scientific Problem
Unsupervised clustering algorithms (such as Fuzzy C-Means, Gustafson-Kessel, Gaussian Mixture Models, and K-Means) are deployed in production systems where data distributions inevitably drift over time ($P_t(X) \neq P_0(X)$). Unlike supervised systems where ground truth labels eventually arrive to reveal performance drops, unsupervised clustering systems operate with **zero inference-time labels**. Consequently, partition degradation occurs silently and invisibly.

## Primary Research Question
> **Under distribution shift $P_t(X) \neq P_0(X)$, can label-free geometric and membership signals internal to soft clustering algorithms predict true partition degradation ($\Delta\text{ARI} = \text{ARI}_{\text{clean}} - \text{ARI}_{\text{shifted}}$) on unseen datasets and unseen shift mechanisms better than raw covariate data-drift measures alone?**

---

## Secondary Research Questions
1. **Probe Bank Independence (RQ1)**: Can a dual-probe architecture—contrasting a fixed reference probe bank against a contemporaneous current probe bank—disentangle algorithm-induced artifacts from genuine geometric manifold deformation?
2. **Signal Specificity (RQ2)**: Do distinct shift mechanisms (e.g., scale expansion vs. local boundary overlap vs. subpopulation prevalence shift) produce distinct structural signatures across membership entropy ($D_U$), prototype displacement ($D_V$), and cluster mass divergence ($D_M$)?
3. **Out-of-Distribution Transferability (RQ3)**: Can a predictive model trained on structural signals from a set of known datasets accurately predict the magnitude of clustering degradation on an entirely novel dataset (Leave-One-Dataset-Out / LODO generalization)?
4. **Superiority Over Classical Validity (RQ4)**: Do differential drift signals ($Z_{\text{struct}}$) offer predictive advantage over classical static internal cluster validity indices (Silhouette, Partition Coefficient, Xie-Beni, Partition Entropy)?

---

## Project Conclusion & Falsification
As documented in the Phase-7 falsification execution and confirmed in the master audit on branch [`hypothesis-y1-master-audit`](https://github.com/mdshoaibuddinchanda/when-clusters-drift/tree/hypothesis-y1-master-audit):
- **The answer to the primary research question is NO.**
- Internal structural signals ($P4$) underperformed raw covariate drift ($P0$, MMD) by $-8.97\%$ in LODO MAE, failed win-ratio thresholds, and lost to a simple non-learning median baseline.
- In accordance with preregistered protocol, the project was permanently terminated after Phase 7.
