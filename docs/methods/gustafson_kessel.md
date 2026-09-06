# Gustafson-Kessel Fuzzy Clustering (GK) Method Reference

## 1. Method Name & Citation
- **Method Name**: Gustafson-Kessel Algorithm (GK)
- **Primary Citations**:
  - Gustafson, D. E., & Kessel, W. C. (1979). "Fuzzy clustering with a fuzzy covariance matrix." *1979 IEEE Conference on Decision and Control including the 16th Symposium on Adaptive Processes*, pp. 761–766.
  - Babuska, R. (1998). *Fuzzy Modeling for Control*. Springer Science & Business Media.

---

## 2. Motivation & Background
Standard Fuzzy C-Means uses the Euclidean distance metric, which implicitly assumes hyperspherical clusters of equal volume. In real-world data, clusters often have unequal volumes, differing orientations, and elongated elliptical geometries.

The Gustafson-Kessel algorithm extends FCM by associating each cluster $k$ with an adaptive Mahalanobis distance metric induced by a positive-definite matrix $A_k$. By constraining the volume of the cluster shapes ($\det(A_k) = \rho_k = 1$), GK enables each cluster to adaptively adjust its ellipsoidal shape to the local covariance of data points.

---

## 3. Mathematical Formulation

### 3.1 Objective Function
$$J_{\text{GK}}(U, V, \{A_k\}; X) = \sum_{i=1}^n \sum_{k=1}^K u_{ik}^m d_{ik}^2$$

where the Mahalanobis squared distance is:
$$d_{ik}^2 = (x_i - v_k)^T A_k (x_i - v_k)$$

subject to:
$$u_{ik} \ge 0 \quad \forall i, k, \qquad \sum_{k=1}^K u_{ik} = 1 \quad \forall i$$
$$\det(A_k) = \rho_k = 1 \quad \forall k$$

where $m > 1$ is the fuzzy weighting exponent (typically $m = 2.0$).

---

## 4. Update Equations

### 4.1 Fuzzy Covariance Matrix ($\Sigma_k$)
The fuzzy covariance matrix of cluster $k$ is computed as:
$$\Sigma_k = \frac{\sum_{i=1}^n u_{ik}^m (x_i - v_k)(x_i - v_k)^T}{\sum_{i=1}^n u_{ik}^m}$$

### 4.2 Norm-Inducing Metric Matrix ($A_k$)
From the constraint $\det(A_k) = 1$, the optimal metric matrix is:
$$A_k = [\det(\Sigma_k)]^{1/d} \Sigma_k^{-1}$$
where $d$ is the dimensionality of feature space $\mathbb{R}^d$.

### 4.3 Fuzzy Membership Update ($u_{ik}$)
$$u_{ik} = \left[ \sum_{j=1}^K \left( \frac{d_{ik}}{d_{ij}} \right)^{2 / (m - 1)} \right]^{-1}$$

### 4.4 Prototype Update ($v_k$)
$$v_k = \frac{\sum_{i=1}^n u_{ik}^m x_i}{\sum_{i=1}^n u_{ik}^m}$$

---

## 5. Numerical Safeguards & Covariance Regularization

In high dimensions, with small cluster sizes or linearly dependent features, the sample fuzzy covariance matrix $\Sigma_k$ can become singular ($\det \Sigma_k = 0$) or ill-conditioned (condition number $\kappa(\Sigma_k) \gg 10^8$).

### 5.1 Deterministic Regularization (Babuska 1998)
To prevent singular metrics while maintaining cluster geometry:
1. **Condition Number Check**:
   Compute condition number $\kappa(\Sigma_k) = \lambda_{\max} / \lambda_{\min}$. If $\kappa(\Sigma_k) > 10^8$ or $\lambda_{\min} < 10^{-6}$:
2. **Eigenvalue / Trace Regularization**:
   $$\Sigma_{k, \text{reg}} = (1 - \beta) \Sigma_k + \beta [\det(\Sigma_0)]^{1/d} I_d$$
   or ridge regularization:
   $$\Sigma_{k, \text{reg}} = \Sigma_k + \epsilon_{\text{reg}} \cdot \text{tr}(\Sigma_k) \cdot I_d$$
   where $\epsilon_{\text{reg}} = 10^{-5}$ and $I_d$ is the identity matrix.
3. **Log-Determinant Computation**:
   To avoid floating point underflow or overflow when taking $[\det(\Sigma_k)]^{1/d}$, compute:
   $$[\det(\Sigma_k)]^{1/d} = \exp\left( \frac{1}{d} \sum_{j=1}^d \ln \lambda_j \right)$$
   using eigenvalue decomposition $\Sigma_k = Q \Lambda Q^T$.
4. **Failure Tracking**:
   If positive definiteness cannot be achieved or $\Sigma_k$ remains non-invertible, set `status_ = "SINGULAR_METRIC"` or `"NUMERICAL_FAILURE"` and record the warning. Never silently fall back to Euclidean FCM without documenting the regularization event.
