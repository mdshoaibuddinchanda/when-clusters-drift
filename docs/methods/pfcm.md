# Possibilistic Fuzzy C-Means (PFCM) Method Reference

## 1. Method Name & Citation
- **Method Name**: Possibilistic Fuzzy C-Means (PFCM)
- **Primary Citation**:
  Pal, N. R., Pal, K., Keller, J. M., & Bezdek, J. C. (2005). "A possibilistic fuzzy c-means clustering algorithm." *IEEE Transactions on Fuzzy Systems*, 13(4), 517–530. DOI: 10.1109/TFUZZ.2004.840099.

---

## 2. Motivation & Background
Standard Fuzzy C-Means (FCM) partitions data by constraining the sum of memberships of each point across clusters to 1 ($\sum_{k=1}^K u_{ik} = 1$). While this prevents trivial solutions, it makes FCM sensitive to outliers: an extreme outlier must still distribute a total membership of 1 across clusters, distorting cluster prototypes.

Possibilistic C-Means (PCM; Krishnapuram & Keller, 1993) relaxes this constraint to model absolute typicality $t_{ik} \in [0, 1]$, but frequently suffers from prototype coincidence (multiple prototypes collapsing onto the same dense region).

PFCM unifies FCM and PCM by jointly optimizing fuzzy memberships $u_{ik}$ (which partition the data and prevent prototype collapse) and possibilistic typicalities $t_{ik}$ (which measure absolute typicality and mitigate the influence of outliers and noise).

---

## 3. Mathematical Formulation

### 3.1 Objective Function
Let $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$ be the dataset, $V = \{v_1, \dots, v_K\} \subset \mathbb{R}^d$ be cluster prototypes, $U = [u_{ik}]_{n \times K}$ be the fuzzy partition matrix, and $T = [t_{ik}]_{n \times K}$ be the possibilistic typicality matrix.

The PFCM objective function is:
$$J_{\text{PFCM}}(U, T, V; X) = \sum_{i=1}^n \sum_{k=1}^K (a \cdot u_{ik}^m + b \cdot t_{ik}^\eta) \|x_i - v_k\|_2^2 + \sum_{k=1}^K \gamma_k \sum_{i=1}^n (1 - t_{ik})^\eta$$

subject to:
$$u_{ik} \ge 0 \quad \forall i, k, \qquad \sum_{k=1}^K u_{ik} = 1 \quad \forall i$$
$$0 \le t_{ik} \le 1 \quad \forall i, k$$

where:
- $m > 1$ is the fuzzy exponent (fuzzifier, typically $m = 2$).
- $\eta > 1$ is the possibilistic exponent (typically $\eta = 2$).
- $a > 0$ is the weight of the fuzzy membership component (default $a = 1$).
- $b > 0$ is the weight of the possibilistic typicality component (default $b = 1$).
- $\gamma_k > 0$ is the scale / penalty parameter for cluster $k$.

---

## 4. Parameter Definitions & Scale Parameter Formulation

### 4.1 Hyperparameters
1. **$a$ and $b$**: Balance the relative influence of fuzzy membership vs. possibilistic typicality in updating cluster centers. When $a > 0, b = 0$, PFCM reduces to standard FCM.
2. **$m$ and $\eta$**: Control the fuzziness and typicality softness. In Phase 3, we lock $m = 2.0$ and $\eta = 2.0$.
3. **Scale Parameter $\gamma_k$**: Determines the distance threshold at which typicality becomes $0.5$. In Pal et al. (2005), $\gamma_k$ is computed from an initial FCM run or data geometry as:
   $$\gamma_k = K_{\text{scale}} \frac{\sum_{i=1}^n u_{ik}^m \|x_i - v_k\|_2^2}{\sum_{i=1}^n u_{ik}^m}$$
   where $K_{\text{scale}} = 1.0$ (standard default in the literature).

---

## 5. Update Equations

By setting partial derivatives of the Lagrangian of $J_{\text{PFCM}}$ with respect to $U$, $T$, and $V$ to zero:

### 5.1 Fuzzy Membership Update ($u_{ik}$)
For a point $x_i$ with $\|x_i - v_j\|_2 > 0$ for all $j$:
$$u_{ik} = \left[ \sum_{j=1}^K \left(\frac{\|x_i - v_k\|_2}{\|x_i - v_j\|_2}\right)^{2 / (m - 1)} \right]^{-1}$$

### 5.2 Possibilistic Typicality Update ($t_{ik}$)
$$t_{ik} = \left[ 1 + \left( \frac{b}{\gamma_k} \|x_i - v_k\|_2^2 \right)^{1 / (\eta - 1)} \right]^{-1}$$

**Crucial Semantics**:
- $u_{ik}$ satisfies $\sum_{k=1}^K u_{ik} = 1$ and represents relative partition membership.
- $t_{ik} \in [0, 1]$ has **no sum-to-one constraint** across clusters. It represents absolute degree of belonging to cluster $k$. It must never be normalized across clusters or interpreted as a probability distribution.

### 5.3 Prototype Update ($v_k$)
$$v_k = \frac{\sum_{i=1}^n (a \cdot u_{ik}^m + b \cdot t_{ik}^\eta) x_i}{\sum_{i=1}^n (a \cdot u_{ik}^m + b \cdot t_{ik}^\eta)}$$

---

## 6. Numerical Safeguards

1. **Coincident Prototypes ($d_{ik} = 0$)**:
   If $\|x_i - v_k\|_2 = 0$ for one or more prototypes:
   - For coincident prototypes ($k \in \mathcal{C}_i$ where $|\mathcal{C}_i| = c \ge 1$):
     $$u_{ik} = \frac{1}{c}, \qquad t_{ik} = 1.0$$
   - For non-coincident prototypes ($k \notin \mathcal{C}_i$):
     $$u_{ik} = 0.0, \qquad t_{ik} = \left[ 1 + \left( \frac{b}{\gamma_k} \|x_i - v_k\|_2^2 \right)^{1 / (\eta - 1)} \right]^{-1}$$

2. **Near-Zero Fuzzy Mass ($\sum_i u_{ik}^m < \epsilon$)**:
   To prevent division by zero, denominators are guarded by $\epsilon = 10^{-15}$. If an entire cluster mass collapses below machine precision, the status is flagged as `EMPTY_CLUSTER`.

3. **Convergence Criterion**:
   $$\|V^{(t)} - V^{(t-1)}\|_F < \text{tol}$$
   with maximum iterations capped at $\text{max\_iter}$.
