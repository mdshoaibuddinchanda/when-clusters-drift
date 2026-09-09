# Hypothesis Y1 design repair for any future RISA study

This is a design review only. No Phase 8 code, data generation, or outcome experiment was started.

## Candidate-specific risk target

The Phase 7 target is the degradation of one frozen source model. It cannot select an adaptation strength because it assigns the same risk target to every candidate. A future selector must train on candidate-indexed examples

\[
  (Z_{\mathrm{base},t}, Z_t(\alpha), \alpha) \longmapsto
  R_t(\alpha)=1-Q_t(\alpha),
\]

where `Z_base` contains label-free batch/source context, `Z(alpha)` contains label-free effects of the actual candidate model, and `alpha` is the dimensionless adaptation strength below. Candidate features must be computed using only source state and the current unlabeled deployment batch. External labels enter only offline target construction/evaluation. Including `alpha` alone is insufficient: the same strength can have different effects, so each realized candidate's memberships, prototype movement, convergence status, and cost must be represented.

## Objective and valid regret bound

Use one objective for the selector and oracle:

\[
J_t(\alpha)=R_t(\alpha)+\gamma C_t(\alpha),\qquad
\widehat J_t(\alpha)=\widehat R_t(\alpha)+\gamma C_t(\alpha).
\]

Let \(\alpha^*=\arg\min J_t\) and \(\hat\alpha=\arg\min\widehat J_t\). If the finite candidate set has a uniform held-out error guarantee \(|\widehat R_t(\alpha)-R_t(\alpha)|\le\epsilon\), and the same fixed, known label-free cost appears in both objectives, then

\[
J_t(\hat\alpha)\le\widehat J_t(\hat\alpha)+\epsilon
\le\widehat J_t(\alpha^*)+\epsilon
\le J_t(\alpha^*)+2\epsilon.
\]

Thus \(J_t(\hat\alpha)-J_t(\alpha^*)\le2\epsilon\). The theorem in the draft is not valid for a risk-only oracle paired with a risk-plus-cost selector; it must be replaced by this matched-objective statement. The assumption is strong and must be established on held-out groups, not asserted from training error.

## Dimensionless adaptation strength

For the proximal FCM center update

\[
v_{k,new}=\frac{S_k\bar x_k+\lambda_kv_{k,old}}{S_k+\lambda_k},
\]

set \(\lambda_k=\rho S_k\), giving

\[
v_{k,new}=\frac{1}{1+\rho}\bar x_k+\frac{\rho}{1+\rho}v_{k,old}.
\]

Equivalently select \(\alpha=\rho/(1+\rho)\in[0,1]\), with \(v_{new}=(1-\alpha)\bar x+\alpha v_{old}\). `alpha=0` is full refit and `alpha=1` is freeze. This makes nominal shrinkage comparable across cluster masses. If \(S_k\) is empty or below a preregistered threshold, the candidate is invalid or that cluster is frozen; it must not divide by a near-zero mass.

## Adaptation cost

The draft denominator \(\|V_{old}\|_F^2\) changes under translation and can make identical motion look different after shifting the coordinate origin. Use a source-scale denominator instead:

\[
C_t(\alpha)=
\frac{\sum_k w_k\|v_{k,new}-v_{k,old}\|_2^2}
     {\sum_k w_k(s_k^R)^2+\epsilon},
\]

where `s_k^R` and `w_k` are frozen source radius and mass. This is translation invariant and, under a common rescaling, dimensionless. Report per-cluster costs as well so a small cluster is not hidden by the aggregate.

## Predictive uncertainty

Ensemble dispersion \(\mu+\kappa\sigma\) is a score, not automatically an upper confidence bound. A future study must reserve calibration datasets/groups, construct intervals without reusing training groups (for example group conformal or calibrated quantile regression), and report marginal and per-dataset-family coverage, interval width, and coverage conditional on predicted risk. The selector may call a bound conservative only after its target coverage is demonstrated on untouched groups.

## Negative adaptation reporting

For a tolerance \(\delta\) fixed before outcomes, report all of:

- adaptation coverage \(P(A=1)\);
- unconditional harm \(P(A=1, Q_{adapt}<Q_{freeze}-\delta)\);
- conditional harm \(P(Q_{adapt}<Q_{freeze}-\delta\mid A=1)\), reported as undefined with a zero-adaptation flag when its denominator is zero;
- mean, median, and worst harmful magnitude \(\max(0,Q_{freeze}-Q_{adapt})\).

The original conditional NAR alone can be improved by almost never adapting and is not an adequate headline measure.

## Degradation curves

Mild/severe labels are not commensurate physical coordinates across location, missingness, outliers, prevalence, and overlap. Report family-specific curves. Pool an AUDC only if each family receives a preregistered, scientifically meaningful normalized dose scale and the estimand explicitly averages those normalized curves.

## Synthetic soft truth

For a Gaussian component, use the Gaussian density in Bayes' rule. For the heavy-tailed Student-t mixture, use its own density:

\[
\tau_{ik}=\frac{\pi_k t_{\nu_k}(x_i\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j t_{\nu_j}(x_i\mid\mu_j,\Sigma_j)}.
\]

The multivariate Student-t density contains the degrees-of-freedom-specific gamma normalizer and \([1+(x-\mu)^T\Sigma^{-1}(x-\mu)/\nu]^{-(\nu+D)/2}\). Reusing the Gaussian posterior for `s08_student_t_mixture` would create incorrect soft ground truth.

The current generator implementation was checked and already branches to a multivariate Student-t log density for the repository family `s08_student_t_mixture`; this is **not** a current code defect. The draft calls the corresponding planned family S09, so the defect is the draft’s unrestricted Gaussian-only formula/generalization unless the family-specific implementation is made explicit.

## Terminology constraint

Phase 7 estimates contemporaneous hard-ARI degradation from a current unlabeled batch. The defensible phrase is **target-label-free estimation of hard clustering degradation at deployment**. “Future failure prediction,” “early warning,” and “soft-membership failure prediction” require separate temporal and soft-truth experiments.
