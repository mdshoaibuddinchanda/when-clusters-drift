# Hypothesis Y1 FCM convergence root-cause audit

Status: post-hoc diagnostic; the frozen Phase 7 verdict is unchanged.

## Finding

Phase 7 marked all 4,500 rows `usable`, although only 3,865 (85.89%) had both a converged source and converged candidate fit. Source convergence was 3,945/4,500 (87.67%) and candidate convergence was 4,071/4,500 (90.47%). The 635 accepted rows lacking one or both convergence guarantees are a confirmed methodological defect.

Letter Recognition is the dominant pathology: no source row converged (0/375), only 31/375 candidate rows converged, and therefore 0/375 rows had both fits converged. Other source convergence shortfalls occurred in Glass (84%), HAR (92%), Mice (92%), Pendigits (92%), Iris (96%), and Sonar (96%).

## Diagnostic design

The diagnostic reran all unique originally failed source fits, every non-Letter failed candidate fit, a deterministic Letter sample covering every condition and every fold/seed pair, and two matched converged controls per dataset. It retained the original data, k-means++ initialization, dimension-adaptive fuzzifier, seed, and `tol=1e-5`; only the horizon increased from 150 to 600. This yielded 207 reruns, of which 161 represented original failures.

## Evidence

- 39/161 failed fits were already within ten tolerances at iteration 150.
- 110/161 formally converged by iteration 300; 153/161 by iteration 600.
- No rerun showed a material objective increase, numerical failure, empty cluster, or degenerate solution.
- Four unresolved fits at 600 were near-converged; four remained clearly non-converged.
- The eight unresolved fits comprised seven Letter fits and one Satimage outlier-severe candidate fit. Two Letter cases had very large terminal shifts (over 1,000 times tolerance), showing that not every failure is merely a boundary rounding case.
- Failed-fit effective fuzzifier values ranged from 1.01 to 1.50, dimensions from 4 to 561, and K from 2 to 26. Failures are therefore not exclusive to one dimension regime, although large K/Letter is the strongest concentration.

## Root-cause classification

The principal cause is `max_iter=150` truncating slowly converging FCM trajectories under particular dataset/initialization/shift combinations. The original tolerance is sometimes strict relative to the tail rate, but changing it is not justified: 122/161 sampled failures were more than ten tolerances away at iteration 150. Dimension-adaptive fuzzification and high K plausibly contribute to slow dynamics, especially on Letter, but this diagnostic does not establish either as a sole causal mechanism. There is no evidence here of objective oscillation or an implementation defect in the FCM update equations.

## Consequence

Finite non-converged outputs can still produce signals and ARI values, but their numerical target is iteration-budget-dependent. They must not silently enter a primary scientific comparison. The corrected policy in `hypothesis_y1_convergence_policy.yaml` requires both fits to converge by a performance-blind 600-iteration horizon and reports attrition explicitly. The converged-only predictive sensitivity separately determines whether this contamination materially changed the P4 conclusion.
