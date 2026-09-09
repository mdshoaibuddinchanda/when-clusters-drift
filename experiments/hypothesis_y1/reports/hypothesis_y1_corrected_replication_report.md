# Hypothesis Y1 corrected-replication report

Status: post-hoc methodological correction and failure analysis. It does not replace the frozen preregistered Phase-7 result.

## Side-by-side metrics

“Duplicate/class-corrected” is one joint split arm because the same outcome-blind grouped split repair fixes exact-row/entity leakage and Phase-7 target-class coverage. The 150-iteration arm isolates that split correction. The full arm adds the diagnostic-fixed 600-iteration horizon and requires both source and candidate convergence.

| Analysis | Evaluation | Block | MAE | RMSE | R² | n predictions |
|---|---|---|---:|---:|---:|---:|
| ORIGINAL PHASE 7 | LODO | P0 | 0.046705 | 0.077379 | -0.121098 | 4200 |
| ORIGINAL PHASE 7 | LODO | P3 | 0.063926 | 0.091623 | -0.571841 | 4200 |
| ORIGINAL PHASE 7 | LODO | P4 | 0.050895 | 0.079636 | -0.187478 | 4200 |
| ORIGINAL PHASE 7 | LOSFO | P0 | 0.047815 | 0.078698 | -0.159650 | 4200 |
| ORIGINAL PHASE 7 | LOSFO | P3 | 0.043299 | 0.075153 | -0.057544 | 4200 |
| ORIGINAL PHASE 7 | LOSFO | P4 | 0.046241 | 0.078588 | -0.156428 | 4200 |
| INDEPENDENT REPLICATION | LODO | P0 | 0.046705 | 0.077379 | -0.121098 | 4200 |
| INDEPENDENT REPLICATION | LODO | P3 | 0.063926 | 0.091623 | -0.571841 | 4200 |
| INDEPENDENT REPLICATION | LODO | P4 | 0.050895 | 0.079636 | -0.187478 | 4200 |
| INDEPENDENT REPLICATION | LOSFO | P0 | 0.047815 | 0.078698 | -0.159650 | 4200 |
| INDEPENDENT REPLICATION | LOSFO | P3 | 0.043299 | 0.075153 | -0.057544 | 4200 |
| INDEPENDENT REPLICATION | LOSFO | P4 | 0.046241 | 0.078588 | -0.156428 | 4200 |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LODO | P0 | 0.049697 | 0.081124 | -0.127180 | 3604 |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LODO | P3 | 0.066523 | 0.097250 | -0.619833 | 3604 |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LODO | P4 | 0.053394 | 0.083695 | -0.199764 | 3604 |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LOSFO | P0 | 0.051209 | 0.083059 | -0.181584 | 3604 |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LOSFO | P3 | 0.046579 | 0.080645 | -0.113902 | 3604 |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LOSFO | P4 | 0.049291 | 0.082210 | -0.157551 | 3604 |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LODO | P0 | 0.042651 | 0.073346 | -0.094376 | 4200 |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LODO | P3 | 0.062779 | 0.092118 | -0.726270 | 4200 |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LODO | P3_delta | 0.044069 | 0.072287 | -0.063010 | 4200 |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LODO | P4 | 0.046511 | 0.076168 | -0.180215 | 4200 |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LOSFO | P0 | 0.043835 | 0.075110 | -0.147654 | 4200 |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LOSFO | P3 | 0.043840 | 0.077797 | -0.231228 | 4200 |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LOSFO | P3_delta | 0.043237 | 0.073836 | -0.109044 | 4200 |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LOSFO | P4 | 0.043265 | 0.074637 | -0.133249 | 4200 |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LODO | P0 | 0.042760 | 0.073337 | -0.080727 | 4145 |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LODO | P3 | 0.064472 | 0.094942 | -0.811296 | 4145 |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LODO | P3_delta | 0.043784 | 0.072140 | -0.045744 | 4145 |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LODO | P4 | 0.048387 | 0.078292 | -0.231690 | 4145 |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LOSFO | P0 | 0.044843 | 0.075908 | -0.157824 | 4145 |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LOSFO | P3 | 0.043541 | 0.075716 | -0.151987 | 4145 |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LOSFO | P3_delta | 0.043311 | 0.073803 | -0.094518 | 4145 |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LOSFO | P4 | 0.043507 | 0.075103 | -0.133395 | 4145 |
| FAIR DELTA-VALIDITY CONTROL | LODO | P3-delta | 0.045597 | 0.078293 | -0.147750 | 4200 |
| NO-SKILL BASELINE | LODO | B0_mean | 0.042690 | 0.073843 | -0.020980 | 4200 |
| NO-SKILL BASELINE | LODO | B0_median | 0.038008 | 0.075039 | -0.054331 | 4200 |
| FAIR DELTA-VALIDITY CONTROL | LOSFO | P3-delta | 0.049255 | 0.086668 | -0.406450 | 4200 |
| NO-SKILL BASELINE | LOSFO | B0_mean | 0.043155 | 0.074705 | -0.044967 | 4200 |
| NO-SKILL BASELINE | LOSFO | B0_median | 0.038284 | 0.075283 | -0.061211 | 4200 |

MAE/RMSE/R² above are pooled over the retained held-out predictions. The win counts and paired bootstrap below deliberately give each held-out dataset or shift family one inferential unit; where convergence filtering makes unit sizes unequal, their mean contrast need not equal the difference between pooled MAEs.

## Effect of correction on MAE

Positive corrected-minus-original differences are explicitly marked as worse; none are filtered from the report.

| Corrected arm | Evaluation | Block | original MAE | corrected MAE | corrected − original | direction |
|---|---|---|---:|---:|---:|---|
| duplicate/class-corrected 150 | LODO | P0 | 0.046705 | 0.042651 | -0.004054 | better |
| duplicate/class-corrected 150 | LODO | P3 | 0.063926 | 0.062779 | -0.001147 | better |
| duplicate/class-corrected 150 | LODO | P4 | 0.050895 | 0.046511 | -0.004384 | better |
| duplicate/class-corrected 150 | LOSFO | P0 | 0.047815 | 0.043835 | -0.003980 | better |
| duplicate/class-corrected 150 | LOSFO | P3 | 0.043299 | 0.043840 | 0.000540 | worse |
| duplicate/class-corrected 150 | LOSFO | P4 | 0.046241 | 0.043265 | -0.002976 | better |
| full corrected 600/both | LODO | P0 | 0.046705 | 0.042760 | -0.003945 | better |
| full corrected 600/both | LODO | P3 | 0.063926 | 0.064472 | 0.000547 | worse |
| full corrected 600/both | LODO | P4 | 0.050895 | 0.048387 | -0.002508 | better |
| full corrected 600/both | LOSFO | P0 | 0.047815 | 0.044843 | -0.002972 | better |
| full corrected 600/both | LOSFO | P3 | 0.043299 | 0.043541 | 0.000242 | worse |
| full corrected 600/both | LOSFO | P4 | 0.046241 | 0.043507 | -0.002734 | better |

## Unit wins and grouped bootstrap

Every contrast is `reference MAE − candidate MAE`; a positive value and a win mean the candidate named on the right has lower MAE. Bootstrap resampling uses the preregistered 10,000 draws and seed at the dataset or shift-family unit, never 4,200 rows as independent units.

| Analysis | Evaluation | Contrast | right-side wins | mean contrast | 95% grouped-bootstrap CI |
|---|---|---|---:|---:|---:|
| ORIGINAL PHASE 7 | LODO | P0 MAE − P4 MAE | 5/12 | -0.004190 | [-0.011854, 0.002437] |
| ORIGINAL PHASE 7 | LODO | P3 MAE − P4 MAE | 10/12 | 0.013031 | [0.002951, 0.023306] |
| ORIGINAL PHASE 7 | LOSFO | P0 MAE − P4 MAE | 4/7 | 0.001573 | [-0.000678, 0.003900] |
| ORIGINAL PHASE 7 | LOSFO | P3 MAE − P4 MAE | 3/7 | -0.002942 | [-0.006420, 0.000718] |
| INDEPENDENT REPLICATION | LODO | P0 MAE − P4 MAE | 5/12 | -0.004190 | [-0.011854, 0.002437] |
| INDEPENDENT REPLICATION | LODO | P3 MAE − P4 MAE | 10/12 | 0.013031 | [0.002951, 0.023306] |
| INDEPENDENT REPLICATION | LOSFO | P0 MAE − P4 MAE | 4/7 | 0.001573 | [-0.000678, 0.003900] |
| INDEPENDENT REPLICATION | LOSFO | P3 MAE − P4 MAE | 3/7 | -0.002942 | [-0.006420, 0.000718] |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LODO | P0 MAE − P4 MAE | 7/11 | -0.003565 | [-0.012681, 0.003431] |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LODO | P3 MAE − P4 MAE | 7/11 | 0.013184 | [0.001635, 0.026172] |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LOSFO | P0 MAE − P4 MAE | 4/7 | 0.001896 | [-0.001922, 0.006273] |
| CONVERGED-ONLY (S3, ORIGINAL ROWS) | LOSFO | P3 MAE − P4 MAE | 3/7 | -0.002685 | [-0.007405, 0.002022] |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LODO | P0 MAE − P4 MAE | 6/12 | -0.003860 | [-0.010478, 0.001376] |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LODO | P3 MAE − P4 MAE | 8/12 | 0.016268 | [0.002882, 0.032782] |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LODO | P3_delta MAE − P4 MAE | 4/12 | -0.002442 | [-0.009088, 0.003506] |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LOSFO | P0 MAE − P4 MAE | 4/7 | 0.000570 | [-0.002664, 0.004075] |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LOSFO | P3 MAE − P4 MAE | 4/7 | 0.000575 | [-0.002078, 0.003397] |
| DUPLICATE/CLASS-CORRECTED (150 ALL) | LOSFO | P3_delta MAE − P4 MAE | 3/7 | -0.000027 | [-0.002755, 0.002731] |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LODO | P0 MAE − P4 MAE | 7/12 | -0.005672 | [-0.016315, 0.001453] |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LODO | P3 MAE − P4 MAE | 8/12 | 0.015869 | [0.002912, 0.032239] |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LODO | P3_delta MAE − P4 MAE | 4/12 | -0.004458 | [-0.015393, 0.003329] |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LOSFO | P0 MAE − P4 MAE | 5/7 | 0.001340 | [-0.002371, 0.005329] |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LOSFO | P3 MAE − P4 MAE | 4/7 | 0.000030 | [-0.003713, 0.003344] |
| FULL CORRECTED Y1 (600, BOTH CONVERGED) | LOSFO | P3_delta MAE − P4 MAE | 3/7 | -0.000190 | [-0.002914, 0.002738] |
| NO-SKILL BASELINE | LODO | B0_mean MAE − P0 MAE | 3/12 | -0.004014 | [-0.006789, -0.001561] |
| NO-SKILL BASELINE | LODO | B0_mean MAE − P3 MAE | 1/12 | -0.021235 | [-0.033777, -0.009275] |
| NO-SKILL BASELINE | LODO | B0_mean MAE − P4 MAE | 4/12 | -0.008204 | [-0.015351, -0.001930] |
| NO-SKILL BASELINE | LODO | B0_median MAE − P0 MAE | 3/12 | -0.008696 | [-0.013157, -0.003768] |
| NO-SKILL BASELINE | LODO | B0_median MAE − P3 MAE | 1/12 | -0.025918 | [-0.039518, -0.013276] |
| NO-SKILL BASELINE | LODO | B0_median MAE − P4 MAE | 3/12 | -0.012887 | [-0.022784, -0.003917] |
| NO-SKILL BASELINE | LOSFO | B0_mean MAE − P0 MAE | 0/7 | -0.004659 | [-0.007505, -0.002858] |
| NO-SKILL BASELINE | LOSFO | B0_mean MAE − P3 MAE | 5/7 | -0.000144 | [-0.005156, 0.003985] |
| NO-SKILL BASELINE | LOSFO | B0_mean MAE − P4 MAE | 2/7 | -0.003086 | [-0.005204, -0.000822] |
| NO-SKILL BASELINE | LOSFO | B0_median MAE − P0 MAE | 1/7 | -0.009530 | [-0.017328, -0.002891] |
| NO-SKILL BASELINE | LOSFO | B0_median MAE − P3 MAE | 2/7 | -0.005015 | [-0.013376, 0.001672] |
| NO-SKILL BASELINE | LOSFO | B0_median MAE − P4 MAE | 1/7 | -0.007957 | [-0.014101, -0.002067] |
| FAIR DELTA-VALIDITY CONTROL | LODO | P0 MAE − P3-delta MAE | 7/12 | 0.001108 | [-0.005415, 0.007194] |
| FAIR DELTA-VALIDITY CONTROL | LODO | P4 MAE − P3-delta MAE | 7/12 | 0.005298 | [-0.001842, 0.013378] |
| FAIR DELTA-VALIDITY CONTROL | LOSFO | P0 MAE − P3-delta MAE | 3/7 | -0.001441 | [-0.016444, 0.010995] |
| FAIR DELTA-VALIDITY CONTROL | LOSFO | P4 MAE − P3-delta MAE | 3/7 | -0.003014 | [-0.016953, 0.008002] |
| FULL CORRECTED NO-SKILL | LODO | B0_mean MAE − P0 MAE | 2/12 | -0.003790 | [-0.006622, -0.001358] |
| FULL CORRECTED NO-SKILL | LODO | B0_mean MAE − P3 MAE | 2/12 | -0.025331 | [-0.049807, -0.006166] |
| FULL CORRECTED NO-SKILL | LODO | B0_mean MAE − P3_delta MAE | 5/12 | -0.005004 | [-0.010358, 0.000451] |
| FULL CORRECTED NO-SKILL | LODO | B0_mean MAE − P4 MAE | 3/12 | -0.009462 | [-0.020737, -0.001732] |
| FULL CORRECTED NO-SKILL | LODO | B0_median MAE − P0 MAE | 3/12 | -0.007018 | [-0.011306, -0.002645] |
| FULL CORRECTED NO-SKILL | LODO | B0_median MAE − P3 MAE | 2/12 | -0.028559 | [-0.054372, -0.007959] |
| FULL CORRECTED NO-SKILL | LODO | B0_median MAE − P3_delta MAE | 4/12 | -0.008232 | [-0.015094, -0.000959] |
| FULL CORRECTED NO-SKILL | LODO | B0_median MAE − P4 MAE | 3/12 | -0.012689 | [-0.025371, -0.003442] |
| FULL CORRECTED NO-SKILL | LOSFO | B0_mean MAE − P0 MAE | 1/7 | -0.005106 | [-0.008386, -0.002285] |
| FULL CORRECTED NO-SKILL | LOSFO | B0_mean MAE − P3 MAE | 2/7 | -0.003796 | [-0.006730, -0.000767] |
| FULL CORRECTED NO-SKILL | LOSFO | B0_mean MAE − P3_delta MAE | 2/7 | -0.003576 | [-0.007904, 0.000492] |
| FULL CORRECTED NO-SKILL | LOSFO | B0_mean MAE − P4 MAE | 2/7 | -0.003766 | [-0.007915, -0.000171] |
| FULL CORRECTED NO-SKILL | LOSFO | B0_median MAE − P0 MAE | 0/7 | -0.008965 | [-0.014402, -0.004285] |
| FULL CORRECTED NO-SKILL | LOSFO | B0_median MAE − P3 MAE | 0/7 | -0.007656 | [-0.012151, -0.003472] |
| FULL CORRECTED NO-SKILL | LOSFO | B0_median MAE − P3_delta MAE | 1/7 | -0.007436 | [-0.011520, -0.003648] |
| FULL CORRECTED NO-SKILL | LOSFO | B0_median MAE − P4 MAE | 1/7 | -0.007626 | [-0.011391, -0.003446] |

Selected headline checks:

- Original LODO P0−P4: 5/12 dataset wins; mean Δ=-0.004190, 95% grouped bootstrap CI [-0.011854, 0.002437].
- Original LODO P3−P4: 10/12 dataset wins; mean Δ=0.013031, 95% grouped bootstrap CI [0.002951, 0.023306].
- Original LOSFO P0−P4: 4/7 family wins; mean Δ=0.001573, 95% grouped bootstrap CI [-0.000678, 0.003900].
- Original LOSFO P3−P4: 3/7 family wins; mean Δ=-0.002942, 95% grouped bootstrap CI [-0.006420, 0.000718].
- Converged-only LODO P0−P4: 7/11; mean Δ=-0.003565, 95% grouped bootstrap CI [-0.012681, 0.003431].
- Duplicate/class-corrected LODO P0−P4: 6/12; mean Δ=-0.003860, 95% grouped bootstrap CI [-0.010478, 0.001376].
- Full-corrected LODO P0−P4: 7/12; mean Δ=-0.005672, 95% grouped bootstrap CI [-0.016315, 0.001453].
- Full-corrected LODO P3-delta−P4: 4/12; mean Δ=-0.004458, 95% grouped bootstrap CI [-0.015393, 0.003329].
- Full-corrected LOSFO P0−P4: 5/7; mean Δ=0.001340, 95% grouped bootstrap CI [-0.002371, 0.005329].
- Full-corrected LOSFO P3-delta−P4: 3/7; mean Δ=-0.000190, 95% grouped bootstrap CI [-0.002914, 0.002738].

## Retention and validity

The full corrected arm retains 4145/4200 shifted scenario rows (98.69%). The split verifier reports zero cross-fold connected-group leakage in all corrected outer and inner folds. All twelve Phase-7 datasets have every manifest class in source and target after correction. Five Ecoli target folds in the broader 30-dataset split audit remain class-incomplete because its two rarest classes contain only two samples each, making five-fold target coverage mathematically impossible; Ecoli is not in the corrected Phase-7 panel.

| Dataset | retained | excluded by policy | retention |
|---|---:|---:|---:|
| breast_cancer_wisconsin_diagnostic | 350 | 0 | 100.00% |
| glass | 350 | 0 | 100.00% |
| human_activity_recognition | 349 | 1 | 99.71% |
| iris | 350 | 0 | 100.00% |
| letter_recognition | 296 | 54 | 84.57% |
| madelon | 350 | 0 | 100.00% |
| mice_protein_expression | 350 | 0 | 100.00% |
| pendigits | 350 | 0 | 100.00% |
| satimage | 350 | 0 | 100.00% |
| sonar | 350 | 0 | 100.00% |
| spambase | 350 | 0 | 100.00% |
| waveform | 350 | 0 | 100.00% |

The exploratory frozen-row fair control uses each clean target-fold scenario as its label-free reference. In the corrected arms, `P3_delta` is stricter and deployment-oriented: `current − source/reference-probe` FPC, PE, XB and silhouette, plus D_X. These two controls answer the same fairness objection but are not numerically interchangeable, so the report does not pretend they are one result.

## Interpretation

Correcting real methodological defects does not rescue structural P4 against P0 and grouped no-skill evidence. The fair P3-delta control is reported regardless of direction. No case where a correction worsened a metric has been omitted. Final decision: `FAILURE_CONFIRMED_STOP_PROJECT`.
