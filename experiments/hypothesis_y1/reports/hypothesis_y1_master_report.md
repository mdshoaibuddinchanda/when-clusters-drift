# Hypothesis Y1 master scientific audit

## 1. Executive conclusion

`FAILURE_CONFIRMED_STOP_PROJECT`. The historical negative result is computationally reproducible. Convergence eligibility, duplicate/group splitting, target-class coverage, comparator fairness, dependence, and post-label implementation timing contain genuine defects, but the isolated corrected analysis does not recover predictive skill for P4. The grouped median no-skill predictor remains a decisive benchmark. **DO NOT CONTINUE ORIGINAL PHASE 8–15 ROADMAP.**

## 2. Historical frozen Phase-7 result

At commit `c9280bb390492f4528752f5457e341fb3d9c1040`, LODO MAE was P0 `0.046704663478981584`, P3 `0.06392587674262011`, and P4 `0.05089470458393738`. Relative improvements were R04 `-0.08971354877316329` and R34 `0.2038481570014151`. P4 won 5/12 datasets against P0 and 10/12 against P3, and 4/7 and 3/7 families respectively. Verdict: `FAILS_PRIMARY_FALSIFICATION`. Frozen inputs and results were not edited.

## 3. Independent reproduction

The independent script used only the two frozen CSVs and preregistered YAML and imported none of the original evaluator/model/bootstrap/verification modules. Agreement status is `PASSED` at tolerance `1e-12`; all continuous discrepancies are below `1e-12`, discrete wins/rules match, and the verdict is identical. An initial reporting-order bootstrap mismatch is retained as negative implementation evidence and was resolved by restoring the preregistered dataset/family order, not by changing values or rules.

## 4. Confirmed methodological problems

| ID | Classification | Confirmed problem |
|---|---|---|
| M01 | A. FIX_NOW_METHODOLOGICAL | `usable` accepted 635 rows without both FCM fits converged |
| M02 | A. FIX_NOW_METHODOLOGICAL | exact duplicates crossed frozen splits in four Phase-7 datasets |
| M03 | A. FIX_NOW_METHODOLOGICAL | four target folds lacked at least one Oracle-K class |
| M04 | A. FIX_NOW_METHODOLOGICAL | no grouped no-skill baseline was in the gate |
| M05 | A. FIX_NOW_METHODOLOGICAL | raw validity levels were compared with structural changes |
| M06 | A. FIX_NOW_METHODOLOGICAL | five seeds/scenarios are dependent and row-level inference would pseudoreplicate |
| M07 | A. FIX_NOW_METHODOLOGICAL | major evaluator code changed after quality labels were available |
| M08 | D. DOCUMENTATION_ONLY | classification outcomes were over-described as natural cluster truth |
| M09 | D. DOCUMENTATION_ONLY | contemporaneous hard-ARI estimation was called future soft-clustering failure prediction |

## 5. Problems rejected as non-issues

The independent evaluator rejects a bad join, wrong metric aggregation, hidden target column, changed HGBR grid/tie rule, and wrong verdict implementation as explanations. The convergence diagnostic found no objective increase, empty cluster, degeneracy, or numerical failure in 207 reruns. Source-only preprocessing and dataset-grouped outer evaluation were correctly implemented. Empty future files, the unfinished bibliography, and manuscript placeholders are expected incomplete rather than scientific defects.

## 6. Exact fixes implemented

Fifteen bounded fixes were implemented without altering a frozen result:

- M-FIX-01 require both FCM fits to converge under the outcome-blind 600-iteration policy
- M-FIX-02 group exact duplicates and protected entities while preserving temporal splits
- M-FIX-03 enforce/report source and target class coverage for Oracle-K folds
- M-FIX-04 add grouped mean and median no-skill baselines
- M-FIX-05 add clean-referenced conventional validity deltas
- M-FIX-06 use dataset/family/scenario grouping for uncertainty and within-dataset validation
- S-FIX-01 remove executable joblib cache deserialization
- S-FIX-02 digest and shape-check cached numerical payloads
- S-FIX-03 bind scalar caches to provenance fingerprints
- S-FIX-04 bind cache keys to data/split/probe/protocol/method identities
- S-FIX-05 reject path traversal and unsafe recursive cache roots
- S-FIX-06 use unique atomic cache temporaries
- S-FIX-07 pin, bound, verify, and atomically publish direct archive downloads
- S-FIX-08 atomically publish the final verification manifest
- S-FIX-09 atomically publish canonical data, split, manifest, and baseline-audit artifacts

## 7. FCM convergence findings

Source convergence is 3945/4500 (87.67%); candidate convergence is 4071/4500 (90.47%); both is 3865/4500 (85.89%). Letter is severe at 0/375 both. Of 161 diagnosed original failures, 110 converged by 300 and 153 by 600; eight remained unresolved. Root cause is principally slow convergence truncated at 150, not demonstrated update-equation failure. S3 remains adverse: pooled LODO P0 `0.049697`, P4 `0.053394`. The full 600-iteration policy retains 4,145/4,200 shifted rows; 54 excluded rows are Letter and one is HAR, and exclusions remain explicit rather than silently usable.

## 8. Duplicate leakage findings

Across all 30 controlled datasets, nine have at least one frozen fold with exact-row crossing: banknote_authentication, glass, haberman_survival, image_segmentation, ionosphere, iris, letter_recognition, spambase, yeast. The largest target leakage fraction is 19.35% (Haberman). The Phase-7 subset is glass, iris, letter_recognition, spambase; its maximum is 11.94% (Spambase). Connected exact-row/protected-entity groups eliminate all corrected outer and inner crossings.

## 9. Class/K validity

Oracle K is explicitly sourced from manifest labels and is acceptable only as a controlled external-recovery protocol. Frozen source folds cover all classes, but 4 Phase-7 target folds do not: Glass folds 0/2 and Mice folds 1/4. Corrected Phase-7 folds all cover K. The broader corrected audit leaves 5 Ecoli target folds incomplete because two classes have only two observations, so five-fold target coverage is impossible; future Ecoli work must reduce folds or change the estimand before outcomes.

## 10. No-skill baseline

Grouped frozen-row LODO mean MAE is `0.042690` and median MAE is `0.038008`. Grouped LOSFO mean/median are `0.043155` / `0.038284`. The median beats original P0, P3, and P4 in MAE. In the fully corrected arm, LODO mean/median no-skill MAE is `0.038936` / `0.035675`, again below every learned block. Negative learned-model R² reinforces the absence of magnitude-prediction skill.

## 11. Fair validity-delta control

Replacing raw FPC/PE/XB/silhouette levels with same-dataset/fold/seed clean-reference deltas yields LODO MAE `0.045597` and LOSFO `0.049255`. It beats P0 on 7/12 datasets but only 3/7 families and remains worse than the LODO median no-skill baseline. The correction improves comparator fairness; it does not rescue the structural thesis.

## 12. Original vs corrected replication

The full side-by-side metrics, wins, grouped bootstrap intervals and cases worsened by correction are in `hypothesis_y1_corrected_replication_report.md`. Full corrected retention is 98.69%; LODO P0/source-reference P3-delta/P4 are `0.042760` / `0.043784` / `0.048387`, with median no-skill `0.035675`. Unlike the clean-target-referenced exploratory control in section 11, corrected P3-delta uses current minus source/reference-probe validity values. The correction does not materially change the negative answer.

## 13. Construct validity of labels

The 54-unit construct inventory has A/B/C counts `4/8/42`. Only Iris is tier A in Phase 7; the other eleven are supervised classes/outcomes used as stress proxies. The results are defensible as external hard-label recovery stress tests, not proof about natural latent clusters in general.

## 14. Hard ARI vs soft-clustering terminology

Phase 7 predicts `ARI_clean−ARI_shifted` after hardening FCM assignments. Soft memberships generate candidate signals, but the outcome is not soft-membership fidelity. Claims are restricted to target-label-free contemporaneous estimation of hard clustering degradation. Soft failure requires posterior-membership targets on families with valid latent posteriors.

## 15. Within-vs-cross dataset result

P4 improves P0 in 6/12 within-dataset models and 5/12 LODO results. Directions agree in only 3/12, and only satimage improves in both. This is active counterevidence against a stable dataset-calibration rescue.

## 16. Signal calibration result

Several raw signal levels track dataset identity: FPC vs source entropy Spearman `-0.839`, PE vs source entropy `0.811`, XB vs dimension `0.823`, D_H vs source entropy `0.720`, and D_U_R vs K `0.611`. D_U and D_H between/within variance ratios exceed one, while D_V is `0.067`. One predeclared normalization per signal family was registered; no outcome-driven transform sweep was conducted.

## 17. D_V mechanism

D_V is already normalized by frozen reference radius. Its overall Spearman with degradation is `0.054`; positive Letter/Mice associations coexist with negative Sonar/Spambase associations. D_V may describe prototype motion, but no transferable degradation mechanism repeats strongly enough to justify a D_V-only hypothesis.

## 18. D_H mechanism

Absolute D_H loses direction and relates strongly to source entropy. The single preregistered source-relative test gives LODO MAE `0.050961`, versus original P4 `0.050895`. It does not rescue P4, so “D_H alone destroys transfer” is rejected.

## 19. Dimension/fuzzifier mechanism

Effective m is deterministically tied to dimension under the adaptive rule. Non-convergence is concentrated in high-K Letter, but diagnosed failures span D=4–561 and K=2–26. A longer fixed horizon resolves 153/161 sampled failures without changing tolerance. The evidence supports an iteration-budget interaction, not a causal claim that adaptive fuzzification alone caused predictive failure.

## 20. Shift-family result

Original P4 beats P0 for scale, measurement noise, class prevalence, and local overlap, but loses for location, MCAR, and outliers. Against P3 it wins only outliers, measurement noise, and local overlap. Effects are small and inconsistent; LOSFO learned models also lose the median no-skill comparator.

## 21. Regression-vs-risk-ranking result

All original HGBR blocks have negative LODO R² (P4 `-0.187`). P4 ranking Spearman is `0.042`. At ΔARI>0.05 its AUROC is `0.184` and balanced accuracy `0.500`. Neither fixed-threshold classification nor ranking rescues exact regression.

## 22. Signal redundancy

D_U_R and D_U_C have raw cross-dataset Spearman `0.971`, with VIF about 15 and 18; D_M is also strongly correlated with them. Redundancy can impair stable attribution, but HGBR can tolerate correlation, so it is a plausible contributor rather than a proven root cause.

## 23. Dataset influence

Madelon accounts for about 69% of the net adverse mean P0−P4 sum. Removing Madelon, Sonar, and Letter flips the mean sign, but the official verdict never changes under the recorded influence sequence. No dataset was dropped; the analysis demonstrates heterogeneity and warns against post-hoc panel selection.

## 24. Severity consistency

Severe exceeds mild for ΔARI in only `65.14%` of matched scenarios. D_X is monotone in `91.43%`, while structural signals range roughly 65.7–78.5%. Generator dose order therefore does not guarantee performance-loss order and raw family severities are not commensurate.

## 25. Post-label code-change audit

Between Pass-B and preflight, 6,311 insertions and 338 deletions touched the joined table and core evaluator/verification files. Scientific configs remained unchanged, and the independent evaluator matches outputs. This is a confirmed procedural exposure with no demonstrated numerical bias. Full evidence is in `hypothesis_y1_post_label_change_audit.md`.

## 26. Security/integrity findings

The focused scan found 9 real software/scientific-integrity defects and repaired all: executable joblib cache loading, missing payload digests, filename-only scalar cache acceptance, incomplete provenance keys, unsafe path components/clear containment, predictable temp files, unpinned/non-atomic archives, non-atomic final verification manifest, and direct canonical scientific writes. Possible committed credential matches: 0; current unsafe-deserialization/`shell=True`/`os.system` matches: 0. Provider-managed acquisition byte pinning and an empty lockfile remain bounded reproducibility limitations.

## 27. Repository cleanup

Only clearly generated Python/pytest caches and documented duplicate runtime cache material are eligible for removal. Frozen results, raw/canonical data, manifests, historical configs, manuscript, placeholders, and negative exploratory evidence are retained. Exact actions and retained categories are in `hypothesis_y1_repository_cleanup_report.md`.

## 28. Remaining limitations

The corrected run is post-hoc, uses the same twelve heavily observed datasets, and cannot become a new confirmatory result. Class labels are usually weak proxies for natural clusters. The 600-iteration rule is diagnostic-derived and still leaves explicit attrition. Only one model family and one bounded normalization per motivated signal were examined. Provider libraries are not pre-acquisition byte-pinned. No literature novelty audit was repeated.

## 29. Evidence against preferred explanation

The strongest explanation is weak, heterogeneous, dataset-calibrated structural association rather than one broken signal. Counterevidence was actively retained: P4 wins 5/12 original datasets and 4/7 families, D_M has overall Spearman `0.128`, and some within-dataset fits improve. Against it, only one dataset improves both within and cross settings, fair validity changes outperform P4 but not no-skill, ranking/classification fail, D_H normalization fails, corrected methodology fails to supply a robust reversal, and directions vary by family. The favorable cases are insufficiently coherent.

## 30. Decision matrix

| mechanism | supporting evidence | contradictory evidence | datasets supporting | families supporting | effect magnitude | confidence | would it invalidate Phase 7? | does it justify new hypothesis? | untouched test required? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-7 evaluator implementation bug | Post-label evaluator rewrite created opportunity for error. | Independent evaluator reproduces all headline values/verdict within 1e-12. | none | none | max headline absolute difference <= 8.4e-17 | high confidence rejected | No | No | No |
| non-convergence contaminated structural signals | 635/4500 usable rows lacked both convergence; Letter 0/375 both. | S3 still has P4=0.0534 vs P0=0.0497. | Letter dominant | mixed | S3 P0-P4=-0.0037 | high defect; high no-rescue | Defect yes; conclusion no | No | Corrected run completed |
| duplicate leakage contaminated folds | Phase7 leakage in glass, iris, letter_recognition, spambase; max Phase7 rate 0.1194. | Duplicate-corrected P0=0.0427, P4=0.0465. | glass, iris, letter_recognition, spambase | all families affected through datasets | corrected P0-P4=-0.0039 | high | Only if corrected reversal; observed in report | No | Completed on same panel |
| missing target classes invalidated folds | Four Phase7 target folds: Glass 0/2 and Mice 1/4. | All Phase7 corrected folds cover K; joint corrected result does not rescue premise. | Glass, Mice | not family-specific | 4/60 original folds | high | Local folds yes; overall conclusion no | No | Completed on same panel |
| raw validity controls were unfair | P3 used levels while P4 used changes. | Fair-delta LODO=0.0456, but median no-skill=0.0380 and LOSFO=0.0493. | 7/12 beat P0 | 3/7 beat P0 | LODO gain vs P0=0.0011 | moderate | Comparator fairness only | No | Yes for any future validity hypothesis |
| raw structural signals are dataset-specific | Strong metadata correlations and inconsistent within/cross directions. | Only ['satimage'] helps in both; some family wins exist. | 6/12 within help, 5/12 cross help | 4/7 P4 vs P0 | same direction 3/12 | high | No | No coherent repeat | Would require untouched data |
| D_V contains real transferable signal | Within-dataset rho up to 0.392 Letter and 0.316 Mice. | Overall rho=0.054; negative Sonar/Spambase; no D_V-only model selected. | Letter, Mice | outliers strongest rho 0.242 | overall rho 0.054 | low | No | No | Yes if separately preregistered |
| D_H destroys transfer | D_H correlates with source entropy and can lose direction. | Source-relative P4 LODO=0.0510, essentially no better than P4=0.0509. | mixed | sign changes across families | delta MAE=0.000066 | high confidence rejected as sole cause | No | No | No |
| dimension-adaptive fuzzification causes instability | Letter/high K is dominant convergence pathology; signal levels relate to D/effective m. | Failures also occur from D=4 to 561 and K=2 to 26; no numerical/degenerate failures. | Letter strongest | not isolated | 153/161 sampled failures converge by 600 | moderate | No | No | Would need fixed-m preregistered comparison |
| exact Delta ARI regression is unsuitable | All learned frozen HGBR R2 values are negative and median no-skill wins MAE. | Signed magnitude remains a legitimate estimand if a model has skill. | all pooled | all pooled | P4 R2=-0.187 | high for current model | No | No | Only after new target justification |
| risk ranking works while magnitude regression fails | P4 ranking rho=0.042. | P4 AUROC at ΔARI>.05=0.184, balanced accuracy=0.500. | none coherent | none coherent | near-zero/anti-skill | high confidence rejected | No | No | No |
| classification labels are poor cluster truth | Construct tiers A/B/C=4/8/42. | ARI still valid for explicitly framed external class-recovery stress tests. | only Iris tier A in Phase7 | all controlled families | 42/54 tier C | high | Invalidates broad natural-cluster claim, not computed comparison | No | Use latent-ground-truth synthetics |
| structural signals are genuinely weak | P4 loses P0 overall, median no-skill wins, structural rho values <= 0.128 overall. | P4 wins selected datasets/families and D_M has modest positive association. | heterogeneous | heterogeneous | original P0-P4=-0.0042 | high for tested formulation | No | No | No automatic continuation |
| dataset generalization differs from shift-family generalization | LODO P4 loses P0 while LOSFO P4 slightly beats P0; P3 relation also reverses. | Both learned evaluations lose grouped median no-skill and bootstrap evidence is weak. | 5/12 P4 wins | 4/7 P4 wins | LODO Δ=-0.0042; LOSFO Δ=0.0016 | high descriptive | No | No | Would require independent datasets and families |
| signal redundancy hurts model transfer | D_U_R/D_U_C Spearman=0.971; VIFs about 15 and 18. | Tree models can tolerate correlated predictors; redundancy is not causal proof. | all | all | rho 0.971 | moderate | No | No | No unless mechanism specified |

## 31. Final project decision

`FAILURE_CONFIRMED_STOP_PROJECT`. The original failure reproduces, genuine repairs do not rescue the premise, and no repeated non-cherry-picked mechanism meets the new-hypothesis criteria.

**DO NOT CONTINUE ORIGINAL PHASE 8–15 ROADMAP.**

## 32. One new hypothesis, ONLY if justified

No new hypothesis is justified. The most promising observations (fair validity deltas, D_M association, selected family wins, and within-dataset improvements) conflict across datasets/families or lose to no-skill. Promoting one would be post-hoc selection forbidden by Y1.

## 33. Untouched confirmatory data

Discovery panel: iris, glass, sonar, breast_cancer_wisconsin_diagnostic, spambase, waveform, satimage, pendigits, letter_recognition, madelon, human_activity_recognition, mice_protein_expression. The following frozen broader-panel datasets remain untouched for **Y1 shift-degradation outcome testing**: wine, seeds, ecoli, yeast, vehicle_silhouettes, image_segmentation, optdigits, banknote_authentication, ionosphere, pima_diabetes, heart_disease, haberman_survival, dermatology, balance_scale, isolet, electricity, bank_marketing, aps_failure. Their schemas, duplicates, groups, and class feasibility were audited, but no corrected Y1 shift/outcome model was run on them. They are reserved only if an independent reviewer defines a genuinely new preregistered question.

## 34. Minimum preregistered next test

Not applicable under `FAILURE_CONFIRMED_STOP_PROJECT`; no confirmation should be run. If an independent review later supplies a coherent mechanism, it must first freeze one target, one transform family, one model family, grouped no-skill baselines, corrected entity/duplicate/time splits, effect threshold, grouped interval, and kill rule, then test the smallest untouched panel. This audit does not propose or execute that experiment.

## 35. Cleanup manifest

`experiments/hypothesis_y1/cleanup/hypothesis_y1_cleanup_manifest.txt` enumerates every retained Y1 file/directory and every deletion record. Test evidence: y1: 19 passed, 0 failed, 0 skipped, exit 0; falsification: 79 passed, 0 failed, 0 skipped, exit 0; full: 402 passed, 0 failed, 0 skipped, exit 0.
