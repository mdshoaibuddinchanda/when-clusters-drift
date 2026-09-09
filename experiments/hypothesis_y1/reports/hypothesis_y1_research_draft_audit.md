# Hypothesis Y1 audit of `RESEARCH_DRAFT.md`

## Overall assessment

The draft is an internally useful research plan, but it is not currently a defensible description of completed evidence. Its central structural-prediction premise failed the frozen Phase-7 falsification, the learned regressors lost to a grouped median no-skill predictor, and most downstream RISA claims remain untested. The draft must not be used as a results manuscript or as authority to start Phases 8–15.

This audit did not edit `RESEARCH_DRAFT.md`. Line references below address all thirteen sections. Literature/bibliography assertions were not independently re-reviewed because the master instructions classify the unfinished bibliography as expected incomplete; none of those assertions is relied upon in the scientific decision.

Classifications are the required Y1 categories:

- `A. FIX_NOW_METHODOLOGICAL`
- `B. FIX_NOW_SOFTWARE_INTEGRITY`
- `C. FUTURE_METHOD_REDESIGN`
- `D. DOCUMENTATION_ONLY`
- `E. EXPECTED_INCOMPLETE`
- `F. NOT_A_REAL_PROBLEM`

## Findings by draft location

| Lines | Finding | Classification | Consequence |
|---:|---|---|---|
| 1–7, 501–506 | The title, “core algorithm,” venue claim, and “core novelty” read as established contributions although the gating hypothesis failed and RISA has not been evaluated. | D | Reframe as a discontinued/conditional research program unless a future independent hypothesis succeeds. |
| 17–21, 25–26 | “Failure prediction,” general soft-clustering applicability, and signals “impossible in hard clustering” are overclaims. Several signals have hard/prototype analogues, and Phase 7 measured contemporaneous hard-ARI degradation for one FCM policy. | D | Narrow terminology and scope. |
| 32–39 | Neighboring fields are called “closed or saturated” based on an unfinished citation record. Those categorical novelty claims are not established by this repository audit. | E | Treat as literature-review placeholders, not findings. |
| 41–53, 82–89 | The flow and RQs call the task “future” prediction and immediately assume a risk-to-adaptation pipeline. Phase 7 uses signals and degradation from the same shifted target batch; it has no prediction horizon. RQ5–RQ8 are downstream and untested. | D | Call the measured task contemporaneous target-batch degradation estimation; stop downstream claims. |
| 56–72 | The motivating distinction between distribution shift and clustering degradation is conceptually sound, but the final arrow to usable risk-guided adaptation is asserted rather than established. | F / D | Preserve the motivation; remove the claimed accomplished chain. |
| 93–95 | H1 compares expectations of predictions rather than prediction error: `E[hat ΔQ(P4)] < E[hat ΔQ(P0)]` is not the stated hypothesis. | A | Define a loss comparison, e.g. `E[L(ΔQ,g(P4)) - L(ΔQ,g(P0))] < 0`, with unit and weighting specified. |
| 96–106 | H2/H3 lack an explicit chance/no-skill criterion, and H4–H6 presume the failed risk model. H4 also needs paired units and an estimand; H5 needs a frozen non-inferiority margin; H6 inherits the NAR defect below. | C | Redesign before any confirmatory work. |
| 116–124 | Source-only preprocessing and the classical FCM formulas are appropriate. Zero-distance membership handling is omitted from prose but implemented in code. | F / D | No scientific repair needed; document numerical edge handling if published. |
| 125, 391, 507 | The draft says primary `m=2`, while Phase 7 used `fcm_adaptive` with `m=max(1.01,1+2/d)`. The five seeds are independent scenario replicates, not an `m=2` primary run. | D | Distinguish the aspirational full benchmark from the actual frozen pilot. |
| 131–141 | Absolute `λ` is confounded by target sample size, soft cluster mass, dimension/fuzzifier, and preprocessing scale. F07 subtracts `v_{t-1}` although the objective anchors to aligned `tilde v_{t-1}`. “Strictly monotonic” holds only conditionally with fixed memberships/nonzero displacement; joint FCM reoptimization does not establish the claimed global theorem. | C | Use dimensionless mass-normalized adaptation and state/prove a matched conditional result. |
| 145–186 | Alignment and probe definitions are reasonable label-free constructions. However, all signals are contemporaneous; `D_H=|H_t-H_{t-1}|` discards direction; and “structural” does not itself imply information about external-label loss. | F / D | Retain as candidate diagnostics, not validated failure predictors. |
| 190–201, 418–421 | P3 uses raw current validity levels, whereas structural predictors are mostly source-to-current changes. This gives the conventional baseline less comparable information. | A | The Y1 fair control subtracts same-fold/seed clean-reference validity levels before grouped evaluation. |
| 192 | “Verify” is outcome-assuming language for a falsification gate. | D | Use “test whether” or “attempt to falsify.” |
| 205–213 | `ΔQ=ARI_clean−ARI_shifted` can be negative (improvement), so calling every value “risk” is misleading. The listed ElasticNet/RF/linear controls were not the frozen primary Phase-7 comparison. | D | Use signed degradation and distinguish planned from executed models. |
| 214–217 | Bootstrap ensemble spread and `μ+κσ` are not a calibrated predictive upper bound. No coverage target or held-out calibration is defined. | C | Use grouped out-of-fold conformal/quantile calibration and report coverage/width. |
| 207–228 | The Phase-7 target is source-model degradation, but the selector requires candidate-specific post-adaptation risk `R_t^(λ)`. A single generic degradation model cannot rank adaptation candidates without a candidate-conditioned target/features. | C | Define candidate-specific utility/loss and train/evaluate at the candidate level. |
| 225–228 | Cost normalization by `||V_{t-1}||²` depends on coordinate origin and can change under a benign translation. | C | Use aligned displacement normalized by source cluster scale, with an explicit batch/mass convention. |
| 230–234 | The theorem minimizes risk alone while the actual selector minimizes estimated risk plus `γC`; its oracle and deployed objectives do not match. The statement also does not cover the added uncertainty score. | C | The future-design report supplies the corrected uniform-error bound for the same penalized objective. |
| 238–244, 332–345, 510 | The Gaussian posterior formula is correct for specified Gaussian mixtures, but cannot be reused for the Student-t family S09. Nonparametric trajectory families also require an explicit generative posterior or must not claim soft ground truth. | C | Define family-specific latent posteriors; use the Student-t density for S09. |
| 248–258 | ARI/AMI are valid external hard-partition metrics and useful stress outcomes. They do not validate “soft-membership failure”; `E_U` is the relevant soft target only where a defensible latent posterior exists. | D | Separate hard label-recovery conclusions from soft-membership conclusions. |
| 259–261, 511 | NAR conditions on “adaptation occurred,” so a method can improve its rate by adapting rarely; it also ignores harm magnitude. | C | Report unconditional harmful-decision rate, adaptation coverage, mean harm/benefit, and regret together. |
| 262–263, 467 | AUDC integrates raw family-specific severity coordinates whose units/endpoints differ, so pooled values are not comparable. | C | Normalize a preregistered severity coordinate within family and report family-specific curves before aggregation. |
| 267–273, 472–473 | Dataset-level Friedman/Wilcoxon/Holm can be appropriate, but seeds and shifted variants are dependent and must not be treated as independent `N`. Small dataset counts, ties/zeros, effect intervals, and family-level replication require explicit handling. | A | Aggregate at the preregistered independent unit and use grouped/hierarchical uncertainty. |
| 277–330 | “Core real-world clustering datasets” conflates natural grouping with supervised class/outcome recovery. The construct audit rates only 4/54 listed/generated datasets tier A, 8 tier B, and 42 tier C; only Iris is tier A among the twelve Phase-7 datasets. | D | Report label-recovery stress results and construct tiers; do not equate classes with natural clusters. |
| 332–345 | Synthetic data are valuable because their latent structure can be controlled, but “soft ground truth” is valid only when the exact generative posterior is defined and retained. | C | Preserve latent component and posterior provenance per family. |
| 349–364 | This is an aspirational 8-family × 3-severity suite, not the executed Phase-7 panel. Frozen Phase 7 used 7 families × 2 severities plus clean = 15 conditions. | D | Label planned and executed protocols separately. |
| 370–376 | The listed components total 54,100 units before ablations and about 59,100–64,100 with the stated ablation range. The rounded 60,000–65,000 claim is close but not the literal sum; more importantly, “unit” mixes executions with seeds/conditions and all values are plans, not completed evidence. | D | State the exact sum and define “unit” if the roadmap is ever revived. |
| 380 | Unlabelled random five-fold splitting omitted exact-duplicate grouping, subject grouping for HAR/Mice, temporal ordering where applicable, and target class-coverage safeguards. Frozen Phase 7 leaked duplicate feature rows across folds in Glass, Iris, Letter, and Spambase; four target folds lacked at least one oracle class. | A | Corrected Y1 splits group exact duplicates, preserve known entities/time, and enforce class safeguards without selecting based on model outcomes. |
| 381–382 | “`n_init=5` … retaining the minimal objective solution” does not describe Phase 7. Seeds 1–5 are retained as separate rows and are highly dependent within a dataset/fold/condition. | D / A | Correct seed semantics and account for dependence. |
| 386–407 | Most listed methods/adaptation strategies are future baselines, not Phase-7 evidence. Oracle adaptation is an analysis-only upper bound and cannot enter deployment selection. | E / D | Keep as a possible future benchmark only; do not imply completion. |
| 413–422 | The ablation omits a grouped mean/median no-skill predictor. Frozen LODO median MAE is `0.038008`, better than P0 (`0.046705`), P3 (`0.063926`), and P4 (`0.050895`). P4 also failed both preregistered superiority comparisons. | A | Always include no-skill baselines and absolute model-skill criteria. |
| 426–440 | Source-only fitting, label isolation, Oracle-K disclosure, domain-feature masking, and dataset-grouped meta-validation are good principles. Rule 3 is worded too broadly because the current deployment probe is intentionally sampled from target `X_t`; the prohibited operation is target-label/outcome-driven design, not all target-data use. | F / D | Clarify the target-data versus target-label distinction. |
| 430–435 | The asserted AST enforcement is narrower than a proof of semantic label isolation and refers partly to planned modules. Phase 7 did keep outcome columns out of predictors, which the independent reconstruction confirms. | F / D | Retain tests but avoid claiming an AST scan proves all forms of leakage absent. |
| 444–472 | All figures/tables are a plan. Figures 2–6 and Tables 3–7 would be misleading if populated under the current affirmative narrative; they must prominently show the failed gate/no-skill result if retained for a failure report. | E / D | Do not generate a success-shaped manuscript from negative evidence. |
| 476–493 | Phase 7 is called “verifying” and the roadmap presumes automatic Phase 8–15 continuation. That conflicts with falsification logic and the actual negative verdict. | D | Do not continue the original Phase 8–15 roadmap. |
| 497–514 | “Frozen decisions” include invalidated or untested conclusions: title precision/novelty, adaptation core, fixed `m=2`, Gaussian-only soft truth, NAR, and claimed leakage elimination. Repository/package names and the prohibition on Auto-K/neural expansion are harmless organizational choices. | C / D / F | Retain only neutral identifiers and scope constraints; redesign or retire the scientific claims. |

## Confirmed defects versus rejected suspicions

Confirmed defects are: convergence eligibility did not require convergence; exact-duplicate group leakage; four target class-coverage failures; missing no-skill baseline; unfair level-versus-change validity comparison; dependent seeds/scenarios needing grouped inference; post-label evaluator refactoring; overbroad construct/soft/future terminology; and the mathematical design defects in the proposed selector.

Rejected as explanations of the frozen negative result are: a wrong Phase-7 join, a hidden outcome column in predictors, a metric/verdict recomputation error, numerical divergence/degenerate FCM solutions in the diagnosed sample, or a simple `D_H` reference correction that rescues P4. The independent implementation reproduces the historical values and verdict.

## Draft disposition

Keep the file as historical planning evidence. Do not silently revise it into a success narrative, do not treat it as a completed manuscript, and do not execute its original Phase 8–15 sequence. Any future work would need a new preregistration built from the corrected design report and untouched data; it cannot be described as continuation of the failed Phase-7 hypothesis.
