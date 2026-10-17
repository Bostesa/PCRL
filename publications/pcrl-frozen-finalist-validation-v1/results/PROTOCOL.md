# Protocol: PCRL frozen-finalist validation and confirmation-readiness study

Registered before any new reader fit. Campaign start 2026-10-17T21:03:32Z. Branch `research/pcrl-frozen-finalist-validation-v1`, created from the source tip `9e2b6c567`.

## 0. What this is

**Data.** Statistical machine-learning research on public-use U.S. Census Bureau American Community Survey (ACS) 2019 1-Year PUMS person records for Florida, in the existing development roles of the protected-tree replication.
- Cohort: housing-unit residents aged 25–64, civilian employed (ESR 1 or 2), WKHP 1–99.

**The service.** From six permitted raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP) it predicts:
- an income class: 2 classes, PINCP × ADJINC / 10^6 above $50,000;
- an occupation group: 5 classes, OCCP mapped to the five civilian broad groups.

Each recipient receives only its own float64 probability vector and argmax decision.

**What is measured.** SEX is the dataset's recorded demographic column. Separate evaluation classifiers ("readers") try to infer SEX from the released predictions: each output alone, and the two outputs aligned to the same record (the linked pair). Reader recovery (ROC AUC for SEX) measures how much the releases expose to these readers. It is not a privacy guarantee.

**Status of the evidence.** This is development robustness evidence on already-used records:
- DEV_REPORT was evaluated once by the source campaign, and its results are published.
- This campaign re-reads it once, in a new logged follow-up event; that is not fresh confirmation.
- Numerical reproduction does not erase the source's adaptive selection history.
- The sealed assessment stays closed. No new population, download, release retraining, release reselection or retuning is authorized or done.

**Platform event, disclosed.** A platform safety classifier stopped the coordinator's first response to this brief before any of its tool calls ran. The campaign restarted from state inspection at 21:03:32Z. No variable or task description has been renamed or obscured; SEX, income and occupation keep their actual names throughout.

## 1. Frozen candidate and compact bank

Everything is admitted by hash (`SOURCE_ADMISSION.json`: 414 copies, no problems) and frozen in `FROZEN_CANDIDATE_MANIFEST.json`.

**The candidate: `B3CTL_UNCOUPLED_MIX50_C6A`.**
- **Occupation output.** The stored JOINT_TREE output, bitwise identical to H_LJ's occupation output.
- **Income statistic.** d = 0.5 × LOCAL_TREE + 0.5 × JOINT_TREE E20 income log-odds.
- **Income code.** level = `searchsorted(thresholds, d, side="left")`, using the frozen label-free MODEL_FIT thresholds (3 levels on each side of 0).
- **Income probability.** The CALIBRATION-fitted per-level value, released with its argmax decision.

No parameter, threshold, calibration value, rounding, routing or output is changed.

**Bank: six effective releases per seed**, with source roles and source MODEL_SELECT eligibility labels (source `ANALYSIS_LOCK.json`):

| Release | Role here | Source eligibility |
|---|---|---|
| FINALIST | frozen candidate | — |
| NEURAL_RAW | eligible reference | eligible |
| H_LJ | eligible reference; same occupation output | eligible |
| TASKONLY | matched coarse task-only control (source claim M, step 1) | present (TO needs only presence) |
| B2TWIN | LOCAL-only coarse twin (source claim J) | eligible |
| PUB_IBSI | diagnostic comparator | **ineligible**: cannot be counted as beaten |

Bank details:
- **Exact aliasing.** FINALIST, H_LJ and B2TWIN share one occupation wire on every role and seed (identical hashes). Their occupation-only readers are therefore one fitted bank, with identical predictions.
- **Unique single-recipient wires per seed:** 6 income and 4 occupation.
- **NEURAL_RAW seed 1.** Its output is admitted by hash; the missing raw-input package is a deployment coverage gap.

## 2. Responsibilities and resources

| Responsibility | Who |
|---|---|
| Coordination and implementation | the main session |
| Output-mapping and exposure review, including the population inventory | a separate agent |
| Statistical and readiness review | a separate agent; reviews this protocol and the reader lock before any fit, and the feasibility calculation after |
| Independent numerical verification | a separate agent with its own code; one separately logged DEV_REPORT replay |
| Deployment and claims review | a separate agent |

**Budget.**
- 6 elapsed hours and 10 CPU-hours.
- At most two heavy jobs at once, one numerical thread per job.
- Zero paid or cloud compute.

**Queue.**
- Every numerical unit runs through `ffv.jobqueue`. Failed attempts, reader fits, refits and verification all count, and agent work is charged at upper estimates.
- The last ~90 minutes are reserved for verification and closeout.

## 3. Role access

**ATTACK_FIT and ATTACK_VALIDATION.**
- ATTACK_FIT (8,001 people; features and SEX) is used to fit readers and their label-free transformations.
- ATTACK_VALIDATION (3,000 people; features and SEX) is used for reader orientation and selection only.
- Every SEX read is logged.

**DEV_REPORT** (5,943 people in 3,955 households) is read without labels until the freeze. Its truth is re-read only through `ffv.common.dev_report_truth`, under two new events:
- `DEV_REPORT_TRUTH_REREAD_BY_FFV_COORDINATOR`: one logged coordinator evaluation, after this campaign's committed and published REPORTING_LOCK;
- `DEV_REPORT_TRUTH_REREAD_BY_FFV_INDEPENDENT_VERIFIER`: one separately logged verifier replay.

Both events are appended to this campaign's access log and to the lineage-wide replication log. The source campaign's logs and one-use counters are preserved, not reset or bypassed.

**MODEL_FIT, CALIBRATION and MODEL_SELECT.** Only features, MODEL_FIT raw inputs and household ids are admitted, for the output-mapping and exposure checks. No task label or SEX column of these roles is admitted or read.

**SEALED_ASSESSMENT.** Refused by every loader.

## 4. Exposure and selection audit

`EXPOSURE_AND_SELECTION_AUDIT.md` gives a component ledger:
- which records and labels fitted each teacher and each privacy correction;
- which records set the income thresholds;
- which income labels fitted the calibration;
- which roles selected the mixture, the number of levels and the finalist;
- which source outputs and DEV_REPORT results were already known;
- which claim definitions were finalized after promotion, including the source PROTOCOL §9 timing deviation (claims written after selection, before final reader fitting and reporting).

**Household overlap.** It is checked from the admitted household ids for every pair of admitted roles. Overlap with the sealed assessment is not checkable here, because its ids are not opened. It is reported as UNKNOWN except for what the role-assignment code shows. Missing evidence is UNKNOWN, not proof of independence.

## 5. Output-mapping check (label-free; before any reader fit)

`ffv.mapping` reports, for each admitted role and seed and for the deduplicated union of the six roles, using the exact transmitted float64 bytes (no rounding):
- records, distinct permitted-input vectors and distinct encoded-feature vectors;
- distinct occupation vectors, distinct income vectors and distinct linked pairs;
- occupation-vector group sizes and the singleton fraction;
- whether one identical occupation vector ever accompanies different income codes, within a role and across roles;
- whether the same input → output mapping holds across roles.

**Domain-wide structure** (kept separate from the empirical counts):
- From the frozen oblivious-tree split borders, the exact number of routing cells over the valid input domain (AGEP 25–64, WKHP 1–99, and each categorical's encoder levels plus its OTHER column) for the occupation teacher, the income teacher and their common refinement.
- A bounded search over valid invented inputs for (i) two distinct inputs with bitwise-identical occupation vectors, and (ii) two inputs with bitwise-identical occupation vectors but different income codes.
- Every reported counterexample is confirmed through the frozen deployment package.
- Failure to find one is not a proof.

**Last-bit batch dependence.** The package arithmetic can yield vectors differing by about 1e-16 depending on request batch size (`FROZEN_CANDIDATE_MANIFEST.json`, `small_batch_last_bit_observation`). This is characterized, not repaired. Exact-byte counts are reported for the evaluated batch outputs.

**Pair-trained reconstruction with occupation-only evaluation input** (diagnostic; pair access only):
- An HGB classifier with the reader HGB15 settings, fitted on ATTACK_FIT linked pairs, predicts the finalist's income level from its occupation output.
- Its level accuracy is reported on ATTACK_VALIDATION and DEV_REPORT outputs. No SEX label is used.
- It is not an ordinary occupation-only adversary.

No population-optimal claim is made from any empirical lookup table.

## 6. Adversary access (unchanged from the source)

- A single-recipient reader receives only that recipient's released output, plus the granted demographic-labelled auxiliary sample (ATTACK_FIT, 8,001 people; selection on ATTACK_VALIDATION, 3,000).
- A pair reader receives the two aligned released outputs and the same pair-access auxiliary sample.

No reader receives raw Census inputs, hidden scores, teacher outputs, parameters, the finalist's internal mixed income score, MODEL_* labels or DEV_REPORT labels. Label-free transformations of received probabilities are fitted on ATTACK_FIT only. Each person receives one release under the existing service.

## 7. Enhanced reader bank (frozen in `READER_BANK_LOCK.json`)

### 7.1 Source bank (retained, not refitted)

The source bank keeps its 8-reader slate:
- OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB;
- the corrected exact-vector LOOKUP reader.

It is retained with its fitted models, its stored DEV_REPORT scores (admitted by hash under the source REPORTING_LOCK), its recorded ATTACK_VALIDATION AUCs and its original winners.

A predefined subset is refitted with the source code and compared bitwise with the stored scores: per seed, the source pair winners of FINALIST, NEURAL_RAW and H_LJ, plus the FINALIST occupation winner.

### 7.2 New reader recipes

Each recipe is applied to every unique single-recipient wire and every linked pair, per seed. The model seed is `MODEL_SEED_BASE + s` for every recipe, and every recipe has the same fixed effort.

| Id | Model |
|---|---|
| ET5 / ET25 | ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5 / 25, max_features=1.0, n_jobs=1) |
| HGB15 / HGB31 | HistGradientBoostingClassifier(max_leaf_nodes=15 / 31, max_iter=200, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, early_stopping=False) |
| MLP64 / MLP6432 | MLPClassifier(hidden_layer_sizes=(64,) / (64, 32), alpha=0.01, batch_size=256, max_iter=300, early_stopping=False); other parameters at sklearn defaults (adam, relu) |
| CATI01 / CATI1 | Regularized categorical-interaction logistic readers (§7.3); L2 LogisticRegression(C=0.1 / 1.0, lbfgs, max_iter=5000) |

**Features for ET, HGB and MLP**, per received output with k classes:
- the probabilities p (k);
- log max(p, 1e-300) (k), using the source clipping convention;
- the one-hot argmax decision (k).

A pair concatenates the income and occupation features, in that order. Every column is standardized with the source FiniteScaler, fitted on ATTACK_FIT only; this is applied to all recipes, and trees are invariant to it.

No early stopping and no internal validation split is used anywhere.

### 7.3 Categorical-interaction readers (CATI)

**Income representation.** If the received income vector takes at most 32 distinct exact values on ATTACK_FIT, it is categorical: one indicator per distinct ATTACK_FIT vector, ordered by P(income = 1). An unseen vector at scoring time gets all indicators zero. Otherwise it is grouped into six label-free quantile groups of its own received P(income = 1) (ATTACK_FIT quantiles 1/6, …, 5/6; ties go to the lower group), and the continuous features p and log p are kept as well.

**Occupation features.** The 5 probabilities and the 4 log-contrasts log(p_j / p_0), j = 1–4, with the source clipping.

**Pair CATI design:**
- income indicators (or group indicators plus the continuous income features);
- occupation features;
- every income indicator × every occupation feature.

**Single-view CATI designs** (fixed construction; subject to the reviewer's agreement before fitting):
- **Income view.** Categorical case: the indicators alone. Otherwise: the group indicators, p, log p, and a cubic B-spline basis of logit P(income = 1) (SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="constant"), fitted on ATTACK_FIT).
- **Occupation view.**
  - The 9 occupation features.
  - A cubic B-spline basis of each log-contrast: SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="constant"), fitted on ATTACK_FIT.
  - The 6 pairwise products of the 4 log-contrasts.
  - The one-hot occupation decision.

Smoothing comes only from the L2 penalty; there is no target encoding. Every design is standardized with FiniteScaler on ATTACK_FIT.

### 7.4 Candidate sets, orientation and selection (all on ATTACK_VALIDATION; never DEV_REPORT)

- **Orientation** follows the source fitting convention: the score is the reader's predicted probability of the SEX value coded 1. Its sign is fixed by the source rule (+1 if the ATTACK_VALIDATION AUC ≥ 0.5, else −1) before any DEV_REPORT read.
- **Original winner (retained).** The source's selected winner per release, view and seed, over the source slate. For pairs this is 8 pair readers plus 16 ignore-one alternatives. Its DEV_REPORT score is the admitted stored score.
- **Enhanced winner.** The candidate with the maximum oriented ATTACK_VALIDATION AUC, in a source-first order (adopted from the statistical review, item A1):
  - a single view has 16 candidates: the 8 source readers in source order, then the 8 new recipes in the order of §7.2;
  - a pair has 48 candidates: the 24 source candidates in source order (8 source pair readers, 8 source IGNORE_occupation alternatives, 8 source IGNORE_income alternatives), then 8 new pair readers, 8 new IGNORE_occupation and 8 new IGNORE_income alternatives.
- **Source candidate values.** Source candidates enter with their stored oriented ATTACK_VALIDATION AUCs and are scored with their stored DEV_REPORT scores. A reproduction refit that is not bitwise is a reported discrepancy, never a substitute.
- **Ties** are exact float equality and go to the earlier candidate. The enhanced winner therefore equals the original winner unless a new candidate is strictly higher on ATTACK_VALIDATION.
- **Near ties.** Each winner's ATTACK_VALIDATION gap to the runner-up and to the original winner is reported. A gap below 0.010 (about one AUC SE on 3,000 people) is flagged `AV_NEAR_TIE`.
- Both winners' DEV_REPORT results are reported, frozen. Neither is chosen after evaluation.
- No architecture or setting is added after DEV_REPORT is read.

### 7.5 Controls

**Source definitions,** recomputed for the new pipeline on the FINALIST wire, per view and seed:
- a shuffled-SEX null: a fixed permutation of ATTACK_FIT SEX, seed 2026101602 + s;
- a planted SEX bit: the reader is fitted on the view's features plus the SEX column. On DEV_REPORT the bit is scored as both 0 and 1, and is combined with truth only inside the logged evaluation.

These run for OUT_HGB (as in the source) and for the new recipes ET5, HGB31, MLP6432 and CATI1.

**Validity bands** (source): planted AUC ≥ 0.95; |shuffled AUC − 0.5| ≤ 0.05, on ATTACK_VALIDATION and DEV_REPORT.

**The CV conjunct.**
- **ORIGINAL scorecard:** the source run's stored OUT_HGB controls, as in the source.
- **ENHANCED scorecard:** all five control recipes (OUT_HGB, ET5, HGB31, MLP6432, CATI1) × 3 views × 2 seeds must sit inside the bands on both ATTACK_VALIDATION and DEV_REPORT (statistical review, item A5).
- The recomputed OUT_HGB controls are also compared bitwise with the source's stored controls.

A coordinator draft had limited the enhanced conjunct to OUT_HGB. The concern was that more band checks raise the chance of a false control failure; in the invented-data dry run, CATI1 read 0.538 on the occupation view, inside the band but near its edge. The reviewer required all five recipes, and that stricter rule is adopted.

**No retuning.** No recipe is dropped, replaced or retuned after any control is seen. An ATTACK_VALIDATION control outside its band is logged as a deviation before the reporting lock, and any repair applies to every release and view. Failed attempts are preserved.

### 7.6 Pre-registered reader-seed variation diagnostic (never used for selection)

- **Scope.** Per seed: the enhanced pair winners of FINALIST, NEURAL_RAW and H_LJ.
- **Random-seed component.** Seed-dependent winners (ET, MLP) are refitted with model seeds MODEL_SEED_BASE + s + 1000·j, for j = 1–4. Seed-invariant winners are labelled "seed-invariant" rather than "measured zero variance": HGB here has no early stopping and no subsampling, and LR, CATI and LOOKUP have no random component (review item A4).
- **Auxiliary-sample component** (review option O5, adopted). Each winner except LOOKUP is refitted on 4 household-bootstrap resamples of ATTACK_FIT (seed 2026101805 + 10·s + j), with the same model seed.
- **What is measured.** The spread of their DEV_REPORT AUCs gives a partial, lower-bound reader-refit component for §10.
- **Not measured.** Release-training variation; the release is frozen and only two model seeds exist.

## 8. Endpoints and claims (computed in one logged evaluation)

**Inference** is the source household-aware paired inference (source `ANALYSIS_LOCK.json`, unchanged):
- household bootstrap over DEV_REPORT households: `task_headroom.inference.group_weights`, B = 2048, seed 2026101705, G = 3955;
- the weights are shared across releases, endpoints, both reader banks and both seeds, which share people;
- paired differences are taken within each replicate;
- Wald bounds are point ± t(3954) × bootstrap SE;
- critical values are unchanged: clause c1 = 2.3949991 (one-sided α = 0.025/3); family c_P = 3.304031 (K_P = 52) and c_S = 3.025281 (K_S = 20); marginal 1.9605641.

Multiplicity handling is frozen as the source's: each of B, M and J keeps its α/3 clause criterion, and the 0.020 test spends no extra α. All other results are descriptive, with marginal 95% intervals.

**Two scorecards, evaluated in the same run:**
- **ORIGINAL:** the source endpoint roster, conjunctions and labels (source `ANALYSIS_LOCK.json`, endpoints restricted to this bank), using the original winners. It must reproduce the source DEV_REPORT_RESULTS.json values (claims E, B, M, J and the 0.020 nested test); any difference is a discrepancy to report.
- **ENHANCED:** the same roster, conjunctions, margins, signs, critical values and labels, with every reader-dependent endpoint recomputed from the enhanced winners:
  - pair benefits against NEURAL_RAW and H_LJ (claim B), TASKONLY (M, step 1) and B2TWIN (J);
  - the 8 recipient non-worsening clauses of B and the recipient clauses of M and J;
  - the descriptive pair-minus-view keys.

  The utility (24) and subgroup-guard (4) endpoints are reader-free and identical in both. CV for ENHANCED is the new pipeline's control set (§7.5).

**Signs.** benefit = AUC(comparator) − AUC(FINALIST), positive favouring the candidate; recipient worsening = AUC(FINALIST view) − AUC(comparator view) ≤ 0.010; material target = benefit clause bound > 0.020.

**Eligibility.** NEURAL_RAW and H_LJ are the eligible references. PUB_IBSI is source-ineligible: its difference is descriptive only and cannot be counted as beaten.

**Paired reader-class changes** (descriptive; marginal 95%), for every release, view and seed: AUC(enhanced winner) − AUC(original winner). The change in each benefit is reported as well.

**ENHANCED E** (review item B2) is re-evaluated: its 8 recipient point gates use the ENHANCED single-view winners. Inside the ENHANCED scorecard, the `claim E` condition of B, M and J refers to ENHANCED E. IDENTICAL_RELEASE occupation clauses stay exact zeros, because the winner is shared by construction.

**Interpretation rule** (frozen; applied to B, the nested 0.020 test, each M step and the M label, and J; review item B1):

| ORIGINAL | ENHANCED | Label |
|---|---|---|
| holds | holds | `SURVIVED_THIS_READER_BANK`: it survived this particular larger bank, not arbitrary readers |
| holds | fails (CV valid) | `READER_SENSITIVE`, with the first failing condition in source condition order and a side attribution (below). This is a substantive finding, not automatically a software defect. |
| fails | holds | `ENHANCED_ONLY_PASS`, descriptive only. The source label stands: a second look at the same people cannot rescue J or the 0.020 target. |
| fails | fails | `NOT_ESTABLISHED_EITHER_BANK`; the source label stands |
| any | ENHANCED CV fails | `ENHANCED_NOT_INTERPRETABLE` (the ENHANCED claim label is CONTROL_FAILURE) |
| does not reproduce the source | any | `DISCREPANCY`; interpretation is suspended for that claim until resolved |

**Side attribution** for READER_SENSITIVE, from the paired reader-class changes of the failing benefit:
- `FINALIST_READ_HIGHER`: the FINALIST enhanced winner reads higher than its original winner;
- `COMPARATOR_READ_LOWER`: the comparator enhanced winner reads lower than its original;
- `BOTH`.

COMPARATOR_READ_LOWER is described as ATTACK_VALIDATION selection instability, not as added finalist exposure.

**Reader-matched descriptive benefits** (review option O3, adopted). For each of the 16 pair recipes, the same recipe is applied to FINALIST and to each comparator, giving comparator AUC − FINALIST AUC. These are DEV_REPORT-descriptive only and are never used for selection. They separate the effect of the reader class from selection noise.

The utility, subgroup and recipient requirements are reported in full for both scorecards.

**Stop rule.** The release is not revised and no reader search is started after the report.

## 9. Freeze before the DEV_REPORT re-read

**What the REPORTING_LOCK fixes:**
- the fitted readers (model-file hashes);
- the selected winners;
- AV and DEV scores (no labels);
- the bank, the report code hashes and the endpoint roster.

**Order of steps:**
1. The lock is committed.
2. A sanitized reporting receipt is published through the established workflow.
3. The coordinator runs the one logged evaluation.
4. The verifier runs a separately logged replay.

## 10. Confirmation feasibility (after the evaluation; from existing evidence only)

**Data.** The evaluation also writes, privately, the full-precision per-replicate bootstrap draws of every endpoint. `ffv.feasibility` then works from those.

**Method.**
- The joint covariance of all endpoint estimators comes from the 2048 shared household-bootstrap replicates, scaled to n independent households: Σ(n) = Σ_DEV × 3955 / n.
- Estimates are approximated as multivariate normal; this is justified by smooth functionals of household means and U-statistics.
- Joint power for a prospective study with the same scorecard is P(all required conditions hold), by Monte Carlo (fixed seed).
- Required households for 80% and 90% joint power are found by search over n.

**Estimand** (review item F6). The primary estimand is the frozen releases with the frozen fitted readers and frozen winners, applied to new independent households from the same population. A confirmation is then "against these fitted readers". If readers were re-selected prospectively, the endpoint would be a maximum over candidates, which is not smooth; the MVN approximation does not cover that case.

**Success definitions** (CV and availability are assumed to pass):

| Definition | Requires |
|---|---|
| D1, source scorecard | the 24 utility and 4 guard point gates, claim B's 8 recipient clause bounds (U1 ≤ 0.010; the 2 IDENTICAL_RELEASE clauses pass deterministically), and the 4 claim-B pair-benefit clause bounds > 0 |
| D2, confirmatory | D1, but every utility and guard condition needs a one-sided upper bound (c1) ≤ cap |
| D3 | D2 plus the 4 benefit clause bounds > 0.020 |

**Computation** (review items F3–F5):
- The grid of household counts is log-spaced from 3,955 to 10^8. Each grid point uses 40,000 Monte Carlo draws with fixed seed 2026101803; the MC SE is ≤ 0.0025. The full power curve and the adjacent-grid bracket are reported.
- "Not reached by 10^8" is kept separate from "the n → ∞ limit is below target". The analytic limit is reported for each definition.
- Σ is factored by eigendecomposition, with negative eigenvalues clipped to zero. Zero-variance (IDENTICAL_RELEASE) endpoints are deterministic. The perfectly collinear U0/Ucal accuracy keys are kept, and move together.
- Both scorecards are run. The less favourable one governs the §13 decision.

**Scenarios** (the 75% shrinkage scenario is presented first, because the plug-in benefits come from the favourably selected result that motivated this follow-up):
1. Observed development effects and margins (plug-in).
2. Recovery improvements shrunk to 75% and 50% of observed.
3. Less utility headroom. The binding clauses, named by key (review item F2), are `UTILITY|income|Ucal|brier_excess|s0`, `UTILITY|income|Ucal|brier_excess|s1`, and `UTILITY|income|U0|accuracy_loss|s1` with its identical Ucal twin. They are moved halfway to their caps and fully to their caps. At the cap, the per-clause limits are 0.5 (D1) and Φ(−c1) = 0.0083 (D2), reported as unreachable.
4. An added non-shrinking AUC variance component, on AUC endpoints only:
   - measured: §7.6 reader refit, a lower bound;
   - hypothetical: release retraining, |benefit(s0) − benefit(s1)| / √2. That has 1 degree of freedom and is confounded with the reader seed and with same-people noise, so it is labelled hypothetical.
5. An assumed-true-slack grid for the binding utility clauses, s ∈ {−0.001, 0, plug-in, 0.0005, 0.001, 0.002}, with the other endpoints at plug-in. The plug-in slack is 0.03–0.11 SE, so the data do not show that the true slack is positive.
6. **Assurance:** power averaged over μ ~ N(μ̂, Σ_DEV), with the same Monte Carlo seed.

**Sensitivity:** a B-only prospective intersection-union test at one-sided α = 0.025 (c = 1.960). It is shown, never registered.

Binding clauses, the uncertainty in each plug-in, and the transport assumptions are reported:
- households iid from the same population;
- an unweighted analysis;
- no transport to another state or year.

A plug-in effect below a target is described as not reliably establishable at that target as n grows, not as impossible under every true effect. The statistical reviewer and the verifier check the calculation independently.

## 11. Population inventory (read-only)

`POPULATION_INVENTORY.md` uses only existing usage logs, reservations, manifests, campaign records and authoritative metadata. There are no downloads, no new person-level data, no new labels and no sealed-assessment access.

For each candidate state-year it gives:
- its status: KNOWN_USED, RESERVED, NO_USE_FOUND_IN_SEARCHED_RECORDS or EXPOSURE_UNKNOWN;
- the historical sealed-label extraction disclosure, where relevant;
- documented sample availability (unknown counts stay unknown);
- whether it would test the frozen release, transport a recipe, or require a different benchmark.

No population is chosen from task or recovery outcomes. Opening any population needs a separate authorization and a frozen protocol.

## 12. Independent verification and deployment

**Refit roster (frozen now).** Per seed:
- the ET5, HGB31, MLP6432 and CATI1 pair readers of FINALIST, NEURAL_RAW and H_LJ;
- the enhanced pair winners of those three releases;
- the source pair winners of those three (bitwise against the stored scores).

**The verifier separately recomputes:**
- source hashes, release identity and the exposure ledger;
- the exact-vector mapping diagnostics;
- reader feature-access restrictions;
- the roster fits;
- the selected readers and primary endpoints, via its own logged replay;
- utility and subgroup headroom;
- the power and sample-size calculation;
- reporting labels and limitations.

**Deployment.** Raw six-field deployment is rerun from the admitted source packages and from the source backup alone. Thresholds and calibration stay unchanged, including the documented threshold-tie behaviour. The following are characterized, not repaired:
- the threshold-tie floating-point boundary (source D3);
- the batch-size last-bit dependence (§5).

The package APIs and the documented wrappers are tested separately, and the report states where strict schema rejection comes from a wrapper rather than the underlying source function.

**Custody.** Same-device restoration is done, plus off-device custody if a drive is available. No unmount is forced, and a warm read-back is not described as a cold check.

## 13. Decision options (exactly one is recommended at the end)

1. Preserve the frozen prototype and prepare a separately authorized prospective study.
2. Preserve it as a reader-sensitive engineering result without confirmation yet.
3. Retire it as a competitive candidate, with the evidence and runnable artifact retained.
4. Mark validation incomplete, with exact missing units and no scientific overstatement.

A fresh population is not treated as a cure for inadequate power, weak readers or an unsupported novelty claim.

## 14. Registered forecasts (coordinator; before any fit)

| Event | Probability |
|---|---|
| ENHANCED claim B holds (clause bound > 0 against NEURAL_RAW and H_LJ on both seeds, with E) | 0.45 |
| ENHANCED 0.020 nested target established | 0.03 |
| ENHANCED claim M step 1 holds (against TASKONLY) | 0.80 |
| ENHANCED claim J holds | 0.20 |
| The enhanced FINALIST pair winner reads higher on DEV_REPORT than the original winner, on both seeds | 0.55 |
| A domain-wide counterexample is found (bitwise-identical occupation vector, different income code) | 0.90 |
| Distinct valid inputs with bitwise-identical occupation vectors exist in the domain | 0.99 |
| Under the plug-in scenario, D2 reaches 80% joint power with at most 100,000 independent households | 0.10 |
| The ORIGINAL scorecard reproduces every source claim label exactly | 0.97 |

Each forecast is scored with the Brier score at closeout.

## 15. Deviations

Any deviation from this protocol is recorded with its time and reason in `PROTOCOL_DEVIATIONS.md` before the stage it affects, or, if discovered later, disclosed as post hoc.

## 16. Statistical review record (before any reader fit)

The review is in `roles/statistical_reviewer/PROTOCOL_REVIEW.md`. Its decision on the §7.3 construction was **AGREED_WITH_CHANGES**.

**Adopted before the reader lock:**
- **A1** source-first candidate order (§7.4);
- **A2** estimators built directly, with full `get_params()` and library versions locked and asserted;
- **A3 / C1–C6** CATI details:
  - the formulas;
  - explicit unique-quantile spline knots, with a block omitted below 2 knots;
  - the quantile groups;
  - categorical income, with unseen counts reported and every wire's regime and fitted state recorded in the lock;
  - a fixed column order;
  - convergence recorded and flagged, never refitted;
- **A4** seed-dependent versus seed-invariant refits (§7.6);
- **A5** the ENHANCED CV over all five control recipes, with no retuning after controls (§7.5).

**Adopted before the reporting lock:**
- **B1** the complete interpretation table and side attribution (§8);
- **B2** ENHANCED E;
- **B3** constants unchanged;
- **B4** the frozen feasibility specification (§10);
- **B5** lock contents.

**Optional items:**
- Adopted: O3 reader-matched descriptive benefits, O4 near-tie flags, O5 auxiliary-sample refits.
- Declined, with reasons recorded in `READER_BANK_LOCK.json`: O1, O2 and O6. O2's asymmetry is conservative for claim B.
