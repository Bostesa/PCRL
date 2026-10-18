# Selection rules of the overnight occupation-headroom campaign (statistical and baseline reviewer)

**Status.** Written 2026-10-18 04:48-05:30Z, before any model outcome of this campaign existed, for the registration commit.
Machine-readable form: `RULES.json` (schema `OHN1-rules-v1`). Implementation: `ohn/rules.py` (pure functions; `load_rules()`
refuses a rule file whose constants disagree with the code). Invented-data tests: `ohn/test_rules.py`, units `OHN_RULES_TEST_V1`
(23 of 23 PASS) and `OHN_RULES_TEST_V2` (24 of 24 PASS, rerun after the final edit of `RULES.json` and the added cross-fitting
flag; its stdout records the rule file's SHA-256, e18f1c68f4646ce9d1395960ab1a364a80c12541b815a55e0ba06b65074cbbb3). Claims: `CLAIMS_DRAFT.md`. The only inputs used to write these rules are the source campaign's published
records (`results/pcrl_income_headroom_successor_v1/`: REFERENCE_MS_TABLE.csv, RESERVE_RULE.json, NOMINATION.json,
EXPERIMENT_LEDGER.csv, report/PRIMARY_ENDPOINTS.csv, RESEARCH_DECISION.md).

**Data context.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The
service predicts an income class (2 classes) and an occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP. SEX is
the dataset's recorded demographic column (sex0 = men, sex1 = women); readers estimate it from the released probability vectors.

**What every rule here is.** A development selection heuristic applied to MODEL_SELECT, a role this campaign and its
predecessors use adaptively. **No rule changes a cap, and no rule proves a population guarantee.** A configuration that fails a
screen still passes or fails the original conditions at the point exactly as before.

## 0. Original limits (unchanged)

| Condition | Limit | Count |
|---|---|---|
| Accuracy loss, NLL excess (nats), summed multiclass Brier excess, against U0 and Ucal, income and occupation | 0.010, 0.010, 0.005 | 12 per seed, 24 in all |
| Within-SEX occupation summed-Brier excess against U0, men and women | 0.005 each | 2 per seed, 4 in all |
| Recipient non-worsening: each single output's SEX-recovery AUC minus the comparator's (income view, occupation view) | 0.010 | per comparator, view and seed |
| Material improvement of the linked-pair benefit | 0.020 AUC | reported separately from positive superiority |

## 1. Whole-release development reserve (R1)

| Item | Rule |
|---|---|
| Released object | The **complete release**: income output (the H3 anchor, or an income-branch output) + occupation output, scored together |
| Conditions | All **28**: 12 income clauses, 12 occupation clauses and 4 within-SEX guards (both seeds) |
| Condition | margin = cap - value **>= 2 x se_used** (inclusive). A non-finite value or SE fails |
| SE | **se_used = max(own SE, floor SE)**; own SE = the household-clustered paired SE on MODEL_SELECT (guards: linearized clustered SE), exactly as `ihs.evaluate.utility`; floor = **H_LJ's MODEL_SELECT SE of the same condition and seed** (the source's registered conservative rule, values copied unchanged), rescaled by sqrt(5306 / households) |
| Zero or tiny own SE | Uses the floor (always > 0); recorded as `own_se_zero` / `se_source = floor`. A missing, zero or non-finite floor and a negative SE are refused as errors |
| No escape | Unlike the source, a changed occupation output gets **no non-worsening alternative**: every occupation condition must have its own reserve |
| Status | `RESERVE_MET` only if both seeds meet all 28. `RESERVE_ONE_SEED_ONLY`, `RESERVE_NOT_MET`, `RESERVE_PENDING_SECOND_SEED` otherwise (none of them meets the reserve) |
| Ranking quantity | The ascending vector of the 28 ratios margin / se_used, compared **leximin** (larger minimum first; an exact tie, which happens when the same anchor condition binds, passes to the next-smallest ratio) |

**Why.**
- *Whole release.* The source tie-break used only the 12 income ratios and nominated H3 while the inherited occupation output's
  men's guard sat 0.32 / 0.62 SE from its cap. Ranking on all 28 conditions makes the weakest condition of either output decide.
- *k = 2 at the MODEL_SELECT scale.* The source's derivation (`UTILITY_RESERVE_RULE.md` section 2.3) is unchanged: k = 2 leaves
  1.369 SE for a one-sided 5% bound with 80% power at the about 17,510 same-population households that exist, plus 0.631 SE for
  development noise and selection. It is a heuristic, not a correction: the source's H3 met it on MODEL_SELECT (seed-1 income
  accuracy 2.86 SE) and still eroded on DEV_REPORT (0.83 DEV SE).
- *Floor = H_LJ's SE.* Own SEs mislead for outputs close to a reference (they shrink towards 0). H_LJ's occupation SEs exceed
  those of the equally exposed task-only learners (T_TREE's by about 15-20%), so the floor is conservative for a stronger tree
  occupation learner and decisive only for near-reference outputs. A replacement would need a reason; none was found.
- *Scale check (source numbers, no new outcome).* H3's income ratios are shared by every H3-anchored release (income clauses do
  not depend on the occupation output): minimum 2.862 (seed-1 accuracy), so an H3-anchored release's whole-release minimum is
  at most 2.862. The inherited JOINT_TREE occupation fails R1 (men's guard 0.32 / 0.62); NEURAL_RAW's occupation output fails by
  a hair (men's guard seed 0: 1.95); the equally exposed task-only occupation outputs pass (T_TREE minimum 2.60, seed-1 men's
  guard; TASK_REFIT about 3.2). The bar is attainable by an ordinary strong occupation learner and discriminates.

## 2. Stage 1: task-only occupation witnesses (utility evidence alone)

**Candidates.** Occupation configurations whose training never used SEX (kind `task_model`, or a control declared task-only),
each combined with the H3 income anchor and evaluated as a complete release.

**Labels (first applicable):** `NOT_TASK_ONLY`, `NOT_CONSIDERED_ONE_SEED_ONLY`, `OCCUPATION_POINT_INELIGIBLE` (an occupation
clause or guard fails at the point), `RELEASE_POINT_INELIGIBLE` (an income clause fails at the point), `OCCUPATION_RESERVE_NOT_MET`
(any of the 16 occupation-side conditions below 2 floored SEs), **`PROMISING_WITNESS`**.

**Selection.** Up to **two** `SELECTED_WITNESS`, ranked by (1) the leximin vector of the 16 occupation-side ratios, (2) fewer
fitted components (n_model_fits + n_calibrator_fits, both seeds), (3) the earliest ledger entry. Others are
`PROMISING_NOT_SELECTED` or `NOT_A_WITNESS`.

**Why.** "Genuine reserve on the occupation constraints" is read as R1 on every occupation-side condition. Because eligibility is
a conjunction and ranking starts from the minimum, **a stronger aggregate NLL cannot compensate for a failed subgroup guard**
(test `t_witness_guard_cannot_be_compensated`). All 28 constraints are reported for every learner, whichever binds.

**Measured and reported for every promising learner (never gates):** accuracy, NLL and summed Brier against U0 and Ucal; men's
and women's occupation Brier excess with SEs; differences against the equally exposed task-only controls T_TREE and TASK_REFIT
(and their occupation-side minimum ratio under the same rule: 2.60 and about 3.2 from the published values, the yardstick a
witness should beat to be called stronger); cross-fitted MODEL_FIT and MODEL_SELECT utility; deployed minus fold-average
MODEL_SELECT occupation NLL and Brier, flagged `CROSSFIT_DEPLOYED_GAP` when its absolute value exceeds that clause's se_used.

**Meaning.** `PROMISING_WITNESS` / `SELECTED_WITNESS` is the label **"useful task-only component"** on development data. It
says nothing about SEX recovery.

## 3. Conditional income branch (at most 8 seed-specific configurations)

**Trigger, evaluated once at stage-1 close (the witness selection record), before any privacy-stage or branch fit:**

`INCOME_BRANCH_TRIGGER = PROMISING_WITNESS exists ∧ (leg A ∨ leg B)`, with trigger clauses `income|U0|accuracy_loss` and
`income|Ucal|accuracy_loss`:
- **Leg A (MODEL_SELECT).** The R1 reserve of (H3 income + top-ranked witness occupation) is not `RESERVE_MET`, and every failing
  condition is a trigger clause (either seed).
- **Leg B (published DEV_REPORT values of the frozen anchor; no new read).** Some trigger clause fails the registered
  final-stage replication rule: margin_DEV >= 1.7267108 x max(SE_DEV(anchor), SE_DEV(H_LJ)) (the source's R2 rule).

**Known before any fit (stated so that nobody reads it as a finding):** leg A is FALSE for every H3-anchored release (published
MODEL_SELECT seed-1 accuracy ratio 2.862 >= 2, independent of the occupation output). Leg B is TRUE: H3's seed-1 income accuracy
loss on DEV_REPORT is 0.0070671, margin 0.0029329 = **0.83** x se_used 0.0035353 < 1.7267. Since the income clauses on DEV_REPORT
depend only on the income output, every H3-anchored finalist would carry exactly this eroded clause, so income accuracy does
prevent a complete H3-anchored candidate from meeting the registered whole-release reserve at its final-stage replication. The
trigger therefore reduces to "a promising witness exists". **A branch opened through leg B is DEV-guided** (its motivation is a
DEV_REPORT result of the source); its outputs are selected on MODEL_SELECT only, and the disclosure travels with every claim.
If the coordinator prefers no DEV-guided step, deleting leg B makes the branch inert by the published numbers.

**Branch configurations:** a new income output paired with a `SELECTED_WITNESS` occupation output (or a privacy-stage output
derived from one); its matched component comparison is the same occupation output with H3 income; same reserve, stability and
nomination rules, nothing relaxed. If the trigger does not fire, no branch configuration may be fitted.

## 4. Comparator set (frozen now; never changed after any outcome)

| Comparator | What it is | Published MODEL_SELECT point feasibility | Role |
|---|---|---|---|
| NEURAL_RAW | neural release | all 28 pass (reserve one seed only) | **required** |
| H_LJ | LOCAL income + JOINT_TREE occupation | all 28 pass | **required**; target reference |
| B2TWIN | 6-level LOCAL income code + JOINT_TREE | all 28 pass | **required** |
| RETIRED_B3CTL | retired B3CTL (JOINT occupation + 6-level income code) | all 28 pass (reserve not met) | **required** |
| SUCC | H3 income + JOINT_TREE occupation (the source successor) | all 28 pass | **required**; matched-income component (CC) |
| TWJ | HC_STRONG income + JOINT_TREE occupation | all 28 pass | **required** |
| PUB_IBSI | published IBSI adaptation | seed-1 men's guard 0.0050142 > 0.005 | descriptive |
| T_TREE, TASK_REFIT | equally exposed task-only releases | pass | stage-1 utility yardsticks; recovery descriptive |
| U0, Ucal | utility references | - | define the clauses |

**Eligibility rule.** A frozen release is an eligible comparator iff all 24 utility clauses and 4 guards pass at the point on
both seeds (published values) and a frozen package exists for both seeds. Reserve status, recovery and the candidate's results
play no role. **"Strongest eligible set" = every eligible member of the universe the coordinator listed**: superiority over all
of them implies superiority over the strongest of them, and choosing a subset would be a choice that could be tuned. In
particular RETIRED_B3CTL is required: it was retired because of its reserve and validation outcomes, which are not point
feasibility, and it is the strongest code comparator (published DEV_REPORT pair AUC 0.6130 / 0.6168 against SUCC's 0.6272 /
0.6263). The original 0.020 target keeps its original reference set, NEURAL_RAW and H_LJ (section 7).

## 5. Development reader grid conditions (SCORE roles only)

Cells: `S0R1k0 S0R1k1 S0R2k0 S0R2k1 S1R1k0 S1R1k1 S1R2k0 S1R2k1` (release seed x repetition x reader seed). Signs: pair benefit
b = comparator AUC - candidate AUC (positive favours the candidate); recipient difference r = candidate AUC - comparator AUC.
For each (candidate, comparator, view): per-seed mean d_s over its 4 cells; v_rep = pooled within-seed variance over the 8 cells
(df 6; reader seeds, splits and the different SCORE sets of the two repetitions); v_samp = mean of the 8 squared cell bootstrap
SEs (SCORE households, readers fixed); **sd_total = sqrt(v_rep + v_samp)**.

**Stability screen, `STABLE_POSITIVE` (gate, every required comparator):**
1. all 8 cell benefits > 0 (strict), and
2. for each seed s: d_s - 2 x sd_total > 0.

Other labels: `POSITIVE_NOT_STABLE` (every cell positive, 2 fails), `SIGN_CHANGES` (8-cell mean > 0 but some cell <= 0),
`NOT_POSITIVE`, `INCOMPLETE` (a registered cell, a cell SE or a finite value missing; fails).

**Why.** "Frequently changes sign" is made precise twice. Observed: no sign change in any of the 8 matched cells (the source's
every-cell rule). Predicted: under a normal approximation, a new matched repetition of either seed flips sign with probability
Phi(-d_s / sd_total) <= Phi(-2) = 0.023. One frozen reader with a positive interval cannot pass, because sd_total contains the
across-refit spread. Adding the bootstrap term is conservative (part of it is already in v_rep) and stops four nearly identical
cells from looking certain. Scale check from source numbers: DEV_REPORT benefit SEs were about 0.002 (vs H_LJ) and 0.005 (vs
NEURAL_RAW) at 5,943 people, so about 0.0033 and 0.0082 at the about 2,400 SCORE people; a benefit like the source successor's
(about +0.012 vs H_LJ, +0.020 vs NEURAL_RAW) passes narrowly, a benefit below about 2 cell SEs does not. The source's seed-1
reversals under final reader refits (3 of 4 negative against H_LJ) are the failure this screen targets.

**Recipient protection.** Gate: every cell r <= 0.010, income and occupation views, every required comparator
(`RECIPIENT_POINT_FAILED` otherwise). Recorded, not a gate: `RECIPIENT_STABLE` when d_s + 2 x sd_total <= 0.010 on both seeds,
else `RECIPIENT_EDGE` (not computed when the grid gives no recipient cell SEs).

**Material target on the grid (recorded, never a gate),** against NEURAL_RAW and H_LJ: `DEV_GRID_TARGET_STABLE` (every cell >
0.020 and d_s - 2 sd_total > 0.020), `DEV_GRID_TARGET_POINT_ONLY` (8-cell mean > 0.020), `DEV_GRID_TARGET_NOT_MET`.

**Recorded, never gating:** VALIDATION near ties (< 0.010) and winner changes per cell; the paired benefit distribution (cells,
mean, min, max, per-seed means, both SD components, predicted flip probability); nested learning curves at 1/4, 1/2 and the full
FIT size, flagged `BENEFIT_SHRINKS_WITH_READER_DATA` when the full-size mean benefit is below the quarter-size one by more than
0.005 (smaller samples cannot establish full-budget protection; a shrinking benefit warns that the 8,001-person final readers may
erase it).

## 6. Nomination (at most one whole-release finalist)

**Admissible:** both seeds evaluated; runnable package for both seeds; release computed from the six permitted raw fields; grid
entries against all 6 required comparators; not an alias of a required comparator. A control-labelled configuration is
admissible and competes on equal terms.

**Gates, in order, with the first failure as the label:** `UTILITY_POINT_INELIGIBLE` (24 clauses + 4 guards at the point) ->
`RECIPIENT_POINT_FAILED` -> `RESERVE_NOT_MET` (R1; flag `EDGE_STABLE_BENEFIT_WITHOUT_RESERVE` if the benefit is stable: the retired
candidate's profile) -> `BENEFIT_NOT_STABLE` (any required comparator not `STABLE_POSITIVE`) -> `NO_MATCHED_COMPONENT_COMPARISON`
(no comparison on the same 8 cells with the configuration's matched component: its task-only occupation with the same income, or
for a task-only configuration SUCC; for a branch configuration the same occupation with H3 income) -> **`QUALIFYING`**.
Earlier: `NOT_CONSIDERED_ONE_SEED_ONLY`, `NOT_ADMISSIBLE`, `ALIAS_OF_<COMPARATOR>`.

**Score:** the **minimum over the 6 required comparators of the 8-cell mean pair benefit** (both seeds x 2 repetitions x 2 reader
seeds); higher is better. **Near-tie set:** score >= best - 0.005 (about half the largest source cell range and the lower end of
the validation's reader-refit SD). **Tie-breaks inside it:** (1) the leximin whole-release reserve vector (minimum whole-release
reserve first), (2) simpler implementation: fewer fitted components (n_model_fits + n_calibrator_fits, both seeds), (3) earliest
ledger entry of the seed-0 evaluation. No income-only and no Brier-specific tie-break. The chosen configuration is
`NOMINATED_FINALIST` (with `CONTROL_DESIGNATED_FINALIST` if it is a control); other qualifiers `QUALIFYING_NOT_CHOSEN`. If none
qualifies: `NO_QUALIFYING_CONFIGURATION`, **DEV_REPORT stays closed**, and the frontier (score against the minimum whole-release
ratio) and every label are reported.

**Flags (never gates):** `RECIPIENT_EDGE`, `PAIR_NEAR_TIE` (some required mean benefit < 0.005).

## 7. Outcome labels: one record each, never merged

| Outcome | Labels | Meaning |
|---|---|---|
| Point eligibility | `POINT_ELIGIBLE` / `POINT_INELIGIBLE` | 24 clauses, 4 guards (MODEL_SELECT) and every recipient cell (grid) at the point |
| Reserve (R1) | `RESERVE_MET` / `RESERVE_ONE_SEED_ONLY` / `RESERVE_NOT_MET` / `RESERVE_PENDING_SECOND_SEED` | whole-release development heuristic |
| Conditional confidence | `UTILITY_CONFIDENCE_RESOLVED` / `UTILITY_CONFIDENCE_UNRESOLVED (n of 28)` | value + 3.1237346 x own SE <= cap (Bonferroni over 28, one-sided 0.025); conditional on the fitted configuration; does not cover the search |
| Recipient protection | `RECIPIENT_STABLE` / `RECIPIENT_EDGE_OR_UNCOMPUTED` / `RECIPIENT_POINT_FAILED` | grid SCORE cells |
| Reader stability | `BENEFIT_STABLE_ALL_REQUIRED` / `BENEFIT_NOT_STABLE` (per comparator: `STABLE_POSITIVE`, `POSITIVE_NOT_STABLE`, `SIGN_CHANGES`, `NOT_POSITIVE`, `INCOMPLETE`) | section 5 |
| Material benefit | `DEV_GRID_TARGET_STABLE` / `DEV_GRID_TARGET_POINT_ONLY` / `DEV_GRID_TARGET_NOT_MET` | 0.020 vs NEURAL_RAW and H_LJ |
| Matched component | `PRESENT` / `MISSING`; incremental protection label vs that component | section 6 |
| Stage 1 | `PROMISING_WITNESS`, `SELECTED_WITNESS`, `PROMISING_NOT_SELECTED`, `NOT_A_WITNESS`, `NOT_TASK_ONLY`, `OCCUPATION_POINT_INELIGIBLE`, `RELEASE_POINT_INELIGIBLE`, `OCCUPATION_RESERVE_NOT_MET` | section 2 |
| Branch | `INCOME_BRANCH_TRIGGER` fires / does not fire, with legs A and B | section 3 |
| Nomination | section 6 labels; `FINALIST_NOMINATED` / `NO_QUALIFYING_CONFIGURATION` | at most one |

**The six separate claims labels** (development forms here, final forms in `CLAIMS_DRAFT.md`): (1) **useful task-only component**
(`SELECTED_WITNESS`); (2) **incremental reader-relative protection** (stability against the matched component); (3) **positive
superiority over all required comparators** (`STABLE_POSITIVE` against all 6); (4) **the original 0.020 criterion**; (5) **a
joint-design advantage**: `NOT_CLAIMABLE` here, because only occupation is optimised; the pair term is a **coalition-aware
occupation correction**, not a two-output joint design; (6) **algorithmic novelty**: `NOT_CLAIMED`. Labels 4-6 never follow from
1-3, and **a partial pass never becomes a competitive claim**.

## 8. What the coordinator's records must carry (inputs of `ohn.rules`)

- Per seed, the evaluator's utility dict with `clauses` (task, reference, metric, value, cap, se), `guards` (group, value, cap,
  se) and `households`.
- Per configuration: `task_only` (stage 1), `n_model_fits`, `n_calibrator_fits`, `entry` (seed-0 ledger entry), `kind`,
  `runnable_package`, `six_raw_fields`, `alias_of`, and `matched_component` = {config_id, grid entry on the same 8 cells}.
- Per (configuration, comparator): `pair`, `income`, `occupation`, each with `cells` and `cell_se` keyed by the 8 cell ids;
  near-tie and winner records may be added as extra keys (passed through as diagnostics).
