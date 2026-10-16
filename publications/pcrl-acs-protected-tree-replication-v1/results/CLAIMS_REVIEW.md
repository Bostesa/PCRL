# Claims review: ACS protected-tree recipe replication

**Role:** claims reviewer (read-only except this file). **Reviewed at:** 2026-10-16T00:46Z. No private data (`.private/`) was opened.

**Ground truth used:** `DEV_REPORT_RESULTS.json` (claims, endpoints, per_seed, descriptive, planning), `PRIMARY_ENDPOINTS.csv`, `SERVICE_TABLE.csv`, `SUBGROUP_TABLE.csv`, `SELECTION_LOCK.json`, `SELECTION_DETAIL.json`, `STAGE_A_HEADROOM.json`, `SCIENCE_LOCK.json`, `POPULATION_DECISION.json`, `EXTRACTION_SUMMARY.json`, `DEPLOYMENT_RESULTS.json`, `BACKUP_VERIFICATION.json`, `FORECAST_SCORES.json`, `baselines/FAGTB_AUDIT.md`. For context I also read `INDEPENDENT_VERIFICATION.json`, `REPORTING_LOCK.json`, `OCCUPATION_MAPPING.json`, `acs_replication/extract.py`, `acs_replication/leafcorr.py` and `acs_replication/schema.py`.

**The documents changed while I was reviewing them** (the lead added verification text). This review applies to the versions below. Anything edited after 00:46Z needs a fresh check.

| file | sha256 (first 12) |
|---|---|
| START_HERE.md | `d88e3b240187` |
| ADVISOR_BRIEF.md | `440c5c53da2b` |
| RESEARCH_DECISION.md | `ef36cb517cd8` |
| POPULATION_AND_SCHEMA.md | `fe4d8f16e2d2` |
| SEALED_ASSESSMENT_STATUS.md | `cbac79f41386` |
| MANUSCRIPT_CLAIMS_HANDOFF.md | `bff14eaa5d7f` |
| QUICKSTART.md | `4ef071997701` |
| LOCK_NOTES.md | `67fc7a19af28` (uncommitted edits present) |

**Verdict.**
- The three decision labels, the 40 primary endpoints and the utility counts are reported correctly.
- The three tables match the JSON exactly: 40/40 endpoint rows and 18/18 service rows, to 1e-12.
- No prohibited claim is made outright.
- 17 fixes are REQUIRED. They fall into five groups:
  - three numbers that are rounded wrongly;
  - female-subgroup and teacher-headroom wording that the lock's labelling rules do not allow;
  - a sealed-assessment statement that the extraction code contradicts;
  - an undisclosed recipe access change in the manuscript handoff;
  - several headline sentences that drop the seed-1 and subgroup qualifiers.

Line numbers below refer to the hashed versions.

---

## REQUIRED (must fix; current text -> replacement)

### R1. Seed-1 coalition benefit is rounded wrongly ("0.018")

`RECIPE|coalition_benefit_vs_TASK_REFIT|s1` point = 0.0174957. That is 0.0175 at 4 dp (as in the RD table) and 0.017 at 3 dp, never 0.018.

- START_HERE.md l8:
  - Current: `recovery fell only 0.007–0.018 AUC against matched continued task fitting, below the pre-registered 0.020 margin (an established shortfall on seed 0).`
  - Replacement: `recovery fell only 0.007 and 0.0175 AUC (seed 0 / seed 1) against matched continued task fitting, below the pre-registered 0.020 margin at the point: an established shortfall on seed 0, unresolved on seed 1.`
- ADVISOR_BRIEF.md l28:
  - Current: `It fell only 0.007 and 0.018, against the 0.020 target. On seed 0 the shortfall is established.`
  - Replacement: `It fell only 0.007 and 0.0175, against the 0.020 target. On seed 0 the shortfall is established; on seed 1 it is unresolved.`
- RESEARCH_DECISION.md l18:
  - Current: `That coalition reduction was +0.007 and +0.018 AUC`
  - Replacement: `That coalition reduction was +0.007 and +0.0175 AUC`
- MANUSCRIPT_CLAIMS_HANDOFF.md l7:
  - Current: `and 0.018 (seed 1, unresolved)`
  - Replacement: `and 0.0175 (seed 1, unresolved)`
- MANUSCRIPT_CLAIMS_HANDOFF.md l24 (`0.007–0.018 AUC`): replaced by R15.

### R2. JOINT_TREE coalition AUC is rounded wrongly

The JOINT_TREE pair AUCs are 0.603492 (s0) and 0.606488 (s1), so the correct 3-dp values are 0.603 and 0.606. The values 0.604 / 0.607 come from double rounding.

- RESEARCH_DECISION.md l94:
  - Current: `cut the coalition AUC to 0.604 / 0.607`
  - Replacement: `cut the coalition AUC to 0.603 / 0.606`

ADVISOR's "about 0.605" is fine (the mean is 0.6050).

### R3. T_TREE descriptive NLL excess is rounded wrongly

The seed-0 T_TREE occupation NLL excess vs U0 on DEV_REPORT is −0.011486, which rounds to −0.011.

- RESEARCH_DECISION.md l64:
  - Current: `T_TREE has NLL excess −0.012 / −0.009`
  - Replacement: `T_TREE has NLL excess −0.011 / −0.009`

### R4. The male subgroup bounds are misstated and the two families are mixed

The one-sided α/3 upper bounds are 0.00508 and 0.00710. Written as "0.005", the first appears not to cross a 0.005 guard. The UNRESOLVED label comes from the K = 40 intervals, not from these bounds.

- ADVISOR_BRIEF.md l38:
  - Current: `The occupation Brier excess is +0.0002 and +0.0025, but the upper bounds (0.005 and 0.007) cross the +0.005 guard.`
  - Replacement: `The occupation Brier excess is +0.0002 and +0.0025. The one-sided α/3 upper bounds (0.0051 and 0.0071) exceed the +0.005 guard, so the group is not qualified. The K = 40 simultaneous intervals ([−0.0065, +0.0068] and [−0.0038, +0.0087]) cross the guard, so harm is not established either.`

### R5. The female subgroup is reported as passing on the basis of marginal clause bounds

The lock does not allow this.
- `SCIENCE_LOCK.inference.design`: per-clause α/3 bounds "can neither establish violations, shortfalls or harm nor be counted after a failed conjunction".
- `claim_rules.SUBGROUP_SUITABILITY`: "All PASS/VIOLATION/SHORTFALL/HARM labels are read from the K = 40 simultaneous intervals".

SUBGROUP_QUALIFIED failed, because the male guards fail. The female simultaneous status is UNRESOLVED on seed 0 ([−0.0084, +0.0058]) and PASS on seed 1 ([−0.0106, +0.0030]).

- START_HERE.md l10:
  - Current: `- **Subgroups:** women's usefulness passes; men's is unresolved.`
  - Replacement: `- **Subgroups:** unresolved. In the K = 40 simultaneous intervals the women's occupation-Brier guard passes on seed 1 and is unresolved on seed 0; the men's is unresolved on both seeds. No harm is established.`
- ADVISOR_BRIEF.md l37:
  - Current: `- **Women:** yes.`
  - Replacement: `- **Women:** not established on both seeds. The point excesses are −0.0013 and −0.0038 (better than U0). The simultaneous interval passes on seed 1 and is unresolved on seed 0 (upper bound 0.0058).`
- MANUSCRIPT_CLAIMS_HANDOFF.md l9:
  - Current: `**Subgroup.** Women: the occupation-Brier guard passes. Men: unresolved. Make no unqualified recommendation.`
  - Replacement: `**Subgroup.** SUBGROUP_UNRESOLVED. Women: the occupation-Brier guard passes in the K = 40 simultaneous interval on seed 1 and is unresolved on seed 0. Men: unresolved on both seeds (one-sided bounds 0.0051 and 0.0071 exceed the 0.005 cap; no established harm). Any recommendation must state that subgroup suitability is unresolved.`
- RESEARCH_DECISION.md l126:
  - Current: `**Partly; unresolved.**`
  - Replacement: `**Unresolved.**`

### R6. Teacher headroom over U0 is called "the gate" and "established"

The pre-registered gate is in `SCIENCE_LOCK.models.T_TREE.headroom_gate`. The benchmark is SUPPORTED iff T_TREE has positive NLL and accuracy gain over constant prediction on MODEL_SELECT for both tasks and seeds, and MODEL_FIT class support passed. The rule adds: "A useful teacher continues ... even if headroom over U0 is small or negative."

So headroom over U0 is not the gate. It has no interval; it was forecast F1 and is a point comparison. The LOCAL_TREE point gate is likewise "a deterministic condition on the estimates, with no error control" (`inference.design`).

- LOCK_NOTES.md l14:
  - Current: `gate SUPPORTED_CONTINUE (teacher better than U0 on MODEL_SELECT for both tasks and seeds)`
  - Replacement: `gate SUPPORTED_CONTINUE (pre-registered usefulness gate: T_TREE has positive NLL and accuracy gain over constant prediction on MODEL_SELECT for both tasks and seeds, and MODEL_FIT class support passed); headroom over U0 was also positive at the point (descriptive; forecast F1)`
- RESEARCH_DECISION.md l143:
  - Current: `| Teacher headroom over U0 (MODEL_SELECT, both tasks and seeds) | **established** (gate; point-based, pre-registered) |`
  - Replacement: `| Teacher headroom over U0 (MODEL_SELECT, both tasks and seeds) | **observed at the point** (descriptive, no interval; forecast F1 occurred). The pre-registered gate (teacher useful over constant prediction, plus class support) is SUPPORTED_CONTINUE |`
- ADVISOR_BRIEF.md l43–46: delete `  - teacher headroom;` and `  - point-level utility;` from **Established**, and add this bullet after it:
  - `- **Observed at the point (no error control):** teacher headroom over U0 on MODEL_SELECT; all 24 LOCAL_TREE utility clauses (14 PASS and 10 unresolved in the simultaneous intervals).`

### R7. ADVISOR gives the wrong reason for FAGTB's descriptive status

FAGTB is descriptive because no λ was feasible on MODEL_SELECT (`SELECTION_LOCK.frozen.FAGTB.status = INELIGIBLE_FALLBACK`). It is not descriptive because its recovery was similar.

- ADVISOR_BRIEF.md l33:
  - Current: `failed every utility clause. Its coalition recovery was about the same as the recipe's, so it is reported descriptively.`
  - Replacement: `had no utility-feasible λ on MODEL_SELECT, so it was ineligible under the frozen rule and is reported descriptively. On DEV_REPORT it failed every utility clause, and its coalition recovery was about the same as the recipe's (benefit −0.009 / +0.007).`

### R8. "The neural control protected the coalition better" is stated without the seed-1 qualifier

NEURAL_RAW benefit −0.0167 (s0) is an ESTABLISHED_SHORTFALL. On s1 (−0.0112) the simultaneous interval is [−0.0254, +0.0029], so it is UNRESOLVED.

- START_HERE.md l9:
  - Current: `the eligible neural adversarial control protected the coalition better.`
  - Replacement: `the eligible neural adversarial control had lower coalition recovery on both seeds at the point (by 0.017 and 0.011 AUC); for the recipe this is an established shortfall on seed 0 and unresolved on seed 1.`
- RESEARCH_DECISION.md l19:
  - Current: `the eligible neural adversarial control protected the coalition better.`
  - Replacement: `the eligible neural adversarial control had lower coalition recovery at the point on both seeds (established on seed 0, unresolved on seed 1).`
- RESEARCH_DECISION.md l113:
  - Current: `The eligible neural control protects the coalition better at feasible utility.`
  - Replacement: `The eligible neural control has lower coalition recovery at feasible utility (established on seed 0, unresolved on seed 1).`
- ADVISOR_BRIEF.md l32:
  - Current: `That control protected the coalition better (by 0.011–0.017 AUC) at feasible utility.`
  - Replacement: `That control had lower coalition recovery (by 0.017 and 0.011 AUC at the point) at feasible utility; established on seed 0, unresolved on seed 1.`
- MANUSCRIPT_CLAIMS_HANDOFF.md l8:
  - Current: `with seed 0 an established shortfall.`
  - Replacement: `with seed 0 an established shortfall and seed 1 unresolved.`

### R9. The headline "protects individual recipients but not their linked coalition" overstates both halves

The recipient result is relative to TASK_REFIT and comes from a finite reader slate. Residual recovery is well above chance: income 0.549–0.558 and occupation 0.582–0.584 AUC.

On the coalition, the seed-1 K = 40 interval is [+0.0063, +0.0286], wholly above 0. By the same reasoning the documents use for recipients, a positive coalition reduction is established on seed 1. What fails is the 0.020 margin.

- START_HERE.md l3:
  - Current: `**Answer:** the local recipe protects individual recipients but not their linked coalition.`
  - Replacement: `**Answer:** relative to matched continued task fitting, the local recipe lowered each recipient's own SEX recovery, but lowered the linked coalition's recovery by less than the pre-registered 0.020 margin.`
- ADVISOR_BRIEF.md l3:
  - Current: `**Answer:** the recipe protects individual recipients, but not the linked coalition of the two.`
  - Replacement: the same sentence as for START_HERE.
- MANUSCRIPT_CLAIMS_HANDOFF.md l6:
  - Current: `**Recipient-level protection with utility.**`
  - Replacement: `**Recipient-level recovery reduction with point-level utility.**`

### R10. The mechanism paragraph overstates and borrows an untested Adult result

The Adult sentence is not supported by any frozen result in this campaign. "Exactly" overstates a single diagnostic that was never selected and was utility-infeasible on both MODEL_SELECT and DEV_REPORT.

- RESEARCH_DECISION.md l94:
  - Current: `In Adult the joint design added nothing. In Florida, protecting the coalition is exactly what the local recipe misses.`
  - Replacement: `Descriptively, in Florida the coalition is where the local recipe's reduction falls short; one unselected, utility-infeasible diagnostic (JOINT_TREE) is consistent with this but does not establish the mechanism.`
  - If the Adult comparison is kept, cite the source publication and add "not re-tested here".

### R11. The sealed-assessment documents say no labels were computed; the code computed them

`acs_replication/extract.py` `eligibility()` builds a `parsed` record for every eligible person before `role_of()` is applied. That record holds the input values, the SEX code, PWGTP, RELSHIPP and both task labels (`'income'`, `'occupation'`). For sealed and capped-out households the record is held in memory and discarded unwritten.

The lead should decide whether this also needs an access disclosure. It sits uneasily with two statements:
- `SCIENCE_LOCK.prohibitions`: "SEALED_ASSESSMENT closed throughout (features, labels and predictions)".
- RD l191: reading sealed eligibility variables is something "the access rule forbids".

- SEALED_ASSESSMENT_STATUS.md l5:
  - Current: `**What was computed for it.** Only the eligibility predicate during extraction, to count them, plus a sha256 of the sorted household-key set: `EXTRACTION_SUMMARY.json → sealed_assessment.household_key_set_sha256`.`
  - Replacement: `**What was computed for it.** The extraction made a single eligibility pass that parsed every eligible record before roles were assigned. For sealed people it therefore read the eligibility and target-validity variables and computed in memory their inputs, SEX code, PWGTP and both task labels (`acs_replication/extract.py`, `eligibility()`). Those values were discarded. Nothing was written, summarised or used except the household and person counts and a sha256 of the sorted household-key set (`EXTRACTION_SUMMARY.json → sealed_assessment.household_key_set_sha256`).`
- SEALED_ASSESSMENT_STATUS.md l7:
  - Current: `No feature matrix, label, support table, weight file or model prediction was written or computed for these households.`
  - Replacement: `No feature matrix, label file, support table or weight file was written for these households, and no model prediction was computed. Labels were computed in memory during the eligibility pass and discarded (see above).`
- SEALED_ASSESSMENT_STATUS.md l19:
  - Current: `record that this campaign read only eligibility for them.`
  - Replacement: `record that this campaign's extraction parsed these households' records in memory (eligibility and target-validity variables, inputs, SEX and both labels), wrote only counts and the key-set hash, and used nothing else.`
- RESEARCH_DECISION.md l28:
  - Current: `They stayed closed: no features, labels or predictions were produced.`
  - Replacement: `They stayed closed: no features, labels or predictions were written or used, and no model saw them. The extraction's eligibility pass parsed their records in memory only (`SEALED_ASSESSMENT_STATUS.md`).`

### R12. The recipe's transport description leaves out the registered access change

`SCIENCE_LOCK.models.LOCAL_TREE.recipe.privacy_pool` = "MODEL_FIT rows with MODEL_FIT SEX (ACS-CHANGE: authorised access)". The `leafcorr.py` docstring adds: "The source split UPDATE (SEX) / GUARD (no SEX) does not exist here." This is a change to data access, not a role name.

- MANUSCRIPT_CLAIMS_HANDOFF.md l6:
  - Current: `transported with only the declared dimension, class-count and role-name changes; labelling disclosure D1`
  - Replacement: `transported with only the registered ACS changes: input width, five occupation classes, critic widths 3/6/9, role names, and an access mapping in which both the task pool and the privacy pool are MODEL_FIT (the source's separate UPDATE-with-SEX / GUARD-without-SEX split does not exist here); labelling disclosure D1`
- POPULATION_AND_SCHEMA.md l12:
  - Current: `input width, the five occupation classes, critic widths 3/6/9, role names, schema labels and the appended Ucal procedure fixed in the SCIENCE_LOCK.`
  - Replacement: `input width, the five occupation classes, critic widths 3/6/9, role names, the privacy-pool access (MODEL_FIT with SEX replaces the source's UPDATE/GUARD split), schema labels and the appended Ucal procedure fixed in the SCIENCE_LOCK.`

### R13. START_HERE overstates what the verifier reproduced

The verifier recomputed the result values from stored outputs. It only spot-checked the document numbers, and R1–R3 are document errors that passed that spot-check.

- START_HERE.md l11:
  - Current: `an independent verifier with separate code reproduced every number and decision.`
  - Replacement: `an independent verifier with separate code reproduced every decision and every result value it recomputed from the stored outputs (document numbers were spot-checked only).`

### R14. The manuscript handoff leaves out the lock's inference scope

Source: `SCIENCE_LOCK.inference.scope` and `design`.

- MANUSCRIPT_CLAIMS_HANDOFF.md l3:
  - Current: `(custom two-task benchmark; SEX protected; one DEV_REPORT opening; sealed 20% closed):`
  - Replacement: `(custom two-task benchmark; SEX protected; DEV_REPORT opened once for analysis; sealed 20% closed; all intervals and decisions are conditional on the two fitted seed pairs and the frozen 7-reader slate, which bounds recovery from below; they include no between-fit (refit) variance and no survey-design variance):`

### R15. The suggested manuscript sentence drops the qualifiers

It carries the wrong number from R1, omits seed 1, and gives no subgroup clause, which the separation rule requires.

- MANUSCRIPT_CLAIMS_HANDOFF.md l24:
  - Current: `> In a Florida ACS 2019 development replication, the routing-preserving local recipe retained point-level utility and cut each recipient's own SEX recovery, but a linked coalition of the two recipients recovered nearly as much as from the unprotected teacher. The coalition benefit over matched continued fitting was 0.007–0.018 AUC, below the pre-registered 0.020 margin. An eligible neural adversarial control did better.`
  - Replacement: `> In a Florida ACS 2019 development replication (conditional on two fitted seed pairs and a frozen reader slate), the routing-preserving local recipe passed all 24 utility clauses at the point and lowered each recipient's own SEX recovery relative to matched continued task fitting (by 0.014–0.018 AUC for income and 0.061–0.067 for occupation). The linked coalition's recovery fell only 0.007 and 0.0175 AUC, below the pre-registered 0.020 margin (an established shortfall on one seed, unresolved on the other). An eligible neural adversarial control had lower coalition recovery at the point on both seeds (established on one). Usefulness within the male group was unresolved.`

### R16. The population-decision disclosures are incomplete, and LOCK_NOTES puts the decision in the wrong order

`POPULATION_DECISION.json` lists three disclosures and `decided_utc` 2026-10-16T00:06:00Z. The public documents carry only the first disclosure. LOCK_NOTES lists the decision before the 23:55Z metadata admission, which in fact preceded it.

- POPULATION_AND_SCHEMA.md l38:
  - Current: `- **Disclosure:** a script defect first auto-advanced to Pennsylvania. It was stopped, and nothing was written or used.`
  - Replacement: `- **Disclosures** (all three are in `POPULATION_DECISION.json`): (1) a script defect first auto-advanced to Pennsylvania; it was stopped, and nothing was written or used. (2) Case-insensitive and ERE searches were abandoned for speed and replaced by case-sensitive PCRE searches with explicit state-name variants; earlier partial runs wrote nothing. (3) The metadata admission (2026-10-15T23:55Z, SERIALNO only) ran after the manual review of the hits but before the corrected automated receipt finished (00:04Z) and before the decision was recorded (2026-10-16T00:06:00Z).`
- LOCK_NOTES.md l9: append to the note, `Decision recorded 2026-10-16T00:06:00Z, after the metadata admission (23:55Z) and the corrected receipt (00:04Z). Also disclosed: case-insensitive/ERE searches were replaced by case-sensitive PCRE searches, and the metadata admission ran before the corrected receipt finished.` Alternatively, move the row below the metadata-admission row.

### R17. "Opened once" needs the verifier re-read next to it

LOCK_NOTES l25 discloses that the verifier re-read DEV_REPORT truth between 00:21:17Z and 00:32:43Z. The headline documents say "opened once" with no qualification.

- ADVISOR_BRIEF.md l9:
  - Current: `DEV_REPORT, which was opened once.`
  - Replacement: `DEV_REPORT, which was opened once for analysis (the independent verifier later re-read it only to recompute the reported numbers; `LOCK_NOTES.md`).`
- RESEARCH_DECISION.md l51:
  - Current: `- DEV_REPORT was opened **once**, at 2026-10-16T00:20:20Z.`
  - Replacement: `- DEV_REPORT was opened **once** for analysis, at 2026-10-16T00:20:20Z (the verifier's later re-read, for recomputation only, is disclosed in `LOCK_NOTES.md`).`
- MANUSCRIPT_CLAIMS_HANDOFF.md l3: covered by R14.

---

## RECOMMENDED

1. **Income recipient reduction.** "About 0.015" covers −0.0138 and −0.0184. Use "0.014–0.018" in START_HERE l7, ADVISOR l27 and HANDOFF l6, matching the occupation range style.
2. **Name the family behind the recipient "reduction".**
   - RD l83: replace `so these reductions are firm` with `so the reductions are established in the K = 40 reporting family (approximate simultaneous 95% coverage, conditional on the fitted models and frozen readers); the pre-registered clauses test only non-worsening ≤ +0.010, which all four PASS`.
   - RD l145: append `(reporting family; not a pre-registered claim decision)`.
   - ADVISOR l45: append `(K = 40 simultaneous intervals wholly below 0)`.
3. **RD l14.** `(headroom gate SUPPORTED_CONTINUE)` -> `(usefulness gate SUPPORTED_CONTINUE: teacher beats constant prediction; class support passed)`. The lock's key is named `headroom_gate`, but its criterion is usefulness.
4. **RD l59.** Income accuracy `+0.1 / +0.06 pp` -> `+0.12 / +0.06 pp` (0.00125 and 0.00062).
5. **RD l92.** `reach 0.05–0.08 for every probability-based reader`: across seeds the five probability-input readers (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP) range 0.048–0.082 (s1 OUT_HGB is 0.0477). CONF_HGB, which reads the max probability, and DEC_LR show −0.008 to +0.023. Name the five readers and give 0.048–0.082.
6. **FAGTB wording (RD l114, l119–121).**
   - The 3-logit cap from 300 × 0.01 is exact only at λ = 0. All runs used λ > 0, where leaves can exceed 1 (FAGTB_AUDIT §8). So "underfit" is an interpretation.
   - Label the "what separates" bullets as descriptive. Add that the utility contrast is confounded with the faithful configuration's fixed shrinkage schedule, so it is not evidence that routing preservation causes it. This also guards against reading the passage as "FAGTB beaten".
7. **FARE "large utility cost" (ADVISOR l34, RD l115).**
   - The no-fairness twin Z1 also fails 12/12 clauses per seed.
   - F1's cost beyond Z1 is small: occupation accuracy +0.5 / +0.6 pp and occupation NLL +0.005 / +0.007, with income mixed (+0.2 / −0.1 pp).
   - Suggested text: `removed more coalition signal than its no-fairness twin (0.640/0.633 vs 0.667/0.666), but both F1 and its twin fail every utility clause; most of the cost is shared with compression.`
8. **RD l128–130.**
   - Fill the female "One-sided upper bound" cells with 0.0040 / 0.0012 instead of "—".
   - Rename the "IUT" column `α/3 clause bound ≤ 0.005 (marginal; not counted after the failed conjunction)`.
9. **RD l180.** `The informative finding is mechanistic` -> `The informative finding is descriptive`.
10. **RD l185.** `show the same ordering` -> `show the same ordering TASK_REFIT > LOCAL_TREE > NEURAL_RAW on both seeds`. Weighted U0 vs LOCAL_TREE flips on seed 0 (0.6471 vs 0.6466).
11. **Margin symbol.** The lock's rule is a lower bound **> 0.020**. Replace "≥ 0.020" in SEALED l15, RD l85 and RD l146.
12. **"Restored from backup alone" (ADVISOR l64, RD l176).**
    - What the evidence shows: all 736 files were restored byte-identically on the **same device**, and the restore-use check re-ran only LOCAL_TREE_S0 and its pair reader from the restored copy.
    - The off-device copy is NOT DONE (`BACKUP_VERIFICATION.json`). Say so in both places.
13. **ADVISOR l65 FUTILE.** Add `(planning treats the development points as the true values for the same frozen models and readers; the seed-1 coalition interval [+0.006, +0.029] still includes values above 0.020)`.
14. **RD l17.** `sharply cut` -> give the numbers. The income cut is 0.014–0.018, which is modest.
15. **HANDOFF claims 1 and 4.**
    - Claim 1: add `(point estimates on MODEL_SELECT; no interval)`.
    - Claim 4: `The faithful FAGTB port and official FARE were utility-infeasible` -> `FAGTB (faithful binary port plus documented multiclass adaptation; seed-1 tree random_state is a registered deviation) and official FARE had no MODEL_SELECT-feasible configuration and failed every DEV_REPORT utility clause`.
16. **POP l86 and l88–95.**
    - The "Capped out, unopened" column counts people (households: 4,483 / 0 / 561 / 601 / 2,012 / 0). Label it so.
    - Change "unopened" to "not exported": their records were also parsed in memory by the eligibility pass.
17. **SEALED l12 and RD l191.**
    - SEALED l12: `never receives sealed rows` -> `never receives sealed feature, label or weight values (the verifier read only the SERIALNO column, including sealed serials)`.
    - RD l191: `which the access rule forbids` -> `which the access rule forbids after extraction`.
18. **Subgroup labels.** SUBGROUP_TABLE.csv uses `sex0`/`sex1` with no legend. Add `sex0 = male (SEX 1), sex1 = female (SEX 2)` (`schema.SEX_CODE`) to POP or to START_HERE's table list.
19. **ADVISOR §1.** "Beat U0" holds for NLL, Brier and accuracy. Balanced accuracy is mixed (income s0 −0.0015, occupation s1 −0.0018), so qualify the sentence or name the metrics.
20. **START_HERE l56–57.** `RESOURCE_INVENTORY.json` does not exist at review time. Create it or drop the pointer.

---

## NOTES

**Numbers verified correct.** Every number in the eight documents was checked against the JSON/CSV. The following match:

| group | values |
|---|---|
| Coalition benefit | +0.0070 / +0.0175; α/3 lower bounds −0.0020 / +0.0092; K = 40 intervals [−0.0051, +0.0191] and [+0.0063, +0.0286] |
| Recipient differences | −0.014 / −0.018 and −0.067 / −0.061, with all four simultaneous upper bounds < 0 (−0.0100, −0.0119, −0.0452, −0.0460) |
| RD recovery table | all 24 AUCs |
| NEURAL_RAW, FAGTB, FARE benefits | −0.0167 / −0.0112, −0.0088 / +0.0069, −0.0171 / −0.0157 |
| FARE Z1 coalition AUC | 0.667 / 0.666 |
| Utility counts | 24/24 at the point; 17/24 at α/3 (7 failing, 6 of them accuracy); 14 PASS / 10 UNRESOLVED / 0 violations |
| Baseline utility | FAGTB and FARE fail 12/12 per seed; FAGTB accuracy loss 1.1 / 2.8 pp (income) and 5.5 / 5.7 pp (occupation); FARE occupation accuracy 0.469 |
| JOINT_TREE | fails 2 / 4 income clauses |
| Occupation NLL excess | LOCAL_TREE +0.0007 / −0.0003; TASK_REFIT −0.012 / −0.010 |
| Headroom | NLL 0.0066 / 0.0068 and 0.0121 / 0.0082; Brier 0.0041 / 0.0045 and 0.0051 / 0.0036; accuracy +0.64 / +0.61 pp |
| Subgroup | points +0.0002 / +0.0025 and −0.0013 / −0.0038; bounds 0.0051 / 0.0071 |
| Controls | null 0.481–0.519; planted 1.000 |
| Weighted coalition AUCs | all eight values |
| Same-reader coalition range | 0.006–0.019 |
| Forecasts | 6 / 14 occurred; mean Brier 0.0989 |
| Extraction and roles | role and household counts; exclusions 98,305 / 8,202 / 24,696; 74,091 eligible in 49,375 households |
| Sealed | 14,693 people / 9,853 households |
| Support and inputs | support counts; DEV_REPORT households per SEX 2,815 / 2,728; 145 inputs; 525 occupation codes |
| Source file | 195 durable-guarantees files; zip 36,712,920 B; Last-Modified 2020-10-15 |
| Lock times | lock commit times match git (−04:00 offset); DEV_REPORT opened at 00:20:20Z after REPORTING_LOCK 62fc5c606 (00:20:08Z) |
| Readers and grids | 195 reader fits per seed; NEURAL_RAW grid 3 × 4, with only ρ1 E2 feasible; FAGTB λ grid; FARE F1–F3 + Z1–Z3 |

**Recipient-level reductions (check 2).**
- The pre-registered endpoints are no-excess *worsening* clauses with a +0.010 margin. All four PASS both the IUT bound and the K = 40 simultaneous label.
- Calling the *reduction* established is justified, but only through the reporting family. All four two-sided simultaneous intervals lie wholly below 0. Simultaneous coverage (approximate, Bonferroni-t, conditional on the fitted models and frozen readers) covers the sign statements as well.
- It is not a decision-family result. The decision family tested only non-worsening inside RECIPE_EFFECT, which failed as a conjunction, so its clause passes cannot be counted.
- The documents cite the simultaneous intervals, which is the right basis. Recommendation 2 asks them to name the family.
- By the same rule, the seed-1 coalition interval [+0.0063, +0.0286] establishes a positive coalition reduction, and the headline must not imply none (R9).

**Prohibited claims (check 3).** None is made outright.
- Confirmation of Adult packages: explicitly disclaimed.
- Adult numbers and the 317,000-group requirement: neither is used.
- Novelty: disclaimed.
- Race: disclaimed.
- Survey-design variance: disclaimed.
- Certified non-use: disclaimed.
- Seed independence: disclaimed.
- FAGTB: called ineligible and descriptive.
- JOINT_TREE: labelled a diagnostic throughout.
- The only cross-study inference is RD l94 (R10). RD l134 ("differs from the Adult diagnostic") is context and acceptable. The FAGTB "what separates" bullets should be labelled descriptive (recommendation 6).

**Consistency (check 4).**
- Labels agree across all documents.
- FAGTB/FARE ineligibility is described correctly in RD and LOCK_NOTES, but not in ADVISOR (R7).
- The sealed-assessment justification (FUTILE at the development points; RECIPE coalition clauses and both NEURAL_RAW clauses are `FUTILE_AT_DEVELOPMENT_POINT` in `planning.per_clause`) is consistent everywhere. The only issue is the ≥/> nit (recommendation 11).

**Mechanism (check 5).**
- ADVISOR and RD label the interpretation "descriptive", which is right.
- The pair-carries-nearly-all-signal statement is fine as description: LOCAL pair AUC is 0.657 / 0.649 vs teacher 0.664 / 0.664, about 91–96% of the teacher's above-chance AUC.
- The overstatements are in the headline (R9), the last two sentences of RD l94 (R10), and "mechanistic" in RD l180 (recommendation 9).

**The verifier's spot-check and R2.** `INDEPENDENT_VERIFICATION.json` reports "JOINT_TREE 0.604/0.607" as reproduced. The exact values are 0.603492 / 0.606488, so its spot-check tolerance accepted double rounding. This is why R2 was not caught earlier.

**Access log.** The verifier says "the access log was not written", while LOCK_NOTES l25 says the lead recorded the re-read in the private access log. These are compatible only if the lead wrote that entry afterwards. Worth one clarifying clause.

**Verifier concern on commit b2c21894f.** The verifier noted that research commit `b2c21894f` contains in-progress verifier files. LOCK_NOTES l25 now discloses this. Make sure the final verification commit supersedes them.

---

## Resolution (added by the lead after the review; the review text above is unchanged)

- **REQUIRED R1–R17: all applied**, in START_HERE, ADVISOR_BRIEF, RESEARCH_DECISION, POPULATION_AND_SCHEMA, SEALED_ASSESSMENT_STATUS, MANUSCRIPT_CLAIMS_HANDOFF and LOCK_NOTES.
- **R11** was confirmed in `acs_replication/extract.py` and is disclosed as **D2** in `LOCK_AMENDMENTS.md`. It also covers the RELSHIPP data-quality count computed over all eligible people.
- **Recommendations 1–20: adopted.** Every proposed number was re-derived from `DEV_REPORT_RESULTS.json`, `STAGE_A_HEADROOM.json` and `EXTRACTION_SUMMARY.json` before use, and all matched.
- **Recommendation 15** led the lead to find that the settings table in `baselines/FAGTB_AUDIT.md` describes a different seeding from the locked `FAGTB_GRID.json`, which was followed. This is disclosed as **D3**.
- **Recommendations 12 and 20** (off-device custody, `RESOURCE_INVENTORY.json`) were completed after research commit `bd4d9812a`, in the closeout commit. Custody V2 with the off-device copy ran at 00:56–00:58Z, and `RESOURCE_INVENTORY.json` was written at queue close. See `BACKUP_VERIFICATION.json` and `RESOURCE_INVENTORY.json`.
- **Re-check.** The same reviewer re-checked `bd4d9812a` read-only. It found 3 remaining REQUIRED items and 8 minor ones, all applied in the closeout commit:
  - the R9 wording in RESEARCH_DECISION §9;
  - three absolute sealed-role statements, now qualified by the counts, the key-set hash, the data-quality count and the identifier-only reads;
  - two completion claims that were ahead of the files.
- **The verifier's file** (`INDEPENDENT_VERIFICATION.json`) is left as written, including its "0.604/0.607" spot-check note. The corrected values are in RESEARCH_DECISION.
