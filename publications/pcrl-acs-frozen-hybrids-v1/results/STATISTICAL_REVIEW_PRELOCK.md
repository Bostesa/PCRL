# Pre-lock statistical review: ACS frozen hybrids (H_LJ)

Role: statistics reviewer. Date: 2026-10-15. Prospective review of `PROTOCOL_DRAFT.md`, done before any recovery reader is fit.

Read for this review:

- the draft;
- the source's `SCIENCE_LOCK.json` (`inference` section), `STATISTICAL_REVIEW_PRELOCK.md` and `SELECTION_LOCK.json`;
- the source's `DEV_REPORT_RESULTS.json`, for structure only (key names; no values read);
- `acs_replication/report.py`.

I opened no data, no `.private` directory and no DEV_REPORT truth. I computed no statistics. I computed only t critical values.

## Verdict

The architecture follows the source and is sound in these respects:

- one shared paired household bootstrap;
- three claims, each an IUT at one-sided α/3;
- failing-direction labels read only from the primary simultaneous intervals;
- nested decisions that need no extra alpha;
- a claim B gate that matches the source's R3 rule.

Estimand signs and margins are correct. The draft's K_primary = 38 is the right count, and the increment stays in it. **Ten changes are required before lock.** The main ones are:

- K_secondary drops from 28 to 22, because the draft lists recipient endpoints once per hybrid instead of once per wire;
- the null control needs a consequence for the decisions;
- "overall confidence status" is not defined;
- the protocol must say plainly what is already known: 30 of the 38 primary endpoints recompute values the source has already published, 2 more have published points, and only 6 carry new information.

## Checked and correct (no change)

- **Signs.** Benefit vs X = AUC_pair(X) − AUC_pair(H_LJ), where positive means H_LJ reveals less. Worsening = AUC_view(H_LJ) − AUC_view(TASK_REFIT), where positive means H_LJ reveals more. Increment = AUC_pair(H_LL) − AUC_pair(H_LJ). The utility excesses use the source's signs.
- **Margins.** The benefit margin is 0.020. Recipient worsening is capped at 0.010. Accuracy loss and NLL excess are capped at 0.010, and summed Brier excess at 0.005. Claim B uses margin 0, with a nested decision at 0.020. The subgroup cap is 0.005.
- **Claim A.** The IUT is valid for its 6 bound clauses. The 24-clause utility point gate only shrinks the rejection region and carries no error control. The nested utility-bounds decision is a subset of claim A and needs no extra alpha.
- **Claim B.** H_LJ meets the comparators' eligibility rule: MODEL_SELECT point feasibility, which holds because JOINT_TREE failed only income clauses in SELECTION_LOCK. It also meets the stricter DEV_REPORT point gate. NEURAL_RAW is not gated on DEV_REPORT utility. That is conservative, because a control with degraded utility would usually leak less. FAGTB and FARE are correctly out: INELIGIBLE_FALLBACK in SELECTION_LOCK. T_TREE is diagnostic only.
- **Claim C.** The precedence is NOT_ESTIMABLE, then QUALIFIED, then HARM, then UNRESOLVED. QUALIFIED and HARM cannot both hold.
- **Bounds and bootstrap.** Wald-type bounds with t(G − 1), G = 3,955. B = 2,048, with one draw matrix shared across services, views and seeds. Seeds are never averaged.
- **Reader bank.** The fit count of 125 per seed is right: 70 marginal, 49 pair and 6 control fits.

## (a) Required changes

**R1. State the selection and scope at the top of the inference section.** The draft says "exploratory" but leaves out the following facts. Add them as written:

- H_LJ was fixed after the DEV_REPORT results of its components were public. The source published, for both LOCAL_TREE and JOINT_TREE: recovery on all three views, all 12 utility clauses per seed, and both subgroup guards. The α guarantees assume a roster fixed independently of DEV_REPORT. The error rates are therefore nominal, and every label is exploratory, not confirmation.
- **What is already public.** The 24 utility clauses and the 4 subgroup guards are exact recomputations of published values: same outputs, labels and draws, hence the same points, SEs and α/3 bounds. The H_LJ income worsening endpoint is the source's `RECIPE|income_recipient_worsening_vs_TASK_REFIT`. The point of the H_LJ occupation worsening endpoint follows from published AUCs.
- **What is new.** Only the H_LJ and H_JL pair-view recovery is new. Six primary endpoints carry new information: the two coalition benefits, the two COMPETITIVE benefits and the two increments. The utility gates, the claim C outcome apart from the K-dependent HARM label, and the income worsening clause are settled before readers run.
- Before readers run, write these settled outcomes into the lock, so that none is later reported as a finding:
  - the DEV_REPORT utility point gate;
  - the count of utility clauses passing at their α/3 bounds;
  - the four subgroup α/3 passes;
  - the income worsening clause passes;
  - the estimability status.
- **Scope (from the source).** All intervals and decisions are conditional on the frozen defense models and the fitted readers. The two seeds share people and are not independent evaluation populations, so "both seeds" means a two-fit replication. The target is the unweighted DEV_REPORT household superpopulation, not ACS survey-design variance. PWGTP values are points only. The source labels are unchanged, and the sealed assessment stays closed.

**R2. The secondary family double-counts. K_secondary is 22, not 28.** Recipient recovery belongs to a wire, not a hybrid. Of the 6 per-seed recipient entries:

- 2 copy primary endpoints: H_LL income is LOCAL income, and H_JJ occupation is JOINT occupation;
- 2 copy each other: H_JL income equals H_JJ income, and H_JL occupation equals H_LL occupation.

Listing a primary endpoint again gives one statistic two intervals and two statuses. Keep one entry per distinct wire statistic. Add the matching occupation-swap effect under JOINT income, so the 2×2 interaction can be broken down. The final list is in (c).

**R3. Recompute every status under K = 38. Do not inherit K = 40 statuses.** c₃₈ = 3.2148 is less than c₄₀ = 3.2295, so intervals around recomputed source values become about 0.5% narrower. For every endpoint that has a source status:

- show the source status (K = 40) next to the new one;
- label any difference `STATUS_CHANGED_BY_MULTIPLICITY_ONLY`;
- never describe such a change as new evidence.

The source labels are not revised.

**R4. Define "overall confidence status" and keep three fields separate.** The draft uses the term without defining it. Definitions are in (d), section 6. Overall point feasibility, overall confidence status and subgroup status are three fields, and none of them enters the overall conjunction.

**R5. Make the null control decision-bearing and split the controls into two stages.** The draft's null rule (`|AUC − 0.5| > 0.05` flagged, then "investigated") has no consequence, and a post-hoc investigation invites rationalisation. The ATTACK_VALIDATION controls need no DEV_REPORT truth, so check them before truth is opened. A missing or refused control counts as a failure. Full rules are in (e).

**R6. Complete claim B's precedence and fix two ambiguities.**

- Add INCOMPLETE and NO_ELIGIBLE_BASELINE, both never a win.
- Name the eligible set, {NEURAL_RAW}, in the decision.
- The "24 MODEL_SELECT clauses" and the "24 DEV_REPORT clauses" each span both seeds, 12 per seed. Remove "both seeds" after "24-clause", which reads as 48 clauses.

**R7. Keep the inheritance equalities out of Q1 and the conjunction.** An equality failure is a computation or alignment defect. It halts the run. It is not a scientific outcome. Two changes:

- Move the label-free part, bitwise equality of outputs and of wire construction, to before truth.
- Define the conjunction as controls AND claim A AND claim B. Claim A already contains the utility point gate.

**R8. Give a precise rule for the distance columns.** The rule is in (d), section 9.

**R9. Write the critical values into the lock.** They are in the table below.

**R10. Register forecasts before readers run**, following the source's `FORECASTS.json` practice. Cover at least: claim A, claim B, the nested 0.020 target, the sign of the increment, and the control outcomes.

### Critical values (G = 3,955; df = 3,954; Wald-type, not studentized)

| Use | Level | t |
|---|---|---|
| Claim clause (IUT) | one-sided 0.025/3 = 0.00833 | 2.3950 |
| Primary simultaneous, K = 38 | two-sided 95%, 0.05/76 per tail | 3.2148 |
| Secondary simultaneous, K = 22 | two-sided 95%, 0.05/44 per tail | 3.0541 |
| Source K = 40 (reference only, for R3) | 0.05/80 per tail | 3.2295 |
| Marginal descriptive ("unadjusted") | 0.025 per tail | 1.9606 |

## (b) Final primary roster (K = 38)

**Does the increment belong in the primary family? Yes.**

- It isolates the one design change: occupation output switched from LOCAL to JOINT, with the income output fixed.
- The handoff asks for it whatever happens to claims A and B.
- It costs under 0.5% in interval width (c₃₆ = 3.1992, c₃₈ = 3.2148).

It carries **no claim and no α/3 clause status**. It enters no decision and not the conjunction.

Notation:

- **benefit**: higher is better. The interval status is ESTABLISHED, ESTABLISHED_SHORTFALL or UNRESOLVED.
- **no_excess**: lower is better. The interval status is PASS, ESTABLISHED_VIOLATION or UNRESOLVED.
- **Value source**:
  - P = exact recomputation of a published source value;
  - p = point already public, SE new;
  - N = new.
  - On recovery rows (2, 3, 21, 22), P and p assume the replay holds.

Rows 4/7 and 10/13 (and 23/26 and 29/32) are identical statistics, because Ucal is argmax-invariant. So K = 38 covers 34 distinct statistics. Bonferroni stays conservative, as in the source, and this is disclosed.

| # | key | kind | margin | value source |
|---|---|---|---|---|
| 1 | `RECIPE\|H_LJ\|coalition_benefit_vs_TASK_REFIT\|s0` | benefit | 0.020 | N |
| 2 | `RECIPE\|H_LJ\|income_recipient_worsening_vs_TASK_REFIT\|s0` | no_excess | 0.010 | P |
| 3 | `RECIPE\|H_LJ\|occupation_recipient_worsening_vs_TASK_REFIT\|s0` | no_excess | 0.010 | p |
| 4 | `RECIPE\|H_LJ\|utility\|income\|U0\|accuracy_loss\|s0` | no_excess | 0.010 | P |
| 5 | `RECIPE\|H_LJ\|utility\|income\|U0\|nll_excess\|s0` | no_excess | 0.010 | P |
| 6 | `RECIPE\|H_LJ\|utility\|income\|U0\|brier_excess\|s0` | no_excess | 0.005 | P |
| 7 | `RECIPE\|H_LJ\|utility\|income\|Ucal\|accuracy_loss\|s0` | no_excess | 0.010 | P |
| 8 | `RECIPE\|H_LJ\|utility\|income\|Ucal\|nll_excess\|s0` | no_excess | 0.010 | P |
| 9 | `RECIPE\|H_LJ\|utility\|income\|Ucal\|brier_excess\|s0` | no_excess | 0.005 | P |
| 10 | `RECIPE\|H_LJ\|utility\|occupation\|U0\|accuracy_loss\|s0` | no_excess | 0.010 | P |
| 11 | `RECIPE\|H_LJ\|utility\|occupation\|U0\|nll_excess\|s0` | no_excess | 0.010 | P |
| 12 | `RECIPE\|H_LJ\|utility\|occupation\|U0\|brier_excess\|s0` | no_excess | 0.005 | P |
| 13 | `RECIPE\|H_LJ\|utility\|occupation\|Ucal\|accuracy_loss\|s0` | no_excess | 0.010 | P |
| 14 | `RECIPE\|H_LJ\|utility\|occupation\|Ucal\|nll_excess\|s0` | no_excess | 0.010 | P |
| 15 | `RECIPE\|H_LJ\|utility\|occupation\|Ucal\|brier_excess\|s0` | no_excess | 0.005 | P |
| 16 | `COMPETITIVE\|H_LJ\|coalition_benefit_vs_NEURAL_RAW\|s0` | benefit | 0 (nested 0.020) | N |
| 17 | `SUBGROUP\|H_LJ\|occupation_brier_excess_vs_U0\|sex0\|s0` | no_excess | 0.005 | P |
| 18 | `SUBGROUP\|H_LJ\|occupation_brier_excess_vs_U0\|sex1\|s0` | no_excess | 0.005 | P |
| 19 | `INCREMENT\|H_LJ_over_H_LL\|coalition_gain\|s0` | benefit, no claim | 0 (0.020 distance shown) | N |
| 20 | `RECIPE\|H_LJ\|coalition_benefit_vs_TASK_REFIT\|s1` | benefit | 0.020 | N |
| 21 | `RECIPE\|H_LJ\|income_recipient_worsening_vs_TASK_REFIT\|s1` | no_excess | 0.010 | P |
| 22 | `RECIPE\|H_LJ\|occupation_recipient_worsening_vs_TASK_REFIT\|s1` | no_excess | 0.010 | p |
| 23 | `RECIPE\|H_LJ\|utility\|income\|U0\|accuracy_loss\|s1` | no_excess | 0.010 | P |
| 24 | `RECIPE\|H_LJ\|utility\|income\|U0\|nll_excess\|s1` | no_excess | 0.010 | P |
| 25 | `RECIPE\|H_LJ\|utility\|income\|U0\|brier_excess\|s1` | no_excess | 0.005 | P |
| 26 | `RECIPE\|H_LJ\|utility\|income\|Ucal\|accuracy_loss\|s1` | no_excess | 0.010 | P |
| 27 | `RECIPE\|H_LJ\|utility\|income\|Ucal\|nll_excess\|s1` | no_excess | 0.010 | P |
| 28 | `RECIPE\|H_LJ\|utility\|income\|Ucal\|brier_excess\|s1` | no_excess | 0.005 | P |
| 29 | `RECIPE\|H_LJ\|utility\|occupation\|U0\|accuracy_loss\|s1` | no_excess | 0.010 | P |
| 30 | `RECIPE\|H_LJ\|utility\|occupation\|U0\|nll_excess\|s1` | no_excess | 0.010 | P |
| 31 | `RECIPE\|H_LJ\|utility\|occupation\|U0\|brier_excess\|s1` | no_excess | 0.005 | P |
| 32 | `RECIPE\|H_LJ\|utility\|occupation\|Ucal\|accuracy_loss\|s1` | no_excess | 0.010 | P |
| 33 | `RECIPE\|H_LJ\|utility\|occupation\|Ucal\|nll_excess\|s1` | no_excess | 0.010 | P |
| 34 | `RECIPE\|H_LJ\|utility\|occupation\|Ucal\|brier_excess\|s1` | no_excess | 0.005 | P |
| 35 | `COMPETITIVE\|H_LJ\|coalition_benefit_vs_NEURAL_RAW\|s1` | benefit | 0 (nested 0.020) | N |
| 36 | `SUBGROUP\|H_LJ\|occupation_brier_excess_vs_U0\|sex0\|s1` | no_excess | 0.005 | P |
| 37 | `SUBGROUP\|H_LJ\|occupation_brier_excess_vs_U0\|sex1\|s1` | no_excess | 0.005 | P |
| 38 | `INCREMENT\|H_LJ_over_H_LL\|coalition_gain\|s1` | benefit, no claim | 0 (0.020 distance shown) | N |

Definitions:

- Rows 2 and 21 use the LOCAL income wire. Rows 3 and 22 use the JOINT occupation wire.
- Utility rows: income clauses use LOCAL_TREE's income outputs, and occupation clauses use JOINT_TREE's occupation outputs.
- Row 16: AUC_pair(NEURAL_RAW) − AUC_pair(H_LJ).
- Row 19: AUC_pair(H_LL) − AUC_pair(H_LJ).

The report asserts that exactly these 38 keys exist and that every one is finite. Otherwise it halts and issues no decision.

## (c) Final secondary roster (K = 22; descriptive; not decision-bearing)

All entries are two-sided 95% simultaneous intervals over these 22 keys (t = 3.0541). No secondary status feeds any claim, gate or the conjunction. Each status is labelled "descriptive (secondary family)".

| # | key | definition | kind | reference values |
|---|---|---|---|---|
| S1 | `SEC\|H_LL\|coalition_benefit_vs_TASK_REFIT\|s0` | AUC_pair(TR) − AUC_pair(H_LL) | benefit | 0, 0.020 |
| S2 | `SEC\|H_JL\|coalition_benefit_vs_TASK_REFIT\|s0` | AUC_pair(TR) − AUC_pair(H_JL) | benefit | 0, 0.020 |
| S3 | `SEC\|H_JJ\|coalition_benefit_vs_TASK_REFIT\|s0` | AUC_pair(TR) − AUC_pair(H_JJ) | benefit | 0, 0.020 |
| S4 | `SEC\|H_LL\|coalition_benefit_vs_NEURAL_RAW\|s0` | AUC_pair(NR) − AUC_pair(H_LL) | benefit | 0, 0.020 |
| S5 | `SEC\|H_JL\|coalition_benefit_vs_NEURAL_RAW\|s0` | AUC_pair(NR) − AUC_pair(H_JL) | benefit | 0, 0.020 |
| S6 | `SEC\|H_JJ\|coalition_benefit_vs_NEURAL_RAW\|s0` | AUC_pair(NR) − AUC_pair(H_JJ) | benefit | 0, 0.020 |
| S7 | `SEC\|JOINT_income_wire\|recipient_worsening_vs_TASK_REFIT\|s0` | AUC_inc(JOINT) − AUC_inc(TR) | no_excess | 0, 0.010 |
| S8 | `SEC\|LOCAL_occupation_wire\|recipient_worsening_vs_TASK_REFIT\|s0` | AUC_occ(LOCAL) − AUC_occ(TR) | no_excess | 0, 0.010 |
| S9 | `SEC\|INCREMENT_J\|H_JJ_over_H_JL\|coalition_gain\|s0` | AUC_pair(H_JL) − AUC_pair(H_JJ) | benefit | 0, 0.020 |
| S10 | `SEC\|INTERACTION\|pair\|s0` | (AUC_LL − AUC_LJ) − (AUC_JL − AUC_JJ), pair view | difference | 0 |
| S11 | `SEC\|H_LJ\|coalition_benefit_vs_T_TREE\|s0` | AUC_pair(T_TREE) − AUC_pair(H_LJ) | benefit (diagnostic) | 0, 0.020 |
| S12–S22 | the same 11 keys with `\|s1` | | | |

Notes:

- The interaction is reported on the pair view only. On a single-recipient view it is zero by construction, because shared wires give identical scores. It is descriptive and says nothing about synergy.
- A favourable S11 can never stand in for claim B.

**Where each hybrid's recipient statistic lives.** Each displayed value carries the interval of the family that holds it. One statistic never gets two intervals.

| hybrid | income worsening | occupation worsening |
|---|---|---|
| H_LL | primary 2 / 21 | S8 / S19 |
| H_LJ | primary 2 / 21 | primary 3 / 22 |
| H_JL | S7 / S18 | S8 / S19 |
| H_JJ | S7 / S18 | primary 3 / 22 |

**Outside both families.** These carry marginal 95% intervals (t = 1.9606), labelled "unadjusted":

- absolute AUCs for the 17 distinct wires per seed, mapped to the 7 services. The wires are: income {LOCAL, JOINT, NEURAL_RAW, TASK_REFIT, T_TREE}; occupation {the same 5}; pair {H_LL, H_LJ, H_JL, H_JJ, NEURAL_RAW, TASK_REFIT, T_TREE};
- per-recipient AUC differences of each hybrid vs NEURAL_RAW and vs T_TREE;
- utility clauses and subgroup guards for the JOINT income and LOCAL occupation components;
- per-hybrid point feasibility.

PWGTP-weighted values are points only.

**Four-hybrid display (required by the handoff).** One row per hybrid and seed, with these columns:

- absolute income, occupation and pair AUC, each with its marginal interval;
- coalition benefit vs TASK_REFIT and vs NEURAL_RAW;
- income and occupation worsening vs TASK_REFIT.

Each difference carries its family interval, the distance columns from section 9, and its status.

## (d) Final decision rules

**1. Definitions.**

- θ̂ is the unweighted DEV_REPORT point estimate.
- SE is the standard deviation (ddof = 1) of the 2,048 shared household-bootstrap replicates.
- Clause bounds: benefit L₁ = θ̂ − 2.3950·SE; no_excess U₁ = θ̂ + 2.3950·SE.
- Family interval: [L, U] = θ̂ ± c_K·SE, with c₃₈ = 3.2148 (primary) and c₂₂ = 3.0541 (secondary).
- Clause pass: benefit L₁ > m; no_excess U₁ ≤ m.
- Point pass: benefit θ̂ > m; no_excess θ̂ ≤ m.
- Interval status at value m, from the endpoint's own family interval:
  - benefit: ESTABLISHED if L > m; ESTABLISHED_SHORTFALL if U < m; otherwise UNRESOLVED;
  - no_excess: PASS if U ≤ m; ESTABLISHED_VIOLATION if L > m; otherwise UNRESOLVED;
  - difference: ABOVE_ZERO if L > 0; BELOW_ZERO if U < 0; otherwise UNRESOLVED.

An interval that contains its reference value is UNRESOLVED. It never shows a violation, shortfall or harm. Clause bounds decide only positive claim clauses. Every failing-direction label comes from the K = 38 intervals.

**2. Checks before DEV_REPORT truth.** These need no labels. Any failure halts the run before truth is opened.

- Each hybrid's output arrays equal its components' frozen outputs bit for bit, in every role.
- Each hybrid pair wire equals [income wire | occupation wire] of its components, row-aligned by `row_id`.
- All 250 reader fits are complete and finite.
- The replay against the source is recorded (see (e)).
- Stage-1 controls pass (see (e)).

**Checks at the truth reading.** Any failure halts the report, and no decision is issued.

- Per-person utility metric vectors equal those of the components.
- The bootstrap weight matrix reproduces the source's: same seed, same household order. Every endpoint that recomputes a published source value has the published SE, within 1e-12 (floating-point summation order only).
- Exactly 38 primary and 22 secondary keys exist, and all are finite.

**3. Claim A, RECIPE_BENEFIT (one-sided α/3; IUT).**

- If the controls are not valid: `INVALID_CONTROL_FAILURE`.
- Otherwise `RECIPE_BENEFIT_ESTABLISHED` if and only if all of the following hold:
  - keys 1 and 20 have L₁ > 0.020;
  - keys 2, 3, 21 and 22 have U₁ ≤ 0.010;
  - all 24 utility keys (4–15, 23–34) pass at the point. This is a deterministic gate with no error control.
- Otherwise `RECIPE_BENEFIT_NOT_ESTABLISHED`, listing every failed clause.
- Nested, no extra alpha: `RECIPE_BENEFIT_AND_UTILITY_BOUNDS_ESTABLISHED` if and only if claim A is established and all 24 utility keys have U₁ ≤ cap. Otherwise `..._NOT_ESTABLISHED`.

Scope string: "error-controlled for the 6 recovery clauses (approximate Wald, nominal given post-hoc composition, conditional on frozen models and fitted readers); utility gate: point only."

**4. Claim B, COMPETITIVE (one-sided α/3). Take the first rule that applies.**

1. `INCOMPLETE` if NEURAL_RAW pair scores are missing or non-finite for either seed. This is never a win.
2. `INVALID_CONTROL_FAILURE` if the controls are not valid.
3. `NO_ELIGIBLE_BASELINE` if `SELECTION_LOCK.frozen.NEURAL_RAW.status` ≠ ELIGIBLE.
4. `H_LJ_FAILS_SAME_UTILITY_REQUIREMENTS` if H_LJ fails MODEL_SELECT point feasibility (24 clauses, 12 per seed) or the DEV_REPORT 24-clause point gate.
5. `SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL` if keys 16 and 35 both have L₁ > 0.
6. Otherwise `NOT_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL`.

Further rules for claim B:

- Nested, no extra alpha: `MATERIALLY_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL` if and only if the decision is rule 5 and keys 16 and 35 both have L₁ > 0.020. Otherwise `NOT_ESTABLISHED`.
- The decision prints the eligible set {NEURAL_RAW}. It states that FAGTB and FARE had no MODEL_SELECT-feasible configuration and were not compared.
- T_TREE is never part of claim B.
- Report the K = 38 status of keys 16 and 35 at both 0 and 0.020.

**5. Claim C, SUBGROUP_SUITABILITY (one-sided α/3; IUT). Take the first rule that applies.**

1. `NOT_ESTIMABLE` if fewer than 200 DEV_REPORT households contain a member of either SEX group. This is counted from SEX alone.
2. `SUBGROUP_QUALIFIED` if keys 17, 18, 36 and 37 all have U₁ ≤ 0.005.
3. `ESTABLISHED_SUBGROUP_HARM` if any of those keys has K = 38 lower bound L > 0.005. Name the guard(s).
4. Otherwise `SUBGROUP_UNRESOLVED`.

Reader controls do not affect claim C.

**6. Separate status fields.** These are never merged with each other or into the conjunction.

- `overall_point_feasibility`. Two parts, with no error control:
  - DEV_REPORT: `POINT_FEASIBLE_24` or `POINT_INFEASIBLE` (failed keys listed);
  - MODEL_SELECT: the admission result.
- `overall_confidence_status`, from the K = 38 intervals of the 24 utility keys:
  - `CONFIDENCE_FEASIBLE_24` if all 24 are PASS;
  - `ESTABLISHED_UTILITY_VIOLATION` if any is ESTABLISHED_VIOLATION (keys named);
  - otherwise `CONFIDENCE_UNRESOLVED`.
  - Also report the counts of PASS, UNRESOLVED and VIOLATION, and the number of clause bounds passing out of 24. That number is descriptive except inside claim A's nested decision.
- `subgroup_status`: the claim C decision, plus the four K = 38 guard statuses.

**7. Increment (keys 19 and 38).** Report θ̂, SE, the K = 38 interval, the status at 0 and at 0.020, and the distance columns. It has no clause status, no claim and no gate role.

**8. Overall conjunction (reported last).**

- `H_LJ_FIXED_COMPOSITION_SUPPORTED` if and only if the controls are valid, claim A is `RECIPE_BENEFIT_ESTABLISHED`, and claim B is `SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL`.
- Otherwise `H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED`. List every unmet component, in this order:
  1. controls;
  2. DEV_REPORT utility point gate (keys);
  3. MODEL_SELECT feasibility;
  4. coalition benefit, s0 and s1;
  5. recipient worsening (4 keys);
  6. claim B completeness or eligibility;
  7. COMPETITIVE benefit, s0 and s1.
- The conjunction uses no extra alpha. It is false only if claim A or claim B is false, and each is held to α/3.
- It excludes subgroup status, overall confidence status, the nested 0.020 targets and the increment.
- Attach this sentence to every statement of the conjunction:

> Exploratory development result on already-opened DEV_REPORT data; the composition was fixed after its components' DEV_REPORT results were public, so error rates are nominal; conditional on the two fitted seed pairs (shared people, not independent samples), the frozen defense models and the fitted readers, in the unweighted DEV_REPORT household superpopulation; not confirmation; source labels unchanged.

Any recommendation follows the source's separation rule:

- unqualified only with SUBGROUP_QUALIFIED;
- with SUBGROUP_UNRESOLVED, it must say that subgroup suitability is unresolved;
- with ESTABLISHED_SUBGROUP_HARM, no recommendation for that group.

**9. Distance columns.** Each endpoint gets one row for each reference value m in its set:

- coalition benefits (vs TASK_REFIT, NEURAL_RAW, T_TREE) and both increments: m ∈ {0, 0.020};
- recipient worsening: m ∈ {0, 0.010};
- utility and subgroup: m = cap;
- interaction: m = 0.

Columns:

- `point_minus_m` = θ̂ − m;
- `lower_minus_m` = L − m;
- `upper_minus_m` = U − m, where [L, U] is the interval of the endpoint's own family (K = 38 or K = 22);
- for claim clauses only: `clause_bound_minus_m`, which is L₁ − m (benefit) or U₁ − m (no_excess);
- `status_at_m`.

All columns are signed, with one convention: positive means above m. A no_excess endpoint therefore passes when `upper_minus_m` ≤ 0. Store values unrounded in JSON and show 4 decimals in tables. Never compute distances from a marginal interval for an endpoint that belongs to a family. Absolute AUCs get no 0 or 0.020 distances, only AUC − 0.5.

**10. Sensitivity (never decisive).** For each claim clause, a one-sided percentile bound at α/3 from the same draws. Bootstrap skewness for all 38 primary keys. Disclose every disagreement with the Wald clause decision.

## (e) Control rules

**Layout.** Controls run on the H_LJ wire, for each seed s ∈ {0, 1} and each view ∈ {income, occupation, pair}. All use OUT_HGB with model seed MODEL_SEED_BASE + s. No control is oriented.

- **Shuffled null:** trained on ATTACK_FIT with SEX permuted by `default_rng(2026101602 + s)`. This is the source's `NULL_SEED + s`.
- **Planted leak:** trained on [wire, SEX] with true ATTACK_FIT SEX.
  - On ATTACK_VALIDATION, it is scored with that role's SEX column.
  - On DEV_REPORT, it is scored as in the source: bit 1 for SEX = 1 persons and bit 0 for SEX = 0 persons.

This gives 12 null AUCs and 12 planted AUCs: 2 seeds × 3 views × {ATTACK_VALIDATION, DEV_REPORT}.

**Acceptance.**

- `PLANTED_PASS` if and only if all 12 planted AUCs are finite and ≥ 0.99.
- `NULL_PASS` if and only if all 12 null AUCs are finite and lie in [0.45, 0.55].
- `CONTROLS_VALID` if and only if both pass.
- A missing, unfit, refused or non-finite control counts as a failure.

**Stage 1, before DEV_REPORT truth:** the 6 ATTACK_VALIDATION planted values and the 6 ATTACK_VALIDATION null values.

- On failure, stop and do not open truth.
- A defect in code, wire construction or alignment may be fixed. The reader slate, hyperparameters, seeds and these thresholds may not be changed.
- After a fix, refit all 250 fits and rerun Stage 1. Disclose every defect and fix.
- If no defect is found, the campaign ends `ATTACK_PIPELINE_INVALID` and DEV_REPORT is never opened.

**Stage 2, at the single truth reading:** the 6 DEV_REPORT planted values and the 6 DEV_REPORT null values. On failure:

- Claims A and B and their nested decisions become `INVALID_CONTROL_FAILURE`.
- The increment and every recovery status, primary and secondary, become `INVALID`. Their numbers are still reported, marked invalid.
- The overall conjunction is `NOT_SUPPORTED`, with CONTROL_FAILURE listed first.
- Utility point feasibility, overall confidence status and claim C are unaffected, because they involve no reader.
- Nothing is repaired, rerun or substituted within this campaign. An investigation may be disclosed but cannot restore any decision.

**Interpretation.**

- Controls test pipeline validity. Passing shows two things: the readers find a planted leak, and they do not create signal from label-free training. A failed or refused control is never evidence of privacy.
- The DEV_REPORT planted check cannot detect a row misalignment between scores and labels, because the bit is injected with the same SEX vector used to score it. The `row_id` and wire-construction checks in (d), section 2, cover alignment.

**Why these thresholds.**

- **Planted:** the planted bit determines SEX exactly, and the source scored 1.000 on every view and seed.
- **Null:** the sampling SD of a null AUC is about 0.0075 on DEV_REPORT (about 2,970 persons per SEX) and about 0.0105 on ATTACK_VALIDATION (about 3,000 persons). That is before household clustering and before any chance alignment between a noise-trained reader and SEX. The source's six nulls fell in 0.481–0.519. A band of ±0.05 is at least 4.7 sampling SDs and 2.6 times the source's widest deviation, so chance alone almost never trips it.

**Disclosure only.**

- Each DEV_REPORT null AUC is shown with its household-bootstrap SE (same draws) and (AUC − 0.5)/SE.
- `NULL_WIDE` flags any null AUC outside the source's forecast band [0.47, 0.53] but inside [0.45, 0.55]. It has no effect on decisions.

**Replay (no labels; before truth).** Readers on the LOCAL, JOINT, TASK_REFIT, NEURAL_RAW and T_TREE wires are compared with the source's stored DEV_REPORT scores.

- A mismatch caused by a construction or alignment defect is fixed before truth, with a full refit and disclosure.
- Any other mismatch is recorded, and the run continues. Statements that recovery values equal the source's are then withdrawn.
- A replay result is never a reason to change the slate, hyperparameters or winners.

## Alpha, B and the bootstrap seed

- **α.** Keep the source's α: one-sided familywise 0.025, Bonferroni across the three claims (α/3 = 0.00833 each), IUT within claims. Do not switch to Holm. Do not change any of this after results.
- **B = 2,048** (the minimum required). One household draw matrix is shared across every service, view, endpoint and both seeds.
- **Seed 2026101603: I agree with reusing it.**
  - It is fixed before results, so no seed is chosen.
  - With the same household order, it reproduces the source's draw matrix exactly. Every endpoint that recomputes a source value then keeps its source SE, and any interval difference from the source comes only from K, which keeps R3 clean.
  - Condition: the reproduction assertion in (d), section 2.
  - Do not change B or the seed later. A different B would break the reproduction.

## Recommended (optional)

- Report the DEV_REPORT AUC of every slate reader on the pair view for H_LJ, H_LL, TASK_REFIT and NEURAL_RAW (descriptive). Label the envelope (the maximum over readers) "selected on report data". This shows whether H_LJ's benefit depends on which reader won on ATTACK_VALIDATION.
- Disclose the Monte Carlo error of the SE: about 1.6% relative at B = 2,048, as in the source.
