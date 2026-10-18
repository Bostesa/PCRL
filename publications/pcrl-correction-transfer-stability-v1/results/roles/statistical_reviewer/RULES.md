# Statistical rules of the correction-transfer campaign (CTS1; statistical reviewer)

**Status.** Written 2026-10-18, about 19:00-19:20Z, before any fit, label read or outcome of this campaign.
- Machine-readable constants: `RULES.json` (schema `CTS1-rules-v1`).
- Implementation: `cts/rules.py`. It uses pure numpy/scipy and reads no file except RULES.json. `load_rules()` refuses a rule
  file whose constants disagree with the module.
- Invented-data tests: `python3 -m cts.test_rules` (26 of 26 PASS; the printed summary records the RULES.json SHA-256).
- Inputs used: the coordinator's DESIGN_BRIEF.md; the source rules (OHN1-rules-v1) and code (`ohn/rules.py`, `ohn/protect.py`,
  `ihs/evaluate.py`); the source's published RESEARCH_DECISION.md and published income-noise components; and the per-view
  dispersion priors computed label-free by the reader investigator from the frozen source grid.
- I read no person-level data and ran no real-data computation.

**What every rule here is.**
- Every rule is a development rule on previously used records.
- No rule changes a cap.
- Every statement involving uncertainty is asymptotic and model-based. None is a finite-sample or distribution-free guarantee.
- The two-SE reserve is a development robustness objective, not a guarantee.
- Identity (strength 0) is always a valid outcome, and it earns no protection claim.

Data context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida. SEX is the recorded demographic
column (sex0 = men, SEX code 1; sex1 = women, SEX code 2). It is a training-time label and an evaluation label, never a runtime input.

## 0. Original limits (unchanged)

| Condition | Limit | Count |
|---|---|---|
| Accuracy loss, NLL excess (nats), summed multiclass Brier excess vs U0 and Ucal, income and occupation | 0.010, 0.010, 0.005 | 24 clauses (2 tasks x 2 references x 3 metrics x 2 source seeds) |
| Within-SEX occupation summed-Brier excess vs U0, men and women | 0.005 each | 4 guards (2 groups x 2 seeds) |
| Recipient non-worsening: single-output SEX-recovery AUC minus the comparator's | +0.010 | per comparator, view and seed |
| Material benefit target | 0.020 AUC | reported separately from positive superiority, never a gate |

## 1. Constraint set and incremental costs

**Conditions.** Each partition has 8 occupation-side conditions. They are evaluated on MODEL_SELECT for the partition's witness,
against the references of the associated source seed (D0, R0 use seed 0; D1, R1 use seed 1):
- `occupation|U0|accuracy_loss`, `occupation|U0|nll_excess`, `occupation|U0|brier_excess`;
- `occupation|Ucal|accuracy_loss`, `occupation|Ucal|nll_excess`, `occupation|Ucal|brier_excess`;
- `guard|sex0`, `guard|sex1`.

**Incremental cost.** Corrected release minus uncorrected release, on the same people. There are five statistics, mapped to
conditions exactly as `ohn.protect.SLACK_OF`:

| Statistic | Definition (mean over people) | Conditions it spends |
|---|---|---|
| `men_brier` | summed-Brier increase among men | `guard|sex0` |
| `women_brier` | summed-Brier increase among women | `guard|sex1` |
| `brier` | summed-Brier increase | both `brier_excess` clauses |
| `nll` | NLL increase | both `nll_excess` clauses |
| `acc_loss` | uncorrected accuracy minus corrected accuracy | both `accuracy_loss` clauses |

An increment does not depend on the reference. The tighter of the two references' slacks therefore binds.

**Metric definitions (unchanged).**
- They are those of `scm_bench.common.task_vectors`: accuracy = 1[argmax p = y]; NLL = -log(max(p_y, 1e-12)); summed Brier =
  sum_k (p_k - 1[y = k])^2.
- **Disclosure:** the NLL keeps the existing 1e-12 probability floor, so one person contributes at most 27.63 nats.
- No new loss is introduced.
- Probabilities are the softmax of centred logits (`acs_replication.trees.probabilities`).
- A test checks that every definition is bitwise identical to its source.

**Estimators of the increment** (per strength s and statistic):
- **OOF:** fold pipeline k on fold k of FIT, with the cross-fitted witness base. These are exactly the `ohn.protect.project` costs.
- **GUARD:** the deployed release (and, separately, the deployed ensemble) on GUARD.
- **CHECK:** the realized increment on CHECK.
- **MS:** the realized increment on MODEL_SELECT.

**Standard errors.**
- Household-clustered SE of a mean: `cluster_se` (cluster sums of centred values, G/(G-1) correction).
- SEX-group statistics: the linearized household-clustered SE of a ratio of means, computed on the whole set with the group mask
  (`subgroup_se`), i.e. the SE is computed within the SEX group and clustered by household.
- All are identical to `ihs.evaluate`.

## 2. Baseline slack (the reserve)

**Rule.**
- rho_c = margin_c - 2 x se_used_c, where margin_c = cap_c - (witness value on MODEL_SELECT).
- se_used_c = max(own household-clustered SE, floor SE).
- The floor is H_LJ's MODEL_SELECT SE for the same condition and seed, copied unchanged from OHN1, rescaled by sqrt(5306 / households).

**Zero or tiny own SE** (the `ohn.rules` conventions):
- An own SE of 0, or one below the floor, uses the floor. This is recorded as `se_source = floor` and `own_se_zero`.
- A floor that is missing, non-finite or <= 0 is refused as an error.
- A negative own SE is refused as an error.
- A non-finite margin or own SE gives rho = -inf: that condition admits no correction.

**Reporting.** The baseline slack and the incremental cost are always reported separately.

**Why.** This is the frozen source reserve rule, used unchanged so that the source's failure can be studied on its own terms.

**The baseline's uncertainty is not treated as known free budget.**
- The 2 x se_used deduction is itself a buffer against MODEL_SELECT noise in the witness value.
- OLD spends at most f x rho, SIMPLE at most gamma x rho, and UNCERTAINTY additionally checks the population cap with the baseline
  SE explicitly (U2, section 5).

**What it does not guarantee.**
- The source's H3 met k = 2 on MODEL_SELECT and still eroded on DEV_REPORT.
- MODEL_SELECT has been used adaptively by every earlier campaign (the witness recipe itself was chosen on it).

## 3. OLD rule (the source projection, reproduced)

**Definition.**
- Costs are the OOF costs within FIT.
- Budget per statistic = f x the smallest rho over the statistic's mapped conditions.
- f = 0.8 (`B80`, primary) and 0.5 (`B50`, secondary).
- The scale is the largest s on the 101-point grid 0, 0.01, ..., 1 such that every grid point up to s is feasible, refined by 30
  bisection steps toward the first infeasible grid point.

**Implementation.**
- **Re-implemented identically** (`cts.rules.project`, `old_rule`): the same floating-point operations as the source, without
  importing `ohn.common`'s file layout.
- `t_old_equals_ohn_project` checks bitwise equality of the chosen scale, the costs and the whole grid curve against
  `ohn.protect.project`. It covers 3 invented examples x 2 levels with interior scales, plus the s = 1, near-zero and negative-slack edges.

**Notes.**
- OLD selects a continuous scale, not a ladder point, so its release is a separate release specification.
- Quirk kept for exact reproduction: when rho <= 0, f x rho >= rho, so the budget is looser than the slack. It is flagged
  `OLD_NONPOSITIVE_SLACK`.

**Known weakness (the reason for this campaign).**
- In the source, the projection's men's-guard transfer error was +0.0016 to +0.0029 on seed 0 (30-54% of slack 0.0054). On seed 1
  it was -0.0008 to -0.0012.
- The OOF costs come from fold pipelines on a cross-fitted base, not from the deployed release.

## 4. SIMPLE rule (conservative scalar shrinkage on GUARD)

**Definition.**
- Accept s iff for every one of the 8 conditions: Delta_GUARD,stat(c)(s) <= b_c.
- b_c = gamma x rho_c if rho_c > 0, else rho_c. The threshold is never looser than the slack.
- The ladder is S = {0, 0.1, ..., 1.0}, tested in ascending fixed sequence. The last accepted strength is selected.
- Identity results if 0.1 fails. A non-finite cost fails.

**gamma = 0.5 (fixed now).**
1. It equals the source B50 level, the only source budget level at which the PAIR arms kept the reserve on both seeds. OLD-B50 and
   SIMPLE therefore differ only in where cost is measured (OOF within FIT vs the deployed release on GUARD) and in the strength
   set. That is the matched comparison question A needs.
2. A 50% buffer absorbs transfer errors of the size the source projection made on seed 0 (30-54% of the men's slack), except its
   largest (54%). This is disclosed.
3. It is not tuned to any outcome of this campaign.

GUARD estimates remove the fold-versus-deployment mismatch. They add sampling noise, because GUARD has about one third of the OOF people.

**What it does not guarantee.**
- It is a point rule with no uncertainty.
- Sampling noise in the GUARD costs or a GUARD-to-target bias can still produce crossings, and these are recorded.

## 5. UNCERTAINTY rule

**Definition.** For condition c, its statistic k and population g (all people; men for `men_brier`; women for `women_brier`):
- se_k(s) = max(SE_k(s), phi_k). The increment SE floor is phi_acc_loss = 1 / n_GUARD (one changed decision); phi = 0 for the
  continuous statistics.
- tau_k(s) = se_k(s) x sqrt(1 + n_GUARD,g / n_MS,g). This is the predictive SE of the increment on a fresh sample the size of
  MODEL_SELECT, with people counts per SEX group for the guards.
- **U1 (keeps the reserve on the benchmark role):** Delta_k(s) + z x tau_k(s) <= rho_c.
- **U2 (population cap, baseline uncertainty explicit):** Delta_k(s) + z x sqrt(se_k(s)^2 + se_used_c^2) <= margin_c.
- Accept s iff U1 and U2 hold for all 8 conditions. Ascending fixed sequence; the last accepted strength is selected; identity
  results if 0.1 fails.
- **alpha = 0.05 one-sided; z = 1.644853626951472.**

U2 is implied by U1 whenever z <= k = 2, because sqrt(a^2 + b^2) <= a + b and tau >= se. It is still checked explicitly (test
`t_uncertainty_U1_implies_U2`). It shows how the baseline's MODEL_SELECT SE enters: whatever U1 spends, the cap keeps a
one-sided 5% buffer for the witness's own sampling error.

**Error event.**
- Fix everything fitted on FIT (witness, routing, direction) and the MODEL_SELECT baseline (m_c, se_used_c, rho_c).
- delta_k(s) is the population mean increment. mu_c(s) = mu_c(0) + delta_k(s) is the population value of the corrected release.
- A nonzero strength s is **infeasible** iff, for some c, delta_k(s) > rho_c or mu_c(s) > cap_c.
- The error event is **E = {the rule selects a nonzero strength that is infeasible}**.

**Assumptions.**
- A1 *Exchangeable households.* GUARD and MODEL_SELECT households are independent draws from one population, independent of the
  FIT fits: no transfer bias between GUARD and the target people. For **FROZEN routing this fails**, because the admitted T_TREE
  saw GUARD and CHECK labels; the statement then does not apply.
- A2 *Approximately normal clustered means.* (Delta_k(s) - delta_k(s)) / SE_k(s) is asymptotically N(0, 1), with consistent clustered
  SEs (many households, bounded household size). se_used_c is at least the true SE of the witness value.
- A3 *Fixed thresholds.* rho, m, se_used, the ladder and z are fixed before GUARD is examined, and GUARD is used for nothing else.
- A4 *Fixed direction.* No GUARD or MODEL_SELECT label enters any fit.

**Claim (asymptotic): P(E) <= alpha + o(1), without assuming monotone costs.**
1. *Component tests.*
   - Under delta_k(s) > rho_c, accepting needs Delta + z tau <= rho. Hence (Delta - delta) / SE <= -z tau / SE <= -z, which has
     probability at most alpha + o(1), since tau >= SE.
   - Under mu_c(s) > cap_c, accepting needs (V_c - mu_c(0)) + (Delta - delta) < -z sqrt(se^2 + se_used^2). The left side has
     asymptotic variance <= se^2 + se_used^2, because MODEL_SELECT and GUARD are independent (A1).
2. *Intersection-union (Berger 1982).* "s infeasible" is a union of 16 component nulls (8 conditions x 2 parts). Accepting s
   requires rejecting every component, including the true one. So the level is alpha with no Bonferroni adjustment. The price is
   power (all 16 must pass), not error.
3. *Fixed sequence.* Let s_j* be the first infeasible nonzero ladder strength.
   - The procedure can select s >= s_j* only if it accepts s_j*, which has probability <= alpha.
   - Any s < s_j* is feasible by the definition of j*.
   - So P(E) <= P(accept s_j*) <= alpha, without using monotonicity.
   - A simulation with a non-monotone true curve and one infeasible point on the boundary gives 0.0007
     (`t_fixed_sequence_fwer_simulation`).

**Predictive part.**
- The inflation sqrt(1 + n_G / n_MS) makes each single acceptance also cover the realized increment on MODEL_SELECT: for fixed s and
  c, P(accept and realized MODEL_SELECT increment > rho_c) <= alpha + o(1).
- This is a per-condition statement. No family-wise statement is made for realized values across conditions or for the selected strength.

**What it does NOT guarantee.**
- No finite-sample guarantee. A normal (or bootstrap) upper bound is not a distribution-free certificate.
- Nothing about transfer bias: FROZEN routing, a shift between the MODEL_FIT and MODEL_SELECT roles, or MODEL_SELECT's adaptive history.
- Nothing about refits: the statement is conditional on the fitted direction.
- Nothing across the many applications (4 rules x arms x routings x deployments x partitions) or the recipe selection.
- The corrected release's own se_used on MODEL_SELECT can differ from the witness's, so the realized reserve check uses its own value.
- People counts stand in for household counts in the SE ratio, assuming household sizes are alike across roles.

**Recorded diagnostics (never gating)**, per strength, statistic and condition:
- the GUARD estimate and its SE;
- the realized CHECK and MODEL_SELECT increments and SEs, and the OOF increment;
- signed error (realized - predicted; positive = cost under-predicted), absolute error, and error / stat slack (undefined when the
  slack is <= 0);
- crossings at every **accepted** strength: realized > rho_c (reserve) and realized > margin_c (point cap) (`accepted_crossings`);
- identity selections;
- correction magnitude: mean |s x centred Delta| over people and classes, and the decision flip rate against the witness
  (label-free; `correction_magnitude`);
- the realized MODEL_SELECT reserve with the corrected release's own se_used;
- coverage of the one-sided predictive bound on CHECK, i.e. how often realized CHECK <= Delta_G + z x SE_G x
  sqrt(1 + n_G / n_CHECK), over strengths and statistics.

## 6. Rule evaluation metrics (questions A and B)

**Per rule and partition (and deployment):**
- the selected strength;
- realized slack use on CHECK and on MODEL_SELECT (the maximum over the 8 conditions of realized cost / rho_c, and per condition);
- the reserve-crossing indicator (any realized cost > rho_c);
- the point-cap-crossing indicator (any realized cost > margin_c; on MODEL_SELECT this is exactly a cap failure of the corrected
  release on that condition);
- realized protection from the reader grid (computed by the reader investigator).

**"Improved correction-cost prediction on excluded households" (precise).** On one partition:
- Errors: e_X(s, k) = Delta_CHECK,k(s) - Delta_X,k(s), for X in {GUARD, OOF}, over the 10 nonzero ladder strengths and the 5 statistics.
- Score: MAE_rel(X) = mean over the (s, k) cells with stat slack > 0 of |e_X| / stat slack_k.
- Difference: D = MAE_rel(OOF) - MAE_rel(GUARD). Positive favours GUARD.
- Uncertainty: a paired household bootstrap (B = 2000, seed tag `CTS1_transfer_boot`). Households are resampled independently
  within the FIT (OOF) people, GUARD and CHECK, conditional on the fitted pipelines. D is recomputed on every replicate.
- `PREDICTION_IMPROVED` iff D > 0 and the 5% bootstrap quantile of D > 0.
- Also reported: the bootstrap SE, the 95% quantile, the count of cells where |e_GUARD| < |e_OOF|, mean signed errors per
  statistic (bias), and the men's-guard signed error by strength. The men's guard was the source's failure.

**Language.**
- The bootstrap does not include refit variability.
- There are only two diagnostic and two repair partitions, with overlapping training sets (D: about 36% shared; R: about 56%). Results
  are stated descriptively ("on both diagnostic partitions"), never as rates.

## 7. Claim ladder (generic; evidence fixed before fitting)

Rungs 1-5 are claimable only in order. Higher rungs are still computed and are labelled
`MET_BUT_NOT_CLAIMABLE_LOWER_RUNG_UNMET`. Implemented in `claim_ladder`.

1. **IMPROVED_COST_PREDICTION.** `PREDICTION_IMPROVED` (section 6) on D0, D1, R0 and R1. Evaluation uses the selected recipe's
   routing and deployment; the other combinations are reported. The D results guided the recipe choice, so the repair panels
   (opened after REPAIR_FREEZE) must replicate them.
2. **UTILITY_RETAINED_AT_NONZERO_CORRECTION.** For the selected recipe, on both R partitions:
   - the selected strength is > 0;
   - on MODEL_SELECT, all 8 occupation-side conditions keep the reserve with the corrected release's own se_used;
   - the complete release (H3 income) passes its 12 clauses and 2 guards at the point;
   - there is no reserve crossing on CHECK.
3. **INCREMENTAL_PROTECTION_BEYOND_MATCHED_CONTROLS.** The pair benefit is `STABLE_POSITIVE` under the source screen (8 cells = R0/R1
   x 2 repetitions x 2 reader seeds; every cell > 0 and every partition mean - 2 sd_total > 0) against each matched control:
   - CONT: the same pipeline with critic weight 0, strength chosen by the same rule on its own GUARD costs;
   - LOC: same rule; required only for a PAIR recipe;
   - COMP: the label-free code at matched GUARD utilization, max_c Delta_G,c / rho_c;
   - W: the unchanged witness.
4. **STABLE_FRONTIER_IMPROVEMENT.** On both R partitions:
   - Take the candidate's frontier, realized MODEL_SELECT utilization u(s) against pair AUC, at the reader-evaluated strengths.
   - At every point with u <= 1 (at least two strengths), the benefit against every matched control's frontier, interpolated linearly
     at the same u, is > 0 in every cell.
   - The recipe's rule is not beaten by another frozen rule on the same directions (`compare_rules`, section 9).
5. **ORIGINAL_BENCHMARK_COMPETITIVENESS.** The complete release (R0 seed 0, R1 seed 1) passes:
   - all 24 clauses and 4 guards on MODEL_SELECT;
   - the old per-cell recipient gate against every required comparator (AGG1 reported beside it);
   - `STABLE_POSITIVE` against all six required comparators. This is "positive superiority".

   **Separately:** the 0.020 target label against NEURAL_RAW and H_LJ (`DEV_GRID_TARGET_STABLE` / `_POINT_ONLY` / `_NOT_MET`) and
   the reserve-qualified status (section 11).
6. **JOINT_DESIGN_ADVANTAGE: NOT_CLAIMABLE.** Track F optimizes occupation only: it is a coalition-aware occupation correction,
   never a joint two-output design. **NOVELTY: NOT_CLAIMED**, and it is assessed separately. Neither follows from rungs 1-5.

## 8. Development-recipe selection (at most one)

**Recipe.** A recipe is (routing, arm, deployment, rule), with arm PAIR or LOC. CONT is a control and never a recipe.

**When.** It is chosen once, after the D0 / D1 diagnosis and before any C0 / C1 label is opened (before REPAIR_FREEZE).

**Gates (both D partitions):**
- G1: nonzero selected strength;
- G2: no realized reserve crossing on D CHECK or on MODEL_SELECT at the selected strength;
- G3: the complete release passes the 12 clauses and 2 guards of its seed at the point on MODEL_SELECT;
- G4: a development reader-grid entry against the matched CONT control exists.

**Score.** The minimum over D0 and D1 of (partition mean - 2 x sd) of the pair benefit against the matched CONT control. Here sd =
sqrt(variance of the partition's 4 cells + mean squared cell SE).
- The score rewards a benefit that repeats on both partitions.
- One exceptionally low AUC on one partition cannot win (test `t_select_recipe_tie_breaks`).

**Near ties and tie-breaks.** The near-tie set is score >= best - 0.005. Inside it, in order:
1. the larger minimum realized MODEL_SELECT reserve ratio, (rho_c - realized cost_c) / se_used_c, over both D partitions and the 8
   conditions, compared leximin (stable task-quality margin);
2. REFIT routing before FROZEN (no GUARD or CHECK label exposure, so A1 can hold);
3. full deployment before ensemble (simpler);
4. rule order SIMPLE, UNCERTAINTY, OLD-B50, OLD-B80 (the GUARD-based rules this campaign asks about; simpler first);
5. LOC before PAIR (fewer inputs);
6. earliest ledger entry.

If no recipe passes the gates, the outcome is `NO_RECIPE`.

**No switching.** Every frozen rule is still evaluated on the repair CHECK panels, but the selected recipe is never switched to the
winner there.

**Sensitivity alias D0B.** D0B uses the same households as D0, with the B2_SINGLE witness recipe. It is a bounded sensitivity check
of the witness recipe:
- It is **not** a third D partition.
- It never enters the gates, the score or the tie-breaks of recipe selection.
- It is never pooled with D0 or D1.
- It is reported only as a sensitivity result.

## 9. Repair comparison plan (R0 with panel C0, R1 with panel C1)

**Primary endpoints** (per R partition, for the selected recipe's fitted directions; every rule evaluated):
- (E1) transfer: errors and predictive coverage on CHECK, across the ladder and at each rule's selected strength;
- (E2) realized utility and reserve on MODEL_SELECT (8 occupation-side conditions; the complete release's 28);
- (E3) protection on the reader grid, against matched controls and required comparators.

**Matched comparisons.**
- Rules are compared on the same fitted directions.
- At a common strength, costs and releases do not depend on the rule; rules differ only in which strength they pick.
- Across the observed frontier: reader grids at every rule-selected strength plus {0.2, 0.4, 0.6, 0.8, 1.0}, as the release budget
  allows. Label-free and cost frontiers are reported at all 11 strengths.

**A stricter rule does not improve the method merely by shrinking every correction to zero** (`compare_rules`). For rule A against rule B:
- `A_SAFER_AVOIDS_CROSSING` (A picks a smaller nonzero strength, B crosses, A does not) and `A_STRONGER_WITHOUT_CROSSING` count as
  improvements.
- `A_IDENTITY_NO_IMPROVEMENT`, `A_WEAKER_NO_GAIN` and `A_CROSSES` never count.
- The protection lost by a safer choice is always reported.
- "Improves" requires the category on both R partitions.

**Repeatable recovery benefit.**
- It is judged on both R partitions x both reader seeds x both household repetitions (8 cells).
- Inputs are paired candidate-minus-control differences per cell, with household-bootstrap SEs (SCORE households, readers fixed), and
  the between-cell spread (source screen, z = 2, `stability_summary`).
- Winners within 0.010 on VALIDATION are flagged as near ties, and fixed-recipe sensitivity is reported beside the frozen winner.

**Stable frontier improvement.** As in rung 4.

**D and R are kept separate.**
- D results choose the recipe and are reported as diagnosis.
- R results come only after REPAIR_FREEZE: rules, recipe and code are frozen, and the C0 / C1 labels are then opened.
- The two are never pooled into extra seeds or extra partitions.
- R results never change the recipe, and D results never count as R evidence.

## 10. Screen comparison (question C) and the operating-characteristic study

Specified before any new method outcome. The screens apply to one (candidate, comparator, view), where a cell difference is the
candidate SCORE AUC minus the comparator's.

**(a) Old per-cell gate** (`per_cell_gate`).
- It passes iff every one of the 8 registered cells is <= +0.010 (inclusive), for the income and occupation views, against every
  required comparator.
- A non-finite cell gives `INCOMPLETE`.

**(b) Aggregate paired rule AGG1** (`aggregate_paired_screen`).
- dbar = the mean of all cells (both seeds, repetitions and reader seeds).
- For each repetition R: y_R = the mean of its cells (over seeds and reader seeds), and se_R = the mean of their bootstrap SEs.
  The mean of SEs is an upper bound, because the cells of a repetition share SCORE households.
- v_samp = sum_R se_R^2 / n_R^2.
- phi_emp = [sum_R (y_R - dbar)^2 / (n_R - 1)] / mean_R se_R^2.
- **phi = max(1, phi_prior(view), phi_emp).**
- **U = dbar + z x sqrt(phi x v_samp)**, with z = 1.644853626951472 (one-sided 0.05).
- **Pass iff U <= +0.010.**
- The configuration passes the screen iff every required comparator and both views pass (intersection-union: the error of admitting
  a configuration that is worse than some comparator by more than the allowance stays at about alpha, with no Bonferroni; power
  falls with the number of comparisons).

**Exact alias.** If the candidate's view wire is bitwise equal to the comparator's on both seeds, the difference is exactly 0 by
construction: `EXACT_ALIAS`, pass. An alias whose cached cells are not all 0 is an error.

**phi_prior.** It uses the **gross** construction, one construction for all views:
- phi_prior = mean over the distinct-wire pairs of the frozen source grid of sd_split_gross^2, divided by the mean over the same pairs
  of rms_cell_se^2.
- sd_split_gross^2 = the mean over seeds of (x_{s,R1} - x_{s,R2})^2 / 2, where x_{s,R} is the mean over reader seeds. It includes
  the reader-seed share, so it equals the source's net value + var_k / 2.
- Why gross: AGG1's variance is that of a mean of repetition means that contain reader-seed noise, and phi_emp is computed on that
  same scale.
- Registered values, computed label-free by the reader investigator:

  | View | phi_prior | Wire groups / pairs |
  |---|---|---|
  | income | **5.636511599640258** | 9 / 36 |
  | occupation | **0.7781545862915747** (the floor 1 binds) | 19 / 171 |
  | pair | **1.0867113014695036** | 26 / 325 |

- The net values (5.6152, 0.6958, 0.8286) are recorded in RULES.json and not used.

**Why this form.**
- The multiplicative-dispersion model is the "multiplicative heterogeneity" model of meta-analysis: across-split variance =
  phi x within-split sampling variance.
- It scales with how different the two releases are. The bootstrap SE of a near-identical pair is small, so near-aliases are not
  held to the noise of very different wires.
- With two splits, phi_emp has about 1 degree of freedom, so it is floored by the frozen source grid's ratio.
- Treating the two seeds' cells of one repetition as fully dependent (they share the household split) keeps the bound conservative.

**Recorded, never gating.**
- Per-seed bounds U_s, and the flag `AGG_SEED_EDGE` when some U_s > 0.010 while the pooled screen passes.
- phi_emp.
- The per-cell gate, always reported beside it.

**What a pass means.**
- An AGG1 pass is **permission for further development**, not evidence of protection.
- The old gate stays the registered gate wherever the source rules apply.
- The source campaign's labels are never rewritten.

**Operating-characteristic study (run by the reader investigator).** The laws are below. The results are simulated rejection rates
under these laws only, with no universal error rates.

- **L0a: exact alias under shared fitting randomness.**
  - Candidate = comparator wire (bitwise), read with cached readers. Every cell is 0.
  - Expected: the gate passes; AGG1 gives `EXACT_ALIAS`, pass.
  - Run it on every frozen reference that has an alias: an H3-anchored release's income view against SUCC; candidate income views
    against each other.
- **L0b: identical release under separately fitted readers.**
  - The same frozen release is the "candidate" and the "comparator", but the candidate's readers are refitted with new reader
    seeds (k2, k3 instead of k0, k1) on the same FIT / VALIDATION / SCORE splits, with no cache sharing.
  - The true difference is 0, and the noise is reader-fitting and winner-selection noise only.
  - Use at least 4 frozen releases: SUCC, H_LJ, NEURAL_RAW and a witness, for the income, occupation and pair views.
  - Report the pass rates of both screens and the observed phi_emp.
- **L1: cell-level synthetic law** (exact truth; primary; `synthetic_cells`, `oc_rates`).
  - d_{s,R,k} = mu + A_{s,R} + B_{s,R,k}.
  - A_{.,R} ~ N(0, sd_A^2), jointly normal across the two seeds with correlation rho_seed (shared household splits), and
    independent across R.
  - B ~ N(0, sd_k^2), independent.
  - Every reported cell SE = se_cell.
  - Per view, use the reader investigator's frozen-grid components: sd_A = the NET split SD, sd_k = the reader-seed SD, se_cell = the
    rms cell SE. Income: 0.021237, 0.001849, 0.008962. Occupation and pair: the investigator's own values from the same computation.
  - rho_seed in {0, 0.76}. 0.76 is the source's observed cross-seed correlation of the R1 - R2 contrast.
  - mu in {-0.030, -0.020, -0.010, 0, +0.005, +0.010, +0.020}. A positive mu means the candidate is more readable (worse).
  - At least 20,000 simulations per setting.
  - Multiplicity version: release-level effects (the candidate's a^X shared across its comparisons, each comparator's own a^Y, with
    per-release SD = sd_A / sqrt 2 and sd_k / sqrt 2), 6 required comparators x 2 views (income, occupation) x 2 seeds. Report the
    probability of passing all comparisons at mu = 0 and mu = -0.010 for every comparison, and the probability of passing everything
    while some comparison has mu = +0.010 or +0.020 (the IUT error).
  - Also n_R = 3, to show what a third repetition buys.
- **L2: release-level semi-synthetic law** (a realism check; the truth is approximate).
  - Take a frozen comparator view p (the occupation views of SUCC, H_LJ and NEURAL_RAW; the income view of H_LJ).
  - Make the candidate log p'(x) = log p(x) + eta x (2 SEX - 1) x v, renormalized, where v = e_1 - (1/K) 1 (the first class
    against the rest, in centred-logit space). This is a SEX-dependent score shift on MODEL_SELECT people only, used for this
    simulation only.
  - Calibrate eta by bisection so that the fixed reader LOG_LR (fit on R1 FIT, scored on R1 SCORE, reader seed k0) gains
    Delta_AUC in {0.005, 0.010, 0.020} over p. Delta = 0 is p itself, read by separately fitted readers (L0b).
  - Then run the full grid (both repetitions, both reader seeds, all registered readers with winner selection) for both screens.
  - Report the slate's realized mean shift (truth approximate) next to each rate.
  - Ten extra household splits with the fixed recipe estimate the slate-level true shift.

**Reporting.**
- Each table gives the pass rate of each screen per law and parameter, with a Monte Carlo SE.
- Each row states the law and its parameters.
- The old gate is always shown next to AGG1.

**Invented-data checks of the machinery** (law L1, income components):
- The gate's false-failure rate at mu = 0 with independent seeds is 0.807 (the source published 0.806).
- At mu = +0.010 with rho_seed = 0.76, AGG1 admits 0.032 (<= alpha), while the per-cell gate admits 0.131.
- At mu = 0, AGG1 passes 0.075, against 0.193 for the gate.

These are simulation properties of the invented law, not findings about any release.

## 11. Reserve-qualified frontier vs the original benchmark

**Original-benchmark comparison (unchanged).**
- The candidate is compared with every original point-eligible strong reference: NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC and TWJ.
- References are eligible by point feasibility alone (24 + 4) and need no reserve.
- In the source, only candidates had to meet the reserve: an unequal requirement.

**Reserve-qualified comparison** (`reserve_qualified_frontier`).
- The same whole-release reserve (all 28 conditions, k = 2, H_LJ floor) applies **identically** to candidates, controls and references.
- A point qualifies iff its minimum ratio is >= 2 on both seeds.
- The Pareto frontier of qualified points uses mean pair AUC (lower is better) and minimum ratio (higher is better).
- Historical references that fail are shown, never dropped. From the source: SUCC, H_LJ, B2TWIN and TWJ at 0.321 / 0.624;
  RETIRED_B3CTL at 0.321 / 0.349; NEURAL_RAW at 1.947 / 1.558. All fail.
- **A positive reserve-qualified result is not superiority over every original reference.** Both comparisons are always reported
  together.

**Known before any fit (stated so nobody reads it as a finding).**
- The H3 income wire and its cached reader cells are unchanged in this campaign.
- So every H3-anchored release fails the old per-cell gate through its income view: +0.0309 against H_LJ and +0.0147 against TWJ
  in S0R1k0 / S0R1k1 (published).
- The reader investigator computes AGG1 for the frozen H3 income view against each required comparator before any new outcome and
  records it beside this note.

## 12. Constants

| Constant | Value |
|---|---|
| Ladder | 0, 0.1, ..., 1.0 (j / 10) |
| Reserve k; floor households | 2; 5306 (H_LJ MODEL_SELECT SEs, OHN1) |
| OLD levels; grid; bisection | B80 = 0.8 (primary), B50 = 0.5; 101; 30 |
| SIMPLE gamma | 0.5 |
| UNCERTAINTY alpha; z | 0.05 one-sided; 1.644853626951472 |
| Increment SE floors | acc_loss 1 / n_GUARD; others 0 |
| Baseline SE floor | H_LJ SE x sqrt(5306 / households); a zero or missing floor is refused |
| Prediction comparison | B = 2000 household bootstrap, tag CTS1_transfer_boot, 5% quantile |
| AGG1 z; phi_prior (income / occupation / pair) | 1.644853626951472; 5.636511599640258 / 0.7781545862915747 / 1.0867113014695036 |
| Stability z; near-tie width | 2.0; 0.005 |
| Recipient allowance; target | 0.010; 0.020 (separate) |

## 13. Open points for the coordinator

1. **n_MS by SEX group** is needed for tau. It is a count from the witness's MODEL_SELECT evaluation, so it should be added to the
   evaluator's record.
2. **Rules on FROZEN routing.** UNCERTAINTY's assumption A1 does not hold there. Its results should be labelled
   `A1_NOT_SUPPORTED` and kept out of the claim language.
3. **COMP matching for rung 3** is defined on GUARD utilization, not on OOF as in the source. Confirm that the correction
   investigator fits COMP accordingly.
