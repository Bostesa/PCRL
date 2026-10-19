# Statistical rules of the controlled training-size campaign (TSZ1; statistical reviewer)

**Status.** Writing started 2026-10-19T00:04:26Z (`date -u`; first draft of this file saved by 00:06:40Z); the completion time is
recorded at the end of this file. Written before any new real-label computation, fit or outcome of this campaign. I read no label, ran no real-data fit and called no loader that reads labels.
- Machine-readable constants: `RULES.json` (schema `TSZ1-rules-v1`). Implementation: `tsz/rules.py` (pure functions; reads no file
  except this RULES.json and, through the imported `ohn.rules.reserve`, the source rule file OHN1 `RULES.json`, SHA-256
  e18f1c68f4646ce9d1395960ab1a364a80c12541b815a55e0ba06b65074cbbb3). Invented-data tests: `python3 -m tsz.test_rules`.
- Inputs read: the coordinator's DESIGN_BRIEF.md; `ohn/rules.py`, the OHN1 RULES.json and SELECTION_RULE.md;
  `results/pcrl_occupation_headroom_overnight_v1/WITNESS_SELECTION.json`; the CTS1 RULES.md, `cts/rules.py`, RESEARCH_DECISION.md,
  REVIEW_TRAINING_SIZE_DIAGNOSTIC.md, `diagnosis/TRAINING_SIZE_BET.json` and `TRAINING_SIZE_CURVE.json`; `ohn/devreaders.py`,
  `cts/devreaders.py`, `ffv/readers.py` (seed dependence of the reader recipes); `ohn/protect.py` (correction recipe).
- **Disclosure (required).** The published source witness margins were known when every constant below was chosen:
  A1_PO_16 (100% of MODEL_FIT) met the full reserve, minimum occupation-side ratio 4.30 (seed 1, men's guard: value -0.00229,
  margin 0.00729, se_used 0.00170), then 4.88 (seed 0 men's guard), 5.57 (seed 1 occupation accuracy); the CTS1 refits at
  42-70% and their paired size differences (e.g. P70 - D0 men's Brier -0.0024, SE 0.0010) were also known.

**What every rule here is.** A development rule on previously and adaptively used records (MODEL_SELECT has been used by every
earlier campaign; nothing here is confirmation). No rule changes an original limit. "The two-SE reserve is a development heuristic,
not a new scientific cap or population guarantee." Every uncertainty statement is asymptotic and model-based (household-clustered
normal approximations), never a finite-sample or distribution-free guarantee.

Data context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records. Income (2 classes) and occupation (5
groups) are the useful tasks, predicted from six raw fields. SEX (sex0 = men, SEX code 1; sex1 = women, SEX code 2) is a training
label on MODEL_FIT and an evaluation label on MODEL_SELECT, never a runtime input. "Recovery" always means SEX recovery by the declared
development readers only.

## 0. Original limits (unchanged)

| Condition | Limit | Count |
|---|---|---|
| Accuracy loss, NLL excess (nats), summed multiclass Brier excess vs U0_S<v> and Ucal_S<v>, income and occupation | 0.010, 0.010, 0.005 (inclusive) | 24 clauses = 2 tasks x 2 references x 3 metrics x 2 source versions |
| Within-SEX occupation summed-Brier excess vs U0_S<v>, men and women | 0.005 each (inclusive) | 4 guards = 2 groups x 2 versions |
| Recipient non-worsening: single-output SEX-recovery SCORE AUC minus the comparator's, income and occupation views | <= +0.010 every cell | per required comparator |
| Material-benefit target (pair AUC, vs NEURAL_RAW and H_LJ) | 0.020 | separate from positive superiority; never a gate |

Version v in {0, 1} = source seed v: its references U0_S<v> / Ucal_S<v>, its H3 income S<v>, its H_LJ floor and its procedure seeds.
**The two versions share every training row; they are procedure repeats, never independent replications.**

## 1. Reserve and floors (exact source rule, reused unchanged)

- **Rule.** `ohn.rules.reserve` imported unchanged (OHN1 RULES.json, SHA-256 e18f1c68...). For each condition c of version v:
  margin_c = cap_c - value_c; se_used_c = max(own SE_c, floor_v(c) x sqrt(5306 / G)); the condition meets the reserve iff
  margin_c >= k x se_used_c with **k = 2** (inclusive); ratio_c = margin_c / se_used_c.
- **Own SE.** The household-clustered paired SE on MODEL_SELECT of the clause's per-person difference against the reference
  (G/(G-1) correction), or the linearized household-clustered SE of the within-SEX mean difference (guards), exactly as computed by
  the registered evaluator (`ohn.evaluate.complete_eval` via `cts.evaluate.eval_release`, identical to `ihs.evaluate`).
- **Floor.** H_LJ's MODEL_SELECT SE of the same condition and seed, copied unchanged from OHN1 (RULES.json `reserve.floor_se`,
  5,306 households). Every evaluation here uses all 5,306 MODEL_SELECT households, so the rescaling factor is exactly 1.
- **Edge conventions (OHN1, unchanged).** An own SE of 0 or below the floor uses the floor (recorded `se_source = floor`); a missing,
  non-finite or non-positive floor is refused (error); a negative own SE is refused; a non-finite value or own SE fails the condition.
- **Occupation-side reserve of version v** = the 8 conditions `occupation|{U0,Ucal}|{accuracy_loss,nll_excess,brier_excess}`,
  `guard|sex0`, `guard|sex1` of version v (OHN1 `OCC_SIDE`). Status per version: `OCC_RESERVE_MET` iff all 8 meet k = 2;
  otherwise `OCC_RESERVE_NOT_MET` with the failing conditions listed. Across versions: `OCC_RESERVE_MET_BOTH` /
  `OCC_RESERVE_ONE_VERSION` / `OCC_RESERVE_NOT_MET`.
- **Per-version ratios.** For each version separately: the 8 ratios, their minimum (`min_occ_ratio_v`), the binding condition
  and the usable slack rho_c = margin_c - 2 x se_used_c (metric units). Versions are never averaged or pooled.
- **Income conditions** (12 per version) are inherited from the frozen H3 income and are identical for every H3-anchored release.
  They are reported separately (whole-release status, `min_income_ratio_v`) and never enter the occupation-side reserve, the
  trigger or the strength rule. Known before any fit (OHN1 records): minimum income ratio 2.862 (version 1, income accuracy).
  An H3 income output whose hash differs from the admitted wire is an error.
- The evaluator must pass every one of the 28 conditions per version (ohn.rules.reserve refuses incomplete or re-capped input).

## 2. Size-table quantities (per fit, MODEL_SELECT, all 8,001 people)

A fit cell is (method, size f, ordering o, version v) with method in {A1_PO_16, TASK_REFIT (primary control), T_TREE (control's
first stage)}, f in {50, 70} x o in {O0..O3}, plus f = 100 (one fit per version, aliased to all four orderings: one fit, one result,
listed once). For each fit cell report:
1. Occupation accuracy, NLL (nats, 1e-12 floor), summed Brier, men's and women's summed Brier (absolute values).
2. The 6 occupation clauses (vs U0_S<v> and Ucal_S<v>) and the 2 guards (men's, women's summed-Brier excess vs U0_S<v>): value,
   cap, original point-cap status (value <= cap), margin in metric units (cap - value), own paired household SE, margin / own SE,
   floor SE (exact OHN1 value used), se_used, ratio = margin / se_used, reserve status (k = 2), usable slack rho.
3. Per version: point status of the 8 occupation-side conditions, occupation-side reserve status, `min_occ_ratio_v`, binding
   condition.
4. Income conditions inherited from H3: one separate block per version (identical across fits; checked).
5. Label-free training records: people, households, minimum occupation class count, collapsed levels per block, MODEL_SELECT share
   with a collapsed level per block, MODEL_SELECT share with a level unseen in the subset ("unknown"), T* (A1_PO_16) or tree
   counts (T_TREE), Stage B update count (TASK_REFIT), household overlaps between subsets.
A missing cell (e.g. a class-support failure) is reported as missing; it is never redrawn and never imputed. Every label below that
needs a missing cell returns `INCOMPLETE`.

## 3. Paired contrasts on MODEL_SELECT

### 3.1 Estimator

- Per-person loss vectors on the common MODEL_SELECT people: error indicator 1[argmax p != y] (accuracy loss), NLL, summed Brier.
- A contrast between fits A and B is the mean of the per-person difference d_i = loss_i(A) - loss_i(B) over all people (accuracy
  loss, NLL, Brier) or over one SEX group (men's Brier, women's Brier). It is reference-free (the reference cancels), so the change
  in a clause or guard equals the change in its metric.
- SE: `cluster_se` (household-clustered, G/(G-1)) for all-people metrics; `subgroup_se` (linearized household-clustered ratio of
  means) for the SEX-group metrics; both identical to `ihs.evaluate`. Double differences (section 4) use the per-person double
  difference vector with the same estimators.
- **Sign convention.** Changes are reported in loss orientation: change(a -> b) = loss(fit at b) - loss(fit at a); negative = the
  later fit is better. Labels use the **improvement** g = -change (positive = larger training set better).
- Five metrics: `acc_loss`, `nll`, `brier`, `men_brier`, `women_brier`. **Primary size endpoint (registered): the 50 -> 100
  improvement in men's summed Brier** (the guard that bound the source witness, ratio 4.30, and that left the CTS1 reduced refits
  at 42-53% with reserve slack of only +0.49 / -0.78 / +0.14 / -0.31 x 1e-3). The other four metrics and the 50 -> 70 and
  70 -> 100 contrasts are secondary and reported in full.

### 3.2 Contrasts

Per method, ordering o and version v: 50 -> 70 = S70(o) vs S50(o); 70 -> 100 = S100 vs S70(o); 50 -> 100 = S100 vs S50(o) (nested
subsets of the same ordering). At 100 the same fit enters every ordering.

### 3.3 Summary over orderings (per contrast, metric, method, version)

With ordering values x_o (o = 0..3) and paired SEs s_o:
- mean x_bar = (1/4) sum_o x_o;
- between-ordering spread SD_ord = sqrt(sum_o (x_o - x_bar)^2 / 3) (df 3), range, number of sign reversals (orderings whose sign
  differs from the mean's sign);
- evaluation SE se_eval = sqrt(mean_o s_o^2). The orderings share the MODEL_SELECT people, so their evaluation errors are
  positively correlated; Var(mean) <= (mean_o s_o)^2 <= mean_o s_o^2, so se_eval is an upper bound for the mean's evaluation SE;
- **combined SE se_comb = sqrt(se_eval^2 + SD_ord^2 / 4)**: evaluation noise plus training-draw (subset-to-subset) noise of a mean
  of four draws. SD_ord itself contains part of the evaluation noise, so this double-counts somewhat (conservative). With one fit
  (f = 100 comparisons) se_comb = its own paired SE.
- z = 2 throughout (the two-SE convention of the source; with df 3 for SD_ord a t-quantile would be larger, which is why every
  resolved label additionally requires no sign reversal across the four orderings).

**Per-version label** (improvement orientation):
- `RESOLVED_POSITIVE`: x_bar - 2 se_comb > 0 and every x_o > 0;
- `RESOLVED_NEGATIVE`: x_bar + 2 se_comb < 0 and every x_o < 0;
- `UNRESOLVED` otherwise (flag `SIGN_REVERSAL` if the orderings disagree in sign);
- `INCOMPLETE` if any ordering is missing or non-finite.
**Across versions** (never pooled): `POSITIVE_BOTH_VERSIONS` / `NEGATIVE_BOTH_VERSIONS` (both resolved in the same direction);
`VERSION_DEPENDENT` (exactly one resolved, or opposite resolved signs); `NOT_RESOLVED` (neither); `INCOMPLETE`.

### 3.4 Subset-to-subset variation (reported, never gating)

- Fit-to-fit spread at a fixed size: SD_ord and the range of each metric's value across S50(o) and across S70(o), per method and
  version; all six pairwise paired differences S_f(o) - S_f(o') with their paired SEs.
- Variance share: SD_ord^2 / (SD_ord^2 + 4 se_eval^2) per summary.
- Label-free overlaps: pairwise household overlap of S50(o), S70(o) across orderings (expected about 25% / 49% of the pool).
- Language: "orderings are four nested random draws from one finite pool; they overlap and are not independent datasets".

### 3.5 The four questions (labels)

**(a) Within-recipe training-size effect on this finite development pool** (`size_effect`): the section 3.3 labels of A1_PO_16 for
each contrast and metric. The registered summary is the primary endpoint: `SIZE_GAIN_RESOLVED` iff the 50 -> 100 men's-Brier
improvement is `POSITIVE_BOTH_VERSIONS`; `SIZE_LOSS_RESOLVED` iff `NEGATIVE_BOTH_VERSIONS`; else `SIZE_EFFECT_UNRESOLVED`
(`INCOMPLETE` if missing). Secondary labels are listed beside it, never substituted for it.

**Margin pattern** (`margin_pattern`, from occupation-side reserve statuses): let met100 = W100 meets the occupation-side reserve on
both versions; n50, n70 = number of the 8 (ordering, version) cells at 50% / 70% that meet it.
- `FULL_ONLY`: met100 and n50 < 8 (some 50% fit loses the reserve);
- `ALL_SIZES`: met100 and n50 = n70 = 8;
- `FULL_ONLY_70_RETAINS`: reported as a sub-flag of FULL_ONLY when n70 = 8;
- `NON_NESTED_PATTERN_50_RETAINS_70_LOSES`: sub-flag of FULL_ONLY when n50 = 8 but n70 < 8 (a non-monotone pattern, reported as
  such; it can only come from subset-to-subset variation);
- `FULL_NOT_REPRODUCED`: W100 fails the occupation-side reserve on some version; `INCOMPLETE`: a status is missing.

**(b) Improvement over the equally exposed control** (section 4): `RECIPE_BETTER_RESOLVED` / `RECIPE_WORSE_RESOLVED` /
`NOT_RESOLVED` per size and metric, and the control-explains rule.

**(c) Reproduction of the old full-data source package** (section 5): `REPRODUCED_BITWISE` / `REPRODUCED_WITHIN_TOLERANCE` /
`NOT_REPRODUCED`, plus `MARGIN_REPRODUCED` (W100 meets the occupation-side reserve on both versions) reported separately.

**(d) Sufficient usable margin for protection**: `MARGIN_AVAILABLE` iff the trigger (section 7) fires; else
`MARGIN_NOT_AVAILABLE` with the failing parts. The usable slack rho_c of W100 per condition and version is reported in metric
units beside it. Whether that slack suffices for a useful correction is answered only by section 8, never by this label.

### 3.6 Interpretation rules (computed, all reported; each with its scope sentence)

- **IR1** (`FULL_ONLY`): "Full-data refitting reproduces the occupation-side margin while smaller fits lose it: the previous
  reduced-training experiment did not settle full-data feasibility." If additionally `SIZE_GAIN_RESOLVED`, the size explanation is
  `SUPPORTED_ON_THIS_POOL`; otherwise IR4 applies as well.
- **IR2** (`FULL_NOT_REPRODUCED`): "The source full-data margin does not reproduce under the fixed recipe: inspect exact
  implementation and exposure differences (T*, dictionary, fold tags, training rows, library versions, encoding); no retuning." The
  trigger cannot fire.
- **IR3** (control explains the gain; section 4): "An ordinary equally exposed control shows the same training-size gain (or the
  same margin)"; reported for every metric where `CONTROL_EXPLAINS_GAIN` or `CONTROL_ALSO_MEETS_RESERVE` holds.
- **IR4** (`SIZE_EFFECT_UNRESOLVED`, or `FULL_ONLY` without a resolved primary endpoint): "Uncertainty leaves the size explanation
  unresolved."
- `ALL_SIZES`: "The margin is retained at 50%, 70% and 100% of this pool; the reduced-training loss of the CTS1 partitions is not
  explained by training size between 50% and 100% under this design." The size explanation is then `NOT_SUPPORTED`.
- **Size explanation label** (`size_explanation`): `SUPPORTED_ON_THIS_POOL` (FULL_ONLY and SIZE_GAIN_RESOLVED), `NOT_SUPPORTED`
  (ALL_SIZES, or SIZE_LOSS_RESOLVED), `NOT_APPLICABLE_FULL_NOT_REPRODUCED`, `UNRESOLVED` otherwise.
- Scope of every reading: "Do not claim a population learning curve, independent replication or proof that size caused every
  historical difference." A supported size explanation means only: on this pool, under this recipe, the larger nested training
  sets had resolved lower loss on MODEL_SELECT and the margin pattern matched.

## 4. Control comparison

- **Level.** At every (f, o, v): improvement of A1_PO_16 over TASK_REFIT (primary) and over T_TREE (secondary): g = loss(control) -
  loss(A1_PO_16), positive = recipe better; paired SE on MODEL_SELECT; summarized over orderings as in 3.3 (f = 100: one fit per
  version). Labels per size, metric: `RECIPE_BETTER_RESOLVED` (POSITIVE_BOTH_VERSIONS), `RECIPE_WORSE_RESOLVED`
  (NEGATIVE_BOTH_VERSIONS), `NOT_RESOLVED` / `VERSION_DEPENDENT`.
- **Difference in size gains (DiD).** Per contrast, ordering, version: DiD = g_A1(contrast) - g_TASK_REFIT(contrast), from the
  per-person double difference; summarized as in 3.3.
- **Control-explains rule (per metric, 50 -> 100):**
  - `CONTROL_EXPLAINS_GAIN`: the recipe's size gain is POSITIVE_BOTH_VERSIONS, the control's size gain is POSITIVE_BOTH_VERSIONS,
    and the DiD is not POSITIVE_BOTH_VERSIONS;
  - `RECIPE_SPECIFIC_SIZE_GAIN`: the recipe's size gain and the DiD are both POSITIVE_BOTH_VERSIONS;
  - `NO_RECIPE_SIZE_GAIN`: the recipe's size gain is not POSITIVE_BOTH_VERSIONS;
  - `RECIPE_GAIN_CONTROL_UNRESOLVED`: otherwise.
- **Equal reserve for controls.** The occupation-side reserve is computed for every control fit (TASK_REFIT, T_TREE, each with H3
  income), exactly as for A1_PO_16. `CONTROL_ALSO_MEETS_RESERVE` at size f iff the control meets the occupation-side reserve in every
  (ordering, version) cell at which A1_PO_16 meets it.

## 5. Reproduction check (W100_v vs the source A1_PO_16 S<v>; S100 controls vs the admitted T_TREE / TASK_REFIT of seed v)

- `REPRODUCED_BITWISE`: the float64 occupation probability arrays on MODEL_SELECT (and every other role the source stored) have
  equal SHA-256, and T* (tree counts) are equal, on both versions.
- `REPRODUCED_WITHIN_TOLERANCE`: not bitwise, but max |p_new - p_source| <= 1e-12 on every role, identical argmax decisions on every
  person and identical T*, on both versions. The cause is recorded (library build, summation order); values are reported from the
  new fit.
- `NOT_REPRODUCED`: otherwise. Then report per metric the paired contrast new - source with its SE, both fits' 8 occupation-side
  values, T*, dictionaries, fold tags, training row hashes and library versions; inspect differences; **no retuning**. The trigger
  and every table use the newly refitted W100_v regardless.
- `MARGIN_REPRODUCED` (separate): W100 meets the occupation-side reserve on both versions. A non-bitwise W100 can still reproduce the
  margin, and the label says so.

## 6. Encoding-sensitivity reading (bounded, label-free preprocessing; v = 0 only)

- Per ordering o: E_o = loss(A1_PO_16 S50(o), FIT-only dictionary) - loss(A1_PO_16 S50(o), admitted dictionary), same rows, folds
  and version 0; positive = the admitted dictionary is better. Five metrics; summarized as in 3.3 (one version only).
- `ENCODING_EFFECT_RESOLVED` (RESOLVED_POSITIVE), `ENCODING_EFFECT_RESOLVED_NEGATIVE`, `ENCODING_EFFECT_UNRESOLVED`.
- Share of the 50 -> 100 change: mean E / mean (50 -> 100 improvement, version 0), reported descriptively only when both means are
  positive; never a decomposition claim. The occupation-side reserve status of the four encoding fits is reported.
- Reading: a resolved positive E says part of the 50% shortfall comes from the FIT-only dictionary collapse (preprocessing exposure,
  no label), not from fewer training labels. One version: no procedure repeat; descriptive.

## 7. Trigger for the conditional protection stage (exact; evaluated once, at task-stage close)

`TRIGGER_FIRES` iff ALL of the following hold; otherwise `TRIGGER_DOES_NOT_FIRE` with every failing part listed:
- **T0 (completeness).** Both versions of W100 (the newly refitted 100% A1_PO_16 occupation output with the frozen H3 income of the
  same version) are evaluated by the registered evaluator on MODEL_SELECT; all 28 conditions per version present and finite; H3
  income hash equals the admitted wire.
- **T1 (original limits).** For v = 0 and v = 1: all 6 occupation point caps (accuracy loss <= 0.010, NLL excess <= 0.010, summed
  Brier excess <= 0.005, vs U0_S<v> and Ucal_S<v>) and both within-SEX guards (<= 0.005 vs U0_S<v>), inclusive: 16 conditions.
- **T2 (predefined occupation reserve).** For v = 0 and v = 1: every one of the 8 occupation-side conditions has margin >= 2 x
  max(own SE, OHN1 H_LJ floor SE of that condition and seed x sqrt(5306 / G)), computed by `ohn.rules.reserve` unchanged.
- **T3 (verified deployed implementation).** For v = 0 and v = 1: a package computed from the six raw fields reproduces the stored
  W100_v occupation release bitwise on every evaluated role, and every registered refusal check passes, as recorded by the
  independent verifier's deployment record `results/pcrl_controlled_training_size_v1/roles/verifier/DEPLOY_W100.json` with
  per-version fields `bitwise: true` and `refusals_all_pass: true` (keys "0" and "1"). The coordinator's own `tsz.deploy` check
  (`deploy/DEPLOYMENT_CHECKS_W100.json`, fields `all_bitwise` / `all_refused`) is a second, non-gating record; a disagreement
  between the two records is reported and blocks the trigger until resolved (never resolved by relaxing a check).
- Not part of the trigger: income conditions (reported separately), any smaller subset (never substituted, however good it looks),
  controls, readers, reproduction status. Nothing is relaxed after outcomes. A fired trigger is not a declaration that the
  H3-anchored service qualifies; it opens the correction stage only.

## 8. Conditional correction (only if `TRIGGER_FIRES`)

### 8.1 Arms, strengths, primary arm

- Arms (source P2 recipe, `ohn.protect.train_correction` with guard `'none'`; identical routing, epochs, task loss and effort except
  the critic term): W (unchanged W100_v), CONT (zero-critic continuation), **LOC (occupation-only critic; PRIMARY, declared now)**,
  PAIR (coalition-aware critic reading the frozen H3 income wire S<v>; SECONDARY). Release r_s = softmax(log p_W + s x Delta_arm).
- Guard `'none'` must not consume any MODEL_SELECT quantity: the `targets` argument is the fixed constant {men: 0.005, women: 0.005}
  (the original guard cap; confirmed for the correction investigator), in every outer fold and in the full pipeline, and an
  invented-data check must show theta is bitwise invariant to `targets` under guard `'none'`.
- **Strength set S = {0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}** (9 values, identity included). Reason: the proposed log-spaced
  set plus 0.2 and 0.7, because in CTS1 the cost and benefit transitions happened between 0.1 and 0.5 (minimum reserve ratio 1.78 at
  0.1 vs 0.06 at 0.5; stable benefit vs CONT from 0.3); 0.2 and 0.7 split the two widest steps in that region (0.1 to 0.3 and 0.5
  to 1.0). Costs at the extra strengths are computationally cheap (the same fitted directions, rescaled); readers are run only at
  selected strengths.
- Why LOC is primary: in CTS1 the occupation-only critic carried most of the recovery reduction (+0.032 vs W at s = 1.0) and PAIR's
  addition over LOC was not stable; LOC uses fewer inputs and is an occupation-only correction claim. PAIR cannot rescue a failed
  LOC claim: a PAIR-only qualification is reported as `SECONDARY_ARM_ONLY` (exploratory; section 11).

### 8.2 Selection information (fitting information only)

- Nested 10-fold household cross-fitting of MODEL_FIT (outer folds `household_folds(MODEL_FIT units, 10, 'TSZ1_ncv10')`). For outer
  fold k: on the other 9 folds (about 18,000 people) refit every upstream layer (FIT-only dictionary, A1_PO_16 witness with its own
  inner CV, its own 5-fold household cross-fitting base, T_TREE routing of seed v, the LOC and PAIR corrections); apply to fold k;
  per-person incremental costs against s = 0 (the fold's own witness) at every strength in S. Pool the 10 folds (all 20,000 people).
  Nested CONT is optional: only if the compute budget allows, descriptive, never gating (CONT has no selection of its own).
- Five statistics (exactly `ohn.protect.SLACK_OF`): `men_brier` (guard|sex0), `women_brier` (guard|sex1), `brier` (both Brier
  clauses), `nll` (both NLL clauses), `acc_loss` (both accuracy clauses). Estimate = pooled mean increment; SE = household-clustered
  over the 13,337 MODEL_FIT households (`cluster_se`; `subgroup_se` for the SEX groups). Training sizes of every submodel recorded.
- MODEL_SELECT never enters strength selection. CALIBRATION (no SEX) does not enter it either.
- Disclosure: "pooled cross-fitted cost estimates target reduced-size submodels and do not guarantee the deployed 100% model's
  costs"; the SE ignores refit variability across folds.

### 8.3 Selection rule (one rule, per arm and version)

- **Allowance** (fixed now, no new MODEL_SELECT information): a_j(v) = lambda x min over the conditions c mapped to statistic j of
  floor_v(c), with **lambda = 1.0** (one H_LJ-floor SE unit). Values (RULES.json `selection.allowance`):

  | Statistic | v = 0 | v = 1 | Fraction of the original cap (v = 0 / 1) |
  |---|---|---|---|
  | men_brier | 0.0018138 | 0.0016965 | 0.36 / 0.34 of 0.005 |
  | women_brier | 0.0016982 | 0.0017105 | 0.34 / 0.34 of 0.005 |
  | brier | 0.0011956 | 0.0011886 | 0.24 / 0.24 of 0.005 |
  | nll | 0.0028706 | 0.0027348 | 0.29 / 0.27 of 0.010 |
  | acc_loss | 0.0035086 | 0.0035434 | 0.35 / 0.35 of 0.010 |

  Reason: the reserve is measured in floor-SE units, so the allowance spends exactly one unit of the scale that defines it, whatever
  the witness's realized margin. A witness at ratio r keeps ratio about r - 1 - (transfer error in SE units) after a correction at
  the allowance; with the published source minimum 4.30 that leaves about 1.3 SE (about 0.0022 men's Brier) for transfer error, which
  covers the CTS1 nested men's-cost error (+0.0027, z 2.0) only partly. lambda = 2 would leave 0.3 SE and was rejected. The published
  witness margins were known when lambda was chosen (disclosed above).
- **Pass test** at a nonzero strength s: for all 5 statistics j: U_j(s) = estimate_j(s) + 2 x SE_j(s) <= a_j(v) (inclusive); a
  non-finite estimate or SE fails. Negative estimates are allowed.
- **Selected strength (fixed sequence):** s* = the largest s in S such that every nonzero strength <= s passes; identity (s* = 0) if
  the smallest nonzero strength fails. Identity is always admissible and is stored as the selected release.
- **Non-monotone estimates:** the sequence stops at the first failure; a larger strength that passes after a failure is never
  selected (flag `PASS_AFTER_FAIL`, with the strengths listed). There are no ties (S is strictly ordered).
- **Audit trail:** per strength and statistic: estimate, SE, U, allowance, U / allowance, pass; the first failing strength and its
  binding statistic; the flags; the rules SHA-256.
- No requirement that both versions select a nonzero strength. CONT is evaluated (and read) at the selected strength of the arm it
  matches (CONT@s*_LOC,v and CONT@s*_PAIR,v); it has no selection of its own. At s* = 0 every arm equals W exactly.
- Recorded, never gating: realized MODEL_SELECT increments at every strength (after selection), realized occupation-side reserve of
  the selected release with its own se_used, realized minus predicted cost per statistic, decision flip rate and correction magnitude.

## 9. Reader cells and stability labels

- **Grid (design brief section 7, frozen).** MODEL_SELECT households ordered by SHA-256('TSZ1_devgrid_<R>|<household>') for R in
  {R1, R2}, dealt round-robin into 10 parts (0-4 FIT 50%, 5-6 VALIDATION 20%, 7-9 SCORE 30%); the source 16-recipe bank; winners by
  maximum oriented VALIDATION AUC (pair candidates include the ignore-income and ignore-occupation readers); SCORE AUC; household
  bootstrap of SCORE (B = 400), weights shared by all releases of a cell (paired). The same bank and effort for every release.
- **Cells.** 8 per comparison: 2 versions x 2 repetitions x 2 reader seeds, named `S<v>R<r>k<k>`.
- **Paired benefit** per cell: b = comparator winner SCORE AUC - candidate winner SCORE AUC (pair view; positive = the candidate is
  read less). Recipient differences: candidate minus comparator, income and occupation views.
- **Seed-deterministic winners (known before any outcome).** Only 6 of the 16 recipes depend on the reader seed (ET5, ET25, MLP64,
  MLP6432, OUT_MLP, LOG_MLP; confirmed by the reader grid's invented-data selftest). When both releases' VALIDATION winners of a view
  are seed-deterministic in a (version, repetition), the k0 and k1 cells are bitwise equal: the reader seed then supplies no
  replication there. Per comparison and (v, R) the rules record `k_differs` = (b(k0) != b(k1), exact float comparison), and the
  reader code reports per cell, release and view the winner and whether it is seed-dependent.
  - **Effective distinct cells** n_eff = 4 + (number of (v, R) pairs with k_differs), between 4 and 8. A pair with equal cells is
    one distinct cell.
- **Stability label** (the source screen, z = 2, with one registered correction for duplicated cells):
  d_v = mean of version v's 4 cells (each repetition weighted equally, so a duplicated pair is that repetition's value);
  v_rep = SS / (n_eff - 2), where SS = sum over the DISTINCT cells of (b - d_v)^2 (a duplicated pair enters once);
  v_samp = mean of the 8 squared bootstrap SEs; sd_total = sqrt(v_rep + v_samp).
  When every pair differs (n_eff = 8) this is exactly the source formula (SS over 8 cells, df 6). With duplicates the source formula
  would count each duplicate twice and shrink v_rep by up to a factor 2/3 (anti-conservative); the correction removes that.
  - `STABLE_POSITIVE`: all 8 cells > 0 (strict) and d_v - 2 sd_total > 0 for both versions;
  - `POSITIVE_NOT_STABLE`: all 8 cells > 0, some version fails the bound;
  - `SIGN_CHANGES`: 8-cell mean > 0 and some cell <= 0;
  - `NOT_POSITIVE`: 8-cell mean <= 0; `INCOMPLETE`: a cell, SE or finite value missing.
  - **Reader-seed qualifier (always attached):** `READER_SEEDS_EFFECTIVE_ALL_PAIRS` (n_eff = 8),
    `READER_SEEDS_PARTLY_EFFECTIVE` (5-7), `READER_SEED_REPLICATION_ABSENT` (n_eff = 4: the label rests on 2 versions x 2
    repetitions only). "Genuinely effective reader seeds" for a label means: the grid passed the seed check below AND the
    qualifier is `READER_SEEDS_EFFECTIVE_ALL_PAIRS`. Only then may the text say the result held "across reader seeds"; otherwise it
    must say on how many of the 4 (version, repetition) pairs the reader seed changed the result. The qualifier never changes the
    label itself.
  Reported with it: the 8 cells, per-pair `k_differs`, n_eff, mean, min, max, per-version means, number of sign reversals (cells
  <= 0), sd components. The same computation at threshold 0.020 gives the target labels (`DEV_GRID_TARGET_STABLE` /
  `_POINT_ONLY` (mean > 0.020) / `_NOT_MET`).
- **Effective reader seeds (a validity check of the grid, computed before any stability label):**
  1. the 4 integer reader seeds (v, k) are distinct, recorded, and differ from every OHN1 / CTS1 grid seed (MODEL_SEED_BASE + 300 +
     10 v + k, k = 0..9);
  2. for every seed-dependent recipe (ET5, ET25, MLP64, MLP6432, OUT_MLP, LOG_MLP), every view, every release wire checked (at least
     W100_v and every selected release) and every (v, R): the SCORE scores under k0 and k1 are not bitwise equal. Any equality ->
     `READER_SEEDS_NOT_EFFECTIVE`: the grid is invalid and no stability label is computed until it is repaired (an engineering repair,
     recorded as a deviation);
  3. seed-invariant recipes (LR, HGB without early stopping or subsampling, CATI, LOOKUP) give bitwise equal k0 / k1 scores; a
     difference is flagged `NONDETERMINISTIC_RECIPE` (recorded, not fatal);
  4. per comparison, the number of (v, R) pairs whose two reader-seed cell values differ (0-4), giving n_eff and the qualifier above.
- Language: "reader recipes and overlapping splits are not independent datasets"; repetitions share MODEL_SELECT households.

## 10. Original-benchmark table vs equal-reserve table (always both, never merged)

- **Original benchmark (unchanged from the source).** Candidates (selected LOC, PAIR, the matched CONT, W100) as complete releases
  with H3 income, against every originally point-eligible strong reference: NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ.
  References are eligible by point feasibility alone (24 clauses + 4 guards from published values) and stay in the table even if
  they fail the reserve. Per row: point status (24 + 4), whole-release reserve status and occupation-side reserve status (reported,
  separately, for every candidate and reference), old per-cell recipient gate (<= 0.010 every cell, both views, every required
  comparator), pair-benefit stability label against each comparator, 0.020 target label vs NEURAL_RAW and H_LJ.
- **Equal-reserve comparison.** The same reserve (all 28 conditions, k = 2, H_LJ floor; and separately the 16 occupation-side
  conditions) applied identically to every method: candidates, controls (W, CONT, TASK_REFIT, T_TREE) and references. A point is
  qualified iff its minimum ratio >= 2 on both versions. Pareto frontier of qualified points (lower mean pair AUC, higher minimum
  ratio). Failing references are shown, never dropped. A positive equal-reserve result is not superiority over every original
  reference.
- Known before any fit: the CTS1 equal application recorded every required reference as failing the whole-release reserve; H3's
  income view failed the old per-cell gate on the CTS1 grid (new splits here, so the outcome is not literally known).

## 11. Claim labels and their exact scope (kept separate; each computed by `tsz.rules.claim_labels`)

| Claim | Label if met | Exact condition | Scope sentence |
|---|---|---|---|
| C1 task-quality margin restored by training size | `MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL` | size_explanation = SUPPORTED_ON_THIS_POOL | within the fixed recipe, this MODEL_FIT pool, adaptively used MODEL_SELECT; not a population learning curve; not causal for every historical difference |
| C2 nonzero correction preserves utility (per arm; **used by the closure rule**) | `UTILITY_PRESERVED_BOTH_VERSIONS` / `_ONE_VERSION` | for each version with s* > 0: the complete release passes all 12 clauses + 2 guards at the point and keeps the occupation-side reserve with its own se_used on MODEL_SELECT; any nonzero selected release that fails -> `UTILITY_NOT_PRESERVED`; no nonzero selection -> `NO_NONZERO_SELECTION` | development utility on MODEL_SELECT; costs selected on cross-fitted 90% submodels; the reserve part is a development heuristic |
| C2p original limits only (per arm; reported beside C2, never used for closure) | `UTILITY_POINT_PRESERVED_BOTH_VERSIONS` / `_ONE_VERSION` | the same, with the original point limits only (12 clauses + 2 guards per version, i.e. 24 + 4 over both versions), no reserve; failure -> `UTILITY_POINT_NOT_PRESERVED`; no nonzero selection -> `NO_NONZERO_SELECTION` | original limits on MODEL_SELECT |
| C3 recovery falls beyond continuation | `RECOVERY_BELOW_CONTINUATION` | pair benefit vs CONT at the same strength STABLE_POSITIVE (8 cells) | declared development readers only |
| C3w recovery falls beyond the unchanged witness | `RECOVERY_BELOW_WITNESS` | pair benefit vs W100 STABLE_POSITIVE | same |
| C4 recovery falls beyond the local correction (PAIR only) | `RECOVERY_BELOW_LOCAL_CORRECTION` | PAIR_sel vs LOC at LOC's own selected strength STABLE_POSITIVE (matched-strength LOC reported descriptively) | same; coalition-aware occupation correction, not a joint design |
| C5 strongest-reference competitiveness | `COMPETITIVE_WITH_ALL_REQUIRED` | 24 + 4 at the point, old per-cell recipient gate vs all six, STABLE_POSITIVE vs all six | original benchmark; reserve reported separately |
| C6 original 0.020 material criterion | `DEV_GRID_TARGET_STABLE` / `_POINT_ONLY` / `_NOT_MET` | stability at threshold 0.020 vs NEURAL_RAW and H_LJ | distinct from positive superiority; never implied by C5 |
| C7 joint-design advantage | `NOT_CLAIMABLE` | fixed | "Occupation-only optimization is not a new two-output joint method." |
| C8 novelty | `NOT_CLAIMED` | fixed | assessed separately |

- For every selected release (identity included) and both versions, the point status (12 + 2, original limits) and the
  occupation-side reserve status are reported separately, so the original limits and the reserve heuristic are always visible.
- A protection claim (C3, C3w, C4, C5) is reported as `MET_WITHOUT_UTILITY` (never claimable) unless C2 is
  `UTILITY_PRESERVED_BOTH_VERSIONS` for that arm. A version that selects identity contributes exact-alias zero cells, so an
  8-cell STABLE_POSITIVE is then impossible; its per-version cells are reported descriptively.
- **Arm qualification:** an arm QUALIFIES iff C2 = `UTILITY_PRESERVED_BOTH_VERSIONS` and it improves recovery beyond its strongest
  relevant controls: LOC: C3 and C3w; PAIR: C3, C3w and C4.
- **Closure rule:** "If no selected nonzero release preserves utility and improves recovery beyond its strongest relevant
  controls, close this correction recipe": `CORRECTION_RECIPE_CLOSED` iff neither arm qualifies. "Preserves utility" here is C2
  (point limits AND the occupation-side reserve), not C2p. If only PAIR qualifies the outcome
  is `SECONDARY_ARM_ONLY` (exploratory; two arms were tried; the primary claim failed). If LOC qualifies: `PRIMARY_ARM_QUALIFIES`.
  If the trigger did not fire: `NOT_RUN_TRIGGER_NOT_FIRED`.
- "Lower recovery for declared readers is not arbitrary-reader protection."

## 12. What is NOT claimable

- A population learning curve; independent replication (versions share rows; orderings are nested draws from one pool; reader
  recipes and overlapping splits are not independent datasets); proof that training size caused every historical difference.
- Confirmation of anything (MODEL_SELECT is adaptively used); any statement about DEV_REPORT or the attack roles (closed).
- That the two-SE reserve is a scientific cap or a population guarantee; that cross-fitted costs guarantee the deployed model's
  costs; that a fired trigger means the H3-anchored service qualifies.
- That a smaller subset is a witness, or that the best-looking smaller fit says anything about the trigger.
- Protection against arbitrary readers; a joint-design advantage; novelty.
- Any claim from a release whose utility fails, or from PAIR in place of a failed primary LOC claim.

## 13. Constants (RULES.json)

| Constant | Value |
|---|---|
| Reserve k; floor households; floor SEs | 2.0; 5306; OHN1 H_LJ MODEL_SELECT SEs (copied) |
| Contrast z; sign-reversal requirement | 2.0; every ordering same sign for a resolved label |
| Primary size endpoint | 50 -> 100 improvement in men's summed Brier, both versions |
| Reproduction tolerance | max abs probability difference 1e-12, identical decisions, identical T* |
| Strength set | 0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 |
| Allowance | lambda = 1.0 x min mapped floor SE per statistic and version |
| Selection z | 2.0 (U = estimate + 2 SE <= allowance, inclusive) |
| Nested CV | 10 outer household folds, tag TSZ1_ncv10 |
| Stability z; cells; spread df | 2.0; S<v>R<r>k<k>, v in {0, 1}, r in {1, 2}, k in {0, 1}; n_eff - 2 over distinct cells (6 when all reader seeds differ) |
| Nested CONT | optional, descriptive, never gating |
| T3 record | roles/verifier/DEPLOY_W100.json, per version `bitwise`, `refusals_all_pass` |
| Recipient limit; target | 0.010; 0.020 (separate) |
| Primary / secondary arm | LOC / PAIR |

## 14. Open points for the coordinator

1. The verifier writes `roles/verifier/DEPLOY_W100.json` with per-version `bitwise` and `refusals_all_pass` (T3; RULES.json
   `trigger.deployment_record`).
2. The correction investigator passes guard `'none'` with the fixed `targets` constant {men: 0.005, women: 0.005} and records the
   invariance check (8.1).
3. The reader code reports per cell, release and view the winner and its seed dependence, and records the 4 reader seeds for the
   effective-seed check (section 9).
4. Whether LOC at PAIR's selected strength is read (descriptive only) depends on the reader budget; it never enters a label.

## 15. Change record (before registration; no outcome existed)

- Coordinator requests applied before freezing: nested CONT made optional and descriptive (8.2); the point-only label C2p added
  beside C2, with the closure rule stated to use C2 (11); the T3 record path and fields named (7; RULES.json `trigger`); the
  `targets` constant confirmed (8.1).
- Reader-seed handling made explicit after the reader grid's invented-data selftest showed 10 of 16 recipes seed-deterministic
  (9): per-pair `k_differs`, n_eff, the distinct-cell degrees of freedom in v_rep (identical to the source formula when n_eff = 8)
  and the always-attached reader-seed qualifier.
- Self-review corrections before registration: the men's-guard sentence in 3.1 now states the CTS1 reduced-refit slacks exactly
  (+0.49 / -0.78 / +0.14 / -0.31 x 1e-3; two of four met the reserve); the margin-pattern sub-flag for a non-nested pattern
  (n50 = 8, n70 < 8) is stated in 3.5 and RULES.json.

## 16. Completion record

Completed 2026-10-19T00:18:01Z (`date -u`). `python3 -m tsz.test_rules`: 16 of 16 PASS (invented data only). SHA-256 at
completion: RULES.json 7bb2d251e951f1745e592602539656769671537e6f29a038194093e6f83e2460; tsz/rules.py
be06bfa81164f2def3690b87c1a03509e49c74ef5bb22e81a04bb7aa6747a66d; tsz/test_rules.py
c4359b25430f1956f43047f1f7127ef651978a30b026455fe65284c599606f6c. No label was read and no real-data computation was run.
