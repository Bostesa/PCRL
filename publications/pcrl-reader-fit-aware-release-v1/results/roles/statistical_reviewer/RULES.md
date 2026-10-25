# Decision rules of the reader-fit-aware release campaign (RFA1; statistical reviewer; schema RFA1-rules-v1)

**Status.** Final for registration. Writing started 2026-10-25T21:19Z (`date -u`), before any outcome of this campaign existed; the
wave-1 draft followed the coordinator's planned structure, and section 0.2 adapts it to card `RF_RFIT_v1` (read 21:44Z with the
coordinator's answers q1-q4), still before any outcome. I read no label, ran no real-data computation, fitted nothing and called no
loader that reads labels. Every number I computed came from invented data (`python3 -m rfa.test_rules`, an in-memory mutation check
and a normal-approximation power calculation in my scratch directory). A later change is made before registration or never.
- Machine-readable constants: `RULES.json` (schema `RFA1-rules-v1`). Implementation: `rfa/rules.py` (pure functions; imports
  `nrm.rules` unchanged as NR; `load_rules()` refuses a RULES.json that disagrees with the module, a changed NRM1 RULES.json or a
  changed `nrm/rules.py`). Invented-data tests: `python3 -m rfa.test_rules` (prints JSON; exit 0 iff all pass).
- The review of `MEASUREMENT_CONTRACT.md` (numbered findings, REQUIRED / ADVISORY) is returned in the hand-back message, to be saved
  verbatim by the coordinator.
- **Inputs read** (code and records only): `MEASUREMENT_CONTRACT.md` (21:06Z draft), `sources/EDG1_ESTIMATOR_DIAGNOSTIC.md`,
  `sources/EDG1_REVIEW_REPORT.md`, DRF1's `RESEARCH_DECISION.md` and its statistical reviewer's `RULES.md` (sections 0-4),
  SAR1's `RULES.md` / `RULES.json`, `sar/rules.py`, `sar/test_rules.py`, `nrm/rules.py`, `nrm/test_rules.py` (helpers), the
  `drf/rules.py` D2R function, `sar/evaluate.py`, and the usefulness chain (`ohn/rules.py`, `ohn/evaluate.py` `complete_eval`,
  `ihs/evaluate.py`, `ihs/common.py`, `scm_bench/selection.py`, `scm_bench/common.py`, `tsz/evaluate.py` `refs_of`), plus the
  constants of the coordinator's `rfa/devreaders.py`. Nothing under `.private/`, no label, no model, nothing of MODEL_SELECT.

**What every rule here is.** A development rule on previously and adaptively used records (MODEL_FIT and MODEL_SELECT were used by
earlier campaigns, and NEURAL_RAW and the references were chosen on them): everything is exploratory and nothing here is
confirmation. No rule changes an original limit. Point eligibility (G1) gates; the confidence conjunction and the two-SE reserve are
separate results and never gate. Every uncertainty statement is asymptotic and conditional, and no interval corrects the accumulated
adaptive search of this and earlier campaigns. Versions 0 and 1 share every training row and the MODEL_SELECT people: "both
versions" is a procedure-repeat requirement, never replication.

**Signs** (NR's, unchanged): pair benefit b = comparator AUC - candidate AUC (positive favours the candidate); recipient difference
r = candidate AUC - comparator AUC (positive = the candidate's single view is more recoverable); paired costs loss(candidate) -
loss(comparator) (negative favours the candidate; accuracy enters as a loss); bracket spend d = cost(CAND) - cost(point) (d <= 0:
the point spends at least as much); learning-curve change d_LC = b(FULL FIT) - b(HALF FIT) (negative: the benefit shrinks when the
readers get the larger auxiliary sample).

## 0. Inherited unchanged, and where the definitions live

From NRM1 (`nrm.rules`, imported unchanged; RULES.json 84f18776..., nrm/rules.py d5df8686..., both checked by `load_rules()`; never
monkeypatched, test `t_nrm_rules_unchanged_not_monkeypatched`):
- **G1** (`point_eligibility`, nrm/rules.py lines 157-182): all 28 original usefulness conditions at the point, inclusive. Per
  version 12 task clauses = 2 tasks x {U0_S<v>, Ucal_S<v>} x {accuracy loss = accuracy(reference argmax) - accuracy(release argmax)
  <= 0.010; NLL excess <= 0.010 nats; summed Brier excess (summed over classes) <= 0.005} and 2 within-SEX guards (occupation summed
  Brier excess vs U0_S<v> among men, SEX 0, and among women, SEX 1, <= 0.005); 28 = 2 x (12 + 2). The values are computed by
  `scm_bench/selection.py::clauses` (lines 38-49) with `scm_bench/common.py` CAPS (line 45) and `task_vectors` (lines 213-220: argmax
  accuracy, NLL with a 1e-12 floor, summed Brier), and by `ihs/evaluate.py::utility` (lines 75-99: household-clustered SEs
  `cluster_se` / `subgroup_se`, lines 57-72; the guards, cap `ihs/common.py` GUARD line 63), called by `ohn/evaluate.py::complete_eval`
  (line 120) from `sar/evaluate.py::utility_of` (lines 160-191; U0 / Ucal from `tsz/evaluate.py::refs_of`, lines 38-46); they are
  named and cap-checked by `ohn/rules.py::conditions` (lines 103-130; CAPS line 30, GUARD_CAP line 31, a changed cap is refused).
  Test `t_usefulness_definitions_at_the_cited_code` checks these lines from the source text.
- The **two-SE reserve** (`reserve_status`, k = 2, H_LJ floor SE): reported beside, never gating.
- The **aggregate paired screen** (`aggregate`, `benefit_screen`, lines 255-341): 8 registered cells = 2 versions x 2 repetitions
  (R1, R2) x 2 reader seeds (k0, k1); blocks = version x repetition, block means over reader seeds; se_agg = max(sqrt(se_hh^2 +
  se_rs^2 + se_mc^2), 0.001); resolved iff mean - 2 se_agg > threshold and both version means > threshold; an exact alias is
  `ALIAS_ZERO`, never positive.
- The **recipient gate** (`recipient_screen`, lines 356-384; RECIPIENT_LIMIT 0.010 line 48) inside `comparison` (lines 387-394):
  neither single view more recoverable than the comparator's by more than 0.010 AUC per version at the aggregate point.
- The **frontier rule** (`frontier_comparison`, lines 569-622): Q* = the cheapest path point whose pooled MODEL_SELECT paired
  total-NLL cost is >= CAND's; AGG_POSITIVE vs Q*; no secondary metric (the 8 per-task metrics incl. the men's and women's occupation
  Brier) resolved higher vs Q*.
- `robustness`; `claims` G4 (D3), G5 (D4: `target_screen`, lines 344-353, 0.020 line 49) and G6 (D5); the nine-reference benchmark
  with its published point labels.
From SAR1 / DRF1 (copied and adapted, not imported): grid validity with the extra-trees support and few-valued explanations, refit
effectiveness, the reader-budget credit rule, the alias collapse, reader targeting, the confidence conjunction, the cheaper-point
domination conjunct and the recipient gate at Q*, the single-point spend rule (DRF1 4.2), the robustness recipient gate (DRF1 D2R),
D5's collapse rules, D6, the registered / adaptive labelling.

### 0.1 Changes from SAR1 (all before any outcome)

1. One family `RFA`: START = NEURAL_RAW (frozen); CAND = one MODEL_FIT-selected operating point per version of the reader-fit-aware
   continuation. UM, the strength machinery, `strongest_nonzero` and CALIBRATION bracket data are removed.
2. D2 gates on START, CONT (always) and three mechanism controls in a fixed order: STOPFIT, ALTCRIT, RAWJ (section 4). Met label
   `COMPONENT_GAIN_BEYOND_START_AND_MECHANISM_CONTROLS`. Development bounds are reported for every gating comparison.
3. New: the learning-curve reading (section 5), descriptive, with a learnability limitation carried into the headline.
4. D2R: the decisive set adds every gating control; the robustness recipient gates are required and their failures carried into
   the headline (DRF1); the beside list is de-duplicated (DRF1's cosmetic defect).
5. Five separately reported outcomes and the outcome classes of section 7.
6. Reader grid: offset +1200, offsets 300, 400, ..., 1100 forbidden (180 seeds), bootstrap base 2026102520 (as
   `rfa/devreaders.py` BOOT_SEED), B = 400 recorded, split tag `RFA1_devgrid_R{r}`.

### 0.2 Card RF_RFIT_v1 (registration adaptation, before any outcome)

1. **Family.** The card's code family `FAM` is the rules' family `RFA` (an alias, canonicalised to RFA: two registered readings under
   the two names are refused). Card `RF_RFIT_v1` is revision 0; the one planned revision R1, with its own STOPFIT, is revision 1.
2. **Configurations.** CAND = `RF_FULL` (occupation network only; the income network and released income vector are START's,
   bitwise), CONT = `RF_TASK`, STOPFIT = [`RF_STOP_LO`, `RF_STOP_HI`], ALTCRIT = [`RF_ALTC_HI`], RAWJ = [`RF_RAWJ_HI`].
3. **LO not saved.** ALTCRIT and RAWJ save only their HI point; their LO is recorded on MODEL_FIT, not saved (`lo_not_saved`, flag
   `LO_NOT_SAVED_SPEC_BUDGET_<key>`): their path is START + HI, so the cheaper-point conjunct (c3) covers START only and the
   interpolation uses START as the lower point. A bracket point that does not exist is not saved: the path is read on START and the
   saved points (a missing HI usually gives NO_BRACKET).
4. **NEAR_ALIAS.** ALTCRIT is a NEAR_ALIAS of STOPFIT iff the recorded median relative critic lag on the deployed fit is < 0.01 (the
   card's rule, applied by the rules from the recorded value). A near alias **still gates** (an intersection can only be harder; its
   own artifact is read) but is **never counted as an independent control** (flag `ALTCRIT_NEAR_ALIAS_OF_STOPFIT_NOT_INDEPENDENT`;
   scope sentence `D2_SCOPE_NEAR_ALIAS`). A recorded `altcrit_near_alias` that disagrees with the lag rule is INCOMPLETE; a missing lag
   gives flag `ALTCRIT_LAG_NOT_RECORDED`. The exact-alias kind stays available (identical wires) but is not expected.
5. **F0 gate.** The card's MODEL_FIT, SEX-based gate at START decides only whether the family is fitted. A version reading
   `MECHANISM_INERT` saves nothing, so a reading with any inert version has CAND status `MECHANISM_INERT`: D1
   `NOT_FITTED_MECHANISM_INERT`, D2-D5 and the learning curve `NOT_ASSESSED_MECHANISM_INERT`, class `MECHANISM_INERT` (a design failure
   on these data, not a usefulness or protection result; flag `MECHANISM_INERT_IN_ONE_VERSION_ONLY`). A FITTED reading with an inert
   version, or the status without the inert record, is INCOMPLETE. Disclosure flag `F0_SEX_BASED_FIT_GATE_DISCLOSED`.
6. **CAND's selection.** k* uses only the 5-fold MODEL_FIT out-of-fold usefulness spends (estimate + 2 SE within SAR1 FB_PFO_R1's
   occupation budgets): selection data `MODEL_FIT_OOF`, no SEX-recovery diagnostic (the record key is
   `selection_uses_sex_recovery_diagnostic`; the draft's flag is renamed `CAND_SELECTED_ON_MODEL_FIT_SEX_RECOVERY_DIAGNOSTIC`).
7. **Learning curve.** HALF = FIT fraction 0.5 by the inherited `RFA1_devgrid_LC` fold rule (the FIT households in folds 0-1 of 4); the
   0.25 fraction is not run; the HALF cells reuse the SCORE households and the identical bootstrap weights, so `diff_cell_se` is the
   paired SE (the unpaired bound would be flagged as a deviation).
8. **Falsifiers.** The card's registered falsifiers are reported in `decisions()['falsifiers']` and as `FALSIFIER_<k>_TRIGGERED` flags,
   never changing a label: F0 from the F0 record; F2 TRIGGERED iff the STOPFIT frontier label at Q* is neither
   `DOMINATES_CONTROL_FRONTIER` nor `SEPARATED_ON_PRIMARY_COST_ONLY` (`NOT_ASSESSABLE_NO_BRACKET` without a bracket); F4 TRIGGERED iff the
   learning curve reads `LEARNABILITY_LIMITATION`; F1, F3, F5 and F6 as recorded by the coordinator (`falsifier_records`), else
   `NOT_RECORDED`.

## 1. The reading and its comparators

| Comparator | What it is | Role |
|---|---|---|
| CAND | NEURAL_RAW's two networks continued with the held-out reader-fit-aware objective (derivative through a reader fitted on MODEL_FIT auxiliary episodes); one operating point per version chosen by the card's rule on MODEL_FIT (`MODEL_FIT_OOF` or `MODEL_FIT_EPISODES`, recorded; never MODEL_SELECT) | the reading |
| START = NEURAL_RAW | frozen, both versions | D2 gate (simple comparison); a D3 reference |
| CONT | zero-privacy continuation: identical task rows, optimizer and steps | D2 gate (simple comparison) |
| STOPFIT | identical episodes, readers, loss, schedule and lambda, the fitted reader parameters detached from their fitting path | D2 gate, PATH from START: isolates the fit-aware derivative |
| ALTCRIT | an ordinary alternating persistent critic of the same family and loss | D2 gate: a PATH from START, a POINT (selected by CAND's own MODEL_FIT rule) or an exact alias of STOPFIT (identical wires shown: artifact reused, counted once) |
| RAWJ | NEURAL_RAW's own online-critic recipe continued beyond E2 to matched spend (SAR1's FB ALT) | D2 gate, PATH from START |
| nine references | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (U0 / Ucal are utility references) | D3; D4 against NEURAL_RAW and H_LJ |

The equal-exposure and identity claims of CONT, STOPFIT, ALTCRIT and RAWJ are card statements the rules cannot verify (challenger
and verifier). The revision R1 is a separate reading of the same structure with its own isolating STOPFIT control (section 7).

## 2. D1 usefulness (`d1`, `confidence_conjunction`)

- **Gating (G1)**: all 28 original conditions at the point at CAND: `TASK_FEASIBLE`, `CAND_INFEASIBLE`, `NO_FEASIBLE_POINT` (no
  operating point met the card's MODEL_FIT rule: no candidate, no fallback), `INCOMPLETE` (also when CAND's selection data are not
  recorded or not a MODEL_FIT source).
- **Confidence conjunction (reported, never gating)**: for CAND, START and CONT, each condition from its own household-clustered SE
  (sqrt(se^2 + se_mc^2) for a randomized release; no floor): **PASS** iff value + 2 se <= cap; **VIOLATION** iff value - 2 se > cap;
  **UNRESOLVED** otherwise. 28 one-sided per-condition bounds, not a simultaneous statement.
- **Reserve (reported, never gating)**: NR's two-SE reserve of CAND, START and CONT.
- The three usefulness results are separate. "Task feasible" means G1; "confidence-resolved" only with `CONFIDENCE_ALL_PASS`; no
  reserve or confidence rule removes an otherwise eligible reference (D3 uses the published point labels, all nine eligible). Flags
  `START_POINT_INELIGIBLE`, `CONT_POINT_INELIGIBLE`, `POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED`, `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`,
  `CAND_SELECTED_ON_MODEL_FIT_SEX_RECOVERY_DIAGNOSTIC` (disclosure: CAND's operating-point rule used a SEX-recovery MODEL_FIT diagnostic that the
  spend-matched control points do not use).

## 3. Resampling and what each uncertainty covers

1. **Household sampling.** Per registered cell, household bootstrap of SCORE, B = 400, with the release, the fitted readers, the
   orientation and the VALIDATION-selected winner held fixed. se_hh = the RMS of the cell SEs with the aggregate's cell weights: an
   upper bound under any cross-cell correlation, which does not shrink with more cells. It does not cover refitting the readers,
   re-selecting the winner or refitting the release. The usefulness SEs are household-clustered over MODEL_SELECT (release fixed).
2. **Release-fitting, reader-split and reader-seed variation.** se_rs = sqrt(sum_b (m_b - mean)^2 / 4) over the four version x
   repetition block means, with reader seeds averaged inside each block. It mixes the version difference (two procedure repeats with
   different U0 / NEURAL_RAW and inner-episode seeds), the split difference (R1 / R2 deal the same people) and their interaction;
   reader-seed variation enters only through the block means. Each factor has about one degree of freedom, so se_rs is a crude
   component and the screen is approximate. Two versions do not estimate release-fitting variance.
3. **Numerical randomness.** se_mc from cell MC SEs (0 for deterministic releases, the planned case).
4. **Selection devices.** The bracket SEs (household-clustered SEs of MODEL_FIT out-of-fold paired costs) and CAND's MODEL_FIT rule
   select points; they are never inference. Inner auxiliary episodes, inner surrogate values, inner held-out AUCs and out-of-fold
   reader checks are training data and training diagnostics of the release; their spread over folds or episodes is never an SE.
5. **Not replication.** Versions, repetitions R1-R4, reader seeds k0-k3, the two learning-curve budgets and overlapping folds share
   people; none is an independent replication.

## 4. D2 gain beyond the mechanism-specific controls (`d2`, `control_spec`, `brackets`, `grid_fineness`)

### 4.1 Labels in order of precedence

1. `NOT_ASSESSED_NO_FEASIBLE_POINT`; 2. `NOT_ASSESSED_EXACT_ALIAS` (CAND = START: no protection experiment); 3. `INCOMPLETE` (CAND
status, selection data or utility); 4. `NOT_ASSESSED_TASK_INFEASIBLE` (never negative, never protection); 5.
`NOT_COMPUTED_GRID_VALIDITY_UNKNOWN` / `NOT_COMPUTED_GRID_INVALID`; 6. `INCOMPLETE` (START not NEURAL_RAW; a control kind not
allowed: STOPFIT and RAWJ are PATH, ALTCRIT PATH / POINT / ALIAS_OF_STOPFIT; a POINT naming other than exactly one point; an alias
without `altcrit_alias_wire_check` True); 7. `NOT_ASSESSED_CONTROLS_NOT_RUN` (a gating control recorded as not run for budget; never
inferred); 8. `INCOMPLETE` (a gating control's selection data not recorded or not allowed, flag
`CONTROL_SELECTION_DATA_NOT_ALLOWED_<key>`, a re-selection being a POST HOC repair; any needed input missing); 9. the conjuncts:
- **(a)** G1 at CAND;
- **(b) START** and **(b') CONT**: `NR.comparison` passes (AGG_POSITIVE pair benefit and both recipient gates);
- **(c) each PATH control** (STOPFIT, a PATH ALTCRIT, RAWJ; path = START + its distinct saved points): (c1) `NR.frontier_comparison`
  = `DOMINATES_CONTROL_FRONTIER`; (c2) the recipient gate vs Q*; (c3) no cheaper path point resolved less recoverable than CAND
  (aggregate upper bound < 0 and both version means < 0);
- **(c') a POINT ALTCRIT** by the single-point spend rule (4.3);
- **(d) reader targeting**: the non-table winner's point pair benefit > 0 in both version means vs START, CONT and every gating Q* /
  point.
10. `COMPONENT_GAIN_BEYOND_START_AND_MECHANISM_CONTROLS` iff every gating comparison and (d) pass; `READER_TARGETED` iff every
gating comparison passes and (d) fails; `NOT_SEPARATED` iff none passes; otherwise `PARTIAL_<passed>_ONLY` (joined by `+` in the
order START, CONT, STOPFIT, ALTCRIT, RAWJ) with a statement naming what passed and what each failure leaves open: without STOPFIT the
gain may come from the held-out reader penalty itself or from task spending and the fit-aware derivative is not isolated; without
ALTCRIT an ordinary alternating critic may reproduce it; without RAWJ running NEURAL_RAW's own recipe further may; without CONT
continued task training alone may; without START it is not lower than the start. **No partial outcome is a claim**; NOT_SEPARATED
is the default reading, never "equal". An exact-alias ALTCRIT is not a second comparison (flag
`ALTCRIT_EXACT_ALIAS_OF_STOPFIT_COUNTED_ONCE`). Scope sentence of the met label (`D2_SCOPE`): a resolved linked-pair benefit with
both recipient gates beyond the unchanged start, its zero-privacy continuation, the stop-fit control and the alternating critic at
matched MODEL_SELECT total-NLL spend where bracketed, and beyond NEURAL_RAW's own online-critic recipe continued to matched spend,
for the declared finite-data reader procedure on reused MODEL_SELECT people; evidence about this procedure and auxiliary budget,
not an arbitrary-attacker guarantee and not information removal; not D3, not D4.

### 4.2 Paths, no bracket, residual mismatch

- Two points per path and version are saved before any MODEL_SELECT read on MODEL_FIT out-of-fold paired total-NLL spend over START:
  LO = the largest path point with spend <= CAND's (START counts as 0) and HI = the smallest with spend >= CAND's + 2 SE (SE = the
  household-clustered SE, same data, of that point's paired difference against CAND). On MODEL_SELECT, NR's rule picks Q*.
- **No bracket** (every path point spends less MODEL_SELECT total NLL than CAND): that comparison reads `NOT_SEPARATED`, reason
  `NO_BRACKET`, flag `NO_BRACKET_<key>`: a limit of the control path, never evidence of equality or of no effect.
- Bracket at START: flag `<key>_BRACKET_IS_ORIGIN` (it repeats the START comparison); an empty saved list `<key>_AT_ORIGIN`; CAND
  identical to a bracketing point: `CAND_ALIAS_OF_<key>_<id>` (never a pass).
- Residual mismatch (never gating): cost(CAND) - cost(Q*) pooled with SE and by version, the smallest overspend vs a cheaper point,
  NR's interpolation. The bracket is read on total NLL; the 8 secondary metrics are the guard.

### 4.3 The single-point spend rule (POINT ALTCRIT; DRF1 4.2)

d = pooled MODEL_SELECT paired total-NLL cost(CAND) - cost(point) (NR._primary: mean of versions, SE = mean of version SEs):
`SPENDS_AT_LEAST_AS_MUCH` (d <= 0); `SPENDS_LESS_UNRESOLVED` (0 < d, d - 2 SE <= 0: credited, flag `ALTCRIT_SPENDS_LESS_UNRESOLVED`);
`SPENDS_LESS_RESOLVED` (d - 2 SE > 0: `NOT_SEPARATED`, reason `NOT_SEPARATED_FROM_SPEND`, `passed_if_spend_matched` reported).
Otherwise pass iff AGG_POSITIVE pair benefit, both recipient gates and no secondary metric resolved higher than the point
(`SEPARATED_ON_PRIMARY_COST_ONLY` otherwise). A POINT's selection data are CAND's own MODEL_FIT sources.

### 4.4 Fineness of the saved path points (`grid_fineness`; reported, never gating)

Per PATH control and version on MODEL_FIT out-of-fold costs with s = CAND's spend over START and SE from the same data: `FINE` iff
LO >= 0.75 s (and <= s) and HI >= s + 2 SE and <= s + 2 SE + 0.25 s; `COARSE_BELOW` / `COARSE_ABOVE` / `COARSE_BOTH`;
`CONTROL_PATH_MAY_NOT_BRACKET` (largest spend < s + 2 SE: extend the path, e.g. more STOPFIT epochs, before reading);
`SAVED_POINT_RULE_NOT_FOLLOWED`; `NOT_APPLICABLE_CAND_NOT_ABOVE_START`; `SELECTION_DATA_NOT_ALLOWED` (anything but
MODEL_FIT_OOF); `INCOMPLETE`; `NOT_GIVEN`. Computed before any MODEL_SELECT read.

### 4.5 Power, bounds and multiplicity (normal approximation; invented settings)

One comparison resolves a true benefit delta with probability Phi(delta / se_agg - 2): 50% at 2 se_agg, about 80% at 2.84 se_agg.
With se_agg 0.006 (START), 0.007 (CONT) and 0.008 (each control), correlation 0.5, D2 (all five resolved; version-mean condition
and targeting ignored) resolves a common true benefit of 0.010 / 0.015 / 0.020 / 0.025 / 0.030 with probability about 0.04 / 0.17 /
0.42 / 0.71 / 0.90 (0.06 / 0.22 / 0.49 / 0.76 / 0.92 with ALTCRIT an exact alias); when STOPFIT reproduces half of the gain, 0.02 /
0.08 / 0.18 / 0.31 / 0.44; when STOPFIT and ALTCRIT both do, 0.01 / 0.05 / 0.10 / 0.18 / 0.28. **Bounds**: every gating comparison
reports mean, se_agg and mean +- 2 se_agg (`bounds`): development bounds for this procedure on reused people, conditional on the
releases fitted, not population or arbitrary-attacker bounds, not corrected for the adaptive search. A D2 look is an
intersection-union conjunction of up to five AGG_POSITIVE comparisons plus targeting: its false-pass rate is at most that of its
weakest member (one comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 known laws and about 8%
when one version or split dominates, NRM1 RULES.md 5.5); with K looks at most K times the per-look rate. No multiplicity correction
is applied or claimed.

## 5. The learning curve (`learning_curve`; budget sensitivity; descriptive)

- **Budgets** (fixed before any outcome): FULL = the declared FIT part; HALF = a fixed half of each FIT part's households by hash
  rank (HALF within FIT). VALIDATION, SCORE, recipes, reader seeds and the bootstrap draws are unchanged, so the per-cell
  difference has a paired-bootstrap SE.
- **Read** at CAND when D2 reached its conjuncts, on the registered cells, against START, CONT, every gating Q* / point and H_LJ.
- **Per comparator**: d_LC = b(FULL) - b(HALF) per cell; SE = the paired-bootstrap SE when recorded for every cell, else
  sqrt(se_FULL^2 + se_HALF^2) (flag `LC_SE_UNPAIRED_BOUND`; an upper bound when the two budgets' SCORE errors are non-negatively
  correlated); aggregated by `NR.aggregate`: `BENEFIT_SHRINKS_RESOLVED` (upper bound < 0 and both version means < 0),
  `BENEFIT_SHRINKS_POINT` (mean < 0), `NO_SHRINKAGE_AT_POINT`. A **limitation** iff the full-budget point benefit is > 0 and mean
  d_LC < 0 (a point rule on purpose: the limitation is disclosed whenever the point estimate shrinks).
- **Family**: `LEARNABILITY_LIMITATION` if any comparator has a limitation (its `headline_caveat` is carried into the outcomes
  statement and flag `LEARNABILITY_LIMITATION_CARRIED_INTO_HEADLINE`); `INCOMPLETE` if a comparator is not run; `NOT_RUN`;
  `NO_SHRINKAGE_SEEN` otherwise. A shrinking benefit is a possible learnability effect and is never presented as information
  removal; no shrinkage between two budgets is not evidence that information was removed, and budgets above the declared FIT were
  not tested. The card records the inner episodes' auxiliary training size (flags `LC_INNER_EPISODE_SIZE_NOT_RECORDED`,
  `INNER_EPISODE_BUDGET_AT_LEAST_EVALUATION_FIT`). Never changes D1-D6 or the class.

## 6. D3, D4, D5, D6

Read at CAND whenever CAND is fitted, G1-feasible and selected on MODEL_FIT and the grid is valid (also when D2 is INCOMPLETE or
NOT_ASSESSED_CONTROLS_NOT_RUN); otherwise they carry D2's gate label or INCOMPLETE. **D3** = NRM1 G4: per reference of the nine
(published point labels; no reserve rule removes one), resolved pair benefit with both recipient gates;
`SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES` only if every one passes; passed and failed lists. NEURAL_RAW (START) is one of the nine.
**D4** = NRM1 G5: `MATERIAL_0020_RESOLVED` iff against each of NEURAL_RAW and H_LJ the aggregate mean - 2 se_agg > 0.020 (strict)
and both version means > 0.020, with both recipient gates; lower levels kept (`MATERIAL_0020_POINT_ONLY`,
`POSITIVE_BELOW_MATERIAL_TARGET`, `NOT_MET`); never weakened. At se_agg about 0.010 this needs a point benefit near 0.040. Both carry
the reader-budget credit rule (a READER_BUDGET_ASYMMETRIC reference is credited only if the non-table winner's point benefit is
positive in both versions; missing input is INCOMPLETE). Several references emit few-valued codes while CAND is continuous: a
fitted-reader comparison across such different outputs can favour the harder-to-learn release (EDG1's direction for finer
outputs); the learning curve against H_LJ is the planned check, and its size is not quantified. **D5** = NRM1 G6 only when a card
claims joint design (a local twin and both sequential orders built); RFA1 builds no local twin: `NOT_APPLICABLE_NOT_JOINT`. **D6** =
`NOT_CLAIMED` unless a separate prior-art review supplies a label; never inferred from performance.

## 7. Outcomes, outcome class, registration, campaign summary (`decisions`, `adaptive_tag`, `campaign_summary`)

- **Five separately reported outcomes** (`outcomes`, one statement): (1) usefulness = D1 with the confidence conjunction and the
  reserve; (2) gain beyond the mechanism-specific controls = D2 with D2R and the learning curve; (3) strong references = D3; (4)
  material target = D4; (5) joint design (D5) and novelty (D6). None implies another.
- **Outcome class**, first applicable: `MECHANISM_INERT` (the F0 gate stopped the family, section 0.2); `TASK_INFEASIBLE` (no
  feasible point, or CAND and every other recorded point fail G1);
  `SELECTED_POINT_INFEASIBLE` (CAND fails G1, another recorded point passes: never promoted); `INCOMPLETE` (D1 incomplete; D2
  incomplete or not computed); `NO_OUTPUT_CHANGE` (CAND = START); `MECHANISM_NOT_ASSESSED` (a gating control not run); `INCOMPLETE`
  (D3 undecided); `COMPETITIVE_CANDIDATE` (D2 and D3 met; D4 separate); `MECHANISM_COMPONENT_NOT_COMPETITIVE` (D2 met, D3 not);
  `REFERENCE_ADVANTAGE_NOT_ATTRIBUTED` (D3 met, D2 not; `partial` = D2's label); `BOUNDED_NEGATIVE_EXPERIMENT` (otherwise; `partial`
  = D2's label, e.g. `NOT_SEPARATED`, `READER_TARGETED`, `PARTIAL_START+CONT_ONLY`, with the development bounds of 4.5). D4, D5, D6,
  D2R and the learning curve never change the class. Flags: `D3_WITHOUT_D2`, `D4_WITHOUT_D2`, `POINT_FEASIBLE_NOT_*`,
  `NO_POINT_MEETS_MODEL_FIT_RULE`, `CAND_OPERATING_POINT_DIFFERS_BY_VERSION`, `REVISION_BEYOND_PLANNED`,
  `D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE`, `D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE`, `LEARNABILITY_LIMITATION_CARRIED_INTO_HEADLINE`
  and the D1, D2, D5, learning-curve and MC flags.
- **Registration**: registered iff revision 0 of `RFA` with its card committed before any MODEL_SELECT outcome. The one planned
  substantive revision R1 (its own isolating STOPFIT) is `REVISED_R1` if its card preceded every MODEL_SELECT outcome, else
  `ADAPTIVE_R1`; a replacement family is `FS_<name>`; every label of a non-registered reading is prefixed `<tag>__`. A revision
  triggered by a MODEL_SELECT outcome is adaptive as a whole. Any other family name is refused.
- **Campaign summary**: the registered reading, every other reading with its tag, K = the number of D2 looks, the readings with D2,
  D3, D4 met and with a learnability limitation, flag `D2_MET_ONLY_IN_NON_REGISTERED_LOOKS`, and the multiplicity statement every
  campaign-level sentence carries.

## 8. D2R and refit effectiveness (`d2r`, `refit_effectiveness`)

- **Trigger, fixed now**: D2R is read at CAND whenever CAND is G1-feasible and the registered grid is valid (budget permitting; a
  skipped D2R reads NOT_RUN and is disclosed), whatever D2's label. R3 / R4 with reader seeds k2 / k3; nothing retuned.
- **Decisive set**: START, CONT and the registered STOPFIT, ALTCRIT (unless an alias) and RAWJ brackets or point, never re-chosen on
  robustness cells. Per comparator `ROBUSTNESS_CONFIRMED` iff `NR.robustness` confirms (AGG_POSITIVE on the robustness cells and on
  all cells), both recipient gates hold on the robustness cells, the registered cells are present and the refit is
  `REFIT_GENUINELY_EFFECTIVE` (at least one of R3 / R4 with recorded, different household assignments and SCORE overlaps in [0, 1);
  all four reader seeds valid; flag `ONE_ROBUSTNESS_SPLIT_ONLY`). Beside: the references present plus NEURAL_RAW and H_LJ.
- A sign change or a failed robustness recipient gate on any comparator, decisive or beside, is carried into the headline.
  Descriptive; never changes D2. R3 / R4 are new splits of the same people: robustness to classifier refitting, never replication.

## 9. Reader-grid validity (`grid_validity`; prospective)

Reader seeds S<v>k<k> recorded as distinct integers equal to MODEL_SEED_BASE + 1200 + 10 v + k (2026102405-2026102408,
2026102415-2026102418), none equal to MODEL_SEED_BASE + {300, ..., 1100} + 10 v + k (k = 0..9; 180 seeds); bootstrap B recorded
and equal to 400; bootstrap seeds S<v><R> recorded and equal to 2026102520 + 100 v + repetition index (no seed shared with SAR1's or
DRF1's sets). Per seed pair, every seed-dependent recipe checked and every seed-dependent equality explained: off the minimum set by
NRM1 A-SC1 (iii); on it by the extra-trees support explanation (ET5 / ET25, distinct inputs <= 8001 // min_leaf); anywhere by the
few-valued explanation (distinct inputs <= 320 and the same recipe differs elsewhere in the same version, repetition and seed pair),
counted once (n_eff). Any other equality invalidates the grid. **Minimum set** = CAND, START, CONT and the saved STOPFIT, ALTCRIT and
RAWJ points (for the refit: the D2R release and its decisive comparators). The strict NRM1 reading is reported beside. Claims need
`grid_valid` recorded and exactly True; the learning curve needs its own `grid_valid` True as well.

## 10. Required records (the coordinator's inputs per reading)

1. `family`, `card`, `revision`, `card_committed_before_any_model_select_outcome`; CAND's status, config_id, per-version
   `operating_point`, `selection_data` (MODEL_FIT_OOF / MODEL_FIT_EPISODES), `selection_uses_sex_diagnostic`, `alias_of_start`;
   `points` (every fitted operating point saved, with utility); START and CONT records with utility.
2. `paths` (STOPFIT, ALTCRIT, RAWJ), `control_kind['ALTCRIT']`, `altcrit_alias_wire_check` (only for an alias), `selection_data`
   per control, `controls_not_run` (budget only), `alias_of`, `fineness` (from `grid_fineness`, with the full path recorded).
3. Grid entries (pair and both views) of CAND vs START, CONT, every saved control point and the nine references; paired costs vs
   START, CONT and every saved point (`total|nll` + the 8 secondary metrics); `nolookup` per-cell benefits vs START, CONT, every saved
   point and the nine; `enhanced` (reported); the reader-budget flags.
4. Grid validity inputs (seeds, bootstrap seeds, B, seed checks with `k_pair`, distinct-input counts per wire, version and view, the
   minimum set).
5. `learning_curve`: `grid_valid`, `budgets`, `inner_episode_fit_size`, `half` entries vs START, CONT, every saved point and H_LJ
   (cells under the registered names S<v>R<r>k<k>, not the `@0.5` names), and `diff_cell_se` (paired-bootstrap SEs of the per-cell
   difference).
6. Card RF_RFIT_v1: `f0` = {v: MECHANISM_ACTIVE | MECHANISM_INERT}; `altcrit_median_relative_lag` (deployed fit) and optionally
   `altcrit_near_alias`; `lo_not_saved` = ['ALTCRIT', 'RAWJ']; `falsifier_records` for F1, F3, F5, F6.
7. For D2R: merged registered + robustness entries (pair and both views), the decisive `comparators`, the split records.

## 11. Limitations

- Version- or split-dominated laws give about 8% false pass per AGG_POSITIVE comparison (NRM1 5.5); two versions and two
  registered splits rest on one degree of freedom each.
- Selection of CAND and of the saved path points on MODEL_FIT treats refitted pipelines as fixed: a selection device, not
  inference. If CAND's rule uses a SEX-based MODEL_FIT diagnostic, CAND had a selection opportunity on privacy that the
  spend-matched control points do not (flagged; the controls' value of the same diagnostic should be reported beside).
- A reader-fit-aware objective is trained against readers fitted at its inner auxiliary budget; a fitted-reader evaluation at a
  similar budget is the setting most favourable to such a mechanism. The two-budget learning curve can reveal shrinkage between half
  and full FIT only; it cannot show behaviour at larger budgets.
- The recipient gate and the learning-curve limitation are point rules: under the NRM1 calibration a true single-view worsening of
  exactly 0.010 passes the recipient gate about 32-34% of the time.
- The few-valued and extra-trees explanations are necessary conditions of their mechanisms, not sufficient ones.
- All readings use the same adaptively used MODEL_SELECT people; registration status and the look count are disclosed instead of a
  correction.

## 12. Completion record

- Wave-1 draft completed 2026-10-25 about 21:36Z; final for registration (section 0.2) completed at the time recorded in the final
  hand-back (`date -u`), which also records the SHA-256 of RULES.json, rfa/rules.py and rfa/test_rules.py (this file cannot contain
  its own hash).
- `python3 -m rfa.test_rules`: 24 of 24 PASS, exit 0 (invented data only; about 0.45 CPU-s; one numerical thread), including
  `t_card_rf_rfit_v1` (the card's record shape: FAM alias, HI-only paths, LO not saved, near alias by the lag rule, F0 inert in one or
  both versions, inconsistent F0 records, a missing HI, the falsifier mapping, one family under two names refused).
- In-memory mutation check (26 single-site seeded bugs of `rfa/rules.py`, among them: offset 1100 not forbidden, B not checked or
  not required, CONT not gating, an alias accepted without its wire check, a resolved-cheaper point credited, cheaper-point
  domination ignored, a missing bracket passing, targeting ignored, the learning-curve limitation without a full-budget benefit, the
  paired LC SE ignored, H_LJ dropped from the learning curve, the robustness recipient gate ignored, the reader-budget credit dropped,
  D3-without-D2 promoted, bracket or CAND selection data unchecked, the confidence PASS read at the point, the bound's sign, a
  revision registered, the learning curve changing the class, START not checked, the D2R beside list not de-duplicated, the point
  rule's secondary guard dropped, a not-run control inferred): caught 26 of 26; extended after section 0.2 by 9 more (a near alias
  dropped from gating, the 1% lag rule not strict, a disagreeing near-alias declaration accepted, MECHANISM_INERT reclassified, a
  FITTED reading with an inert version accepted, the LO-not-saved consistency check removed, the FAM alias not canonicalised, F2
  inverted, the near-alias scope dropped): caught 35 of 35; no repository file was changed by it.
- Inherited files verified unchanged: NRM1 RULES.json 84f18776..., nrm/rules.py d5df8686... (load_rules checks both).
