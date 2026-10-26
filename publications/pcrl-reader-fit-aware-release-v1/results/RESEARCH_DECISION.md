# Research decision (campaign `pcrl_reader_fit_aware_release_v1`, RFA1)

Coordinator, 2026-10-25. Wording follows the statistical reviewer's claims reviews (`roles/statistical_reviewer/HANDBACK_2.md`,
`HANDBACK_3.md`). The governing decision records are `claims/RFA_v1_final/` and `claims/RFA_R1_final/` (deviation 10); the endpoint
values are in `ENDPOINTS.json` and `TABLES.md`.

Every number is development evidence on adaptively reused MODEL_SELECT people, for the declared reader procedure: the admitted
16-recipe bank, fitted on FIT, oriented and selected on VALIDATION, scored once on SCORE. A lower reader AUC is not an
arbitrary-attacker privacy guarantee. No interval corrects the accumulated adaptive search. D5 is not applicable (no local twin); D6
NOT_CLAIMED.

## Decision

**A bounded negative experiment.** NEURAL_RAW's occupation network was trained against readers fitted afresh in each training
episode, with the derivative passing through the readers' weight fit. F0 found the fit-path term active on MODEL_FIT (orthogonal share
0.10-0.11). The candidate's recovery by the declared readers was not separated from any mechanism-specific control (the control
paths bracketed on MODEL_FIT and compared at matched MODEL_SELECT total-NLL spend). "Not separated" is not "no effect". This holds
for the registered reading and for the one adaptive revision. Against the unchanged start NEURAL_RAW:
- In v1, the candidate's pair recovery on the registered cells was resolved below NEURAL_RAW's, but that benefit was not confirmed
  on the second split.
- In R1, the benefit over NEURAL_RAW was point only (+0.0098, se 0.0054), and it was not confirmed on the second split either.

It is not a competitive method, and no mechanism component was established.

| Reading | Outcome class | D1 | D2 | D3 | D4 | Learning curve | Second split (D2R) |
|---|---|---|---|---|---|---|---|
| RFA_v1 (registered; card RF_RFIT_v1) | BOUNDED_NEGATIVE_EXPERIMENT (partial PARTIAL_START_ONLY) | TASK_FEASIBLE | PARTIAL_START_ONLY | NOT_SUPERIOR | NOT_MET | LEARNABILITY_LIMITATION (point rule) | ROBUSTNESS_NOT_CONFIRMED |
| RFA_R1 (ADAPTIVE_R1; card RF_RFIT_R1) | BOUNDED_NEGATIVE_EXPERIMENT (labels carry the ADAPTIVE_R1 prefix, except the second-split label, which the D2R code writes unprefixed; deviation 11) | TASK_FEASIBLE | NOT_SEPARATED | NOT_SUPERIOR | NOT_MET | LEARNABILITY_LIMITATION (point rule) | ROBUSTNESS_NOT_CONFIRMED |

The rules' campaign summary (`claims/CAMPAIGN_SUMMARY.json`), verbatim:

> D2 was met in 0 of 1 registered readings and in 0 of 1 non-registered readings (2 D2 looks in total); D3 was met in 0 and D4 in 0
> of 2 readings; a learnability limitation was carried in 2; one AGG_POSITIVE comparison passes a true null in about 0.07-2% of
> replicates under six moderate NRM1 known laws and about 8% when one version or split dominates (NRM1 RULES.md 5.5); a D2 look is an
> intersection-union conjunction of up to five such comparisons (START, CONT, STOPFIT at its bracket, ALTCRIT at its bracket or point
> unless an exact alias of STOPFIT, RAWJ at its bracket) plus reader targeting, so its false-pass rate is at most that of its weakest
> member; with K D2 looks the chance that at least one passes by chance is at most K times the per-look rate; no multiplicity
> correction is applied or claimed.

Here K = 2.

## The five separate outcomes

### 1. Usefulness

- **At the point.** Both readings are point-feasible on the original 28 conditions in both versions. The income output is START's
  network (deviation 7: it differs from the frozen START wire by at most 6.8e-8).
- **Confidence.** Not confidence-resolved.
  - v1 has 26 of 28 resolved. The unresolved two are in version 1: occupation NLL vs Ucal (0.00737 +- 0.00133 of 0.010) and the men's
    guard (0.00442 +- 0.00074 of 0.005).
  - R1 has 27 of 28. The unresolved one is the version-1 men's guard (0.00457 of 0.005).
- **Reserve.** Not reserve-qualified.
- **Tightest condition on MODEL_SELECT.** At the selected points it is the version-1 men's occupation guard: 0.885 of its cap in v1,
  0.915 in R1.
- **What stopped k\* on MODEL_FIT.** The selection rule takes the largest S_k such that S_10 ... S_k all keep estimate + 2 SE within
  BUDGET_OCC (MODEL_FIT out-of-fold, against the fold start). In all four fits the first failing checkpoint is k\* + 10. Its failing
  components (`claims/CAMPAIGN_SUMMARY.json`, `binding_model_fit_budget_components`):

  | Fit | k\* | First failing checkpoint | Failing component (estimate + 2 SE > budget) |
  |---|---|---|---|
  | v1, version 0 | S20 | S30 | women's occupation Brier 0.00299 > 0.0028 |
  | v1, version 1 | S70 | S80 | occupation NLL 0.00695 > 0.0060; women's occupation Brier 0.00293 > 0.0028 |
  | R1, version 0 | S30 | S40 | women's occupation Brier 0.00294 > 0.0028 |
  | R1, version 1 | S60 | S70 | occupation NLL 0.00628 > 0.0060 |

- **Version 0 barely moved in v1.**
  - k\* = S20, and the candidate's MODEL_FIT out-of-fold total-NLL spend over the start is 0.0008 nats (se 0.0002).
  - The STOPFIT and ALTCRIT HI points sit at the same S20. In every fit the HI points of these two controls are saved at S_k\*.
  - Their LO bracket points (S10, saved for STOPFIT only) lie well below the candidate's spend. The version-0 brackets therefore read
    coarse: COARSE_BELOW for STOPFIT and ALTCRIT, COARSE_BOTH for RAWJ (the CONTROL_GRID_COARSE flags; reported, never gating).
    Version 1 reads FINE, and so does every R1 bracket.
  - The per-version pair benefits over START are +0.0100 / +0.0154 (`claims/RFA_v1_final/DECISIONS.json`,
    `D2.targeting.NEURAL_RAW.version_means`).

### 2. Gain beyond the mechanism-specific controls

#### v1 (registered)

On the registered cells, the candidate's linked-pair recovery by the declared readers was resolved below NEURAL_RAW (+0.0127, se
0.0059, with both recipient gates). This is v1's only D2 pass. It was not separated from any mechanism-specific control:

| D2 comparison | Compared at | Registered pair benefit (se) | Second split: robustness-cell mean (se) |
|---|---|---|---|
| START | NEURAL_RAW | +0.0127 (0.0059), resolved | +0.0120 (0.0075), not confirmed |
| CONT, zero-privacy continuation (not spend-matched) | RF_TASK | +0.0086 (0.0092) | +0.0102 (0.0074) |
| STOPFIT, the isolating control | RF_STOP_HI | +0.0028 (0.0045) | -0.0009 (0.0036): sign change |
| ALTCRIT, the lagging critic | RF_ALTC_HI | +0.0005 (0.0010, at the SE floor) | -0.0023 (0.0049): sign change |
| RAWJ, NEURAL_RAW's own recipe | RF_RAWJ_HI | -0.0016 (0.0084) | -0.0044 (0.0090) |

- **The controls' usefulness.** "At comparable usefulness" does not hold for every control:
  - The RAW-J bracket point fails G1 in version 1 (RF_RAWJ_HI max value/cap 1.059). The ALT (RAWJ) comparison is therefore against a
    control that is not itself usefulness-feasible in version 1.
  - CONT (RF_TASK, at S_k\*) is not spend-matched.
  - The STOPFIT and ALTCRIT points are G1-feasible in both versions.
- **Not a component gain.** PARTIAL_START_ONLY does not isolate the fit path, the ingredient tested. F2, the registered falsifier, is
  triggered (registered prediction: tie, 0.75).
- **Development bounds.** On this grid, a gain beyond the stopped-fit control larger than about 0.012 is not supported (upper
  +0.0117). Beyond the alternating critic the bound is about 0.0025, set by the SE floor. These bounds are conditional on the fitted
  releases, for this procedure on reused people. They are not population bounds and are not adjusted for the adaptive search.
- **Same-checkpoint spend (descriptive and post hoc; not a registered endpoint).** In every fit the STOPFIT and ALTCRIT HI points are
  saved at S_k\*. At that checkpoint they spent 0.00018-0.00057 nats more MODEL_FIT out-of-fold total NLL than FULL, each 3.1-6.1
  paired household-clustered SE above zero (`fam/FIT_RF_*.json`, `selection.same_checkpoint_spend_vs_cand`):

  | Fit | STOPFIT minus FULL (se) | ALTCRIT minus FULL (se) |
  |---|---|---|
  | v1, version 0 | 0.000218 (0.000054) | 0.000183 (0.000058) |
  | v1, version 1 | 0.000571 (0.000093) | 0.000466 (0.000096) |
  | R1, version 0 | 0.000258 (0.000063) | 0.000328 (0.000083) |
  | R1, version 1 | 0.000434 (0.000085) | 0.000517 (0.000094) |

  - On MODEL_SELECT the compared points also spent more than the candidate (`D2.residual_spend_mismatch.at_bracket`): in v1, STOPFIT
    0.00039 (se 0.00009) and ALTCRIT 0.00041 (0.00010); in R1, 0.00034 (0.00010) and 0.00040 (0.00012).
  - So, at equal step count, the fit path reached recovery not separated from its controls' at a slightly lower task cost. This is a
    very small, resolved cost difference, not a recovery difference, and it does not change D2.
  (An earlier draft wrongly called these differences unresolved point estimates without recorded uncertainty; verifier hand-back 4,
  D-A1.)
- **Second split (ROBUSTNESS_NOT_CONFIRMED).** This is descriptive and never changes D2. The refit is genuinely effective (6,912
  checks).
  - The registered benefit over NEURAL_RAW was not confirmed on the second split. On the robustness cells it is +0.0120 (se 0.0075),
    not resolved, and all cells together are point only.
  - The point benefits changed sign against RF_STOP_HI (+0.0028 to -0.0009), RF_ALTC_HI (+0.0005 to -0.0023) and SUCC (+0.0037 to
    -0.0036).
  - A single-view recipient condition failed on the robustness cells against:
    - RF_RAWJ_HI (income view);
    - H_LJ (occupation);
    - RETIRED_B3CTL (both views);
    - SUCC (occupation).
  - Coverage: B2TWIN, TWJ, W, TR100 and TT100 were not in the robustness bank (NOT_RUN).
- **Enhanced readers.** With the enhanced reader set, the point benefit over RF_RAWJ_HI is not positive
  (READER_SENSITIVE_ENHANCED_RF_RAWJ_HI). It was not positive in the registered reading either.

#### R1 (ADAPTIVE_R1)

Against NEURAL_RAW the benefit is not resolved: +0.0098 (se 0.0054, point only).

| D2 comparison | Compared at | Registered pair benefit (se) | Second split: robustness-cell mean (se) |
|---|---|---|---|
| START | NEURAL_RAW | +0.0098 (0.0054), point only | +0.0124 (0.0070), not confirmed |
| CONT (not spend-matched) | RF1_TASK | +0.0090 (0.0051) | +0.0118 (0.0064) |
| STOPFIT | RF1_STOP_HI | -0.0014 (0.0041) | -0.0007 (0.0048) |
| ALTCRIT | RF1_ALTC_HI | -0.0030 (0.0045) | +0.0016 (0.0049): sign change |
| RAWJ | RF1_RAWJ_HI | -0.0046 (0.0052) | -0.0011 (0.0077) |

- **The controls' usefulness.** RF1_RAWJ_HI fails G1 in version 1 (max value/cap 1.137), and CONT is not spend-matched.
- **Second split (ROBUSTNESS_NOT_CONFIRMED).** The refit is genuinely effective.
  - The point benefits changed sign against RF1_ALTC_HI (-0.0030 to +0.0016) and SUCC (+0.0008 to -0.0031).
  - A single-view recipient condition failed on the robustness cells against:
    - RF1_RAWJ_HI (income);
    - H_LJ (occupation, resolved worse);
    - RETIRED_B3CTL (income; occupation resolved worse);
    - SUCC (occupation, resolved worse).
  - The same five references were not in the robustness bank.
- **Enhanced readers.** The point benefit with the enhanced reader set is not positive against RF1_STOP_HI, RF1_ALTC_HI and
  RF1_RAWJ_HI (READER_SENSITIVE_ENHANCED flags). None of these comparisons was a pass.
- **Family-change reading FC_R1** (beside, never gating; RULES_ADDENDUM_1_R1). This compares RF1_FULL with the v1 candidate RF_FULL.
  - It reads ADAPTIVE_R1__FAMILY_CHANGE_NOT_POSITIVE: -0.0029 (se 0.0046; version means -0.0016 / -0.0042).
  - Spend state: R1_SPENDS_MORE_UNRESOLVED (+0.00019 +- 0.00013 total NLL).
  - The men's occupation Brier is resolved higher (+0.00041 +- 0.00008).
  - The income view is an exact alias; the occupation view reads RECIPIENT_AGG_EDGE.

  R1's pair recovery was not lower than v1's.
- **Adaptive targeting.** R1 was designed after v1's MODEL_SELECT winners were seen. Its family targets the recipes that won there,
  so any fall in those recipes' recovery on the same SCORE households would be adaptively targeted.

#### Learning curve

This covers both readings, under the point rule. The half-budget grid validity is effective (1,536 checks in v1, 1,728 in R1).

- **v1.** Limitations against RF_TASK (change -0.0011, se 0.0111), RF_ALTC_HI (-0.0001, 0.0037) and H_LJ (-0.0045, 0.0132). For each,
  the point benefit was smaller with readers fitted on the full FIT than on its half; none is resolved. Against START the benefit was
  larger at the full budget (+0.0038, 0.0081).
- **R1.** R1's learning-curve limitation is against H_LJ only (-0.0067, se 0.0116, point only). The point estimates against
  RF1_STOP_HI (-0.0014) and RF1_ALTC_HI (-0.0021) also fell, but from non-positive full-budget benefits, so they are not limitations
  by the rule.
- **Budgets.** The inner episodes used about the evaluation's own FIT budget (about 4,096 rows against about 4,002), the setting most
  favourable to this mechanism. Budgets above FIT were not tested.

#### Falsifiers

The descriptive inputs are in `claims/<reading>_final/FALSIFIER_DETAIL.json`.

- **F0 and F1: not triggered.** MECHANISM_ACTIVE in every fit, and the orthogonal share never falls below 0.05 at any logged step up
  to k\*. Its minimum is 0.08595, in R1 version 0.
- **F2: triggered in both readings.** There is no resolved benefit over STOPFIT.
- **F3: triggered in both readings,** on tiny descriptive margins with no uncertainty.
  - v1: the LOG_LR gain over RF_STOP_HI is +0.0007, against a tree/table mean of -0.0002.
  - R1: +0.0003 against -0.0005.
  - LOOKUP's gain is a structural 0.0, and that zero enters the tree/table mean: on continuous outputs every SCORE wire is unseen by
    the table.
  - The R1 card's per-recipe falls are measured against START, a different contrast from F3's.
- **F4: triggered in both readings** (the learning-curve limitation).
- **F5: not triggered.** The code implements only the "mostly negative orientation" disjunct; the negative-orientation fraction is 0.0
  in every logged term. The other disjunct, an unchanged oriented fresh AUC:
  - is not met in v1, where the START comparison is resolved;
  - is not excluded at the resolved level in R1, where the START benefit is point only (lower -0.0010).
- **F6: not triggered.** It was only weakly assessable in version 0 (2-3 logged steps and 1 monitor, because k\* came early).

### 3. Strong references

NOT_SUPERIOR in both readings.

| Reading | Resolved below, with both recipient gates | Resolved below, recipient gate fails | Not resolved |
|---|---|---|---|
| v1 | NEURAL_RAW, W, TR100, TT100 | TWJ (occupation gate) | B2TWIN, H_LJ, SUCC (point only, occupation gates fail); RETIRED_B3CTL (not positive) |
| R1 | W, TR100, TT100 | TWJ (occupation gate) | NEURAL_RAW (point only); B2TWIN, H_LJ, SUCC (point only, occupation gates fail); RETIRED_B3CTL (not positive) |

- **Where the recipient gates fail.** On the **occupation** view. The candidate's occupation view (version means 0.5971 / 0.5969 in
  v1; 0.5979 / 0.5983 in R1) sits against the tree-based references' shared occupation output at 0.5665 / 0.5682. That is r = +0.031
  / +0.029 in v1 (+0.031 / +0.030 in R1), above the 0.010 allowance.
- **Income.** On the registered cells income passes; the largest income r is RETIRED_B3CTL's in version 0, +0.0077. On the
  robustness cells, RETIRED_B3CTL's income gate also fails.
- **The references' own cost.** Their pair recovery comes from a different occupation output, at their own usefulness cost:
  - Their tightest condition is at 0.883 / 0.788 of its cap (max value/cap, versions 0 / 1), the men's guard at 0.00442 / 0.00394 of
    0.005.
  - RETIRED_B3CTL's tightest condition is at 0.883 in both versions.
  - The values come from the frozen reference usefulness records and are tabulated in `TABLES.md` (the reference table).

### 4. Material target

NOT_MET in both readings.
- v1: +0.0127 against NEURAL_RAW and +0.0168 against H_LJ, neither resolved above 0.020.
- R1: +0.0098 and +0.0139.

### 5. Joint design and novelty

Not applicable (no local twin is built); not claimed.

## Registered expectations against outcomes

Each expectation is judged by its own registered definition.

### PROTOCOL.md section 12 (coordinator, before any label read)

| Registered expectation | Probability | Outcome |
|---|---|---|
| CAND G1-feasible at the point | 0.7 | yes, in both readings |
| the fit path changes the release measurably: CAND and STOPFIT outputs differ beyond rounding | 0.9 | yes: the occupation wires differ, with no exact alias in any grid. Occupation-view version means are 0.5971 / 0.5969 against RF_STOP_HI's 0.5986 / 0.5989 (R1: 0.5979 / 0.5983 against RF1_STOP_HI's 0.6002 / 0.6011) |
| D2 against STOPFIT resolved | 0.10 | no |
| D2 against all gating controls | 0.06 | no |
| D3 | 0.02 | no |
| D4 | 0.01 | no |

The registered most likely outcome was: "CAND lowers NEURAL_RAW's occupation view by 0.005-0.015 at the point, close to STOPFIT and
ALT at matched spend; a bounded negative experiment." The outcome was close to that:
- The occupation view fell 0.012 / 0.016 (version 1 just above the predicted range).
- That is close to RF_STOP_HI's fall (0.010 / 0.014) and RF_RAWJ_HI's (0.005 / 0.0155).
- The outcome class is a bounded negative experiment.

### Card RF_RFIT_v1, section 6 (all 11 predictions)

| Prediction | Probability | Outcome |
|---|---|---|
| F0 active | 0.8 | yes |
| k\* between S20 and S200 | 0.7 | yes (S20 / S70; S20 is the boundary) |
| CAND G1-feasible on MODEL_SELECT at the point | 0.7 | yes |
| point pair benefit over START between +0.003 and +0.020 | 0.6 | yes (+0.0127) |
| resolved over START | 0.3 | yes on the registered cells; not confirmed on the second split |
| FULL vs STOPFIT at matched spend: tie (NOT_SEPARATED) | 0.75 | yes |
| FULL vs STOPFIT: resolved FULL benefit | 0.10 | no |
| resolved FULL benefit over ALT | 0.10 | no |
| D3 | 0.02 | no |
| D4 | 0.01 | no |
| F4 triggered given a positive benefit | 0.5 | yes (point rule) |

### Card RF_RFIT_R1, section 3 (all 9 predictions, registered before the R1 fit)

| Prediction | Probability | Outcome |
|---|---|---|
| F0 active, with a larger orthogonal share than v1's | 0.6 | no: active, but 0.098 / 0.101 against v1's 0.102 / 0.112 |
| RF1_FULL G1-feasible on MODEL_SELECT at the point | 0.7 | yes |
| FULL_R1 resolved over STOPFIT_R1 at matched spend | 0.10 | no |
| FULL_R1's pair AUC below v1 RF_FULL's at the point | 0.45 | no: it was higher (FC_R1 -0.0029; version means 0.6115 / 0.6066 against 0.6099 / 0.6024) |
| the ET25 / CATI1 / HGB15 pair AUCs fall more than under v1 (falls against START, as in the card) | 0.45 | no: only CATI1 fell more. ET25 -0.0086 against v1's -0.0088; CATI1 -0.0094 against -0.0091; HGB15 -0.0048 against -0.0051 (`claims/PER_RECIPE_PAIR_MEANS_corrected.json`; deviation 11) |
| D2 met | 0.04 | no |
| D3 | 0.03 | no |
| D4 | 0.01 | no |
| the learning-curve limitation again | 0.6 | yes (H_LJ, point) |

## What is not claimed

- No protection against arbitrary readers, and no information removal from a lower fitted-reader AUC.
- No confirmation, and no replication from versions, repetitions or reader seeds. Repeated reader splits and seeds on the same people
  are not replication.
- No held-out evidence from the inner episodes, which are all MODEL_FIT training data.
- No mechanism contribution, competitiveness or novelty.
- Nothing about the closed roles; no anonymity.

## Interpretation (not a tested claim)

The gap to the strong references sits in the occupation output. In the tree-based references that output is about 0.03 AUC less
recoverable, at their own men's-guard cost. On this evidence, closing the gap looks like a question about the output representation
rather than about the critic. This is an interpretation; the campaign did not test it.
