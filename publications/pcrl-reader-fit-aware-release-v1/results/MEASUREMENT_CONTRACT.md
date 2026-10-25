# Measurement contract (campaign `pcrl_reader_fit_aware_release_v1`, RFA1)

Coordinator. Written 2026-10-25 from 21:06Z (`date -u`), before any outcome of this campaign and before any label read. The statistical
reviewer's check is in `roles/statistical_reviewer/` and its adopted changes are listed in section 9. Historical numbers and labels of
earlier campaigns are not changed; the interpretive qualifications of section 6 and 7 are an addendum, not edits.

Setting, unchanged from the admitted chain: Florida ACS 2019 development cohort. A service predicts an income class (2) and an
occupation group (5) from six fields. Each recipient receives only its own probability vector and argmax decision; the linked-pair
recipient is granted both outputs of the same person and the specified labelled auxiliary sample. These grants are assumptions of the
benchmark, not claims about a deployment. The recovered variable is the recorded SEX column. The adversary is not new.

## 1. Primary target

The primary quantity, for release r, version v and recipient view u (income, occupation, linked pair), is the AUC on SCORE households
of the SEX reader produced by the **declared finite-data reader procedure**:
1. fit each recipe of the admitted 16-recipe bank (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP, ET5, ET25,
   HGB15, HGB31, MLP64, MLP6432, CATI01, CATI1) on the released outputs and SEX of the FIT households (the declared labelled auxiliary
   sample);
2. orient and select on the VALIDATION households (max oriented VALIDATION AUC; SCORE never used for selection or orientation). For the
   pair, the winner is chosen among the pair readers and the ignore-one-recipient readers;
3. score the selected, oriented reader once on the SCORE households.

FIT, VALIDATION and SCORE are household-disjoint parts of MODEL_SELECT (a new RFA1 deal; `READER_SPLITS.json`). The recipes, seeds and
splits are frozen before any outcome.

**Equal opportunities.** Candidates, controls and references get the same cells, recipes, seeds, selection rule and bootstrap. No reader
is chosen per release. An identical released wire is evaluated and counted once. Any mechanism-specific reader is reported beside and
never gates. Each recipient view and the linked pair are reported separately; the coalition may ignore either view.

**Meaning.** A lower AUC is evidence about this procedure, this auxiliary budget and these recipes on these households. It is not an
arbitrary-attacker privacy guarantee, not evidence that information was removed, and not a population-optimal recovery value.

## 2. Usefulness and the material target (unchanged definitions)

- **Usefulness.** The 28 original conditions on MODEL_SELECT: per version, 12 task clauses (accuracy loss <= 0.010, NLL excess <= 0.010
  nats, summed Brier excess <= 0.005, for 2 tasks against U0_S<v> and Ucal_S<v>) and 2 within-SEX occupation summed-Brier guards
  against U0 (<= 0.005). Definitions are the authoritative `nrm.rules` / `sar.evaluate` ones, verified by the statistical reviewer.
- **Three separate usefulness results.** Point feasibility G1 (all 28 at the point) gates. The confidence conjunction and the inherited
  two-SE reserve are reported beside and never gate; no reserve rule removes an otherwise eligible reference.
- **Recipient conditions.** A pair benefit counts only with the recipient gate: each single view not more recoverable than the
  comparator's by more than 0.010 per version at the aggregate point (`nrm.rules.recipient_screen`).
- **Material target.** A resolved pair benefit of at least 0.020 against both NEURAL_RAW and H_LJ, with the recipient gates; never
  weakened.

## 3. Fresh readers, second split and budget sensitivity

- Evaluation readers are fitted from scratch on MODEL_SELECT FIT parts. They never reuse an inner training reader, its weights or its
  features.
- **Second split.** A useful, promising release is also read on robustness repetitions R3 / R4 (different household deals of the same
  people) with reader seeds k2 / k3, without retuning anything.
- **Budget sensitivity.** A learning curve uses two prospectively fixed auxiliary sizes: the declared FIT (parts 0-4), and a fixed half
  of the FIT households (by hash rank). VALIDATION and SCORE are unchanged. If a candidate's benefit shrinks at the larger budget, that
  is reported as a limitation (a learnability effect), never presented as information removal.

## 4. Inner episodes are training data

Every auxiliary-training and auxiliary-evaluation episode used inside release fitting is drawn from MODEL_FIT. Their labels shape the
release's updates, so they are training data of the release mechanism. Inner surrogate values, inner held-out AUCs and MODEL_FIT
out-of-fold reader checks are training diagnostics, never held-out evidence about the release.

## 5. Empirical optimal AUC is a diagnostic only

The exact optimal-reader AUC of an empirical finite law (same-sample or nested) is not used as an objective, selector, gate or claim in
this campaign. All releases here are continuous network outputs, for which a same-sample empirical law is degenerate. If any
finite-support diagnostic is computed, it is labelled a diagnostic. It is never a population value, an upper confidence bound or a
cardinality correction.

## 6. Qualification of the estimator diagnostic's recommendation (addendum; EDG1 text unchanged)

EDG1 (publication d987e3685) recommended accepting a direction across supports only from the estimator whose support bias opposes it.
That recommendation is qualified here:
- The sign of each estimator's bias for one release does not by itself determine the bias of the difference between two releases'
  estimates. Both biases depend on each release's support, occupancy, class balance and sample size, and they can cancel or reverse.
- An opposing bias in expectation does not validate a significance test. The size of a test also depends on the variance, the
  finite-sample distribution and the dependence between the two estimates. EDG1 calibrated only the rules and settings it simulated.
- EDG1's refinement findings concern refinements of one law (identical recovery, more cells). They supply no universal ordering rule
  for unrelated releases with different laws.

Its finding that fitted readers can lose more on harder-to-learn outputs remains relevant here. It motivates the budget-sensitivity
reading of section 3, not a correction.

## 7. DRF1's objective was not tested (addendum)

DRF1 optimized the exact optimal-reader AUC of a **propensity-estimated** law: in each legal cell, SEX was represented by MODEL_FIT
propensities. EDG1 studied **empirical label-based** laws, so it did not test DRF1's actual objective. The mechanism selected here fits
its inner readers on realized MODEL_FIT SEX labels and does not use a propensity-estimated law. This campaign therefore does not test
that objective either. Its estimator properties remain unresolved, and no synthetic diagnostic campaign is launched for it.

## 8. Uncertainty and what the intervals mean

- **Household uncertainty.** Household bootstrap of SCORE (B = 400) within each cell gives se_hh.
- **Reader-fitting and split variation.** The spread over repetitions R1 / R2 and reader seeds k0 / k1 gives se_rs. Repetitions deal
  the same people differently, so they are not independent replications.
- **Release-fitting variation.** Versions v0 / v1 are two procedure repeats: different U0 / NEURAL_RAW seeds and inner-episode seeds,
  the same training rows and the same MODEL_SELECT people. They are reported separately and must agree in sign. Two repeats do not
  estimate release-fitting variance, and they are never called replications.
- **The screen.** The inherited aggregate screen is unchanged: block means over version x repetition, se_agg = max(sqrt(se_hh^2 +
  se_rs^2 + se_mc^2), 0.001); a benefit is resolved iff mean - 2 se_agg > threshold and both version means exceed the threshold. Overlap
  between folds or cells is never treated as independent replication.
- **Scope.** All results are exploratory on heavily reused development records. The intervals do not correct the accumulated adaptive
  search across this and earlier campaigns.

## 9. Statistical reviewer's check and the adopted corrections (binding; sections 1-8 are not edited)

The reviewer's wave-1 review (`roles/statistical_reviewer/HANDBACK_1.md`, saved verbatim) found the contract sound, with 7 REQUIRED
corrections and 9 ADVISORY items. All are adopted. Where a sentence of sections 1-8 conflicts with this section, this section governs.

- **A1 (se_rs).** se_rs is the spread of the four version x repetition block means, with reader seeds averaged within each block. It
  mixes the version difference (two release-fitting repeats), the reader-split difference and their interaction. Reader-seed
  variation enters only through the block means. Each factor has about one degree of freedom, so se_rs is a crude component and the
  screen is approximate. It is not an estimate of release-fitting variance.
- **A2 (se_hh and other SEs).**
  - Each cell's SCORE bootstrap (B = 400) holds fixed the release, the fitted readers, the orientation and the VALIDATION-selected
    winner.
  - se_hh is the root mean square of the cell SEs, weighted as in the aggregate. It is an upper bound under cross-cell correlation, and
    it does not shrink as cells are added. It is conditional on everything fitted: it does not cover refitting readers, re-selecting
    winners or refitting the release.
  - Usefulness SEs are household-clustered over MODEL_SELECT with the release fixed.
  - Bracket SEs, from MODEL_FIT out-of-fold paired costs, are a selection device only.
- **A3 (definitions).** The usefulness metrics are defined by this chain:
  - `scm_bench/selection.py::clauses` and `scm_bench/common.py` (CAPS, task_vectors);
  - `ihs/evaluate.py::utility` (household-clustered SEs and the two guards) and `ihs/common.py` (GUARD);
  - `ohn/evaluate.py::complete_eval`, reached from `rfa/evaluate.py::utility_of` unchanged;
  - `tsz/evaluate.py::refs_of` for U0 and Ucal;
  - `ohn/rules.py::conditions` and `nrm/rules.py::point_eligibility` for gating.

  The recipient gate is `nrm/rules.py::recipient_screen`. The verifier hash-checks these files.
- **A4 (material target).** Against each of NEURAL_RAW and H_LJ, with both recipient gates: aggregate mean - 2 se_agg > 0.020 and both
  version means > 0.020. At se_agg near 0.010 this needs a point benefit near 0.040.
- **A5 (budget sensitivity).**
  - **HALF.** The fixed fold rule of `rfa/devreaders.py::lc_fit`: FIT households in folds 0-1 of 4 household folds, tag
    `RFA1_devgrid_LC`, nested inside FIT. It reuses the recipes, seeds, VALIDATION, SCORE and the same bootstrap draws, so the FULL - HALF
    difference has a paired SE.
  - **Reading.** Per comparator (START, CONT, every gating bracket or control point, H_LJ), d_LC = b(FULL) - b(HALF). A limitation is
    flagged iff the full-budget point benefit is > 0 and the mean d_LC is < 0; the resolved level is reported beside.
  - **Converse.** No shrinkage between two budgets is not evidence of information removal. Budgets above the declared FIT are untested.
  - **Inner budget.** The inner-episode training size is recorded in the card: about 4,096 A rows, about the declared FIT size.
  - The 0.25 fraction is not run.
- **A6 (second split).** D2R is read at CAND whenever CAND is G1-feasible and the registered grid is valid, whatever D2 shows. It is
  descriptive and never changes D2, and it requires the recipient gates on the robustness cells. If the budget prevents it, it reads
  NOT_RUN and is disclosed.
- **A7 (scope).** CAND and its controls are continuous network outputs; some references are few-valued. No empirical optimal AUC of
  any release is used as an objective, selector, gate or claim.
- **A10 (release-side opportunities).** CAND's operating point and every control's saved points are selected on MODEL_FIT only, by
  rules declared in the card before any MODEL_SELECT read. CAND's rule uses usefulness spends only. The F0 mechanism gate is a SEX-based
  MODEL_FIT diagnostic that decides only whether the family is fitted, not an operating point; it is disclosed.
- **Advisory, adopted.**
  - **A8.** EDG1's support-specific numbers do not transfer to a continuous CAND. Fitted-reader comparisons between continuous and
    few-valued releases can favour the harder-to-learn release; the learning curve against H_LJ is the planned check. EDG1's coverage
    figures do not calibrate this campaign's aggregate screen.
  - **A9.** The final card states that the inner readers are fitted on realized MODEL_FIT SEX labels (verified).
  - **A11.** "Scored once" means once per (release, cell, budget). SCORE is reused across releases, budgets and the ignore-one
    candidates, which makes the comparisons paired. VALIDATION values are never reported as recovery.
  - **A12.** The spread of inner episodes or MODEL_FIT folds is never an SE or a replication.
  - **A13.** The recipient gate is a point rule: a true 0.010 single-view worsening passes it about a third of the time. The confidence
    conjunction and the reserve are defined in RULES.md section 2.
  - **A14.** The screen requires both version means to exceed the threshold (> 0 for D2 and > 0.020 for D4), not only sign agreement.
  - **A15.** The `rfa/devreaders.py` docstring was corrected to bootstrap seed 2026102520.
  - **A16.** Dependence on optimizer and inner-episode seeds beyond the two versions is not covered by any SE.
