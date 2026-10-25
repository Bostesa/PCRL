(Saved verbatim by the coordinator from the statistical reviewer's hand-back of 2026-10-25; the harness did not let the reviewer
write report files. The reviewer reported about 6-8 CPU-seconds.)

# Claims review of the DRF1 decision drafts (statistical reviewer)

Review started 2026-10-25T04:41:56Z and completed 04:51:34Z (`date -u`). Drafts reviewed at commit e0c661652:
`RESEARCH_DECISION.md` and `START_HERE.md`. I confirmed with `git diff` that they, `TABLES.md`, `claims/` and `dr/` are
identical at the later HEAD d939c2dd8.

Method: label-free. I read the committed JSON records with small read-only scripts kept in my scratch directory (one
numerical thread, about 6-8 CPU-s). Records read: claims/PLAN_* (through DECISIONS.json), claims/DRF_V1, DRF_R1 and DRF_R2
DECISIONS.json, both D2R.json, CAMPAIGN_SUMMARY.json, TABLES.md, dr/FIT_*.json, dr/OPTIMISTIC_V*.json, dr/BUDGETS.json, the
prediction lines of the three cards, PROTOCOL.md section 12, the custody reader records, and the verifier's
VERIFICATION_REPORT.md, HANDBACK_1.md and HANDBACK_2.md. No label read, no queue unit, no agent; nothing regenerated.

**Verdict: not ready yet.** The decision (no method claim; registered headline R1 at STAGE_3; v1 at STAGE_2; R2 adaptive) is
correct, and every label and almost every number matches the records. Eleven REQUIRED corrections remove false or overstated
statements before publication.

## Checks requested

1. **Numbers and labels.**
   - These match the records: every D1 / D2 / FC / TWIN / D3 / D4 / D2R label and number in the tables and answers 3-6; the
     recovery table (TABLES.md 2); the usefulness values; registration times; budgets; the optimistic in-sample values; the
     verifier and custody statements.
   - These do not (R1, R2, R3): the binding-constraint statements, the "occupation view" attribution of the objective
     difference, and the reason given for `held_out_survives: []`.
2. **Failure stage.** Correctly assigned by the registered rule: R1 L3 with a useful in-sample optimistic point gives STAGE_3;
   v1 L2 gives STAGE_2. A sensitivity note is recommended (C8).
3. **Separate outcomes.**
   - The three outcomes are kept separate.
   - FC and TWIN never appear inside a D2 sentence, but the TWIN sentence sits in the D2 bullet of answer 4 (C1).
4. **Registered versus adaptive.**
   - Correct in the records: v1 and R1 are both registered (cards 03:14:30Z and 03:27:33Z, before the first MODEL_SELECT read at
     03:34:32Z). R1 is the headline and v1 is superseded (CAMPAIGN_SUMMARY flag MULTIPLE_REGISTERED_READINGS). R2 is
     ADAPTIVE_R2.
   - Wrong in the drafts: answer 3's heading calls v1 "the registered mapping" (R4), and the expectations section labels R2's
     adaptive predictions as registered (R11).
5. **Uncertainty section.** se_rs (fitting variation) is described under household sampling (R10).
6. **Overclaiming.**
   - "reproduced the optimistic pattern" is unsupported (R4).
   - "tied" understates two failed recipient conditions against SUCC (R4, R5).
   - "confirmed the gain" needs its qualifier (R9).
   - Answer 7 is not precise yet (R6, R7, R8).
   - "reproduced" for the reader refits is supported, as computational reproducibility only (C5).
7. **Forecasts.**
   - Answer 3 presents the optimistic diagnostic as reproduced (R4).
   - The nested-estimator agreement should be worded as one check, not as validation (C3).
   - Everything else is labelled as a diagnostic or an expectation.
8. **Answer 7.** Not accurate enough as drafted: wrong "same constraints", an imprecise "not cost-matched", an overstated
   "does not allow", a version-specific recipient difference stated generally, and an incomplete requirement for a method claim
   (R1, R6, R7, R8).

## REQUIRED corrections

**R1. Binding constraints (answer 4, answer 7).**
- Problem: "Both stop at the same binding constraints (occupation NLL, then the men's guard)" is false, and incomplete.
  - The highest-ratio constraint differs by objective in version 1. R1: DR_AUC the men's guard (0.9998), DR_MI occupation NLL.
    R2: DR_AUC occupation NLL, DR_MI the men's guard.
  - In every R1 / R2 fit, both objectives end with three budgets at or near their limits (in-sample MODEL_FIT spend over
    budget): occupation NLL 0.999-1.000, the men's occupation Brier 0.984-1.000 and income Brier 0.990-0.999.
- Replace in answer 4 with: "Both exhaust the same three budgets: in every R1 / R2 fit, occupation NLL (0.999-1.000 of budget),
  the men's occupation Brier (0.984-1.000) and income Brier (0.990-0.999) are at or near their limits for both objectives. The
  constraint with the highest ratio differs by objective in version 1 (R1: DR_AUC the men's guard, DR_MI occupation NLL; R2:
  the reverse)."
- In answer 7, replace "The binding constraints are occupation NLL and the men's occupation Brier guard." with "The constraints
  at budget are occupation NLL, the men's occupation Brier guard and income Brier (in-sample MODEL_FIT ratios 0.98-1.00 in every
  R1 / R2 fit)."

**R2. Answer 4: where the objective difference comes from.**
- Problem: "Most of the AUC objective's lower value comes from the occupation view" holds for version 0 only. R2 version 0: the
  J difference is 0.0040, of which occupation 0.0030. Version 1: the J difference is 0.0092, of which pair 0.0053 and occupation
  0.0034.
- Replace with: "In version 0 most of the J difference comes from the occupation view (0.0030 of 0.0040); in version 1 most
  comes from the pair (0.0053 of 0.0092; occupation 0.0034)."

**R3. Answer 2 and START_HERE: why the optimistic point did not survive held-out evaluation.**
- Problem: `held_out_survives: []` is not a budget result.
  - The rule's nested mode compares recovery at matched nested total-NLL spend and reuses the in-sample `within_budget` flag.
  - On the nested estimator OPT_L's held-out exact pair AUC* (0.6868 / 0.6794) is 0.0443 / 0.0410 above the SEX-blind path at
    matched nested spend (bracket path steps 67 / 68, spend 0.040 / 0.042 against OPT_L's 0.038 / 0.038).
  - The LOOKUP values quoted in the draft (0.6116 / 0.5979) are a different held-out measure.
- Replace answer 2's nested sentence with: "On the nested MODEL_FIT estimator OPT_L was not useful: its held-out exact pair AUC*
  (0.6868 / 0.6794) was 0.044 / 0.041 above the SEX-blind coarsening path's at matched nested total-NLL spend
  (`held_out_survives: []`); its held-out LOOKUP pair AUC was 0.6116 / 0.5979. Its nested spends were also 2.6-4.0 times the
  budgets (occupation NLL 0.0277 / 0.0262 against 0.0097 / 0.0100; men's occupation Brier 0.0119 / 0.0114 against
  0.0035 / 0.0029), so no held-out point is within budget either; the rule's nested label does not itself test the budgets."
- In START_HERE replace "None of it survived the nested held-out estimator, where it ran about three times over the occupation
  budgets." with: "It was not useful on the nested held-out estimator: there its held-out exact pair AUC* was 0.044 / 0.041 above
  the SEX-blind path at matched nested spend, and its nested spends were 2.6-2.9 times the occupation NLL budget and 3.4-4.0 times
  the men's guard budget."

**R4. Answer 3: the heading, "reproduced", and SUCC.**
- Problem: the heading calls v1 "the registered mapping", although R1 is the registered headline.
- Replace the heading with: "**The first registered mapping (v1) did not keep usefulness. The registered headline R1, a
  revision decided on MODEL_FIT evidence before any MODEL_SELECT outcome, kept all 28 conditions at the point. It did not
  reach below the strong start, and it is not a reproduction of the optimistic point.**"
- Problem: the bullet "It reproduced the optimistic pattern ... It did not reach below SUCC, ... resolved." is unsupported.
  - The optimistic point was useful in-sample only, was not useful on the nested estimator, and is an exact AUC* on MODEL_FIT,
    not reader AUC on MODEL_SELECT.
  - "Tied" omits that both recipient conditions against SUCC fail: income version 1 +0.0129, occupation version 0 +0.0167;
    FC recipient labels RECIPIENT_AGG_POINT_FAILED twice.
- Replace that bullet with: "On MODEL_SELECT it is resolved less recoverable than the SEX-blind coarsening bracket at matched
  total NLL (+0.0340, se 0.0096), but only on that primary cost (answer 4). This is not a reproduction of the optimistic point,
  which was useful in-sample only and not on the nested estimator. It did not reach below SUCC, the unchanged strong start:
  +0.0008 (se 0.0120), with both single-view recipient conditions failing against SUCC (income version 1 +0.013, occupation
  version 0 +0.017). It also spent 0.0097 (0.0033) more total NLL than SUCC, resolved."

**R5. START_HERE, R1 bullet on SUCC.**
- Replace "Tied with the unchanged strong start SUCC (+0.0008) while spending more total NLL." with: "Not below the unchanged
  strong start SUCC: +0.0008 (se 0.0120), both single-view recipient conditions fail against SUCC (income version 1, occupation
  version 0), and it spends resolved more total NLL (+0.0097, se 0.0033)."

**R6. Answer 4: an untested attribution.**
- Replace "The difference is therefore partly a re-allocation of usefulness spend from occupation to income, not a recovery gain
  at equal cost." with: "The pair benefit is therefore not shown at equal cost on every usefulness metric; part of it may reflect
  a re-allocation of spend from occupation to income (not tested)."

**R7. Answer 7: make the three findings precise.** Replace the three bullets with:
- "**No objective effect is resolved.** DR_AUC vs DR_MI: +0.0076 (se 0.0117; version means -0.0004 / +0.0156) in R1, +0.0039
  (0.0082) in the adaptive R2, and +0.0054 (0.0110) on the R1 robustness cells. At R1's se_agg a true 0.010 benefit would be
  resolved about 13% of the time (about 22% at R2's), so a small effect is neither shown nor excluded. Both objectives exhaust
  the same three budgets (occupation NLL, the men's occupation Brier, income Brier)."
- "**The separation from SEX-blind coarsening holds on total NLL only.** At the bracket DR spends resolved more income NLL
  (+0.0061, se 0.0018) and income Brier (+0.0038, 0.0013), and resolved less occupation NLL and men's occupation Brier
  (SEPARATED_ON_PRIMARY_COST_ONLY). So the +0.034 pair benefit is not shown at equal cost on every metric. The coarsening points
  themselves fail the men's guard on MODEL_SELECT."
- "**The family's MODEL_SELECT recovery is not below the strong start's, at a higher cost.** DR's pair AUC is 0.605-0.613, against
  SUCC 0.605 / 0.612 and RETIRED_B3CTL 0.603 / 0.605: not below either. It spends resolved more total NLL than SUCC. Against
  SUCC its occupation view is 0.017 more recoverable in version 0 and its income view 0.013 more in version 1, both beyond the
  0.010 allowance (the other versions are within it). In-sample, the search stopped with occupation NLL, the men's guard and
  income Brier at budget: within this family, these budgets and this local search the views were not coarsened further. Whether
  another mapping in the family could coarsen them within budget was not tested."

**R8. Answer 7: what a method claim would need.**
- Problem: D2 is omitted, only three of the nine references are named, and D4 is listed as part of the method claim.
- Replace the final paragraph with: "A method claim (METHOD_CANDIDATE) would need D2 and D3. D2: a resolved pair benefit with the
  recipient gates beyond both family controls, with no secondary metric resolved higher. D3: a resolved benefit with the
  recipient gates against every point-eligible reference of the nine, SUCC, RETIRED_B3CTL and H_LJ included. The material target
  D4 (a resolved 0.020 against NEURAL_RAW and H_LJ) is reported separately and is not implied by either. Any of these would then
  need confirmation on households not used here."

**R9. "Confirmed" needs its qualifier (START_HERE and answer 6).**
- Replace the START_HERE robustness bullet with: "**Robustness reader splits** (R3 / R4, with reader seeds k2 / k3 on R1 / R2; no
  retuning). The pair benefits over the SEX-blind coarsening bracket and over the twin were ROBUSTNESS_CONFIRMED; the coarsening
  comparison remains separated on total NLL only, and the twin comparison is never sufficient. The point benefits over the MI
  objective, NEURAL_RAW and H_LJ were not confirmed. The SUCC comparison changed sign (+0.0008 to -0.0002), and a single-view
  recipient condition failed against SUCC and RETIRED_B3CTL."
- In answer 6 (R1), replace "**Confirmed:** the gain over ..." with: "**Confirmed (pair benefit and robustness recipient
  gates):** over the SEX-blind coarsening bracket (+0.0308, 0.0110; still separated on total NLL only) and over the twin
  (+0.0388, 0.0084; never sufficient)." Make the same change for R2.

**R10. Uncertainty section: se_rs belongs under fitting variation.**
- Replace the "Household sampling" bullets with: "**Household sampling.** The reader-cell SEs are household bootstrap SEs over
  SCORE households, with the release, readers and selections fixed; their weighted RMS (se_hh) is the household part of se_agg,
  an upper bound under correlated cells. The usefulness SEs are household-clustered over MODEL_SELECT. Both are conditional on
  everything fitted, and neither covers refitting the release or the readers."
- Open the "Fitting variation" block with: "se_rs, the spread of the four version x repetition block means (version, reader
  split and reader seed, about one degree of freedom each), is the second part of se_agg = max(sqrt(se_hh^2 + se_rs^2 +
  se_mc^2), 0.001), with se_mc = 0 here."
- Add to the optimizer-seed bullet: "the R1-to-R2 change also includes the readers refitted on the changed releases."

**R11. Expectations section: missing headline predictions, and adaptive labelling.**
- Add after the PROTOCOL bullets: "- R1 card predictions (registered; card committed before any MODEL_SELECT outcome):
  P(DR_AUC_R1 meets all 28 conditions on MODEL_SELECT in both versions) 0.5: yes; P(D2) 0.08, P(D3) 0.02, P(D4) 0.01: none
  met."
- Change "R2 card predictions:" to "R2 card predictions (adaptive: written after the R1 MODEL_SELECT outcomes):".

## RECOMMENDED changes

- **C1.** Answer 4: move the TWIN sentence out of the D2 bullet into its own bullet ("Beside D2, never sufficient: ..."), and
  replace "resolved and robust" with "resolved; ROBUSTNESS_CONFIRMED on the robustness cells".
- **C2.** Decision table, v1 row: "DR_DRF_v1 (registered; superseded by the pre-outcome R1 card)", and cite the CAMPAIGN_SUMMARY
  flag MULTIPLE_REGISTERED_READINGS.
- **C3.** Answer 3: replace "agreed with" by "was within 0.0021 / 0.0019 of", and call it "one check of the nested estimator,
  not a validation".
- **C4.** Answer 3: replace "about 1.1 times the budgets" with "DR_AUC_R1's nested spends were 1.04-1.12 times the budgets on
  its exhausted metrics (occupation NLL 1.11, men's guard 1.12, income Brier 1.04; income NLL 0.99); DR_MI_R1 up to 1.21".
- **C5.** Answer 6, "Reader refits": label these computational reproducibility (same cells, seeds and data), not robustness
  evidence.
- **C6.** Replace "The second household reader split" with "the robustness splits R3 and R4 (24 robustness cells per
  comparator)".
- **C7.** D2R duplicate: name the cause. The duplicate comes from `drf.rules.d2r`, the reviewer's code: it lists START (SUCC) and
  then the nine references, which include SUCC again. Labels and sets are unaffected (the family label uses the decisive set;
  per-comparator entries are keyed once); only the flag and caveat strings repeat SUCC. Disclose it in PROTOCOL_DEVIATIONS as a
  cosmetic rules-code defect, not patched in the frozen rules.
- **C8.** Stage sensitivity (not registered), after the Decision:
  - The registered rule judges the optimistic point in-sample. On the nested mode no optimistic point is useful, and under that
    reading v1 and R1 would both be stage 1.
  - Say which held-out measure the nested label used: exact AUC*, not LOOKUP.
  - The held-out exact AUC* is strongly optimistic for fine partitions (the twin: 0.7475 nested exact against 0.6303 LOOKUP and
    0.6812 in-sample), so the nested comparison favours coarser releases.
- **C9.** Answer 5: tag the R2 lines "(adaptive)".
- **C10.** START_HERE wording:
  - R1 bullet: add "(at the point only: 20 of 28 pass the confidence conjunction, none violated; not reserve-qualified)".
  - v1 bullet: "failed the men's occupation guard and occupation NLL (and occupation Brier in version 1)".
  - Replace "Superior only to NEURAL_RAW" with "Resolved below NEURAL_RAW only (+0.023, se 0.011; recipient gates pass)".
- **C11.** Expectations section:
  - The v1 card predicted the in-sample exact *pair AUC* below DR_MI's, while PROTOCOL 12 says *objective*. Both readings give
    the same outcome (v1 no, R1 / R2 yes); note the wording.
  - The "binds first" bullet: add that v1's DR_AUC version 1 bound on income Brier.
  - Report the rest of the R2 card prediction: J_mi lowered by 0.0015 / 0.0022, and DR_MI_R2's in-sample pair AUC* moved
    +0.0017 / -0.0031.

## Verified as stated (no change needed)

- **Labels.** R1 and R2: D2 NOT_SEPARATED (CTRL frontier SEPARATED_ON_PRIMARY_COST_ONLY at DR_CTRL_HI; OBJ
  NO_RESOLVED_BENEFIT_VS_OBJ_MI); FC FAMILY_CHANGE_NOT_RESOLVED; TWIN BELOW_TWIN_RESOLVED; D3 NOT_SUPERIOR (passed NEURAL_RAW;
  R2 also TWJ); D4 NOT_MET (NEURAL_RAW point only, H_LJ not met); class FEASIBLE_NO_RELIABLE_GAIN. OBJ spend states: R1
  OBJ_MI_SPENDS_LESS_UNRESOLVED (d +0.0011, se 0.0021), R2 OBJ_MI_SPENDS_AT_LEAST_AS_MUCH.
- **v1.** CAND_INFEASIBLE, men's guard +0.0080 / +0.0111, reader stage STOPPED_USEFULNESS_FAILURE, class TASK_INFEASIBLE.
- **Paired numbers.** All paired numbers in answers 3-5 match TABLES.md 3 and DECISIONS.json.
- **D2R numbers** (R1 and R2) match the D2R records. The sign change against SUCC and the recipient failures (SUCC,
  RETIRED_B3CTL) are correct; both refits are REFIT_GENUINELY_EFFECTIVE; grid validity READER_SEEDS_EFFECTIVE with 82 explained
  equalities per reading.
- **R1 usefulness.** Worst value / cap 0.564 / 0.692; men's guard +0.0028 / +0.0035; 20 of 28 confidence passes, none violated;
  not reserve-qualified.
- **Budgets** frozen at 03:12:27Z, before the first fit (03:20:30Z).
- **R2 search.** 4 to 12 seeds, nothing else changed (card and verifier).
- **R2 MODEL_FIT values.** In-sample 0.6212 vs 0.6216 and 0.6205 vs 0.6258; J_auc lowered by 0.0021 / 0.0114; DR_AUC's pair AUC*
  moved +0.0019 / -0.0054.
- **Optimistic values.** OPT_L in-sample 0.6213 / 0.6222 with gains 0.0243 / 0.0226; OPT_S 0.6400 / 0.6344, not useful.
- **Verifier and custody.** The six verifier refits (HGB15, CATI01 twice, LOG_LR, ET25, MLP6432) and the six custody readers
  (two files, three readers each) are as stated.
