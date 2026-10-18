# Manuscript claims handoff (correction-transfer campaign, PCRL, 2026-10-18)

**Use.**
- These are the only statements this campaign supports, with their required qualifiers and records. The independent claims review
  (`roles/claims_review/REPORT.md`) was applied to this text.
- Everything is development evidence on previously used public-use ACS 2019 Florida records (MODEL_FIT, CALIBRATION and the
  adaptively used MODEL_SELECT).
- None of it is confirmation. The final ATTACK and DEV_REPORT labels stayed closed.
- "Recovery" means SCORE AUC of the declared development readers estimating the recorded SEX column from released probability
  vectors. It is relative to those readers, never protection against every reader.

## Statements that may be made (with their qualifiers)

1. **The out-of-fold cost estimate used to choose correction strength over-predicts the NLL increase on MODEL_SELECT on all four
   refitted training partitions** (REFIT-routed PAIR: -1.9 to -4.5e-3, mean over strengths).
   - Non-nested cross-fitting is the likely source.
   - Records: `diagnosis/TRANSFER_FACTORS.json` (MS-OOF, CHECK-NESTED_OOF, MS-NESTED_OOF).
   - Qualifiers:
     - Nesting was run on one partition (D0) only.
     - There it removed PAIR's over-prediction against the CHECK panel (+0.97e-3, z 0.5) but under-predicts against MODEL_SELECT
       (+2.9e-3, z 2.0), and for LOC on both panels (+3.0 / +2.4e-3).
     - Nesting was not run for CONT.
2. **The out-of-fold estimate's men's-cost error is partition-specific.**
   - For the REFIT-routed coalition-aware direction it ranges from +2.1e-3 to -5.1e-3 across the four partitions (CHECK minus OOF,
     mean over strengths).
   - On three of four partitions it exceeds the refitted witnesses' men's reserve slack (+0.5 to -0.8e-3). On D0 it is +0.03
     against a slack of +0.49.
   - Record: `diagnosis/TRANSFER_FACTORS.json`.
3. **An excluded-household (GUARD) estimate is unbiased within noise (|z| up to 2.9) but not more accurate.**
   - Under the registered paired household bootstrap (one resample per set; RULES.md section 6), GUARD predicts realized CHECK
     costs better than the out-of-fold estimate for neither REFIT-routed correction direction (PAIR, LOC) on any of the four
     partitions.
   - No direction does so on all four partitions under either scheme.
   - Record: `diagnosis/QUESTION_A_LABELS.csv`.
   - Qualifiers:
     - The FROZEN-routed PAIR is PREDICTION_IMPROVED on D1 (ensemble) with A1_NOT_SUPPORTED.
     - The primary labels are the registered scheme's (A2 for D, A5 for R): NOT_IMPROVED. The superseded frozen-implementation
       output reads PREDICTION_IMPROVED for three units, flagged SCHEME_DEPENDENT: D0 CONT ensemble, R1 LOC full and R1 Track R PAIR
       full.
     - Report both, primary first, with the sentence: "an improvement appeared only under the superseded implementation; it is not
       established under the registered scheme". Cite deviations 10, 11, 16 and 18.
4. **Honest refitting costs most of the source witness's margin.**
   - Refits of the same recipe on 42-53% of MODEL_FIT give men's-guard excess +0.65 to +2.15e-3, against -3.9 / -2.3e-3 for the
     source witness fitted on all of MODEL_FIT.
   - An adaptive refit on 70% restores a positive men's reserve slack and about half of the gap. This is consistent with
     training-set size being sufficient to explain the loss; a selection component at 100% is not excluded.
   - Records: `roles/coordinator/FINDINGS_NOTES.md` F2 and F7, `diagnosis/TRAINING_SIZE_CURVE.json`, `TRAINING_SIZE_BET.json`.
   - Qualifier: the 70% refit is an adaptive, disclosed diagnostic (amendment 9).
5. **No development recipe passed the registered gates (NO_RECIPE).**
   - The GUARD-based strength rules never improved on the source rule on the reserved repair panels.
   - Records: `diagnosis/RECIPE_SELECTION.json`, `repair/RULE_COMPARISON.json`.
   - Qualifier: this was close to forced. On R1 every rule chose identity, and the registered comparison never counts identity as
     an improvement.
6. **The coalition-aware occupation correction lowers recovery for these development readers repeatably, but only at large
   strengths, and not stably beyond an occupation-only critic.**
   - Beyond a zero-critic continuation from s = 0.3, and beyond the unchanged witness at full strength: +0.041 / +0.042 SCORE AUC
     at full strength on the repair partitions, STABLE_POSITIVE over 8 cells; similar on the diagnostic partitions.
   - At full strength its realized men's cost is 70 times the R0 reserve slack (unbounded on R1).
   - Records: `repair/MATCHED_CONTROLS.csv`, `tables/READER_STABILITY_*`, `repair/FRONTIER_DRIVERS.csv`.
   - Qualifiers: a coalition-aware occupation correction with H3's income frozen, never a joint two-output design; reader-relative.
7. **Under one equal reserve rule for candidates, controls and references, no release that lowers the readers' recovery
   qualifies.**
   - The only qualified points are the zero-critic continuation. The record lists them in its `qualified_protected_points` field
     because it holds every qualified non-reference point, but they give no recovery reduction.
   - Every required reference fails the reserve that candidates had to meet in the source.
   - Record: `repair/RESERVE_QUALIFIED_FRONTIER.json`.
   - Qualifier: not superiority over any original reference.
8. **The old per-cell recipient gate has a low pass rate for true-zero differences on the income view.**
   - Under simulation law L1 at two household repetitions it passes 19% (rho_seed 0) or 31% (rho_seed 0.76). H3's frozen income
     view alone fails it.
   - Records: `screen/SCREEN_STUDY.md` / .json, `screen/PRE_OUTCOME_H3_INCOME_VIEW.json`.
   - Qualifiers: rates are properties of the specified law only. AGG1 does not cure this at two repetitions. The source label is
     unchanged.
9. **Refitting H3's income recipe without the excluded households removed its Brier / NLL headroom.**
   - Income Brier excess vs U0 goes from -2.8 / -3.3e-3 (frozen H3) to +1.6 / +1.1e-3 (refit).
   - Records: the Track R evaluations (private run summary TRACKR_EVAL_MS) and findings F11 with ERRATA E2.
   - Qualifiers: refit on about 53% of MODEL_FIT plus CALIBRATION, one per partition, so training size is not separated from
     recipe. A recipe test on different training subsets, not a test of the frozen H3 package.

## Statements that must not be made

- That any rule, recipe or release here is a validated, competitive or confirmed method, or a protection component.
- That the 0.020 target was met. At full strength the mean benefit against NEURAL_RAW and H_LJ exceeds 0.020 only as a point mean
  (DEV_GRID_TARGET_POINT_ONLY), and that release fails utility at the point.
- That a joint two-output design was tested, or that the method is novel.
- That the source campaign's NO_QUALIFYING_CONFIGURATION is changed, or that an AGG1 pass would be evidence of protection.
- That nesting removes the bias, that training size causes the margin loss, or that the reserve-qualified continuation points are
  protective.
- Any impossibility claim.

## Required disclosures beside any of the above

- **Question-A bootstrap correction** (deviations 10, 11, 16 and 18; adjudications A2 and A5).
  - It was committed after the repair panels had been read with the frozen code, and was triggered by D-stage inputs only
    (verifier P4 PASS).
  - Disclosed window (P4b): the verifier's R1 replays wrote the two SCHEME_DEPENDENT labels to its CHECKS.json 29 s before the
    commit; no coordinator read is logged.
- **The verifier's genuine FAILs:** F3 V2 (frozen rule code changed after the freeze) and C2.1 (P70A record defect, ERRATA E1).
- **Ensemble deployment** (deviation 6): the registered gates were patched for the ensemble before any outcome. The ensemble's
  identity is the uncorrected ensemble e_0, which itself crosses the witness's reserve.
- **The adaptive training-size diagnostic** (deviation 9).
- **Rule selections not reader-evaluated** (deviation 12).
- **Compression twins:** R0 only, and not cost-matched (deviations 13 and 17).
- **Code and queue:** the launch audit (deviation 14) and the queue record defect (deviation 15).
- **Errata:** E1 and E2.
- **Time stamps:** the coordinator's errors (deviations 8, 11 and 19), and the heading-time errors of A2, A5 and the verifier's
  "20:29Z" message.
- **Units:** one failed verifier unit (refused before any label read; rerun) and one verifier unit registered but never launched.
- **The rest are in PROTOCOL_DEVIATIONS.md and ADJUDICATIONS.md:** deviations 1-5 and 7; adjudications A1, A3, A4 and A6.
