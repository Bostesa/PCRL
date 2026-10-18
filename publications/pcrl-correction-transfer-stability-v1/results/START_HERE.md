# Start here: correction transfer, genuine refits and a stable method comparison (PCRL, 2026-10-18)

This folder records one exploratory development campaign on previously used public-use ACS 2019 Florida PUMS development roles. It
asked three separate questions about the coalition-aware occupation correction of the occupation-headroom campaign (whose result,
NO_QUALIFYING_CONFIGURATION, stands unchanged):
- A. Can incremental occupation and subgroup costs be predicted on excluded households accurately enough to choose a useful
  correction strength?
- B. Does an improved strength rule keep actual utility while delivering a repeatable recovery benefit beyond its strongest matched
  controls?
- C. How much do the old recipient screen and unequal reserve requirements affect interpretation?

**Outcome in one line.** A replicated, independently replayed diagnosis and a negative method result (development evidence):
- the out-of-fold cost estimate over-predicts NLL on every partition, most likely through non-nested cross-fitting (a one-partition
  check), and its men's-cost error changes sign between training partitions;
- an excluded-household estimate is unbiased but as noisy as the slack it protects;
- honest refitting leaves almost no slack (NO_RECIPE);
- the coalition-aware correction lowers the declared development readers' recovery repeatably beyond the witness and the zero-critic
  continuation, but only at strengths far outside the utility budget;
- no release that lowers that recovery qualifies under one equal reserve.

## Read in this order

1. `ADVISOR_BRIEF.md`: one page.
2. `RESEARCH_DECISION.md`: the decision and plain answers to the eight campaign questions.
3. `PROTOCOL.md`, `PARTITIONS.json`, `EXPOSURE_MAP.md`, `ACCESS_POLICY.json`, `roles/statistical_reviewer/RULES.md`: the registration
   (receipt verified on GitHub at 19:13:08Z, before any fit or label read).
4. `MECHANISM_CARDS.md`, `REPAIR_FREEZE.json`: what was fitted and what was frozen before the repair panels were opened.
5. `PROTOCOL_DEVIATIONS.md`: every deviation, amendment and disclosure.

## Records

- Admission and preflight: `SOURCE_ADMISSION.json`, `preflight/` (source outputs, source replay, routing reproduction).
- Cost transfer (question A): `diagnosis/` (COST_TRANSFER.csv and _SUMMARY.json, TRANSFER_FACTORS.json, QUESTION_A_LABELS.csv /
  .json with the superseded first labels beside the primary ones, flag SCHEME_DEPENDENT, primary per A2 (D) and A5 (R), RULE_OUTCOMES.csv, RECIPE_GATES.json, RECIPE_SELECTION.json =
  NO_RECIPE, TRAINING_SIZE_BET.json and TRAINING_SIZE_CURVE.json), `figures/COST_TRANSFER_*` (SVG / PNG with a CSV table view).
- Repair comparison and frontier (question B): `repair/` (RULE_OUTCOMES_REPAIR.csv, RULE_COMPARISON.json, MATCHED_CONTROLS.csv /
  .json, FRONTIER_DRIVERS.csv, RESERVE_QUALIFIED_FRONTIER.csv / .json, ORIGINAL_BENCHMARK.csv / .json, CLAIM_LADDER.json),
  `tables/READER_STABILITY_*` (D_FRONTIER, R_FRONTIER, R_COMP0, R_TRACKR), `figures/FRONTIER_D.*`, `figures/FRONTIER_R.*`.
- Screens (question C): `screen/` (pre-outcome H3 income record, SCREEN_STUDY.md / .json, SCREEN_TABLE.csv, SCREEN_OC.csv),
  `roles/statistical_reviewer/SCREEN_REVIEW.md`.
- Track R (whole-service refit): `repair/MATCHED_CONTROLS.*` (Track R rows), `tables/READER_STABILITY_R_TRACKR.*`, findings F11 in
  `roles/coordinator/FINDINGS_NOTES.md`.
- Grid specifications: `grid_specs/` (D_FRONTIER, R_FRONTIER, R_COMP0, R_TRACKR), `DEVGRID_CODE_LOCK.json`.
- Verification and engineering:
  - `roles/verifier/`: separate code; append-only CHECKS.json and CHECK_CHANGES.json; REPORT.md with its addendum on A5; and
    records/V_TRIGGER__V2.json and V_QA_INFLUENCE__R1__V3.json.
  - `roles/engineering_review/`; `roles/claims_review/REPORT.md`.
  - Adjudications A1-A7 in `roles/coordinator/ADJUDICATIONS.md`; the statistical reviewer's adjudications in
    `roles/statistical_reviewer/`; `ERRATA.md` (E1, E2).
- Deployment: `QUICKSTART.md`, `deploy/DEPLOYMENT_CHECKS.json` (10 packages: R0 / R1 x anchor, refitted witness, coalition-aware
  correction at s = 0.5 and 1.0, the decisive zero-critic control at s = 1.0) and `deploy/DEPLOYMENT_CHECKS_P70.json` (the strongest
  honest task witness, refitted on the whole non-panel pool).
- Closeout, all written at closeout: `RESOURCE_ACCOUNTING.json`, `ACCESS_RECEIPT.json`, `EXPERIMENT_LEDGER.csv`,
  `BACKUP_VERIFICATION.json`, `HANDOFF.json`, `MANUSCRIPT_CLAIMS_HANDOFF.md`. `PUBLICATION_RECORD.json` is written after the final
  push and is therefore not part of the commit it describes.
- Coordinator notes: `roles/coordinator/` (COORDINATOR_LOG.md, FINDINGS_NOTES.md F1-F12, DESIGN_BRIEF.md, REPAIR_PLAN.json,
  ADJUDICATIONS.md, briefs). F8 holds the superseded question-A labels (F9 corrects them); F11's "75% of MODEL_FIT" is corrected
  by ERRATA E2.

## What is not here

Person-level records, fitted private models and reader scores stay in the private tree (`.private/`, never published; restorable from
the verified backup). The research branch with its full history stays local. The public copy is the sanitized publication branch
`publication/pcrl-correction-transfer-stability-v1`.
