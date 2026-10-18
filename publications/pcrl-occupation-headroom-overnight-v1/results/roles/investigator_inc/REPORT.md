# Conditional income branch report (investigator A, second assignment)

**Saved by the coordinator.** The investigator (a separate agent) returned this report as the text of its final response, as the
campaign's briefs require; the coordinator saved it here unchanged in substance.

The branch fails on all three counts: it does not fix H3's seed-1 income accuracy, its new income view fails the frozen recipient
gate, and it gives no pair benefit. The POBP repair is a real income task-quality gain, but here that gain shows up as Brier / NLL room
and extra SEX readability, not as better decisions or protection. The branch is DEV-guided: the trigger fired through leg B (the source
anchor's published DEV_REPORT seed-1 income accuracy); no new DEV_REPORT read was made; all fitting used MODEL_FIT and CALIBRATION
income labels only, and no SEX beyond the stored corrections.

## What was built

- Step 1: the source's STRONG income learner (`ihs.fam_h.step1`) unchanged except that the compact grid settings code POBP as its
  ordinal place-of-birth code instead of a native 106-level category; same 7-setting grid, same CV folds (`IHS_H_step1_cv_S<s>`), same
  summed-Brier criterion; the original coding recomputed in the same unit as a check.
- INC_TW_PO (task-only twin, kind control): the repaired HGB stacked with T_TREE income on CALIBRATION, then Platt (the source's
  HC_STRONG with the repaired base).
- INC_H3_PO (branch configuration): the source's H3 step applied unchanged by calling `ihs.fam_h`'s own `budget_select`, `fit_maps` and
  `release_logodds`.
- Occupation output in both: the selected witness B/B2_SINGLE, recomputed with `ohn.fam_b.predict` (bitwise equal to its saved release
  on all six roles).
- 4 of 8 configurations (both configurations x both seeds); no revision (the card allowed one only for a defect).

## Checks that the source step was applied unchanged

- Step 1 reproduces the source exactly on both seeds: same selection (16 leaves / min leaf 20 / 94 trees seed 0; 8 / 80 / 185 seed 1),
  every grid tree count equal, out-of-fold Brier difference 0.0; the original-coding refit is bitwise equal to the source `HGB.pkl` on
  all six roles.
- The fit stage reproduces the source states exactly: the same code on the source HGB returns the source HC_STRONG and H3 states (beta
  equal; stack and Platt coefficients differ by 0.0) and the released H3 income bitwise on all six roles.
- All four INC releases verify bitwise (both tasks, all six roles); selftests V1 and V2 passed (18 and 19 checks).

## Task quality of the repaired income learner

Household cross-fitting of the HGB layer (OHN1 folds; the whole Step-1 procedure refitted per fold for both codings), repaired minus
original: seed 0 Brier -0.00288 (SE 0.00066), NLL -0.00422 (0.00082), accuracy loss -0.00225 (0.00163); seed 1 -0.00282 (0.00067),
-0.00408 (0.00082), -0.00125 (0.00165). The repaired main-CV optimum lies later (169 / 358 trees vs 94 / 185 for the original coding).
CALIBRATION Brier against raw T_TREE: raw repaired HGB -0.00003 / +0.00108 (original +0.00191 / +0.00279); cross-fitted repaired stack
-0.00020 / -0.00012 (original +0.00028 / +0.00033).

## MODEL_SELECT utility (income vs U0; each row pairs the income output with the B2_SINGLE occupation output)

| Release | Seed | beta | Accuracy loss (SE) | NLL | Brier |
|---|---|---|---|---|---|
| INC_H3_PO | 0 | 0.25 | +0.00000 (0.00271) | -0.00586 | -0.00354 |
| H3 + B2_SINGLE | 0 | 0.25 | +0.00012 | -0.00469 | -0.00279 |
| INC_H3_PO | 1 | 0.20 | +0.00175 (0.00290) | -0.00593 | -0.00389 |
| H3 + B2_SINGLE | 1 | 0.20 | +0.00175 | -0.00515 | -0.00330 |
| INC_TW_PO | 0 | - | -0.00375 | -0.00963 | -0.00653 |
| HC_STRONG + B2_SINGLE | 0 | - | -0.00287 | -0.00836 | -0.00572 |
| INC_TW_PO | 1 | - | +0.00075 | -0.00925 | -0.00648 |
| HC_STRONG + B2_SINGLE | 1 | - | +0.00050 | -0.00835 | -0.00583 |

- beta: the unchanged rule returned the source's own values (0.25 / 0.20); the expected-accuracy term was again the binding limit.
- All 12 point clauses and both guards pass; two-seed reserve INC_H3_PO RESERVE_MET at 2.840 (H3 2.862), INC_TW_PO RESERVE_MET at
  3.144 (HC_STRONG 3.296), both bound by seed-1 income accuracy.
- The branch changes 1.8-2.3% of income decisions relative to H3 on every role (label-free check), but the net seed-1 accuracy on
  MODEL_SELECT is identical to H3's.

## Development reader grid (INC_BANK; 8 cells over both seeds)

Spec `roles/investigator_inc/specs/INC_BANK.json`; summary `.private/.../runs/GRID/INC_BANK/SUMMARY.json` (merged with REF_BANK and
WITNESS_BANK). Recipient gate: income-view difference (candidate minus comparator) <= +0.010 in every cell.
- INC_H3_PO income view: fails against SUCC and against the matched component B/B2_SINGLE@H3 (same occupation, H3 income): mean
  +0.0177, max +0.0549, 4 of 8 cells above; against H_LJ mean +0.0069, max +0.0300, 4 cells; against RETIRED_B3CTL mean +0.0038, max
  +0.0207, 3 cells; passes at the edge against NEURAL_RAW (max +0.0061), B2TWIN (max -0.0028) and TWJ (max +0.0020).
- INC_TW_PO income view: fails against all seven comparators (means +0.0009 to +0.0349).
- Occupation view: fails against every comparator except the matched component (+0.0729 against the JOINT-occupation references,
  +0.0212 against NEURAL_RAW); this comes from the task-only witness occupation itself (B/B2_SINGLE@H3 shows the same values).
- Pair benefit (comparator minus candidate): NOT_POSITIVE against all six required comparators (INC_H3_PO means -0.011 to -0.048);
  against the matched component -0.0010 (range -0.0136 to +0.0141); no cell reaches the 0.020 target.
As pre-declared, the occupation witness stays a component beside H3; nothing is relaxed.

## Deviation, refusals, resources

- Both Step-1 units (OHN_INC_STEP1_S0_V1, OHN_INC_STEP1_S1_V1) exited with return code 1 on a TypeError in the final console summary
  print, after the write-once STEP1.json, HGB_PO.pkl and CROSSFIT_INCOME.npz were written (hashes match the record); not rerun (it would
  be an equivalent refit); the print line was fixed at 06:41Z (summary only) before any INC fit; logged in the INC card
  (PROTOCOL_DEVIATIONS item 8).
- Refusals: two heredoc shell commands were refused by the worktree guard ("...this command is too complex to verify that it stays
  inside the worktree. Refusing to run it..."); the Write tool and simpler commands were used instead.
- Resources: 19 queue units (2 failed, as above), 1,010.6 CPU-s (0.28 CPU-h), plus 60 CPU-s of out-of-queue work charged; the two
  Step-1 units account for 818 CPU-s (402 HGB fits per seed).

## Files

`ohn/fam_inc.py`; `roles/investigator_inc/CARD.md` (Step-1, results and grid sections appended after 721a64c15);
`roles/investigator_inc/specs/INC_BANK.json`; releases `runs/INC/{INC_TW_PO,INC_H3_PO}/S{0,1}/`; Step 1 `runs/INC/STEP1/S{0,1}/`;
grid `runs/GRID/INC_BANK/` (private).
