# Verifier hand-back 1 (verbatim; received by the coordinator 2026-10-25, saved 04:18Z)

I've finished checks 1–5 for everything currently available: the v1 and R1 fits and both readings. I found no discrepancy with the campaign's records. The one failed record is a defect in my own integrity script; I kept it, then corrected the script and re-ran it.

# Verifier progress report (DRF1), 2026-10-25T04:10Z

All checks used separate verifier code in `results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/` (`vlib.py` plus check scripts; numpy and the standard library only for every metric, objective and rule). Labels were read only through `drf.common.load(..., 'verify', ...)` inside queue units. All 15 invented-data tests of the library pass.

| Check | Unit / mode | Items | Result |
|---|---|---|---|
| 1. Finite-law objective, v1 + R1, both versions | DRF_VER_C1_V1 (MODEL_FIT labels) | 32,188 | PASS |
| 2. 28 conditions, G1, paired costs, plan records | DRF_VER_C2_V1 (MODEL_SELECT labels) | 5,936 | PASS |
| 3. R1 aggregate screen and D-labels, before and against DECISIONS.json | label-free | 365 + 388 | PASS |
| 3. Stopped v1 reading | label-free | 21 | PASS |
| 3b. Optimistic-diagnostic numbers | label-free | 138 | PASS |
| 3c. Grid validity (RULES.md 9) | label-free | 37 | PASS |
| 3d. Confidence conjunction and two-SE reserve | label-free | 32 | PASS |
| 4. Bootstrap SEs and 6 reader refits | DRF_VER_C4_V1 (MODEL_SELECT SEX, locked interpreter) | 998 | PASS |
| 5. Integrity, first run | label-free | 189 | FAIL (2), my parser defect, kept |
| 5. Integrity, corrected run | label-free | 189 | PASS |

**Check 1 (finite-law objective).**
- I rebuilt the fine cells from the pinned out-of-fold statistics. My identity tables match the twin's stored tables within 1e-12, and no routing near-ties exist.
- Against `dr/FIT_*.json` and every RELEASE.json, within 1e-9 for every saved configuration, under both laws, on merged emitted wires and on atom ids:
  - AUC* for the pair, income and occupation views;
  - the plug-in MI terms;
  - the 8 spends (realized for v1; expected for R1, with the realized spends reported beside them);
  - feasibility and the binding constraint.
- My independent SEX-blind coarsening path reproduces all 71 steps, and the saved CTRL points sit on it at their recorded steps.
- The MODEL_SELECT and MODEL_FIT releases recomputed from features alone match RELEASE.npz bitwise.
- The CTRL brackets recomputed from the nested records match: steps 66/69 and 67/69 for v1, 59/62 and 60/63 for R1. Fineness reads FINE in all four, as the plans record.
- The v1 candidate's expected-cost spends are 2.3–3.4 times its occupation-NLL and men's-guard budgets, while its realized spends sit at the budgets. This independently supports the R1 card's MODEL_FIT trigger.

**Check 2 (task metrics).**
- All 14 conditions per version agree within 1e-10. The twin also matches FRC1's published ledger to 6 decimals.
- G1:
  - DR_AUC_R1 and DR_MI_R1 pass all 28.
  - v1 DR_AUC and DR_MI fail on occupation NLL and the men's guard (and occupation Brier in some versions).
  - All four CTRL points fail.
  - The twin, SUCC and all nine references pass.
- This matches your reading. Every recorded paired cost recomputes within 1e-10, including DR_AUC_R1 against both CTRL points, DR_MI_R1, the twin and SUCC.

**Check 3 (reader comparisons and labels).** All 360 recorded cell values reproduce exactly. I computed my labels before DECISIONS.json existed, and they all match it:
- D1 TASK_FEASIBLE.
- D2 NOT_SEPARATED:
  - Against the bracket DR_CTRL_HI_R1 the pair benefit is +0.034 (AGG_POSITIVE), but income Brier and income NLL are resolved higher, so the frontier reads SEPARATED_ON_PRIMARY_COST_ONLY.
  - Against DR_MI_R1 the benefit is +0.0076 and point-only (version-0 mean −0.0004), with OBJ_MI_SPENDS_LESS_UNRESOLVED.
- FC FAMILY_CHANGE_NOT_RESOLVED; TWIN BELOW_TWIN_RESOLVED.
- D3 NOT_SUPERIOR: only NEURAL_RAW passes. W, TR100 and TT100 fail the income gate; H_LJ, TWJ and B2TWIN fail the occupation gate.
- D4 NOT_MET.
- USEFUL_OPTIMISTIC_POINT, STAGE_3, class FEASIBLE_NO_RELIABLE_GAIN.

The stopped v1 reading also matches every label: CAND_INFEASIBLE, STAGE_2, TASK_INFEASIBLE.

For grid validity I found the same 82 seed equalities with the same explanations, across 3,072 seed checks.

**Check 4 (refits).**
- My SCORE AUCs, all 400 bootstrap draws (against BOOT.npz), and the 168 paired cell SEs match within 1e-12.
- All 6 refits reproduce SCORE and VALIDATION AUCs exactly, with scores bitwise equal to the cache: HGB15, CATI01 twice, LOG_LR, ET25, MLP6432.

**Check 5 (integrity).**
- Order: the receipt was verified at 03:15:46Z, before the first label read at 03:16:17Z. Each card was committed before its fit reads. Both plans were committed at 03:34:13Z, before the first MODEL_SELECT read at 03:34:32Z.
- 20 specifications, with no duplicates.
- No MODEL_SELECT read involved drf.dr, and no closed-role label was read.
- The first run failed two items because my revision parser predated R2; the corrected parser is stricter, not looser.

**Disclosures.**
- My code was uncommitted when my three units ran. During DRF_VER_C1_V1, `vlib.py` changed; the changes only inserted functions check 1 does not call.
- I registered a clean re-run, DRF_VER_C1_V2, then withdrew it at 04:08:45Z while it was still waiting for a slot. It never started and read no label; its directory holds only CODE_AT_START.json.
- My uncommitted files also appear in your units' queue records; those units don't import verifier code.
- I could not recompute:
  - the OPT_L/OPT_S values, because their mappings aren't saved;
  - the nested held-out values;
  - the reader-budget flags, which I took as recorded.
- OPT_S ended at a worse objective than OPT_L even though its search space contains OPT_L's: an optimizer shortfall that doesn't change the label.
- Out-of-queue CPU is charged as VERIFIER_OUT_OF_QUEUE_1 (60 s) and VERIFIER_OUT_OF_QUEUE_2 (15 s). The queue units used 29.5, 1.3 and 4.1 CPU-s.

**Requests and pending.**
- Please commit `roles/verifier/code/`, `CHECKS.json`, `VERIFICATION_REPORT.md` and `out/`. The file hashes are listed in the report.
- Then I'll run the clean re-run of check 1 as DRF_VER_C1_V3 over v1, R1 and R2. For R2 it also checks that R1's four per-seed objectives are reproduced exactly and that R2's best objective is no worse.
- Also pending: checks 2–4 for the adaptive R2 reading, D2R on DRF1_R1_ROB, the deployment packages, and a final check 5.

Files are in `results/pcrl_direct_recovery_frontier_v1/roles/verifier/`:
- `VERIFICATION_REPORT.md`
- `CHECKS.json`
- `out/`
- `code/`
