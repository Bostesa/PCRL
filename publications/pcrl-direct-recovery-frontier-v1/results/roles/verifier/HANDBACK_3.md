# Verifier hand-back 3 (verbatim; received by the coordinator 2026-10-25, saved before 04:51:09Z, the `date -u` of its commit)

Items 1–3 are done (04:50Z). The R2 reading, R2's robustness record and all eight deployment packages agree with the records, and no new check failed. The only non-PASS records in `CHECKS.json` (167 records) are still the two from the kept check-5 parser-defect run.

**1. R2 decisions.**
- I computed my labels with `claims/DRF_R2/DECISIONS.json` unopened, then compared: 400 of 400 items pass, unprefixed and with the `ADAPTIVE_R2__` prefix. The registered flag (false) and the adaptive tag match too.
- D1 TASK_FEASIBLE.
- D2 NOT_SEPARATED:
  - Against the bracket DR_CTRL_HI_R2 the pair benefit is +0.0322 (AGG_POSITIVE), but income Brier and income NLL are resolved higher.
  - Against DR_MI_R2 it is +0.0039, point-only, and DR_MI_R2 spends at least as much.
- FC NOT_RESOLVED; TWIN BELOW_TWIN_RESOLVED.
- D3 NOT_SUPERIOR: NEURAL_RAW (+0.0207) and TWJ (+0.0238) pass, both credited under the reader-budget rule; the rest fail.
- D4 NOT_MET, STAGE_3, class FEASIBLE_NO_RELIABLE_GAIN.
- Grid validity for DRF1_R2 matches the record (82 equalities, all explained), as do the confidence conjunction and reserve.
- Refits (DRF_VER_C4_V2, 998 of 998):
  - SCORE AUCs, bootstrap draws and the 168 paired cell SEs match within 1e-12.
  - All six refits are exact: OUT_MLP three times, LOG_LR, CATI01, OUT_HGB.

**2. R2 robustness.**
- I computed my D2R at 04:38:25Z, before `claims/DRF_R2/D2R.json` was written (04:38:37Z), then compared: 434 of 434 pass.
- Result: ROBUSTNESS_NOT_CONFIRMED.
  - Against DR_CTRL_HI_R2 it is confirmed (robustness benefit +0.0302).
  - Against DR_MI_R2 it is not (+0.0040, point-only).
- No sign change; the robustness income gate fails against SUCC.
- Refit is genuinely effective: all 232 equalities in 6,144 seed checks are explained.
- The record repeats SUCC in `recipient_failures` and in the headline caveat, the same cosmetic issue as R1's record; the sets match.

**3. Deployment (DRF_VER_C6_V1 and DRF_VER_C6_V2, 244 of 244 each).** For DR_AUC_R1, DR_MI_R1, DR_AUC_R2 and DR_MI_R2 in both versions:
- All 23 dependency hashes match, and the release hash agrees across the package, RELEASE.json and the evaluation record.
- The stored arrays equal RELEASE.npz bitwise on all six roles, and my own emission equals them.
- Used as a black box, every package matches my emission bitwise:
  - on the raw fields of all 20,000 MODEL_FIT rows;
  - on MODEL_SELECT as a whole batch;
  - in requests of 333 rows (all rows), 7 rows (first 700) and 1 row (first 100). The full request-size sweep is in your report.
- Decisions are the argmax, and only probability and decision are exported.
- The SEX-field, hidden-output and reordered-field requests are all refused.
- Your two reports' four summary flags are all true.

**Bookkeeping.**
- Changed since 6b4f27750 and awaiting a commit: `code/check6_deploy.py`, `VERIFICATION_REPORT.md` (new section 7), `CHECKS.json` and eight new files in `out/`.
- Queue CPU: DRF_VER_C4_V2 4.5 s, DRF_VER_C6_V1 25.8 s, DRF_VER_C6_V2 25.6 s. Out-of-queue I charged VERIFIER_OUT_OF_QUEUE_4 (30 CPU-s).

Next is the final check 5, once you tell me the closeout records exist. Everything is in `results/pcrl_direct_recovery_frontier_v1/roles/verifier/`.
