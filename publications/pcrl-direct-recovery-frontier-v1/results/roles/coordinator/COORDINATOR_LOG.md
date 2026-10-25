# Coordinator log (campaign `pcrl_direct_recovery_frontier_v1`, DRF1; append-only; times from `date -u`)

- 02:17:51Z campaign start (first `date -u`); 121 GiB free; YOTUO not attached (02:21:25Z).
- 02:30:39Z worktree `pcrl-direct-recovery-frontier-v1` created from the SAR1 tip 73931682e; session switched into it.
- 02:32:32-02:32:37Z working cache: APFS clones of the nine inherited private trees (verified in SOURCE_ADMISSION.json).
- 02:33:05Z `drf/` derived from `sar/` (mechanical rename); then `drf/solver.py` and `drf/dr.py` written (selftests on invented data).
- agents assigned after the design brief: statistical reviewer (rules; started 02:38:01Z by its own record), investigator A (exact finite-law AUC and
  tiny exact instances; started 02:38:33Z by its own record).
- 02:51:28Z source admission ok; 02:51:46Z reader splits (label-free); 02:53:57Z queue opened; deviation 1 recorded (a reader-grid
  selftest ran without the one-thread environment, charged 1,085 CPU-s); 02:59:25Z reader-grid code lock (torch interpreter).
- investigator A's tiny exact instances (V1 ended 03:00:16Z by its record: no implementation defect; local optima) -> atom-merge polish and 2 seeds x 2
  restarts added to the solver before registration (card section 7); V2 re-run 03:05:20-03:11:56Z: 4 / 3 of 348 shortfalls (AUC / MI).
- 03:08:00Z rules frozen by the reviewer (24 / 24 tests); REQUIRED findings R1-R12 adopted (PROTOCOL 15); 03:12:27Z budgets frozen.
- 03:14:24Z protocol completed; registration commit 62412796f; receipt dae5914f4 (root commit of
  `publication/pcrl-direct-recovery-frontier-v1`) pushed 03:15:40Z and verified on the GitHub remote 03:15:46Z; private owner scan
  of the receipt tree and message: no owner identifier (two mount-point paths in code only).
- 03:16:17Z / 03:16:19Z fit units DRF_FIT_V0_V1 / DRF_FIT_V1_V1 started; first label read 03:16:17Z (MODEL_FIT), after the verified receipt; independent
  verifier assigned (separate code under `roles/verifier/code/`).
- 03:21:00Z v1 fits done (281.9 / 277.9 CPU-s). MODEL_FIT nested estimator: the realized-label in-sample constraints do not transfer
  (nested occupation NLL spend +0.028 against a 0.0097 budget, men's Brier +0.012 against 0.0035) -> revision R1 decided on MODEL_FIT
  evidence, before any MODEL_SELECT read.
- card DR_DRF_R1 committed in 6a3dc2f34 (03:27Z, before any MODEL_SELECT read); R1 fit units 03:27:50Z-03:33:20Z (328.6 / 322.6 CPU-s):
  nested spends now about 1.1 x the budgets.
- 03:34:13Z both reading plans (DRF_V1, DRF_R1) committed (13d9dbdb9); 03:34:32Z first MODEL_SELECT read of the campaign (utility).
- MODEL_SELECT usefulness: v1 DR_AUC and DR_MI fail G1 in both versions (occupation NLL, men's guard; as the nested estimator
  predicted): the v1 reader stage is stopped (STOPPED_USEFULNESS_FAILURE), releases preserved; R1 DR_AUC_R1 and DR_MI_R1 pass all 28
  conditions at the point; the R1 CTRL points fail (men's guard, occupation NLL). R1 reader grids started 03:34:32Z / 03:34:42Z.
- 03:44:57Z / 03:45:12Z R1 reader grids done (622.9 / 628.2 CPU-s); 03:45:13Z-03:54:02Z optimistic units OPT_V0 / OPT_V1 (MODEL_FIT
  only; 485.7 / 511.6 CPU-s): OPT_L useful in-sample, none survives the nested estimator; OPT_S not useful in-sample.
- 03:55:02Z claims units: DRF_V1 stopped reading (CAND_INFEASIBLE, STAGE_2) and DRF_R1 (registered; TASK_FEASIBLE, D2
  NOT_SEPARATED, D3 NOT_SUPERIOR, D4 NOT_MET, STAGE_3); decision records committed 99696243e (03:59:40Z) with the card of revision R2
  (DR_DRF_R2: search budget only, 12 seeds; ADAPTIVE, trigger the R1 reading together with the MODEL_FIT search records).
- 03:59:56Z-04:12:42Z R2 fit units (760.0 / 762.3 CPU-s); robustness bank DRF1_R1_ROB committed (2d0616339, 04:00:11Z) before any
  robustness output; 04:12:40Z-04:27:15Z R1 robustness grids (862.7 / 871.1 CPU-s).
- 04:18Z R2 plan written; committed ffb524dcc 04:19:12Z with the R2 fit records and the verifier's checks 1-5 for v1 and R1
  (hand-back 1 saved verbatim). 04:27:06Z first R2 MODEL_SELECT read (utility): DR_AUC_R2 and DR_MI_R2 pass all 28 conditions; the
  R2 CTRL points fail the men's guard. 04:27:10Z-04:29:07Z R2 grids (75.0 / 110.7 CPU-s; most reference cells were cached).
- 04:29:13Z D2R of DRF_R1: ROBUSTNESS_NOT_CONFIRMED (CTRL bracket and twin confirmed; DR_MI_R1, NEURAL_RAW and H_LJ not confirmed;
  SUCC changes sign). 04:29:21Z claims DRF_R2 (adaptive; the same labels as R1; D3 passed NEURAL_RAW and TWJ). 04:29:12Z-04:29:17Z
  reader custody (5 readers bitwise); 04:29:08Z-04:29:27Z deployment store and build of the R1 packages.
- 04:25:55Z YOTUO attached: disk6s1, ExFAT, volume UUID 3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D, 578.8 GB free (read with `diskutil
  info`; nothing mounted or changed).
- 04:30:19Z R2 robustness bank DRF1_R2_ROB written after the R2 decisions and before any robustness output (committed f9efd4d31,
  04:30:30Z); R2 robustness grids, R2 packages and the R1 deployment check queued.
- 04:30:38Z-04:38:21Z R2 robustness grids; 04:38:37Z D2R of DRF_R2: ROBUSTNESS_NOT_CONFIRMED (CTRL bracket and twin confirmed; DR_MI_R2,
  NEURAL_RAW and H_LJ not confirmed). 04:29:28Z-04:48:39Z deployment checks of the R1 and R2 packages (8 packages bitwise at every
  request size, all refusals); 04:48:48Z validity audit of all 8 packages.
- 04:36Z-04:41Z decision drafts written (RESEARCH_DECISION.md, START_HERE.md; committed e0c661652); 04:41:56Z-04:51:34Z the
  statistical reviewer's claims review (11 required corrections, 11 recommendations; saved verbatim as CLAIMS_REVIEW.md); the
  corrections were applied 05:04-05:06Z (committed af82db399).
- Verifier hand-backs 2 (04:32Z) and 3 (04:50Z) saved verbatim: every R1 / R2 label, both D2R records, the R2 refits and the 8
  packages reproduced with separate code; the only non-PASS records are the two kept from its own parser defect.
- 04:50:17Z-05:10:50Z custody rehearsal (DRF_CUSTODY_REHEARSAL_V1, tip d939c2dd8, 566.5 MB): restore from the backup alone passed in
  both interpreters (8 models and 8 packages bitwise, 6 custody readers bitwise, 96 bank-member wires with their recorded hashes).
- 05:11:38Z-05:32:00Z final backup (DRF_CUSTODY_FINAL_V1, tip 5a791d7f1, 568.2 MB): restore from the backup alone passed in both
  interpreters. YOTUO identity re-checked 05:32:16Z (disk6s1, ExFAT, UUID 3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D); 05:32-05:52:52Z
  off-device copy to PCRL_private_backups/pcrl_direct_recovery_frontier_v1/ with a WARM restore check from the drive copy (passed);
  identity re-checked 05:53:02Z; the 22 existing folders untouched.
- 05:33:23Z out-of-queue charges for the reviewer's claims review (10 CPU-s) and the coordinator's label-free work (240 CPU-s, upper
  estimate through 06:00Z).
- 05:53:36Z queue closed; closeout receipts: 3.596 h elapsed, 3.994 CPU-h charged, 58 units done, 0 failed, 28 of 32
  specifications, 38 label reads (none before the receipt, none of a closed role), lineage log unchanged, no process left.
