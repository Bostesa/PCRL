# Reader-robustness investigator: Phase 2 log (UTC, real clock; append-only)

- **15:54Z** Coordinator's Phase 2 message.
  - The extras proposal was accepted and integrated before the lock: devreaders 7222b0105, lock 1128afd67.
  - The instance map is now R1 0, R2 1, R3 6, R4 7, with offsets on R1/R2 only.
  - FB2 bank members were named.
- **15:56-15:57Z** Robustness bank frozen, outcome-blind, at commit 724a81f71: NRM1_ROBUST_REPS (R1-R4 x k0/k1) and NRM1_ROBUST_SEEDS
  (R1/R2 x k2/k3); eight members; plan ROBUST_PLAN_V1.
  - The first draft of the two specs carried a written time of 15:58Z, ahead of the clock (15:55:52Z). It was corrected to 15:56:06Z
    before the commit; no committed record carries the wrong time.
- **15:58-16:00Z** `nrm.readerbudget` follows the new instance map: a refused repetition/offset combination is recorded as None
  instead of raising. Selftest 19/19, unit RR_BUDGET_SELFTEST_V3, queued and run at 16:17Z.
- **16:08-16:21Z** `nrm.readerstab` and its refinements committed before any grid output was read: fcf55f224, 78aa98106
  (single-instance variant), 29699f02b (validity scoped to bank members) and 5c803c086 (explicit run folders). Selftest 25/25, unit
  RR_STAB_SELFTEST_V1 at 16:17Z, which tested the code of 29699f02b. 5c803c086 is re-tested by RR_STAB_SELFTEST_V2 before the
  summaries run.
- **16:17:12-16:17:17Z** NRM_GRID_MAIN_S0_V1 and _S1_V1 ended (done). Only then were the robustness units submitted:
  - RR_ROB_REPS_S0_V1 started at 16:17:49Z, slot1;
  - RR_ROB_REPS_S1_V1 waited, then started at 16:18:38Z in slot0, when VER_GRID_MAIN_V1 ended at 16:18:37Z.
- **16:17Z** RR_BUDGET_ROBUST_V1 (label-free budget of the bank): done, 4.2 CPU-s.
- **16:20Z** Coordinator's scheduling rule: at most one robustness unit running or waiting. It asked me to withdraw the waiting ticket
  of RR_ROB_REPS_S1_V1.
  - **Withdrawal.** By then the unit had already started (unit_start 16:18:38Z), so it was not a waiting ticket any more.
  - To free the slot for the critical path, at 16:20:10Z I sent SIGTERM to that unit's reader process only (PID 99818, the child of
    its `nrm.jobqueue run` process).
  - The queue recorded unit_end with return code -15, status failed and 91.2 CPU-s charged.
  - Effects:
    - RR_ROB_SEEDS_S1_V1 was never submitted (the chain used `&&`);
    - the empty default output folder runs/GRID/NRM1_ROBUST_REPS/S1 that the unit created is kept as part of the failed attempt;
    - completed reader-cache entries, which are written atomically, stay valid for the rerun.
  - RR_ROB_REPS_S0_V1 was not touched.
  - The freed slot was taken by VER_REFIT_MAIN_S0_V1 (16:20:14Z).
- **Rerun plan, one unit of mine in the queue at a time.** After the version-0 chain (RR_ROB_REPS_S0_V1, then RR_ROB_SEEDS_S0_V1)
  finishes, in order:
  1. RR_STAB_SELFTEST_V2;
  2. RR_ROB_REPS_S1_V2 with `--out runs/GRID/NRM1_ROBUST_REPS/S1_V2`, read by `nrm.readerstab --run-dir`;
  3. RR_ROB_SEEDS_S1_V1.
- **16:17:49-16:31:49Z** Version-0 chain done:
  - RR_ROB_REPS_S0_V1, 16:17:49-16:24:36Z: 403.8 CPU-s, 1,472 reader fits, 100 cache hits;
  - RR_ROB_SEEDS_S0_V1, 16:24:36-16:31:49Z: 431.3 CPU-s, 1,472 reader fits, 4 cache hits.
  - The cache hits beyond the 96 registered view-cells (4 in each run) suggest an exact alias of one view between two bank members.
    The wire hashes will check this, and the alias is counted once.
- **16:32Z** Version-1 chain submitted, one unit at a time: RR_STAB_SELFTEST_V2, then RR_ROB_REPS_S1_V2, then RR_ROB_SEEDS_S1_V1.
- **16:32-16:45Z** Version-1 chain done:
  - RR_STAB_SELFTEST_V2: 26/26;
  - RR_ROB_REPS_S1_V2, 16:32:15-16:37:45Z: 328.0 CPU-s, 119 cache hits, 19 of them completed entries of the stopped V1 unit;
  - RR_ROB_SEEDS_S1_V1, 16:37:45-16:44:54Z: 423.3 CPU-s.
- **16:45Z** RR_STAB_RUN_V1 wrote `readers/ROBUSTNESS.json`:
  - 56 comparisons;
  - 0 cell conflicts;
  - PAIRED_MATCH;
  - grid validity of the bank READER_SEEDS_EFFECTIVE.
  - The report was returned as text to the coordinator (the harness refuses report files from sub-agents).
- **CPU:** 1,695.3 CPU-s in the queue over 15 RR units (2 failed: the RR_BUDGET_SELFTEST_V1 expectation and the withdrawn
  RR_ROB_REPS_S1_V1), plus 5 CPU-s charged.
