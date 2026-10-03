# Activity log

Times are UTC on 2026-10-03.

- **Branch and worktree.** Created `research/combined-output-aware-removal-v1` from 70f978ffc0afff55ecdf49cf1a74908cc3db5f49. The name was free. Other worktrees were not touched.
- **Stage 0, admission (no fits).** Ran `scripts/s0_admission.py`:
  - All index-referenced inputs and checkpoints match their recorded sha256.
  - All 9,717 reused benchmark outputs (units, defenses, shared, SIGMA_STAR) match the drive SHA256SUMS.
  - Overlap reproduced: 17 Adult and 42 HMDA test-split rows have a record equal to an encoder-training record, split by role 6/4/7 and 21/7/14. No record is shared between any two admitted roles.
- **Correction found during admission.** The benchmark RESEARCH_DECISION / HANDOFF called the 17/42 rows "assessment rows". They span all three test roles; only 7 Adult and 14 HMDA are assessment rows. PROTOCOL.md had it right. Added to the correction ledger.
- **FARE admission.** Started in parallel: official eth-sri/fare, isolated environment, wrapper, synthetic tests. The earlier durable-guarantees use pinned 89cb1b66; its installation knowledge is reused, its baseline labels are not.
- **Predictions.** Registered in PREDICTIONS.md before any new fit.
- **Custody check started.** The locked runner inference reruns from saved predictions to reproduce the 24 primary rows.
- **Stage 1 done.** Corrections C1 (50 rows) and C2 (9 rows) computed. The custody check reproduced 24/24 primary endpoints at 1e-12. Committed as 70dbdf2.
- **Stage 2 started.** Exposure sensitivity under the committed rule (7a4a086).
- **FARE admission done.**
  - Official eth-sri/fare 89cb1b66ed268c16659cbf7428c43e60da2df641 in an isolated environment. The reproduction gate is bit-exact.
  - 7 documented adaptations (see METHOD_ADMISSION.md). 12/12 wrapper tests pass.
  - Coordinator decisions before the lock:
    - renamed the zero-fairness uid (FZ) to avoid a collision;
    - identical-cell configurations count as aliases;
    - declared a secondary HMDA certificate on race {0,1,2};
    - certificate refusals are recorded as UNAVAILABLE.
- **Lock v1** pushed at 3130c8d before any new fit.
- **Adult run.** 2026-10-03 05:35–05:53Z, 1,066 CPU-s, complete. FARE nominees 4 / 2 / 4, all admissible.
- **HMDA run attempt 1 failed** after 28 s with FileNotFoundError, before any defense or attacker fit; only REF and the three O_* units completed.
  - Cause: the policy-set LEACE map id was built with sorted attribute names (C_ethnicity+race) instead of the benchmark's declared order (C_race+ethnicity).
  - Repair: `leace_map_id()` now uses the declared order, plus a regression test. 17/17 tests pass.
  - **Lock amendment L1:** the lock was rebuilt, and v1 is kept as EXECUTION_LOCK_v1.json. Only oar/study.py changed; no scientific definition changed.
  - Adult is unaffected: its map names are the same under both orders.
- **Certificates UNAVAILABLE on Adult.** The wrapper refuses when a certification row's feature vector equals a fit row's (4 rows), even though these are distinct records in disjoint roles. This is handled post-run as dated amendment A1: the original UNAVAILABLE results are kept, and the guard is checked by row identity. Certificates are outside the primary family.
