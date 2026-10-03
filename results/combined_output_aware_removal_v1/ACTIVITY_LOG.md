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
