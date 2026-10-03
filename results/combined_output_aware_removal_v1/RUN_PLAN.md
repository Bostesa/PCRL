# Run plan: corrected benchmark and complete-release protection

**Base:** research/combined-matched-removal-benchmark-v1 @ 70f978ffc0afff55ecdf49cf1a74908cc3db5f49. Its original
lock is 3274fe11a45f4e29f458ef7c7eff1e97df133456.

**Ceilings:**
- 12 aggregate CPU-h for scientific fitting, inference and verification. Setup and compilation are recorded
  separately.
- 6 GiB peak memory.
- At most 2 workers, with 1 BLAS thread each.
- At most 8 GiB of extra laptop storage, keeping at least 10 GiB free.
- No cloud.

| Stage | Content | Depends on | Mandatory | CPU-h estimate |
|---|---|---|---|---|
| 0 | Input admission by hash; overlap inventory; registered predictions; run plan; status file | — | yes | < 0.01 |
| 1 | Corrections: seed-paired contrasts for the truncated E3 pair, stable ρ₁², separation of MC from deterministic errors, wording ledger, custody check of the 24 original primary rows | 0 | yes | ≈ 0.3 |
| 2 | Exposure sensitivity: remove the pre-identified training-overlapping assessment groups; recompute from saved predictions | 0 | yes | ≈ 0.5 |
| 3 | New matched protocol; push EXECUTION_LOCK.json before any new fit | 0, FARE admission | yes | 0 |
| 4 | Output-surface comparison: full logits, probability, hard prediction, label reference, constant; outputs alone and with the noisy representation at the frozen σ\* | 3 | yes | ≈ 1 |
| 5 | FARE: at most 6 configurations per encoder plus a zero-fairness control, chosen on validation only. Controls: untreated, target LEACE, policy LEACE (descriptive), σ\* noise, constant. Three access views; heads on each protected representation. | 3 + admission | yes, if official FARE runs | ≈ 4–6 |
| 6 | Defense-aware attackers (cell-conditional for FARE); real shuffled-label and planted-leak controls per new interface | 5 | yes | inside stage 5 |
| 7 | Independent verification, backup and restore proof, reports, commit, push, private handoff | all | yes | ≈ 1.5 |
| opt | Kernelized adversarial concept erasure | spare budget only | no | — |

**Order and fallbacks:**
- Corrections and exposure sensitivity do not depend on FARE, so they run first.
- If official FARE is unusable within 2 hours of compatibility work, the exact blocker is published. Stages 1, 2 and 4
  are still completed.
- On memory pressure, concurrency is reduced before any scientific unit is removed.
- Timing calibration uses synthetic data or fitting-role rows only.
