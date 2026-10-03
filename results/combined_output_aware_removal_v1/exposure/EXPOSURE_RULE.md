# Exposure-sensitivity rule

**Fixed 2026-10-03, before any sensitivity result was computed.**

**Classification.** This is a post hoc exposure sensitivity. It does not replace the original primary analysis or
serve as a confirmation.

**Rows dropped.**
- Every assessment row whose `canon_key` (the established record-matching key from the benchmark admission) occurs
  among the encoder-training split rows is removed.
- Its whole record group (`assess_unit`) goes with it.
- The groups are identified from keys and roles only: Adult 7 assessment rows, HMDA 14. No outcome is used to choose
  them.
- Equal records are repeated records, not proven identical people. HMDA has no applicant identifier.

**What stays fixed.** Every fitted encoder, LEACE map, noise draw, attacker, head, probe and reference. No refit of
any kind.

**What is recomputed.**
- Every statistic is recomputed from the saved per-row predictions on the retained assessment rows.
- The bootstrap is over retained groups, with the same B and seeds as the original (exploratory 2,000 / 20261003;
  primary 20,000 / 20261004).
- Every paired comparison uses the identical retained rows on both sides.

**Support.**
- Per-class and per-pair assessment support is recomputed on the retained rows against the frozen thresholds (100 per
  class).
- A class that falls below its threshold is reported as not estimable, with the changed denominator. Sparse classes
  are never pooled.

**Native check.** PCRL's historical in-sample N0 check is recomputed on the test split with all 17 / 42
training-overlapping test rows removed, from every role. The same fixed function and λ are used.

**What is reported.**
- All 24 original primary endpoints, with point, simultaneous bounds and decision.
- The exploratory headline rows.
- A label per headline: STABLE (same decision), CHANGED, or UNRESOLVED.
