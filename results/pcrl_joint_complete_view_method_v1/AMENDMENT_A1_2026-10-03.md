# Amendment A1 (2026-10-03, before any inner or outer result existed)

**What happened.**
- The first inner-audit unit (`nn__s0__U`) failed with "Input X contains infinity".
- `finalize.outputs` built the released centred logits as `log(predict_proba) − mean`. When a deployed head's probability underflows to exactly 0, this gives −∞.
- The U units on all three seeds were affected: income, 390 rows each. Every other unit was finite.

**Repair (technical, narrow).**
- Centred logits are now `z − mean(z)`, where `z` is the head's `decision_function`. For a binary head with margin `d`, the logits are `(0, d)`.
- These are the same quantity in exact arithmetic, but they stay finite.
- Probabilities (`predict_proba`) and hard decisions are unchanged; the runner asserts the hard decisions are identical.

**Scope.**
- Outputs are recomputed for every finalised unit from its saved head and saved features (`jcv.run --stage amend_a1`). There is no refit, and no change to encoders, LEACE maps, heads, labels, roles, gates, families or selection.
- The previous unit versions are kept as `<unit>.quarantined`.
- A regression test is added (`test_centred_logits_finite_under_probability_underflow`).
- The original lock is preserved as `LOCK_v1.json` (commit b67664f). `LOCK.json` re-pins the code with this amendment.

No inner, selection or outer result had been produced when this amendment was made.
