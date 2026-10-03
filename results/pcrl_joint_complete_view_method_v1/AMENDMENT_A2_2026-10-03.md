# Amendment A2 (2026-10-03, during outer scoring; affected units had produced no output)

**What happened.**
- The outer scoring crashed on the first FARE arm (`outer__s0__F` and `outer__s2__F`) with `IndexError: index 2 is out of bounds`.
- The cause is the secondary race stress audit. It fits the cell-conditional attacker on finite (FARE) releases, and `CellConditional` defaulted to K = 2 classes; race has 5 labels.
- Both workers stopped there. Seed 1 had not started.

**Repair (technical, narrow).**
- `CellConditional.fit` now sets `K = max(K, max(y) + 1)`.
- A regression test is added (`test_cell_conditional_multiclass_labels`).

**What was not affected.**
- Neural-arm outer units never fit the cell-conditional attacker on more than two classes: their race audit is on continuous views, and SEX is binary.
- The 14 completed outer units (seeds 0 and 2: U, E, L, J, JP, S12, S21) are therefore unaffected and are kept.
- No F or F0 outer unit existed, and no seed-1 outer unit existed.

**Rerun.** Only the missing units: F and F0 on seeds 0 and 2, and all of seed 1. Then the controls.

**Locks.**
- `LOCK_A1.json` is preserved, and `LOCK.json` re-pins the code.
- `SELECTION_LOCK.json` (197f323) is unchanged. Selection, gates, families and the audit settings are unchanged.
