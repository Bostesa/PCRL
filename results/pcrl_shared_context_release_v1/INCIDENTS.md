# Incidents and deviations

All times are UTC on 2026-09-24. None of these changed a margin, endpoint, nominee or comparison rule after an outcome was seen.

| # | Time | What happened | Effect | Resolution |
|---|---|---|---|---|
| 1 | 17:33 | The first full worktree checkout failed with ENOSPC. The tracked tree is 7.8 GB against 7.1 GiB free; git removed the partial tree itself. | None | Sparse worktree limited to the study, predecessor and dependency packages (46 MB) |
| 2 | 17:48–18:08 | Timestamps in RUN_STATUS, DESIGN_SPEC and RESPONSE were written as estimates that ran ahead of the clock. | Documentation only | Corrected to git commit times at 18:08. The content is unchanged. |
| 3 | ~18:05 | Engineering smoke (30% of anchor 0; no inner_check or outer rows): the raw per-token cost oracle deviated from D17 for about 90% of people with no task gain. | Design | Pre-fit amendment M2 (paired oracle). Its fixed-margin limitation is recorded; the full-data policies still deviated for 52–93%. |
| 4 | ~18:25 | Baseline smoke showed that frozen-attacker prices, co-trained adversaries and task-keyed early stopping weakened RD_PRIV and ADV. | Design | Pre-fit amendments M3 and M5, which strengthen the baselines |
| 5 | ~18:40 | With the M3 closing refit, NM smoke rounds could all be infeasible. The predecessor code raised an error in that case. | Design | Pre-fit amendment M4: a D17-witness fallback, identical for every privacy-constrained family |
| 6 | ~18:41 | An unquoted shell heredoc executed one backtick span in the M3 amendment text, which dropped the `--select privacy` flag name. | Documentation only | Repaired immediately |
| 7 | 19:54 | The first post-lock `prepare+unlock` command ran while the lockcheck sparse clone was still checking out, and failed with `ModuleNotFoundError`. It wrote nothing: the gate was never passed and no outer data was read. | None | Waited for the checkout, verified detached HEAD e5d555a with a clean tree, and reran. The unlock was verified at 19:56:26. |
| 8 | 19:45 | The first host capacity-report attempt from a separate checkout refused to run because it looked for sanitized inputs relative to its own root. | None | Symlinked the same sanitized files; the report reads no label arrays |

## Technical failures of scientific units

Zero. 58 of 58 queue units completed on their first attempt, with no retries (`RUN_MANIFEST.json`, `technical_failures_and_retries: []`). Every probability matrix passed shape, finiteness, nonnegativity and row-sum validation in `laws.person_law`. No invalid zero-sum probability rows were seen; unlike on the Mac, the host had no memory pressure.

## Registered deviations and limitations, carried into the decision

- **Weak ADV and RD_PRIV comparators.** Under the shared M4 rule, RD_PRIV and all four ADV units selected D17. Their comparator groups merged with D17, and the primary family has 40 endpoints rather than 80.
- **DET_SEL4 is not a privacy-first deterministic counterpart.** It chooses on task within the NM4_U closing bank. Its comparison with NM4_P is therefore not a clean randomization ablation.
- **Inner-only predictions not scored.** Predictions 3 and 12 needed inner contrasts that the registered selection code does not compute. They are marked "not scored" rather than scored post hoc.
- **Shared audit households.** Audit attackers share audit_fit households with the optimization bank: separate fits, not independent data.
- **Continuous erasers not refitted.** LEACE and SPLINCE were not refitted as 17-token controls.

## Note on ADV outcomes (checked 20:20)

The privacy-selected ADV units that ended in `WITNESS_FALLBACK` did train. For example, a0_ADV_B1_P ran 45 epochs in 1,212 s. Their inner_selection task loss never came within the registered D17 + .001 cap: the balanced task was about .48 at every epoch, and the unit reported `NO_ELIGIBLE_EPOCH`.

By design, pruning keeps the per-epoch history in `SELECTED.json` but not the unselected network weights, so those archives are small (for example 44 KB). The PPAN-style adversarial categorical baseline therefore did not reach D17's task loss on this data. That makes it a weak matched comparator: an honest outcome of the registered training budget, not a pipeline fault.
