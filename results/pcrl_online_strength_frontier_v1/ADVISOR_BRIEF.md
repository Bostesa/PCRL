# Advisor brief: online strength frontier (2026-10-05)

**Bottom line: no advantage (EXPERIMENTAL_NO_ADVANTAGE, complete).** We matched the incumbent's actual update strength and tried fixed recipient allocations. Neither a normalized joint method nor ordinary raw joint training met its development comparison. The best runnable model is still the incumbent, ordinary online raw joint training at β 0.3.

## What we did

- **Bank.** One fixed bank of 21 configurations × 3 seeds, every run 40 epochs from the same warm start on a common minibatch order:
  - raw penalty β 0.1, 0.3, 0.6;
  - normalized ratio ρ 1.5, 3, 5;
  - fixed income/occupation allocations a = 0.5 and 2 at ρ 3 and 5;
  - each in a joint and a local version.
- **Instrumentation.** Per-step strength receipts were logged for both encoders. The admitted incumbent runs were replayed bit for bit to obtain their receipts.
- **Selection.** Selection used inner roles only. The single assessment was locked and pushed first, then scored on 13,936 previously used rows (four pools combined for precision; not fresh data).

## What we found

1. **The incumbent's strength is about 3 on average, but it is not constant.** Its per-step penalty/task ratio averages 3.0 / 2.9 (income / occupation encoder), with a 10–90% range of 1.7–4.5. Symmetric ρ 3 holds exactly 3.
2. **Matching that strength reproduces the incumbent's trade-off rather than beating it.**
   - NORM-J ρ 3 leaks a little less than the incumbent: pair AUC 0.786 vs 0.802.
   - It pays 0.5 occupation-accuracy points for it, and failed a task gate on inner selection.
   - It sits on the line connecting raw β 0.3 and β 0.6 (post hoc interpolation).
3. **Every route to lower recovery costs occupation accuracy.**
   - Higher ρ: ρ 3 → 5 lowers pair AUC by 0.013 and costs 1.2 occupation points.
   - Higher β: β 0.3 → 0.6 lowers pair AUC by 0.038 and costs 1.1 occupation points.
   - The descriptive nominees lose 1.4 (normalized) and 2.0 (raw) occupation points against U. Both are beyond the registered one-point guard.
4. **Fixed allocation only moves leakage.** The occupation-heavy allocation lowered the occupation recipient's leak and raised the income recipient's, or traded protection for occupation accuracy.
5. **Joint beats local at the same budget, but the margin does not survive the task gates.** Joint vs local at ρ 3: pair −0.055.
6. **Even the incumbent still leaks.** Pair AUC 0.80 is substantial recovery of SEX, and the incumbent costs 0.9 occupation points against U.

## Process

- Four pushed locks, and two engineering amendments that repaired a diagnostic check (not training).
- An early pre-lock reference run was quarantined and reproduced after the lock.
- 109 tests; an independent verifier with its own code (see `INDEPENDENT_VERIFICATION.json`).
- About 2 CPU-hours locally, $0 cloud.
- The external drive was not mounted, so only a same-device backup with restore tests exists. Off-device backup is pending.

## Recommendation

1. **Close the strength/allocation family.** The limit is the occupation-accuracy cost of removing SEX recovery, not update strength.
2. **Package RAW-J β 0.3 as the best development model.**
3. **Do not spend a confirmation population.** A new direction would need a mechanism that removes SEX information without spending occupation accuracy, registered before any new fit.
