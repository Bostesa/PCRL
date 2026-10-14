# Advisor brief: separate-pool utility guard

**Question.** The previous safeguard checked only the current update minibatch. By epoch 5 it had lost about 0.04 FIT2 NLL, and selection returned the unchanged start. Can a safeguard that checks a whole separate fitting pool, under start-anchored absolute budgets, keep occupation useful on a slice it never saw in this continuation?

**Answer for this fixed recipe: no.** The outcome is COMPLETE_NEGATIVE_FEASIBILITY, with the privacy comparison NOT_TESTED.

| Seed, epoch 5 | Accepted updates | GUARD NLL / Brier change (its own pool) | VALIDATION NLL / Brier change | VALIDATION accuracy loss |
| --- | --- | --- | --- | --- |
| 0 | 80/160 | +0.0079 / +0.0050 | +0.0156 / +0.0077 | 0.0065 |
| 1 | 102/160 | +0.0084 / +0.0050 | +0.0159 / +0.0075 | 0.0088 |

The VALIDATION caps are NLL ≤ +0.010, Brier ≤ +0.005 and accuracy loss ≤ 0.010. Epochs 1 and 3 also fail NLL and Brier in both seeds.

- **Real movement.** The privacy term drove the accepted steps. In each seed, all 8 scanned updates differed from the no-privacy counterfactual by about 0.009–0.012 in L2, against tolerances around 1e-8. The occupation wire and decisions changed.
- **The guard behaved as specified.** It never exceeded its absolute budgets on GUARD. Brier was the binding constraint. An independent replay confirmed every budget state and every VALIDATION score.
- **No generalization margin.** Held-out loss grew about 1.5–2 times faster than guard-pool loss. GUARD also feeds the shared task objective, so it is fitting data.
- **Controls.**
  - GLOBAL_UPDATE_GUARD, which checks the 8,180-row UPDATE pool, passed every cap in seed 0 and narrowly failed NLL in seed 1 (+0.0106 to +0.0109). This is descriptive and not a registered effect.
  - ORDINARY and OLD_BATCH_GUARD lost +0.031 to +0.043 NLL by epoch 5.
  - TASK_ONLY stayed within the caps, with no privacy gradient applied.
- **Not affected.** The frozen-income interval uncertainty from the corrected trial is unchanged, and no attacker metric exists for these models.

**Compute.** About 57 CPU-seconds of queued numerical work in total, about 17 s per training seed, all under the 20 CPU-hour ceiling, with $0 cloud. Single-threaded queue units peaked at about 0.6 GB RSS.

**Next.** Do not tune budgets, pools or epochs on this VALIDATION slice. A future design would need a check that certifies held-out loss. Examples are a larger excluded guard pool that is not part of the task objective, or a margin set from a held-out calibration fitted before training. It must be frozen separately, and confirmation still needs an unexposed population.
