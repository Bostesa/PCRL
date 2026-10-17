# Compact failure map (from the published record; no new data read)

All AUCs are fresh-reader DEV_REPORT recovery of the recorded SEX column from the released outputs, seed 0 / seed 1, as published by earlier campaigns on these same Florida ACS 2019 development records. They are descriptive inputs to this campaign's design, not results of it.

| Release (source) | Linked pair | Income output | Occupation output | Utility status (DEV, point) | What it shows |
|---|---|---|---|---|---|
| T_TREE teacher (CatBoost; replication) | 0.664 / 0.664 | 0.569 / 0.568 | 0.650 / 0.649 | feasible; headroom over U0 (MODEL_SELECT): income NLL 0.0066, occupation NLL 0.0121 | a stronger task learner gives ~0.007 / ~0.012 nats of headroom; it reads as high as anything |
| TASK_REFIT (same leaf-correction loop, privacy off) | 0.664 / 0.666 | 0.572 / 0.567 | 0.650 / 0.645 | feasible | matched task-only control of the tree recipe |
| LOCAL_TREE = H_LL (per-recipient critics) | 0.657 / 0.649 | 0.558 / 0.549 | 0.582 / 0.584 | 24/24; occupation NLL excess vs U0 +0.0007 / −0.0003 | each output alone is much harder to read, almost for free, but **the linked pair still reads ~0.65**: coalition synergy |
| JOINT_TREE = H_JJ (adds the pair critic) | **0.604 / 0.607** | 0.535 / 0.537 | 0.560 / 0.583 | **infeasible**: income Brier excess vs U0 0.0054 / 0.0067; income accuracy loss 0.0086 / 0.0140 | the pair term removes most of the synergy; **income utility is the measured bottleneck** |
| H_LJ (LOCAL income + JOINT occupation; eligible reference) | 0.641 / 0.643 | 0.558 / 0.549 | 0.560 / 0.583 | 24/24; men's occupation Brier 0.0032 / 0.0040 | recombining outputs from different objectives restores synergy |
| NEURAL_RAW ρ1 E2 (eligible neural reference) | 0.640 / 0.638 | 0.546 / 0.551 | 0.622 / 0.623 | 24/24 | the competitive bar: pair ≤ 0.620 / 0.618 for the 0.020 target |
| C0 (512 shared cells) | 0.662 / 0.663 | 0.575 / 0.570 | 0.651 / 0.651 | feasible | shared cells make the pair redundant with the occupation output |
| SCM t = 0.008 (shared-cell merging) | 0.623 / 0.633 | 0.567 / 0.574 | 0.608 / 0.617 | 24/24 point; men's guard 0.0054 / 0.0064 > 0.005 | coarsening a fixed partition is expensive per unit of occupation-readability removed; burden falls on men; MODEL_SELECT constraints drifted on DEV |
| PUB_IBSI γ = 2 (adapted) | 0.630 / 0.623 | 0.558 / 0.558 | 0.620 / 0.613 | seed 1 fails 3 clauses on DEV | the published privacy objective is as good as SCM |
| ORD (task-only merging) | 0.657 / 0.658 | 0.596 / 0.597 | 0.647 / 0.662 | feasible | ordinary coarsening barely moves recovery |

## Measured bottlenecks (design requirements for this campaign)

1. **Coalition synergy.** Per-recipient protection lowers each output's own recovery but leaves the linked pair near the teacher. Only the pair-aware objective (JOINT_TREE) or shared cells (SCM, pair redundant with occupation) moved the pair.
2. **Income utility under pair protection.** JOINT_TREE misses the income Brier (both seeds) and accuracy (seed 1) caps. The fixed-coefficient recipe has no per-task cost control.
3. **Subgroup burden.** Coarsening and loss spending have fallen mostly on men (SCM men's occupation Brier excess 0.0054 / 0.0064).
4. **Role drift.** Constraints that held on fitting or selection people moved on DEV_REPORT (paired SE of held-out NLL differences 0.0016–0.0020; SCM spend 0.0082 → 0.0113 on seed 0). Edge selection is fragile.
5. **Strong comparators.** NEURAL_RAW and H_LJ are the eligible references; PUB_IBSI matches SCM; ordinary coarsening is not an explanation. H_LJ's occupation output reads 0.560 / 0.583, so the occupation recipient non-worsening margin (+0.010) against H_LJ is tight.
6. **Reader limits.** Finer partitions can be under-read by the fixed 8,001-person budget; continuous outputs can be harder to fit. Reader recovery is not a guarantee.

## Questions already answered (not repeated here)

SCM stopping thresholds and matched-loss comparison (closed, inconclusive); refreshed-critic schedules, strength-matched feedback and online strength frontiers (Adult, closed); decoder-only rescue and held-out calibration (closed); LEACE / linear erasure and noise channels (closed); class-preserving and task-only codes (closed); the local tree recipe's coalition benefit (not established).

## What remains open

Whether a pair-aware mechanism can keep JOINT_TREE-level coalition protection while meeting every utility clause and both subgroup guards on held-out people, and whether removing the coalition synergy structurally (one output a function of the other) or removing the SEX-mixing pathway of the predictor does better than adversarial correction at the same utility.
