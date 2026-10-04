# Research decision: refreshed critics with per-recipient local feedback (J-G)

**Status: EXPLORATORY DEVELOPMENT evidence** on already-exposed Adult rows. DEVELOPMENT_ASSESSMENT (3,397 rows, 3,393 groups) is a new development partition, not fresh confirmation.

Locks were pushed before each stage they govern:

| Lock | Commit (time) | Pushed before |
|---|---|---|
| CODE_LOCK.json | `d7367e0` (05:22Z) | any nonzero fit |
| Stage B freeze | `2be8cea` (05:31Z) | Stage C |
| EVALUATION_LOCK.json | `3affa49` (05:49:43Z) | the single assessment (05:49:56–05:59:33Z) |

Dated code amendments A1–A3 lock the later-written diagnostics, inference, deployment, reporting and closeout code. No fitted component was changed after its lock.

**Model label: EXPERIMENTAL_NO_ADVANTAGE.**

## Verdict

**Neither registered claim is established.**

1. **J-G had no feasible nominee on any seed.** Inner selection required J-G's local AUC on each recipient to be no higher than min(L-R, L-G, C\*).
   - C\* (the lowest-coalition feasible control) was **J-O** on every seed: the joint penalty trained with the inherited online critics and no refits.
   - J-O had the lowest inner local recovery of every trained arm, so J-G's guard was J-O's local levels. No J-G checkpoint met them.
   - J-G was therefore scored descriptively at its declared closest checkpoint: β = 0.3, epochs 10, 20 and 10 on seeds 0, 1, 2.
2. **L-G also had no feasible nominee on seed 0.** Its occupation-view inner AUC (0.822) exceeded L-R's (0.821).
3. **On the assessment, the clauses fail as well.**
   - Claim A (J-G vs L-G) passes 7 of 9 clauses. Claim B (J-G vs C\* = J-O) passes 6 of 9.

A missed conjunction is not a success, whatever the individual clauses show.

## Primary family (18 slots, z = 2.991316)

| Clause | J-G vs L-G (claim A) | J-G vs C\* = J-O (claim B) |
|---|---|---|
| Coalition improvement > 0.02 | +0.012 [0.006, 0.018], **NOT_ESTABLISHED** | **−0.039** [−0.049, −0.029], NOT_ESTABLISHED (J-G leaks more) |
| Income-view increase < 0.01 | −0.007 [−0.016, 0.001], PASS | +0.029 [0.017, 0.040], NOT_ESTABLISHED |
| Occupation-view increase < 0.01 | +0.004 [−0.003, **0.011**], **NOT_ESTABLISHED** | +0.050 [0.037, 0.062], NOT_ESTABLISHED |
| Accuracy vs U > −0.01 (income / occupation) | −0.003 / +0.003, PASS / PASS | aliases |
| 80% retention (income / occupation) | PASS / PASS | aliases |
| Gain over constant > 0.03 (income / occupation) | +0.108 / +0.189, PASS / PASS | aliases |
| **Claim** | **NOT_ESTABLISHED**: 7/9 clauses; J-G not a nominee on any seed; L-G not a nominee on seed 0 | **NOT_ESTABLISHED**: 6/9 clauses; J-G not a nominee |

## Levels (assessment; seed means)

| Arm | Income accuracy | Occupation accuracy | v1 AUC | v2 AUC | Coalition AUC | Coalition OLS R² (fitting rows) |
|---|---|---|---|---|---|---|
| U (task only, matched budget) | 0.844 | 0.478 | 0.857 | 0.873 | 0.882 | 0.37 |
| L-R (frozen local reference) | 0.842 | 0.477 | 0.829 | 0.847 | 0.862 | 0.26 |
| L-O (local, online critics) | 0.848 | 0.477 | 0.742 | 0.822 | 0.855 | 0.28 |
| L-G (local + feedback) | 0.843 | 0.479 | 0.813 | 0.842 | 0.867 | 0.26 |
| J-R (joint, refreshed) | 0.842 | 0.479 | 0.802 | 0.845 | 0.859 | 0.20 |
| **J-O (joint, online; C\*)** | 0.842 | **0.468** | **0.777** | **0.796** | **0.816** | 0.17 |
| **J-G (candidate, descriptive)** | 0.841 | 0.481 | 0.805 | 0.846 | 0.855 | 0.19 |
| E (official LEACE on U) | **0.787** | 0.444 | 0.842 | 0.852 | 0.863 | 0.00 |
| F (official FARE) | 0.842 | **0.455** | 0.682 | 0.634 | **0.701** | 0.14 |
| F0 (zero-fairness tree) | 0.852 | 0.468 | 0.777 | 0.857 | 0.867 | 0.41 |

The constant classifiers reach 0.733 (income) and 0.292 (occupation). J-G's balanced accuracy, minority recall, Brier score and calibration error are all within 0.01 of U's (`ACTUAL_TASK_UTILITY.csv`).

## The registered questions (PROTOCOL §8, prompt §12)

1. **Did refitting genuinely close the measured critic gap?** Partly, and only relative to bounded critics (`CRITIC_TRACKING.csv`, registered mean paired CE gap, CALIB rows).

   | Arm | Start | Midpoint | Final aligned snapshot |
   |---|---|---|---|
   | J-O (online critics) | 0.06–0.16 nats | 0.06–0.08 | 0.06–0.07 |
   | J-G / J-R (refreshed) | ≈ 0 (−0.001 to +0.005) | 0.03–0.07 | ≈ 0 (−0.004 to +0.005) |
   | J-R at β = 0 (baseline, same start) | — | — | 0.01–0.06 |

   - Refits removed the online lag at the snapshots right after a block settled, but not mid-block.
   - Caveat: fresh critics read **less** from the refreshed arms' training views (best CE 0.55–0.59) than from J-O's (0.44–0.49).
   - The bounded critics are also blind under every transform to a planted 1e-6 clue that a scaled linear reader detects perfectly (`WHITENING_DIAGNOSTIC.json`).
   - So "tracking" here means keeping pace with a bounded critic family. It does not establish that the arms are hard to attack.
2. **Did it lower independently read recovery at useful accuracy?** **No.**
   - At matched β = 0.1, epoch 20, the online arm minus the refreshed arm (J-O − J-R) is +0.004 on the coalition (NOT_RESOLVED) and −0.019 on the income view (BELOW 0). The online schedule protected the income view more.
   - Among the selected configurations, J-O's coalition AUC is 0.816, against 0.859 for J-R and 0.855 for J-G.
   - `GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv` gives the measured reason. The online arms' per-step floored transform makes their penalty gradient about 3× larger at the same β: penalty/task norm ratio 1.55 (J-O) vs 0.52 (J-R) at β = 0.3, and 2.89 (L-O) vs 0.61 (L-R).
   - The whitening diagnostic shows the floored transform amplifies critic input gradients 30–400× relative to raw centring.
   - The online arm's advantage is therefore at least partly a gradient-scale effect, not a better critic. It cost occupation accuracy: J-O is 1.0 point below U.
3. **Did local feedback prevent the income-recipient leakage shift?** The shift did not recur, but **not because of the feedback**.
   - The multipliers stayed essentially inactive: the largest λ was 0.09 of the base weight on any run, and most runs stayed near 0.01 or 0.
   - The calibrated budgets (L-R's own level + 0.005) were rarely violated, because Stage C started from L-R and local recovery fell from there.
   - J-G's income view leaked less than L-G's (−0.007), but its occupation view narrowly failed the guard (+0.004; upper bound 0.011).
   - The fixed-β contrast J-R − J-G is NOT_RESOLVED on every view (pair −0.001, v1 +0.003, v2 +0.002), so there is no measurable feedback effect.
4. **Did J-G beat the refreshed local control rather than an old weaker schedule?** Not as registered.
   - J-G lowered coalition recovery relative to L-G by 0.012 [0.006, 0.018]. That is resolved above 0 but below the 0.02 target, and nomination failed.
   - The "old weaker schedule" (online critics) was in fact the strongest control.
5. **Did it beat J-R, or did ordinary refreshed joint training do equally well?** Equally well. J-R − J-G on the coalition is +0.004 (lower −0.003). The selected J-R and J-G are indistinguishable.
6. **Which privacy weights or constraints were active, and did they cost utility?**
   - Only the base penalty terms were active.
   - All selected trained configurations sit at the top of the registered grid (β = 0.3).
   - The local multipliers were inactive.
   - Utility cost: none measurable for J-G (−0.3 / +0.3 points vs U). J-O cost 1.0 occupation point.

## Strongest comparisons

**Favourable:**
- At U-level accuracy on both tasks, J-G lowered coalition SEX recovery relative to U by 0.026 [0.019, 0.034] and relative to L-G by 0.012 [0.006, 0.018].
- It lowered the income view's proper-loss recovery relative to L-G (LLR difference 0.033, ABOVE 0) and the income view's race recovery (0.054, PASS).
- Coalition linear R² is 0.19, against U's 0.37.

**Adverse:**
- The joint arm with the inherited online critics (J-O) beats J-G by 0.039 on the coalition and by 0.029 and 0.050 on the recipients, at equal income accuracy but 1.3 points lower occupation accuracy.
- FARE leaks far less (coalition 0.70) but fails the occupation utility gate.
- Every neural arm still leaks heavily in absolute terms (coalition AUC 0.82–0.87). The coalition adds recovery beyond the best local view for every arm (+0.009 to +0.025, ABOVE 0).
- LEACE restores linear guardedness (R² 0) but costs 5.7 income points, with nonlinear coalition recovery still at 0.863.

## Predictions (PROTOCOL §10)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Refreshed arms close most of the critic gap | Hit at the start and final snapshots; miss at the midpoint |
| 2 | L-R a nonzero nominee on ≥ 2 seeds | Hit (3/3, β = 0.3) |
| 3 | Multipliers small | Hit (largest λ ≤ 0.09 of base) |
| 4 | Claim A NOT_ESTABLISHED | Hit, via both the coalition margin and nomination |
| 5 | Claim B NOT_ESTABLISHED, C\* = L-G or L-R | Claim hit; **C\* miss** (C\* = J-O) |
| 6 | J-G within 1 point of U | Hit |
| 7 | J-G ≈ J-R; λ/(β/3) < 1 | Hit |
| 8 | J-G NO_FEASIBLE on ≥ 1 seed | Hit (3/3) |

## Counts, incompleteness and custody

**Units, all complete and hash-verified:**
- Task line: 3 runs, 24 checkpoints.
- Stage B: 18 runs, 72 checkpoints.
- Stage C: 36 runs, 144 checkpoints.
- 3 Stage C task-only receipts.
- 3 calibrations and 3 LEACE units.
- 42 re-headed FARE units: 6 configurations × 2 purposes × 3 seeds, plus the 6 zero-fairness twins at the selected config 1. All 42 reuse predecessor trees after verification; no new FARE fit was needed.
- 300 inner audits.
- 737 complete units in total.
- 15 tracking units.
- 1 whitening diagnostic.
- Optional ablation: 3 runs, 12 checkpoints.
- 39 outer units and 1 controls unit.

**Engineering and training health:**
- Parity: 21/21 receipts, and the three Stage C β = 0 receipts all bitwise.
- Rescues: 0. Nonfinite gradients: 0.
- Clip hits: 31 over 57 runs.

**Incomplete units:** none. **Not triggered:** the half-learning-rate rescue.

**Optional ablation** (capped transform, J-G β = 0.1; diagnostic, never a nominee): inner coalition AUC 0.843–0.853, against 0.844–0.858 for the floored transform. This is a small difference with no conclusion.

## Measured bottleneck and justified next decision

**The bottleneck is not critic staleness.** The refreshed critics kept pace with fresh bounded critics, yet independently audited recovery did not fall. Two things limited this candidate:
- **Penalty-gradient scale:** the protection achieved at a given β is governed mainly by the transform-amplified penalty gradient.
- **Loose local budgets:** they never engaged the feedback.

**Next decision justified by this outcome.** Do not develop J-G further as specified. If the method line continues, register a new study on fresh data that:
1. matches the penalty-gradient magnitude across critic schedules, for example a capped transform with a β grid calibrated to equal penalty/task ratios, so that schedule and scale are separated;
2. sets local budgets below the reference rather than at it, so the feedback can engage;
3. compares against the online joint arm as the expected strongest control.

`CONFIRMATION_PLAN.md` lists candidate populations and their usage histories. This development result does not justify spending a confirmation population on J-G.

Novelty is assessed separately and is not claimed (`MATH_REVIEW.md`, prior art). A failed recipe does not bound future methods.
