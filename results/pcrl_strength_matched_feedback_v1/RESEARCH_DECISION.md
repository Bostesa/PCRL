# Research decision: matched update strength and active local feedback (J-F)

**Status: EXPLORATORY DEVELOPMENT evidence** on already-exposed Adult rows. NEW_DEVELOPMENT_ASSESSMENT (3,796 rows, 3,793 groups) was withheld from this procedure until the evaluation lock. It is not fresh confirmation.

Staged locks were pushed before the stages they govern:

| Lock | Commit (time) | Pushed before |
|---|---|---|
| DATA_AND_ENGINEERING_LOCK | `6f307ee` (00:53Z) | warm starts, parity, U, raw controls |
| PHASE_A_PROTOCOL_LOCK | `65a6eeb` (01:06Z) | Phase A |
| PHASE_A_FREEZE | `15a9db9` (01:17Z) | the controller preflight and Phase B |
| PHASE_B_PROTOCOL_LOCK | `90270f0` (01:20Z) | Phase B |
| EVALUATION_LOCK | `74af248` (01:33:58Z) | the assessment, run 01:33:59–01:46:26Z |

Amendments:
- A1: audit, baselines and assessment code;
- A2 and A3: preflight, with A3 a defect fix;
- A4: inference, deployment and reporting.

**Label: EXPERIMENTAL_NO_ADVANTAGE.**

## Verdict

**The full development advantage was not established.**
- J-F had no feasible nominee on any seed. Against the strongest control (C\* = RAW-J, the faithful raw-penalty joint arm at β = 0.3, which won on every seed) its local recovery was 0.017–0.046 AUC above the +0.005 nomination buffer.
- Scored descriptively, J-F passes 8 of 9 clauses against L-F (claim A) but misses the coalition margin. It passes 6 of 9 against C\* (claim B).
- Missing clauses and the status requirement each defeat both claims.

| Assessment, seed means | Income acc. | Occupation acc. | Income view AUC | Occupation view AUC | Coalition AUC | Coalition OLS R² |
|---|---|---|---|---|---|---|
| U (task only, 40 epochs) | 0.852 | 0.480 | 0.860 | 0.880 | 0.882 | 0.36 |
| **C\* = RAW-J** (raw penalty, β 0.3, 40 epochs) | 0.853 | 0.474 | **0.768** | **0.796** | **0.810** | 0.19 |
| **J-F** (candidate; descriptive, ρ 0.25) | 0.851 | 0.481 | 0.797 | 0.846 | 0.864 | 0.22 |
| L-F (local + feedback) | 0.851 | 0.476 | 0.823 | 0.848 | 0.870 | 0.26 |
| J-N / L-N (no-feedback twins) | 0.847 / 0.848 | 0.476 / 0.478 | 0.826 / 0.825 | 0.858 / 0.858 | 0.868 / 0.873 | — |
| E (official LEACE on U) | **0.799** | 0.434 | 0.847 | 0.867 | 0.871 | 0.00 |
| F (official FARE) | 0.847 | **0.462** | 0.699 | 0.636 | 0.721 | 0.13 |

Constant classifiers reach 0.749 (income) and 0.290 (occupation).

Utility is preserved for J-F: balanced accuracy, minority recall and Brier score are all within 0.01 of U (`ACTUAL_TASK_UTILITY.csv`).

## Primary family (18 slots, z = 2.991316)

| Clause | J-F vs L-F (A) | J-F vs C\* = RAW-J (B) |
|---|---|---|
| Coalition improvement > 0.02 | +0.006 [0.001, 0.012], **NOT_ESTABLISHED** | **−0.054** [−0.067, −0.041], NOT_ESTABLISHED |
| Income-view increase < 0.01 | **−0.026** [−0.035, −0.017], PASS | +0.028 [0.015, 0.042], NOT_ESTABLISHED |
| Occupation-view increase < 0.01 | −0.002 [−0.007, 0.003], PASS | +0.050 [0.036, 0.064], NOT_ESTABLISHED |
| Accuracy vs U (income / occupation) | −0.001 / +0.001, PASS / PASS | aliases |
| Retention / gain over the constant | PASS (all four) | aliases |
| **Claim** | **NOT_ESTABLISHED** (8/9; J-F no nominee on any seed) | **NOT_ESTABLISHED** (6/9) |

## The five questions

**1. After matching actual update strength, did the schedule still matter?** No measurable difference.
- Phase A (inner, mean selected joint coalition AUC):

  | Schedule | Mean selected joint coalition AUC |
  |---|---|
  | ONLINE | 0.859 |
  | REFRESHED | 0.857 |
  | ONLINE_MATCHED | 0.865 |

- Assessment, fixed ρ = 0.75 joint points:
  - REFRESHED − ONLINE −0.003 [−0.008, 0.003];
  - ONLINE_MATCHED − ONLINE −0.001 [−0.005, 0.003];
  - REFRESHED − ONLINE_MATCHED −0.002 [−0.007, 0.004].

  All are NOT_RESOLVED.
- **Freshness.** REFRESHED critics did stay closer to fresh critics: registered final-snapshot gap 0.035 nats, against 0.139 (ONLINE) and 0.246 (ONLINE_MATCHED), on CONTROLLER_CALIB rows. This did not translate into lower audited recovery.
- **Zero-direction stalls.** At ρ ≥ 0.75 the REFRESHED encoder pushed its block-fixed critics below the constant predictor between refits. The protection direction was then exactly zero on many steps: about a quarter of local steps at ρ = 1.5. The realized strength fell below the declared ρ (`GRADIENT_MATCHING.csv`). ONLINE and ONLINE_MATCHED realized exactly ρ.
- **Strength.** At the strengths actually run, strength is what mattered. The faithful raw penalty's measured penalty/task ratio was 1.2–1.6 (β 0.1) and 2.6–3.2 (β 0.3), above the whole normalized grid (≤ 1.5). RAW-J minus the norm-controlled J-N on the coalition was −0.059 [−0.073, −0.044] (BELOW 0), at equal occupation accuracy (−0.002, NOT_RESOLVED).
- **Limit.** Norm matching does not match directions or trajectories, and stronger normalized settings were not run. So this shows that the schedule does not matter at equal strength within ρ ≤ 1.5. It does not show which strength is optimal.

**2. Did the controller truly activate and improve the protected recipient?**
- **It activated.** 16 of 18 feedback runs reached asymmetric allocations (ACTIVE or ACTIVE_ASYMMETRIC). One local run was INACTIVE_OR_ALIAS (common-mode only, identical to its twin) and one joint run was ACTIVE_COMMON_ONLY (`CONTROLLER_SUMMARY.json`).
- **It acted almost like a switch.** On seeds 0 and 2 the income recipient fell below its target within one epoch, so its weight dropped to the floor (0.25). The occupation recipient stayed above its target, so its weight rose to the cap (8). Nearly all of the fixed RMS budget went to the occupation encoder.
- **It did not improve the protected recipient.** Even at maximum allocation (s ≈ 1.41 × ρ), the occupation recipient's calibration AUC stayed above target.
- **Assessment.** At fixed ρ = 0.75, feedback versus no feedback is NOT_RESOLVED on every view:

  | Arms | Coalition | Income view | Occupation view |
  |---|---|---|---|
  | Joint | +0.002 | +0.007 | 0.000 |
  | Local | — | +0.008 | −0.005 |

- So the logged weight changes did act (parameters and updates differ from the twins), but they produced no measurable benefit.

**3. Did the joint method improve the pair without shifting recovery or sacrificing utility?** Partly.
- Against L-F, J-F lowered coalition recovery by 0.006 [0.001, 0.012]. That excludes 0 but falls far short of the registered 0.02.
- Neither recipient leaked more: the income view was −0.026 and the occupation view −0.002. Task accuracy was unchanged.
- The predecessor's income-recipient shift did not recur.

**4. Did it beat the strongest simple control and its own no-feedback twin?** No.
- C\* (RAW-J) is better than J-F by 0.054 on the coalition and 0.03–0.05 on each recipient, at equal income accuracy and 0.7 occupation points lower.
- Against its selected no-feedback twin J-N, the coalition difference is J-N − J-F = +0.005 [−0.002, 0.011], NOT_ESTABLISHED.

**5. Which component earned evidence, and which did not?**
- **Earned:**
  - update strength (the raw β = 0.3 penalty, at roughly twice the strongest normalized ratio, protects clearly more at useful accuracy);
  - the joint pair term, as a small coalition gain over local protection without recipient shift (0.006, below target).
- **Did not earn evidence:**
  - critic-refresh schedule at matched strength (indistinguishable; it also causes zero-direction stalls at high ρ);
  - the AUC-driven local controller (active but switch-like, with no measurable benefit).
- **No conclusion** that local management cannot help, or that refreshed critics are useless in general.

## Strongest comparisons

**Favourable:**
- At task-only accuracy, J-F leaks less than L-F on the income view (−0.026 [−0.035, −0.017]) with a resolved, small coalition gain (0.006).
- J-F lowers income-view proper-loss recovery relative to L-F (LLR difference 0.033, ABOVE 0).
- It lowers coalition recovery relative to U by 0.019 [0.012, 0.026].

**Adverse:**
- The faithful raw-penalty joint arm (RAW-J) beats J-F by 0.054 on the coalition and on both recipients, at equal accuracy.
- The normalized ρ grid never reached RAW-J's measured strength.
- FARE leaks far less (coalition 0.721) but loses 1.8 occupation points and fails the gates.
- LEACE removes linear signal (R² 0) but costs 5.3 income points, and its nonlinear coalition AUC remains 0.871.
- All neural arms still leak substantially (coalition 0.81–0.88), and the coalition adds signal beyond the best local view for every arm (ABOVE 0).

## Best development baseline

**C\* = RAW-J** (the inherited raw-penalty joint arm with online critics, β = 0.3, 40 epochs) is packaged as the strongest development baseline (`MODEL_MANIFEST.json`). On the assessment:

| Metric | RAW-J | U |
|---|---|---|
| Coalition SEX AUC | 0.810 | 0.882 |
| Income accuracy | 0.853 | 0.852 |
| Occupation accuracy | 0.474 | 0.480 |

**What is still missing.**
- RAW-J was a control, not the registered candidate. Its advantage over U and over the other controls is supported here only descriptively and through the secondary endpoints.
- A claim for it would need its own pre-registered comparison on fresh data.
- Its strength (penalty/task ratio about 2.6–3.2) was set by β and the transform, not matched.

## Predictions (PROTOCOL §9)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Schedules within 0.01 at matched strength | **Hit** |
| 2 | ρ = 1.5 fails a gate somewhere; selected joint ρ mostly 0.75 | Hit for REFRESHED (ρ 1.5 failed on seeds 0 and 2; selected 0.75, 0.75, 0.25). ONLINE selected ρ 1.5 everywhere |
| 3 | The controller activates | **Hit** (16/18; 8/9 J-F asymmetric) |
| 4 | Claim A NOT_ESTABLISHED via the coalition margin | **Hit** (0.006) |
| 5 | Claim B NOT_ESTABLISHED, C\* = RAW-J or J-N | **Hit** (RAW-J) |
| 6 | J-F within 1 point of U | **Hit** |

## Counts, custody and incomplete units

**Units.**
- 3 fresh warm starts, 36 parity receipts (all bitwise), U (3 runs) and the raw controls (12 runs).
- Phase A: 54 normalized runs; matched templates shared with their REFRESHED twins, not refitted.
- Calibration (3) and preflight (3, plus superseded receipts kept).
- Phase B: 36 runs, each with 21 probe measurements.
- Baselines: LEACE (3) and new official FARE fits (36 trees plus 6 zero-fairness twins).
- 165 inner audits, 15 tracking units, 51 outer units, 1 controls unit.
- Nonfinite gradients: 0. Rescues: 0. Clip hits: 2 (normalized arms) and 18 (raw arms).
- Incomplete units: none.

**Superseded receipts.** Kept and quarantined, never deleted: 36 pre-review parity receipts and 3 preflight receipts. Two of the preflight receipts were written by the pre-A3 code before it crashed, and one by the first A3 run, before the full three-seed rerun.

**Custody.** The drive copy holds 2,412/2,412 files, re-read uncached. U, J-F and L-F restore bitwise from it, and a recorded attacker refit reproduces its saved predictions exactly (`BACKUP_VERIFICATION.json`).

## Next research decision

**Do not develop the refreshed-critic schedule or this AUC controller further.** At matched strength neither showed a measurable effect, and the controller's switch-like allocation could not meet the binding recipient's target within a fixed budget. More of the same controller is not justified.

**The measured lever is update strength.** A justified next study would be pre-registered on fresh data and would:
1. extend the normalized ρ grid upward (for example to 3 and 5), so that it spans the raw arms' measured penalty/task ratios;
2. use the ONLINE schedule, which has no zero-direction stalls;
3. compare normalized and raw updates at matched measured strength;
4. report the accuracy-protection frontier rather than a single nominee.

Do not spend a confirmation population (`CONFIRMATION_PLAN.md`). A failed recipe does not bound future methods, and no novelty is claimed (`MATH_REVIEW.md`, prior art).
