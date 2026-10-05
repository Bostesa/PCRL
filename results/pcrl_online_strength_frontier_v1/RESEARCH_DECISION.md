# Research decision: online strength frontier and fixed allocation

## Verdict

**EXPERIMENTAL_NO_ADVANTAGE (complete).** The study is complete and valid, and it is a negative result.

- Neither development criterion was met. Claims A, B and C are all NOT_ESTABLISHED.
- On inner selection, the normalized joint nominee N\* and the raw joint nominee R\* were both NO_FEASIBLE_NOMINEE. Their primary clauses are therefore descriptive only.
- The deployable best development model, fixed on inner selection before any assessment, is the incumbent **RAW-J β 0.3** (ordinary online raw joint training; C\*).

This is exploratory development on a reused benchmark: 13,936 previously used Adult rows. It is not confirmation. Intervals condition on the fitted models and the declared attacker slate, and do not correct for the adaptive history of these rows.

## Results

Means over seeds 0–2 on OSF_DEVELOPMENT_ASSESSMENT (13,936 rows, 13,929 exact-record groups). Lower AUC means less recovery.

| Model | Role | Pair SEX AUC | Income-recipient AUC | Occupation-recipient AUC | Income accuracy | Occupation accuracy |
|---|---|---:|---:|---:|---:|---:|
| U | task only | 0.883 | 0.859 | 0.878 | 0.844 | 0.475 |
| **RAW-J β 0.3** | C\*, incumbent, deployable best | **0.802** | 0.756 | 0.787 | 0.845 | 0.467 |
| NORM-J ρ 3, a = 1 | N\*, descriptive fallback | 0.786 | 0.746 | 0.758 | 0.844 | 0.462 |
| RAW-J β 0.6 | R\*, descriptive fallback | 0.763 | 0.704 | 0.742 | 0.840 | 0.456 |
| NORM-L ρ 3, a = 1 | L\*, best local | 0.841 | 0.751 | 0.777 | 0.844 | 0.466 |
| FARE (official) | reference, task-infeasible | 0.705 | 0.685 | 0.634 | 0.844 | 0.451 |
| LEACE (official) | reference, task-infeasible | 0.871 | 0.846 | 0.864 | 0.794 | 0.436 |

The constant predictor (fitted on OSF_DEFENSE_FIT) scores 0.746 on income and 0.284 on occupation. Full levels for every grid point, seed and view are in `ALL_LEVELS.csv`. Secondary slots named N\* or R\* score the descriptive fallback configurations, not valid nominees. The frontier is in `figures/fig_frontier.pdf`.

## Answers

### 1. Did online normalized training reach RAW-J's actual strength and performance?

**Strength: in mean, yes; in profile, no.**
- The incumbent's per-step realized penalty/task ratio averaged 2.98 (income encoder) and 2.85 (occupation encoder). It varied widely within a run: 10th–90th percentile 1.66–4.53, coefficient of variation 0.40. It was similar early and late (epochs 0–4 about 2.6; epochs 30–39 about 2.4–2.7).
- Symmetric NORM-J ρ 3 holds a constant 3.00 on both encoders.
- In both arms the task and protection gradients are nearly orthogonal: mean cos(t, p) between −0.02 and −0.01.
- Equal average strength does not equalise the time profile or trajectory. With a constant ratio ≠ 1, a normalized update has no rest point (MATH_REVIEW §8).

**Performance: comparable, not better at equal utility.**

NORM-J ρ 3 minus the incumbent:

| Quantity | Difference [interval] | Resolved? |
|---|---|---|
| Pair AUC | −0.016 [−0.021, −0.011] | yes |
| Income-recipient AUC | −0.010 [−0.016, −0.004] | yes |
| Occupation-recipient AUC | −0.030 [−0.036, −0.024] | yes |
| Income accuracy | — | not resolved |
| Occupation accuracy | −0.005 [−0.010, −0.000] | yes |

(Secondary family, z = 3.346.)

On inner selection, NORM-J ρ 3 failed a task gate on one seed (summed shortfall 0.004), so it could not be nominated.

### 2. Was there an improvement across the useful accuracy–protection frontier?

**No.**
- Every configuration lies on one trade-off between occupation accuracy and recovery. Less recovery is bought with occupation accuracy:
  - NORM-J from ρ 1.5 to ρ 3: pair −0.066, occupation −1.4 points;
  - from ρ 3 to ρ 5: pair −0.013, occupation −1.2 points.
- Interpolating the raw joint line between β 0.3 and β 0.6 at NORM-J ρ 3's occupation accuracy gives pair 0.784; NORM-J ρ 3 measured 0.786. The normalized point sits on the raw line, not below it. This interpolation is post hoc and descriptive.
- Joint training beats local training at matched budgets. NORM-J vs NORM-L at ρ 3: pair −0.055 [−0.063, −0.047], with occupation accuracy not resolved.

### 3. Did any fixed allocation improve protection without shifting leakage or hurting a task?

**No.** Allocation moved leakage between recipients, or traded it for occupation accuracy:

| Contrast at ρ 3 (vs a = 1) | Pair | Recipients | Occupation accuracy |
|---|---|---|---|
| Joint, a = 0.5 | +0.0005 [−0.005, +0.006], not resolved | income recipient +0.010 worse; occupation recipient −0.008 better | −0.6 points |
| Joint, a = 2 | +0.053 worse | occupation recipient +0.068 worse | +1.2 points |
| Local, a = 0.5 | −0.015 better | occupation recipient −0.030 better | −0.8 points, inner-infeasible |
| Local, a = 2 | +0.021 worse | — | — |

### 4. Did normalized training earn an advantage over strong controls?

**No.** N\* had no feasible nominee. Its descriptive fallback (NORM-J ρ 3) would have shown:
- vs L\* (pair): lower by 0.055 [0.048, 0.063];
- vs C\* (pair): lower by only 0.016 [0.011, 0.020], under the 0.02 margin;
- occupation accuracy vs U: −0.014 [−0.020, −0.007]. That is a **point loss beyond the one-point guard**, not just imprecision.

### 5. Did ordinary raw joint training meet its predeclared development comparison against strong local methods?

**No.** R\* had no feasible nominee:
- RAW-J β 0.3 is task-feasible but exceeded L\*'s local AUC + 0.005 on at least one seed.
- RAW-J β 0.6 failed the task gates.

The descriptive fallback (β 0.6) leaks less than L\* (pair lower by 0.078 [0.069, 0.086]). It loses 2.0 occupation points against U ([−2.7, −1.2]), beyond the guard. The RAW_JOINT_DEVELOPMENT_CRITERION is not met.

### 6. What is the strongest adverse comparison?

**Lower recovery costs occupation accuracy everywhere.**
- The incumbent itself costs 0.9 occupation points against U ([0.35, 1.43]).
- Its pair AUC of 0.802 is still substantial recovery of SEX; it is not safe disclosure.
- Every model that leaks less than the incumbent pays more occupation accuracy:

| Model vs incumbent | Pair | Occupation accuracy | Income accuracy |
|---|---|---|---|
| NORM-J ρ 3 | −0.016 | −0.5 points | — |
| RAW-J β 0.6 | −0.038 | −1.1 points | −0.5 points |
| FARE | −0.097 | −1.6 points; −2.4 points against U | inner-infeasible |

- LEACE keeps pair recovery at 0.871 and costs 5.0 income points.

## Process and validity

| Lock | Commit | Governs |
|---|---|---|
| DATA_AND_ENGINEERING_LOCK | `0179033` | admission, parity, fidelity, replay, timing |
| AMENDMENT_A1 | `ad5d867` | engineering amendment 1 of 2 |
| AMENDMENT_A2 | `cbe72dd` | engineering amendment 2 of 2 |
| TRAINING_PROTOCOL_LOCK | `b8b0e9e` | full bank chosen from timing |
| SELECTION_AND_AUDIT_LOCK | `a72664f` | references, inner audits, selection, tracking |
| EVALUATION_LOCK | `3554235`, pushed 04:19:24Z | the assessment, opened 04:19Z |

Every lock and amendment was on origin before its stage ran (`osf.lock.verify_lock` enforces this).

**Engineering checks (all pass):**
- admission: 3 warm starts replayed bitwise; 15 admitted releases rebuilt bitwise;
- parity: 12/12 bitwise;
- raw fidelity: 6/6, bitwise against pinned rgj;
- frozen-minibatch equivalence: relative L2 ≤ 2.9e-5;
- instrumented replays: 15/15 bitwise with the admitted checkpoints;
- bank: 48 new fits, 0 nonfinite, 0 rescued.

**Amendments A1 and A2.** Both repaired the equivalence check's design, and neither changed training. The stated cause in A1/A2 ("the constant wins every view with the snapshot critics") is corrected by the reviewer's A8: epoch checkpoints pair θ_e with the whitening transform of θ_{e−1}. On these rank-deficient views the stale transform amplifies a one-step change by about 10⁴. Training receipts show 0% no-gradient steps (VALIDATION.md).

**Deviation.** The audit owner ran the reference admission, inner gates and FARE reselection at about 03:32Z, before any lock. All 96 units were quarantined (kept) and rerun under SELECTION_AND_AUDIT_LOCK. The rerun's data files are identical, and the selected statuses are unchanged.

## Predictions (registered at `9159c57`, before any real fit)

| ID | Prediction | Outcome |
|---|---|---|
| PR1 | NORM-J ρ 3 within ±0.01 of the incumbent: P = 0.35; better by more than 0.01: P = 0.15 | Better by 0.016 on the pair, with 0.5 fewer occupation points, and inner-infeasible. **Low-probability branch, qualified.** |
| PR2 | ρ 5 task-feasible: NORM-J P = 0.45, NORM-L P = 0.50 | **Miss**: both infeasible |
| PR3 | Allocation helps: joint P = 0.25, local P = 0.20 | **Hit**: no |
| PR4 | Claim C passes: P = 0.15 | **Hit**: no (no valid R\*) |
| PR5 | Claims A and B: P = 0.10 / 0.05; N\* valid: P = 0.45 | **Hit**: no; N\* invalid |
| PR6 | C\* from RAW-J: P = 0.65 | **Hit**: RAW-J β 0.3 |
| PR7 | Replays bitwise: P = 0.9 | **Hit** |

## Decision

1. **Close this strength/allocation family without escalation.** Neither fixed higher strength nor fixed allocation helped at useful task performance. The measured limit is the occupation-accuracy cost of removing SEX recovery, not a shortage of update strength. Normalized updates at the incumbent's average strength reproduce the incumbent's trade-off rather than improving it.
2. **Package the incumbent RAW-J β 0.3 as the best development model** (ordinary raw joint training; `MODEL_MANIFEST.json`, `osf.deploy`). It still leaves substantial recovery (pair AUC 0.80).
3. **No confirmation population is spent.**

A complete negative here does not show that the desired method is impossible. What would change the picture is a mechanism that removes SEX information without spending occupation accuracy. A stronger push along the same directions does not do that.
