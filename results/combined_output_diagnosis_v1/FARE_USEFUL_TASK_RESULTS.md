# FARE with a non-trivial utility requirement (stage 5: conditional, run)

**Status.** Run, because the mandatory stages finished well inside budget. This is development analysis, not a
retroactive replacement for the earlier HMDA verdict.

**Method.** Official FARE (eth-sri/fare 89cb1b66), the admitted isolated environment and adaptations, and the frozen
six-setting grid. Registered before any stage-5 fit: lock amendment L1 (c677ec2) and `S5_ADMISSION_NOTE.md`.

## Part A: replay of the existing Adult and HMDA FARE results

Same complete contract: features plus a head fitted only on them. `FARE_REPLAY_EXISTING.csv`; saved predictions only.

**Task gain** = U2 refitted-probe accuracy minus the attacker_fit-majority constant. **Share** = gain / untreated
gain.

| Cell, seed | Untreated: gain / recovery | Target LEACE: gain / recovery | FARE nominee: cells, gain, share, recovery | Zero-fairness twin: gain / recovery |
|---|---|---|---|---|
| Adult s0 | 0.049 / 0.674 | 0.049 / 0.679 | 10 cells, 0.048, 0.99, **0.528** | 0.052 / 0.614 |
| Adult s1 | 0.101 / 0.910 | 0.101 / 0.907 | 50 cells, 0.101, 1.00, **0.589** | 0.099 / 0.690 |
| Adult s2 | 0.094 / 0.865 | 0.094 / 0.866 | 10 cells, 0.074, **0.79**, **0.533** | 0.094 / 0.635 |
| HMDA s0 | 0.020 / 0.993 | 0.021 / 0.990 | 4 cells, 0.019, 0.92, **0.504** | 0.020 / 0.665 |
| HMDA s1 | 0.006 / 0.629 | 0.006 / 0.629 | **1 cell: a constant release**, 0.000, **0.00**, 0.500 | 0.008 / 0.619 |
| HMDA s2 | 0.023 / 0.988 | 0.020 / 0.970 | 4 cells, 0.020, 0.87, **0.500** | 0.020 / 0.532 |

**Reading:**
- On HMDA the whole task gain is 0.6–2.3 points. The seed-1 FARE nominee is a constant release that keeps none of it.
- Adult s2's nominee keeps 79 % of the gain, so it would fail this study's ≥ 80 % rule.
- The zero-fairness twin, with the same tree budget, already removes much of the recovery. On HMDA s2 it removes
  almost all of it (0.532).

## Part B: one new useful-task cell

**Screen** (validation only; `run/s5/SCREEN.json`).
- Eligibility required a frozen-head validation gain ≥ 0.03 on average and > 0 on ≥ 2 seeds, an untreated U2
  validation gain ≥ 0.03 on average and > 0 on every seed, and support.
- **Eligible:** Adult income/race, employment × {race, age_group, marital_status}, education × {sex, race, income};
  HMDA pricing × {race, sex}.
- **Ineligible:** HMDA underwriting/ethnicity (gain 0.009); HMDA fair_lending (gain 0: a constant head).
- **Chosen:** **Adult employment_analysis / age_group**. It has the highest mean validation gain (0.68), and ties are
  broken lexicographically, which puts age_group first.
- **Caveat:** the occupation-group task is easy to keep (untreated accuracy about 0.97). Earlier notes call it a
  near-recoding of the inputs, but the verifier found that no single input column determines it.
- age_group has 4 classes, all supported. The γ ceiling for this cell is 0.864, so grid settings 4–6 collapse to 1
  cell, as declared beforehand.

**Grid and nominees** (`FARE_USEFUL_TASK_FRONTIER.csv`; validation only).
- **Feasible** (U2 validation accuracy within 1 point of untreated **and** ≥ 80 % of the untreated gain): settings 1–2
  on every seed.
- **Infeasible:** settings 3–6, which have 1–3 cells and 0.28–0.70 validation accuracy.
- **Nominees:** seed 0 setting 1 (5 cells); seed 1 setting 1 (5 cells; setting 2 is an alias); seed 2 setting 2 (11
  cells).
- There is no NO_FEASIBLE_NOMINEE seed.

**S5 endpoints** (Bonferroni over 4, z = 2.4977; same paired bootstrap). Acc = U2 probe on assessment; complete
contract = features plus own head.

| Endpoint | Point | Lower bound | Decision |
|---|---|---|---|
| R(target LEACE) − R(FARE) > 0.02 | +0.151 | +0.137 | **PASS** |
| R(zero-fairness twin) − R(FARE) > 0.02 | +0.007 | +0.002 | **NOT_ESTABLISHED** |
| Acc(FARE) − Acc(untreated) > −0.01 | −0.0004 | −0.0025 | **PASS** |
| Retention: mean[A_F − 0.8·A_A − 0.2·const] > 0 | +0.139 | +0.135 | **PASS** |

**Levels:**
- Complete-contract recovery of age_group: untreated 0.731, LEACE 0.713, FARE 0.562, zero-fairness twin 0.569.
- These values follow the registered plus-surface alias rule (amendment S1, `AMENDMENT_S1_2026-10-03.md`): the first S5
  table had omitted it. No decision changed.
- Task accuracy: untreated 0.975, FARE 0.974, against a constant of 0.277.
- The FARE tree's own task accuracy is 0.93–1.00 (a compatibility diagnostic only).

**Answer.** On this cell, FARE keeps the useful task and lowers recovery far below LEACE. But **the zero-fairness
compression control does almost exactly as well**: no contribution of the fairness term beyond compression was
established. The task's structure lets a 5-cell tree keep it while discarding most age information whatever γ is.

## Native certificate

- **Original run:** UNAVAILABLE on every seed. The wrapper's feature-hash guard refused; its message counts *distinct
  shared vectors* (1 / 2 / 1). In rows, 418 / 266 / 360 of the 1,500 cert rows share a collapsed encoder vector with
  FARE fit rows, while 0 cert records are shared with fit records. This is the same rows-versus-vectors issue as repair
  R2. The A1 premise text "no certificate row byte-identical to a fit row" is inaccurate as worded; the guard actually
  applied is record identity.
- **Under the dated A1 row-identity guard** (`run/s5/CERTIFICATES_A1.json`):
  - nominee: **3.30 and 3.30 (vacuous: > 1)** on seeds 0–1, UNAVAILABLE on seed 2 (a cell missing from a certificate
    split);
  - zero-fairness twin: UNAVAILABLE.
- With 4 age groups (6 pairs) and about 1,500 certification rows, the certificate is not informative. That is a sample
  size and role limitation of this study, not evidence about FARE's safety.
- The certificate bounds demographic parity for cell-only classifiers. It is not an AUC bound.
