# Training and optimization diagnostics

Source: `TRAINING_DIAGNOSTICS.csv`, one row per finalised neural unit (51 units = 17 per seed × 3). The private per-epoch logs (surrogate R traces per step, guard losses, active-guard counts, rejections) are in each unit's `record.json`.

## Budgets and counts (identical across arms within a seed)

**Shared schedule.**
- Warm start: 20 Adam epochs, shared bitwise by every neural arm of the seed.
- Protection: 20 epochs × 76 steps = **1,520 updates per encoder in every arm**.
- S12/S21 run two stages, each updating one encoder, so they show 3,040 protection steps in total. Each encoder still gets 1,520.

**Critic budget.** 5 critic steps per encoder step, 6 critics per arm, so 45,600 critic updates per arm (S12/S21: 2 + 4 critics over two stages, the same total).

**Parameter counts.**
- Each encoder: 10,576 parameters.
- Training heads: 34 and 102 parameters.
- Deployed heads: affine StandardScaler + logistic regression, C selected on defense_val.

**Wall time per run** (CPU, one thread):

| Arm | Wall time | Notes |
|---|---|---|
| U | 0.7 s | — |
| JP | 17 s | — |
| L | 29 s | Backtracking guard evaluations dominate |
| J | 29 s | Backtracking guard evaluations dominate |
| S12 / S21 | 34 s | — |

All neural training took about 15 minutes of wall time on two workers.

## The proposed update in practice

**The projection is exact.** The maximum constraint violation over all 82,080 projected protection steps in the study (L, J, S12 and S21; 36 units) was 6.8e-13.

**The guards bind most of the time** (seed 0, number of active guards per step, counted as 0 / 1 / 2):

| Arm, β | 0 active | 1 active | 2 active | Rejected protection steps |
|---|---|---|---|---|
| L, 0.1 | 1,089 | 259 | 172 | 33 / 1,520 |
| L, 1 | 49 | 320 | 1,151 | 422 |
| L, 10 | 1 | 204 | 1,315 | 718 |
| J, 0.1 | 1,075 | 288 | 157 | 37 |
| J, 1 | 34 | 171 | 1,315 | 572 |
| J, 10 | 0 | 128 | 1,392 | 822 |
| S12, 10 | — | — | — | 1,065 / 3,040 |
| S21, 10 | — | — | — | 880 / 3,040 |
| JP (all β) | — | — | — | 0 (no acceptance rule) |

At β ≥ 1, both task guards are active on most steps, and 28–54% of the projected protection steps fail the backtracking acceptance and fall back to task-only steps.

**Cause: LEACE pushes the task losses above the budget.**
- The budget is the warm-start guard loss + 0.01 nats, and the warm start has no LEACE.
- The first in-loop LEACE map raises both task losses well above it. Guard losses after epoch 0 reach about 1.6 for income and 1.8 for occupation group, against budgets of about 0.31 and 1.24.
- Every epoch-boundary refit of the map shifts the release under the head again. The guard loss then spikes and recovers within the epoch.
- The guard rule therefore mostly allows privacy steps that keep the task loss from rising. It cannot restore utility that the linear erasure itself removes.
- **This is the scientific limitation the protocol anticipated** ("projection that removes every privacy direction while utility binds"). It is not a defect, and the constraint was not changed.

**No engineering rescue was triggered.** No nonfinite gradient occurred, and no arm rejected every protection step.

## Critics

- **Final-epoch training surrogates (seed 0).**
  - L at β = 1: R_v1 = 0.14, R_v2 = 0.17.
  - J at β = 1: R_v1 = 0.12, R_v2 = 0.17, R_pair = 0.26.
  - JP at β = 1: 0.02 / 0.03 / 0.05.
  - S12 stage 2 at β = 1: R_pair = 0.48, against a frozen first encoder.
- **The surrogates do not track the audit.** JP's surrogates are near zero, yet fresh attackers still recover SEX at 0.71–0.77. The in-training critics are weaker than the audit slate, which most often selects the defense-aware whitened MLP.
- **Training critics never establish privacy.** Only fresh attackers decide.

## LEACE

- **All 48 erasure units (E, L, J, JP, S12, S21 × 3 seeds; the 45 trained plus 3 E) pass LEACE's native check on the fitting rows.** Fit-row cross-covariance is about 1e-15 relative, the OLS R² about 1e-16, and no singular value was truncated.
- **On assessment rows**, the max absolute correlation of any released coordinate with SEX is about 0.014–0.021 for the erasure arms, against 0.31 for U (`NATIVE_VS_AUDIT.csv`).

## Utility cost of the mandatory final LEACE (descriptive, post hoc)

Means over seeds, assessment rows.

| Arm | Predicted >50K rate, men − women | True gap | Income accuracy |
|---|---|---|---|
| U | 0.18 | 0.20 | 0.843 |
| E (U + final LEACE) | 0.014 | 0.20 | 0.794 |
| L, J, JP | 0.02–0.04 | 0.20 | 0.77–0.80 |
| F (FARE, task-first tree) | 0.16 | 0.20 | 0.842 |

- Linear guardedness of a 16-dimensional task representation equalises the class-conditional means of every affine score. Here that drives the head toward demographic parity of decisions, and accuracy falls for **both** sexes (women 0.917 → 0.868, men 0.806 → 0.757 for E).
- This matches the known accuracy–parity trade-off when base rates differ (Zhao & Gordon). The finding is that this trade-off makes the registered one-point gate infeasible for every LEACE-based arm on this task pair.

## Fixture-stage repairs and amendments

- **Fixture-stage critic repairs.** See `THEORY_AND_LIMITS.md`: whitening, ZCA refreshed each step, 5 critic steps at learning rate 3e-3. These were registered before the real data were used.
- **A1:** finite centred logits from the head's decision function. It applies to every unit, with hard decisions asserted unchanged; made before any inner result.
- **A2:** the cell-conditional attacker accepts multiclass labels. It applies to the FARE race audit only; made before any F/F0 outer output existed.
- **A3** is descriptive only: FARE certificates computed under the record-identity guard (`FARE_CERTIFICATES_A3.json`).
