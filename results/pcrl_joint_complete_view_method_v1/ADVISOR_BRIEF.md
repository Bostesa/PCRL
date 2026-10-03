# Advisor brief: joint complete-view method prototype

2026-10-03. Development evidence on reused Adult rows; not fresh confirmation. The protocol was locked before any real fit (`b67664f`), and the selection was locked before outer scoring (`197f323`).

## What was built

A runnable prototype; `jcv.deploy` reproduces the saved releases bitwise.

- **Two recipients on the same people:** income and occupation group. Both are forbidden SEX.
- **Inputs:** permitted columns only. The input contract was corrected: no sex, race, income, occupation or fnlwgt.
- **Release:** two separate encoders → official LEACE → affine heads with centred logits.
- **Training:** whitened critics on each recipient's view and on the coalition. A GEM/A-GEM-style projection keeps privacy steps from raising the task losses.
- **Controls:**
  - task-only (U); U + LEACE (E);
  - local-only critics (L), the key ablation;
  - a plain penalty (JP);
  - two sequential designs (S12, S21);
  - official FARE (F) and its zero-fairness twin (F0).

## What happened

| | Income accuracy | Occupation accuracy | Coalition SEX recovery |
|---|---|---|---|
| U (task only) | 0.843 | 0.476 | 0.873 |
| E (U + LEACE) | 0.794 | 0.439 | 0.861 |
| L (local critics) | 0.782 | 0.436 | 0.830 |
| **J (candidate)** | 0.774 | 0.441 | **0.814** |
| JP (plain penalty) | 0.795 | 0.434 | 0.773 |
| F (official FARE) | **0.842** | 0.438 | **0.717** |

The constant classifier scores 0.749 on income and 0.277 on occupation group.

1. **No LEACE-based arm kept both tasks within one point of U on any seed**, so neither registered claim could be established.
   - The cause is LEACE itself: E alone costs 4.9 and 3.7 points.
   - It forces near-equal predicted >50K rates for men and women (gap 0.014, against a true 0.20).
2. **J vs L, descriptively.** J lowered coalition recovery by 0.017 [0.009, 0.024]: below the 0.02 target, though the interval excludes zero. It also halved the coalition-specific gain (0.021 → 0.009), with no local cost.
3. **The plain penalty beat the projected update.** JP was 0.040 lower on coalition recovery, with better income accuracy.
4. **FARE had the best trade-off for the income recipient:** U-level accuracy with recovery 0.72. It lost 3.8 points on occupation, though, and its certificate was vacuous or unavailable.

## Answers in one line each

| Question | Answer |
|---|---|
| Joint vs no coupling? | Not established (both arms infeasible; small descriptive gain). |
| Vs strongest control? | None feasible. Descriptively, both JP and FARE beat J. |
| Both tasks useful? | No for J. Income gain 0.025 < 0.03; >50K recall 0.19 vs 0.61. |
| Certified privacy? | No. LEACE's linear check holds on the fitting rows (all 48 units); nonlinear recovery stays at 0.71–0.86. |
| Novel? | Low. Established ingredients; the tested integration did not win. |

## Corrections and repairs disclosed

**Fixture-stage critic repairs** (before the real data were used):
- whitening;
- 5 critic steps per encoder step.

**Amendments:**
- A1: finite logits; affected no result.
- A2: race attacker on FARE cells; affected no result.
- A3: a descriptive FARE certificate.

**Selection-lock annotation:** F0's own gates pass, but it is labelled GATES_FAILED (no decision changes).

## Decision needed

**Whether the combined paper should report this as a negative method result,** alongside the empirical analysis, rather than as a method contribution. If yes, the next method attempt needs a new prospective protocol. It must either:
- drop the mandatory final linear eraser for label-dependent tasks; or
- register a utility gate that admits the known parity cost.

Either way it needs new data, not a rescore of these outer results.
