# Advisor brief: matched update strength and active local feedback

2026-10-05. This is development evidence on already-exposed Adult rows, using a new 3,796-row partition withheld until the evaluation lock. Five staged locks were pushed before the stages they govern.

## What we tested

The last study suggested that the old online-critic arm protected more only because its protection push was about 3× larger. We tested this directly:
1. Make the protection push a fixed fraction (ρ) of each encoder's task push, then compare critic schedules at equal strength (online, refreshed, and online with extra updates matched to the refreshed effort).
2. Steer protection effort between recipients with a controller that reads each recipient's actual SEX AUC and has a target slightly below the local reference.
3. Compare joint and local protection, with and without feedback, under identical inputs, budgets and attacks. A faithful raw-penalty arm was kept as a serious baseline.

## What happened

| | Income acc. | Occupation acc. | Coalition SEX AUC |
|---|---|---|---|
| Task only (U) | 0.852 | 0.480 | 0.882 |
| **Strongest control: raw penalty, β 0.3 (RAW-J)** | 0.853 | 0.474 | **0.810** |
| J-F (candidate; no valid nominee) | 0.851 | 0.481 | 0.864 |
| L-F (local + feedback) | 0.851 | 0.476 | 0.870 |

1. **No development advantage.** The candidate never qualified as a nominee, because the raw-penalty control was clearly stronger. Against the local twin it gained only 0.006 on the pair, against a 0.02 target.
2. **At equal strength the schedule does not matter.** Online, refreshed and matched schedules are within 0.003 on the assessment. The refreshed critics stay closer to fresh critics but give no protection advantage, and at high strength they stall: the encoder pushes them below the constant predictor.
3. **The controller really engaged, but it did not help.** It switched nearly all effort to the occupation recipient and still could not bring that recipient to its target. Feedback versus no feedback is unresolved everywhere.
4. **Strength is the lever.** The raw arm's measured push (2.6–3.2× the task push) exceeded the whole normalized grid (≤ 1.5×), and it protected 0.059 better on the pair than the matched joint arm at equal accuracy.

## Decision needed

- Close J-F as EXPERIMENTAL_NO_ADVANTAGE, and keep RAW-J as the best development baseline. Both are packaged and runnable.
- If the line continues, register a strength-frontier study on fresh data. It would extend ρ upward to cover the raw arm's strength, use the online schedule, and compare normalized and raw updates at matched measured strength.
- Do not pursue the refresh schedule or this controller further. Do not spend a confirmation population.
