# Erasure on vs off: PN (no erasure) vs JP (official LEACE in the loop and as the final map)

**What is compared.** The penalty update, critics, warm starts, data order, schedule, β and head-refit rule are matched.
- The contrast measures **the whole LEACE package**: in-loop refits plus the final map.
- The trajectories differ from the first step, because JP's maps feed the task gradient, critic inputs and penalty gradient. In-loop and final-map effects are **not** separated (`METHOD_DELTA.md`).

**Data status.** EXPLORATORY DEVELOPMENT evidence on reused Adult rows. Values are assessment-row means over 3 seeds.

## Registered secondary endpoints (β-matched; mean over β ∈ {0.1, 1, 10} and seeds; z = 3.144)

| PN − JP | Point | Simultaneous interval | Classification |
|---|---|---|---|
| Income accuracy | **+0.043** | [+0.034, +0.052] | **ABOVE 0**: removing erasure restores accuracy |
| Occupation-group accuracy | **+0.017** | [+0.010, +0.024] | **ABOVE 0** |
| Coalition SEX AUC | +0.003 | [−0.004, +0.010] | NOT_RESOLVED |
| Recipient-1 (income) SEX AUC | +0.006 | [−0.003, +0.015] | NOT_RESOLVED |
| Recipient-2 (occupation) SEX AUC | −0.004 | [−0.011, +0.003] | NOT_RESOLVED |

## Per β (descriptive; seed means)

| β | Income accuracy, PN / JP | Occupation accuracy, PN / JP | Coalition AUC, PN / JP |
|---|---|---|---|
| 0.1 | **0.842** / 0.773 | **0.473** / 0.440 | 0.840 / 0.817 |
| 1 | 0.834 / 0.801 | 0.444 / 0.431 | 0.751 / 0.758 |
| 10 | 0.810 / 0.784 | 0.419 / 0.414 | 0.726 / 0.733 |

References: U is 0.843 / 0.476 with coalition AUC 0.873. E (U + final LEACE) is 0.794 / 0.439 with coalition AUC 0.861.

## Answer

- **Removing the mandatory erasure restored task accuracy.**
  - At β = 0.1, PN matches U on income (0.842 vs 0.843; >50K recall 0.60 vs 0.61; log loss 0.335 vs 0.334). On occupation group it is within 0.3 points.
  - Erased JP at β = 0.1 lost 7 income points and 3.6 occupation points.
- **Nonlinear protection did not measurably change.** Averaged over matched β, the measured coalition and local SEX recovery changed by amounts whose intervals include zero (coalition +0.003). The β = 0.1 coalition point, 0.840 vs 0.817, is 0.023 higher without erasure; at β ≥ 1 it is slightly lower. The registered average is not resolved.
- **The linear guarantee was lost, and that loss is measurable.**
  - OLS R² of SEX on the coalition features [r1, r2], fitting rows, seed means: **0.229 for PN at β = 0.1**, against 0.000 for JP and E (zero up to rounding by construction). LN at β = 0.1 is 0.315 and U is 0.369; the shuffled-label null is about 0.002.
  - Held-out R² (fitted on attacker_fit, scored on assessment): PN 0.221, LN 0.304, U 0.355, JP and E ≈ 0.
- **Conclusion.** On these models and this task pair, mandatory LEACE bought exact fitted-row linear guardedness and no measurable reduction in nonlinear recovery, at a cost of 2–7 accuracy points. Removing it restores accuracy and forfeits linear guardedness.
- **Accuracy restoration alone is not a competitive privacy-method result.** See `RESEARCH_DECISION.md`.

## Scope notes

- This does not show that exact linear erasure always forces decision-level demographic parity, nor that every erased representation must lose this much accuracy. Equal conditional means of affine scores do not by themselves imply equal rates of thresholded decisions.
- The cost was measured for these models and this task pair.
- LEACE's own theorem holds for the erased arms. Nonlinear recovery outside its scope is not a refutation of it.
