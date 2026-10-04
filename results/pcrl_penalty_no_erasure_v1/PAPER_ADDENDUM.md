# Paper addendum: focused method improvement (outcome-specific integration text)

**Scope.**
- This is for the combined empirical-plus-method paper; the empirical manuscript is unchanged.
- No method advantage appears in the abstract: none was established.
- Whether to split the papers is not decided here.

## Suggested paragraph (method section, after the negative joint-method result)

**Motivation.** Our first prototype applied official LEACE as a mandatory final transform, and it failed the utility requirement. On these data, LEACE alone cost 4.9 points of income accuracy and 3.7 of occupation-group accuracy.

**Design.** We therefore ran one focused, pre-registered improvement on the same exposed development rows:
- the ordinary penalty update;
- joint local-plus-coalition critics (PN) against local-only critics (LN);
- no erasure at any point.

At zero penalty both reproduce the task-only model bit for bit.

**Results: removing erasure.** Removing erasure restored accuracy (β-matched: +4.3 points on income, +1.7 on occupation) with no resolved change in nonlinear SEX recovery (coalition +0.003, simultaneous interval −0.004 to 0.010). It forfeited LEACE's exact fitted-row linear guardedness: the linear R² of SEX on the coalition features rose from 0 to 0.23.

**Results: PN vs LN.** At the selected strength, PN matched task-only accuracy on both tasks and lowered coalition SEX recovery relative to LN by 0.029 AUC (0.021–0.037). About two-thirds of this came from the occupation recipient's view and one-third from reduced coalition-specific signal. However:
- the registered income-recipient guard failed (+0.010, upper bound 0.019 against 0.01);
- on one seed PN missed the frozen local allowance at selection.

The registered claims are therefore not established.

**Diagnostic.** On frozen training snapshots, fresh critics of identical architecture read more SEX information than the online critics in every case (+0.047 nats). The penalty mostly made SEX harder for the online critics to read rather than removing it, which locates the next optimiser change.

## Table rows (keep LEACE)

| Method | Native criterion | Utility (income / occupation accuracy; U = 0.843 / 0.476) | Coalition SEX AUC |
|---|---|---|---|
| LEACE (E) | Fulfilled (linear, fit rows) | 0.794 / 0.439 | 0.861 |
| Joint penalty + LEACE (JP, β = 0.1) | Fulfilled (linear, fit rows) | 0.773 / 0.440 | 0.817 |
| Joint penalty, no erasure (PN, β = 0.1) | None (linear R² 0.23) | 0.842 / 0.473 | 0.840 |
| Local penalty, no erasure (LN, β = 0.1) | None (linear R² 0.32) | 0.842 / 0.475 | 0.869 |
| FARE (F) | Certificate vacuous or unavailable | 0.842 / 0.438 | 0.717 |

**Footnote.** Exploratory development evidence on reused rows. Intervals are conditional on fitted models, and every attack is against a declared fitted slate. The model is released as EXPERIMENTAL_NO_ADVANTAGE.
