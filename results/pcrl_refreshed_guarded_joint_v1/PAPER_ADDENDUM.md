# Paper addendum: refreshed critics and local feedback (outcome-specific integration text)

**Scope.**
- This text is for the combined empirical-plus-method paper. The manuscript branch is not edited.
- No method advantage goes in the abstract, because none was established.
- Splitting the papers is not decided here.

## Suggested paragraph (method section, after the no-erasure penalty result)

**Motivation.** Removing mandatory erasure restored accuracy, but the follow-up showed two problems:
- the training critics lagged the encoder;
- joint protection partly shifted leakage onto one recipient.

**What we tested.** We ran one pre-registered repair on a new development partition of the same, already-exposed rows:
- every four epochs the critics are refit on a frozen snapshot, keeping the better of a continued and a fresh critic (a bounded refit, not a best response);
- each recipient's protection weight rises when its own leakage exceeds a budget set by a refreshed local-only reference;
- nomination requires both tasks to stay useful and each recipient to be no worse than every comparator.

**What we found.**
- **Critic lag closed after refits.** The measured critic lag fell to about zero after each refit, against 0.06–0.07 nats for online critics.
- **Independently audited recovery did not fall.** At the same strength, the online-critic joint arm leaked less on the coalition: AUC 0.816, against 0.855 for the candidate. Its per-step input whitening makes the penalty gradient about three times larger, at a cost of one occupation-accuracy point.
- **The local multipliers never engaged.**
- **No feasible nominee.** The candidate had no feasible nominee on any seed.
- **Below target against the local control.** Scored descriptively, it lowered coalition recovery relative to the refreshed local control by 0.012 (0.006–0.018), below the 0.02 target, with an occupation-view guard that narrowly failed.
- **Both registered claims are not established.**

The measured lever was the scale of the penalty gradient, not critic freshness.

## Table rows (keep LEACE and FARE)

| Method | Native criterion | Utility (income / occupation; U = 0.844 / 0.478) | Coalition SEX AUC | Coalition linear R² |
|---|---|---|---|---|
| LEACE on U (E) | Fulfilled (linear, fitting rows) | 0.787 / 0.444 | 0.863 | 0.00 |
| FARE (F) | Certificate not recomputed (cert pool excluded) | 0.842 / 0.455 | 0.701 | 0.14 |
| Local, refreshed + feedback (L-G) | None | 0.843 / 0.479 | 0.867 | 0.26 |
| Joint, online critics (J-O) | None | 0.842 / 0.468 | 0.816 | 0.17 |
| **Joint, refreshed + feedback (J-G; descriptive)** | None | 0.841 / 0.481 | 0.855 | 0.19 |

## Favourable and adverse evidence (both must be reported)

**Favourable:**
- At task-only accuracy, the candidate lowered coalition recovery by 0.026 (0.019–0.034) relative to task-only training and by 0.012 relative to L-G.
- It lowered the income view's log-loss and race recovery relative to L-G.

**Adverse:**
- The candidate failed nomination on every seed and both claims.
- An older, simpler schedule protected more.
- Absolute leakage stays high for every neural arm (coalition AUC 0.82–0.87).
- FARE leaks far less but fails the occupation utility gate.
- LEACE removes linear signal but costs 5.7 income points and leaves nonlinear coalition recovery at 0.863.

**Footnote.**
- Development evidence on reused rows.
- Every attack is against a declared fitted slate.
- Intervals are conditional on the fitted models.
- Released as EXPERIMENTAL_NO_ADVANTAGE.
