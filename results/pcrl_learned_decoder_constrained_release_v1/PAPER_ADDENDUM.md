# Paper addendum — lcr (for manuscript owners)

**Status:** MECHANISM_GATE_NOT_MET. There is no Adult result. Nothing here changes any existing manuscript number.

## Allowed summary (drop-in)

> We pre-registered a mechanism gate for learned class-preserving decoders combined with confidence-constrained
> assignment search. On four registered known-law fixtures, no privacy-trained partition with the learned decoder
> reduced pair mutual information by the registered 0.01 nats beyond the strongest feasible task-only compression.
> This outcome was structurally expected and registered before the fixture stage. Every release determines both
> predicted classes, so no release can carry less pair information than the decision-only release. In these laws that
> release has zero pair information and fits the confidence budgets (in the calibrated-null fixture no task-only
> release fits). The laws were deliberately not changed. All implementation correctness checks passed. As registered,
> no Adult fit was run, so there is no Adult result.

## Supporting facts (with sources in this package)

| Fact | Value | Source |
|---|---|---|
| Gate verdict | GATE_NOT_MET, NO_FIXTURE_TRIGGERED | FIXTURE_GATE.json (commit 18e41ab), FIXTURE_LOCK 9ac4cb7 |
| Correctness checks C1–C7 | pass on all four fixtures | FIXTURE_GATE.json |
| Calibrated null, D1 vs D0 | max \|Δq\| 8.1e-13 per token | FIXTURE_GATE.json C2 |
| Miscalibrated fixture, D1 on identical tokens | log loss −0.033 to −0.083 nats, Brier −0.024 to −0.049; MI identical; 27/27 maps budget-feasible vs 0/27 under D0 | DECODER_ONLY_ABLATION.csv |
| Decoder certificates | 228 releases, all certified; stationarity ≤ 3.1e-16 | DECODER_CERTIFICATES.json |
| Search optimality on fixtures | constrained arms and C-TASK EXHAUSTIVE_OPTIMAL on F2–F4 | OPTIMIZATION_RECEIPTS.json |
| Forecast | LP1 (gate met) 0.85, scored as a miss; pre-stage update LP1-U1 0.01 | PREDICTIONS.json |

## May be said only with its label

- **Post-hoc, unregistered.** At equal (zero) pair information, decoded privacy maps keep 0.042 (F2) and 0.004 (F3)
  nats more task utility than the decision-only release. The new constrained search adds about 0.0006 nats beyond
  decoded old maps (POST_HOC_EQUAL_LEAKAGE_UTILITY.json). These are fixture numbers, not population or Adult evidence.

## Do not write

- "the method failed on Adult", "negative on Adult" or "competitive negative";
- "learned decoders / constrained search do not work", or "privacy compression is impossible";
- "the fixtures were fixed", "fixed before the run" or "designed to fail";
- any Adult number presented as a test of this method;
- any novelty claim;
- "beats Taylor / Privacy Funnel";
- "the joint program is convex";
- "fitted MI is a population guarantee".

## Prior art and scope

- Hard utility constraints, privacy-funnel objectives (Makhdoumi, Salamatian, Fawaz and Médard, arXiv:1402.1774),
  proper-loss calibration, greedy partition search and sequential collusion accounting (Taylor, Vippathalla and Coon,
  arXiv:2601.21859 v2) are established ideas.
- The sequential arms are matched adaptations, not the official Taylor solver.
- The decision-only floor is the standard decision-containment / data-processing argument; it is not a new result.
- Exposure: every Adult role has been used historically (EXPOSURE_LEDGER.md). No lcr computation used an Adult task label or SEX. Admission
  checked teacher forward passes and release re-encoding only, and the assessment rows were never unsealed.
