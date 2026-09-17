# Review index: locked 2017 transport of the fixed 14-interface ACS slate

Start here. Branch `ablations-facct-2026-07-24`; study directory `results/redesign_20260917_acs_spectral_transport_v1`.

| Question | File |
|---|---|
| What is the scientific answer, and what happens next? | [RESEARCH_DECISION.md](RESEARCH_DECISION.md) |
| The manuscript | [PAPER_DRAFT.md](PAPER_DRAFT.md), bibliography [references.bib](references.bib) |
| What was fixed before any 2017 outcome was opened? | [PROTOCOL.md](PROTOCOL.md) (+ dated amendments), [COMPARISONS.json](COMPARISONS.json), [PROTOCOL_FREEZE.json](PROTOCOL_FREEZE.json) |
| What was sealed, and when? | [TRANSPORT_LOCK.json](TRANSPORT_LOCK.json), [TRANSPORT_LOCK_AMENDMENT_1.json](TRANSPORT_LOCK_AMENDMENT_1.json), [TRANSPORT_LOCK_AMENDMENT_2.json](TRANSPORT_LOCK_AMENDMENT_2.json), [FINAL_ACCESS_LOG.json](FINAL_ACCESS_LOG.json) |
| Transport results, both modes | [TRANSPORT_RESULTS.md](TRANSPORT_RESULTS.md) |
| 2018 development results (reused, not rerun) | [DEVELOPMENT_RESULTS.md](DEVELOPMENT_RESULTS.md), and the development study's [DECISION_REPORT.md](../redesign_20260910_acs_residual_spectral_v1/DECISION_REPORT.md) |
| What is proved and what is not | [METHOD_AND_SCOPE.md](METHOD_AND_SCOPE.md) |
| Prior work and novelty | [PRIOR_WORK_AND_NOVELTY.md](PRIOR_WORK_AND_NOVELTY.md) |
| Why 2017 is admissible, and its limits | [DATA_ADMISSION.md](DATA_ADMISSION.md) |
| Machine-readable results | [PER_SEED.csv.gz](PER_SEED.csv.gz), [AGGREGATE.csv.gz](AGGREGATE.csv.gz), [FAMILIES.csv](FAMILIES.csv), [evidence/](evidence/) |
| Figures | [figures/](figures/) |
| Verification | [VALIDATION.md](VALIDATION.md), [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json), [PREDICTION_REPLAY.json](PREDICTION_REPLAY.json), [VALIDATION_SCORE_AUDIT.json](VALIDATION_SCORE_AUDIT.json) |
| How to rerun it | [REPRODUCE.md](REPRODUCE.md), [RUN_STATUS.md](RUN_STATUS.md) |

## Reading the result in one paragraph

The channel preserves the published service outputs exactly and adds real reusable capability (≈0.025–0.030 nats of residence prediction over the service alone, versus ≈0.017–0.019 for the frozen neural channel J). On the locked new-year evaluation, coalition conditioning beats both its equal-strength and its equal-total-mass local control on sensitive recovery, under simultaneous intervals and both weightings — the criterion development had failed for lack of resolution. It nevertheless leaks several times more sex and race than J at the same width, so the tested design is not competitive: an evaluation finding and a qualified tradeoff, with a negative method verdict.

## Schemas

`PER_SEED.csv.gz`: seed, mode (A frozen / B fresh), condition, scope, budget, weight, kind (utility_loss, attack_loss, absolute_recovery, additional_recovery, utility_gain_vs_H), endpoint, value, selected_candidate, coverage_complete.
`AGGREGATE.csv.gz`: the same keys at budget 360 with seed_0/1/2, seed_mean, seed_sd, n_negative_seeds and, for the three bootstrapped scopes, boot_estimate/boot_low/boot_high.
`FAMILIES.csv`: family, left, right, endpoint, weight, estimate, per-seed values, se, unadjusted and adjusted interval endpoints, degenerate flag, critical_value.
