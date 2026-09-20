# RUN_STATUS — `pcrl_utility_extension_v1`

## Incidents

**Incident 1 (2026-09-20 06:21Z) — all nine reference audits failed on the cloud host.**
`acs_spectral_audits.build_audits` re-scores every routed H-ancestor candidate and asserts
`actual == c.metadata['validation_scores']` with EXACT dict equality
(`AssertionError('A/public_coverage', 'anchor__wire__mlp_0')`). Tier 1 gate FAILED, the scheduler
took the closeout path and stopped the instance, as designed. No scientific output was produced
and none was discarded; the Tier 1 reanalysis (which needs no audit) had already completed.

Diagnosis, measured on the cloud host before any code change
(`infra/.../diag_scores.py`, 196 routed candidates, seed 0):

| metric | max abs difference, x86-64 host vs stored arm64 records |
|---|---|
| `log_loss` | 3.9e-09 |
| per-class `auroc` | 1.5e-05 |
| `macro_auroc` / `observed_macro_auroc` | 7.6e-07 / 1.9e-06 |
| counts, supports, accuracy, precision, recall, f1, prevalence, schemas | **identical** |
| None-vs-value flips | none |

The first version of this diagnostic did NOT route the ancestor columns and reported differences
of up to 2.5 nats; that was an error in the diagnostic, not in the harness, and it is retained
here because it calibrates what a genuinely wrong route looks like.

**Amendment 1 (declared before Tier 1 was re-run; no new-release outcome had been opened).**
The check's purpose is route correctness: a routed ancestor reads the same `H` columns of the new
wire that it was fitted on, so it must reproduce its recorded scores. Exact float equality
additionally requires identical floating-point execution, which does not hold across CPU
architectures. `experiments/pcrl_utility_extension_v1/portability.py` keeps the check and wraps
`audit.metrics` for the duration of this study's audits so the comparison is tolerant:
`log_loss` within 1e-7 (25x the measured deviation, four orders of magnitude below the .001-nat
decision scale), tie-order-sensitive ranking metrics within 1e-3, **everything else exact**.
The historical module is not edited. A mis-routed candidate is still rejected (2.5 nats is
thousands of times the tolerance); fixture tests in
`tests/pcrl_utility_extension_v1/test_extension_math.py` assert acceptance of the measured
deviation and rejection of a 1e-6 log-loss shift, a 2.5-nat shift, a changed count, a changed
precision, a missing key and a None-vs-value flip. Every comparison and its realised maximum
deviation are recorded in `PORTABILITY_AMENDMENT_1.json`.

This is an implementation fix for a platform difference, supported by fixtures and measured
before the change. It is not an outcome-driven design change: no new release had been scored.

## Operational events

| utc | event |
|---|---|
| 2026-09-20 05:45 | AWS session available; private bucket, least-privilege roles, live price check |
| 2026-09-20 05:52 | host 1 bootstrapped; execution bundle restore reported FAILED with 0 bad files |
| 2026-09-20 06:00 | cause: `s3api get-object` writes response metadata to stdout after the body, corrupting the verification stream hash; readers moved to a FIFO (commit `fc771c6b`); host relaunched |
| 2026-09-20 06:07 | host 2: all three execution chunks RESTORED+VERIFIED |
| 2026-09-20 06:10 | both independent stop paths rehearsed against the verified instance ID (CloudWatch alarm stop; EventBridge Scheduler stop); START written |
| 2026-09-20 06:11 | **T0 PASS** (restore smoke, mapper parity, probe reproduction, forecast $2.62) |
| 2026-09-20 06:21 | **T1 FAIL** (incident 1); scheduler closed out and stopped the instance |
| 2026-09-20 06:28 | host 3 bootstrapped for diagnosis; measurement above; Amendment 1 |
