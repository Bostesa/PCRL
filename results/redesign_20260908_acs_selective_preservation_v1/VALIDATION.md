# Focused validation and independent replay

The complete selective study passes the focused implementation checks and independent replay. No scientific result was invalidated, no execution source/configuration amendment was needed, and no operational retry occurred. The protocol and scientific source dependency closure were frozen before the first teacher fit. Historical source hashes remain historical.

**Training and data boundaries.** [Focused checks](focused_tests.txt) passed32 tests covering fixed raw-coordinate targets/scales, paired teacher-label provenance, complete-case joint counts/masks, fitting-only construction, immutable teacher maps, rho0's omitted decoder gradient/Adam updates, mapper-only teacher gradients, exact full C/D forks and schedules, frozen snapshots, reserved-label timing, affine fitting and nested audits. [Reporting checks](reporting_validation.json) passed25 focused tests for fixed selection/weighting, full-schema support, original parent margins, signed contrasts, incomplete seeds and four-score paired interactions; these overlap mature helper suites and are not claimed as57 distinct new tests. Subsequent report-only changes compacted repeated coordinate metadata and clarified titles/prose without changing scores or comparison logic. Existing torch.jit deprecation notices were the only test warnings.

The [miniature historical regression](HISTORICAL_REGRESSION.json) loads the exact reviewed `e1691955c330d4782ccd600d171ae9571f23fdb5` training source using `git show` and compares an artificial R/rho.1 unit against its historical beta1-persistent path. Allseven initialization/base/fork/final checkpoint trees match bitwise, including model/observer tensors, complete Adam states, counters and schedules. No historical scientific matrix was refitted.

**Training replay.** [TRAINING_REPLAY.json](TRAINING_REPLAY.json) verifies all15 new learned units independently:420 released arrays replay bitwise, and105 mapper-gradient diagnostic points have maximum scalar discrepancy **0**. Historical initial tensors, fitting-only statistics, real labels/masks, original household rows and batch schedules match. C/D forks contain equal tensors and full Adam states. rho0 decoder tensors remain at initialization with no decoder Adam entries. Observers retain their real-label training path. I/W disposable-observer gradients and shared-fork D-objective diagnostics are explicitly distinct from applied C gradients. Runtime3.64seconds.

**Prediction, audit and teacher replay.** [SCORE_REPLAY.json](SCORE_REPLAY.json) checks all18 completed units using the existing independent scoring/literal inference mechanism, without fitting scientific models:

| Evidence | Replayed |
| --- | ---: |
| New candidate rows, including both nested budgets | 1,470 |
| Independent weighted/unweighted score dictionaries | 5,880 |
| Bitwise saved-model prediction sets | 2,940 |
| Selection records | 399 |
| Actual terminal model/Adam/RNG checkpoints | 408 |
| Nested120/360 trajectory prefixes | 204 |
| Saved direct-coordinate observer fidelity checks | 120 |
| Independently checked affine decoder fits | 252 |

All5880 score dictionaries preserve real masks, PWGTP, full SEX2/RAC1P9 schemas, undefined scores and prior signs. The maximum scalar discrepancy is **1.11e-15**, with zero errors. Fresh/catch-up candidates retain their exact fitting examples, standardizers, seeds, selected checkpoints and actual terminal states. Saved observers are linked to their own final training checkpoint and attribute; catch-up begins from exact direct-coordinate predictions with reset Adam. The separate saved-observer candidate is excluded from pooled selection, but the historical catch-up rule retains epoch0 as an eligible checkpoint. Three new catch-up candidates select epoch0 and win pooled selection (seed1 SEX: E/.1/D, E/0/C, E/0/D), so those gaps precede additional optimization. This clarifies candidate naming under the unchanged rule; it is not a post-run recipe change. Selection is frozen on unweighted validation before development scores; weighted results never select a different winner. Runtime70.55seconds.

Teacher replay independently checks the single jointly permuted complete-case assignment, joint counts, missing masks, original fitting rows/means/scales, frozen map hashes and affine application, covariance diagnostics and declared ranks. It does not refit LEACE. Geometry replay independently solves the recorded fixed-rcond system using SVD and checks predictions, prior errors, variances, ratios, direct errors and frozen coefficients. Maximum coefficient difference8.94e-8 and fitting-prediction difference6.44e-9 reflect independent algebra on ill-conditioned float32 teacher designs; their tiny singular directions and distinction from the exact projection rank remain reported. No numerical policy was changed to improve the result.

The final source/teacher/metric/selection/completion hashes, intended publication files and entry-point links are checked separately in [PUBLICATION_VALIDATION.json](PUBLICATION_VALIDATION.json) and [publication_manifest.json](publication_manifest.json). Models, raw data, releases, person rows and prediction arrays stay local, with each unit's `local_artifacts.json` recording their paths/hashes. Historical compatible artifact hashes were verified once per static seed and historical predictions/metadata are reused without fitting or changing their original source identities. This is targeted validation of the new study, not a repository-wide historical audit.

Read-only commands (report paths must be fresh):

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python scripts/verify_acs_selective_training_regression.py --report <fresh-regression.json>
.venv/bin/python scripts/verify_acs_selective_training.py --report <fresh-training-replay.json>
.venv/bin/python scripts/verify_acs_selective.py --report <fresh-score-replay.json>
.venv/bin/python scripts/summarize_acs_selective.py --out results/redesign_20260908_acs_selective_preservation_v1
```

The miniature regression fits only disposable artificial examples; the other verifier commands do no training. Reporting takes the saved compact results and performs no scientific fitting. [reporting_execution.json](reporting_execution.json) identifies the final rendering command, source hash and measured runtime. Independent scientific/text reviews checked the decision's numerical claims, seed signs, weighting, audit exposure, geometry interpretation and one next-design recommendation.

Finite predictive audits do not establish privacy or fix absent category support. All nine race classes remain visible; original racecode4 remains absent from attacker fitting/validation. No I/W protection audit is inferred from teacher audits. All original evaluation households remain DEVELOPMENT EVALUATION.
