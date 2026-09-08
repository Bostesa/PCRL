# Focused validation and evidence replay

All historical results and execution source hashes were preserved. No scientific
run was invalidated or repeated. No representation, final eraser, attacker, or
head was refit to recreate reporting.

- [Focused tests](FOCUSED_TESTS.json): **23 passed in2.17s**. Two existing
  `torch.jit.script` deprecation warnings. Checks include source-label/column
  whitelist and missing masks, fitting pool, validation-independent final tensors,
  actual post-warmup adversary initialization, true epoch0, exact model/Adam-state
  clones, fixed final epochs and schedules, detached adversary gradients, frozen
  adversary parameters during mapper updates, zero C protection gradient, absent
  protection gradients for source heads/decoder, direct-coordinate catch-up
  fidelity, Adam reset, immutable source states, and saved selection before
  development evaluation. Reporting regressions preserve original-PCA margins,
  signed/undefined quantities, diagnostic-only saved adversaries and all five
  paired tasks. [Log](focused_tests.log).
- [Artificial full pipeline](PIPELINE_CHECK.json): .8618s, real miniature mapper,
  heads, independent auditors and catch-up;48 new candidate rows,96 probability
  arrays,14 fitting suites,4 catch-up fits. Its51 historical reference rows are
  explicitly mocked immutable fixture data, not claims of scientific replication.
  A fixture-only nested metadata assertion was corrected before passing; no
  scientific output was affected. [Fixture log](pipeline_regression_tests.log).
- [Read-only reference precheck](REFERENCE_PRECHECK.json):137 seed0 candidates
  reproduced validation probabilities exactly before scientific fitting;278
  binary references checked,3.626s. Each full seed subsequently performs the
  required reference identity/reuse checks again, included in its timed run.
- [Artifact replay](INDEPENDENT_VERIFICATION.json): all three seeds pass in1.374s.
 42 learned release applications match bitwise across all seven pools,321 new
  binary files match hashes, original household rows/groups and parent files
  match. Reconstructed phase schedules, actual saved Adam counters, exact C/D
  forks, source/attribute labels and known-label exposure, fit-only standardizers,
  fixed final epoch80, source-only fitting and freeze/selection gates all pass.
- [Independent score/model replay](SCORE_REPLAY.json):555 candidate records,
 2,220 score dictionaries and174,863 numeric values; maximum discrepancy
 6.66e−16,13.001s. All411 reused reference records identical. All288 new saved-model
  prediction sets and42 mapper outputs match bitwise. Checked42 primary,
 42 independent and42 AUROC choices,66 MLP schedules/selected epochs,33 matched
  schedule groups,12 initial-adversary fidelities and24 saved/catch-up coordinate
  and inherited-exposure records. No fitting warnings, fallback or selection error.

Execution sources were frozen at06:05:29UTC before seed0 began06:05:44UTC.
Seed0 selection was saved before its development release inference and scoring;
the same guard passed for seeds1/2. Original evaluation outcomes were previously
known and informed this stage: these checks establish within-run separation,
not untouched confirmation. All nine race categories remain in the schema;
coverage and exposed-control failures remain limitations, not passes.

Safe replay, requiring local raw records and fitted artifacts:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python scripts/verify_acs_bottleneck_artifacts.py --out results/redesign_20260908_acs_bottleneck_v1 --report /tmp/acs_bottleneck_artifact_replay_new.json
.venv/bin/python scripts/verify_acs_bottleneck_scores.py --out results/redesign_20260908_acs_bottleneck_v1 --report /tmp/acs_bottleneck_score_replay_new.json
```

Use fresh report paths. Both verifiers refuse to overwrite their output. The
artifact checker reuses small read/scoring-label helpers from the published
prior artifact verifier; these actual dependencies are in the repository.
Independent numerical replay is not independent retraining or reconstruction
of every optimization action. Physical representation-training actions total
27,500 mapper and62,500 training-adversary Adam steps across seeds; inherited
warm states are not counted twice. Downstream/fresh/catch-up fits are separate.
