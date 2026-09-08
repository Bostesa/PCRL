# Focused validation and new-only replay

[FOCUSED_TESTS.json](FOCUSED_TESTS.json) records18 focused tests across training,
runner and reporting commands. A further26-test comparison-helper command includes
five of those same new tests; it is not added as distinct coverage. No broad
repository suite or historical experiment was rerun. Existing Torch deprecation
warnings did not affect the checks.

- Ten training checks (six existing, four new) passed2.37s: analytical overwrite,
  constant-coordinate/validation parity, RNG/nonmapper identity, unused readout
  gradients, I/W immutability, source-label whitelist, C/D forks and counts.
- Three runner checks (two existing, one new) passed3.57s. The artificial miniature
  new branch completed I/W and both final arms, fitted68 candidates in24 suites,
  saved136 validation/evaluation probability arrays, and checked all representation
  training finishes before reserved-label heads. Evaluation followed saved choices.
  I/W had no new auditors or transferred adversaries.
- Five stage/selection/margin reporting checks passed.02s. Independent and inclusive
  selectors stay distinct; I/W cannot acquire fabricated audit scores; signed
  stage comparisons and tiny nonzero differences remain visible. The targeted
  combined comparison-helper command passed26 tests in.03s.

[TRAINING_VALIDATION.json](TRAINING_VALIDATION.json) additionally records an
artificial comparison with the exact pre-edit source: all seven default-random
nested checkpoints, model/Adam states, counters, curves and schedules remained
identical. Added descriptive metadata/source/runtime fields were not expected to
be identical. The old temporary source was not archived into this publication.

## Scientific artifacts and scores

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python scripts/verify_acs_pca16_init.py \
  --out results/redesign_20260908_acs_pca16_init_v1 \
  --report results/redesign_20260908_acs_pca16_init_v1/SCORE_REPLAY.json
```

The verifier refuses to replace an existing SCORE_REPLAY.json; omit `--report`
for a later stdout-only replay. Exact object replay requires local fitted files.
[SCORE_REPLAY.json](SCORE_REPLAY.json) passed in7.579s:

- 204 new candidate records,816 score dictionaries,58,492 numeric comparisons,
 maximum absolute metric error4.44e-16;
- 408 bitwise saved-model prediction replays and84 bitwise release applications;
- 72 primary and72 independent choices,180 family choices,96 MLP curves/schedules,
 180 fitting-only standardizers,24 catch-up coordinate checks and12 saved-adversary
 fidelity checks;
- Original genuine zero-update nonmapper/preprocessing tensors, analytical I
 weights, W unchanged through adversary warmup, exact C/D complete Adam/model
 forks, every saved Adam step and identical historical schedules/budgets;
- All615 copied historical raw records unchanged, with original files/provenance
 hashes retained and no old scientific inference in this replay.

Maximum I/PCA16 coordinate error was4.7684e-7; the tolerance remained1e-5 absolute
and relative. Per-pool maximum errors are in the replay; RMS and disposable
nonzero readout gradients are in each seed's training.json. Every seed's frozen
maps, buffers, outputs, saved adversaries and selections remained unchanged.

The verifier's first local invocation had an alias-shadowing typo before reading
scientific data; only the verifier was corrected. No scientific run, score,
checkpoint, selection or frozen execution source was replaced or regenerated.

Training and verification both use only the original sampled rows. The raw CSV
is read to reconstruct the prescribed cohort; unused households contribute no
training, selection or evaluation examples. The final household pool is explicitly
development evidence, not untouched confirmation.

All9 race categories and existing support/exposed-control limitations remain.
Inspect [original support](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md),
new `seed_*/support.json`, and [new class scores](PER_CLASS.csv). Absent code4 and
failed control coverage do not count as protected. No universal inference guarantee.
