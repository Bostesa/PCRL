# Reproduction and exact replay

Read [PROTOCOL.md](PROTOCOL.md), [config.json](config.json), [comparison_rules.json](comparison_rules.json), [PREFIT_REVIEW.md](PREFIT_REVIEW.md) and the immutable protocol/reuse manifests. This is an explicit new study; historical runners and original fitting-source identities remain unchanged.

Use the existing `.venv`, local ACS cohort, original PCA maps and retained coalition forks. The raw cohort file remains `data/folktables/2018/1-Year/psam_p06.csv`, SHA256 `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`. No installation, dataset/model download or cloud compute is needed. Exact checkpoint replay requires the recorded local fitted artifacts; compact reports alone cannot regenerate person-level predictions.

A new prospective rerun requires an unused output directory with its actual UTC start, the same full protocol/configuration/numerical policy, exact historical inputs/forks and a new prefit freeze. Do not overwrite this study or count historical aliases as new fits. Commands below document the finite authorized execution; completed hash-bound units are skipped, and differing/partial evidence requires explicit diagnosis.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python -m experiments.run_acs_source_guard --out NEW_DIRECTORY --phase prepare
.venv/bin/python scripts/run_acs_source_guard_bounded.py --out NEW_DIRECTORY --phase train --max-units 1
.venv/bin/python scripts/run_acs_source_guard_bounded.py --out NEW_DIRECTORY --phase train --max-units 1
# After first shared T and guarded unit timing, complete the fixed remaining order.
.venv/bin/python scripts/run_acs_source_guard_bounded.py --out NEW_DIRECTORY --phase train --max-units 19
.venv/bin/python -m experiments.run_acs_source_guard --out NEW_DIRECTORY --phase freeze
.venv/bin/python scripts/run_acs_source_guard_bounded.py --out NEW_DIRECTORY --phase evaluate --max-units 1
.venv/bin/python scripts/run_acs_source_guard_bounded.py --out NEW_DIRECTORY --phase evaluate --max-units 23
```

The external wrapper enforces remaining scientific-process wall time5400s and total14400s with1800s reserved for publication. Whole process includes checkpoint loads, two disposable Adam proposals, projection, diagnostics, fitting, selection and scoring, including any failures. Source-only F/P share one forward trajectory per seed but retain distinct observer checkpoints and audits. There is no source/observer prefix refitting, new reserved-label fitting before global freeze, or training after that freeze. The program does not promise continuation after a stopped/sleeping session.

Utility heads select on their prescribed unweighted validation labels. Auditors use one360 path with genuine nested120/360 checkpoints and distinct terminal model/Adam/RNG states. Catch-up resets Adam and keeps epoch0 eligible, using each system's own observer. Projected singleton and public-source-head candidates are references to legal within-system fits; P wire/probability fits are deduplicated. No alternative release is concatenated or used as an auditor's input.

Tables use development evidence; native fitting-loss constraints do not certify downstream source feasibility or privacy. A rerun never turns this repeatedly studied cohort into a fresh test.

For the completed published result, restore only the compact derived score/selection JSONs, regenerate reports, and independently check the comparison arithmetic:

```sh
.venv/bin/python scripts/compact_acs_source_guard_evidence.py --out results/redesign_20260909_acs_source_guard_v1 --restore
.venv/bin/python scripts/report_acs_source_guard.py --out results/redesign_20260909_acs_source_guard_v1
.venv/bin/python scripts/verify_acs_source_guard_comparisons.py --out results/redesign_20260909_acs_source_guard_v1 --report COMPARISON_REPLAY_NEW.json
```

Historical compact score files referenced by the reuse manifest must also be restored using their original reproduction instructions if absent in a fresh clone. This restores existing derived bytes; it never retrains a reference or changes its source identity. Report regeneration does not require raw people or fitted objects. Its reporting runtime fields naturally describe the new render.

Exact local release, predictor and guard replay additionally requires every object identified in [LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md) and [LOCAL_ARTIFACTS.json](LOCAL_ARTIFACTS.json). Run read-only verification in a separate numerical slot with the same one-thread environment:

```sh
.venv/bin/python scripts/replay_acs_source_guard.py --mode training --report TRAINING_REPLAY_NEW.json
.venv/bin/python scripts/replay_acs_source_guard.py --mode scores --report SCORE_REPLAY_NEW.json
```

Training replay reconstructs literal source/full/task gradients, both Adam proposals, retained full moments, independent projection geometry, cast bounds and checkpoint identities. Score replay reconstructs released values and fitted-candidate predictions from saved objects, then independently computes masked raw-label unweighted/PWGTP metrics and validation selection. It never fits an auditor. Comparison replay uses separate standard-library arithmetic, without importing the reporter's numerical helpers. None of these commands chooses a release or consumes a new evaluation population.

The original T fitting stream omitted each-step displacement norms and dots. A separately frozen, bounded diagnostic reconstruction of its three existing source-only trajectories fills those fields without replacing fitted outputs. It must reproduce every original source-loss record, the five saved pre/post points and both interface final model/Adam states exactly. Its full process time is conservatively charged to the same scientific ceiling. Run only after the global release freeze:

```sh
.venv/bin/python scripts/run_acs_source_guard_reconstruction_bounded.py --out NEW_DIRECTORY
```

Its dedicated freeze, exact diagnostics and `REPLAY.json` live in `t_source_reconstruction/`. Canonical verifier sources are published; duplicate source snapshots remain local. This is diagnostic reconstruction of existing results, with zero new selected models or observer/auditor fits.

For focused implementation checks, compilation/source identities and completed numerical verification, use [FOCUSED_FINAL_VALIDATION.json](FOCUSED_FINAL_VALIDATION.json) and [VALIDATION.md](VALIDATION.md). The exact focused test command is:

```sh
.venv/bin/python -m pytest -q tests/test_acs_source_guard_training.py tests/test_acs_source_guard_runner.py tests/test_acs_source_guard_replay.py tests/test_acs_source_guard_reporting.py tests/test_acs_source_guard_comparisons_replay.py tests/test_acs_coalition_strength_training.py::test_beta_point_one_full_continuation_exactly_matches_historical
```

[SCIENTIFIC_RUNTIME.json](SCIENTIFIC_RUNTIME.json) accounts for every full scientific process; [runtime.json](runtime.json) separately records read-only verification/report intervals and elapsed work through the publication checkpoint. This is a completed matrix with no skipped scientific units, no operational retry and no new follow-up experiment. [FAILURES.json](FAILURES.json) preserves the limited test/reporting/provenance events and the exact T diagnostic reconstruction.
