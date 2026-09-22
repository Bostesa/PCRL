# Reproducing and verifying the task-directed release study

This is a source-audited command guide, not a result or completion certificate. Use [REVIEW_INDEX.md](REVIEW_INDEX.md) to locate the evidence and pending outputs. The ACS cohorts are historically reused 2018 development data; a new run on them is not fresh confirmation. **2016 remains sealed.** These commands neither fetch nor authorize opening it.

## Source, environment, and synthetic checks

Use the published source commit identified by the final handoff/archive index, once generated. Match the scientific Python and package versions in [ENVIRONMENT.json](ENVIRONMENT.json); it records the execution environment, not a portable environment lock. The historical source commits are pinned in [CONFIGS.json](CONFIGS.json) and [ARTIFACT_DEFINITIONS.json](ARTIFACT_DEFINITIONS.json). Figure rendering additionally requires Matplotlib. Archive commands require `zstd` and an authenticated AWS CLI with access to the exact private archive objects.

Run from that checkout's repository root, using the intended isolated Python environment:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python experiments/pcrl_task_directed_release_v1/fixtures/pcrl_score_refinement_fixture.py
python -m pytest -q tests/pcrl_task_directed_release_v1
```

The first command is a standard-library rational fixture and prints its independently checkable result; it reads no ACS data. The second runs synthetic/integrity tests. Neither establishes empirical ACS performance. The fixture's original source, manifest and output are under `experiments/pcrl_task_directed_release_v1/fixtures/`; independent checking is documented in `FIXTURE_VERIFIED.json` and `INDEPENDENT_FIXTURE_REPLAY.json`.

## Verify preserved artifacts without refitting

Public Git contains source, registrations, aggregate reports and hashes. Private archive members include original-person identities, labels, weights, predictions/losses, split manifests, fitted models, channels and empirical tables. Keep the download cache, extracted inputs, models and detailed replay reports in task-owned private storage. Do not print row arrays, upload them to public Git, or resolve missing files through an unrelated checkout.

The original execution root is `/opt/pcrl/work`. A restored root is separate and explicit. Saved absolute model paths are remapped from `original_root` to `artifact_root`; verification rejects missing or escaping paths and never falls back to the original directory.

### Download, authenticate and restore the archive

Wait for a finalized, published `ARCHIVE_INDEX.json`. Its bucket, object keys, version IDs and compressed hashes are the retrieval contract. The private manifest has its own published hash and identifies every member. The archive module's CLI **publishes** an archive; restoration uses `archive.verify_and_restore` as below.

Set three task-specific absolute paths. `PCRL_ARCHIVE_CACHE` and `PCRL_RESTORE_ROOT` must be new directories; `PCRL_ARCHIVE_INDEX` is the trusted public index file. This recipe restores all three anchors. To reproduce the narrower publication readback instead, use `index['restore_prefixes']` as the prefixes.

```sh
export PCRL_ARCHIVE_INDEX=/absolute/path/to/published/ARCHIVE_INDEX.json
export PCRL_ARCHIVE_CACHE=/absolute/path/to/new-private-archive-cache
export PCRL_RESTORE_ROOT=/absolute/path/to/new-private-restored-study
python - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
from experiments.pcrl_task_directed_release_v1.archive import chunk_records, verify_and_restore

index = json.loads(Path(os.environ['PCRL_ARCHIVE_INDEX']).read_text())
cache = Path(os.environ['PCRL_ARCHIVE_CACHE']); cache.mkdir(parents=True, exist_ok=False)
restore = Path(os.environ['PCRL_RESTORE_ROOT'])
assert not restore.exists(), 'Restore target must be new'
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def get(key, version, path):
    args = ['aws', 's3api', 'get-object', '--bucket', index['archive_bucket'], '--key', key]
    if version:
        args += ['--version-id', version]
    subprocess.run(args + [str(path)], check=True, stdout=subprocess.DEVNULL)

manifest_path = cache/'MANIFEST.private.json'
get(index['manifest_key'], index.get('manifest_version_id'), manifest_path)
assert sha(manifest_path) == index['manifest_sha256']
manifest = json.loads(manifest_path.read_text())
groups = chunk_records(manifest['files'])
assert len(groups) == len(index['chunks'])
study = 'pcrl_task_directed_release_v1'
prefixes = [f'experiments/{study}', f'tests/{study}', f'results/{study}',
            'experiments/acs_transfer_data.py']
for number, (chunk, records) in enumerate(zip(index['chunks'], groups)):
    path = cache/f'part-{number:04d}.tar.zst'
    get(chunk['key'], chunk.get('version_id'), path)
    assert path.stat().st_size == chunk['bytes'] and sha(path) == chunk['sha256']
    checked = verify_and_restore(path, records, restore_root=restore, restore_prefixes=prefixes)
    assert checked['verified_files'] == len(records)
print('All indexed streams and members authenticated and requested members restored.')
PY
```

Every archive member is hashed, including members omitted by a narrower restore. The restore API rejects undeclared/duplicate members, unsafe paths, sealed-year paths, symbolic links, changed bytes and overwrites. A compressed hash alone does not establish per-file integrity. Model deserialization belongs after these authenticity checks and is limited to these trusted study artifacts.

### Check the frozen raw-to-PCA input adapter

This feature-only command reads the ten permitted raw covariates on the first 128 frozen representation-fit rows per anchor. It checks saved PCA/standardization, unchanged H_A and exported-map reload parity; it reads no labels or test members and fits nothing. It needs all owned historical input bundles, which are included even in the representative anchor-0 restore. Choose unused output paths:

```sh
python -m experiments.pcrl_task_directed_release_v1.inference \
  --inputs-root "$PCRL_RESTORE_ROOT/results/pcrl_task_directed_release_v1/private/inputs" \
  --report-path "$PCRL_RESTORE_ROOT/results/pcrl_task_directed_release_v1/private/feature-recheck.json" \
  --export-root "$PCRL_RESTORE_ROOT/results/pcrl_task_directed_release_v1/private/feature-recheck-exports" \
  --rows 128
```

### Full independent replay of preserved models

With all three anchor trees restored and final selection/evaluation artifacts present:

```sh
python -m experiments.pcrl_task_directed_release_v1.verify \
  --artifact-root "$PCRL_RESTORE_ROOT" \
  --original-root /opt/pcrl/work \
  --out-dir "$PCRL_RESTORE_ROOT/results/pcrl_task_directed_release_v1/private/full-replay" \
  --include-evaluation --wire-repetitions 512
```

Omit `--include-evaluation` for validation-only replay; that report does not verify test predictions. The output directory must be new and inside the explicit restored study's `private` directory. This reconstructs frozen selection and the complete contrast family from accepted validation summaries, loads preserved model weights, checks expected token log loss and probabilities, H/ancestor parity, mathematical channel diagnostics, and sampled-wire expectations. It does not refit or nominate again using evaluation data. Explicit supplemental eraser slice/moment checks are included when those releases are replayed.

`parallel_verify` is the bounded original-study execution wrapper. Its CLI accepts the same explicit roots, `--writers-closed`, `--max-workers 8`, `--deadline-utc` and an optional `--public-report`. It verifies every endpoint configuration plus H/J at all three anchors, all 16 roles and wire512, after one global selection reconstruction. It enforces the registered execution cutoff; it is not a general later-date replay command. Use the serial `verify` command above for later artifact verification.

### Representative anchor-0 restore

Publication readback restores global study documents/sources, all inputs and deployment input maps, the complete indexed anchor-0 tree, and all indexed `private/numerical_recovery` and `private/numerical_originals` artifacts needed for repaired-map provenance. Other anchors are still verified in the archive stream. Original and retry versions remain preserved; their old registries require explicit version-aware path relocation against their preservation manifests and are not immediately executable at moved paths. That representative scope can run the helper below without reading anchor-1/2 active audit registries. Use the selection hash from the separately completed full original-root verification, not a hash invented by this representative check:

```sh
export PCRL_VERIFIED_SELECTION_SHA256=sha256_from_the_full_original_root_verification
python - <<'PY'
import json, os
from pathlib import Path
from experiments.pcrl_task_directed_release_v1.verify import verify_restored_unit
root = Path(os.environ['PCRL_RESTORE_ROOT'])
out = root/'results/pcrl_task_directed_release_v1'
frozen = json.loads((out/'SELECTION.json').read_text())
name = frozen['routes']['utility_first']['nominee'] or 'H'
result = verify_restored_unit(name, anchor=0, artifact_root=root,
    original_root='/opt/pcrl/work', out_dir=out/'private/representative-anchor0-replay',
    include_evaluation=True, wire_repetitions=512,
    expected_selection_sha256=os.environ['PCRL_VERIFIED_SELECTION_SHA256'])
assert result['passed'] and result['selection_recomputed'] is False
PY
```

This is a representative preserved-model replay, not full three-anchor selection reconstruction. Missing anchors never authorize fallback to original files.

## Historical full-fitting and inference entry points

A full refit trains teachers, channels and audit models; verification above reuses saved models. The commands here document the scientific execution order when those writes were authorized. They are **not** instructions to resume a closed study. A later fresh replication needs its own dated execution/resource registration and separate output namespace. Do not change this study's immutable registrations or bypass scheduler guards.

The primary scheduler requires the frozen `RESOURCE_SCHEDULE.json`, matching sources/configuration and restored permitted inputs. Its actual DAG is `scheduler.phases()`: prepare, H, the mandatory positive-budget T0 privacy block, then the registered unconstrained/refined/zero-budget maps and matched controls. Run the primary and only the already registered/scheduled extension commands:

```sh
python -m experiments.pcrl_task_directed_release_v1.scheduler
python -m experiments.pcrl_task_directed_release_v1.scheduler --extension baseline
python -m experiments.pcrl_task_directed_release_v1.scheduler --extension A
python -m experiments.pcrl_task_directed_release_v1.scheduler --extension C
```

The baseline extension requires `BASELINE_SUPPLEMENTS.json` and `BASELINE_SUPPLEMENT_SCHEDULE.json`, containing all 12 mandatory audits. Its four variants preserve original 48% controls and add mechanism40/union88 fits. Before their original fits, `baseline_supplement.register()` wrote metadata, that source/registry was committed and published, and `freeze_schedule(resource_decision={'registry_commit': FULL_40_CHARACTER_COMMIT, ...})` pinned the schedule. These are one-time registration steps, not commands to repeat on a restored finished run.

Branch A uses `EXTRA_CONFIGS.json` and `EXTRA_RESOURCE_SCHEDULE.json`. Its one-time orchestration entry point is `python -m experiments.pcrl_task_directed_release_v1.programme branch-A --resource-decision PATH.json`; `--include-outer-budgets` changes which prospectively permitted blocks are scheduled and is not interchangeable with the frozen executed schedule. Branch C uses `robustness.register_robustness()` followed by `freeze_robustness_schedule(resource_decision=...)`, then `scheduler --extension C`. Both branches require their registered trigger and immutable schedule before affected fits. Registered but resource-unscheduled settings remain visible in the combined ledger.

For an individual historically authorized unit, the actual CLI is `python -m experiments.pcrl_task_directed_release_v1.run audit --anchor 0 --name CONFIGURATION`; `prepare`, `map` and `benchmark` are other fitting entry points. They may fit or create artifacts and are not verification commands. Receipts pin original caches, models, roles and dependencies. Numerical recovery has separate prospective documentation and preserved diagnostics; do not substitute a diagnostic recovered channel for an accepted original fit or rewrite accepted caches.

Close all scientific writers before final validation selection:

```sh
python -m experiments.pcrl_task_directed_release_v1.programme freeze
```

This checks scheduled completion, collects validation, freezes common-configuration choices and both-weighting predictor identity, binds accepted audit/source hashes, and writes `SELECTION.json`, `CONTRASTS.json` and `SELECTION_INPUTS.json`. Missing scheduled units require an explicit truthful `--closeout-reason`; they remain unavailable and cannot create scientific passes. Publish the frozen selection and contrasts before evaluation access. Only then:

```sh
python -m experiments.pcrl_task_directed_release_v1.scheduler --evaluate
python -m experiments.pcrl_task_directed_release_v1.reporting inference
```

Evaluation consumes frozen predictors. Inference uses the explicit contrast family, common household multiplicities across arms/anchors, and recomputed survey ratios. It writes `PAIRED_BOUNDS.json` and `CLAIM_RESULTS.json`. All results remain 2018 development evidence; neither a favorable validation screen nor an attribution availability flag is an adjusted scientific claim.

## Derive aggregate reports, plots and exact equivalence

After the relevant accepted artifacts and frozen inference outputs exist, derive the inventory and evidence. The ledger command changes only public bookkeeping and preserves predecessor bytes/hash; it never schedules or fits a unit. Add `--check-only` to inspect counts without writing.

```sh
python -m experiments.pcrl_task_directed_release_v1.ledger
python -m experiments.pcrl_task_directed_release_v1.evidence \
  --ledger results/pcrl_task_directed_release_v1/ALL_RELEASES.json
python -m experiments.pcrl_task_directed_release_v1.attack_calibration
```

Evidence consumes all registered units and separately counts scheduled, resource-unscheduled and incomplete scheduled units. Per-person artifacts remain private. Attack calibration summarizes stored validation candidate/checkpoint metadata and inherited H slack; it does not refit attacks. Subjective prediction assessment is the pure `predictions.assess_predictions` API, using explicit native validation, frozen claim results, exact zero certificates and selection/provenance inputs under `PREDICTION_ASSESSMENT_RULES.json`.

Figures have a Python API, not a CLI. Use the receipt-bound aggregate grid and combined ledger metadata so branch controls and fine-conditioning maps are labelled correctly. Choose a fresh output directory:

```sh
python - <<'PY'
import json
from pathlib import Path
from experiments.pcrl_task_directed_release_v1.figures import render_figures
out = Path('results/pcrl_task_directed_release_v1')
ledger = json.loads((out/'ALL_RELEASES.json').read_text())
metadata = {r['configuration']: r for r in ledger['records']}
render_figures(out/'EVALUATION_GRID.json', out/'figures', metadata=metadata)
PY
```

`POINTS.json` and `FIGURE_MANIFEST.json` bind plotted values, receipt declarations, metadata and image hashes. Descriptive frontiers do not become simultaneous inferential claims.

Exact channel-equivalence collection needs the **original map specifications**, including branch provenance. Combined ledger records contain additional bookkeeping fields and must not be substituted for receipt-bound map specs:

```sh
python - <<'PY'
import hashlib, json
from pathlib import Path
from experiments.pcrl_task_directed_release_v1.equivalence import collect_equivalence
out = Path('results/pcrl_task_directed_release_v1')
specs = json.loads((out/'CONFIGS.json').read_text())['maps']
for name, pin, source in (
    ('EXTRA_CONFIGS.json', 'extra_registration_sha256', 'branch_source_sha256'),
    ('ROBUSTNESS_CONFIGS.json', 'robustness_registration_sha256', 'robustness_source_sha256')):
    path = out/name
    if path.exists():
        raw = path.read_bytes(); record = json.loads(raw)
        specs += [{**s, pin: hashlib.sha256(raw).hexdigest(), source: record[source]}
                  for s in record['maps']]
collect_equivalence(out_root=out, specs=specs, out_path=out/'EXACT_EQUIVALENCE.json')
PY
```

This groups exact saved channels/actions and records support, witness and missing/invalid-artifact distinctions. It does not count near-equal floating-point outputs as an exact proof or establish equality outside its declared support. The continuous eraser supplements are not Q-map equivalence candidates. `python -m experiments.pcrl_task_directed_release_v1.zero_certificate` separately reconstructs exact supported zero-budget feasibility certificates and keeps rational witnesses private.

For an independent empirical mathematical check of original-root primary maps, the API is `math_replay.verify_anchor(anchor, include_evaluation=False, out_dir=NEW_PRIVATE_DIR, public_path=NEW_AGGREGATE_JSON, inputs_root=OWNED_INPUTS)` for each anchor. It loads accepted frozen maps and does not solve a new optimization. Unlike the explicit-root `verify` adapter, this helper uses the executing checkout's `run.OUT` for fitted artifacts; use the archive adapter for relocated models.

## Publish and read back the private archive

Archive publication is an external write, used only after scientific writers are closed and the destination is the authorized task-private encrypted/versioned store. The actual publication CLI is:

```sh
python -m experiments.pcrl_task_directed_release_v1.archive \
  --bucket AUTHORIZED_PRIVATE_BUCKET --prefix TASK_OWNED_PREFIX \
  --directory /absolute/task-private/new-archive-staging \
  --restore-root /absolute/task-private/new-readback-root \
  --source-commit FULL_PUBLISHED_SOURCE_COMMIT
```

If required by the bucket permission model, provide both `--bucket-attestation PATH` and `--bucket-attestation-sha256 SHA256`; their validation is not optional when supplied. Publication uploads encrypted objects, retrieves the exact versions, checks compressed and uncompressed hashes, performs the declared representative restore, and writes `ARCHIVE_INDEX.json`. It does not delete originals and does not itself execute restored-model prediction replay; run the representative helper above and retain its report. Keep any final report generated after archive sealing separately hash-pinned, or publish a subsequent declared archive version rather than claiming it was in an earlier manifest.

No restart/resume is needed to review a closed positive or negative study. Final completeness, exclusions and verification coverage must come from generated receipts/reports, not from this command guide.
