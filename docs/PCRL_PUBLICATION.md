# Publication scope and reproduction

The [review index](PCRL_REVIEW_INDEX.md) is the entry point. This publication
packages the completed September 7 redesign work on
`ablations-facct-2026-07-24`, whose pre-publication HEAD was
`15dbcc3c3338f6707e7a0b3901d738d99651a77d`. Original experiment commits, hashes,
protocols, selections and results are preserved verbatim. They describe execution
in the then-uncommitted working tree, not execution at the new publication commit.
Older protocol statements about not pushing and proposed next experiments are
historical chronology. The user's subsequent publication authorization applies
to this packaging step.

## What GitHub contains

The four main reports; implementation repairs and their callers; all eleven new
experiment/helper modules, four reporting scripts and nine new test files; an
observed-version [environment snapshot](../requirements-redesign.txt); and compact
original evidence from all five synthetic stages plus the application screen.

Original evidence includes every `seed_*/metrics.json`, root summaries, protocols,
tables, analyses, configuration, split manifests, selections saved before tests,
available verification and execution logs, per-arm test scores, frozen-diagnostic
learning curves, and small PNG/PDF plots. Some raw JSON files exceed GitHub's
inline preview limit; use its Raw/download interface or a clone. No scores were
recomputed or altered during packaging.

The [artifact manifest](../results/redesign_publication_20260907/artifact_manifest.json)
lists **all 3,570 original files** in the six completed result directories with
paths, SHA256 hashes, byte counts and availability. **250 files / 30,589,282 bytes**
are published; **3,320 files / 204,565,835 bytes** remain local. The manifest itself
and new publication notes/checks are additional packaging records. Redundant
per-arm/probe metadata remain represented inside the canonical per-seed JSON;
their separate original files are listed as omitted rather than silently deleted.

The [source equivalence record](../results/redesign_publication_20260907/source_equivalence.json)
maps omitted source-snapshot files to identical canonical source files where
available. All four nonlinear frozen runner/dependency manifests match the
published executable sources. Old status/report snapshots may represent earlier
text and are not substituted for the current reports. The README review notice
and status publication append are explicitly later documentation changes.

## What remains local

| Artifact class | Paths and limitation |
|---|---|
| Encoder, head and training-adversary checkpoints | Each result directory's `seed_*/pretraining/`, `adaptation/` or `training/*.pt`. Initialization/selected/final tensors are not downloadable from this GitHub publication. |
| Final calibrated erasers and original attacker preprocessing | `seed_*/*/erasers.npz`, `probes/**/preprocessing.npz`, and saved training-eraser maps. Exact final-release replay needs these original maps; refitting is a new reproduction. |
| Fitted audit/continued attackers | `probes/**/*.pt`, `model.npz`, `model.joblib`, and frozen-diagnostic `fitting/B/selected_target_*.pt`. Published scores alone do not permit independently replaying their predictions. |
| Release/test arrays, IDs and optimizer/batch schedules | `release_cache/*.npz`, `fresh_test.npz`, schedule arrays and optimizer checkpoints. Generator parameters, split seeds, array hashes and actual exposure counts remain in published JSON. |
| Application-screen model arrays | `results/redesign_20260907_application_screen_v1/learned_task_reference.npz` includes learned coefficients/scaling, row indices and per-row predictions. Aggregate development evidence is published; these arrays remain local. |
| Duplicate source archives | `source_snapshot/**` and Gaussian `changes.patch`. SHA256 inventories preserve identities; canonical source dependencies are published once. |
| Public datasets and local processed data | `data/UCI HAR Dataset/`, `data/diabetes/`, `data/diabetes_processed/`, and other inventoried datasets. No raw data, environments, credentials or unrelated untracked results are added. |

Paths in the manifest are repository-relative. The original machine's checkout
was `/Users/nathansamson/PCRL`; occasional absolute paths in unchanged logs are
provenance, not portable download locations. No external checkpoint archive or
public retrieval URL is available in this publication. If the original local
artifacts are supplied separately, restore them at those relative paths and
verify each SHA256 before claiming exact saved-checkpoint replay.

Official public-data sources and the four files actually used by the screen are
recorded in [its inventory](../results/redesign_20260907_application_screen_v1/local_data_inventory.json).
HAR is UCI HAR v1.0 in the official extracted layout. The existing
[Diabetes preprocessing script](../experiments/preprocess_diabetes.py) and
[dataset instructions](DATA.md) describe producing `data/diabetes_processed/`.
Match the recorded processed `train.npz` hash for exact replay; regeneration
under a different dependency version is not asserted to produce identical bytes.
The screen's documented legacy preprocessing limitations still apply. No data
download or application experiment was performed during publication.

## Review and report regeneration without checkpoints

Run `python scripts/verify_redesign_publication.py` to verify published artifact
hashes, executed-source identities and canonical evidence links without models
or data. Add `--include-local` only if all original omitted files are available.
The [compact per-seed view](../results/redesign_publication_20260907/compact_per_seed.json)
is a packaging-only projection of original metrics, without rescoring or selection.

Read the committed tables, per-seed JSON, per-target CSV, selections and
verification records directly. The reporting scripts read canonical per-seed
metrics without fitting models. To avoid rewriting historical artifacts, copy
only the relevant per-seed JSON into a new directory before invoking a reporter:

```sh
python - <<'PY'
from pathlib import Path
import shutil
source = Path('results/redesign_20260907_prediction_audit_v1')
out = Path('results/review_prediction_table')
out.mkdir(exist_ok=False)
for path in source.glob('seed_*/metrics.json'):
    dest = out / path.relative_to(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
PY
python scripts/summarize_prediction_audit.py --out results/review_prediction_table
```

Other stage reporters are linked from the review index. Reporter output is a new
derivation; it does not replace the historical analysis or claim to reproduce
the original model fitting. Packaging verifies imports and evidence integrity;
it does not rerun the research program.

## Regenerating omitted synthetic artifacts

The runners use fixed repository-relative prior-stage paths. Changing only
`--out` for a later stage does not redirect its input checkpoints. A clean
reproduction therefore needs a **separate clone**, preserving the published
evidence there before recreating the canonical paths. Do not run these commands
in the original evidence workspace. Use the publication commit supplied with
the review, not a moving branch, and install the observed dependency versions:

```sh
git clone https://github.com/Bostesa/PCRL.git PCRL-reproduction
cd PCRL-reproduction
git checkout <publication-commit-SHA>
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements-redesign.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

python - <<'PY'
from pathlib import Path
import shutil
names = ['gaussian', 'nonlinear_upstream', 'nonlinear_release',
         'frozen_adversary', 'prediction_audit']
archive = Path('results/published_redesign_evidence')
archive.mkdir(exist_ok=False)
for name in names:
    original = Path(f'results/redesign_20260907_{name}_v1')
    saved = archive / original.name
    shutil.move(str(original), saved)
    original.mkdir()
    shutil.copy2(saved / 'PROTOCOL.md', original / 'PROTOCOL.md')
PY

python experiments/run_redesign_conflict.py --out results/redesign_20260907_gaussian_v1 --seeds 0 1 2
python experiments/run_nonlinear_conflict.py --out results/redesign_20260907_nonlinear_upstream_v1 --prepare-only
python experiments/run_nonlinear_conflict.py --out results/redesign_20260907_nonlinear_upstream_v1 --seeds 0 1 2
python scripts/summarize_nonlinear_conflict.py --out results/redesign_20260907_nonlinear_upstream_v1
python experiments/run_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1 --prepare-only
python experiments/run_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1 --seeds 0 1 2
python scripts/summarize_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1
python experiments/run_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1 --prepare-only
python experiments/run_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1 --seeds 2 0 1
python scripts/summarize_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1
python experiments/run_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1 --prepare-only
python experiments/run_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1 --seeds 2 0 1
python scripts/summarize_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1
```

This is a dependency-complete **new fitting recipe**, not a checkpoint download
or a claim that the recipe was executed during publication. Seeds, objectives,
budgets and generators are unchanged. New invocation commits, timing records,
source inventories and checkpoint serialization may differ; compare predictive
scores and tensor hashes separately from file hashes. Original wall times were
2.31, 32.23, 76.98, 11.39 and 52.24 seconds for these stages on a one-thread Apple
M4 Pro CPU. Other hardware/library builds can vary.

The application screen is optional and separate. After obtaining the recorded
public data, run `python experiments/screen_pcrl_applications.py --out-dir
results/application_screen_reproduction` in a fresh directory. That reproduces
development checks; it does not start the proposed ACS task-family experiment.

## Validation provenance

Original test logs remain attached to their stages: 156 foundation tests, 31
upstream checks, 44 release checks, 19 frozen-diagnostic checks, and 15 final-stage
checks. These counts overlap across stages and must not be added into a claim of
distinct tests. Packaging checks and the final staged-file inventory are separate
in [VALIDATION.md](../results/redesign_publication_20260907/VALIDATION.md).
