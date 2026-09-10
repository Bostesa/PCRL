# Reproduction and finite completion

Use the original PCRL local environment and data. Start from this commit on `ablations-facct-2026-07-24`; reviewed model commit is `8a3c181dcba44a570b6bb5a3daebd19c6ff59bc8`. No later fitted object is substituted. Python3.13 and the recorded Apple M4 Pro/CPU package environment are bound in LOCK.json. All numeric libraries and Torch use one thread; one inference worker. No new scientific fit or optimizer update is required or allowed.

## Local requirements

The exact original CSV, historical v2 cache used only for exposure reconciliation, original transfer maps/preprocessors, parent18 frozen checkpoints and historical selected utility/audit/anchor objects remain local. [FROZEN_OBJECTS.json](FROZEN_OBJECTS.json) records1,549 paths, hashes and metadata for486 candidate directories, all eighteen graphs, prior/exposed controls and context references.1,528 files match historical manifests/packed-export hashes or reviewed Git bytes; the other21 metadata JSON files exactly match nested historical selection records, checked in OBJECT_IDENTITY_VALIDATION.json. No required object is missing locally. Prior vectors are explicit original aggregate probabilities; fitted preprocessing arrays are omitted and represented by exact hashes/shapes.

[LOCAL_ARTIFACTS.json](LOCAL_ARTIFACTS.json) binds omitted new local artifacts, including household/person use manifests, empty sample manifests, logs, initial inventories/lock and local review receipts. The repository's `local/` and `*.local.*` exclusions prevent committing identifiers. Reconstruct exposure manifests from the exact CSV/cache using the independent verifier; never replace frozen models by refitting. The parent [LOCAL_ARTIFACTS.md](../redesign_20260909_acs_fixed_predictions_v1/LOCAL_ARTIFACTS.md) explains its omitted model storage. Compressed historical aggregate exports may be restored using the parent's hash-verifying packaging tool if absent, but exact inference additionally requires the original local binary objects. This assignment did not need a download or model restoration.

## Verified commands from repository root

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m scripts.inventory_acs_locked_objects
.venv/bin/python scripts/verify_acs_locked_independence.py
.venv/bin/python -m scripts.run_acs_locked_evaluation --phase verify
.venv/bin/python -m scripts.report_acs_locked_evaluation
.venv/bin/python -m pytest tests/test_acs_locked_evaluation.py tests/test_acs_locked_independence.py tests/test_acs_fixed_predictions.py tests/test_acs_transfer_data.py tests/test_acs_transfer_heads.py -q
```

Existing compatible inventories/completion artifacts are checked without changing timestamps. The independent exposure verifier reconstructs and byte-compares its outputs; incompatible completed artifacts raise an error. `freeze` is idempotent after LOCK.json exists and verifies its recorded identities. This is a **locked empty cohort**: infer/score/uncertainty intentionally raise `Scientific evaluation blocked: no defensibly unused locked households`. Do not remove that gate or use the misleading50,329-person transfer-only remainder.

A new explicit historical plumbing replay can be written to a new local filename:

```sh
.venv/bin/python -m scripts.verify_acs_locked_objects \
  --report results/redesign_20260910_acs_locked_evaluation_v1/local/HISTORICAL_REPLAY.recheck.json
```

The tool refuses an existing report path. This checks already available historical fixtures only. It does not run old prepare/train/evaluate wrappers, refit/pretrain/reselect objects, or reuse the parent's expired clock. Repeated full historical replay is not necessary unless an artifact or implementation change needs verification.

The existing independent comparison verifier writes an output file in its input directory. Reproduce its arithmetic against an isolated read-only input tree so it cannot overwrite historical evidence:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.verify_acs_fixed_predictions_comparisons import main
parent = Path('results/redesign_20260909_acs_fixed_predictions_v1').resolve()
with TemporaryDirectory(prefix='pcrl-comparison-replay-') as temporary:
    shadow = Path(temporary)
    for source in parent.iterdir():
        if source.name != 'COMPARISON_REPLAY.json':
            (shadow/source.name).symlink_to(source, target_is_directory=source.is_dir())
    main(shadow)
    print((shadow/'COMPARISON_REPLAY.json').read_text())
```

The original completed historical replay/comparison receipts are kept in this study; never write new verification into the parent's directory. No new scientific score/comparison/interval can be independently replayed because none was computed. All unavailable cells and pending counts are explicit in TABLE.md and COMPLETION.json. The observed-point tradeoff figure is unavailable because no observations exist.

## Locks and publication

START.json preserves the actual initial HEAD/time and main references. The exposure audit completed before protocol drafting; PROTOCOL.md states that chronology. Initial full inventory and first lock remain local; PUBLICATION_REDACTION.json binds their hashes and explains the later public inventory's removal of preprocessing arrays. LOCK.json records the actual final lock time and source/model/sample identities. No sample, candidate, class, weighting, comparison or scientific outcome changed between these records; no outcome was scored.

Aggregate CSV/JSON/Markdown and inference/verification code are the publishable artifacts. Fitted models, raw records, caches, person identifiers/predictions/losses and full process logs remain local. The final local publication receipt records the pushed SHA and authenticated commit-pinned byte verification for decision, data independence, table, protocol and these reproduction instructions. That receipt stays local because a committed file cannot contain its own future commit SHA. Main remains unchanged; no PR or force push is used.
