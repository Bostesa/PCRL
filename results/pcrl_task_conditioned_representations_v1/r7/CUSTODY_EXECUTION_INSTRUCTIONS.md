# Custody sequence for the coordinator

These are prepared execution instructions. No backup or restore has run yet. Root owns every queue enqueue/run action. Commit the final scientific results, all current helper code and public receipts before preparing the immutable plan. The plan refuses uncommitted helper files and tracked changes.

Use the extended source manifest and the explicit legacy source supplement. The supplement copies only already admitted P0/S12 role laws, public priors, source metadata, and pinned policy receipt/control files. It does not walk predecessor private directories or follow additional pointers inside old source metadata. Same original parent files remain adjacent, including S12 `POLICY_RECEIPT.json` and `control.npz`.

```sh
python -m task_conditioned_representation.build_custody_units plan V1 \
  .private/task_conditioned_representation/r7/PREPARED_CUSTODY_PLAN_V1.json \
  --manifest .private/task_conditioned_representation/r1/SOURCE_PATHS_WITH_HISTORICAL_REFERENCE.json \
  --release-lock results/pcrl_task_conditioned_representations_v1/ATTACKS_LOCK.json \
  --reader-lock results/pcrl_task_conditioned_representations_v1/REPORTING_LOCK.json \
  --supplemental-metadata .private/task_conditioned_representation/r5/LEGACY_ADMITTED_METADATA.json
```

Enqueue the spec path in the returned preparation record and run it through `task_conditioned_representation.scheduler`. The plan will be `.private/task_conditioned_representation/r7/custody/V1/PLAN.private.json`. Preserve Git HEAD between successful plan capture and successful snapshot completion; commit their public receipts afterwards.

```sh
python -m task_conditioned_representation.build_custody_units snapshot V1 \
  .private/task_conditioned_representation/r7/PREPARED_CUSTODY_SNAPSHOT_V1.json \
  --plan .private/task_conditioned_representation/r7/custody/V1/PLAN.private.json
```

Enqueue and execute that returned spec. The default new snapshot directory is `r7/custody/V1/SNAPSHOT` under the same private campaign root. The worker copies every captured file, verifies uncached SHA256 readback, creates/verifies the private Git bundle, and writes its snapshot receipt. Neither the plan nor a snapshot overwrites an existing destination.

```sh
python -m task_conditioned_representation.build_custody_units restore V1 \
  .private/task_conditioned_representation/r7/PREPARED_CUSTODY_RESTORE_V1.json \
  --snapshot .private/task_conditioned_representation/r7/custody/V1/SNAPSHOT
```

Enqueue and execute that returned spec. The default restored tree is `r7/custody/V1/RESTORED`. All source and campaign bytes are copied from the snapshot alone, followed by uncached hash verification. `runtime/SOURCE_PATHS.json` and `runtime/SUPPLEMENTAL_SOURCES_0.json` refer to the copied files. Original predecessor source metadata is archived as provenance; primary packages are independently runnable state-only exports.

Then prepare the required all-22 numerical restore jobs using the backed-up live index:

```sh
python -m task_conditioned_representation.build_deployment_units restore \
  .private/task_conditioned_representation/r7/custody/V1/RESTORED/files/campaign/r7/LIVE_DEPLOYMENT_UNITS_V2.json \
  .private/task_conditioned_representation/r7/custody/V1/RESTORED \
  .private/task_conditioned_representation/r7/RESTORED_DEPLOYMENT_UNITS_V1.json \
  --tag LOCAL
```

Root enqueues every spec in that index. Each runs the copied runtime code in `RESTORED/files/code`, uses only copied raw83 inputs and state-only packages, compares actual full and score wires to copied live outputs, and replays recipient/pair LR_C1 plus the actual strongest frozen reader from each reported envelope. Pair ignore-other aliases dispatch to the appropriate copied recipient reader. This is prediction replay of fixed models, not refitting. Reader paths derive from backed-up `PREPARED_ALL_READERS_V1.json`; no original private directory is needed. The builder reads metadata only and uses recorded hashes rather than opening person arrays.

R6 separately executes its independent backup-only check of 12 declared packages/readers and four independent refits. Both the private refit weights/scores and all independent helper code are captured by the plan. Its scope remains independent of the R7 implementation.

The plan captures all tracked source, all Python files in both new namespaces, the runtime requirements, the conditional_release reader/metric/inference dependencies, and the conditional_release custody source used solely for adapter provenance. It copies only this campaign's private files plus explicit admitted source pins. The current custody subtree and nonterminal queue logs are excluded to prevent recursion and mutable captures. The executed environment is Python 3.13.7, NumPy 2.4.2, Torch 2.10.0, SciPy 1.17.1, scikit-learn 1.8.0, Joblib 1.5.3 and Psutil 7.2.2.

The public plan/snapshot/restore receipts contain relative aliases, byte counts, hashes, code versions and statuses. They omit original source paths, manifest bodies and supplemental metadata bodies. Root should apply its existing public scanner to the receipts before publication.

YOTUO is currently unavailable. Off-device custody is pending. If it becomes a writable separate mounted filesystem, prepare `build_custody_units off-device` with an explicit new versioned target, enqueue and execute that copy/readback, then restore from the off-device snapshot and run the corresponding numerical restore jobs with a distinct tag. Do not delete originals. A different folder on the source filesystem is refused as an off-device target.
