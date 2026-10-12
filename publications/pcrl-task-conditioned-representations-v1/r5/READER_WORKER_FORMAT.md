# Private reader worker format for R00

Entry point will be `python -m task_conditioned_representation.attack_worker --spec PRIVATE_CONFIG` under the new authenticated queue, stage `attacks`.

Private configuration fields: `unit_id`, `arm_id`, `seed`, `manifest`, `manifest_sha256`, `checkpoint`, `checkpoint_sha256`, `science_lock`, `science_lock_sha256`, `attack_lock`, `attack_lock_sha256`, `private_output`, `public_output`. The release lock must have `status=RELEASES_AND_ENDPOINTS_FROZEN` and list exact per-arm seed checkpoint hashes. Source path manifest must exclude INNER SEX. The worker loads ATT features+SEX and INNER features only, reruns raw83 checkpoint inference, fits on ATT training people, tunes orientations on ATT validation, and saves immutable frozen reader scores for INNER before INNER SEX access.

Private freeze artifacts: `ATT_RELEASE.npz` (row_id,unit,r1,r2,sex,train,validation); `INNER_RELEASE.npz` (row_id,unit,r1,r2); `INNER_SCORES.npz` (view__reader scores, pair ignore aliases, shuffled and two planted-bit predictions); `MODEL_view__reader.joblib`; `FROZEN_COMPLETE.json`. Public receipt copies aggregates/contracts/hashes, without weights, per-person values, filesystem private paths, or private schemas containing identifiers.

Reporting entry point will require a separate R00 global `ALL_READERS_FROZEN` lock containing exact freeze receipt pins for every required release, endpoint registry/K pin, the owner-only INNER SEX archive pin, nomination seal pin, and arm/seed identities. No reporting label archive is read or hashed before this lock. Reporting outputs aggregate reader AUCs, envelope AUCs, controls, and hashes. Models/orientations remain fixed.

Inference entry point will read this report state and all frozen score archives, and use one shared 2048-by-group bootstrap index stream across every release and source seed. Every required missing endpoint retains its original ID and family slot.
