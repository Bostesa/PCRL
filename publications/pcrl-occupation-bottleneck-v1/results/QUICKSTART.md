# Runnable occupation diagnosis artifacts

The campaign completed diagnosis and fresh-reader tests on saved models. The bounded repair pilot did not trigger; its training implementation is available but has no Adult repair result. Private people, source models and reader weights are not on GitHub.

## Label-free deployment from a restored copy

Use the existing environment: Python3.13.7, numpy2.4.2, torch2.10.0, scipy1.17.1, scikit-learn1.8.0, joblib1.5.3. Set `PYTHON` to that interpreter and `RESTORE` to the verified private RESTORED_V1 directory. No dependency upgrade is needed.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$RESTORE/files/code"
"$PYTHON" -m occupation_bottleneck.deployment \
  --package "$RESTORE/files/campaign/r6/packages/RAW_JOINT_S0_head_row_space" \
  --manifest-sha256 5d9d4905be2c09d928b025b6b829fc4b3bee6f2b2f5189f4c6593c8080a40595 \
  --input "$RESTORE/files/campaign/r6/inputs/RAW_JOINT_S0_head_row_space.npz" \
  --input-sha256 49fb9c70ace65443ec37fe64bd039ba166c2b8f159c504f7cc31ae665ad27cba \
  --output "$PRIVATE_OUTPUT/release.npz"
```

`PRIVATE_OUTPUT` must be an existing private directory; the output must not exist. Input NPZ contains only `X` and `feature_names`, with83permitted columns in exact pinned order. Wires contain16feature coordinates, centered logits, probabilities and decision for each recipient:21income/29occupation/50coalition dimensions. Projection effective ranks are1and5. Alloutputs are computed from actual transmitted features. Runtime rejects labels, IDs, additional/reordered columns, hidden flags, teacher outputs, nonfinite values and mismatched pins.

This CLI was run on all12packages. Backup-only replay then checked all12packages and72representative reader predictions, plus two separately implemented anchor forward/reader checks. See [MODEL_MANIFEST.json](MODEL_MANIFEST.json), [VALIDATION.md](VALIDATION.md), and [BACKUP_VERIFICATION.json](BACKUP_VERIFICATION.json).

## Commands actually run

Commands below are historical execution receipts, with paths expressed relative to the research worktree. The final queue is closed; these are not commands to reopen the experiment.

```sh
"$PYTHON" -m occupation_bottleneck.host_queue --count 1
"$PYTHON" -m occupation_bottleneck.host_queue --count 12
"$PYTHON" -m occupation_bottleneck_independent.replay --spec .private/occupation_bottleneck/root/configs/OB_INDEPENDENT_HEADLINES_R7.json
"$PYTHON" -m occupation_bottleneck_independent.release_replay_worker_r7 --spec .private/occupation_bottleneck/root/configs/OB_INDEPENDENT_R7_ALL_RELEASE_FORWARDS.json
"$PYTHON" -m occupation_bottleneck_independent.restore_worker --spec .private/occupation_bottleneck/root/configs/OB_INDEPENDENT_BACKUP_ONLY_REPLAY.json
```

Workers were invoked by queue specs, inheriting authenticated queue descriptors; direct execution of guarded scientific workers fails closed. Immutable private specs preserve the exact argv, source hashes, input roles, dependencies, thread/RSS limits and receipts. Independent replay uses independently written arithmetic, forward functions, SVD and resampling. Public code supports deployment and inspection; a scientific replay also requires the admitted private restore inventory.

## Pending off-device custody command

No writable different-filesystem backup volume was available. After a suitable volume is mounted, set `EXTERNAL_TARGET` to a new private directory on it and `SNAPSHOT` to the private SNAPSHOT_V1 directory. This is a bounded custody-only operation with the existing heavy-slot semaphore; it does not reopen the scientific campaign or fit models.

```sh
"$PYTHON" -m occupation_bottleneck.off_device_backup \
  --snapshot "$SNAPSHOT" \
  --receipt-sha256 2a26ba43092f078a238b2a2c36300adc6b23cdb3cde4febb8c10ac5e63735f28 \
  --target "$EXTERNAL_TARGET" \
  --queue-root "$CAMPAIGN_PRIVATE/compute" \
  --wall-seconds 600
```

The helper verifies every copied byte uncached, rejects the same filesystem and existing targets, checks shared process capacity and15GiBfree-disk floor, and leaves originals intact. This pending helper was syntax-reviewed; an off-device run has not occurred. Then set `RESTORE="$EXTERNAL_TARGET"` and run the label-free deployment command above. A separate representative frozen-reader restore receipt is still required before declaring off-device custody complete.

## New study only under a new identity

[NEXT_METHOD_EXECUTION_PROMPT.txt](NEXT_METHOD_EXECUTION_PROMPT.txt) supplies the concrete next occupation safeguard trial. It must be prospectively locked with its own resource budget. It is not a result of this campaign. All present results reuse Adult development people and provide no independent confirmation.
