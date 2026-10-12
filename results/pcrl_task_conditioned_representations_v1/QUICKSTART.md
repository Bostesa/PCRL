# Runnable packages and executed commands

The clean publication has aggregate evidence and source under `code/`. Execute from that code directory or set `PYTHONPATH` to it. Models, source inputs, release arrays and reader weights remain private. Do not run an all-role loader. Private operational indices provide exact authorized paths and SHA256 pins; public hashes identify the same artifacts.

The environment used Python 3.13.7, NumPy 2.4.2, Torch 2.10.0, SciPy 1.17.1, scikit-learn 1.8.0 and Joblib 1.5.3. `task_conditioned_representation/requirements-deployment.txt` pins deployment dependencies. A clean install was not tested. One numerical thread was used throughout.

## Deployment actually executed

All 22 packages ran on 2,235 admitted raw-feature rows and matched their actual selected checkpoint wires. The independent replay used separately written encoder/head arithmetic. `r7/MODEL_RUNTIME_INDEX_V1.json` is a private operational index, not a public download. Each package has `DEPLOYMENT.json` and `model_state.pt`. It needs only NumPy/Torch and does not need the old model, estimators, labels or private source path.

The following command ran for every package inside its queued deployment worker. Variables denote exact private values in that worker's immutable config. Use a new output filename.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m task_conditioned_representation.deployment \
  --package "$PACKAGE" --manifest-sha256 "$MANIFEST_SHA" \
  --input "$INPUT" --input-sha256 "$INPUT_SHA" \
  --output "$OUTPUT" --contract full
```

Input NPZ fields must be exactly `X` (finite floating matrix, 83 columns) and `feature_names` (exact pinned order). Wrong/reordered/extra columns, SEX, true task labels, IDs, flags, nonfinite inputs or mismatched model/schema are refused. Output fields `recipient1`/`recipient2` contain actual z16, centered own logits, own probabilities and first-argmax decision: 21/29 scalars, aligned pair50. `--contract output_only` gives probabilities/decision only, 3/7 scalars, pair10. This narrower view has separate evidence.

## Fitting and replay actually executed

All fitting and numerical review used one durable queue. Exact unit argv/config/code/source/output pins are retained privately; public UNIT_MANIFEST contains counts, CPU and statuses. The coordinator repeatedly executed:

```sh
python -m task_conditioned_representation.host_queue --count 1
```

Longer bounded `--count` waves used the same queue and machine semaphore. Workers acquire their authenticated slot before source access. Direct launches of source/fitting/evaluation workers are refused. The now-closed study is not reopened by a deployment command; deployment inference remains runnable without a training queue.

Executed workers include `train_worker` (including fixed calibration modes), `utility_worker`, `attack_worker`, `legacy_attack_worker`, `attack_worker` reporting mode, `inference_worker`, standalone deployment, custody, and separate `task_conditioned_representation_independent` replay workers. Private configs bind their actual commands. No claim is made that public aggregates alone can reproduce private training.

## Restore

RESTORE_INDEX and BACKUP_VERIFICATION identify the versioned private snapshot and numerical restore receipts. The snapshot includes admitted role-only inputs, original useful model/heads/calibration, historical reference, P0/S12 policies/laws, all new models/readers, source code and environment pins. Snapshot-only deployment and reader replay use remapped private paths. Original files and execution ancestry are preserved. Off-device custody is pending if no writable external drive is mounted.
