# Corrected deployment and custody adapter

The standalone runtime schema is `CIS1-raw83-own-head-deployment-v1`. Runtime code is `compressed_income_safeguard/deployment.py`; it imports only standard Python, NumPy and Torch. Each private package contains `DEPLOYMENT.json`, `model_state.pt` and `geometry.json`. No private source path, training label, teacher score, identifier or per-person output occurs in the manifest.

The exact manifest fields are schema, campaign, service_id, seed, per_recipient_variants, feature_names, feature_schema_sha256, class_orders, model_sha256, source_checkpoint_sha256, provenance, contract, parameter_count and geometry. Geometry contains file (`geometry.json`), sha256, binding_sha256, checkpoint_sha256 and head_tensor_sha256. The geometry file is the complete unchanged admitted CIS1 per-seed binding. It binds the original head/basis identity separately from the current trained checkpoint. Only projected recipients must retain their admitted own head; the hybrid occupation head may change.

Raw83 inference uses whole-role batches of 1024, Torch float32 encoders, factorized Torch float64 `q = z @ B.T; released = q @ B`, and recomputed own-head centered logits, probabilities and fixed-order argmax. Both recipients are full for U0_FULL/RAW_FULL; income is projected and occupation full for RAW_HYBRID and all four continuation arms; both are projected for U0_PROJ/RAW_PROJ. FIT fold2 output is sliced after the full role forward. Nominal wire widths are 21/29/50; projected income has rank-one feature capacity.

`deployment_specs.prepare_live(index, output)` reads metadata only and requires exactly nine services by two seeds. Each instance needs arm_id, seed, checkpoint, checkpoint_sha256, geometry `{path, sha256}`, geometry_binding_sha256 and bank_wire `{path, sha256}`; role defaults to INNER_SELECTION. The optional bank_slice is `FIT_fold2`, which requires role OSF_DEFENSE_FIT. The index needs manifest `{path, sha256}`, and may supply python, science_lock, diagnostic_lock and dependencies. Root registers and launches all resulting units. Live checks compare runtime output exactly with both the canonical source forward and the pinned bank wire; run the actual CLI inside the same admitted worker process; and test forbidden inputs and wrong pins. Inputs and expected outputs remain private.

`prepare_reader_index` binds frozen LR and ATT-selected envelope representatives from reader metadata. `prepare_restore` generates all 18 numerical restore units using snapshot files only. The snapshot must include the complete code closure, standalone package files, whole-role feature-only input, expected wires, representative reader weights and expected reader scores. Runtime restore refuses any bytes outside the restored tree and requires snapshot code execution plus exact environment pins. Independent deployment verification is implemented separately by the independent agent and does not import this forward.

`prepare_custody` builds exact-file plan, snapshot, restore and off-device copy units. Every copy and uncached hash readback happens under the durable queue. `off_device_backup --spec SPEC` now also requires an admitted custody unit and never acquires a new independent slot. Public custody receipts expose safe relative aliases/hashes/sizes, never original paths. Off-device completion requires a different mounted filesystem and separate queued numerical replay; a same-device snapshot is not off-device completion.

ROOT must queue the private finite engineering config R5_DEPLOYMENT_ENGINEERING_V2 before claiming runtime validation. Its synthetic checks cover all nine literal wires over 2049 rows, changed occupation heads in hybrid continuation models, income compression and output parity, exact canonical batching and forbidden inputs. Metadata compilation has passed; numerical execution remains queued. Actual 18 packages, restore receipts and figures can only be completed after root freezes the nominees and reader banks.

The finite unit uses the existing `admission` queue stage with operation `engineering`. The unrun V1 config with the unsupported stage is preserved privately; V2 supersedes it.

## Post-nomination metadata helper

The new helper `.private/compressed_income_safeguard/r5/build_deployment_custody_specs.py` reads frozen JSON and code only; model/NPZ/reader hashes come from actual queued receipts. It preserves differing prior metadata, refuses missing nominees/banks, and never enqueues.

Run with the study Python executable:

```sh
python .private/compressed_income_safeguard/r5/build_deployment_custody_specs.py --phase live --nomination-lock results/pcrl_compressed_income_safeguard_v1/NOMINATION_LOCK.json --amendment-sha256 LATEST_QUEUE_AMENDMENT_SHA256
```

This generates `FINAL18_LIVE_INPUT_INDEX.json`, exactly 18 export specs in `GENERATED_LIVE.json`, and `FINAL18_RESTORE_READER_INDEX.json` with linear and ATT-frozen complete54-envelope representatives. Source-code pins are refreshed from current safe code; ROOT must bind any code differences through its dated amendment before admission.

After all 18 live receipts pass, `--phase custody-plan --execution-commit COMMIT --reporting-seal SEAL_JSON --amendment-sha256 SHA` generates an exact-file custody plan unit. The optional seal adds only its admitted INNER-label pins; these require ROOT's pending custody guard correction and the exact committed seal/environment binding. The custody plan includes code closure, admitted sources, every training checkpoint, all standalone packages, feature-only whole-role input, expected/CLI wires and representative frozen readers/scores. Public aliases normalize sealed labels without exposing source paths.

Then generate `--phase snapshot --target FRESH_VERSIONED_TARGET`; `--phase restore-copy --snapshot SNAPSHOT --target FRESH_RESTORE`; and `--phase numerical-restore --restore-root RESTORED_TREE --tag LOCAL`, each with the latest amendment SHA. The numerical restore phase produces all 18 queued package/reader replay specs. `--phase off-device --snapshot SNAPSHOT --target NEW_EXTERNAL_TARGET` prepares the separate filesystem copy; a later numerical restore is still required.

Custody guard correction is preserved as `.private/compressed_income_safeguard/r5/custody_worker_after_E.py` with SHA256 `56c17ae34c7b5465cdfcb9c7eb8c4de8908a194fa97db203f92536c842f35f4f`. ROOT restored active HEAD after the source guard refused admission, before any affected fit started. The current reader/trajectory queue continues under its existing pins. Apply the pending correction only through ROOT's later engineering amendment E. The four mocked-open checks prove byte-access guard behavior; they are not numerical trial evidence.
