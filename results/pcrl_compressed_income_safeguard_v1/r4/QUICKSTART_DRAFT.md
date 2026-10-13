# Runnable command draft and evidence status

This is a prospective ROOT integration draft. It does not claim pending jobs ran. Run status is determined by durable queue receipts, not by this command list.

Already run: source admission and independent source replay, finite engineering, both seed training jobs, independent stored-log training audit, and synthetic standalone deployment engineering. Fresh complete readers and42 FIT2 trajectory jobs are in ROOT's active queue. Not yet established here: frozen nominees,54 completed independent refit attempts, sealed INNER report/K240 inference, final independent replay, selected18 live deployment packages, backup-only numerical restores, off-device custody, and verified publication.

## Metadata preparation in the authorized private execution checkout

Commands below only prepare metadata. ROOT alone registers and launches their generated specs. Use the execution environment's verified Python and dependencies.

```sh
python .private/compressed_income_safeguard/r4/build_reporting_specs.py --phase trajectory
python .private/compressed_income_safeguard/r4/build_reporting_specs.py --phase nomination
python .private/compressed_income_safeguard/r4/build_reporting_specs.py --phase packages
python .private/compressed_income_safeguard/r4/build_reporting_specs.py --phase report --seal PRIVATE_REPORTING_SEAL_JSON
python .private/compressed_income_safeguard/r4/build_reporting_specs.py --phase inference --seal PRIVATE_REPORTING_SEAL_JSON
python .private/compressed_income_safeguard/r4/build_closeout_tables.py
```

Trajectory V2 specs already exist; repeated unchanged metadata preparation is idempotent and differing prior files are preserved. Nomination preparation requires actual32 frozen banks and34 relevant trajectories; package preparation requires ROOT's frozen nomination lock. Report preparation requires the committed reporting seal and complete final roster. Inference preparation requires the actual report input index. The final command refuses to write review tables before report/inference aggregates exist.

ROOT admits a generated unit and runs it through the existing new queue. These command patterns are available; substitute exact generated spec filenames and the admitted campaign queue environment:

```sh
python -m compressed_income_safeguard.scheduler enqueue GENERATED_UNIT_SPEC_JSON
python -m compressed_income_safeguard.host_queue --count 1
```

Never invoke trajectory, reader fitting, nomination, INNER reporting, inference, refits, source deserialization, deployment or restore workers outside the admitted queue. The active campaign uses one numerical thread per process and its recorded resource limits. Do not open a new queue or reuse the closed predecessor queue.

## Selected deployment and custody preparation

These preparations are not completed deployment evidence:

```sh
python .private/compressed_income_safeguard/r5/build_deployment_custody_specs.py --phase live --amendment-sha256 ADMITTED_AMENDMENT_SHA256
python .private/compressed_income_safeguard/r5/build_deployment_custody_specs.py --phase custody-plan --execution-commit VERIFIED_EXECUTION_COMMIT --amendment-sha256 ADMITTED_AMENDMENT_SHA256
python .private/compressed_income_safeguard/r5/build_deployment_custody_specs.py --phase snapshot --target FRESH_PRIVATE_SNAPSHOT_ROOT --amendment-sha256 ADMITTED_AMENDMENT_SHA256
python .private/compressed_income_safeguard/r5/build_deployment_custody_specs.py --phase restore-copy --snapshot VERIFIED_PRIVATE_SNAPSHOT_ROOT --target FRESH_PRIVATE_RESTORE_ROOT --amendment-sha256 ADMITTED_AMENDMENT_SHA256
python .private/compressed_income_safeguard/r5/build_deployment_custody_specs.py --phase numerical-restore --restore-root VERIFIED_PRIVATE_RESTORE_ROOT --tag LOCAL --amendment-sha256 ADMITTED_AMENDMENT_SHA256
```

ROOT registers each generated unit and checks its actual receipt. Off-device copying and restore need a verified mounted external filesystem and the separate R5 `--phase off-device` metadata plan. Record pending commands when the drive is absent.

## Public standalone runtime closure

The exact repository-native code closure for **standalone deployed inference** consists of these two safe files under the publication's `code` root:

```text
code/compressed_income_safeguard/__init__.py
code/compressed_income_safeguard/deployment.py
```

`deployment.py` imports only Python standard library, NumPy and Torch; its `RuntimeModel`, wire construction and package loader are self contained. Independent verification and trial replay require their separately published code closures, rather than just these two runtime files. The inherited verified deployment dependency manifest is `task_conditioned_representation/requirements-deployment.txt`; it pins NumPy2.4.2, Torch2.10.0, SciPy1.17.1, scikit-learn1.8.0, joblib1.5.3 and psutil7.2.2. Only NumPy and Torch are needed by standalone inference. Final package environment pins must match the verified live/restore receipts.

Each privately restored package directory contains exactly the required runtime assets:

```text
DEPLOYMENT.json
model_state.pt
geometry.json
```

Its manifest has no absolute source path, label or teacher dependency. Model weights and all per-person input/output files remain private. The authorized recipient supplies a finite feature-only NPZ containing `X` and `feature_names`, with exactly83 predictors in the package's declared order. Runtime output contains the literal income `r1` and occupation `r2` wires, widths21/29; the aligned pair is their ordered concatenation, width50. Both tasks use their own released representation and head. Projected income has rank-one feature capacity.

After authorized package restoration and verified manifest/input pins, the standalone invocation is:

```sh
PYTHONPATH=code python -m compressed_income_safeguard.deployment \
  --package PRIVATE_PACKAGE_DIRECTORY \
  --manifest-sha256 VERIFIED_DEPLOYMENT_JSON_SHA256 \
  --input AUTHORIZED_RAW83_INPUT_NPZ \
  --input-sha256 VERIFIED_INPUT_SHA256 \
  --output FRESH_PRIVATE_RELEASE_NPZ
```

This command is a documented future invocation until a corresponding live/restore receipt proves it ran. The controlled campaign validates the same CLI within its admitted deployment worker. It rejects reordered or forbidden predictor names, nonfinite data, extra input fields, wrong manifests/models, or incompatible geometry. It requires no training labels or original absolute private source location.
