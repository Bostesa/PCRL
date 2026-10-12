# Post-freeze reporting preparation

The globally frozen reporting stage fixes 26 service banks, the FIT label-only oracle and four independent reader refits before the reporting protected-label archive is opened or hashed in this campaign. R00 seals and publishes that freeze before admitting the reporting custody job. No encoder, head, reader, orientation or candidate selection changes at this stage.

`reporting_custody_worker.py` is a queue-authenticated opaque copy. It verifies all 28 global freeze receipts and the four actual refit model hashes against their aggregate, checks the published freeze, then verifies the owner-only source archive has exactly row_id/unit/sex fields and copies its bytes into a fresh owner-only target. It verifies the copied SHA and records no labels or private paths publicly. It does not deserialize arrays. The root queue may hash the registered source while admitting this job, which is authorized only after the same global freeze is sealed.

`prepare_reporting_units.py` reads the coordinator's private reporting preparation input and creates a custody unit, 22 continuous-service reports, four literal legacy score-wire reports and a paired-bootstrap inference unit. Preparation never opens or hashes the protected source or target archive. Target hashes are copied from the frozen custody metadata, and every dependent job waits for custody. Inference waits for all 26 reporting receipts.

Portable form of the executed metadata preparation:

```sh
python -m task_conditioned_representation.prepare_reporting_units --input .private/task_conditioned_representation/root/REPORTING_PREPARATION_INPUT_V1.json --attacks-amendment results/pcrl_task_conditioned_representations_v1/REPORTING_LOCK.json --python "$PCRL_PYTHON" --output .private/task_conditioned_representation/r5/PREPARED_REPORTING_INFERENCE_V1.json
```

The prepared index exposes exact private config/spec pins for R00's shared queue. Custody reserves/caps 120 seconds, each report 180 seconds and inference 3600 seconds; every job has a 2 GiB RSS cap and one numerical thread. All source entries contain exactly path and SHA256. This command creates metadata and executes no scientific job.

Inference retains all 51 primary, 240 utility and 33 secondary slots. Utility uses the original nominee probability keys from the complete 40-instance utility archive. The fixed actual banks retain original nominee eligibility and identity flags. H_RAW_CSTAR adds descriptive recovery intervals without changing the primary family. New output-only and literal historical code-wire results remain separate contracts. Per-seed and paired-seed-mean absolute intervals are descriptive; the primary conjunction uses its registered simultaneous intervals.
