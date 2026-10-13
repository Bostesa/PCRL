# Runtime commands

Run under the campaign's registered queue specification using the pinned compatible Python environment. The command accepts only a private feature-only raw83 inference archive.

```sh
python -m occupation_bottleneck.deployment \
  --package PRIVATE_PACKAGE \
  --manifest-sha256 MANIFEST_SHA256 \
  --input PRIVATE_RAW83_INPUT.npz \
  --input-sha256 INPUT_SHA256 \
  --output PRIVATE_RELEASE.npz
```

The output contains only `r1` and `r2`, the declared complete recipient wires. No source teacher service is required at runtime. Hashes and exact package locations are obtained from private deployment receipts; public package aliases are for provenance rather than a claim that private weights are published.

Prepare metadata without executing numerical work:

```sh
python -m occupation_bottleneck.deployment_specs live PRIVATE_LIVE_INDEX.json PRIVATE_PREPARED_LIVE.json
python -m occupation_bottleneck.deployment_specs custody PRIVATE_CUSTODY_CONFIG.json PRIVATE_PREPARED_CUSTODY.json
python -m occupation_bottleneck.deployment_specs restore PRIVATE_PREPARED_LIVE.json PRIVATE_RESTORE_ROOT PRIVATE_READER_INDEX.json PRIVATE_PREPARED_RESTORE.json
```

The coordinator registers each emitted specification with the new campaign queue. Restore specifications execute from the copied `files/code` tree, so original models and source loaders are unavailable to numerical replay. Frozen-reader records bind view, reader name, orientation, saved score field, model hash and expected-score archive hash. Live12/restored12 packages are required for the three source anchors and two projection contracts across both seeds; a triggered pilot adds8 live and8 restored packages.
