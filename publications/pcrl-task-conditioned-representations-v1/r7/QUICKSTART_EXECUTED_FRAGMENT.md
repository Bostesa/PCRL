# Executed deployment and package contract

All 22 frozen arm/seed packages were deployed on 2,235 admitted INNER feature rows. Each package exactly matched its selected saved encoder/head checkpoint, and the standalone CLI exactly matched those actual full wires. Inputs contained no labels, SEX, row identifiers or group identifiers. The receipt table is `LIVE_DEPLOYMENT_ALL22_VERIFIED_V1.json`; private package paths and hashes are in `MODEL_RUNTIME_INDEX_V1.json` under the campaign private `r7` directory.

The command below was executed for each package by its registered deployment worker. `PACKAGE`, `MANIFEST_SHA`, `INPUT`, `INPUT_SHA` and `OUTPUT` stand for the exact private paths and bound hashes recorded in that worker's config and receipt. Outputs require a new filename.

```sh
python -m task_conditioned_representation.deployment \
  --package "$PACKAGE" \
  --manifest-sha256 "$MANIFEST_SHA" \
  --input "$INPUT" \
  --input-sha256 "$INPUT_SHA" \
  --output "$OUTPUT" \
  --contract full
```

The input NPZ must contain exactly `X` (a finite floating matrix with 83 columns) and `feature_names` (the exact ordered names in the pinned package manifest). The package contains `DEPLOYMENT.json` and `model_state.pt`; it does not require an original source model, source estimator, source home directory, labels, IDs, optimizer state or training arrays. Runtime inference imports only NumPy and Torch.

Full output NPZ fields are `recipient1` and `recipient2`, with 21 and 29 scalars per row. Each row contains the actual 16-dimensional representation, centered own-head logits, own-head probabilities and own-head decision. The aligned coalition concatenation has 50 scalars. `--contract output_only` returns 3 and 7 scalars (own probabilities and decision), for 10 aligned coalition scalars. The 22 live workers tested the full CLI contract; output-only wires were also computed from the same deployed models and stored for backup replay.

Wrong, reordered, missing or extra columns, nonfinite values, schema/model mismatches, and additional SEX/task/flag/source-output inputs are refused. Six packages are recorded exact source aliases; the two historical H_RAW_CSTAR packages remain descriptive, point-ineligible references. Deployment parity does not change their scientific status.

The executed environment pins were NumPy 2.4.2, Torch 2.10.0, SciPy 1.17.1, scikit-learn 1.8.0 and Joblib 1.5.3. The adjacent runtime requirements file records these versions. A clean environment install was not part of the live replay; package numerical inference and all 22 standalone CLI commands were actually executed in the admitted environment.

Versioned backup and numerical restore receipts will establish the separate custody stage. Off-device custody remains pending while the external drive is unavailable.
