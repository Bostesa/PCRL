# Raw-input release deployment

The runnable adapter is `python -m conditional_release.deployment`. It consumes the pinned private role-safe source manifest, an NPZ containing exactly `X` and `feature_names` in the original 83-column order declared in [RAW83_SCHEMA.json](RAW83_SCHEMA.json), provider-only canonical person keys, a private 32-byte sampling root and a persistent cache. It never consumes task or SEX labels.

The permitted deployment kinds are `P0`, `V_J`, `V_S21`, `ALLOCATION`, `TEMPLATE` and `CONTROL`. Policy kinds select their registered joint, independent-recipient, sequential or deterministic sampler automatically. CONTROL accepts only the pinned narrow or full-bank receipt schemas. Every output is the original bank's complete recipient wire.

An actual command shape, with private locations supplied from the restore index, is:

```sh
PYTHON -m conditional_release.deployment \
  --source-manifest SOURCE_MANIFEST \
  --source-manifest-sha256 SOURCE_MANIFEST_SHA256 \
  --seed 0 --kind ALLOCATION \
  --policy-dir POLICY_DIRECTORY \
  --policy-receipt-sha256 POLICY_RECEIPT_SHA256 \
  --input RAW83_NPZ --input-sha256 RAW83_SHA256 \
  --person-keys PERSON_KEYS_JSON --person-keys-sha256 PERSON_KEYS_SHA256 \
  --sampling-root PRIVATE_32_BYTE_ROOT --cache PRIVATE_CACHE_SQLITE \
  --output-r1 PRIVATE_R1_JSONL --output-r2 PRIVATE_R2_JSONL
```

The same person key returns the same cached tuple after repeated calls or process restarts. A changed raw input under that key is refused. Providers must assign stable, unique canonical keys across duplicate records; integer and string keys with the same textual form are the same key. Keys, roots, source states, contexts and fallback flags are never transmitted. Distinct-person coin independence is the declared ideal PRF assumption; the deployed HMAC sampler uses a 53-bit uniform grid.

Executed commands, exact private pins and outputs are retained in each deployment unit's private `EXECUTED_COMMAND.json`. Public `RAW83_PRIMARY_DEPLOY_S0_V1.json` and `S1_V1.json` record successful CLI/API byte parity for P0, source V_J/S21, G_STRICT, C_STRICT and N_STRICT on 34 allowed fitting inputs per arm, including duplicate records. The additional `RAW83_WIDE_DEPLOY_S0_V1.json` and `S1_V1.json` receipts record the same successful checks for the full-bank W_STRICT package. `RAW83_UT_DEPLOY_S0_V1.json` and `S1_V1.json` cover its utility-only control; `RAW83_D_W_S0_V1.json` covers deterministic seed 0; `RAW83_L_W_S0_V1.json` and `S1_V1.json` cover the local controls. They also record cache/restart parity, finite-law sampler parity and refusal of reordered columns, extra columns, nonfinite values and changed raw inputs under an existing key. This is implementer verification; independent numerical replay has its own receipts.

A runnable diagnostic package is not a utility or privacy pass. Scientific eligibility and comparisons are in the campaign decision.

## Custody and restored replay

`conditional_release.custody` creates versioned private plans, snapshots/Git bundles, restored source manifests and separate-filesystem copies. It reads only the filtered admitted source pins and this campaign's private artifacts; it never inventories predecessor archives. Hashes are read uncached before copying and again afterward.

`conditional_release.restore_deployment_worker` replays the raw inputs and original private sampling roots from a restored snapshot, comparing recipient wires byte-for-byte against the wires stored in that same backup. It rejects artifact inputs outside the restore tree. Actual restore unit receipts, when run, state their coverage; the small synthetic custody fixture is not a real-data restore.
