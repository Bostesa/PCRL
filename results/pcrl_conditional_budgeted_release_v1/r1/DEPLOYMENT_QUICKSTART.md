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

The same person key returns the same cached tuple after repeated calls or process restarts. A changed canonical float32 input under that key is refused. Providers must assign stable, unique canonical keys across duplicate records; integer and string keys with the same textual form are the same key. Keys, roots, source states, contexts and fallback flags are never transmitted. Distinct-person coin independence is the declared ideal PRF assumption; the deployed HMAC sampler uses a 53-bit uniform grid.

Executed commands, exact private pins and outputs are retained in each deployment unit's private `EXECUTED_COMMAND.json`. Public `RAW83_PRIMARY_DEPLOY_S0_V1.json` and `S1_V1.json` record successful CLI/API byte parity for P0, source V_J/S21, G_STRICT, C_STRICT and N_STRICT on 34 allowed fitting inputs per arm, including duplicate records. The additional `RAW83_WIDE_DEPLOY_S0_V1.json` and `S1_V1.json` receipts record the same successful checks for the full-bank W_STRICT package. `RAW83_UT_DEPLOY_S0_V1.json` and `S1_V1.json` cover its utility-only control; `RAW83_D_W_S0_V1.json` and `S1_V1.json` cover the deterministic controls; `RAW83_P_W_S0_V1.json` and `S1_V1.json` cover independent product controls; `RAW83_L_W_S0_V1.json` and `S1_V1.json` cover the local controls. Both sequential orders are covered by `RAW83_S12_W_S0_V1.json`/`S1_V1.json` and `RAW83_S21_W_S0_V1.json`/`S1_V1.json`. They also record cache/restart parity, finite-law sampler parity and refusal of reordered columns, extra columns, nonfinite values and changed raw inputs under an existing key. This is implementer verification; independent numerical replay has its own receipts.

A runnable diagnostic package is not a utility or privacy pass. Scientific eligibility and comparisons are in the campaign decision.

## Custody and restored replay

`conditional_release.custody` creates versioned private plans, snapshots/Git bundles, restored source manifests and separate-filesystem copies. It reads only the filtered admitted source pins and this campaign's private artifacts; it never inventories predecessor archives. Hashes are read uncached before copying and again afterward.

`conditional_release.restore_deployment_worker` replays the raw inputs and original private sampling roots from a restored snapshot, comparing recipient wires byte-for-byte against the wires stored in that same backup. It rejects artifact inputs outside the restore tree. Actual restore unit receipts, when run, state their coverage; the small synthetic custody fixture is not a real-data restore.

## Actual recovery checks and required layout supplement

The captured scientific commit is `75b2ac258b451e17029e6d4c7426f46c942f8217`. The actual private snapshot contains 8,738 files. Its separate-filesystem copy and both restored trees passed uncached byte verification. All eight implementer W_STRICT/S12_SELECTED wire checks (both paired seeds, both trees) and both independent raw and frozen-reader replay pairs passed. These are representative recovery checks; they do not numerically rerun every package. No scientific model or reader was refitted. `CUSTODY_NUMERICAL_SUMMARY.json` binds the actual receipts.

A first restored runtime failed because the versioned aliases separated a historical policy receipt from its package, while the pinned predecessor loader requires adjacent files. Correction T was pushed before affected reruns. The original failed unit and three unrun superseded units remain recorded. The scientific snapshot, original manifest, package bytes and release laws were preserved.

After `conditional_release.custody restore` and before deployment, execute the generic [restore_layout_companions.py](restore_layout_companions.py) with a custody queue lease. It creates four byte-identical `policy.npz` companions next to the historical V_J/V_S21 receipts, using only restored, hash-bound inputs. Its executed command shape was:

```sh
PYTHONPATH=SOURCE_ROOT PYTHON restore_layout_companions.py --config PRIVATE_LAYOUT_CONFIG
```

The private restore index and closeout supplement retain the two actual configuration files. For a new restore location, create a new metadata configuration with `unit_id`, `restore_root`, the actual `manifest_sha256` for `runtime/SOURCE_PATHS.json`, `amendment_sha256` from correction T, and a new `public_output`; register it through the durable custody queue. The helper refuses existing companion targets rather than overwriting them. Then use the restored manifest in the normal deployment command above.

The layout helper and metadata supplement helper are separately hash-bound closeout artifacts outside the frozen 165-file scientific export. The supplement contains helpers, indices, configurations and actual receipts only. It does not recapture person data. Numerical restore integrity scans and queue preflight hash admitted archives, including admitted label archives; inference does not deserialize or require those label arrays.

## Public engineering fixtures

From the published `source` directory, the complete campaign fixture command is:

```sh
python -m pytest -q conditional_release conditional_release_independent
```

The export contains the fixed synthetic fixture JSONs and exact package requirements. This command tests engineering laws and correctness; it does not reproduce private person-level fits. The final actual export is bound by `EXPORT_SOURCE_MANIFEST_V4.json`; `PUBLIC_EXPORT_FIXTURES_V2.json` records all 112 tests passed, with zero failures, errors or skips, from that 165-file tree. The earlier V1 source manifest and fixture receipt are preserved.

## Runtime label independence

`RUNTIME_LABEL_INDEPENDENCE_V1.json` records exact original wire parity for all 20 fitted packages through the unchanged CLI after every role/label/role-wire/reader/sample record was removed from its runtime manifest. An audit hook installed before deployment imports observed no denied-path access. The unit preflight contains no role-data or label archive pin. `RUNTIME_LABEL_GUARD_SCOPE.md` states the boundary: this verifies the current Python loaders and is not an operating system sandbox. Earlier comprehensive queue preflight label hashes remain documented.
