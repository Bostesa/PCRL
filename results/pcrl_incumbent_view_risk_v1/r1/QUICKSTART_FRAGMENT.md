# R1 raw release commands

The queue ran `incumbent_view.r1_replay` from the six V_S21 O specs and
the fourteen P0/V_L/V_J M specs. Each worker invoked the CLIs below on
the admitted 32-row FIT raw83 input. The source, policy, input and coin
paths in those private specs are owner-only; this public command form uses
variables for the same hash-pinned arguments.

For a new permitted 83-column input archive, set `SOURCE_MANIFEST` to
the private P0 source manifest, `INPUT` to the new input, `SEED` to one
of the fitted seeds, and `PYTHON` to the tested virtualenv interpreter.
The input column names and order must equal the frozen schema.

```sh
PYTHONPATH=. "$PYTHON" -m incumbent_view.r1_deploy \
  --source-manifest "$SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --seed "$SEED" --recipient 1 --input "$INPUT" --output "$P0_R1_OUT"
```

The second P0 recipient uses `--recipient 2` and its own output path.
For the nominated V_L channel or the V_S21 strongest control, set
`POLICY_DIR`, `POLICY_RECEIPT_SHA256`, `ARM_ID`, and a private uniform
archive matching the declared input rows and realization. V_L and V_S21
consume two independently derived recipient-domain uniforms per person.
Use the frozen CAMPAIGN_LOCK SHA as `POLICY_LOCK_SHA256`.

```sh
PYTHONPATH=. "$PYTHON" -m incumbent_view.r1_selected_deploy \
  --source-manifest "$SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --seed "$SEED" --policy-dir "$POLICY_DIR" \
  --policy-receipt-sha256 "$POLICY_RECEIPT_SHA256" \
  --policy-lock-sha256 "$POLICY_LOCK_SHA256" --arm-id "$ARM_ID" \
  --input "$INPUT" --input-sha256 "$INPUT_SHA256" \
  --uniforms "$UNIFORMS" --uniforms-sha256 "$UNIFORMS_SHA256" \
  --output-r1 "$R1_OUT" --output-r2 "$R2_OUT"
```

The CLI requires fresh output paths and writes complete recipient JSONL
wires containing original P0 code IDs, fixed decoded probabilities,
decisions and public metadata. It does not require the research queue
for ordinary inference. The one-fixed-release-per-person contract applies.
For deterministic P0, no coin file is accepted. All three research draws
of P0 are exact aliases. The owner-only O and M queue specs provide exact
executed argument values and hashes.

## Versioned restore and replay

The shared queue executed the Q snapshot and snapshot-only restore using
`results.pcrl_incumbent_view_risk_v1.r1.r1_custody`. The owner-only
`r1/.private/custody/EXECUTED_COMMANDS_V3.json` records exact argv, queue
spec hashes and receipt hashes for the failed P copy, successful Q copy and
restore, 18 R1 numerical replays, and independent R5 replay. These commands
were submitted to the one global queue; copying large artifacts or replaying
protected rows should use that same queue.

The successful restore command had this form, with variables bound to the
exact hashes in [RESTORE_INDEX_V3.json](RESTORE_INDEX_V3.json):

```sh
PYTHONPATH=. "$PYTHON" -m results.pcrl_incumbent_view_risk_v1.r1.r1_custody restore \
  --snapshot "$SNAPSHOT_V3_Q" \
  --snapshot-receipt-sha256 "$SNAPSHOT_RECEIPT_SHA256" \
  --plan-sha256 "$PLAN_SHA256" --target "$RESTORE_V3_Q"
```

The R1 restored-code checker used `incumbent_view.r1_restore_replay` from
`$RESTORE_V3_Q/files/code` with `PYTHONPATH` set to that directory. It matched
32/32 public wires for both recipients on each draw-zero P0, V_L, V_J and
V_S21 check across two seeds. Historical P0 readers and eight fresh R4 pair
HGB readers reproduced 32 predictions each with maximum absolute gap zero.
The separately authored R5 checker covered all three contract draws and
reproduced every public wire and eight frozen reader arrays exactly. Both
checks used restored files only and read no task or SEX labels.

A new permitted 83-column input can use the restored
`runtime/P0_SOURCE_PATHS.json` as `SOURCE_MANIFEST`, the restored public
banks, fitted policy package, and a private coin file matching the declared
sampling contract. P0 does not use a coin. The restored archive carries the
Git bundle at the captured tip; later final-report commits are separate.
Off-device custody is pending because the external drive was absent.
