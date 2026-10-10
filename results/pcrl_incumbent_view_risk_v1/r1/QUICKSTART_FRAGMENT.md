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

Snapshot/restore commands and numerical verification receipts will be
added after the prospective custody amendment and actual restore run.
