# Bank-anchored raw-input deployment adapter

`python -m bank_anchored.r1_deploy` is the one-recipient provider adapter. It
reads the ordered 83 permitted predictors, computes both frozen U task routes
privately, samples the selected R2 policy, and emits only `code`, `prob`, and
`decision` for the requested recipient. The output is an owner-only NPZ with
ten little-endian code bits per row. Each probability and decision must equal
the entry in the frozen public W_TSTAR decoder table indexed by that code.

The recipient does not receive the private teacher scores, fine cells, other
recipient's code, coins, source labels, or a branch/fallback flag. CLI stdout
contains only constant run metadata: schema, arm, seed, purpose, row count,
policy receipt hash, public table hash, and emitted NPZ hash. Those constants
are part of the declared complete view, even though they have no variation
within a release unit.

The provider must supply the frozen Hcal `source_root`, QPC `qpc_root`, public
W_TSTAR table, ordered schema, selected R2 policy directory, and an NPZ of raw
predictors. The policy receipt, source binding, table, model, and ordered
schema are byte and identity checked before release. The selected B0 package
uses no sampling coin. J/T/S policies require one frozen private uniform per
person for the pair; P/L policies require two independent recipient uniforms.
The draw NPZ must have only a float64 `uniforms` member and be owner-only and
hash pinned. D is deterministic and supplies no draw file. This adapter
models one fixed release per person; repeated queries are a separate contract.

Example after restoring the dependency manifest and setting local paths:

```sh
python -m bank_anchored.r1_deploy \
  --purpose 0 --seed 0 \
  --source-root "$SOURCE_ROOT" --source-sha256 "$SOURCE_MODEL_SHA256" \
  --qpc-root "$QPC_ROOT" \
  --public-table "$PUBLIC_TABLE" --public-table-sha256 "$PUBLIC_TABLE_SHA256" \
  --schema "$ORDERED_SCHEMA" --schema-sha256 "$ORDERED_SCHEMA_SHA256" \
  --policy-dir "$SELECTED_POLICY_DIR" \
  --policy-receipt-sha256 "$POLICY_RECEIPT_SHA256" \
  --source-binding-sha256 "$SOURCE_BINDING_SHA256" \
  --input "$PERMITTED_INPUT_NPZ" --output "$PRIVATE_RELEASE_NPZ"
```

For stochastic arms, append `--draws "$PRIVATE_DRAWS_NPZ"` and
`--draws-sha256 "$PRIVATE_DRAWS_SHA256"`. The frozen HMAC realization-0 draws
for the admitted 32 FIT sample were derived inside the queued replay worker
and checked against R2's full-FIT cached release before this CLI ran. The
draw files remain private.

The final B0/J draw-bound dependency inventory is private at
`r1/.private/R1_DEPLOYMENT_SNAPSHOT_DEPENDENCIES_B0_J_DRAW_BOUND.json`.
The versioned same-device snapshot and restore use the reviewed private
`R1_CUSTODY_PLAN_B0_J_READER_V2.json`. Public hashes and relative labels are
in `r1/RESTORE_INDEX.json` and `r1/RESTORE_REPLAY_SOURCE_PLAN.json`.
Both B0 CLI units, `R1_DEPLOY_S0_B0` and `R1_DEPLOY_S1_B0`, passed actual
32-person raw-input, fixed-decoder, original-wire, and cached-code checks.

Both J units, `R1_DEPLOY_S0_J` and `R1_DEPLOY_S1_J`, passed the same actual
raw-input CLI and frozen-coin parity checks. J seed 0 happened to emit the
same 32 sample wires as B0; that sample equality does not alias their laws.
The original INNER utility endpoints passed independent numerical replay for
both seeds. Separate FIT proxy boundary disagreements remain unresolved. The
snapshot-only numerical CLI and nonlinear-reader replay is queued for a
separate verification amendment; no J recovery or competitive claim follows
from the deployment receipts alone.
