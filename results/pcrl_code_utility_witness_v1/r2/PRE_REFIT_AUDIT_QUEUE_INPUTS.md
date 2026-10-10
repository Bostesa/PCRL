# R2 queued pre-refit audit inputs

The worker is `r2/loss_audit_worker.py`, invoked through the single R0 queue as
`<venv-python> -m results.pcrl_code_utility_witness_v1.r2.loss_audit_worker
--config <owner-only config JSON>`. Direct launch rejects before reading the
config. Use stage `witness`, `cwd` equal to this campaign worktree, and set
one numerical thread per the queue. R0 sets the unit's wall, CPU and memory
caps. The config, queue spec and output private directory stay owner-only.

The R0 seal must have `role_manifest_sha256` equal to the admitted repair
role manifest's byte SHA, and `source_files_sha256` must map
`r2/loss_audit_worker.py` and `r2/loss_metrics.py` to their exact byte SHAs.
The worker checks these, its active queue slot and each input's queue source
record before opening a role archive or saved release.

## Owner-only config schema

Use absolute local paths in the private config; never publish it. The public
role manifest is the repair package's `m1/DATA_ACCESS_MANIFEST.json`. Its
admitted SHA for these old releases is
`605bcdc9111f8d228a8526a649aa5cdec34c3f932d798d65e0cdf9672d86c45c`.
All `{path,sha256}` pairs are byte hashes of the named existing files. Hash
them by streaming file bytes; do not load NPZ members to build the spec.

```json
{
  "schema": "pcrl-r2-pre-refit-loss-config-v1",
  "unit_id": "R2_PRE_REFIT_LOSS",
  "seal": {"path": "<effective-seal>", "sha256": "<seal-byte-sha>"},
  "role_manifest": {"path": "<repair>/m1/DATA_ACCESS_MANIFEST.json", "sha256": "605bcdc9111f8d228a8526a649aa5cdec34c3f932d798d65e0cdf9672d86c45c"},
  "seeds": {
    "0": {
      "first_receipt": {"path": "<current>/D_FIRST_S0_R1_RECEIPT.json", "sha256": "<sha>"},
      "followup_receipt": {"path": "<current>/D_FOLLOWUP_CONTROLS_S0_RECEIPT.json", "sha256": "<sha>"},
      "first_private_root": "<old>/.private/d_first_s0",
      "followup_private_root": "<old>/.private/d_followup_s0",
      "public_tables": {"path": "<current>/r1/W_BANK_PUBLIC_TABLES_S0.json", "sha256": "<sha>"},
      "banks": {
        "W_Q": {"release": {"path": "<new-private-W_Q-INNER-release>", "sha256": "<sha>"}},
        "W_TSTAR": {"release": {"path": "<new-private-W_TSTAR-INNER-release>", "sha256": "<sha>"}}
      }
    },
    "1": {
      "first_receipt": {"path": "<current>/D_FIRST_S1_RECEIPT.json", "sha256": "<sha>"},
      "followup_receipt": {"path": "<current>/D_FOLLOWUP_CONTROLS_S1_RECEIPT.json", "sha256": "<sha>"},
      "first_private_root": "<old>/.private/d_first_s1",
      "followup_private_root": "<old>/.private/d_followup_s1",
      "public_tables": {"path": "<current>/r1/W_BANK_PUBLIC_TABLES_S1.json", "sha256": "<sha>"},
      "banks": {
        "W_Q": {"release": {"path": "<new-private-W_Q-INNER-release>", "sha256": "<sha>"}},
        "W_TSTAR": {"release": {"path": "<new-private-W_TSTAR-INNER-release>", "sha256": "<sha>"}}
      }
    }
  },
  "output_private_dir": "<this-r2>/.private/pre_refit_loss",
  "output_csv": "<this-r2>/LOSS_DECOMPOSITION.csv",
  "output_receipt": "<this-r2>/LOSS_AUDIT_RECEIPT.json"
}
```

`banks` and `public_tables` may both be omitted for an old F/T/D-K-only
audit. Include both only after R1 has produced pinned INNER releases. Each
bank release has the eight hard-code wire arrays. The worker independently
decodes the ten bits against the public table and requires exact equality to
saved probabilities and decisions.

## Exact queue source inventory

The queue spec's `sources` list must contain the private config, effective
seal, repair role manifest, the two R2 source files, and the following for
**each** seed:

- Four admitted `INNER_SELECTION` role archives: `income.npz`,
  `occupation.npz`, `references_s0.npz`, `references_s1.npz`. Their exact paths
  and SHAs come from the role manifest's `role_file_relative_path` and
  `role_file_sha256`; the worker checks them against both that manifest and
  queue registration.
- Public old first and followup receipts named above. The followup receipt
  must pin the first receipt SHA.
- For `T` and `D-K` under the old first private root, and `F` under the old
  followup private root: `/<arm>/manifest.json`, `/<arm>/weights.npz`, and
  `/release__<arm>__INNER_SELECTION.npz`. Get expected SHAs from the public
  receipt's `packages[arm]` and `roles.INNER_SELECTION[arm]`; all actual bytes
  must match before arrays are opened. Source and role identities must match.
- If evaluating current-role banks, one public `W_BANK_PUBLIC_TABLES_S<seed>.json`
  and the `W_Q` and `W_TSTAR` private INNER release NPZs. Use R1 release
  receipt SHAs. Both bank arms are scored against the same admitted row order.

The queue spec itself requires `unit_id`, `stage`, `argv`, `cwd`, `max_wall_s`,
`cpu_reserve_s` (at least wall seconds), `max_rss_gib` in `(0,8]`,
`projected_output_bytes`, and the `sources` array. The private config file
mode and its parent directory must be owner-only. Create `<this-r2>/.private`
mode `0700` before the worker runs; the worker creates only the new final
`pre_refit_loss` directory. Use a fresh unit ID/output name if rerunning
under a documented amendment. Do not overwrite a prior public result.

Outputs are public aggregate `LOSS_DECOMPOSITION.csv` and
`LOSS_AUDIT_RECEIPT.json`, plus an owner-only SHA/row-identity receipt.
They contain no person-level rows. Differences are second minus first on
identical INNER people. Distillation KL uses the **uncalibrated** head and
Ucal teacher; public NLL/Brier use the calibrated transmitted scores.
