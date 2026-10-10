# Quickstart draft for R00

Status: `R19_VERIFIED_INTEGRATION_COMMANDS_ONLY`. Run these from `results/pcrl_latent_and_sampled_methods_v1/r19/` in the campaign worktree. They read public aggregate evidence and R03 receipt metadata; they do not fit a model or replay private person-level arrays.

```sh
python3 -m unittest -q test_queue_unit_status.py
python3 derive_d_k567_proposal.py
python3 build_primary_endpoint_proposal.py
python3 queue_unit_status.py
```

R19 actually ran the first three commands and read-only queue export. The three queue-helper tests passed. The D builder is idempotent and emits 665 rows with SHA-256 `4ec0adda1dc9aa83df64093e51cfdbcda842730820e32d1ed94da1820d68cd12`. The live queue output changes as R03 units finish; do not pin an old row count as closeout.

**R00 fill from exact run receipts:** command/argv and hashes for R01 role/source admission, D T/D-K/L/S12/S21/C/F fit and release, R02 fresh attacks and anchor readers, R04 independent all-arm and K567 replay, R18 one-row deployment and custody, and any admitted A fit/release/attack/interval/replay. Record each as run, skipped, failed or aliased. Do not replace actual queued commands with inferred examples from source files.

The public [D endpoint schema](../PRIMARY_ENDPOINTS_SCHEMA_PROPOSAL.md) and [R03 status helper](../QUEUE_UNIT_STATUS_HELPER.md) explain sign, eligibility and receipt fields. Private checkpoints, row-level predictions and labels are excluded from this quickstart draft.
