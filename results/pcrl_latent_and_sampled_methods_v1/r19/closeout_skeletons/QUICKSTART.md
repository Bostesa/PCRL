# Quickstart draft for R00

Status: `R19_VERIFIED_INTEGRATION_COMMANDS_ONLY`. Run these from `results/pcrl_latent_and_sampled_methods_v1/r19/` in the campaign worktree. They read public aggregate evidence and R03 receipt metadata; they do not fit a model or replay private person-level arrays.

```sh
python3 -m unittest -q test_queue_unit_status.py
python3 derive_d_k567_proposal.py
python3 build_primary_endpoint_proposal.py
python3 queue_unit_status.py
python3 a_220_status.py --attempt-output A_FIXED_ATTACK_PHYSICAL_ATTEMPTS_PROPOSAL.csv > A_220_UNIT_STATUS_PROPOSAL.csv
```

R19 actually ran these commands as integration checks. The D builder is idempotent and emits 665 rows with SHA-256 `4ec0adda1dc9aa83df64093e51cfdbcda842730820e32d1ed94da1820d68cd12`. The queue export and 220-row A plan join change as R03 units finish; refresh the files after any later units. The A helper distinguishes 14 queue-completed null FITs and their 116 untriggered children from two failed SEQ21 FITs and their 20 technically blocked children. It maps 12 corrected R1 physical attacks to their original logical cells and writes a separate 20-row physical attempt ledger retaining two failed originals. Eleven focused R19 status/alias tests passed.

**R00 fill from exact run receipts:** command/argv and hashes for R01 role/source admission, D T/D-K/L/S12/S21/C/F fit and release, R02 fresh attacks and anchor readers, R04 independent all-arm and K567 replay, R18 one-row deployment, local restore and numerical replay, all 16 A FITs, and every later admitted fixed A release/attack/interval/replay. Record each as run, skipped, failed, aliased, not triggered or incomplete. Do not replace actual queued commands with inferred examples from source files.

The public [D endpoint schema](../PRIMARY_ENDPOINTS_SCHEMA_PROPOSAL.md) and [R03 status helper](../QUEUE_UNIT_STATUS_HELPER.md) explain sign, eligibility and receipt fields. Private checkpoints, row-level predictions and labels are excluded from this quickstart draft.
