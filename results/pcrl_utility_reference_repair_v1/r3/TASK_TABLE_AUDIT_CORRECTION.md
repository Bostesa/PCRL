# R3 task-table audit correction — 2026-10-09

Before any FIT SEX access, the first run of R3's aggregate-only audit stopped on
an audit-code identity error. The audit compared `TASK_TABLE_RECEIPT.bank_sha256`,
which is the frozen bank's canonical content identity, with the SHA-256 of the
`MASK_BANK.json` file. Those are distinct identities. A second attempted check
looked for a `bank_sha256` field in the JSON file; that file does not contain
one. Neither attempt wrote an audit result or changed q12 scientific outputs.

The corrected audit separately verifies the `MASK_BANK.json` file SHA against
the q12 lock and requires the canonical bank content identity to agree across
the q12 receipt, task table and T selection. It then independently checks all
mask task flags, metrics, margins, head identities and T ranking. The corrected
run passed and wrote `TASK_TABLE_SELECTION_AUDIT.json` (SHA-256
`f462d3a731696ec507c1249d66e9915e500a28acaa55cf0ad6c37e2ab922ba65`).
The original audit source is preserved in commit `bdf7987c4`; the correction
and successful aggregate audit are in commit `625ff1726`. No person-level
role file or protected SEX label was opened by either audit attempt.
