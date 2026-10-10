# Off-device custody pending

The external drive was absent during this campaign. No off-device copy or
restore was attempted. The write-once same-device V3 Q snapshot and its
snapshot-only restore passed uncached hash checks; the restored numerical
checks also passed. The private snapshot includes 4,774 planned artifacts and
a Git bundle of research tip `de3752cbe167c1cbe63cc7a829fafd077fff07f2`.
Later public closeout commits are not in that captured bundle.

When the drive is mounted under a separately admitted custody step, copy the
entire versioned Q snapshot without deleting the local original. Pin the copy
to the snapshot receipt SHA-256
`a635f9cd93c1dc1a5d04f35cc5cabf9d210499ae5c3df7b5b870e19e912149f1`,
read every copied file uncached against the V3 plan, verify the bundle and
restore representative raw83 and reader outputs from the off-device copy.
Record the drive copy and restore receipts before changing this status.

Exact local commands already executed, including the failed P attempt, are in
the owner-only `r1/.private/custody/EXECUTED_COMMANDS_V3.json`. Exact local
source paths and proposed future off-device command templates are in the
owner-only `r1/.private/custody/OFF_DEVICE_PENDING_V3.json`; its mount
destination is deliberately unset. No new data acquisition or cloud storage
is authorized by this pending note.
