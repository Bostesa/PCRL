# R2 proposal-builder role-file read deviation — 2026-10-09

At approximately 12:50 UTC, an **unsealed, uncommitted** downstream unit
proposal builder hashed four distinct role-only export files while making a
future input map: `ATTACK_FIT_NEW:features`, `ATTACK_FIT_NEW:sex`,
`INNER_SELECTION:features`, and `INNER_SELECTION:sex`. Its loops made 43
transient hash passes in total (11 over each features file, 19 over ATT SEX,
and 2 over INNER SEX). Those reads preceded the separate sealed release,
ATT fit, and privacy-score stages. This violated the role-access rule even
though the files were already present in the role-only export.

The builder read bytes solely to compute SHA256, compared each to the sealed
manifest, and wrote a proposed path/hash/dependency map. It did **not** open
the NPZ members, materialize arrays, inspect label values, fit any reader,
score INNER labels, or use the values for scientific selection. No downstream
real-data unit had been registered or run at correction time. The sealed q13
FIT SEX pair search is a separate stage and does not import this builder.

The unsealed original proposal is retained privately as
`.private/access_deviations/UNSEALED_PROPOSAL_WITH_ROLE_HASH_READS.json`, SHA256
`d1f3b7e6d45e574a65120ce7ecfa343aba17283bb2a954b687d7e602d968b046`.
It must not be bound into a stage lock or treated as scientific evidence.

The corrected builder takes each role path and SHA directly from the sealed
`DATA_ACCESS_MANIFEST.json`, validates path confinement and SHA syntax, and
does not open any role file. A data-free test using a nonexistent role file
failed under the original code and passes under the correction. The corrected
proposal SHA256 is
`ffac532f3c42eb81de7987a0a8a1afcf04ce1527fb2eb6be94aa0b02dcaf3012`.
R0 and R3 were notified before the proposal could be used for a lock.

The shared queue also hashes declared input files before a worker starts. A
future independently reviewed and sealed privacy-score stage may authorize
that hash-only role verification after F and ATT custody are frozen. Such a
read is queue admission, before worker-side F checks, and must be described
that way. It does not excuse this builder's reads: this builder had no stage
seal or role permission when it opened the four files.
