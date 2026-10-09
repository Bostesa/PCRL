# ATT feature loader correction — 2026-10-09

The first corrected q16 RAW_ONLY unit failed before producing an output. The
sealed privacy stage allowed `ATTACK_FIT_NEW:features`, but
`PrivacyRoleStore.read` delegated that field to the inherited task-role
reader, whose allowlist intentionally excludes `ATTACK_FIT_NEW`. This was a
loader integration error. The unit did not fit a reader, score a release or
inspect an INNER_SELECTION SEX label.

`PrivacyRoleStore` now handles only the already stage-allowed
`ATTACK_FIT_NEW:features` field in its own subclass. It verifies the sealed
file SHA-256, exact ZIP members, ordered people and groups, 83 finite float32
predictors, feature-name schema SHA-256 and forbidden-column rule before
returning a copy. The base task-role allowlist is unchanged. Synthetic tests
first reproduced the denial, then verified the admitted field in release,
RAW_ONLY and ATT stages and denial of ATT task/reference labels and INNER SEX.

The failed q16 attempt is preserved under
`.private/lock_attempts/q16_att_feature_loader/queue_g16/QUEUE.json`,
with its original lock, seal, review and registry in the archive root. Its charge is
`0.2250022920779884` CPU seconds. The corrected q16 queue carries forward
the q15 scientific queue charge and records this failed attempt as a separate
campaign debit; the global 20 CPU-hour accounting must include both. No
output from the failed unit may be reused or aliased. R0 must seal a new
governed source hash and obtain independent prefit review before retrying.
