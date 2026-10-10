# R18 A real one-row deployment contract

Status: prefit interface draft. No A person row has been opened by R18.

## Parents and source identity

Use one selected `A_JOINT_STOCH` FIT package and one completed R12
`INNER_SELECTION` release realization for the same seed. R03 must show both
unit IDs done with rc0 and byte hashes for the selected `FIT_RECEIPT.json`,
`POLICY_RECEIPT.json`, `policy.npz`, `RELEASE_RECEIPT.json`,
`release_codes.npz`, and `cache.sqlite3`. The R18 unit must be separately
listed in a dated effective A-stage amendment and have one fresh private
output. The parent release's cache is an input, never an output of R18.

Bind the original campaign lock, current A amendment, R01 source audit,
repair role manifest, T* seed bank admission receipt, both T* bank files,
the selected role's `tokens_INNER_SELECTION.npz` and
`references_s{seed}.npz`, the source U checkpoint identity, public stream-root
manifest and the sealed private stream-root file. Require the exact R12
policy, cache and release implementation source SHAs from the A amendment.

## One-row operation

After the R03 slot and parent/source checks, read only the admitted
`INNER_SELECTION` fine-cell and reference-score exports. Verify their row and
unit arrays against the repair manifest and against each other. Select
predeclared row index 0. This gives the two U0 score vectors and fine cells;
no income, occupation or SEX label file is opened. Verify the selected
source-binding SHA and recipient-global bank/action/mechanism hashes against
the fitted policy receipt and the queued release receipt.

Copy the parent `cache.sqlite3` into a fresh R18 mode-0700 private directory
using an exclusive mode-0600 file, then rehash it against the parent release
receipt. Restore `JointPolicy` from the selected package and construct
`CachedJointRelease` on the copied cache with the exact precommitted private
root, release ID, seed, role and realization. Query both recipients for the
same provider unit, then query both again. Compare the first and repeated
seven-field wires byte-for-byte. Require each code ID to equal the saved
`release_codes.npz` value at row 0, and independently reconstruct probability
and decision from the sealed global bank. The wire must contain no fine token,
action index, q value, score vector, stream root or routing flag.

Challenge the same cache key with one changed score input that remains a
valid score vector. It must refuse without a new public release. Hash the
R18 cache copy after the challenge and retain it privately. Leave the R12
parent cache byte-identical to its precheck hash. A second mismatch in source,
policy, bank, role or private-root identity must also refuse before release.

## Evidence boundary

The sole R18 output is a write-once mode-0600 private JSON receipt containing
the one-row input needed for later snapshot restore, the two exact public
recipient wires, hashes of the copied cache and source packages, and check
booleans. The public R18 receipt records only the R03 receipt SHA, private
output SHA, aggregate pass/fail checks, source and package hashes, charged
CPU/wall/RSS, and the remaining custody state. No per-person values, IDs,
private paths or random-root bytes enter public files.

Freeze source hashes, parent IDs, output path, cap and argv only after the A
selected FIT and role release have their completed R03 receipts. Run the
negative schema/wire/cache fixture through R03 separately; its current queued
status is not a PASS.
