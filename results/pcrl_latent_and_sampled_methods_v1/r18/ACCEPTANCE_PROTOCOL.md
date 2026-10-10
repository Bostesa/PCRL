# R18 deployment acceptance, staged version 3

The source prompt requires a **run** on one real admitted input for each
representative. The D-K seed-0 one-row worker completed under the R03 slot;
its hash-only public receipt is `D_REAL_DEPLOY_ACCEPTANCE_RECEIPT.json`
(SHA-256 `6675a2ce2dd174e41987b04cae3bcb1390989bd8891cb305cfc1c15d5ee24c6d`).
The A representative and the separate data-free negative fixture still need
their own queued receipts.

## Source and one-row custody

- Use the repaired role-only `INNER_SELECTION:features` export, or a separately
  sealed permitted role. Verify the public role manifest SHA and the export's
  byte SHA before opening it. Inspect its ZIP member list and reject anything
  beyond `X`, `feature_names`, `row_id`, and `unit`. Read no labels or SEX.
- Require exactly one finite `float32` row in the ordered 83-predictor schema.
  The newline-joined feature-name SHA is
  `7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975`.
  Reject reordered/missing/extra columns and the excluded source fields
  `sex`, `race`, `income`, `occupation`, and `fnlwgt`.
- Keep the row, row ID, group ID, provider cache key, and all paths in private
  output. Public receipts contain unit ID, hashes, status, timing and aggregate
  pass/fail checks only. Reject any assessment-role input.

## D hard-code acceptance

R07's service API is `load_service(manifest_path, expected_manifest_sha256,
expected_source_sha256)` followed by
`HardCodeRelease.recipient_from_raw(purpose, raw, columns, encoder)`. The
frozen source checkpoint must hash to the sealed
purpose/seed pin before load. The **postfit** private service manifest and
public fit receipt must bind the effective D seal and completed R03 fit unit;
model, head and CAL hashes are verified from that postfit chain, since those
hashes cannot exist in the prefit lock. Run both purposes against the same admitted raw
row through their own frozen encoders. Require ten binary `uint8` bits per
recipient, at most 1,024 states, and exactly `code`, `probabilities`,
`decision`, and fixed `metadata` on the wire. Recompute probability and
decision from a separately loaded, hash-pinned code-only head, the inherited
H-GLOBAL-TEMP calibration record, and its documented float64 transform;
compare exactly. Require public `source_sha256`, `model_sha256`,
`head_sha256`, `calibrator_sha256`, `calibration_family`, and bit order to
equal the sealed service manifest.
Repeat the deterministic service call and compare every public byte. Mutate
column order, add an excluded field, change checkpoint SHA, and attempt a
latent/teacher/soft-bit export; each must refuse before a release is recorded.
No teacher scores or source latent features are transmitted.

## A sampled-channel acceptance

R12's provisional service takes both frozen score vectors at a central
provider, samples one joint action, and caches that joint realization under
an opaque provider key `(release_id, source_seed, role, realization, unit)`
plus a source-input digest binding both score vectors and fine cells. Both
recipient outputs must have exactly `schema`, `recipient`, `code_id`,
`probability`, `decision`, `release_id`, and `public_metadata`, with schema
`SCV1-score-release-v1` and globally constant metadata. Check each ID against
the **global** bank for that seed and recipient, with no
input-local action alias. Repeat the same request and require byte-identical
outputs for both recipients. A mismatched input digest or incompatible bank,
source, or policy pin must refuse rather than return a stale release. Reject
fine tokens, action indices, gate probabilities, random coins/seeds, clean
teacher scores, unsampled expectations and hidden routing flags on the wire.
Save the sampled realization privately and replay it from that cache; never
substitute expected probabilities for the actual sampled output.
The A real worker must use the dated effective A amendment and fitted
policy/bank/cache manifests as its trusted pins. It cannot infer them from
the service response being checked.

## Queue and evidence boundary

The data-free fixtures in `test_adapter_acceptance.py` and
`test_role_input.py` must be registered under R03 before running. They test
the contract but do not establish a real deployment. The later real units are
`R18-D-DEPLOY-REAL-S0` and `R18-A-DEPLOY-REAL-S0`, with one heavy slot each,
exact `argv`/`cwd`/source and manifest SHAs, private output, CPU/wall/RSS
ceilings and a fresh R03 receipt. The D worker at `r18/run_real_d_acceptance.py`
is bound to the completed seed-0 D-K fit receipt and hash-pinned service,
source checkpoint, effective lock, and canonical frozen U implementation.
R10 statically reviewed its direct-script preaccess path; R00 admitted it in
dated A4 amendment SHA-256
`9bb03642f7f42d44f26bf611ed2aedb53bea6ab438456757c5c0623250560b3a`.
R03 completed it with receipt SHA-256
`98f053d640bdd5cdc69b52caf342421aea974c2cfd12d0e6c5f17b0e6fcf137b`,
rc0 and one private result SHA-256
`7bcc5dd95de2013c0e6a917ca4d26639e2df24d651857dbe8674a64a19db312b`.
A will bind R12's fitted API and policy/bank/cache manifests when they exist. If A is not fitted
or no sampled bank exists, mark its real unit untriggered under the sealed
condition; do not describe a synthetic fixture as a deployed result.

R07's completed seed-0 D-K package is hash-bound to its public fit receipt.
The R18 real worker reconstructed probabilities from separately opened head
tensors and CAL records, without calling the service's code-head method. The
real positive check passed; the synthetic negative fixture remains R03 queued
and held. Its wrong/reordered/excluded schema, incompatible checkpoint and
hidden-wire refusal cases have no numerical PASS receipt yet.
