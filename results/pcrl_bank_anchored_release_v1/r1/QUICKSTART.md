# R1 release quickstart

Run from the bank-anchored research worktree with the project scientific
environment. [R1_EXECUTABLE_RELEASE_MANIFEST.json](R1_EXECUTABLE_RELEASE_MANIFEST.json)
binds B0 and J for both source seeds to the public tables, policy receipts,
32-person raw-input deployment receipts, and fixed draw hashes. The private
files stay under `r1/.private`; paths in this page are relative to the
campaign package.

## Commands executed

The four shared-queue units `R1_DEPLOY_S0_B0`, `R1_DEPLOY_S1_B0`,
`R1_DEPLOY_S0_J`, and `R1_DEPLOY_S1_J` ran
the project virtual environment's `python -m bank_anchored.r1_deploy`
on the admitted 32-row raw FIT sample for each recipient. Their public
receipts bind exact emitted NPZ hashes and report PASS. The worker generated
the command arguments from the private source and policy receipts; it did
not supply task or SEX labels to deployment.

The following custody commands were run, in order, from the worktree root:

```sh
python3 -m bank_anchored.r1_custody snapshot \
  --source results/pcrl_bank_anchored_release_v1/r1/.private/R1_CUSTODY_PLAN_B0_J_READER_V2.json \
  --target results/pcrl_bank_anchored_release_v1/r1/.private/custody/snapshot_20261010_v1
python3 -m bank_anchored.r1_custody restore \
  --source results/pcrl_bank_anchored_release_v1/r1/.private/custody/snapshot_20261010_v1 \
  --target results/pcrl_bank_anchored_release_v1/r1/.private/custody/restore_20261010_v1
python3 -m bank_anchored.r1_custody snapshot \
  --source results/pcrl_bank_anchored_release_v1/r1/.private/R1_CUSTODY_PLAN_B0_J_READER_V4.json \
  --target results/pcrl_bank_anchored_release_v1/r1/.private/custody/snapshot_20261010_v2
python3 -m bank_anchored.r1_custody restore \
  --source results/pcrl_bank_anchored_release_v1/r1/.private/custody/snapshot_20261010_v2 \
  --target results/pcrl_bank_anchored_release_v1/r1/.private/custody/restore_20261010_v2
```

Both commands are write-once and refuse existing targets. Use a new versioned
target for another copy. Their owner-only receipts and public hashes appear
in [RESTORE_INDEX_V2.json](RESTORE_INDEX_V2.json). The original V1 copy
and failed replay remain preserved. This is a same-device restore;
off-device custody remains pending because no writable external drive was
mounted.

## Recipient release contract

`bank_anchored.r1_deploy` accepts one NPZ with exactly `X` and
`feature_names`: a finite preprocessed float matrix with the pinned 83
permitted columns in order. It validates the schema hash, frozen U source,
W_TSTAR table, selected policy package and private draw hash. B0 needs no
draws. J requires the owner-only float64 `uniforms` NPZ with one frozen draw
per person. The interface models one fixed release per person.

The one-recipient output NPZ contains exactly `code` (`uint8[n,10]`, least
significant bit first), `prob` (`float64[n,2]` or `float64[n,6]`) and
`decision` (`int64[n]`). The probability and decision are determined by the
public W_TSTAR table entry indexed by the code. CLI stdout has constant run
metadata only. Scores, fine cells, coins, other-recipient codes, fallback
flags, and labels are absent from the recipient wire.

The data-free adapter tests ran with:

```sh
"$PROJECT_VENV/bin/python" -m pytest -q \
  bank_anchored/test_r1_deploy.py bank_anchored/test_r1_runtime.py \
  bank_anchored/test_r1_release.py bank_anchored/test_r1_deployment_replay.py \
  bank_anchored/test_r1_snapshot_manifest.py
```

Set `PROJECT_VENV` to the local PCRL virtual environment. Result: 17 passed.
The tests cover wrong or reordered schema, incompatible
source/seed identity, wrong or extra private draw fields, hidden predictor
fields and hidden emitted wire fields. The actual 32-person raw-input CLI
replays check the two recipient views and exact frozen bank decode.

## Restored numerical replay

The snapshot and restore contain both seed routers, public bank tables,
B0/J policy packages, sealed J sample draws, raw permitted samples, a pinned
runtime source archive and one completed B0 nonlinear probability reader.
[RESTORE_REPLAY_SOURCE_PLAN.json](RESTORE_REPLAY_SOURCE_PLAN.json) lists the
first restore's relative source labels and hashes. Its queued
`R1_RESTORE_B0_J_READER` attempt failed at the first restored CLI import:
the runtime source archive omitted a transitive `latent_repair_v1` module.
The failure and original snapshot remain preserved in
[R1_RESTORE_ATTEMPTS.json](R1_RESTORE_ATTEMPTS.json). The corrected runtime
archive adds only that tracked module family and passes an isolated import
and seven synthetic deployment tests. The separate V2 snapshot and restore
have 55/55 uncached hash checks. Their relative source labels and hashes are
in [RESTORE_REPLAY_SOURCE_PLAN_V2.json](RESTORE_REPLAY_SOURCE_PLAN_V2.json).
The successor shared-queue numerical replay awaits a verification amendment.

The successor numerical replay will execute eight restored one-recipient
CLI calls and one reader inference; its status must come from its queued
receipt.
The reader takes the emitted B0 seed-0 probabilities concatenated as
`[n,2] + [n,6]` and returns a finite probability per person. It never uses
SEX at inference.
