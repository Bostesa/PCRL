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
python3 -m bank_anchored.r1_custody snapshot \
  --source results/pcrl_bank_anchored_release_v1/r1/.private/R1_CUSTODY_PLAN_B0_J_READER_V5.json \
  --target results/pcrl_bank_anchored_release_v1/r1/.private/custody/snapshot_20261010_v3
python3 -m bank_anchored.r1_custody restore \
  --source results/pcrl_bank_anchored_release_v1/r1/.private/custody/snapshot_20261010_v3 \
  --target results/pcrl_bank_anchored_release_v1/r1/.private/custody/restore_20261010_v3
```

The commands are write-once and refuse existing targets. Use a new versioned
target for another copy. Their owner-only receipts and public hashes appear
in [RESTORE_INDEX_V3.json](RESTORE_INDEX_V3.json). The V1 and V2 copies
and failed numerical replay receipts remain preserved. This is a same-device restore;
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

The V3 snapshot contains both seed routers, public bank tables, B0/J policy
packages, sealed J sample draws, raw permitted samples, a pinned runtime
source archive and one completed B0 nonlinear probability reader. The private
manifest specifies all 55 entries; [RESTORE_REPLAY_SOURCE_PLAN_V3.json](RESTORE_REPLAY_SOURCE_PLAN_V3.json)
lists their relative labels and hashes. The V3 runtime archive adds the
tracked `lra` dependency required by the restored decoder; its 450 prior
members are byte-identical. V3 passed independent uncached source, snapshot
and restored-file hash checks.

The shared-queue unit `R1_RESTORE_B0_J_READER_V3` executed eight restored
one-recipient raw-input CLI calls on 32 admitted FIT rows and one restored
nonlinear reader inference. Its [public receipt](R1_RESTORE_B0_J_READER_V3_RECEIPT.json)
reports PASS. Each emitted wire file has the exact hash of the earlier
raw-input deployment receipt, the cached codes agree, and all code, probability
and decision values decode through the fixed bank. The B0 seed-0 reader takes
the concatenated emitted probabilities (`[n,2] + [n,6]`) and returned 32 finite
probabilities in [0,1]. No task or SEX label entered this replay.

The separate [independent X replay](../r5/R5_VERIFY_RESTORE_V3_RECEIPT.json)
confirmed all eight wire hashes, fixed bank and cached code parity, and the
reader's 32 predictions with maximum absolute difference 0. It accessed no
labels. The V1 replay failed because the source archive omitted `latent_repair_v1`;
V2 reached decoder reconstruction but failed because the archive omitted
`lra`. Both failures and their original outputs are retained in
[R1_RESTORE_ATTEMPTS_V3.json](R1_RESTORE_ATTEMPTS_V3.json). These representative
32 FIT rows verify executable custody and replay; they do not establish
population deployment or privacy performance.
