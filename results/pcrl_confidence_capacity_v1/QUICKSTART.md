# Quickstart

## Placeholders and test status

| Placeholder | Meaning |
|---|---|
| `<WORKTREE>` | A checkout of branch `research/pcrl-confidence-capacity-v1` |
| `<python>` | The project virtualenv interpreter |
| `<PRIVATE_CACHE>` | The private, never-committed store |
| `<DRIVE_ROOT>` | The external drive, once mounted |

Run everything from `<WORKTREE>`. Every study command refuses unless `OMP_NUM_THREADS=1`. **Every heavy command runs under the shared two-slot semaphore**: `python -m qpc.sema --label <who> -- <cmd>`. When it is called by file path, add `-P`.

Every command below was run on 2026-10-06 against the completed study. Two exceptions:
- Drive-dependent commands were rehearsed with `--dry-run` only, because the drive is absent.
- The assessment and inference were not re-run after completion, so the assessment was opened only once.

## 1. Run or resume the study

`work.sh` lives in `<PRIVATE_CACHE>/qpc_v1/run/`. It wraps `qpc.sema` and `/usr/bin/time -l` around `python -m qpc.run --lock <LOCK> --stage <stage> [--shard i/n]`.

Each stage refuses unless:
- its lock and every amendment after it are byte-identical on origin;
- every locked file is unchanged;
- every study module it loads is locked with its hash.

Completed units are skipped, so the same command resumes after an interruption.

```sh
W=<PRIVATE_CACHE>/qpc_v1/run/work.sh
$W SOURCE_ADMISSION_LOCK admit
$W STAGE_A_LOCK stagea ;  $W STAGE_A_LOCK gate               # gate -> CAPACITY_GATE.json (selected rates)
$W STAGE_B_LOCK partition 0/2 & $W STAGE_B_LOCK partition 1/2
$W STAGE_B_LOCK fit 0/2 & $W STAGE_B_LOCK fit 1/2
$W AUDIT_AND_SELECTION_LOCK inner 0/2 & $W AUDIT_AND_SELECTION_LOCK inner 1/2
$W AUDIT_AND_SELECTION_LOCK inner_src                        # composed U source bank (needs every code's inner unit)
$W AUDIT_AND_SELECTION_LOCK controls 0/2 & $W AUDIT_AND_SELECTION_LOCK controls 1/2
$W AUDIT_AND_SELECTION_LOCK select
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.eval_lock results/pcrl_confidence_capacity_v1/EVALUATION_LOCK.json
#   commit + push EVALUATION_LOCK.json BEFORE the assessment (qpc.assess refuses otherwise)
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.sema --label A:assess_0of2 -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.assess --evaluation-lock results/pcrl_confidence_capacity_v1/EVALUATION_LOCK.json --shard 0/2
#   (and --shard 1/2 in a second slot)
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.sema --label A:infer -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.infer --evaluation-lock results/pcrl_confidence_capacity_v1/EVALUATION_LOCK.json
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.report --part stagea ; OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.report --part stageb
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.report_outer
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m pytest -q qpc/tests
```

**Tested: resume.** `$W AUDIT_AND_SELECTION_LOCK fit 1/2` verified the lock plus AMENDMENT_A1, skipped every completed unit (the unit count was unchanged at 243) and exited 0 in 3 s.

## 2. Deploy a code from the 83-column input

| Release | Teacher unit | Policy |
|---|---|---|
| Q\* = U\|DIRECT-TASK\|i8o64 | `<PRIVATE_CACHE>/qpc_v1/admitted/rel__s{k}__U` | `<PRIVATE_CACHE>/qpc_v1/run/units/pol__s{k}__U_DIRECT-TASK_i8o64/policy.json` |
| P\* = U\|JOINT\|i8o64\|l0.1 | same | `.../pol__s{k}__U_JOINT_i8o64_l0.1/policy.json` |

Q* is confidence-feasible. P* is inner-eligible, but its confidence preservation is NOT established on the assessment.

The input npz holds exactly `X` (n × 83) and `feature_names`, the pinned names in pinned order. The reconstructed authorised input is `<PRIVATE_CACHE>/qpc_v1/inputs/deploy_input.npz`, with `schema.json` beside it.

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.deploy --unit <PRIVATE_CACHE>/qpc_v1/admitted/rel__s1__U \
    --policy <PRIVATE_CACHE>/qpc_v1/run/units/pol__s1__U_DIRECT-TASK_i8o64/policy.json \
    --X <PRIVATE_CACHE>/qpc_v1/inputs/deploy_input.npz --schema <PRIVATE_CACHE>/qpc_v1/inputs/schema.json --out <release.npz>
```

**Tested: deployment.** Q* and P* (seed 1, 39,170 rows) return exit 0, are BOUND, and output only these six arrays:

| Recipient | Arrays |
|---|---|
| 1 | `tokens_1`, `probs_1`, `decision_1` |
| 2 | `tokens_2`, `probs_2`, `decision_2` |

Both are bitwise equal to the stored releases.

**Tested refusals** (each exits 2):

| Attempt | Refusal |
|---|---|
| An 84-column input | refused |
| A reordered schema | refused |
| `--export-fine-ids` | refused |
| `--raw-scores` | refused |
| A seed-0 teacher with a seed-1 policy | "policy was fitted for a different teacher model.pt" |
| An unknown flag (`--verbose`) | refused |

**Baseline.** U's continuous output is the admitted teacher, with no protection.

## 3. Restore and check the private store

The current copy is on the same device: `<PRIVATE_CACHE>/qpc_v1_local_copy_20261006/qpc_v1` (925 files, 625,755,727 bytes). **This is not an off-device backup.**

```sh
cd <PRIVATE_CACHE>/qpc_v1_local_copy_20261006 && shasum -a 256 -c SHA256SUMS --quiet       # every file
rsync -a <PRIVATE_CACHE>/qpc_v1_local_copy_20261006/qpc_v1/ <PRIVATE_CACHE>/qpc_v1/        # restore (only if lost)
```

**Tested: checksums.** `shasum -c` exited 0 on all 925 entries.

The restore checks from the copy alone, run by `qpc.closeout`, all passed (`BACKUP_VERIFICATION.json`, `RESTORE_INDEX.json`):
- teachers U and RAW-J, by an own forward pass, bitwise;
- Q* and P*, re-encoded and deployed from the copy, bitwise;
- the selected pair attacker, refit from the copy, max diff 0.

## 4. New off-device backup (PENDING: drive absent)

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.closeout status                        # drive probe by content; writes nothing
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.closeout all --dry-run --plan-only --targets <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.sema --label E:closeout -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.closeout all --targets <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json
```

**Tested:** `status` (exit 0, not mounted) and `all --dry-run --plan-only` (exit 0, PENDING, wrote nothing).

**With the drive mounted,** `all` does the following:
1. Identifies the drive by content.
2. Completes the dpc off-device backup.
3. Repairs the osf/smf predecessor custody.
4. Writes a new versioned qpc copy, re-reads every file uncached, and restores from the copy alone.

## 5. Predecessor custody repair (osf, smf and dpc; PENDING)

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.closeout dpc-backup                # dpc off-device backup (documented dpc command, in-process)
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.closeout predecessor --restore     # osf backup + smf drive restore + 2 late smf logs
```

**Tested:** `dpc-backup --dry-run` and `predecessor --dry-run` (both exit 0, PENDING without the drive).

Receipts are redirected into `provenance/dpc_custody/` and `provenance/predecessor_custody/`. The closed dpc, osf and smf results trees are never written.

## 6. Independent replay

```sh
OMP_NUM_THREADS=1 <python> -P <WORKTREE>/qpc/sema.py --label F:replay -- env OMP_NUM_THREADS=1 <python> results/pcrl_confidence_capacity_v1/verification/replay_qpc.py
```

The verifier imports no study code (import guard) and writes `INDEPENDENT_VERIFICATION.json`. Options are listed by `--help`; see `VALIDATION.md`.
