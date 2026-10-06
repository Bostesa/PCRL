# Quickstart

## Placeholders and test status

| Placeholder | Meaning |
|---|---|
| `<WORKTREE>` | A checkout of branch `research/pcrl-decision-preserving-compression-v1` |
| `<python>` | The project virtualenv interpreter |
| `<PRIVATE_CACHE>` | The private, never-committed store |
| `<DRIVE_ROOT>` | The external drive, once mounted |

Run every command from `<WORKTREE>`. Every dpc command refuses unless `OMP_NUM_THREADS=1`.

Every command below was run on 2026-10-06 against the completed study (outputs summarised under each command). Two exceptions:
- Commands needing the external drive (`--dest <DRIVE_ROOT>`) were rehearsed with `--dry-run` only, because the drive is absent.
- The assessment and inference were not re-run after completion, so the assessment was opened only once.

## 1. Run or resume the study

Each stage runs as a timed worker. `work.sh` lives in `<PRIVATE_CACHE>/dpc_v1/run/`; it records CPU and peak memory and logs to `work_<stage>[_<i>of<n>].log`.

A stage refuses unless its lock file is committed, byte-identical, and on `origin`. A stage that is re-run skips every unit that already has a `COMPLETE.json`, so the same command resumes after an interruption.

```sh
W=<PRIVATE_CACHE>/dpc_v1/run/work.sh
$W ENGINEERING_LOCK admit                                       # admit teachers U, RAW-J beta 0.3 and refs E, F, F0
$W TRAINING_LOCK partition 0/2 &  $W TRAINING_LOCK partition 1/2 # fine KL partitions (two workers at most)
$W TRAINING_LOCK fit 0/2 &        $W TRAINING_LOCK fit 1/2       # 252 mapping-pair units + 6 CLASS units
$W SELECTION_AND_AUDIT_LOCK inner 0/2 & $W SELECTION_AND_AUDIT_LOCK inner 1/2
$W SELECTION_AND_AUDIT_LOCK controls
$W SELECTION_AND_AUDIT_LOCK select
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.eval_lock results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json
#   commit + push EVALUATION_LOCK.json BEFORE the next command (dpc.assess refuses otherwise)
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.assess --evaluation-lock results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json --shard 0/2
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.assess --evaluation-lock results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json --shard 1/2
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.infer  --evaluation-lock results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.report --part all
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m pytest -q dpc/tests
```

**Tested: resume.** `$W SELECTION_AND_AUDIT_LOCK partition 0/2` and `$W SELECTION_AND_AUDIT_LOCK fit 1/2` each verified the lock, skipped every completed unit and exited 0 (3 s and 2 s).

`dpc.assess` resumes in the same way (one memoised unit per seed and label). It was not re-run after the single assessment opening.

## 2. Deploy a code from the 83-column input

The deployed teacher is `<PRIVATE_CACHE>/dpc_v1/admitted/rel__s{k}__U`, and the policy is `units/pol__s{k}__<config>/policy.json`. For the best runnable compact policy, the config is `U_JOINT_m8_l0.1`.

**This code is NOT ELIGIBLE under the score contract.** It preserves every decision, but fails the occupation confidence allowance.

Build an input file, or supply your own npz with exactly two arrays: `X` (n × 83) and `feature_names` (83, pinned names in pinned order).

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -c "
import json, numpy as np
from dpc import data as DA
D = DA.load(); names = [str(x) for x in D['feature_names']]
np.savez('<in.npz>', X=D['X'], feature_names=np.asarray(names))
json.dump(names, open('<schema.json>', 'w'))"
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.deploy --unit <PRIVATE_CACHE>/dpc_v1/admitted/rel__s1__U \
    --policy <PRIVATE_CACHE>/dpc_v1/run/units/pol__s1__U_JOINT_m8_l0.1/policy.json \
    --X <in.npz> --schema <schema.json> --out <release.npz>
```

**Tested: deployment.** The command returned `binding: BOUND`, 39,170 rows and alphabets [16, 41]. The output contains only these six arrays:

| Recipient | Arrays |
|---|---|
| 1 | `tokens_1`, `probs_1`, `decision_1` |
| 2 | `tokens_2`, `probs_2`, `decision_2` |

Tokens, decoded probabilities and decisions are bitwise equal to the study's stored release for that unit.

**Tested refusals** (each exits with code 2 and writes nothing):

| Attempt | Message |
|---|---|
| An 84-column input | `deployment accepts exactly the 83 permitted columns; got 84` |
| A reordered schema | `input columns are reordered relative to the pinned schema (first difference at column 0 …)` |
| `--export-fine-ids` | `would export a non-released field (fine, export)` |
| `--raw-scores` | `would export a non-released field (raw, score)` |
| A seed-1 policy with the seed-0 teacher | `policy was fitted for a different teacher model.pt` |

Any flag outside the allow-list is refused. The output never contains fine-cell IDs, teacher probabilities, logits, features or distances.

**Eligible baseline.** The only release meeting the score contract is U's continuous output: the admitted `rel__s{k}__U` heads, as packaged by the predecessor (`osf.deploy`). It carries no protection.

## 3. Restore the private store from a copy, and check it

The current copy is on the same device: `<PRIVATE_CACHE>/dpc_v1_local_copy_20261006/` (2,364 files, 673,460,066 bytes).

**This is not an off-device backup.**

```sh
cd <PRIVATE_CACHE>/dpc_v1_local_copy_20261006 && shasum -a 256 -c SHA256SUMS --quiet     # every file
rsync -a <PRIVATE_CACHE>/dpc_v1_local_copy_20261006/dpc_v1/ <PRIVATE_CACHE>/dpc_v1/       # restore (only if lost)
```

**Tested: checksums.** `shasum` exited 0 over all 2,364 entries.

The restore checks from the copy alone were run when the copy was made. All seven targets passed (`BACKUP_VERIFICATION.json`):
- teacher U;
- teacher RAW-J;
- the U|JOINT|m8|l0.1 code, re-encoded from the restored teacher;
- the U|FINE-TASK|m8 code;
- the C_global = U continuous release;
- the RAW-J teacher unit;
- the final pair attacker, refit at seeds 0–2 with a maximum difference of 0.

To repeat these checks read-only against the live store:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout backup --dry-run --targets <PRIVATE_CACHE>/dpc_v1/run/closeout_targets.json
```

**Tested:** all 7 PASS, in 4 s.

## 4. New-study off-device backup (PENDING: drive absent)

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout status        # drive probe by content; writes nothing
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout backup --dest <DRIVE_ROOT> --targets <PRIVATE_CACHE>/dpc_v1/run/closeout_targets.json
```

**Tested:** `status` (exit 0; `mounted: false`, 0 matching volumes) and `backup --dry-run` (exit 0; would write a new versioned same-device copy `_v2`, and all restore checks PASS).

**How the backup command behaves:**
- It identifies the drive by content: the closed smf study's `SHA256SUMS` hash. It never uses a volume name.
- It copies to `<DRIVE_ROOT>/private_dpc_v1_<UTC date>[_vN]/dpc_v1`, re-reads every file uncached, and repeats the restore checks from the drive copy alone.
- It rewrites `BACKUP_VERIFICATION.json` and `RESTORE_INDEX.json`. Commit those files afterwards.

## 5. Predecessor custody repair (osf / smf; PENDING: drive absent)

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout predecessor --restore
```

This runs the documented `python -m osf.closeout backup --dest <DRIVE_ROOT>` and `python -m osf.closeout predecessor --restore` in-process.
- **Covers:** the osf drive copy, the smf drive restore by the smf verifier, and the two later smf logs.
- **Receipts:** redirected into `provenance/predecessor_custody/`.
- **Safety:** the closed osf and smf results trees are checked byte-identical before and after.

**Tested:** `predecessor --dry-run` and `predecessor --restore --dry-run` (both exit 0). With no drive, both report three gaps:
- osf has a same-device copy only;
- the smf drive restore has not been executed;
- two smf logs exist on the internal disk only.

`provenance/predecessor_custody/STATUS.json` records PENDING.

## 6. Independent replay

```sh
OMP_NUM_THREADS=1 <python> results/pcrl_decision_preserving_compression_v1/verification/replay_dpc.py              # full (phase 1 + 2)
OMP_NUM_THREADS=1 <python> results/pcrl_decision_preserving_compression_v1/verification/replay_dpc.py --selftest-only --no-write
```

The verifier imports no study code: an import guard refuses `dpc`, `osf`, `smf`, `rgj`, `jcv`, `pnx`, `oar` and `stored_model_eval`. It writes `INDEPENDENT_VERIFICATION.json`. The options:

| Option | Effect |
|---|---|
| `--no-refit` | Skips attacker refits |
| `--search sample` | Replays a sample of the search traces |
| `--light-update` | Re-runs only the light sections |

**Tested:** `--selftest-only --no-write` (exit 0, 1.6 s). The verifier's own full run produced the committed report.
