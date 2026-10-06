# Quickstart (cbp)

| Placeholder | Meaning |
|---|---|
| `<WORKTREE>` | A checkout of branch `research/pcrl-confidence-budgeted-privacy-v1` |
| `<python>` | The project virtualenv interpreter |
| `<PRIVATE_CACHE>` | The private store, never committed |
| `<DRIVE_ROOT>` | The external drive, once mounted |

Run everything from `<WORKTREE>`.

- **Threads:** every study command refuses unless `OMP_NUM_THREADS=1`.
- **Semaphore:** heavy commands run under the shared two-slot semaphore: `python -m cbp.sema --label <who> -- <cmd>`.
  When it is called by file path, add `-P`.
- **What was run:** every command in sections 1–3 was run on 2026-10-06 against this study, exactly as written apart
  from the placeholders. Section 4 records the closeout commands and their tested status.

## 1. Run or resume the study

`work.sh` is in `<PRIVATE_CACHE>/cbp_v1/run/`. It wraps `cbp.sema` and `/usr/bin/time -l` around
`python -m cbp.run --lock <LOCK> --stage <stage> [--shard i/n]`.

A stage refuses to run unless:
- its lock is byte-identical on origin;
- every locked file is unchanged;
- every study module it loads is locked with its hash.

Completed units are skipped by hash, so the same command resumes after an interruption.

```sh
W=<PRIVATE_CACHE>/cbp_v1/run/work.sh
$W SOURCE_ADMISSION_LOCK admit
$W AUDIT_AND_SELECTION_LOCK fit                              # endpoint parity (33 reused units), then the 48 new fits
$W AUDIT_AND_SELECTION_LOCK inner 0/2 & $W AUDIT_AND_SELECTION_LOCK inner 1/2
$W AUDIT_AND_SELECTION_LOCK inner_src 0/2 & $W AUDIT_AND_SELECTION_LOCK inner_src 1/2   # composed U source (needs every code's inner unit)
$W AUDIT_AND_SELECTION_LOCK controls 0/2 & $W AUDIT_AND_SELECTION_LOCK controls 1/2
$W AUDIT_AND_SELECTION_LOCK select                           # selection.json / SELECTION.json / tables (+ inner validation)
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.sema --label A:eval_lock -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.eval_lock results/pcrl_confidence_budgeted_privacy_v1/EVALUATION_LOCK.json
#   commit + push EVALUATION_LOCK.json BEFORE the assessment (cbp.assess refuses otherwise)
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.sema --label A:assess_0of2 -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.assess --evaluation-lock results/pcrl_confidence_budgeted_privacy_v1/EVALUATION_LOCK.json --shard 0/2
#   (and --shard 1/2 in the second slot; --seeds / --labels restrict the locked list; completed units are skipped)
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.sema --label A:infer -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.infer --evaluation-lock results/pcrl_confidence_budgeted_privacy_v1/EVALUATION_LOCK.json
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.report --part fits      # CLASS_PRESERVATION.json, OPTIMIZATION_RECEIPTS.json
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.report --part inner     # figure 1, INNER_STATES_VS_RECOVERY.csv
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.report --part outer     # ASSESSMENT_COMPARISON.csv, figures 2-4
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.sema --label A:report-post -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.report_post   # post-lock, presentation only
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m pytest -q cbp/tests cbp/review_tests
```

**Tested.** Every line above ran with exit 0. `cbp.report_post` was run after the assessment and is a post-lock
reporting completion. It only adds files (ENDPOINT_RECEIPTS.json, INNER_STATES_VS_RECOVERY_v2.csv, figure 1b and the
`_annotated` figures 2–4) and changes no locked output. Locks and timings are in RUN_STATUS.json, and the attempt log is in
ASSESSMENT_ATTEMPTS.json. One note on the assessment run:
- Two extra load-balancing workers used `--seeds`/`--labels`.
- One of them overlapped shard 1/2 on 5 units. Both copies were bitwise identical, and the duplicates were retained
  privately.

## 2. Deploy a code from the 83-column input

| Release | Teacher unit | Policy | Status |
|---|---|---|---|
| Q = U\|DIRECT-TASK\|i8o64 | `<PRIVATE_CACHE>/cbp_v1/admitted/rel__s{k}__U` | `<PRIVATE_CACHE>/cbp_v1/run/units/pol__s{k}__U_DIRECT-TASK_i8o64/policy.json` | confidence feasibility established (P34–P37 PASS); task-only, no privacy claim |
| P\* = U\|JOINT\|i8o64\|l0.01 | same | `.../pol__s{k}__U_JOINT_i8o64_l0.01/policy.json` | inner-eligible; confidence clauses PASS; privacy criterion NOT established (pair benefit 0.002 vs T\*) |
| J\* fallback = U\|JOINT\|i8o64\|l0.04 | same | `.../pol__s{k}__U_JOINT_i8o64_l0.04/policy.json` | DESCRIPTIVE_ONLY (no eligible JOINT nominee) |

The input npz holds exactly `X` (n × 83) and `feature_names`, the pinned names in the pinned order. The reconstructed
authorized input is `<PRIVATE_CACHE>/cbp_v1/inputs/deploy_input.npz`, with `schema.json` beside it.

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.sema --label A:deploy -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.deploy \
    --unit <PRIVATE_CACHE>/cbp_v1/admitted/rel__s1__U \
    --policy <PRIVATE_CACHE>/cbp_v1/run/units/pol__s1__U_JOINT_i8o64_l0.01/policy.json \
    --X <PRIVATE_CACHE>/cbp_v1/inputs/deploy_input.npz --schema <PRIVATE_CACHE>/cbp_v1/inputs/schema.json --out <release.npz>
```

**Tested: deployment.** Q and P\* were run on seed 1 (39,170 rows). Both returned exit 0 and were BOUND. Each wrote only
`tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2`, bitwise equal to the stored release (`tok`, `q`, `hard`).

**Tested refusals.** Each exits 2.

| Attempt | Result |
|---|---|
| `--include-sex` | refused |
| `--export-fine-ids` | refused |
| `--raw-scores` | refused |
| `--debug` | refused |
| `--foo` (unknown flag) | refused |
| seed-0 teacher with a seed-1 policy | refused |
| unregistered λ = 1 policy (U\|JOINT\|i8o64\|l1) | refused |
| reordered columns | "input columns are reordered relative to the pinned schema" |
| 84 columns | "deployment accepts exactly the 83 permitted columns; got 84" |
| 82 columns | "deployment accepts exactly the 83 permitted columns; got 82" |
| an extra `sex` array in the input | "input npz must contain exactly X and feature_names" |

**Baseline.** U's continuous output is the admitted teacher itself, with no protection.

## 3. Independent replay

```sh
OMP_NUM_THREADS=1 <python> -P <WORKTREE>/cbp/sema.py --label F:replay -- env OMP_NUM_THREADS=1 <python> results/pcrl_confidence_budgeted_privacy_v1/verification/replay_cbp.py
```

The verifier imports no study code; an import guard refuses it. It writes `INDEPENDENT_VERIFICATION.json`. Its options
are listed by `--help`. See `VALIDATION.md`.

## 4. Backup, restore and custody

The current copy is on the **same device**, so it is **not** an off-device backup:

- location: `<PRIVATE_CACHE>/cbp_v1_local_copy_20261006`;
- contents: the 1,143-file store plus the bundled pinned input `dependencies/jcv_v1/inputs/adult_jcv.npz`;
- size: 1,144 files, 635,850,300 bytes after the final refresh at 21:41:03Z.

```sh
cd <PRIVATE_CACHE>/cbp_v1_local_copy_20261006 && shasum -a 256 -c SHA256SUMS --quiet          # every entry
rsync -a <PRIVATE_CACHE>/cbp_v1_local_copy_20261006/cbp_v1/ <PRIVATE_CACHE>/cbp_v1/            # restore (only if lost)
CO="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.sema --label F:closeout -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.closeout"
$CO status                                                         # drive probe by content; writes nothing
$CO all --dry-run --plan-only --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json
$CO all --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json            # drive absent: versioned same-device copy + restores
$CO refresh --dry-run --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json
$CO refresh --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json        # bring the copy up to date; deletes nothing
```

**Tested.**

| Command | When run | Result |
|---|---|---|
| `all` | 19:18Z, by role F, drive absent | exit 0; status `LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING` |
| `refresh --dry-run` | 20:50Z | exit 0 |
| `refresh` | 20:59Z (role F) and 21:41Z (lead) | exit 0 both times |
| `status`, `all --dry-run --plan-only`, `refresh --dry-run` | re-run by the lead at 21:01Z | each exit 0; wrote nothing; drive not mounted |
| `shasum -a 256 -c SHA256SUMS --quiet` in the copy | about 21:02Z, by the lead | exit 0 on all 1,144 entries |
| `rsync` restore line | not run | it would overwrite the live store; the restores above were done from the copy alone instead |

What `all` (drive absent) did:
- copied 1,144 files;
- re-read every file uncached (F_NOCACHE), with 1,144/1,144 matches;
- restored from the copy alone: teachers U and RAW-J (seed 1), Q, P\*, the J\* fallback and the selected pair attacker. All restores were bitwise; see BACKUP_VERIFICATION.json and RESTORE_INDEX.json.

What `refresh` did:
- appended the semaphore ledger and kept the previous SHA256SUMS;
- did not overwrite the two rewritten semaphore lock files, which is disclosed;
- deleted nothing.

**Pending: off-device backup and the earlier studies' custody (drive absent).** With the drive mounted, the drive is
found by content (never by name) and one command runs everything in order:

```sh
$CO all --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json
```

It runs the pending qpc / dpc / osf / smf custody step, in-process from this worktree, with receipts redirected to
`provenance/qpc_custody/`. It then makes the cbp drive copy at `<DRIVE_ROOT>/private_cbp_v1_<date>`, re-reads every
file uncached, and restores from the drive copy alone.

**Not tested:** this drive path has run only as `--dry-run`, because the drive is absent.
