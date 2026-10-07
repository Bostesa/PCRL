# Quickstart — lra

Every command marked **Tested** was executed in this study, under the shared two-slot semaphore with one thread per
process. Run everything from the worktree root.
- `<python>` is the pinned venv interpreter.
- `<PRIVATE_CACHE>` is the private store root.
- `PKG` is `results/pcrl_adult_learned_decoder_release_v1`.
- `TENV="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=."`

## 1. Locks

```
env $TENV <python> -m lra.lock verify PKG/SCIENCE_LOCK.json --stage d1
```

**Tested.** It returns `"ok": true`. Every stage refuses an unpushed lock, and CBP_LOCAL_ONLY cannot bypass that.

## 2. Stages (worker form)

```
env $TENV <python> -m lra.sema --label A:<stage> -- /usr/bin/time -l env $TENV <python> -m lra.run \
    --lock PKG/<LOCK>.json --stage <stage> [--shard i/n]
```

This is exactly what `<PRIVATE_CACHE>/lra_v1/run/work.sh <LOCK> <stage> [i/n]` executes.

**Tested**, with these stage/lock pairs:

| Stage | Lock | Shards |
|---|---|---|
| `admit` | SOURCE_ADMISSION_LOCK | — |
| `correctness` | CORRECTNESS_LOCK | — |
| `d1` | SCIENCE_LOCK | — |
| `ctask` | SCIENCE_LOCK | — |
| `fit` | SCIENCE_LOCK | 2 |
| `inner` | SCIENCE_LOCK | 2 |
| `inner_src` | SCIENCE_LOCK | — |
| `controls` | SCIENCE_LOCK | 2 |
| `select` | SCIENCE_LOCK | — |
| `d0same` | SCIENCE_LOCK | — |

- **Gate.** Every science stage refuses without a lock-bound, pushed ENGINEERING_READY.
- **Resume.** A re-run skips units whose COMPLETE.json verifies (unit-level resume). Each shard is restartable on its
  own.

## 3. Evaluation lock, assessment, inference

```
env $TENV <python> -m lra.sema --label A:eval-lock-build -- env $TENV <python> -m lra.eval_lock PKG/EVALUATION_LOCK.json
env $TENV <python> -m lra.sema --label A:assess_0of2 -- /usr/bin/time -l env $TENV <python> -m lra.assess \
    --evaluation-lock PKG/EVALUATION_LOCK.json --shard 0/2          # and --shard 1/2
env $TENV <python> -m lra.sema --label A:infer -- /usr/bin/time -l env $TENV <python> -m lra.infer \
    --evaluation-lock PKG/EVALUATION_LOCK.json
```

**Tested**, once each, as registered.
- The lock builder refuses on any technical failure, and there is no escape flag.
- `infer` writes run/inference.json, PKG/PRIMARY_ENDPOINTS.csv and PKG/ALL_LEVELS.csv.

## 4. Reports (read committed and private artifacts; no refitting)

```
env $TENV <python> -m lra.sema --label A:report -- env $TENV <python> -m lra.report fit work inner assess
```

**Tested** (each part separately). It writes:
- DECODER_ONLY_ABLATION.csv, BUDGET_FEASIBILITY.csv, DECODER_CERTIFICATES.json, OPTIMIZATION_RECEIPTS.json;
- DECODER_UTILITY_ABLATION.csv, DECISION_FLOOR_AND_FEASIBILITY.csv;
- RUN_STATUS.json, ACTUAL_WORK_ACCOUNTING.json;
- figures/.

## 5. Deployment (83-column input; teacher, map and decoder bound)

```
env $TENV <python> -m lra.sema --label A:deploy -- env $TENV <python> -m lra.deploy \
    --unit <PRIVATE_CACHE>/lra_v1/admitted/rel__s1__U \
    --policy <PRIVATE_CACHE>/lra_v1/run/units/<unit>/policy.json [--decoder <PRIVATE_CACHE>/lra_v1/run/units/<unit>/decoder.json] \
    --X <PRIVATE_CACHE>/lra_v1/inputs/deploy_input.npz --schema <PRIVATE_CACHE>/lra_v1/inputs/schema.json --out <release.npz>
```

**Tested** (DEPLOYMENT_RECEIPT.json).
- On seed 1, these releases deploy BOUND and bitwise equal to their stored releases:
  - Q (pol__s1__U_DIRECT-TASK_i8o64);
  - P\* (pol__s1__U_JOINT_i8o64_l0.1);
  - C-TASK D1 (new__s1__U_C-TASK_i8o64_D1, with `--decoder`);
  - K-SEQ-21 D1;
  - the D1 JOINT λ0.01 control.
- Nine refusals exit with code 2 and write nothing.
- The two D1 releases also deploy from the same-device copy.

## 6. Independent verifier (role E; imports no study module)

```
OMP_NUM_THREADS=1 <python> -P lra/sema.py --label E:phase3 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 <python> PKG/verification/replay_lra.py --phase 3 --label PHASE_3
```

**Tested** by role E. Phases 0, 1 and 2 use `--phase 0|1|2`. The result is in INDEPENDENT_VERIFICATION.json.

## 7. Custody

```
env $TENV <python> -m lra.sema --label F:closeout -- env $TENV <python> -m lra.closeout all \
    --targets <PRIVATE_CACHE>/lra_v1/run/closeout_targets.json
```

**Tested** by role F after the assessment opening.
- The drive was absent, so this made a versioned same-device copy, verified uncached and restored from.
- **Not run:** the off-device `backup --dest <DRIVE_ROOT>` and the predecessor custody, because the drive was absent.
  Both commands are in BACKUP_VERIFICATION.json and provenance/cbp_custody/STATUS.json.
