# Quickstart — lcr

The study ended at the mechanism gate (MECHANISM_GATE_NOT_MET), so no Adult fit command was ever run. Every command
marked **Tested** below was executed in this study, with what it did. The one command marked **Not run** (off-device
backup) is stated as such. Run everything from the worktree
root. `<python>` is the pinned venv interpreter and `<PRIVATE_CACHE>` is the private store root. Numerical commands go
through the shared two-slot semaphore (`lcr.sema`) with one thread.

```
TENV="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=."
```

## 1. Check the lock

```
env $TENV <python> -m lcr.lock verify results/pcrl_learned_decoder_constrained_release_v1/FIXTURE_LOCK.json --stage fixture
```

**Tested.** It returns `"ok": true`. Both FIXTURE_LOCK.json and AMENDMENT_A1_FIXTURE_C5_SCOPE.json are on origin, and
every locked file hash matches.

## 2. The registered fixture stage

```
env $TENV <python> -m lcr.sema --label A:fixture -- /usr/bin/time -l env $TENV <python> -m lcr.run \
    --lock results/pcrl_learned_decoder_constrained_release_v1/FIXTURE_LOCK.json --stage fixture
```

This is exactly what the private worker script `<PRIVATE_CACHE>/lcr_v1/run/work.sh FIXTURE_LOCK fixture` executes. It
also appends start and exit lines to `run/work_fixture.log`.

**Tested.**
- It ran as attempt 2 (about 27 CPU-s, 0.4 GiB) in one process with no `--shard`.
- It writes FIXTURE_GATE.json, fixture_oracle/*.csv and the private units fix__<FID>.
- The engine is deterministic. A re-run reproduces every registered field and rewrites only the timings, but it would
  overwrite the registered file, so prefer step 3 for checking.
- The stage refuses to run unless the laws and the rule equal the lock's documents_sha256.

## 3. Reports (replay checked against the registered gate)

```
env $TENV <python> -m lcr.sema --label A:report -- env $TENV <python> -m lcr.report all
```

**Tested.**
- It re-runs the locked fixture engine and refuses unless the replay equals FIXTURE_GATE.json (timings excluded).
- It then writes:
  - DECODER_ONLY_ABLATION.csv, BUDGET_FEASIBILITY.csv;
  - DECODER_CERTIFICATES.json, OPTIMIZATION_RECEIPTS.json, CLASS_PRESERVATION.json;
  - RUN_STATUS.json, POST_HOC_EQUAL_LEAKAGE_UTILITY.json;
  - the NOT_RUN records for the Adult tables;
  - figures/.

## 4. Deployment (83-column input; teacher, map and decoder bound)

```
env $TENV <python> -m lcr.sema --label A:deploy -- env $TENV <python> -m lcr.deploy \
    --unit <PRIVATE_CACHE>/lcr_v1/admitted/rel__s1__U \
    --policy <PRIVATE_CACHE>/lcr_v1/run/units/pol__s1__U_DIRECT-TASK_i8o64/policy.json \
    --X <PRIVATE_CACHE>/lcr_v1/inputs/deploy_input.npz --schema <PRIVATE_CACHE>/lcr_v1/inputs/schema.json \
    --out <release.npz>
```

**Tested** (DEPLOYMENT_RECEIPT.json).
- Q on seed 1, 39,170 rows: exit 0, BOUND. It wrote only tokens_1/probs_1/decision_1/tokens_2/probs_2/decision_2, and
  every array is bitwise equal to the admitted release.
- These refusals exit with code 2 and write nothing:
  - 84 columns or 82 columns;
  - an extra `sex` array;
  - `--export-logits`;
  - an unknown flag;
  - a RAW-J teacher with a U policy.
- A D1 configuration also takes `--decoder <decoder.json>`, plus optional `--decoder-sha256`. That path is tested only in
  `lcr/tests/test_deploy.py` on synthetic data, because no Adult D1 decoder exists.

## 5. Tests

```
env $TENV <python> -m lcr.sema --label A:test-all -- env $TENV <python> -m pytest -q -p no:cacheprovider lcr/tests
```

**Tested.** All 173 pass. They use synthetic data and the registered fixture laws; no real labels are read.

## 6. Custody

```
env $TENV <python> -m lcr.sema --label F:closeout -- env $TENV <python> -m lcr.closeout status
env $TENV <python> -m lcr.sema --label F:closeout -- env $TENV <python> -m lcr.closeout all \
    --targets <PRIVATE_CACHE>/lcr_v1/run/closeout_targets.json
env $TENV <python> -m lcr.sema --label F:closeout -- env $TENV <python> -m lcr.closeout refresh \
    --targets <PRIVATE_CACHE>/lcr_v1/run/closeout_targets.json
```

**Tested.**
- `status`: the drive is absent, and the installer image is skipped by name.
- `all`: a versioned same-device copy of 731 files (01:49Z), verified uncached and restored from. The U and RAW-J teachers,
  the Q release and the Q deployment are all bitwise. Status: LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING.
- `refresh`: run at 02:28:12Z. It added 23 new files and re-copied 1 grown ledger, so the copy holds 754 files, all read back
  uncached. It deleted nothing and kept the previous SHA256SUMS (BACKUP_VERIFICATION.json `refreshes`).
- Off-device, when a drive is mounted:
  `lcr.closeout backup --dest <DRIVE_ROOT> --targets <PRIVATE_CACHE>/lcr_v1/run/closeout_targets.json`. **Not run:**
  there was no drive.

## 7. Independent verification (role E; imports no study module)

```
OMP_NUM_THREADS=1 <python> -P lcr/sema.py --label E:phase3 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 <python> results/pcrl_learned_decoder_constrained_release_v1/verification/replay_lcr.py \
    --phase 3 --label PHASE_3
```

**Tested** by role E: final run at 02:38:42Z (40.2 s wall); an earlier run was at 02:19:44Z. Result: PHASE_3, WARN with
0 FAIL, written to
INDEPENDENT_VERIFICATION.json (VALIDATION.md §6). Add `--out <file>` to write nothing to the package.

## Run / resume and inference

- **Fixture stage: no resume step.** It recomputes all four fixtures in one process (about 27 CPU-s), so a re-run is
  the resume.
- **Adult stages: resume never exercised.** Their unit-level resume (`lcr.run` skips units whose COMPLETE.json
  verifies) never ran on Adult.
- **Inference: not run.** `lcr.infer` needs an assessment, and none exists. No command for these is listed as tested.

## Not runnable as an Adult result

The Adult stages exist as code but were never executed under a lock:
- `--stage d1`, `ctask`, `fit`, `inner`, `inner_src`, `controls`, `select`;
- `lcr.eval_lock`, `lcr.assess`, `lcr.infer`.

`lcr.lock` refuses them without a SCIENCE_LOCK, and no SCIENCE_LOCK exists.

**Tested.** `lcr.run --lock …/FIXTURE_LOCK.json --stage d1` (also `fit` and `select`) stops with `REFUSED: lock does not verify: stage d1 needs SCIENCE_LOCK or later`, before any data is loaded.
