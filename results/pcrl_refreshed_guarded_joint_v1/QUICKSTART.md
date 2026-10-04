# Quickstart (commands tested 2026-10-04)

Run from the worktree root on `research/pcrl-refreshed-guarded-joint-v1`. Set up first:

```
export OMP_NUM_THREADS=1 PYTHONPATH=.
PY=~/PCRL/.venv/bin/python; P=results/pcrl_refreshed_guarded_joint_v1; L=$P/CODE_LOCK.json; E=$P/EVALUATION_LOCK.json
```

**Where the private data live.**
- New units, critics, predictions and logs: `~/PCRL_eval_cache_private/rgj_v1/`. The drive copy is `<drive>/private_rgj_v1_20261004/` (see `RESTORE_INDEX.json`).
- Predecessor inputs, warm starts and FARE trees: `~/PCRL_eval_cache_private/jcv_v1/`. The drive copy is `<drive>/private_jcv_v1_20261003/`.
- Every stage skips complete, hash-verified units, so rerunning a finished stage is a no-op (tested: about 5 s).

| Step | Command | Expected |
|---|---|---|
| Tests | `$PY -m pytest -q rgj/tests` | All pass: pipeline, math-review fixtures, audit, inference |
| Lock | `$PY -m rgj.lock verify $L` | `ok: true` (base lock plus amendments A1–A3) |
| Parity | `$PY -m rgj.run --lock $L --stage parity --shard 0/2` (and `1/2`) | 21 bitwise receipts (about 3 min on 2 workers) |
| Task line and Stage B | `--stage taskline`, then `--stage stageB` (shards 0/2 and 1/2) | 24 task-line checkpoints; 18 runs and 72 checkpoints (about 4 min) |
| Inner audit | `--stage inner` (shards) | `inner__*` for every candidate release |
| Stage B selection | `--stage selectB` | `selection_B.json` (private), `STAGE_B_SELECTION.json`; commit and push before Stage C |
| Calibration and Stage C | `--stage calibrate`, then `--stage stageC` (shards) | `calib__s{k}`; 36 runs, 144 checkpoints, `tc__s{k}` receipts (about 10 min) |
| Baselines | `--stage baselines` (shards) | `lc__s{k}__E`, 42 re-headed FARE units |
| Stage C selection | `--stage inner`, then `--stage selectC` | `SEED_STATUS.json`, `SELECTION_TABLE.csv` |
| Diagnostics (inner only) | `--stage tracking` (shards), `--stage whiten`, `--stage ablation` | `track__*`, `whiten__diag`, capped J-G units |
| Evaluation lock | `$PY -m rgj.eval_lock $E` | Rebuilt seeds are identical to the pushed lock (tested); commit and push before scoring |
| Assessment | `$PY -m rgj.assess --evaluation-lock $E --labels J-G L-G J-R J-O L-R L-O U` (and the other labels) | 39 `outer__*` units; refuses unless the lock is committed and on origin |
| Controls | `$PY -m rgj.assess --evaluation-lock $E --controls-only --seeds 0 --controls-labels J-G L-G J-O U F` | `controls all_ok = True` |
| Inference | `$PY -m rgj.infer --evaluation-lock $E` | `PRIMARY_ENDPOINTS.csv`, `SECONDARY_ENDPOINTS.csv`, `RAW_LEVELS.csv`; a rerun is byte-identical (tested) |
| Reports | `$PY -m rgj.report` | Tracking, gradient, utility, linear, controls, whitening and ablation tables, plus the figure |
| **Deploy** | `$PY -m rgj.deploy --unit ck__C__s1__J-G__b0.3__e20 --X <permitted_inputs.npy> --out release.npz` | Two recipient releases (r_i, centred logits, probabilities, decisions), bitwise equal to the stored release. No labels needed. Refuses anything but 83 columns. Labelled **EXPERIMENTAL_NO_ADVANTAGE** (descriptive J-G checkpoint; no valid nominee) |
| Backup | `$PY -m rgj.closeout --dest <drive folder>` | 3,062/3,062 files read back uncached; U, J-G and L-G restored bitwise from the drive copy; one recorded attacker refit reproduces its saved predictions |
| Independent replay | `$PY $P/verification/replay_rgj.py` | `INDEPENDENT_VERIFICATION.json` |

**Release units per seed.** Units for every label are listed in `MODEL_MANIFEST.json`. For seed 1:
- descriptive J-G: `ck__C__s1__J-G__b0.3__e20`;
- L-G: `ck__C__s1__L-G__b0.3__e20`;
- C\* = J-O: `ck__C__s1__J-O__b0.3__e15`;
- U: `tl__s1__e35`;
- LEACE: `lc__s1__E`.
