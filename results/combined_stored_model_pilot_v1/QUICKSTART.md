# Quickstart: CELL-A stored-model pilot

Run everything from the owned worktree `/Users/nathansamson/PCRL/.worktrees/combined-stored-model-pilot-v1`, on
branch `research/combined-stored-model-pilot-v1`. The runner derives its root from its own location and refuses
any other branch.

Private inputs and outputs live in `~/PCRL_eval_cache_private/pilot_adult_s0/`. A verified copy is on the drive
under `private_pilot_run_v1_20261002/`.

```
S=results/combined_stored_model_pilot_v1/scripts
```

| Purpose | Command | Tested 2026-10-02 |
|---|---|---|
| Tests | `cd stored_model_eval && OMP_NUM_THREADS=1 /Users/nathansamson/PCRL/.venv/bin/python -m pytest -q` | 52 passed, 42 s |
| Plan + dry-run (no fits) | `sh $S/run_pilot.sh` | "plan OK (expected == actual registered unit IDs)", "dry-run OK" |
| **Resume** (lock-verified; skips hash-verified units) | `RESUME=1 EXECUTE=1 sh $S/run_pilot.sh` | 26/26 "complete and hash-verified; skipped", 2.6 s, 0 fits |
| Inference (saved predictions only) | `STAGE=infer sh $S/run_pilot.sh` | 124 s |
| Report tables | `git -C /Users/nathansamson/PCRL show origin/main:results/v2_adult_ROUND4/dominant_axis_audit.json > /tmp/hist_da.json && HIST=/tmp/hist_da.json STAGE=report sh $S/run_pilot.sh` | tables written |
| **Independent replay** (no retraining; system python) | `/opt/homebrew/bin/python3 results/combined_stored_model_pilot_v1/verification/replay.py --inputs-dir ~/PCRL_eval_cache_private/pilot_adult_s0 --run-dir ~/PCRL_eval_cache_private/pilot_adult_s0/run_v1 --report-dir results/combined_stored_model_pilot_v1 --out <path>.json` | 3 min 14 s; PASS 19,358 / FAIL 0 / MC_BORDERLINE 21 / NOTE 1 / INFO 86 (deterministic) |
| Full re-execution from scratch (only after a new amendment) | `EXECUTE=1 sh $S/run_pilot.sh` | 684 s wall, 0.19 CPU-h, 0.51 GB |

**Restoring from the drive.** If the laptop cache is missing, restore it before replaying:

```
ditto /Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-30/private_pilot_run_v1_20261002/pilot_adult_s0 ~/PCRL_eval_cache_private/pilot_adult_s0
```

Then verify the copy with `shasum -a 256 -c` against the drive `SHA256SUMS`, run from `~/PCRL_eval_cache_private`.

**Lock behaviour.** Execution refuses if any pinned code file, manifest or data file differs from `PILOT_LOCK.json`
(identical to `PILOT_LOCK_v2.json`).
