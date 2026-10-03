# Quickstart

**Where to run.** From the worktree root, on branch `research/combined-output-diagnosis-v1`.

**Private data.**
- Inputs: `~/PCRL_eval_cache_private/bench_v1`.
- Reused units: `~/PCRL_eval_cache_private/oar_v1`.
- This study's outputs: `~/PCRL_eval_cache_private/odx_v1`, with a verified drive copy at `private_odx_v1_20261003`.

```
T="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0"
P=results/combined_output_diagnosis_v1
```

| Purpose | Command | Tested 2026-10-03 |
|---|---|---|
| Tests | `env $T ~/PCRL/.venv/bin/python -m pytest -q odx/tests oar/tests` | all pass (repairs, fixtures, canonical utility, FARE wrapper, study runner) |
| Custody and replays (no fits) | `PCRL_DRIVE=<drive> env $T ~/PCRL/.venv/bin/python $P/scripts/s0_custody.py` | inputs 18/18, reused units 372 / 2,286 files OK; encoder, head and attacker replays exact |
| Repairs R1–R3 (no fits) | `~/PCRL/.venv/bin/python $P/scripts/s0_repairs.py` | 10 ledger rows |
| Manifest and coverage (no fits) | `~/PCRL/.venv/bin/python $P/scripts/s1_manifest.py` | 449 planned slots |
| Lock check | `~/PCRL/.venv/bin/python -c "from odx.lock import verify_lock; from pathlib import Path; print(verify_lock(Path('$P/LOCK.json')))"` | ok |
| **Run or resume stages 2/3** (lock-verified; hash-complete units are skipped) | `env $T ~/PCRL/.venv/bin/python -m odx.run --dataset adult --stage s2s3 --lock $P/LOCK.json` (same for `hmda`) | 1,042 s / 805 s; a re-run skips all complete units |
| S4 coalition / controls | `... -m odx.run --dataset adult --stage s4 ...`; `... --stage controls ...` (both datasets) | 115 s / 21 s + 41 s |
| Stage 5 (conditional) | `OAR_RUN_UNITS=~/PCRL_eval_cache_private/odx_v1/run/units env $T ~/PCRL/.venv/bin/python -m odx.fare5 --lock $P/LOCK.json` | 461 s |
| Inference and tables (saved predictions only) | `env $T ~/PCRL/.venv/bin/python $P/report/infer_all.py`; `$P/report/infer_s5.py`; `$P/report/usefulness.py`; `$P/report/fare_replay.py`; `$P/report/s5_certificates_A1.py`; `$P/report/figures.py` | about 1 min each |
| **Independent replay** (no runner imports) | `env $T ~/PCRL/.venv/bin/python $P/verification/replay_odx.py` | see VALIDATION.md |

**Restore from the drive:**
1. `ditto <drive>/private_odx_v1_20261003/odx_v1 ~/PCRL_eval_cache_private/odx_v1`
2. `cd ~/PCRL_eval_cache_private && shasum -a 256 -c <drive>/private_odx_v1_20261003/SHA256SUMS`

**Safe resume.**
- Every runner stage verifies `LOCK.json` and refuses on any code, input or family change.
- Units are written atomically with COMPLETE.json hashes. A partial unit is moved aside, never reused.
