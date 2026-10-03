# Quickstart: output-aware removal study

**Where to run.** Run from the worktree root on branch `research/combined-output-aware-removal-v1`.

**Private outputs.** They are in `~/PCRL_eval_cache_private/oar_v1/`. A verified copy is on the external drive under
`private_oar_v1_20261003/`. The benchmark inputs are in `~/PCRL_eval_cache_private/bench_v1/`.

**Environments.**
- The project venv `~/PCRL/.venv` runs everything except the official FARE fit and certificate.
- Those run in `~/PCRL_eval_cache_private/oar_v1/env_fare`; the recipe is in `notes/fare/FARE_ENV_RECIPE.md`. The
  wrapper `oar/fare_official.py` calls into it.

```
T="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0"
P=results/combined_output_aware_removal_v1
```

| Purpose | Command | Tested 2026-10-03 |
|---|---|---|
| Tests | `env $T ~/PCRL/.venv/bin/python -m pytest -q oar/tests` | 17 passed, 21 s |
| Stage 0 admission (no fits) | `~/PCRL/.venv/bin/python $P/scripts/s0_admission.py` | all hashes OK, 9 s |
| Stage 1 custody (locked inference) | `env $T ~/PCRL/.venv/bin/python $P/scripts/s1_custody_primary.py` | 24/24 identical, 877 s |
| Stage 1 corrections | `env $T ~/PCRL/.venv/bin/python $P/scripts/s1_corrections.py`, then `.../s1_ledger.py` | 39 s |
| Stage 2 exposure sensitivity | `env $T ~/PCRL/.venv/bin/python $P/scripts/s2_exposure.py`, then `$P/report/exposure_tables.py` | 2,524 s |
| Lock check | `~/PCRL/.venv/bin/python -c "from oar.lock import verify_lock; from pathlib import Path; print(verify_lock(Path('$P/EXECUTION_LOCK.json')))"` | ok |
| **Run or resume one dataset** (lock-verified; skips hash-complete units) | `env $T ~/PCRL/.venv/bin/python -m oar.run --dataset adult --lock $P/EXECUTION_LOCK.json` (same for `hmda`) | adult 1,077 s; hmda 1,826 s. A re-run skips all complete units. |
| Inference and tables (saved predictions only) | `env $T ~/PCRL/.venv/bin/python $P/report/infer_report.py` | 170 s |
| Certificate amendment A1 | `env $T ~/PCRL/.venv/bin/python $P/report/amend_A1_certificates.py` | < 1 min |
| **Independent replay** (numpy; no `oar.*` or `stored_model_eval.*` imports) | `env $T ~/PCRL/.venv/bin/python $P/verification/replay_oar.py --exposure-original` | see VALIDATION.md |

**Restoring from the drive.** If the laptop cache is missing, restore it:

```
ditto <drive>/private_oar_v1_20261003/oar_v1 ~/PCRL_eval_cache_private/oar_v1
cd ~/PCRL_eval_cache_private && shasum -a 256 -c <drive>/private_oar_v1_20261003/SHA256SUMS
```

**Lock behaviour.**
- `oar.run` refuses unless `EXECUTION_LOCK.json` verifies. The lock pins code, dependencies, the official FARE tree,
  the inputs, the LEACE maps, the roles and support, and the primary family.
- Adding or changing a pinned file needs a dated lock amendment. L1 is recorded in the lock.
