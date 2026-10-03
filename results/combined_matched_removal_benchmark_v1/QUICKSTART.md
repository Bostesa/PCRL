# Quickstart: matched attribute-removal benchmark

**Where to run.** Run everything from the worktree root, on branch `research/combined-matched-removal-benchmark-v1`.
The runner derives its root from its own location and refuses any other branch.

**Private inputs and outputs** live in `~/PCRL_eval_cache_private/bench_v1/`. A verified copy is on the external
drive under `private_bench_v1_20261003/`.

```
P=results/combined_matched_removal_benchmark_v1
R=$P/scripts/run_benchmark.sh
```

| Purpose | Command | Tested 2026-10-03 |
|---|---|---|
| Tests | `OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m pytest -q stored_model_eval/tests` | 97 passed, 94 s |
| Plan / dry-run (no fits) | `STAGE=plan sh $R`; `STAGE=dry-run sh $R` | plan == registered IDs; dry-run OK (2026-10-02, before fits) |
| **Resume Tier 1** (lock-verified; skips hash-verified units) | `STAGE=tier1 RESUME=1 sh $R` | 126/126 skipped, 0 fits, lock OK, 5.2 s |
| Resume Tier 2 (same trigger and budget rule) | `STAGE=tier2 RESUME=1 sh $R` | Not re-run after the stop; by the rule it would stop again at the same unit, because the ledger is cumulative |
| σ\* (attacker_val only) | `STAGE=sigma-star sh $R` | 2.3 s; Adult 2.0, HMDA 4.0 |
| Inference (saved predictions only) | `STAGE=infer TIER=all sh $R` | 38 min, 2.9 GB peak |
| Report tables / plots | `STAGE=report sh $R`; `STAGE=plots sh $R` | about 10 s |
| **Independent replay** (numpy only; no repo imports; no refits) | see the command below | 83 min; 100,954 checks; primary 183/183 PASS; 185 FAIL, all exploratory and diagnosed (VALIDATION.md §3) |
| Full re-execution from scratch (only under a new lock) | `STAGE=lock sh $R`, then `STAGE=sanity UNITS=… sh $R`, `STAGE=tier1`, `sigma-star`, `tier2`, `infer`, `report`, `plots` | about 5 h wall, 5.0 CPU-h |

**Independent replay command:**

```
OMP_NUM_THREADS=1 /opt/homebrew/bin/python3 $P/verification/replay_bench.py \
  --root ~/PCRL_eval_cache_private/bench_v1 --report-dir $P \
  --out $P/INDEPENDENT_VERIFICATION.json --results-out $P/verification/replay_results_aggregate.json
```

**Restoring from the drive.** If the laptop cache is missing, restore it before replaying:

```
ditto <drive>/private_bench_v1_20261003/bench_v1 ~/PCRL_eval_cache_private/bench_v1
cd ~/PCRL_eval_cache_private && shasum -a 256 -c <drive>/private_bench_v1_20261003/SHA256SUMS
```

**Lock behaviour.**
- Every fitting stage verifies `LOCK.json` and refuses on any change to:
  - the 39 pinned code files or the dependency versions;
  - the official concept-erasure tree hash;
  - the input files, the support freeze, the unit list, the primary family or the inference pins.
- LEACE maps are pinned in `defenses/MAP_PINS.json`. A refit that differs from its pin is refused.
- Paths in `LOCK.json` are home-relative (`~/…`).

**Sanity stage.** The stage needs an explicit `UNITS` list. The registered run used the 6 arm-A Tier-1 units.
