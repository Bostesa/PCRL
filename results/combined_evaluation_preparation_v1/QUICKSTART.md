# stored_model_eval: quickstart

Run everything from the worktree root:
`cd /Users/nathansamson/PCRL/.worktrees/combined-evaluation-preparation-v1`.
`PY=/Users/nathansamson/PCRL/.venv/bin/python` (torch is needed only for `forward`; system `python3` runs everything else).
`N=results/combined_evaluation_preparation_v1/notes/evaluator`, `M=results/combined_evaluation_preparation_v1/notes/methodology/protocol_config.json`.

The default mode performs **no scientific fits** and **no network access**. `--dry-run` is accepted by every command.
Runtimes were observed on 2026-10-02 on this Mac (14 cores, 24 GB).

## Tested commands

| step | command | observed |
|---|---|---|
| tests | `OMP_NUM_THREADS=2 $PY -m pytest stored_model_eval/tests -q` | 26 passed, 2.1-2.4 s |
| plan | `python3 -m stored_model_eval --protocol $M --out $N/pilot_plan_methodology.json plan --spec $N/pilot_plan_spec.json --calibration $N/calibration_methodology_measured.json` | 0.1 s; 72 units, 16 missing/PENDING inputs listed, 0 fits |
| admit (synthetic) | `$PY -c "import sys; sys.path.insert(0,'.'); from stored_model_eval.fixtures import *; write_manifest(make_synthetic('direct', n_units=2000), '/tmp/sme')"` then `python3 -m stored_model_eval admit --manifest /tmp/sme/synthetic_manifest.json` | 0.1 s, ADMITTED |
| admit (real, forward subset) | `python3 -m stored_model_eval admit --manifest ~/PCRL_eval_cache_private/forward_smoke/forward_manifest.json` | 0.06 s, ADMITTED (256 rows); the bad-hash manifest exits 2 |
| recount (b) | `python3 -m stored_model_eval --out $N/recount_pcrl_strict.json recount pcrl-strict --repo /Users/nathansamson/PCRL --ref origin/main` | 0.15 s; 56/60 final, 54/60 best |
| recount (a) | `python3 $N/build_aaai67_spec.py` (builds the 67-config spec, checks sha256 against the drive inventory, calls `recount probs`) | 1.4 s; 64/59/51 |
| forward smoke | `$PY $N/forward_smoke.py` (wraps `python -m stored_model_eval forward --manifest ... --cache-dir ~/PCRL_eval_cache_private/forward_smoke/cache --tag run_a --threads 1`) | 3.9 s total; one forward call 0.65 s |
| synthetic slate | `OMP_NUM_THREADS=1 $PY -m stored_model_eval --protocol $M fit-attackers --synthetic --synthetic-kind direct --n-units 15060 --dim 64 --out-dir <scratch>/syn` | 38.5 s (binary); 147 s with `--synthetic-kind contrast --k 5`; peak RSS 0.33 GB |
| score / infer / report | `python3 -m stored_model_eval score --scores <dir>/scores.npz`; `... --out inf.json infer --scores <dir>/scores.npz`; `... report --scores <dir>/scores.npz --infer inf.json` | score below 1 s; infer at n_eval 4.5k, 2000 replicates, 27 statistics: 20 s (binary) / 65 s (K=5) |
| validation | `$PY $N/build_validation.py` | 2.5 s; writes `validation_results.json` |

`OMP_NUM_THREADS` matters: unpinned, HistGradientBoosting on this machine spent most of its time in thread
spin (16 s wall at 1040 percent CPU for a 2 s job).

## Prepared scientific pilot (NOT executed; needs approval and the protocol lock)

Cell: the stored PCRL Round-4 Adult seed-0 release, which is the encoder durable-guarantees audited. Rows are
the local Adult test split (15,060 rows; 5 exact duplicate records collapse to units). It covers 8
(purpose, attribute) pairs, 3 surfaces and the attacker slate. Roles (methodology shares) give attacker_fit
7,571, attacker_val 2,239 and assessment 5,250 rows.

```
# 1. inputs (private cache, outside git); no fitting
$PY $N/prepare_pilot_adult_s0.py --write-inputs
# 2. frozen forward pass of all test rows (eval mode, no grad, sha256-checked)
$PY -m stored_model_eval forward --manifest ~/PCRL_eval_cache_private/pilot_adult_s0/forward_manifest.json \
    --cache-dir ~/PCRL_eval_cache_private/pilot_adult_s0/cache --tag adult_s0_test --threads 1
$PY $N/prepare_pilot_adult_s0.py --pair-manifests
# 3. per pair (8 pairs): admit, fit (the ONLY step that needs the flag), infer, report
for m in ~/PCRL_eval_cache_private/pilot_adult_s0/manifest_*.json; do
  t=$(basename $m .json)
  python3 -m stored_model_eval admit --manifest $m || exit 1
  OMP_NUM_THREADS=1 $PY -m stored_model_eval --protocol $M fit-attackers --manifest $m \
      --execute-scientific-fits --out-dir ~/PCRL_eval_cache_private/pilot_adult_s0/scores/$t
  $PY -m stored_model_eval --protocol $M --out ~/PCRL_eval_cache_private/pilot_adult_s0/scores/$t/infer.json \
      infer --scores ~/PCRL_eval_cache_private/pilot_adult_s0/scores/$t/scores.npz
  $PY -m stored_model_eval --protocol $M --out ~/PCRL_eval_cache_private/pilot_adult_s0/scores/$t/report.json \
      report --scores ~/PCRL_eval_cache_private/pilot_adult_s0/scores/$t/scores.npz \
      --infer ~/PCRL_eval_cache_private/pilot_adult_s0/scores/$t/infer.json
done
```

**CPU and memory estimate.** This is not a measurement. It was derived from timed synthetic runs at exactly the
pilot row counts (n_fit 7.5k, d=64, single thread), scaled by grid size and class count, and the cost model
reproduced the K=5 GBT time within 1 percent.

* Methodology slate (R01 x5, GBT 9 configs with max_iter 500, MLP 9 configs): **0.31 CPU-h** for seed 0. This
  is 692 s of fits plus 432 s of bootstrap at B=2000.
* Default slate: 0.17 CPU-h.
* Peak RSS is about 0.35 GB.
* Allow a 2x margin for real-data difficulty, mainly MLP epochs: at most about 0.6 CPU-h for seed 0.
* Seeds 1-2 (checkpoints on the external drive, PENDING) would bring it to about 1-2 CPU-h.

**Known before running.**

* Adult race has 61 and 40 assessment rows in two classes. Race per-class and pair statistics will be
  NOT_ESTIMABLE (3/5 classes and 3/10 pairs supported) unless the lock pools categories.
* Adult has no linkage unit: the unit is the de-duplicated record.
* The role split is hashed by record, not stratified by (s, y).

## Coordinator update (2026-10-02): inputs prepared, pilot command tested in dry-run

**Already done** (no fitting; the outputs live in `~/PCRL_eval_cache_private/pilot_adult_s0/`, outside git):

```
PY=/Users/nathansamson/PCRL/.venv/bin/python
N=results/combined_evaluation_preparation_v1/notes/evaluator
$PY $N/prepare_pilot_adult_s0.py --write-inputs
$PY -m stored_model_eval forward --manifest ~/PCRL_eval_cache_private/pilot_adult_s0/forward_manifest.json \
    --cache-dir ~/PCRL_eval_cache_private/pilot_adult_s0/cache --tag adult_s0_test --threads 1
$PY $N/prepare_pilot_adult_s0.py --pair-manifests     # 8 untreated manifests
$PY $N/prepare_pilot_adult_s0.py --noise-manifests    # 144 noise-arm manifests (8 pairs x 6 sigma x 3 seeds)
```

Inputs, forward pass and manifests took 3.9 s in total. 152 of 152 manifests were admitted.

**Tested pilot command, dry-run** (admission plus `fit-attackers --dry-run` for 26 units: 8 untreated pairs and 18
noise arms for income_prediction/sex):

```
sh results/combined_evaluation_preparation_v1/notes/evaluator/run_pilot_cell_a.sh
# -> "units processed: 26 (mode: --dry-run)", 23 s, 0 fits
```

**Prepared scientific pilot (NOT executed):**

```
EXECUTE=1 sh results/combined_evaluation_preparation_v1/notes/evaluator/run_pilot_cell_a.sh
```

For each unit, this runs:
- the attacker slate on the rep, outputs and rep+outputs surfaces;
- the closed-form endpoints (native, R02, held-out ρ₁²);
- cluster-bootstrap inference;
- the report.

**Cost estimate.** This is an estimate from the evaluator's timed synthetic calibration at the pilot row counts, not
a measurement on real data.

| Part | Estimate |
|---|---|
| 8 untreated pairs (several are multiclass) | ≈ 0.31 CPU-h |
| 18 binary noise arms, each ≈ 2.3 min (slate on 3 surfaces + bootstrap) | ≈ 0.7 CPU-h |
| **Total** | **≈ 1.0 CPU-h** |
| Allowance with a 2× margin for real-data MLP epochs | ≤ 2 CPU-h |

- Wall time is about the same as CPU time single-threaded. Units are independent and can run in parallel.
- Peak memory is about 0.35 GB.
- Cloud and GPU: none.

**Extensions** (prepared but not part of the pilot):
- E1: CELL-B HMDA inputs. Port `prepare_pilot_adult_s0.py` to HMDA; the Round-4 HMDA checkpoint is in the private
  cache.
- E2: all 144 noise manifests. Extra cost ≈ 0.7 CPU-h per 18 arms, about 5.5 CPU-h in total.
- E3: Round 5/7 NeurIPS checkpoints. Extract them from the drive and verify against the inventory.
- E4: historical-recount track for the 10 projection arms. This uses stored per-row scores and the `recount`
  subcommand, with no fitting.
