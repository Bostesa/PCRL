# Reproducibility guide

This document gives a step-by-step reproduction recipe for every numeric
claim in the paper. The README's "Headline results" and "Paper-claim →
artifact" tables list which file backs which claim; this document goes
further: which command produces that file, on what hardware, and how long
to expect.

The old universal R²-to-classification-accuracy bound and its nonlinear
smoothing extension are retired as invalid. Historical bound columns must
not be interpreted as guarantees. Current reports retain empirical
least-squares statistics and separately fitted attacker accuracies; see
[the mathematical correction](ACCURACY_CERTIFICATE_RETIREMENT.md).

## Environment

Python 3.10 or 3.12. PyTorch 2.0+. Other deps in `requirements.txt`.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository performs **no external network calls during reproduction**
beyond `pip install` and the dataset downloads in `docs/DATA.md`. There is
no Weights & Biases / sentry / telemetry integration. If you wish to opt in
to W&B logging, set `WANDB_API_KEY` and `WANDB_PROJECT` env vars and edit
the relevant entry-point — none of the canonical scripts read these by
default.

## Datasets

Follow `docs/DATA.md`. Place each dataset under the documented path; the
loaders in `pcrl/data/` and `pcrl/{vision,language}/` read the official
release layout directly.

## Stage 1 — train the V2 grid (60 cells = 3 datasets × pairs × 3 seeds)

```
python experiments/run_v2_dataset.py --dataset adult    --out-tag _ROUND5 --seeds 0 1 2
python experiments/run_v2_dataset.py --dataset hmda     --out-tag _ROUND5 --seeds 0 1 2
python experiments/run_v2_dataset.py --dataset diabetes --out-tag _ROUND7 --seeds 0 1 2
```

Wall-clock on `g4dn.xlarge` (1× T4): Adult ≈ 2.3 h, HMDA ≈ 3.1 h,
Diabetes ≈ 2.1 h. CPU runs are 4–8× slower.

Outputs:
- `results/v2_adult_ROUND5/{per_seed_results,summary}.json`
- `results/v2_hmda_ROUND5/{per_seed_results,summary}.json`
- `results/v2_diabetes_ROUND7/{per_seed_results,summary}.json`
- `checkpoints/v2_{adult,hmda,diabetes}_{ROUND5,ROUND5,ROUND7}_s{0,1,2}/final.pt` (gitignored, written locally)

## Stage 2 — dominant-axis audit (33 / 33 convex-combo identity, claim #8)

CPU-only. Re-uses the trained checkpoints from Stage 1.

```
python scripts/eval_round4_dominant_axis.py --datasets adult hmda --seeds 0 1 2 --tag ROUND5
python scripts/eval_round4_dominant_axis.py --datasets diabetes  --seeds 0 1 2 --tag ROUND7
```

Wall-clock: ≈ 5 min total. Outputs `dominant_axis_audit.json` in the same
result directories as Stage 1.

## Stage 3 — cross-purpose concatenation attack (claim #10)

CPU-only.

```
python experiments/run_cross_purpose_attack_v2.py --datasets adult hmda diabetes --seeds 0 1 2
```

Wall-clock: ≈ 12 min. Outputs `results/v2_cross_purpose/aggregate.json`.

The dual-criterion split (26 / 33 absolute, 22 / 33 incremental) is built
by the LAFTR comparison driver:

```
python scripts/run_laftr_benchmark.py
```

which produces `results/cross_purpose_laftr/DUAL_CRITERIA.json`.

## Stage 4 — Round-4 baseline (claim #6)

The Round-4 results are committed under `results/v2_{adult,hmda,diabetes}_ROUND4/`.
Re-running them requires checking out the pre-fix commit (the optimizer
schedule was changed in commit `dbe0fdc` — the Round-5 numbers are
post-fix). A `git log -- pcrl/training/proxy_lagrangian.py` will surface
the relevant commits if you want to bisect.

## Stage 5 — τ sweep (claim #5)

```
python scripts/tier1_tau_sensitivity.py
```

Wall-clock: ≈ 4 min on CPU. Reads existing checkpoints from Stage 1.
Outputs `results/tier1_analyses/tau_sensitivity.json` plus the LaTeX table.

## Stage 6 — Zhao-Gordon TV-Barycenter certificate (claim #7)

```
python scripts/zhao_gordon_certificate.py
```

Wall-clock: ≈ 2 min on CPU. Reads existing checkpoints. Outputs
`results/reviewer_dropins/zhao_gordon_table.{csv,json}` (20 cells).

## Stage 7 — Cross-purpose Prop 6 bound utilization (claim #11)

```
python results/reviewer_dropins/prop6_bound_utilization/verify_bound_v2.py
```

Wall-clock: ≈ 1 min on CPU. Outputs
`results/reviewer_dropins/prop6_bound_utilization/numerical_verification_v2.json`.

## Stage 8 — CelebA (claims #13, #14)

```
python experiments/run_celeba_medium.py --train --seed 0
python experiments/run_celeba_medium.py --train --seed 1
python experiments/run_celeba_medium.py --train --seed 2
```

Wall-clock: ≈ 75 min per seed on `g4dn.xlarge`. Per-seed outputs land in
`results/v2_celeba_R5_{FULL,seed1,seed2}/`. The 6-architecture audit is
re-evaluated by `experiments/run_celeba_ensemble_eval.py`.

## Stage 9 — Baselines

```
# LEACE-on-raw threat baseline (claim — 0/20 strict pass)
python experiments/run_leace_baseline.py --dataset adult
python experiments/run_leace_baseline.py --dataset hmda
python experiments/run_leace_baseline.py --dataset diabetes

# LAFTR per-purpose benchmark (claim #18)
python scripts/run_laftr_benchmark.py

# INLP three-way benchmark (claim #19)
python scripts/run_inlp_benchmark.py

# SPLINCE 60-cell comparison (claim #15)
python scripts/run_splince_benchmark.py
```

Wall-clocks (CPU): LEACE ≈ 3 min/dataset; LAFTR ≈ 30 min; INLP ≈ 90 min;
SPLINCE ≈ 4 h.

## Stage 10 — Held-out validation (claim #12)

The seed-3 retrain on Adult is committed under `results/v2_adult_HELDOUT_S3/`.
Re-running it requires invoking the Stage-1 driver with `--seeds 3` plus
the held-out config:

```
python experiments/run_v2_dataset.py --dataset adult --out-tag _HELDOUT_S3 --seeds 3
python scripts/heldout_s3_compare.py
```

Wall-clock: ≈ 1 h on `g4dn.xlarge`.

## End-to-end wall-clock

A complete reproduction from a clean clone, on a single `g4dn.xlarge`:
- Stage 1 (V2 grid): ≈ 7.5 GPU-h
- Stage 8 (CelebA): ≈ 3.75 GPU-h
- Stages 2, 3, 5, 6, 7, 9 (CPU): ≈ 6.5 CPU-h
- Stage 10 (held-out s3): ≈ 1 GPU-h

≈ 12.25 GPU-h + 6.5 CPU-h end-to-end.

## Re-using committed artifacts

Most readers will not re-train. The committed `results/` tree backs every
numeric claim, and the analysis scripts in `scripts/` re-derive the LaTeX
tables and Markdown tables-of-numbers from the JSON / CSV files without
needing the trained checkpoints.

To regenerate the LaTeX tables from the committed result JSONs:

```
python scripts/build_dominant_axis_latex.py
python scripts/build_single_purpose_baselines_latex.py
python scripts/build_splince_vs_pcrl.py
python scripts/build_stage2_report.py
```
