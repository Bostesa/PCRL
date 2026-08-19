# PCRL: Purpose-Conditioned Representation Learning

PCRL trains a single shared encoder that produces multiple purpose-specific
representations, each carrying a compliance certificate (linear R² ≤ 0.05)
for a declared list of disallowed sensitive attributes. The method combines
a closed-form joint LEACE warm-start with a proxy-Lagrangian fine-tune on a
held-out R² constraint per (purpose, attribute) pair. Results are evaluated
under both the standard one-hot R² metric and a Dominant-Axis audit that
exposes per-class leakage hidden by aggregate one-hot scoring.

## Headline results

All numbers are sourced from the JSON / CSV files under `results/`. See the
"Reproducing the headline numbers" section below for one command per claim,
and the "Paper-claim → artifact" table at the bottom of this README for the
exact file backing each row.

| Claim | Number | Backing artifact |
|---|---|---|
| Strict pass at τ=0.05 across Adult / HMDA / Diabetes (60-cell grid) | **56 / 60** | `results/v2_{adult,hmda,diabetes}_ROUND{5,5,7}/dominant_axis_audit.json` |
| Cleanly compliant (strict pass + per-dim std ≥ 0.5 + eff. rank ≥ 2.0) | **7 / 60** | same as above + `collapse_diagnostic.json` |
| Round-4 baseline (pre-fix optimizer schedule) | **46 / 60** strict, mean R² = 0.038 | `results/v2_{adult,hmda,diabetes}_ROUND4/summary.json` |
| τ-sweep (R² thresholds 0.01 / 0.025 / 0.05) | **37 / 52 / 56 of 60** | `results/tier1_analyses/tau_sensitivity.json` |
| Convex-Combination Identity audit | **33 / 33** multi-class pair-seeds, max residual < 0.01 | same DA audit JSONs |
| Cross-purpose concatenation attack | **26 / 33** triples flag recovery above majority +1pp under absolute-leakage criterion (Criterion A); **22 / 33** under incremental-leakage criterion (Criterion B) | `results/cross_purpose_laftr/DUAL_CRITERIA.json`, `results/v2_cross_purpose/aggregate.json` |
| LAFTR baseline under PCRL strict-pass criterion | **15 / 60** | `results/laftr_benchmark/STAGE2_ADULT.md` + per-cell artifacts |
| Per-purpose INLP baseline | **20 / 27** | `results/inlp_benchmark/inlp_results.json` |
| SPLINCE comparison (60 cells) | **3 / 60** strict-health pass, **60 / 60** R²-only pass, **2 / 60** R² + Δ-auditor pass | `results/splince_benchmark/splince_vs_pcrl_summary.json` |
| Held-out validation seed 3 (Adult) | **6 / 8** pass, mean Δ R² = +0.0111 | `results/v2_adult_HELDOUT_S3/summary.json` |

## Repository structure

```
.
├── README.md
├── LICENSE                             MIT, anonymous holder
├── requirements.txt
├── pcrl/                               core package: encoder, LEACE warm-start, proxy-Lagrangian
│                                       trainer, per-purpose LoRA adapters, compliance certificates,
│                                       dominant-axis audit, vision (CelebA) and language (BIOS) backbones
├── experiments/                        training and evaluation entry points
├── scripts/                            analysis utilities and table builders
├── tests/                              unit tests for the core package
├── configs/                            YAML hyperparameter configs used by canonical runs
├── results/                            JSON / CSV / Markdown artifacts cited by the paper
├── paper-body/tables/                  LaTeX tables shipped with the paper
├── figures/                            PDF figures referenced by the paper
├── docs/                               REPRODUCIBILITY.md, DATA.md
└── data/                               instructions only — raw datasets are NOT committed
```

## Setup

Python 3.10 or 3.12 is required.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then download the datasets you need by following `docs/DATA.md`. Datasets
are kept out of the repository.

## Reproducing the headline numbers

Each command below assumes the environment in "Setup" is active. Outputs
land under `results/`; the canonical artifact cited in the paper is listed
alongside each command. End-to-end reproduction of the 60-cell grid is on
the order of 8 GPU-hours on a single T4 (`g4dn.xlarge`); the cross-purpose
attack and dominant-axis audit each run on CPU.

### Strict pass at τ=0.05 (56 / 60 across Adult, HMDA, Diabetes)

```
python experiments/run_v2_dataset.py --dataset adult    --out-tag _ROUND5 --seeds 0 1 2
python experiments/run_v2_dataset.py --dataset hmda     --out-tag _ROUND5 --seeds 0 1 2
python experiments/run_v2_dataset.py --dataset diabetes --out-tag _ROUND7 --seeds 0 1 2
```

Outputs `results/v2_{adult,hmda,diabetes}_ROUND{5,5,7}/{per_seed_results,summary,dominant_axis_audit}.json`.

### Convex-Combination Identity audit (33 / 33 multi-class pair-seeds)

```
python scripts/eval_round4_dominant_axis.py --datasets adult hmda --seeds 0 1 2 --tag ROUND5
python scripts/eval_round4_dominant_axis.py --datasets diabetes  --seeds 0 1 2 --tag ROUND7
```

The `convex_combo_residual` field on each multi-class row is the
per-pair-seed identity residual.

### Cross-purpose concatenation attack (26 / 33 absolute, 22 / 33 incremental)

```
python experiments/run_cross_purpose_attack_v2.py --datasets adult hmda diabetes --seeds 0 1 2
```

The `DUAL_CRITERIA.json` aggregator splits the 33 triples by the two
criteria; both are reported in the paper.

- **Criterion A (absolute leakage):** `concat_acc − majority_baseline > 1pp`. **PCRL: 26 / 33.** LAFTR: 29 / 33.
- **Criterion B (incremental leakage):** `concat_acc − best_single_purpose_acc > 1pp`. **PCRL: 22 / 33.** LAFTR: 16 / 33.

### CelebA decoupled erase-layer architecture (n=3 seeds)

```
python experiments/run_celeba_medium.py --train --seed 0
python experiments/run_celeba_medium.py --train --seed 1
python experiments/run_celeba_medium.py --train --seed 2
```

Per-seed outputs land in `results/v2_celeba_R5_{FULL,seed1,seed2}/`.

### LEACE-on-raw threat baseline (0 / 20 strict pass)

```
python experiments/run_leace_baseline.py --dataset adult
python experiments/run_leace_baseline.py --dataset hmda
python experiments/run_leace_baseline.py --dataset diabetes
```

Outputs `results/{adult,hmda,diabetes}_LEACE/leace_baseline.json`. Summary
in `results/LEACE_SUMMARY.md`.

### LAFTR per-purpose benchmark

```
python scripts/run_laftr_benchmark.py
```

Outputs `results/laftr_benchmark/STAGE2_ADULT.md` and per-cell `metrics.json`.

### INLP three-way benchmark

```
python scripts/run_inlp_benchmark.py
```

Outputs `results/inlp_benchmark/{inlp_results,PAPER_PASTE,pcrl_vs_laftr_vs_inlp_table.tex}`.

### SPLINCE 60-cell comparison

```
python scripts/run_splince_benchmark.py
```

Outputs `results/splince_benchmark/splince_results.json` plus per-cell
`metrics.json` and the comparison summary `splince_vs_pcrl_summary.json`.


## Hardware

The canonical experiments were run on a single T4 GPU (`g4dn.xlarge` or
equivalent: 1× T4, 16 GB GPU RAM, 4 vCPU). A single dataset's full
multi-seed run takes 1–4 hours of GPU time. The CelebA decoupled
architecture takes about 75 minutes per seed. The dominant-axis audit and
the cross-purpose attack run comfortably on CPU once the trained
checkpoints are available.



## License

Released under the MIT License — see `LICENSE`.
