# PCRL: Purpose-Conditioned Representation Learning

Supplementary code for the anonymous NeurIPS 2026 submission
*"One Encoder, Many Purposes: Purpose-Conditioned Representation Learning
with Compliance Certificates."*

PCRL trains a single shared encoder that produces multiple purpose-specific
representations, each carrying a compliance certificate (linear R² ≤ 0.05)
for a declared list of disallowed sensitive attributes. The method combines
a closed-form LEACE warm-start with a proxy-Lagrangian fine-tune on a
held-out R² constraint per purpose-attribute pair. Results are evaluated
under both the standard one-hot R² metric and a Dominant-Axis audit that
exposes per-class leakage hidden by aggregate one-hot scoring.

## Repository structure

- `pcrl/` — core method package (LEACE warm-start, proxy-Lagrangian
  trainer, per-purpose LoRA adapters, compliance certificates,
  dominant-axis audit primitives, vision and language backbones).
- `experiments/` — training and evaluation entry points.
- `scripts/` — auxiliary analysis and table-generation utilities.
- `tests/` — unit tests for the core package.
- `configs/` — YAML hyperparameter configs used by the canonical runs.
- `results/` — JSON artifacts cited by the paper's tables and figures.
- `data/` — instructions for obtaining raw datasets (datasets are NOT
  included; see `data/README.md`).
- `paper-body/tables/` — LaTeX tables shipped with the paper.
- `figures/` — figure PDFs referenced by the paper.

## Reproducing the headline numbers

Each command assumes a fresh Python 3.10 or 3.12 environment with the
dependencies in `requirements.txt` installed. Outputs land under
`results/`; the canonical artifact cited in the paper is listed
alongside each command.

### Strict pass at τ=0.05 (56/60 across Adult, HMDA, Diabetes)

The canonical V2 pipeline trains all three seeds for one dataset per
invocation. Use `--out-tag` to name the output directory.

```
python experiments/run_v2_dataset.py --dataset adult    --out-tag _ROUND5 --seeds 0 1 2
python experiments/run_v2_dataset.py --dataset hmda     --out-tag _ROUND5 --seeds 0 1 2
python experiments/run_v2_dataset.py --dataset diabetes --out-tag _ROUND7 --seeds 0 1 2
```

Outputs:
- `results/v2_adult_ROUND5/{per_seed_results,summary}.json`
- `results/v2_hmda_ROUND5/{per_seed_results,summary}.json`
- `results/v2_diabetes_ROUND7/{per_seed_results,summary}.json`

The dominant-axis audit (next section) re-evaluates the trained
checkpoints and writes the canonical `dominant_axis_audit.json` files
that the paper's strict-pass tables read.

### Convex-Combination Identity audit (33/33 multi-class pair-seeds, max residual < 0.01)

Re-evaluates trained checkpoints and emits both `r2_onehot` and the
convex-combination prediction `predicted_r2_onehot_from_convex_combo`
plus the residual:

```
python scripts/eval_round4_dominant_axis.py --datasets adult hmda --seeds 0 1 2 --tag ROUND5
python scripts/eval_round4_dominant_axis.py --datasets diabetes --seeds 0 1 2 --tag ROUND7
```

Cited artifacts (one per dataset):
- `results/v2_adult_ROUND5/dominant_axis_audit.json`
- `results/v2_hmda_ROUND5/dominant_axis_audit.json`
- `results/v2_diabetes_ROUND7/dominant_axis_audit.json`

The `convex_combo_residual` field on each multi-class row is the
per-pair-seed identity residual.

### Cross-purpose concatenation attack (22/33 flagged amplifications)

```
python experiments/run_cross_purpose_attack_v2.py --datasets adult hmda diabetes --seeds 0 1 2
```

Cited artifact: `results/v2_cross_purpose/aggregate.json` — per
(dataset, attribute, attacker) `gain_mean_pp`, `gain_std_pp`, `verdict`.

### CelebA decoupled erase-layer architecture (n=3 seeds)

The `run_celeba_medium.py` script handles a single seed per invocation
and requires the `--train` flag for the full training run:

```
python experiments/run_celeba_medium.py --train --seed 0
python experiments/run_celeba_medium.py --train --seed 1
python experiments/run_celeba_medium.py --train --seed 2
```

Cited artifacts (per seed):
- `results/v2_celeba_R5_FULL/{train_set_r2,cross_purpose,training_log}.json`
- `results/v2_celeba_R5_seed1/...`
- `results/v2_celeba_R5_seed2/...`

### LEACE-on-raw threat baseline (0/20 strict pass)

The script accepts one `--dataset` per invocation:

```
python experiments/run_leace_baseline.py --dataset adult
python experiments/run_leace_baseline.py --dataset hmda
python experiments/run_leace_baseline.py --dataset diabetes
```

Cited artifacts:
- `results/{adult,hmda,diabetes}_LEACE/leace_baseline.json`
- Summary in `results/LEACE_SUMMARY.md`.

### BIOS BERT mechanistic interpretation

```
python experiments/run_mech_interp_stage3.py --device cuda
```

Add `--device cpu` for a CPU-only run (slow). Cited artifacts:
- `results/v2_bios_FINAL/{layer_probe_findings.md,rank_k_sweep.json}`
- Stage 3 outputs (`mech_interp_full_results.json`, `mech_interp_heatmap.pdf`).

## Environment

- Python 3.10 or 3.12.
- PyTorch 2.0 or later.
- All other dependencies pinned in `requirements.txt`.

Install with:

```
pip install -r requirements.txt
```

## Hardware

The canonical experiments were run on a single AWS T4 GPU instance
(`g4dn.xlarge` or equivalent: 1× T4, 16 GB GPU RAM, 4 vCPU). A single
dataset's full multi-seed run takes 1-4 hours of GPU time. The CelebA
decoupled-architecture run takes about 75 minutes per seed.

The dominant-axis audit and the cross-purpose attack scripts run
comfortably on a CPU-only machine in well under an hour total once the
trained checkpoints are available.

## Data

Raw datasets are not committed. See `data/README.md` for download
instructions. The five datasets used are:

- **Adult Census** — UCI Machine Learning Repository.
- **HMDA Loan Application Register** — FFIEC public release, 2023
  California LAR (`hmda_2023_ca.csv`).
- **Diabetes 130-US Hospitals** — UCI Machine Learning Repository.
- **CelebA** — official MMLAB release (`img_align_celeba/` plus
  `list_attr_celeba.csv`, `list_eval_partition.csv`).
- **BIOS** — De-Arteaga et al. 2019 release (top-10 occupations subset).

Preprocessing scripts:
- `experiments/prepare_hmda.py` — reads the raw CA LAR CSV and writes
  `data/hmda_processed/`.
- `experiments/preprocess_diabetes.py` — reads the UCI Diabetes archive
  and writes `data/diabetes_processed/`.

## Anonymization

This repository is supplementary material for an anonymous NeurIPS 2026
submission. Author and affiliation information have been removed. A
non-anonymized version with full attribution will be released after the
review process completes.
