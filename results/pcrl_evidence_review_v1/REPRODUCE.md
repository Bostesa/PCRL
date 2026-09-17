# Reproduce the Terminal B audit

Branch `research/pcrl-evidence-paper-v1`, worktree `/Users/nathansamson/PCRL-terminal-b`, baseline
`349efa454afd907389760fd1f59fd8806a215efd`.

Everything here is CPU-only, single process, one BLAS/OpenMP thread. Nothing refits a model,
re-selects a candidate or opens the 2016 final partition.

## Software

Project venv `/Users/nathansamson/PCRL/.venv`: Python 3.13.7, NumPy 2.4.2, pandas, Matplotlib,
pytest 9.0.2. The reanalysis was additionally run under CPython 3.14 with a different NumPy build and
agreed to the same tolerance; the fixtures and tests use the project venv.

```bash
cd /Users/nathansamson/PCRL-terminal-b
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/Users/nathansamson/PCRL/.venv/bin/python
```

## Required local inputs

The per-person prediction arrays are ignored by git and were **resolved read-only** from the original
checkout; nothing was refitted to recreate them.

* `LOCAL=/Users/nathansamson/.config/superpowers/worktrees/PCRL/residual-spectral-20260910`
* study directory: `$LOCAL/results/redesign_20260917_acs_spectral_transport_v1`
  (15 GB: `final_labels.npz`, `seed_*/<interface>/mode_{A,B}/{predictions.npz,metrics.json}`,
  `seed_*/context/*`, `TRANSPORT_LOCK.json`, three amendments)
* published (committed) copy of the same study, for `COMPARISONS.json` and `FAMILIES.csv`:
  `results/redesign_20260917_acs_spectral_transport_v1`

Integrity of those inputs is not assumed: step 3 re-hashes all 17,639 locked inputs.

## Steps

```bash
# 1. independent regeneration of every pre-declared endpoint  (~80 s, < 1 GB RSS)
$PY -m experiments.pcrl_evidence_review_v1.reanalyse_transport \
      --study   $LOCAL/results/redesign_20260917_acs_spectral_transport_v1 \
      --published results/redesign_20260917_acs_spectral_transport_v1 \
      --out     results/pcrl_evidence_review_v1
# writes INDEPENDENT_FAMILIES.csv, FAMILY_AGREEMENT.json, EQUIVALENCE_F1.csv,
#        MODEB_TRANSPORT_TABLE.csv, SELECTION_RECHECK.json

# 2. tradeoff, direction, scope and matching tables  (~2 s)
$PY -m experiments.pcrl_evidence_review_v1.tradeoff_tables \
      --published results/redesign_20260917_acs_spectral_transport_v1 \
      --out       results/pcrl_evidence_review_v1

# 3. re-verify the lock over all 17,639 inputs with every amendment applied  (~5 s, 1.94 GB hashed)
$PY -m experiments.pcrl_evidence_review_v1.recheck_lock \
      --study $LOCAL/results/redesign_20260917_acs_spectral_transport_v1 \
      --root  $LOCAL --out results/pcrl_evidence_review_v1

# 4. claim audit and evidence ledger  (instant)
$PY -m experiments.pcrl_evidence_review_v1.build_claim_audit --out results/pcrl_evidence_review_v1

# 5. 2016 admission preparation  (~3 s; needs the public download, see below)
$PY -m experiments.pcrl_evidence_review_v1.admit_acs_2016 \
      --data data/acs_2016_admission --out results/pcrl_evidence_review_v1

# 6. tests: mathematical fixtures + sealed-year loader  (21 tests, < 1 s)
$PY -m pytest tests/pcrl_evidence_review_v1 -q

# 7. manuscript tables and figures, then the PDF
$PY -m experiments.pcrl_evidence_review_v1.make_paper_assets \
      --evidence results/pcrl_evidence_review_v1 --out papers/pcrl_evidence_v1
cd papers/pcrl_evidence_v1 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Public downloads for step 5

```bash
mkdir -p data/acs_2016_admission/official data/acs_2016_admission/extracted
cd data/acs_2016_admission/official
curl -L -O https://www2.census.gov/programs-surveys/acs/data/pums/2016/1-Year/csv_pca.zip
mv csv_pca.zip csv_pca_2016.zip
curl -L -O https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict16.txt
curl -L -O https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2017.txt
cd .. && unzip -o official/csv_pca_2016.zip -d extracted
```

Expected SHA256 (checked by step 5 and recorded in the manifest):

| File | SHA256 |
|---|---|
| `csv_pca_2016.zip` | `a434bf9316f174d896766488cd5fd4ae7356a9512258321110b62110c6ded818` |
| `ss16pca.csv` | `9d8b7eb28baa74a88537d3f429b0a361cc36e8c60f67c25d52cac1fb0a7d5c1a` |
| `PUMSDataDict16.txt` | `b21a4dd6500b03fe61de92f5433a3b25b635b07492a9806b51db2d8a49e2b89b` |
| `PUMS_Data_Dictionary_2017.txt` | `ea5bcde2a07b0c76ac145fac108e9013480b686a9d70afc4e3f4a6512ccbfd69` |

Note that `PUMS_Data_Dictionary_2016.txt` does **not** exist; that URL returns a Census 404 page that
saves as a plausible-looking file. The 2016 one-year dictionary is `PUMSDataDict16.txt`.

Raw 2016 records stay under the ignored `data/` path. Only the aggregate manifest is published.

## Expected results

| Step | Expected |
|---|---|
| 1 | critical values 2.930106 / 3.023780 / 2.728486 / 3.205669; max abs difference vs the study 1.11e-15 (estimates), 2.40e-15 (intervals); 0 selection mismatches in 924 cells |
| 3 | `pass_with_all_amendments: true`, 0 changed, 0 missing, 0 invalid, 17,639 files |
| 5 | 0 schema violations; 79,298 eligible people in 53,093 groups; partition 39,697 / 15,917 / 23,684; 0 household overlap |
| 6 | 21 passed |
| 7 | `main.pdf`, 12 pages, 0 overfull boxes, 0 undefined references |

## What is intentionally not reproducible here

Refitting any interface, probe, prior or attacker; opening the 2016 final partition; and anything
that would spend a confirmation set. The audit consumes stored evidence only.
