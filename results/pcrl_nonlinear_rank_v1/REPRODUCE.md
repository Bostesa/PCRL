# REPRODUCE

## Environment

| Component | Version |
|---|---|
| Python | 3.13.7 (`/Users/nathansamson/PCRL/.venv/bin/python`) |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| torch | 2.10.0 |
| joblib | 1.5.3 |
| Machine | Apple CPU, 14 cores, 24 GB. One worker, one BLAS/OpenMP thread throughout. |

Every command below is single-threaded by construction:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

## Worktree

```
cd /Users/nathansamson/PCRL
git worktree add -b research/pcrl-nonlinear-rank-v1 ../PCRL-terminal-a 349efa454afd907389760fd1f59fd8806a215efd
cd ../PCRL-terminal-a
```

## Required local objects (read-only, never refitted)

This worktree is a fresh checkout, so the ignored local artifacts of the completed
studies do not travel with git. `experiments/pcrl_nonlinear_rank_v1/inputs.py`
resolves each one from the first fallback root that has it and records its
sha256 in `EVIDENCE_MANIFEST.json`. Override the search root with
`PCRL_NLR_FALLBACK=<path>`.

| Artifact | Default location |
|---|---|
| Frozen spectral maps, 2018 releases, matrix diagnostics | `results/redesign_20260910_acs_residual_spectral_v1/seed_N/` in the residual-spectral worktree |
| `H` audit ancestors (2018) | same study, `seed_N/H/audits/` |
| PCA32 teacher coordinates, released anchors, pool rows, subset index hashes | `/Users/nathansamson/PCRL/results/redesign_20260909_acs_fixed_predictions_v1/seed_N/` |
| `H` utility heads for the B-view alias | same study, `seed_N/H/fitted/utility/B/` |
| Attacker priors | `/Users/nathansamson/PCRL/results/redesign_20260908_acs_coalition_v1/seed_N/controls/metrics.json` |
| PCA32 / rich-bank / tree-bank reference probes | `/Users/nathansamson/PCRL/results/redesign_20260908_acs_protection_v1/seed_N/metrics.json` |
| Raw 2018 ACS | `/Users/nathansamson/PCRL/data/folktables/2018/1-Year/psam_p06.csv` |
| 2017 partitions (exploratory phase only) | `data/acs_spectral_transport/2017_*.npz` in the residual-spectral worktree |

## Sequence

```
# 0. Falsification fixtures. These run before any new ACS score.
python -m pytest tests/pcrl_nonlinear_rank_v1/ -q

# 1. Rank diagnostic (outcome-free; reads saved eigenvalues only)
python -c "
from experiments.pcrl_nonlinear_rank_v1.rank_diagnostic import diagnose
from experiments.pcrl_nonlinear_rank_v1.inputs import OUT, write_json
write_json(OUT/'RANK_SPECTRUM.json', diagnose())"

# 2. Fit the new maps and build releases (resumable, atomic, ~7 min/seed)
python -m experiments.pcrl_nonlinear_rank_v1.run_fit --seeds 0 1 2

# 3. Mechanism diagnostics: rotation decomposition + nuisance calibration
python -m experiments.pcrl_nonlinear_rank_v1.run_diagnostics --seeds 0 1 2

# 4. 2018 development audits, utility probes and scores (~45 s/unit)
python -m experiments.pcrl_nonlinear_rank_v1.run_dev_2018 --seeds 0 1 2

# 5. Independent prediction replay (separate process; must match bitwise)
python -m experiments.pcrl_nonlinear_rank_v1.run_replay --seeds 0 1 2

# 6. Endpoints, bootstrap intervals, decisions, tables
python -m experiments.pcrl_nonlinear_rank_v1.run_report --seeds 0 1 2
```

Each phase writes a per-unit completion marker and is skipped on resume. Phase 2
short-circuits on `seed_N/fit_complete.json`; phase 4 on
`seed_N/<condition>/complete.json`.

## Determinism

All matrix algebra, eigensolutions, finite moments and verification run in CPU
float64. The random streams are fixed and derived only from the seed and the
rank:

| Quantity | Stream |
|---|---|
| Bandwidth subset | `default_rng(20260918 + 100*seed + r)` |
| Fourier directions and phases | `default_rng(20260919 + 100*seed + r)` |
| Perturbed second start | `default_rng(20260920 + 100*seed + r)` |
| Rotation-invariance probe | `default_rng(20260921 + seed)` |
| Household-cluster bootstrap | `default_rng(20260918)`, 2000 replicates |

Attack, utility and kernel seeds are the historical formulas, unchanged:
role seed `1260000 + 100*seed + 10*view_index + target_index`, kernel seed
`20263910 + 100*seed + 10*view_index + target_index`, utility probe seed
`1250000 + 100*seed + TASKS.index(task)`, utility subsets 2048 at
`1230000 + 100*seed + j`, attacker subsets 4096 at `1240000 + 100*seed + j`.
Phase 4 asserts the subset index hashes equal the historical `indices.json`.

## What is not reproducible from this repository alone

The fitted objects above are local and are not published (`.gitignore` excludes
`maps.joblib`, `releases/`, `fitted/`, `utility_state.joblib` and
`predictions.npz` for this study). Published evidence is the aggregate tables,
diagnostics and manifests. Eigenvector signs are canonicalised, but sign
canonicalisation does not resolve repeated-eigenvalue rotations, so exact
cross-platform coordinate identity is not promised; objectives, losses and
scores are the reproducible quantities.
