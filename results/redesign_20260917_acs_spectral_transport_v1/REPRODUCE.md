# Reproducing the locked 2017 transport study

## Environment

Python 3.13.7, scikit-learn 1.8.0, NumPy 2.4.2, SciPy 1.17.1, PyTorch 2.10.0, pandas 3.0.1, Matplotlib 3.10.8 (`/Users/nathansamson/PCRL/.venv`). Apple CPU, one numerical thread per process, local only; no paid or remote compute. Run from the repository root (the research worktree) with `PYTHONPATH=.`:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

## Required local inputs (not published)

* **Historical 2018 objects** under the original checkout (`HIST_ROOT` in `experiments/acs_spectral_transport_eval.py`, default `/Users/nathansamson/PCRL`):
  * `results/redesign_20260907_acs_transfer_v1/seed_*`: preprocessing, PCA, source encoder and tree;
  * `results/redesign_20260908_acs_protection_v1/seed_*/fitted/transfer`: service heads and reference probes;
  * `results/redesign_20260908_acs_coalition_v1/seed_*/controls`: priors;
  * `results/redesign_20260908_acs_selective_preservation_v1/static`: the E teacher;
  * `results/redesign_20260908_acs_pca16_init_v1`: the initialization;
  * `results/redesign_20260909_acs_fixed_predictions_v1/seed_*`: historical checkpoints, 2018 releases and audits;
  * the 2018 raw PUMS file.
* **Development spectral objects:** `results/redesign_20260910_acs_residual_spectral_v1/seed_*` (maps, audits, utility state).
* **2017 inputs:** `data/acs_spectral_transport/` (the official zip, dictionaries and the five partition NPZ files), recreated by `python -m scripts.audit_acs_spectral_transport --download` (see the development study's `TRANSPORT_ADMISSION.md`).

`TRANSPORT_LOCK.json` lists the SHA256 of every one of these files that the evaluation reads (17,639 files). Historical candidate metadata contain absolute paths. Replay on another machine must restore those paths, or relocate them together with their hash-bound files, and must never substitute refits.

## Commands, in the order executed

```
python -m experiments.run_acs_spectral_transport --phase replay            # 2018 bitwise identity; blocks transport on failure
python -m experiments.run_acs_spectral_transport --phase releases          # 2017 fitting/validation releases
python -m experiments.run_acs_spectral_transport --phase fit --workers 8   # Mode B probes, priors, attackers, selections (resumable)
python -m scripts.dry_run_acs_spectral_transport --dest <scratch>          # optional code exercise on a validation partition
python -m experiments.run_acs_spectral_transport --phase lock              # writes TRANSPORT_LOCK.json (commit and push before scoring)
python -m experiments.run_acs_spectral_transport --phase score --workers 8 # verifies the lock, opens the final partition, Modes A and B
python -m scripts.report_acs_spectral_transport                            # bootstrap, families, decisions, tables, figures
python -m scripts.verify_acs_spectral_transport                            # independent recomputation checks
python -m scripts.summarize_spectral_development                           # development tables from committed CSVs
python -m pytest tests/test_acs_spectral_transport_eval.py tests/test_acs_residual_spectral.py tests/test_acs_spectral_audits.py tests/test_acs_spectral_transport.py -q
```

Every unit writes a completion record with hashes and is skipped on resume. Re-running `--phase lock` refuses to overwrite an existing lock. `--phase score` re-verifies all 17,639 locked hashes and appends to `FINAL_ACCESS_LOG.json`.

## Determinism and what "reproduce" means

All seeds are fixed in code: subset, model, kernel, catch-up and bootstrap (`default_rng(20260917)`) seeds, and withholding uniforms from SHA256 of year-qualified keys. On the same software stack, rerunning `score` and `report` from the locked objects reproduces the published tables exactly. Refitting Mode B from scratch is a new reproduction: prediction values should agree on the same stack, but byte-identical joblib/torch serialization across software versions is not promised.

## Local outputs (kept private; inventoried by hash in the lock and completion records)

* `seed_*/releases_2017/*.npz`: person-level released vectors;
* `seed_*/<interface>/fitted/`: Mode B fitted probes and attackers;
* `seed_*/<interface>/{audit,utility}_selection.json`: pre-final selections;
* `seed_*/<interface>/mode_{A,B}/{metrics.json,predictions.npz}`: final scores and per-person predictions;
* `seed_*/context/`: priors, reference probes, service quality, frozen-nuisance moment diagnostics, final feature support;
* `final_labels.npz`: final labels, weights and identifiers.
