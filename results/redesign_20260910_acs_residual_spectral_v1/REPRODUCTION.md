# Reproduction and local objects

This is a development comparison on the historical 2018 cohort. Public evidence contains code, aggregate scores, diagnostics and SHA256 inventories. Exact object replay additionally requires the original local raw ACS file and fitted artifacts; no model fitting is implied by reproducing tables. Original source models, preprocessing, split rows, released vectors and historical attacks are not refitted.

Run from the publication checkout in an environment matching `requirements-redesign.txt`. The executed interpreter was `/Users/nathansamson/PCRL/.venv/bin/python` (Apple CPU, one numerical thread). Set `PYTHONPATH` to the checkout when using another entrypoint. Set `--historical-root` to the original evidence checkout. Historical candidate metadata contain absolute local model paths; exact replay on a different machine must restore or explicitly relocate those paths together with their hash-bound files. Do not silently substitute refits.

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m pytest tests/test_acs_residual_spectral.py tests/test_acs_spectral_audits.py tests/test_acs_spectral_transport.py tests/test_acs_spectral_reporting.py tests/test_acs_spectral_replay.py tests/test_acs_spectral_scoring.py -q
python -m scripts.replay_acs_residual_spectral --report results/redesign_20260910_acs_residual_spectral_v1/REPLAY_REPRODUCTION.json
python -m scripts.report_acs_residual_spectral
python -m scripts.package_acs_residual_spectral
```

The independent replay reconstructs target masks/weights from source labels and independently checks saved probabilities, selection, legal projections and comparisons. Full report regeneration requires the local predictions, labels, matrix diagnostics and withholding arrays; it consumes already frozen results and does not fit models. Public-only readers can inspect the exported aggregate CSVs and verify their integrity with `python -m scripts.package_acs_residual_spectral --verify`; the public package alone cannot regenerate the full report. See each script's `--help` for output relocation and explicit partial-debugging options; partial outputs are never the complete scientific result.

To reproduce **new fits** in a new empty output directory, copy this study's prospective `PROTOCOL.md` into it and retain all historical evidence at its verified paths. Then:

```
python -m experiments.run_acs_residual_spectral --out results/spectral_reproduction --historical-root /path/to/original/PCRL --phase fit
python -m experiments.run_acs_residual_spectral --out results/spectral_reproduction --historical-root /path/to/original/PCRL --phase evaluate --seeds 0 1 2
```

The first command fits exactly eight maps per seed from original representation-fit rows. The second requires their global freeze, preserves complete units, and fits only new A/AB independent attacks, kernel controls and A utility probes. It reuses H's B fits. A new fitting run is not asserted to create byte-identical joblib/torch serialization across software versions; compare prediction values, mathematical diagnostics and separately recorded hashes.

Local artifacts include each seed's `maps.joblib`, release arrays, all fitted audit/utility objects, candidate predictions, aligned labels/weights, detailed matrix diagnostics, source row/household manifests and hashes. Public exports omit person arrays and fitted vectors. `OPERATIONAL_AMENDMENTS.json` distinguishes evaluation-loader/recovery changes from the immutable mathematical maps. Local `RUNTIME_EVENTS.json` records measured process time, including exceptions; public `RUNTIME.json` summarizes it; unit completion records contain actual counts and elapsed times. Runtime from `.git` birthtime excludes the brief initial inspection before worktree creation and is labeled accordingly.

Future-source admission is separate. Public Census downloads are explicitly requested with `python -m scripts.audit_acs_spectral_transport --download`; no trained model is invoked. Replaying its admission requires `--historical-root` pointing to the original pinned 482f14d evidence checkout and the exact original local inventory, as documented by the transport script. There is no frozen-inventory-only replay mode. A fresh admission after later data use must rescan and review that use. The saved 2017 final partition has never been scored in this study. Admission does not establish distinct people across years or confirm the mechanism.
