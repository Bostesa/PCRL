# Quickstart: SCM engineering benchmark

Run from the repository root:

```
PYTHONPATH=.:<source private pylib>
```

The environment is Python 3.13, numpy 2.4, scipy 1.17, scikit-learn 1.8, torch 2.10 (for the neural reference only), and CatBoost 1.2.8 from the source study's private library. Every numerical unit runs through the queue:

```
python -m scm_bench.jobqueue run --unit <ID> --stage <stage> --rss-gib <r> -- <command>
```

The queue allows two slots, one thread per job, and enforces an 8 CPU-hour and 6-hour budget.

## Reproduce (development roles only; the sealed assessment is never exported)

1. `python -m scm_bench.restore`: copies the admitted role exports and frozen comparator outputs, with hash checks against the public records.
2. `python -m scm_bench.admit`: records the stopped campaign's artifacts by hash.
3. `python -m scm_bench.make_lock`: writes the pre-fit ANALYSIS_LOCK. Commit it.
4. `python -m scm_bench.fit_scm --seed {0,1} --out .private/pcrl_scm_engineering_benchmark_v1/runs/FIT_{s}`: teachers, the fine partition, the 31-release utility bank and its outputs.
5. `python -m scm_bench.locks training`: commit TRAINING_COMPLETE_LOCK.json and FIT_SUMMARY.json.
6. `python -m scm_bench.select_scm --out results/pcrl_scm_engineering_benchmark_v1/SELECTION_LOCK.json`: utility clauses, cross-fitted proxies and nominations on MODEL_SELECT. Commit the output.
7. `python -m scm_bench.bank --out results/pcrl_scm_engineering_benchmark_v1/SERVICE_BANK_MANIFEST.json`
8. `python -m scm_bench.readers --seed {0,1} --out .private/pcrl_scm_engineering_benchmark_v1/runs/READERS_{s}`
9. `python -m scm_bench.locks reporting`: commit REPORTING_LOCK.json and READERS_SUMMARY.json.
10. `python -m scm_bench.report --out results/pcrl_scm_engineering_benchmark_v1/report`: the one logged re-read of DEV_REPORT truth.
11. `python -m scm_bench.deploy_scm --out-root .private/pcrl_scm_engineering_benchmark_v1/deploy --report results/pcrl_scm_engineering_benchmark_v1/DEPLOYMENT_RESULTS.json`

## Deploy a frozen SCM package on raw ACS fields (no labels)

The input is a CSV whose header is exactly `AGEP,WKHP,COW,SCHL,MAR,POBP`, in that order, coded as in the 2019 PUMS dictionary.

```
python -c "from scm_bench import deploy_scm as D; from acs_replication import deploy as S; import numpy as np; \
r = D.release('<package dir>', S.read_raw_csv('people.csv')); \
np.savez('released.npz', **{f'{t}_{k}': v for t, o in r.items() for k, v in o.items()})"
```

The output holds `income_probability` (n × 2), `income_decision`, `occupation_probability` (n × 5) and `occupation_decision`. In service, each task's output goes only to its own recipient.

**Refusals:**
- reordered or extra columns;
- any forbidden column (SEX, for example);
- any request for hidden outputs (cell or group ids, teacher probabilities, propensity).

## Tests (invented data only)

```
python -m scm_bench.test_engine_synthetic
python -m scm_bench.test_proxy_synthetic
python -m structural_pilot.test_lookup_synthetic
```
