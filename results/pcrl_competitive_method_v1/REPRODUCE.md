# REPRODUCE — `pcrl_competitive_method_v1`

Branch `research/pcrl-competitive-method-v1`. Python 3.13 with the repository `.venv`
(`torch 2.10.0`, `numpy`, `scipy 1.17.1`, `sklearn 1.8.0`; exact versions in
`DEPENDENCY_MANIFEST.json`). Every stage is single-threaded
(`OMP_NUM_THREADS=1` etc.), resumable through completion markers, and never
overwrites a completed unit. The frozen 2018/2017 inputs (the `redesign_*` result
trees, ACS extracts) are local, unpublished artifacts resolved by the predecessor
registries; raw survey rows are never published.

```
PY=/path/to/PCRL/.venv/bin/python
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 0. diagnosis of the predecessor's stored checkpoints (no refit)
$PY -m experiments.pcrl_competitive_method_v1.diagnose
$PY -m experiments.pcrl_competitive_method_v1.write_diagnosis

# 1. fixtures, lock, drift check
$PY -m pytest -q tests/pcrl_competitive_method_v1
$PY -m experiments.pcrl_competitive_method_v1.freeze --verify

# 2. fits and 2018 audits (Track E, Track N, references; one global priority queue)
$PY -m experiments.pcrl_competitive_method_v1.run_all --worker 0

# 3. panel nomination (validation split only), then 2018 report and validation
$PY -m experiments.pcrl_competitive_method_v1.panel
$PY -c "from experiments.pcrl_competitive_method_v1 import aggregate as a; a.track_n(); a.track_e(); a.counts()"
$PY -c "from experiments.pcrl_competitive_method_v1 import report; report.run()"

# 4. stronger attack and 2017 exploratory transport, per anchor
$PY -m experiments.pcrl_competitive_method_v1.stress --seed S --conditions <panel + controls + comparators>
$PY -m experiments.pcrl_competitive_method_v1.transport_2017 --seed S --units <panel + controls>

# 5. documents
$PY -m experiments.pcrl_competitive_method_v1.write_docs
```

Resume: rerunning any stage skips completed units by marker; a claimed-but-incomplete
audit unit is re-queued by deleting its `claims/<seed>__<unit>` directory.
