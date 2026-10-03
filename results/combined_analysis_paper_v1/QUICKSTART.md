# Quickstart

**Where to run.** From the worktree root, on branch `research/combined-analysis-paper-v1`.

**Variables:**

```
T="OMP_NUM_THREADS=1"
PY=~/PCRL/.venv/bin/python
P=results/combined_analysis_paper_v1
```

Private inputs live in `~/PCRL_eval_cache_private/{bench_v1,oar_v1,odx_v1,cap_v1}`. A restore from `<drive>` is described in
`PRIVATE_BACKUP_INDEX.json`.

| Step | Command | Expected |
|---|---|---|
| Tests | `env $T $PY -m pytest -q cap/tests odx/tests oar/tests` | 32 pass |
| Lock check | `$PY -m cap.lock verify $P/LOCK.json` | `ok: true` |
| Planned units (no fits) | `$PY -m cap.plan` | 184 units: 48 aliases, 48 + 36 attack, 48 banks, 4 controls |
| Fits (done; resumable) | `env $T $PY -m cap.run --lock $P/LOCK.json --stage fits` | 180 unit directories, about 10 min, peak 0.5 GB |
| Controls | `env $T $PY -m cap.run --lock $P/LOCK.json --stage controls` | nulls 0.48–0.51, planted 0.90–0.93 |
| Inference | `env $T $PY -m cap.infer` | writes `ACTUAL_HEAD_UTILITY.csv`, `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` (19) and `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv` (33); about 40 s |
| Post-hoc diagnostics | `env $T $PY $P/report/posthoc_diagnostics.py` | `report/posthoc_diagnostics.json` (labelled post hoc) |
| Stage A recomputation | `env $T $PY $P/report/stage_a_recompute.py` | `report/corrections_recomputed.json` |
| Independent replay | `env $T $PY $P/verification/replay_cap.py` | `INDEPENDENT_VERIFICATION.json` (refuses runner imports) |
| Figures | `$PY $P/report/figures.py` | `papers/combined_empirical_v1/figures/*.pdf` |
| Manuscript | `cd papers/combined_empirical_v1 && latexmk -pdf main.tex` | `main.pdf`, 10 pages |
| Backup and restore replay | `env $T $PY $P/report/backup_and_restore.py --dest <drive>` | uncached read-back all match; 4 attacker replays from the copy, max diff 0 |

**Re-running.** Completed units are skipped by hash, so re-running the fits is safe. Any change to locked code makes the
runner refuse to start.
