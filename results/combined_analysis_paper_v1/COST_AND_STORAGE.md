# Cost and storage

Everything ran on the laptop. No cloud resource was launched.

## Time and compute

| Item | Value |
|---|---|
| Assignment start | 2026-10-03 17:44:55Z (`~/PCRL_eval_cache_private/cap_v1/START.txt`) |
| Lock pushed (before any fit) | 17:53:29Z (`a0de449`) |
| Fits and controls | 17:53:39Z – 18:03:16Z: 9 min 38 s elapsed; fits 501 s user + 73 s sys; controls 40 s |
| Inference | 41 s elapsed, 32 s user |
| Independent replay | about 65 s (replays all 528 saved attacker models) |
| Post-hoc diagnostics, Stage A recomputation, figures, backup | about 3 min in total |
| **Total elapsed (start → backup)** | **about 40 min**, of which about 10 min was compute |
| **Total CPU** | **about 0.25 CPU-h**, against a ceiling of 12 |
| Model fits | 3,924 attacker fits in 84 new units, plus the 4 control slates. There were no new defense, head or encoder fits. |
| Workers | at most 2 concurrently (fits + controls), `OMP_NUM_THREADS=1` |
| Peak RSS | 0.71 GB (inference), against a ceiling of 6 GiB |
| Free disk | 125 GiB after the run, against a floor of 5 GiB |

**Agents.** Four read-only or write-limited agents were used:
- a claims and literature reviewer;
- the evidence-map builder;
- the independent verifier;
- a manuscript reviewer.

None fitted a model.

## Storage

| Location | Content | Size | Files |
|---|---|---|---|
| `~/PCRL_eval_cache_private/cap_v1` (private) | 180 unit directories (per-person predictions, models), ledger, events, logs, controls, `inference.json` | 112 MB | 955 |
| `~/PCRL_eval_cache_private/backups/private_cap_v1_20261003` (private, versioned copy) | identical copy, `SHA256SUMS`, `BACKUP_RECORD.json` | 112 MB | 955 + 2 |
| `<drive>/private_cap_v1_20261003` | **PENDING**: the drive was not mounted; the single command is in `PRIVATE_BACKUP_INDEX.json` | | |
| Repository (public) | aggregate tables, code, tests, lock, replay script, figures, manuscript (PDF about 0.3 MB) | small | |

**Nothing was deleted.** No original models, results, other terminals' files or caches were touched. The study created no disposable scratch files beyond the rendered page images in the session scratchpad.
