# Quickstart (commands tested 2026-10-04)

Run from the worktree root on `research/pcrl-penalty-no-erasure-v1`. First set:

```
T="OMP_NUM_THREADS=1"; PY=~/PCRL/.venv/bin/python; P=results/pcrl_penalty_no_erasure_v1; L=$P/LOCK.json
```

Private data live in `~/PCRL_eval_cache_private/pnx_v1/`; predecessor inputs and references in `~/PCRL_eval_cache_private/jcv_v1/`. Drive copies are `<drive>/private_pnx_v1_20261003` and `<drive>/private_jcv_v1_20261003`.

| Step | Command | Expected |
|---|---|---|
| Tests | `env $T $PY -m pytest -q pnx/tests` | 6 pass: β = 0 parity fixture, copied-loop fidelity, specs, aliases, complete bank records, critic snapshot |
| Lock | `$PY -m pnx.lock verify $L` | `ok: true` |
| Parity (engineering) | `env $T $PY -m pnx.run --lock $L --stage parity` | All pass: U environment, PN/LN β = 0 bitwise, JP replication (about 95 s) |
| Aliases | `env $T $PY -m pnx.run --lock $L --stage alias` | 117 hash-checked alias records |
| Train | `env $T $PY -m pnx.run --lock $L --stage train --seeds 0 1 2` | 18 PN/LN units (about 3 min on 2 workers; complete units are skipped) |
| Inner / select | `env $T $PY -m pnx.run --lock $L --stage inner`, then `--stage select` | `selection.json` |
| Outer (after `SELECTION_LOCK.json` is pushed) | `env $T $PY -m pnx.outer --selection-lock $P/SELECTION_LOCK.json` | 39 units; refuses unless the lock is pushed |
| Controls | `env $T $PY -m pnx.controls --selection-lock $P/SELECTION_LOCK.json` | Nulls ≤ 0.55, planted leaks > 0.75 |
| Critic gap | `env $T $PY -m pnx.critic_gap --lock $L` | 18 `critic__` units |
| Inference | `env $T $PY -m pnx.infer --selection-lock $P/SELECTION_LOCK.json` | `PRIMARY_ENDPOINTS.csv` (18), `SECONDARY_ENDPOINTS.csv` (30) |
| Reports and figure | `$PY $P/report/report.py` | CSV tables and `figures/fig_erasure_on_off.*` |
| **Deploy** | `env $T $PY -m pnx.deploy --unit pn__s1__PN__b0.1 --X <permitted_inputs.npy> --out release.npz` | Identity-map release, bitwise equal to the saved one. Refuses non-83-column input. **EXPERIMENTAL_NO_ADVANTAGE**. |
| Backup and restore | `env $T $PY $P/report/backup_and_restore.py --dest <drive>` | 561/561 uncached match; PN, LN and E restored from the drive reproduce their releases exactly |
| Independent replay | `env $T $PY $P/verification/replay_pnx.py` | `INDEPENDENT_VERIFICATION.json` |
