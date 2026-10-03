# Quickstart (commands tested 2026-10-03)

Run from the worktree root on branch `research/pcrl-joint-complete-view-method-v1`:

```
T="OMP_NUM_THREADS=1"; PY=~/PCRL/.venv/bin/python; P=results/pcrl_joint_complete_view_method_v1; L=$P/LOCK.json
```

Private inputs and units live in `~/PCRL_eval_cache_private/jcv_v1/`. A verified copy is at `<drive>/private_jcv_v1_20261003/`.

| Step | Command | Expected |
|---|---|---|
| Tests | `env $T $PY -m pytest -q jcv/tests` | 11 pass: projection vs SLSQP, XOR controls, task = S conflict, constant release, signs, seeds, centring, A1 and A2 regressions |
| Lock | `$PY -m jcv.lock verify $L` | `ok: true` |
| Admission (no fit) | `$PY -m jcv.data` | All 39,205 record keys match; writes `DATA_ADMISSION.json` |
| Warm start | `env $T $PY -m jcv.run --lock $L --stage warm --seeds 0 1 2` | About 1 s per seed |
| Train | `env $T $PY -m jcv.run --lock $L --stage train --seeds 0 1 2` | U, E, L/J/JP/S12/S21 × β; about 8 min per seed. Complete units are skipped (a resume takes about 1.5 s). |
| FARE | `env $T $PY -m jcv.run --lock $L --stage fare --seeds 0 1 2` | 36 official trees (FARE environment) |
| Amendment A1 | `env $T $PY -m jcv.run --lock $L --stage amend_a1` | Recomputes outputs from saved heads (idempotent) |
| Inner | `env $T $PY -m jcv.run --lock $L --stage inner --seeds 0 1 2` | 87 inner units, about 2 s each |
| Selection | `env $T $PY -m jcv.run --lock $L --stage select` | `selection.json`, plus F0 at F's budget |
| Outer (after `SELECTION_LOCK.json` is pushed) | `env $T $PY -m jcv.outer --selection-lock $P/SELECTION_LOCK.json --seeds 0 1 2` | 27 outer units, about 1 min each; refuses if the lock is not pushed |
| Controls | `env $T $PY -m jcv.controls --selection-lock $P/SELECTION_LOCK.json` | Nulls ≤ 0.55, planted leaks > 0.75 |
| Inference | `env $T $PY -m jcv.infer --selection-lock $P/SELECTION_LOCK.json` | `PRIMARY_ENDPOINTS.csv` (18), `SECONDARY_ENDPOINTS.csv` (30); about 1 min |
| Tables / figures | `$PY $P/report/report.py inner\|outer\|diagnostics`; `$PY $P/report/figures.py` | CSV tables; `figures/*.pdf` |
| FARE certificates (A3, descriptive) | `env $T $PY $P/report/fare_certificates_a3.py` | `FARE_CERTIFICATES_A3.json` |
| **Deploy** | `env $T $PY -m jcv.deploy --unit nn__s0__J__b0.1 --X <permitted_inputs.npy> --out release.npz` | Releases `v1`, `v2` (18 and 22 dims). Reproduces the saved release bitwise. Refuses any matrix that is not exactly the 83 permitted columns. |
| Backup and restore replay | `env $T $PY $P/report/backup_and_restore.py --dest <drive>` | 2,085/2,085 uncached match. J, L and F restored from the copy reproduce their releases exactly. |
| Independent replay | `env $T $PY $P/verification/replay_jcv.py` | `INDEPENDENT_VERIFICATION.json` (refuses runner imports) |

The deployed model is **EXPERIMENTAL**: its registered advantage was not established.
