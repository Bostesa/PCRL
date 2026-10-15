# Quickstart: run a protected tree service

Packages are private (they are derived from Adult development people). They live under `.private/task_headroom/deployment/<SERVICE>_S<seed>/` in the research worktree, and in the versioned same-device backup listed in `BACKUP_VERIFICATION.json`. The public archive holds the runtime code only.

```bash
# from the research worktree root; numpy only (no CatBoost, torch or labels needed)
PRIV=.private/task_headroom
python - <<'PY'
import numpy as np
# X: float32 (n, 83) raw features in the admitted order; names: the 83 names (see MANIFEST.json "feature_names")
np.savez('X.npz', X=X, feature_names=names)
PY
PYTHONPATH=. python -m task_headroom.deploy \
  --package $PRIV/deployment/L_TREE_S0 \
  --manifest-sha256 <MANIFEST sha256 from DEPLOYMENT_RESULTS.json> \
  --input X.npz --input-sha256 $(shasum -a 256 X.npz | cut -c1-64) \
  --output released.npz
```

- **Output:** `p_income` (n,2), `decision_income` (n,), `p_occupation` (n,6), `decision_occupation` (n,). Nothing else.
- **Refusals:** the runtime refuses each of these (13 checks per package, all passing):
  - a wrong manifest or input hash;
  - missing, reordered, renamed or extra columns (including a SEX column);
  - any label or identity array in the input;
  - nonfinite or integer matrices;
  - `--emit leaves` / `--emit teacher`;
  - a tampered package file;
  - an existing output.
- **Packages:** `T_TREE` (teacher), `L_TREE` (primary protected service), `S12_TREE` (strongest matched control: lowest reported INNER coalition recovery among the point-feasible matched controls, with infeasible S21 excluded; this packaging choice was made post hoc on INNER and affects no endpoint) and `J_TREE` (joint arm), each for seeds 0 and 1. Their manifest hashes are in `DEPLOYMENT_RESULTS.json`.

**Reproducing the campaign:** the queue specs and configs are in `.private/task_headroom/root/` (private), and the workers are in `task_headroom/`. Every numerical step ran through `task_headroom.scheduler` (one thread, RSS-capped). CatBoost 1.2.8 is needed only to refit teachers; the wheel hash is in `SCIENCE_LOCK.json`.
