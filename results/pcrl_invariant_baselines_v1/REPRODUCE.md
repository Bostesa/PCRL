# REPRODUCE

Every stage is resumable: a completed unit writes an atomic marker and is reused, never
refitted. A cache read is never counted as a new fit.

## Environment

```
interpreter : /Users/nathansamson/PCRL/.venv/bin/python   (3.13)
numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, torch 2.10.0, concept-erasure 0.2.4
```

The worktree has **no venv of its own**; `concept_erasure` is only importable through
the interpreter above. Always run with:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=.
```

**One fitting worker throughout.** Audit suites are never run concurrently with a model
job. The machine was at swap capacity for the whole run (see `RUN_STATUS.md`).

## Checkout

```
git worktree add /Users/nathansamson/PCRL-terminal-1-invariant \
    -b research/pcrl-invariant-baselines-v1 c37807e4f568ef38e5528fc09c1506083278bf4d
cd /Users/nathansamson/PCRL-terminal-1-invariant
```

## Read-only inputs

Ignored artifacts of the completed studies do not travel with git. They are resolved
read-only through `experiments/pcrl_nonlinear_rank_v1/inputs.py:FALLBACK_ROOTS`, in
order, and hashed into each stage's `inputs` block. Nothing historical is refitted,
moved or modified.

| Input | Resolved from |
|---|---|
| Frozen 2018 `SpectralModel` (V, U, bases, 15 OOF nuisances) | `~/.config/superpowers/worktrees/PCRL/residual-spectral-20260910/` |
| Pools, PCA teacher, anchors, subset indices, frozen `H/E/A0/L025/L20/J` | `~/PCRL/results/redesign_20260909_acs_fixed_predictions_v1/` |
| Historical `spectral_*` scored units | `~/.config/superpowers/.../results/redesign_20260910_acs_residual_spectral_v1/` |
| Predecessor `spectral_lin8_*`, `spectral_nlr*_*` scored units | `~/PCRL-terminal-a/results/pcrl_nonlinear_rank_v1/` |
| 2017 partitions | `~/.config/superpowers/.../data/acs_spectral_transport/` |
| 2018 raw ACS | `~/PCRL/data/folktables/2018/1-Year/psam_p06.csv` |

Set `PCRL_NLR_FALLBACK` to override the second root.

## Stages, in order

```bash
P=/Users/nathansamson/PCRL/.venv/bin/python

# 3. Validation gate. Must pass before any fitting is trusted.
$P -m pytest tests/pcrl_invariant_baselines_v1 -q            # 40 fixtures, ~10 s

# 4. Fit the 18 repaired conditions (3 policies x 2 ranks x 3 seeds).
$P -u -m experiments.pcrl_invariant_baselines_v1.run_fit --seeds 0 1 2
#    949 s, peak RSS 369 MB. Raises if the closed-form replay is not bitwise.

# 4b. The outcome-free mechanism gate + the cross-family objective matrix.
#     MUST be run before any 2018 or 2017 score is read.
$P -u -m experiments.pcrl_invariant_baselines_v1.run_diagnostics

# 5a. SARL alias audit (fits nothing).
$P -u -m experiments.pcrl_invariant_baselines_v1.alias_audit

# 5b/5c. LEACE and SPLINCE on the A0 auxiliary channel.
$P -u -m experiments.pcrl_invariant_baselines_v1.erasure_baselines

# 5d. OptNet-ARL. Budget was set by a training-only probe (RUN_STATUS amendment 2):
$P -u -m experiments.pcrl_invariant_baselines_v1.optnet_arl --calibrate   # optional
$P -u -m experiments.pcrl_invariant_baselines_v1.optnet_arl --budget 1200

# 6. 2018 development evaluation of the new wires only.
$P -u -m experiments.pcrl_invariant_baselines_v1.run_dev_2018

# 8. Endpoints, bootstrap, decisions.
$P -u -m experiments.pcrl_invariant_baselines_v1.run_report
```

## Resume points

| Marker | Effect if present |
|---|---|
| `seed_N/fit_complete.json` | the seed's 6 repaired fits are reused |
| `seed_N/releases/<arm>/releases.npz` | that arm's release is reused |
| `seed_N/optnet_complete.json` | the seed's 3 OptNet fits are reused |
| `seed_N/<arm>/complete.json` | that arm's 2018 audit unit is reused |
| `seed_N/<arm>/utility_state.joblib` | the utility probes are reused |

Deleting a marker forces exactly that unit to recompute. Nothing else is touched.

## Determinism

Every seed is a frozen formula, all hashed into `PROTOCOL_FREEZE.json`:

| Object | Formula |
|---|---|
| kernel subset | `20260930 + 100*seed + 10*rank + role_index` |
| bandwidth subset | `20260918 + 100*seed + rank` |
| perturbed start | `20260920 + 100*seed + rank` |
| OptNet encoder init | `20260940 + 100*seed + 10*start` |
| OptNet batches | `20260941 + 100*seed + 10*start` |
| bootstrap | `20260918`, 2000 replicates, cohort SERIALNO clusters |
| attacker role / kernel seeds | historical: `1260000 + 100*seed + 10*view + target`, `20263910 + ...` |

## Verifying the two load-bearing claims independently

**The repair is exact, not approximate.** On real ACS data at rank 16:

```bash
$P - <<'PY'
import numpy as np
from experiments.pcrl_nonlinear_rank_v1.inputs import Registry, load_representation_labels
from experiments.pcrl_nonlinear_rank_v1.maps import build_roles
from experiments.pcrl_invariant_baselines_v1.maps import build_rank_context, make_objective
from experiments.pcrl_invariant_baselines_v1.objective import FAMILY_INVARIANT
reg = Registry.new(); state = build_roles(0, reg)
labels, _ = load_representation_labels(0, reg)
ctx = build_rank_context(state, 16, 0, labels)
obj = make_objective(state, ctx, FAMILY_INVARIANT, 'C1'); w = ctx.w_reference
q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(16, 16)))
print('|L(WQ) - L(W)| =', abs(obj.loss(w @ q) - obj.loss(w)))   # 1.1e-16
PY
```

**The chain to the historical baseline is bitwise.** `seed_N/closed_form_replay.json`
reports `all_bitwise: true` with `max_abs_difference: 0.0` for
`spectral_lin{8,16}_{L1,L2,C1}`; the rank-16 arms alias the historical
`spectral_L1/L2/C1`. The fit **raises** rather than continuing if this fails.

## What cannot be reproduced from this worktree alone

* The ignored artifacts above must exist at one of the fallback roots.
* `results/pcrl_invariant_baselines_v1/**/releases.npz`, `predictions.npz`,
  `utility_state.joblib` and `audits/` are **not committed** (171 MB); only compact
  aggregate evidence is. Re-running the stages regenerates them.
* **2016 is not touched by any command above**, and no command in this study can reach
  it.
