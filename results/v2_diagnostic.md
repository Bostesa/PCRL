# V2 Validation — Diagnostic (Step 1)

Reads what trajectory data is recoverable from the failed Adult and HMDA
runs (results/v2_adult/, results/v2_hmda/) and decides between two
hypotheses for the 0/8 + 0/6 compliance failure:

- **H1 (metric mismatch):** the trainer satisfied its HSIC constraint
  (kernel-space independence) but the auditor's metric (linear R²)
  disagreed.
- **H2 (optimisation failure):** lambdas oscillated or HSIC never dropped,
  so the constraint was never satisfied.

## Data available

V2Trainer's `_update_history` only stores per-epoch *aggregated* scalars
(`loss`, `task_loss`, `vicreg_loss`, `verify_loss`, `hsic_mean` (mean
across pairs), `vclub_q_loss`). Per-pair HSIC and per-constraint λ
trajectories live in `Constraint.value_history` / `Constraint.lambda_history`
in memory but are **not** serialised. Checkpoints contain the aggregated
`history` dict but the orchestrator's tarball pulled only `summary.json`
+ `per_seed_results.json`, not `checkpoints/v2_<dataset>/*.pt`.

So full per-epoch HSIC vs λ curves are **not recoverable** without
re-running. Working with terminal state instead — that is sufficient to
distinguish H1 from H2 because of how proxy-Lagrangian dynamics work.

## Proxy-Lagrangian terminal-state argument

For HSIC ≤ 0.05 with direction "<=":

    λ_{t+1} = max(0, λ_t + η * (HSIC_t - 0.05))

If `HSIC_t < 0.05` consistently in late training, λ decays toward 0 and
floors there. **λ → 0 is the unique equilibrium of constraint
satisfaction.** A non-zero λ at convergence means HSIC was still
violating threshold. Conversely, λ ≈ 0 cannot be reached via oscillation
without the constraint being repeatedly satisfied (dual ascent is
monotone in violation; clipping only at zero).

So the final-λ table is enough:

- `λ_final ≈ 0` ⇒ HSIC ≤ 0.05 was achieved.
- `λ_final >> 0` ⇒ HSIC > 0.05 still violating.
- Compare with `linear_r2` from the audit:
  - λ ≈ 0 AND linear_r2 ≫ 0.05 ⇒ **metric mismatch (H1)**.
  - λ ≫ 0 AND linear_r2 ≫ 0.05 ⇒ inconclusive on its own (could be H2 or
    just slow convergence).

## Adult — final λ vs linear R²

| pair | λ s0 | λ s1 | λ s2 | linear R² (s0/s1/s2) | adj_pass |
|---|---|---|---|---|---|
| income_prediction__race | 1.83 | 0.59 | 0.87 | 0.40 / 0.34 / 0.37 | ✗ |
| income_prediction__sex | 0.76 | 1.01 | 1.02 | 0.44 / 0.60 / 0.53 | ✗ |
| employment_analysis__race | 1.90 | 1.80 | 2.00 | 0.37 / 0.32 / 0.36 | ✗ |
| employment_analysis__age_group | **0.003** | 0.39 | 0.11 | 0.11 / 0.16 / 0.14 | ✗ |
| **employment_analysis__marital_status** | **0.0002** | **0.000** | **0.002** | **0.26 / 0.43 / 0.29** | ✗ |
| education_assessment__sex | 0.40 | 0.44 | 0.61 | 0.31 / 0.36 / 0.40 | ✗ |
| education_assessment__race | 1.66 | 1.62 | 1.00 | 0.27 / 0.34 / 0.30 | ✗ |
| education_assessment__income | 0.64 | 0.62 | 0.72 | 0.18 / 0.19 / 0.19 | ✗ |

**Smoking gun:** `employment_analysis__marital_status`. λ_final is
essentially zero (max 0.002) on **all 3 seeds** ⇒ HSIC was driven below
0.05 reproducibly. Yet linear R² = 0.26 / 0.43 / 0.29 and
empirical_best_acc = 0.96 vs 0.52 majority (delta 0.44pp). The trainer
satisfied its constraint while the audit metric continued to expose
44 percentage points of marital-status leakage.

**Secondary evidence:** `employment_analysis__age_group` seed 0 has
λ = 0.003 (≈ 0), linear R² = 0.11. Same pattern, less extreme.

## HMDA — final λ vs linear R²

| pair | λ s0 | λ s1 | λ s2 | linear R² (s0/s1/s2) | adj_pass |
|---|---|---|---|---|---|
| underwriting__race | 1.03 | 1.11 | 1.32 | 0.50 / 0.52 / 0.56 | ✗ |
| underwriting__ethnicity | 0.58 | 0.64 | 0.44 | 0.48 / 0.60 / 0.58 | ✗ |
| pricing_analysis__race | 0.81 | 0.89 | 0.76 | 0.40 / 0.49 / 0.49 | ✗ |
| pricing_analysis__sex | 0.14 | 0.18 | 0.20 | 0.46 / 0.47 / 0.55 | ✗ |
| fair_lending_audit__race | 1.02 | 1.05 | 1.18 | 0.58 / 0.53 / 0.64 | ✗ |
| fair_lending_audit__sex | 0.28 | 0.33 | 0.25 | 0.47 / 0.48 / 0.79 | ✗ |

No λ collapsed to ≈ 0 on HMDA. All λ stayed in [0.14, 1.32]. So HMDA is
*also* failing the HSIC constraint — H2 partially applies (HSIC didn't
drop). But early stopping triggered at `last_epoch=23` for seeds 0+1
(only 57 for seed 2). With 3-purpose forward passes and a 91K-row
dataset on a T4, 23 epochs is short — the dual variables hadn't
converged. This is consistent with both H1 and H2.

The Adult `marital_status` pair is the cleaner signal because it
*reproducibly* satisfies HSIC (λ floors to 0) but *reproducibly* fails
the linear-R² audit. That can only be metric mismatch.

## Verdict

**H1 (metric mismatch) confirmed.** Even when the v2 trainer's HSIC
constraint is satisfied (λ → 0 across all seeds), the auditor's linear
R² remains at 0.26-0.43 for the same pair. nHSIC ≤ 0.05 in the kernel
space we use (Gaussian + median heuristic for continuous, delta kernel
for categorical) does **not** imply linear R² ≤ 0.05 in the embedding
space. The two metrics disagree, and we are training against the wrong
one.

H2 is also partially in play on HMDA (more time / dual tuning needed),
but H1 is sufficient to explain the failure pattern and is the higher-
priority fix.

## Decision

Proceed to **Step 2 (Option A)**: replace the HSIC constraint in
`pcrl/training/v2_trainer.py` with a linear-R² constraint computed by
`VerificationRegularizer` (which already does differentiable OLS via
`torch.linalg.solve`). Threshold 0.05 to match the auditor exactly.
Keep HSIC as a fixed-weight auxiliary loss term (`lambda_hsic_aux=0.1`)
so we still get a non-linear independence signal but it does not drive
the dual variables. vCLUB and VICReg unchanged.

This aligns the optimisation target with the audit metric. After
re-running on Adult, if linear R² truly drops below 0.05 the auditor's
delta will follow, and the adj_pass count should track v1's 6/8.

## Logging gap (note for future)

Per-pair HSIC and per-constraint λ trajectories should be persisted in
the run artifacts. Currently `Constraint.value_history` /
`Constraint.lambda_history` exist but are not serialised. Suggest adding
them to `state.history` (or a sibling dict) so future diagnostics don't
require re-running. Out of scope for this fix but worth a follow-up.
