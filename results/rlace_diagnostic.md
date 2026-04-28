# R-LACE / LEACE diagnostic — Adult

_Generated 2026-04-28T00:41:27Z on `mps` (seed=0, N_test=15060, backbone epochs=50)_

**Question.** Is the linear R²(z, A) < 0.05 constraint achievable on v2's frozen `StandardEncoder` backbone features for Adult, or does the frozen backbone carry nonlinear entanglement that no linear projection can fix?

**Method.** Train a `StandardEncoder` (hidden=[128,128] → 64) on Adult's task labels (no fairness constraint, 50 epochs). For each (purpose, disallowed_attr) pair: apply LEACE (Belrose et al. 2023, closed-form population-optimal) and an INLP/R-LACE-style iterative orthogonal nullspace projection at rank r ∈ {1, 4, 8}. Measure linear R² of the optimal Tikhonov-regularized predictor (matching the auditor), task accuracy preserved (purpose's trained task head), and a 2×64 MLP adversary's accuracy on the erased representation.

**Note on the backbone.** v2 freezes the `StandardEncoder` at random initialization and trains only the per-purpose LoRA adapters. Here we task-train the backbone for a stronger test: if even task-trained features (which entangle attrs with task signal more strongly than random init) are linearly erasable, then v2's stalling is an optimization issue, not a representational wall.

## Per-pair results

| Purpose / Attr | base R² | LEACE R² | R-LACE r=1 | R-LACE r=4 | R-LACE r=8 | best | task acc base→best | MLP base→best | verdict |
|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| income_prediction / race | 0.326 | 0.009 | 0.285 | 0.205 | 0.157 | LEACE (0.009) | 0.851 → 0.836 (+1.48pp) | 0.967 → 0.951 | **GREEN** |
| income_prediction / sex | 0.607 | 0.007 | 0.511 | 0.376 | 0.325 | LEACE (0.007) | 0.851 → 0.818 (+3.33pp) | 0.980 → 0.973 | **RED** |
| employment_analysis / race | 0.326 | 0.009 | 0.285 | 0.205 | 0.157 | LEACE (0.009) | 1.000 → 1.000 (+0.01pp) | 0.967 → 0.951 | **GREEN** |
| employment_analysis / age_group | 0.321 | 0.005 | 0.296 | 0.250 | 0.222 | LEACE (0.005) | 1.000 → 1.000 (+0.00pp) | 0.856 → 0.844 | **GREEN** |
| employment_analysis / marital_status | 0.801 | 0.004 | 0.734 | 0.659 | 0.625 | LEACE (0.004) | 1.000 → 1.000 (+0.00pp) | 0.984 → 0.983 | **GREEN** |
| education_assessment / sex | 0.607 | 0.007 | 0.511 | 0.376 | 0.325 | LEACE (0.007) | 1.000 → 1.000 (+0.00pp) | 0.980 → 0.973 | **GREEN** |
| education_assessment / race | 0.326 | 0.009 | 0.285 | 0.205 | 0.157 | LEACE (0.009) | 1.000 → 0.999 (+0.09pp) | 0.967 → 0.951 | **GREEN** |
| education_assessment / income | 0.433 | 0.006 | 0.433 | 0.431 | 0.414 | LEACE (0.006) | 1.000 → 0.996 (+0.42pp) | 0.851 → 0.850 | **GREEN** |

## Verdict counts

- GREEN: 7/8
- YELLOW: 0/8
- RED: 1/8
- **Overall: GREEN**

## Comparison to v2 Option A

v2 Adult Option A (3 seeds × 200 epochs, R² constraint, LoRA rank 8) achieved 0/8 pass with mean linear R² (across seeds) per pair:

| Pair | v2 Option A R² (mean) | Closed-form best R² | gap |
|---|---:|---:|---:|
| income_prediction/race | 0.231 | 0.009 | +0.222 |
| income_prediction/sex | 0.328 | 0.007 | +0.321 |
| employment_analysis/race | 0.143 | 0.009 | +0.134 |
| employment_analysis/age_group | 0.114 | 0.005 | +0.109 |
| employment_analysis/marital_status | 0.186 | 0.004 | +0.182 |
| education_assessment/sex | 0.193 | 0.007 | +0.186 |
| education_assessment/race | 0.137 | 0.009 | +0.128 |
| education_assessment/income | 0.125 | 0.006 | +0.119 |

## Train-vs-test LEACE R² (distribution-shift diagnostic)

LEACE makes the cross-covariance Σ_{Pz,Z} exactly zero on the **training** set by construction (residual R² ≈ 0 in expectation). Any non-zero R² on the test set is therefore an **out-of-sample** artifact — a finite-sample + distribution-shift floor, not a structural representational wall.

| Pair | LEACE train R² | LEACE test R² | shift |
|---|---:|---:|---:|
| income_prediction/race | 0.0000 | 0.0093 | +0.0093 |
| income_prediction/sex | 0.0000 | 0.0073 | +0.0073 |
| employment_analysis/race | 0.0000 | 0.0093 | +0.0093 |
| employment_analysis/age_group | 0.0000 | 0.0049 | +0.0049 |
| employment_analysis/marital_status | 0.0000 | 0.0042 | +0.0042 |
| education_assessment/sex | 0.0000 | 0.0073 | +0.0073 |
| education_assessment/race | 0.0000 | 0.0093 | +0.0093 |
| education_assessment/income | 0.0000 | 0.0060 | +0.0060 |

## Conclusion and recommended next step

**Universal headroom.** 8/8 pairs have closed-form best R² < 0.05. No pair has a structural linear-erasure floor — the linear R² < 0.05 constraint is achievable on every pair by some closed-form linear projection, with task accuracy preserved within 1.5pp on all but one pair.

**Headroom for v2.** v2 Adult Option A leaves a factor of ~10–50× of linear-erasure quality on the table (e.g. marital_status: v2 R² = 0.186 vs LEACE 0.004; sex/income: v2 ~0.20–0.33 vs LEACE 0.007; income-as-attr: 0.125 vs 0.006; race: v2 0.14–0.23 vs LEACE 0.009). On every pair the constraint is **structurally achievable** and v2 is stalling for purely **optimization reasons** — competing gradients (vCLUB + VICReg + HSIC-aux + task) drowning the R² signal, early stopping firing at epoch ~25 well before the dual variables converge, and lambda saturating at 8–28 (of lambda_max=100) all consistent with this.

**Task-collateral pairs** (task acc drop ≥ 2pp under closed-form linear erasure): income_prediction/sex (+3.33pp). These mark a genuine fairness-utility tradeoff: the disallowed attr is genuinely predictive of the task and removing all linear predictability of the attr forces a small task-acc cost. This is normal — not a representational wall — and orthogonal to v2's optimization stalling. v2 should still hit R² < 0.05 on these pairs; whether the ~3pp utility cost is acceptable is a paper-claim question, not an engineering one.

**Train-vs-test floor.** LEACE drives train R² to exactly 0 on every pair (see table above). The 0.004–0.009 test-time R² is pure finite-sample/distribution-shift noise — not a representational ceiling. The 0.05 threshold has ~5–12× margin over this floor on every pair.

**Recommended next steps (ranked):**

1. **Fix v2's optimizer.** Most actionable: (a) disable early stopping or stop on a constraint-aware composite (`val_task_loss + Σ max(0, R² − τ)`), since the current criterion rewards constraint-violating-but-task-fitting representations; (b) drop `lambda_hsic_aux` to 0.0 — HSIC is a nonlinear gradient that competes with the linear-R² constraint and offers no additional auditable signal (the constraint is already on the auditor's metric); (c) warmup: train task-only for K=20 epochs to establish a non-degenerate representation, then ramp constraints over the next 20; (d) optionally seed v2 with a LEACE-style closed-form projection as the initial LoRA so the optimizer starts inside the feasible set rather than having to descend into it.

2. **Cotter best-iterate stopping** instead of patience on val task loss — track the best feasible iterate (where all R² < τ) and return that, matching standard practice for proxy-Lagrangian training.

3. **Document the income/sex tradeoff** in the paper's compliance section. On Adult, sex is genuinely predictive of income (the dataset's well-known demographic-bias artifact), so any compliant representation will pay a small (~3pp) income-prediction cost. This is a feature of the dataset, not a v2 failure mode.
