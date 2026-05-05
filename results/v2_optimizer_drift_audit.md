# V2 Round 4 Optimizer-Drift Audit

Source: `checkpoints/v2_{adult,hmda}_s{0,1,2}/final.pt` `history` field.
Per-epoch `r2_per_pair_per_epoch` for 205 epochs (5 warmup + 200
constrained). Lambdas only stored as final values; per-epoch λ
trajectories are not in checkpoints (limitation).

Threshold τ = 0.05. Landmark epochs: e0/e4 (warmup), e5 (first
constrained), e10/e25/e50/e100/e150/e199/e204.

## Key universal observation: warmup destroys LEACE init

Joint LEACE warm-start drives epoch-0 R² to ~0 (verified in
`results/leace_rank_probe.json`). But `train()` then runs 5
**task-only warmup epochs** during which only `L_task + L_vicreg`
backprops through the LoRA. With ~76 batches/epoch, that's ~380 SGD
steps of pure-task gradient before the dual ever engages.

End-of-warmup R² (epoch 0 logs the state AFTER one warmup epoch):

| pair | s0 e0 | s0 e4 | s1 e0 | s1 e4 | s2 e0 | s2 e4 |
|---|---|---|---|---|---|---|
| Adult income/race | 0.775 | 0.796 | 0.778 | 0.796 | 0.764 | 0.749 |
| Adult income/sex | 0.924 | 0.925 | 0.909 | 0.918 | 0.949 | 0.956 |
| Adult employment/age_group | 0.569 | 0.544 | 0.583 | 0.537 | 0.569 | 0.540 |
| HMDA underwriting/race | 0.933 | 0.937 | 0.923 | 0.931 | 0.926 | 0.935 |
| HMDA pricing/race | 0.904 | 0.882 | 0.902 | 0.879 | 0.914 | 0.876 |
| HMDA pricing/sex | 0.943 | 0.918 | 0.959 | 0.916 | 0.955 | 0.935 |

The LEACE init is gone by the end of warmup epoch 0. The constrained
phase starts from a fully-task-fitted, **un-erased** state. LEACE init
in Round 4 amounts to a tiny inductive bias on subsequent training
trajectory, not the "start inside the feasible set" property the
warm-start was designed to provide.

## Per-pair classification

| dataset | pair | seed | e0 | e25 | e50 | e100 | e150 | e199 | e204 | λ_final | class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | income/race | 0 | 0.775✗ | 0.252✗ | 0.011✓ | 0.009✓ | 0.045✓ | 0.074✗ | 0.065✗ | 0.11 | OSCILL |
| Adult | income/race | 1 | 0.778✗ | 0.370✗ | 0.053✗ | 0.025✓ | 0.229✗ | 0.275✗ | 0.232✗ | 0.51 | STARVE |
| Adult | income/race | 2 | 0.764✗ | 0.284✗ | 0.017✓ | 0.009✓ | 0.138✗ | 0.366✗ | 0.382✗ | 0.30 | STARVE |
| HMDA | underwriting/race | 0 | 0.933✗ | 0.006✓ | 0.007✓ | 0.119✗ | 0.073✗ | 0.113✗ | 0.080✗ | 3.07 | LATE |
| HMDA | underwriting/race | 1 | 0.923✗ | 0.021✓ | 0.088✗ | 0.102✗ | 0.103✗ | 0.104✗ | 0.094✗ | 50.47 | LATE |
| HMDA | underwriting/race | 2 | 0.926✗ | 0.014✓ | 0.065✗ | 0.142✗ | 0.115✗ | 0.138✗ | 0.123✗ | 2.98 | LATE |
| HMDA | pricing/race | 0 | 0.904✗ | 0.267✗ | 0.016✓ | 0.017✓ | 0.131✗ | 0.074✗ | 0.070✗ | 1.35 | LATE |
| HMDA | pricing/race | 1 | 0.902✗ | 0.105✗ | 0.035✓ | 0.024✓ | 0.038✓ | 0.069✗ | 0.089✗ | 1.08 | LATE |
| HMDA | pricing/race | 2 | 0.914✗ | 0.362✗ | 0.014✓ | 0.057✗ | 0.250✗ | 0.105✗ | 0.106✗ | 0.66 | STARVE |

Aggregate over 9 failing pair-seeds: **5 LATE, 3 STARVE, 1 OSCILL.
Every single failing pair-seed reaches R² < 0.05 by epoch 25-100, then
drifts back above τ before epoch 200.** None are pure EARLY-drift; none
are SATURATE.

### Comparison: confidently-passing pairs

| pair | s0 e100/e204 | λ_final | s1 e100/e204 | λ_final | s2 e100/e204 | λ_final |
|---|---|---|---|---|---|---|
| Adult income/sex | 0.017/0.043 | 8.98 | 0.024/0.048 | 4.69 | 0.028/0.018 | 3.62 |
| Adult employment/age | 0.268/0.031 | 60.16 | 0.259/0.031 | 60.93 | 0.260/0.018 | 64.87 |
| HMDA pricing/sex | 0.017/0.047 | 2.36 | 0.027/0.045 | 6.78 | 0.018/0.050 | 0.02 |
| HMDA fair_lending/sex | 0.056/0.056 | 29.69 | 0.039/0.039 | 15.53 | 0.069/0.067 | 45.09 |
| HMDA fair_lending/race | 0.035/0.035 | 5.72 | 0.024/0.024 | 0.00 | 0.043/0.042 | 17.60 |

Passing pairs share a flat-feasible-tail: once R² drops below τ around
epoch 50-100 it stays within ε of τ for the rest of training. λ either
locks high (employment/age=60+, fair_lending/sex=15-45) or relaxes to
~0 when R² is safely below threshold (HMDA s1 fair_lending/race
λ→0.0, R²=0.024 stable). HMDA s2 fair_lending/sex (λ=45) and HMDA s0
fair_lending/sex (λ=30) are technically just-above τ at end (0.067,
0.056) — these mirror the LATE-drift mode but at much smaller
amplitude.

## Diagnosis

**Dominant mechanism: post-feasibility drift driven by dual relaxation.**
Both LATE and STARVE share the same shape:

1. Constrained phase begins at R² ≈ 0.5–0.95 (warmup destroyed init).
2. Dual ramps up; primal projects R² down → R² < τ around epoch 25-100.
3. Dual decreases (proxy-Lagrangian: when constraint slack, λ
   descends). On easy pairs (sex, age, marital_status, ethnicity in
   most purposes) λ either locks at a steady value or relaxes to ~0,
   and primal-only signal keeps R² robustly below τ.
4. On race in {Adult income, HMDA underwriting, HMDA pricing}: race
   correlates with the task target so the task gradient continually
   pushes the LoRA toward race-encoding directions. As λ relaxes, that
   pressure regrows R². By the time the dual notices, λ has decayed to
   <1 and the ascent rate (`lr_lambda=0.02`) cannot catch the rising
   R² before training ends.
5. The Cotter best-iterate selector picks a much earlier feasible
   epoch (~12 per memory), which is why best.pt looks better than
   final.pt on these pairs — but the paper's stability claim should
   rest on final.pt.

**Why race specifically?** Income, mortgage approval, and pricing all
have strong race correlations in their datasets. A 5-class attr also
has more degrees of freedom for residual signal to hide in than a
2-class attr. The combination — high task-correlation × high
cardinality — is the worst case.

Race in employment_analysis (Adult) doesn't fail because that purpose's
task (occupation grouping) has weaker race correlation than income,
AND because age_group + marital_status drag the dual pressure on the
joint constraint group up via their own large λ values, providing
collateral protection.

## Recommendations (no AWS, ranked by surgical-ness)

### Fix R1: lambda floor

Project dual to `[λ_min, λ_max]` instead of `[0, λ_max]` with λ_min ≈
5-10. One-line change in `pcrl/training/proxy_lagrangian.py` dual_step.

- Addresses STARVE directly (3/9 failing pair-seeds where final λ < 1)
- Slows LATE-DRIFT — dual is already engaged when R² rises, ascent
  starts from λ ≥ λ_min instead of 0
- Trade-off: passing pairs that had relaxed to λ=0 (e.g. HMDA s1
  fair_lending/race, λ=0) will carry λ ≥ λ_min unnecessarily; primal
  sees permanent constraint pressure. May slightly increase task loss.
  Acceptable since those pairs are already feasible by margin.

### Fix R2: skip warmup with LEACE init

Set `warmup_epochs=0` when `leace_init=True`. The original purpose of
warmup was to let LoRA fit the task before constraints engage; with
LEACE init the LoRA already has a structured (Q-I)W projection, so
constrained training and task fitting can co-adapt from epoch 0.

- Preserves LEACE init through the early epochs (current run
  destroys it in ~76 batches of pure task gradient).
- Trade-off: dual variables engage from λ_init=1.0 in batch 1. Could
  destabilize early training if primal+dual co-adaptation is fragile.
  Mitigated by `lr_lambda=0.02`.

### Fix R3: re-fit LEACE periodically

Every K=20 epochs, re-run `leace_warm_start(train_loader)` to re-anchor
the encoder to the feasible manifold. More invasive (~30 LOC for the
hook), and re-projection may invalidate Cotter selection's monotonic-
state assumption (a snapshot at epoch t can suddenly diverge from the
weights at epoch t+1).

### Fix R4: asymmetric dual update

When R² > τ, λ ascent rate = lr_lambda (current). When R² ≤ τ, λ
descent rate = α·lr_lambda with α ≈ 0.2. Slows starvation while
preserving sensitivity to overshoot. ~10 LOC in proxy_lagrangian.

## Verdict

```
Adult/HMDA optimizer drift audit complete.

Dominant drift mechanism: LATE-DRIFT + STARVATION (5 LATE + 3 STARVE
+ 1 OSCILL across 9 failing pair-seeds). All share the same dynamic:
race attr in task-correlated purposes reaches feasibility around
epoch 25-100, dual relaxes, task gradient regrows R² in the second
half of training.

- Adult income/race s0: OSCILL (λ_final=0.11)
- Adult income/race s1: STARVE  (λ_final=0.51)
- Adult income/race s2: STARVE  (λ_final=0.30)
- HMDA underwriting/race s0/s1/s2: LATE
- HMDA pricing/race s0/s1: LATE
- HMDA pricing/race s2: STARVE  (λ_final=0.66)

Recommended fix: combine
  R1 (lambda floor λ_min=5 in proxy_lagrangian dual_step) +
  R2 (warmup_epochs=0 when leace_init=True).
~5 LOC config + ~5 LOC trainer change. Validate via the existing local
50-epoch Adult probe (verify epoch-5 R² < 0.05 instead of ~0.45 in
Round 4). Then a single AWS Adult retrain to compare final.pt
trajectories vs Round 4 baseline.

Estimated impact: 6/9 failing pair-seeds (the STARVE + LATE-with-low-λ
cases) should improve substantially. The 3/9 LATE-with-high-λ cases
(HMDA underwriting/race s1 λ=50, HMDA pricing/race s0/s1 λ=1.35/1.08)
are less certain — the dual was already engaged but R² still drifted,
so the task gradient is genuinely overpowering even active dual
pressure. These may need R3 (periodic re-LEACE) to fix.

Estimated implementation cost: 30-60 min, including a 50-epoch CPU
probe and the proxy-Lagrangian unit-test update.
```

## Files

- `experiments/analyze_drift.py` — script that produced this analysis.
- `results/v2_optimizer_drift_audit.json` — raw per-pair-seed
  classifications + landmark trajectories.
- `checkpoints/v2_{adult,hmda}_s{0,1,2}/final.pt` — source data.
