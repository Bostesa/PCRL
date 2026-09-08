# Nonlinear upstream adaptation: fixed pilot protocol

This protocol is written before fitting the pilot or viewing any final-test
outcome. The generator, budget, strength grid, thresholds and selection rules
are fixed for seeds 0, 1, 2. No generator redesign, search expansion or test-based
tuning is permitted within this run. The original Gaussian linear pilot at
`results/redesign_20260907_gaussian_v1/` remains the unchanged sanity reference.

## Question and population

Does protection-aware training **before** final erasure preserve more task
utility than matched task-only upstream training followed by the same erasure?

L=(U,V,S,E1,...,E5) consists of eight independent standard Gaussian variables.
P1 predicts U and prohibits V,S; P2 predicts V and prohibits U,S. The combined
recipient is authorized to recover U,V, but S remains prohibited. The nuisance
variables E1,...,E5 are independent of all three named signals.

Generate Q1,Q2 by QR of 8×8 independent standard Gaussian matrices, with the
diagonal-sign convention Q=Q*sign(diag(R)), from fixed NumPy RNG seeds 20260908
and 20260909. With r=LQ1, observations are X=(r+0.2*r³)Q2+b, b=linspace(-.4,.4,8).
The cubic is componentwise and strictly monotone (derivative 1+0.6*r²>0), so
the map is invertible. All parameters are independent of method outcomes.
`generator.json` stores both matrices and the offset exactly.

## Splits and preprocessing

Five disjoint RNG streams for each seed s use seeds 500000+100*s+i:

| i | Split | n | Permitted fitting use |
|---|---|---:|---|
| 0 | representation_train | 4096 | observation standardization, shared pretraining, C/D adaptation, independent task probes |
| 1 | calibration | 2048 | final LEACE fitting, after encoder freezing |
| 2 | attacker_fit | 2048 | linear/MLP leakage attackers and their preprocessing |
| 3 | validation | 2048 | pretraining checkpoint, MLP probe checkpoints, protection-strength selection |
| 4 | test | 4096 | final predictive scores and separately labeled descriptive covariance |

Rows are independent Gaussian draws, with globally disjoint numeric row IDs.
The test split is not generated until that seed's model/attacker/configuration
selection is written to `selection_before_test.json`. All methods within a
seed share each split exactly. Validation is reused for the declared finite
selection steps; it is not claimed to be a second independent final test.

Observation means/standard deviations use representation_train only. Every
probe separately fits feature and target preprocessing on its designated
fitting rows. No intercept, scaler or eraser is fit on validation or test.
Target means from an evaluation split occur only in score denominators and
descriptive covariance/variance, not in prediction fitting.

## Task pretraining and matched methods

Shared encoder: Linear(8,32), GELU, Linear(32,32), GELU, Linear(32,8). One affine
head predicts U,V jointly. There is no BatchNorm or dropout. All encoder/head
parameters receive task-MSE gradients during genuine pretraining: Adam lr .002,
batch 256, 80 epochs (1280 steps). Evaluate validation mean U/V MSE at epoch 0
and every five epochs; select its minimum, with earliest ties. Training always
executes the complete budget. Save initialization, selected, final checkpoints,
optimizer counts, state hashes, and before/after direct task-head performance.
Checkpoint selection uses task utility only, not leakage or erasure cost.

| Method | Representation before final evaluation |
|---|---|
| A | Frozen selected task-pretrained encoder; no erasure |
| B | Same frozen encoder; separate LEACE |
| C | Same checkpoint + task-only upstream adaptation; separate LEACE |
| D, strength .1 | Same checkpoint + protection-aware upstream adaptation; separate LEACE |
| D, strength 1 | Same checkpoint + protection-aware upstream adaptation; separate LEACE |

C and both D strengths reuse `PerPurposeLoRAEncoder`: rank4, alpha4, dropout0,
adapters at all three Linear layers. The first and second adapters precede
GELUs, so updates can change nonlinear features. Base checkpoint weights are
frozen; adapter parameters and purpose task heads are trainable. Each purpose
head copies its U or V row from the shared pretrained head. All three arms
use identical adapter/head initial tensors, RNG seeds, minibatch order, Adam
lr .001, batch256, no weight decay, and exactly 400 primal steps. The final
adapter iterate is used for every arm; no extra checkpoint search favors D.
Initial state/schedule hashes and trainable capacity are checked and recorded.

## Protection objective and gradient flow

There is **no erasure during representation training** in C or D. In all
erased methods, the final eraser is fitted once after encoder freezing on the
separate calibration split. Thus training does not fit exact batch erasure
and then penalize its identically zero residual covariance.

C minimizes mean U/V purpose task MSE. D minimizes

L = (MSE_U + MSE_V)/2 + strength/5 * Σ_c λ_c (R²_c − τ_c),

for P1→V, P1→S, P2→U, P2→S, and [P1;P2]→S **before erasure**. The batch proxy
centers H and z on representation-training rows, standardizes feature columns
with RMS floor1e-6, and solves (H'H/n + .001 I)w=H'z/n. R²=1−SSE/SST is
clamped below at zero for this differentiable fitting proxy only. The task and
ridge-solve gradients reach upstream LoRA parameters. Initial unscaled task
and protection gradient norms at first-layer B are checked for finiteness and
nonzero values; final per-layer parameter changes are saved.

Existing `Constraint` / `ProxyLagrangianOptimizer` primitives perform projected
dual ascent: initial λ=1, step .02, interval [0,10]. C performs no dual updates;
D performs 400. Individual τ=.05; combined-S τ=.10. This is the implemented
heuristic, with no asserted global convergence, Pareto, or privacy guarantee.
Unlike the existing categorical V2 pipeline, this pilot uses continuous targets,
upstream LoRA and no LEACE warm-start, training eraser, VICReg or vCLUB. The
intended C/D difference is solely the protection loss and its dual updates.

Final LEACE receives continuous V,S for P1 and U,S for P2; fits float64 with
shrinkage=False, constrain_cov_trace=False, svd_tol=1e-10. Its affine transform
is fixed for all subsequent task/attacker fits and evaluation. Each view keeps
eight ambient coordinates; concatenation has sixteen. Effective ranks and
empirical calibration/validation/test covariance are reported separately.

## Matched task and attacker evaluation

Every method gets new independent affine OLS and nonlinear task probes, one
per purpose, fit on representation_train. Attacker probes fit on attacker_fit:
each individual view and their concatenation predict U,V,S with a multi-output
regressor. Combined U,V results are recorded as authorized recovery, not leakage.

OLS uses a fixed relative singular-value cutoff1e-10, feature/target centering
and scales from fitting rows, and no validation tuning. MLPs use 32×32 ReLU
hidden layers, float32, Adam lr.001, batch256, 80 epochs, and exactly two fixed
initializations per view/role. Each target selects its full MLP checkpoint by
validation MSE over epoch0 and every five epochs, earliest ties. Multi-output
attacker trunks are trained jointly with standardized mean target-MSE, but
checkpoint choice is independent per target. Heads and attacker search budgets
match across methods. Fitting data, seed, candidate history, selected epoch,
steps, preprocessing and weights are saved. This is a small attack budget,
not a claim to have found every possible nonlinear attack.

Primary utility: mean U/V independently fitted MLP task-probe validation R².
Affine task scores are also always reported. The declared strength grid is
0 (C), .1 and 1 (D). For configuration selection, take the stronger validation
score of the linear and MLP attackers for each of the five prohibited pairs.
Each individual pair must be ≤.05 and combined S ≤.10. Undefined evidence is
infeasible. Among feasible candidates choose greatest primary utility, then
smaller strength. If none is feasible, choose minimum summed normalized excess
Σ max(0,R²−τ)/τ, then greatest utility, then smaller strength, explicitly marking
the choice infeasible. Apply this rule both to the full grid and separately to
the two positive strengths. All fixed arms are reported, not only the selected
one; D has two positive settings compared with one critical C control.

Predictive R² is 1−SSE/SST about the evaluated target mean, without clipping;
MSE, n and target variance are saved too. Negative R² means worse squared error
than the evaluation constant baseline, not negative information. Empirical
covariance and same-split fitted projection R² are distinct diagnostics.
Near-zero linear R² is not nonlinear privacy; unsuccessful MLP attacks do not
establish a universal guarantee. Three-seed differences are preliminary.

## Execution boundary and preservation

CPU only, one numerical-library thread; existing Apple M4 Pro, 24 GiB RAM.
First run one complete seed including pretraining, adaptation, final erasers,
all attackers, validation selection and final evaluation. Record its measured
time and estimate the two remaining seeds before continuing. Target total
experiment runtime below roughly30 minutes. If reduction is necessary, retain
the prior run and document a common revised budget before any new final test.
No model downloads, paid resources, broad sweeps, PR or push.

Reproduce with `.venv/bin/python experiments/run_nonlinear_conflict.py --out
results/redesign_20260907_nonlinear_upstream_v1 --seeds 0`, then `--seeds 1 2`.
`--prepare-only` freezes config and source hashes without fitting or generating
test data. Existing invocation/seed outputs cannot be overwritten; source or
configuration changes after freezing require a new retained result directory.
