# R2 pre-fit objective and adapter map

This design was written before any numerical fixture, scientific import, data
label read or optimizer launch by R2. It is a proposal for the coordinator's
science lock, not an outcome-dependent amendment.

The conditional alignment motivation is established by Zhao et al., arXiv
1910.07162v3 sections 3.1–3.4, especially its conditional balanced-error minimax
equation 2. That paper does not establish unconditional SEX privacy for this
service. Deka and Sutherland, PMLR v206 deka23a sections 3.1–3.4 equations 7,
9, 11 and the task-loss display supply the block-power, strongest-bandwidth,
conditional local, and negative target-power terms. Their Adult experiment uses
target and classification coefficients 1 (paper section 5 and official config).

Read sources: https://arxiv.org/html/1910.07162v3 and
https://proceedings.mlr.press/v206/deka23a/deka23a.pdf . Official code pinned by
git ls-remote then git show at full SHA
280a178abc86802cef58f99ab3eff4d2365ddacb; files utils/mmd.py,
trainer/frep_eq.py and config/eq/adult/lambda_1.yml inspected before adaptation.
The exact mmd.py is retained as task_conditioned_representation/
official_reference_mmd.py for reproducible fixture parity.

## Pinned core and paper/code discrepancy

For n paired observations per arm, use off-diagonal one-sample U-statistic
MMD2 = offmean(Kx)+offmean(Ky)-2offmean(Kxy). Let H=Kx+Ky-Kxy-Kxy.T.
The official biased branch computes v=4*(mean_i(mean_j H_ij)^2)-mean(H)^2).
Power is Phi(sqrt(b)*MMD2/sqrt(max(v,1e-6))-Phi^-1(.95)). No clipping of
negative MMD2 occurs. The max variance floor and CDF are essential; this is
not raw squared MMD. The official biased branch ignores its population-size
argument. Paper equation 4 includes division by hypothetical block size B;
we preserve literal code parity and explicitly report the resulting proxy.
This is a pinned-code MMD-B-Fair adaptation, not a claim that the code proxy
is a calibrated power probability. Only fresh attacks establish recovery.

Register n=32 unique groups per SEX arm, b=4 hypothetical blocks, B=8
(32 observations per arm total). The estimator uses all 32 observations and
does not physically split, as expressly stated after paper equation 8. Sample
groups without replacement inside each draw and choose one group row uniformly.
Different objective draws may reuse people; none is claimed to be a new
independent sample or independent empirical test. Groups with inconsistent
task/SEX labels are rejected. Support threshold remains >=32 groups per arm.

## Named adaptations

1. Full-wire sufficient kernel: concatenate z_r with its actual centered own
head logits. The deployed wire also sends probabilities and decisions, which
are deterministic functions of these logits. All transmitted columns enter
fresh attacks. No source scores or latent bypass enter this kernel or release.
2. Scale rule: each coordinate centered/scaled using detached means and
population standard deviations from a fixed <=256-row FIT reference subset.
Refresh at epoch start including epoch 0. Floor=max(1e-12,1e-8*coordinate's
initial standard deviation), retained per coordinate. No coordinate is dropped.
Distance is median strict-upper-triangle normalized squared reference distance,
floored at 1e-12. RBF denominator is distance*(.25,.5,1,2,4)^2. Log floor count,
variance/scale and head norms. Actual releases remain unnormalized.
3. FIT frequency weighted conditional terms: all supported true local task
classes and true task pairs have weight empirical FIT row count/N. Unsupported
mass stays uncovered and is not renormalized. For each recipient/pair term,
first sum weighted power vectors across strata, then select their largest
bandwidth entry; max outside the conditional sum preserves equation 11's
strongest common-kernel convention.
4. Multiclass target contrast: for each task compare each unordered pair of
true classes using the same kernel machinery with weight
n_c*n_d/sum_{a<b}n_a*n_b. Unsupported target contrasts retain missing mass.
Max over bandwidth after weighted contrast sum; negative coefficient 1 per
recipient. Income's two classes reduce exactly to one official target test.
5. Aligned coalition product RBF: multiply same-relative-bandwidth recipient
kernels, equivalent to exponentiating the sum of separately normalized squared
distances. Both encoders see identical person order; shuffled row IDs rejected.
The product kernel can distinguish XOR complementary clues despite identical
recipient marginals. This extension, rather than conditioning or end-to-end
training, is the candidate contribution requiring experimental evidence.

Objective is population-row CE_income+CE_occupation -target0-target1
+lambda_local*(local0+local1)+lambda_pair*pair. Main joint uses equal local/pair
coefficients. LOCAL_TWIN removes only pair at exactly the joint local coefficient.
The stronger local control uses its separate preregistered gradient calibration.
U_TASKPOWER retains the two negative target terms with zero sensitive terms.
Unconditional joint retains the full-wire/target/scaling rules, uses marginal
SEX groups for each local term and coalition, and receives nominal joint lambda.
All task CE is evaluated on a separate uniformly sampled row minibatch.

## Fixed data-free engineering fixtures and API

kernel_fixtures.py fixes source-core parity tolerance 1e-12; within-label no
extra signal <=.05+1e-12; planted extra signal and aligned XOR pair >.99;
shuffled alignment rejection; reciprocal feature/inverse-head scales
1e-3,1e-2,1,1e2,1e3 preserving probabilities and kernel-index with power
error<1e-10; finite nonzero gradient >1e-12; detached reference; sparse-stratum
uncovered mass; no duplicate sampling; contradictory duplicate-group rejection.
These checks concern implementation laws, not Adult superiority.

StrataRegistry(FIT labels/groups).draw(random.Random, family) produces Draws.
KernelObjective.refresh_model_reference(model, Xfixed) runs detached reference
refresh; loss(model, Xfit, draws, family, active=(0,1)) forwards sorted unique
union indices once and returns tensor terms plus diagnostics. model.encode(r,x)
must yield actual protected z, and model.head[r] must read that z. The trainer
controls source initialization, uniform task batch RNG, sequential freezing,
calibration and selection. R2 does not read any private role arrays.

No novelty claim is made for conditional alignment, MMD, block testing,
bandwidth maximization, trainable representations, or fair classification.
