# Nonlinear release protection: frozen pilot protocol

Written before pilot fitting or fresh final-test generation. Three seeds 0,1,2;
two configurations per new protection method. The generator, checkpoints,
budgets, thresholds and selection rule below will not change after results.
Previous Gaussian and nonlinear-upstream artifacts remain unchanged.

## Question and known evidence

Can nonlinear protection reduce residual leakage at task R² >= .99 for EACH
purpose, and does a constrained formulation add value over simple releases
and matched fixed-penalty adversarial training? Previous independent LEACE
lost only about .00024 mean task R², retained substantial MLP leakage, and the
previous protection objective gave no feasible advantage. That objective was
pre-erasure affine ridge R² with projected dual updates, not a nonlinear
adversary. The current E is standard constrained adversarial optimization
using existing PCRL LoRA and an explicit projected dual update, not a new
mechanism or guarantee.

## Fixed generator, starting features, and split roles

Exactly the previous generator: L=(U,V,S,E1,...,E5) iid N(0,I8), r=LQ1,
X=(r+.2*r³)Q2+b. Q1/Q2 are sign-corrected QR Gaussian orthogonal matrices
from RNG seeds 20260908/20260909; b=linspace(-.4,.4,8). The map is invertible.
`generator.json` is copied byte-for-byte from the previous pilot.
P1 predicts U and prohibits V,S; P2 predicts V and prohibits U,S. Combined
access permits U,V, and prohibits S. Independent true U/V releases provide
a known population-feasible oracle diagnostic, unavailable in deployment.

Reuse the three task-pretrained selected checkpoints from
`results/redesign_20260907_nonlinear_upstream_v1/seed_{s}/pretraining/selected.pt`.
The 8→32→32→8 GELU encoder was genuinely trained on U,V for1280 updates;
validation task MSE selected epoch80 for all seeds. No leakage-based feature
selection. All new arms share identical saved starting tensors within seed.
Checkpoint hashes, prior source hashes, and file provenance are recorded.

| Split | n | RNG seed | Fitting/selection use |
|---|---:|---|---|
| representation_train | 4096 | 500000+100*s | observation preprocessing; upstream training; independent task probes |
| calibration | 2048 | 500001+100*s | final LEACE after encoder freezing only |
| attacker_fit | 2048 | 500002+100*s | independent audit fitting/preprocessing only |
| validation | 2048 | 500003+100*s | audit checkpoints/restarts and finite configuration selection |
| fresh test | 4096 | 900004,900104,900204 for s=0,1,2 | final evaluation only |

The first four streams and observation mean/std are identical to the previous
pilot. Do not generate or reuse its old test stream500004+100*s. The fresh test
is generated only after all seed-specific fitting and selection are saved in
`selection_before_test.json`. No fitting occurs after that boundary. Numeric
row IDs use RNG_seed*100000+row_index, and all splits have saved IDs and hashes.
Validation was already used in the prior pilot; finite selection reuse is
disclosed, with the independent new final test reserved for evaluation.

## Controls first and release interfaces

| Key | Release per purpose |
|---|---|
| A_oracle | true U / true V; scalar inaccessible oracle |
| B_prediction_only | saved C_task_only model's existing appropriate task-head scalar, before any LEACE |
| C_saved_task_only | saved C_task_only full 8-dimensional views, freshly calibrated separate LEACE |
| R_prior_0.1 / R_prior_1 | saved previous protection checkpoints, freshly calibrated separate LEACE; references only |
| C_matched_task_only | new training procedure below with weight0, final separate LEACE |
| D_fixed_0.01 / D_fixed_0.1 | fixed nonlinear-adversary penalty .01 / .1, final separate LEACE |
| E_dual_0.01 / E_dual_0.1 | matching nonlinear adversaries; projected dual weights initialized .01 / .1, final separate LEACE |

All references receive newly fitted independent probes and common new-test
evaluation; no old test scores are copied. B uses saved heads, with no refit;
report direct prediction R² and independently probed utility separately.
Scalar success establishes sufficiency for two fixed tasks, not a reusable
representation for future tasks. A/B dimensions1+1 differ deliberately from
the 8+8 full-representation interface. Combined dimensions are2 and16.
Additional exposed_target diagnostic releases (U,V,S) in BOTH views; this
deliberately violates policy and tests attacker competence, including S.

## Matched nonlinear training on released features

C_matched/D/E all use existing per-purpose LoRA at every encoder Linear:
rank4, alpha4, dropout0; two adapters precede GELUs and can change nonlinear
features. Backbone frozen, no BatchNorm/dropout; heads copy pretrained U/V
rows. Model seed630000+s. Exactly1314 trainable encoder/head parameters.
Same initialization, batches, Adam lr.001/no weight decay, batch256,
400 encoder updates, fixed final iterate. No checkpoint search.

The first1024 representation_train rows form a fixed training-eraser holdout;
the remaining3072 supply encoder/adversary update batches. These roles are
identical across C/D/E. At encoder step0 and every25 steps through400, fit
separate continuous LEACE on holdout features/own prohibited targets using
float64, shrinkage=False, constrain_cov_trace=False, svd_tol=1e-10. Fits run
under no_grad; cached float32 affine maps apply to update batches with
gradients through their inputs only. Training adversary feature center/scale
is fit on holdout released features and refreshed with the maps. Training
adversary target center/scale is fit once on the3072 update rows.

Three independent adversaries have32×32 ReLU hidden layers, outputs2 for
P1(V,S),2 for P2(U,S), and1 for combined S. Each minimizes normalized target
MSE. Warm up adversaries100 updates before any encoder update, then do2
adversary updates per encoder update:900 adversary optimizer steps total,
same schedules and initial tensors for every new arm, including weight0.
Adversary Adam lr.001, batch256. For adversary updates, encoder outputs are
detached. For encoder updates, adversary parameters are frozen but derivatives
through adversary inputs reach upstream adapters. Training and independently
fitted audit networks have distinct states, seeds and fitting roles.

Let e_c be the adversary's MSE on a training minibatch for target c normalized
using the fixed update-row target statistics; q_c=1-e_c. Let τ_c=.05 for
individual constraints and .10 for combined S. All three methods minimize

    mean raw U/V task-head MSE on released views + (1/5) Σ_c λ_c (q_c−τ_c).

The negative-MSE term reduces adversary success when minimized by the encoder;
adversaries themselves minimize positive MSE. Task heads consume the same
erased feature interface as adversaries. q_c is a stochastic training proxy
using fixed training variance, not held-out predictive R² or a certificate.
No exact same-batch linear covariance statistic is optimized. No clipping of
MSE/R² is used to manufacture protection. Instability or weak training
adversaries will be recorded rather than hidden.

C fixes λ=0. D fixes all λ=.01 or .1. E starts at the matching weight w and
updates λ_c <- clip(λ_c + .02*w*(q_c−τ_c),0,1) after each encoder update.
Thus E has dual step .0002 or .002, 400 updates. D/E have the same task and
leakage losses; their intended difference is fixed versus adaptive weights.
This is a standard projected constrained adversarial formulation. It does
not enforce task utility mathematically: utility floors are validation filters.

Save step0/final checkpoints, all schedules/hashes/counts, separate task and
protection upstream gradients after warmup, parameter deltas, adversary
learning/constraint/dual histories, and refreshed-map diagnostics. Initial
task and unscaled protection gradients must be finite and nonzero. Verify
the base is unchanged and trainable/adversary budgets match across all arms.

Training fits approximate moving erasers on a training-only1024-row holdout.
Final release fits new erasers on reserved2048-row calibration after freezing
encoders; gradients ignore dependence of eraser fits on encoder parameters.
That finite-calibration/stop-gradient approximation may limit transfer and is
reported. C_saved differs in training loss/interface and fitting rows from
C_matched; C_matched is the clean D/E task-only control.

## Independent audit and selection, frozen before testing

Reuse the previous probe protocol without changing search effort: affine OLS
float64, fitting-only centering/scaling/intercepts, rcond1e-10; and MLP32×32
ReLU, Adam.001, batch256,80 epochs, two independent initialization seeds.
Each MLP target independently chooses its strongest validation checkpoint
(minimum target MSE) from epoch0/every5, earliest ties. Same restart seeds,
architecture/budget for every release. Multitarget audit training uses mean
standardized MSE; full selected states are saved separately per target.
Task probes fit representation_train and score authorized U/V independently;
leakage probes fit attacker_fit and predict U,V,S from each view/concatenation.
Each probe fits its own feature/target preprocessing on its fitting data only.

Primary utility: MLP task R²>=.99 for U AND >=.99 for V, separately per seed.
Protection: P1→V,S and P2→U,S each R²<=.05; combined→S R²<=.10, for BOTH
affine and validation-selected MLP audits. All ten inequalities must hold;
undefined evidence fails. Negative predictive R² is retained, not negative
information. Score is1−sum squared errors/sum evaluation-target deviations
from the evaluation mean squared. Evaluation means are used only in scoring
and separately labeled descriptive covariance, never prediction fitting.

Select D and E independently among their exactly two declared configurations:
first filter by BOTH validation utility floors; minimize the maximum of all
ten leakage/constraint-specific-threshold ratios, then higher minimum-purpose
utility, then lexical key. If no candidate passes both utility floors, label
the method infeasible and select a DIAGNOSTIC only by highest minimum-purpose
utility, then lowest worst ratio, then key. Protection pass is separate and
is never assumed from utility feasibility. Save selection before fresh test.
Report all configs as exploratory tradeoffs, selected primary rows, all
per-seed/per-target values, means/sample SDs and paired E−D differences.
No averaging across purposes or seeds can turn a failed constraint into a pass.

Exposed-target recovery and oracle independence are diagnostics, not tuning
criteria. Report unexpected attack failure/spurious leakage. Passing these
finite empirical tests establishes no universal privacy guarantee. Three
seeds provide preliminary evidence only.

## Execution budget and reproduction

CPU, one numerical-library thread, existing Apple M4 Pro24GiB. First run one
complete seed including audits; save wall time and estimate twice that for
the remaining seeds before expansion. Target much less than30 minutes;
no paid resources/downloads/grid expansion. Any necessary common budget
reduction must precede any final testing and be documented. No changes planned.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest tests/test_nonlinear_release.py tests/test_nonlinear_conflict.py tests/test_redesign_conflict.py tests/test_proxy_lagrangian.py -q
.venv/bin/python experiments/run_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1 --prepare-only
/usr/bin/time -p .venv/bin/python experiments/run_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1 --seeds 0
/usr/bin/time -p .venv/bin/python experiments/run_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1 --seeds 1 2
.venv/bin/python scripts/summarize_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1
```
