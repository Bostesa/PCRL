# PCA16-initialized matched C/D — frozen protocol, 2026-09-08 UTC

Question: does initializing the existing learned mapper to reproduce PCA16
improve the eventual utility/protection tradeoff, and at which fixed training
stage does residential transfer change? Only mapper initialization changes.
This is standard adversarial representation learning, not a PCRL mechanism,
purpose-conditioning/coalition experiment, or novelty claim.

Starting commit `de9a7e499c30802e320fec7ac01f0d6f72af8a48`, research branch
`ablations-facct-2026-07-24`. Protocol, configuration, execution closure and
reference records are hashed before fitting. Historical evidence remains intact.
The [bottleneck protocol](../redesign_20260908_acs_bottleneck_v1/PROTOCOL.md)
and [PCA16 protocol](../redesign_20260908_acs_pca16_v1/PROTOCOL.md) supply unchanged
model/head/audit details. No new dataset, task, dimension/weight search, erasure,
regularizer, bypass, coordinate search, stopping rule or reference refitting.

## Data and information boundaries

Identical California 2018 public ACS 30,000-person household sample, seeds 0/1/2,
original seven household-disjoint pools and all missing/eligibility masks.
Age 19–34, positive PWGTP; raw SHA256
`dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`.
The [original schema](../../docs/ACS_2018_SCHEMA_NOTES.md), original PCA32 maps,
allowlist, sample/split seeds, training-only preprocessing and fitting rows remain
unchanged. These are **DEVELOPMENT EVALUATION** households already used to shape
this question. No unused households are accessed; no confirmatory/survey claim.
Seeds share one cohort and SDs are descriptive, not independent population samples.

Only representation-fitting PCA32, three source binaries (PINCP>50000,
ESR==1 civilian employed **and at work**, PUBCOV==1 public coverage) and recorded
SEX(2)/RAC1P(9) enter representation fitting. Missing labels remain -1 and masked.
Source validation may be logged but cannot change training or select checkpoints.
MIG==1 means same residence one year ago; JWMNP>20 is eligible only at valid
integer minutes 1…200. These reserved labels enter independent downstream heads
only after **all scientific representation training for that seed** finishes and
I/W/final snapshots are frozen. They never influence initialization, training,
continuation, checkpoint choice or reporting which release is selected.

## Analytical initialization and fixed training

Construct the original full model with the original seeded RNG consumption,
then overwrite only mapper tensors under no_grad. Saved representation-fitting
mean mu and scale sigma standardize raw PCA32 to float32 exactly as before.
Mapper Linear(32,64) -> ReLU -> Linear(64,16) is initialized:

- First weight is vertical `[I32; -I32]`; first bias is zero.
- S is 16x32, S[j,j]=sigma[j] for j<16, otherwise zero.
- Second weight is horizontal `[S, -S]`; second bias is mu[:16].

In real arithmetic output is raw PCA32[:,:16]. Float32 rounding means numerical,
not bitwise, parity. Predeclared `atol=1e-5, rtol=1e-5`; report maximum absolute
and RMS coordinate errors on representation fitting and source validation before
training; later other pools after release freeze, evaluation after saved choices.
Failure stops for diagnosis, never a relaxed tolerance. Verify unchanged shapes,
parameter counts, RNG and non-mapper tensor identities against the original
recipe and saved zero-update checkpoint, including saved input means/scales.
Check finite forward/backward and nonzero gradients to initially zero readout
connections on a disposable model. All mapper weights remain independently
trainable, including unused-coordinate readouts; no tied weights/skip/frozen map.
Actual I remains zero-update. No historical post-warmup Adam state is reused.

Source heads (three independent linear binary logits), decoder16->64ReLU->32,
and two attribute adversaries16->64ReLU->32ReLU->K retain original initialization
process and dedicated seeds. Base loss is mean masked source BCE + .1 mean
row/coordinate reconstruction MSE to standardized PCA. This shared reconstruction
may retain attribute information and does not guarantee transfer.

Adam .001, betas(.9,.999), eps1e-8, no decay/clipping, batch256. Common base
warmup60 epochs; adversary warmup20 on its frozen mapper; exact C/D clone of
mapper/heads/decoder/adversaries and both relevant Adam states. Continue80 epochs
with identical minibatches. Each mapper batch receives three detached adversary
updates on mean separately masked attribute CE. C has observer adversaries and no
protection gradient. D adds `-.1 mean(CE_SEX/H_SEX, CE_RAC1P/H_RAC1P)` to the base
objective. H is fixed empirical representation-fitting prior entropy; undefined
or zero entropy fails explicitly. Adversary parameters are frozen for mapper
backpropagation, gradients through their inputs retained; source heads/decoder
receive only base gradients. Preserve original source/adversary missing-batch
handling, seeds, orders, logging and update counts without schedule changes.

Snapshots: I genuine initialization; W after common60 epochs; C_init unprotected
fixed80 continuation; D_init protected fixed80 continuation. W must remain
unchanged across adversary warmup. Save original phase/fork/final checkpoints,
state/buffer/Adam/schedule hashes and actual steps/exposure. All training finishes
before any snapshot transfer scoring. Intermediate utility is diagnostic, never
selectable: no early stop, choosing W instead of final, or favorable numerical
fallback. Numerical failure is retained/reported.

## Utility, immutable releases and final audits

Release only deterministic float32 mapper output, 16 coordinates/64 bytes;
no raw PCA, source/decoder/adversary outputs appended, no final LEACE. Freeze all
maps before heads/audits and verify hashes afterward. Fit fresh utility heads
for all five tasks on **each I/W/C_init/D_init**, including I even when near PCA16.
Exactly2048 eligible fitting rows/task and original subset/head seeds; logistic
C1/lbfgs500/tol1e-4 and MLP64/32ReLU/Adam.001/batch256/40epochs. Standardize only
head-fitting rows. Validation log loss chooses epochs (0/every5/final) and family,
with original earliest-epoch/lexicographic rules. Same labels and opportunities.

Final C_init/D_init receive five independent attribute auditors: logistic;
two120-epoch64/32MLPs with second seed+10000; HistGradientBoosting150iterations,
15leaves,lr.1,L2=1,noearlystop,minleaf20/5. Original attacker-fitting rows(cap4096),
validation rows/seeds and fitting-only standardizers. Additionally evaluate each
saved final adversary and fit one120-epoch catch-up per attribute/final arm,
reset Adam, identity preprocessing/direct mapper coordinates. Verify predictions
before first update. Catch-up inherits20+240 representation-fitting passes before
120 attacker-fitting passes; record counts separately from fresh candidates.

Compare matching five independent candidates primarily; separately report the
validation-selected winner among independent+catch-up, plus saved/fresh/caught-up
scores and per-family/restart curves. Saved adversary is diagnostic, not selectable.
No new I/W attack suites: original PCA16 audit is a named historical reference,
not an executed I audit. Save selections before opening development outcomes.
No candidate or component is selected by evaluation performance.

Reuse verified PCA32, PCA16, PCA32+LEACE, old C16/D16 and rich neural/tree banks
(unchanged/erased), plus prior/exposed control predictions/metrics. Verify original
files, row/preprocessing/recipe identities and prior replay records; no historical
training or audit refits. Original source hashes remain historical, never replaced
by this extended source or publication SHA. Raw/fitted local objects retain hashes.

Unweighted natural log loss is primary (original 1e-12 clip/renormalization).
Report signed prior-relative attack gain, secondary AUROC/accuracy/balanced
accuracy, per-class support and PWGTP sensitivity on identical selected predictions.
All9 race classes remain. Code4 absent from all attacker-fitting/validation pools;
original exposed-control failures remain explicit. Unsupported/undefined scores
never count as protected; complete all-attribute assessment remains unavailable.
No finite unsuccessful attack proves universal privacy or classification bounds.

## Predeclared analysis and execution budget

Every seed, validation and development separately, descriptive mean/sample SD:
I−PCA16 (coordinate and freshly fitted score differences), W−I, C_init−W,
D_init−C_init, C_init−historicalC16, D_init−historicalD16, each task and relevant
attribute. Attribute differences use both independent and catch-up-inclusive
scopes. Other releases contextualize the final tradeoff; none is reselected.

Carry unchanged original-PCA32 references for learned arms and stage utility:
source loss <= PCA32+.01 for each binary; retain half of PCA32's positive
residential advantage over the better unprotected rich bank per seed/split,
otherwise undefined. Final arms: report halving of each positive PCA32 attack
gain over fitting prior, and feature-versus-bank residence at least .01 lower
with each attribute gain at most .005 higher. Preserve failed inequalities,
negative gains, undefined headroom and race-coverage limitations. These are
provisional descriptive margins, not privacy budgets or noninferiority tests.
Give weighted residential sensitivity explicit space; no official survey inference.

A favorable initialization result concerns this baseline, not PCRL novelty.
A negative result concerns this fixed schedule, not impossibility. Stage changes
do not uniquely identify objective, geometry or optimization causes. A baseline
need not meet the full policy before a method can address a measured weakness.

Run only focused new checks and an artificial miniature pipeline. Time the first
complete scientific seed including training, snapshot heads, final audits and
controls/reference verification. On M4 Pro CPU, one numerical thread, expand to
seeds1/2 only if three-seed estimate fits900seconds. Otherwise preserve full one
seed; never weaken audits/config or adjust to outcomes. Publish compact source,
metrics, selections, parity, training diagnostics, stage/final tables, runtime,
reproduction/omitted hashes and decision to research branch. No PR/force/main.
