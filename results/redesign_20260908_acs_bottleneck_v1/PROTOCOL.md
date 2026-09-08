# Fixed ACS utility-aware nonlinear bottleneck — 2026-09-08

**DEVELOPMENT EVALUATION.** Original test households informed this question.
Their reuse is exploratory development evidence, not untouched confirmation.
No unused household, new split, task search, protection grid, or synthetic rerun.
Question: does a fixed nonlinear adversarial bottleneck improve on the identical
unprotected bottleneck while preserving source utility and residential transfer?
This is standard adversarial representation learning, not a PCRL mechanism or
novelty claim. No purpose conditioning, coalition, LoRA, new eraser, or new data.

## Data, policy, and immutable references

Start from `948169361c38fa5d37657fd45c5ab45c84f1fef5`. Use the original
[ACS schema](../../docs/ACS_2018_SCHEMA_NOTES.md),
[transfer protocol](../redesign_20260907_acs_transfer_v1/PROTOCOL.md), and
[erasure protocol](../redesign_20260908_acs_protection_v1/PROTOCOL.md).
Public California 2018 ACS, age19–34, positive PWGTP, fixed30,000-person
whole-household sample seed1200000; split seeds1210000+[0,1,2]. Original seven
household-disjoint pools remain representation fitting(.35), source validation(.10),
downstream fitting(.15), downstream validation(.10), attacker fitting(.10),
attacker validation(.10), development evaluation(.10). Raw CSV SHA256 is
`dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`.
Keep grouping/deduplication, masks, row order, ten allowed input covariates,
train-only preprocessing, and original PCA maps unchanged.

The hypothetical one-time recipient may learn income>50000, ESR==1 (civilian
employed **and at work**), PUBCOV==1(public coverage), MIG==1(same residence one
year ago), and JWMNP>20(valid integer minutes1…200 only). PINCP signed valid range
−19998…4209995 includes zero/negative income; ESR1…6, PUBCOV1…2, MIG1…3.
Missing/inapplicable labels are −1 and masked, not negative examples. SEX1…2
and RAC1P1…9 are recorded audited categories, retained without collapsing.
Public records simulate a release; no confidentiality against public linkage,
survey population inference, or arbitrary-future-task claim is made. Authorized
outcomes can reveal attributes. Commute eligibility through example membership
is common to all interfaces; numerical-release audits do not cover extra lists.

Central arms: **A** original32D PCA (`E_pca`); **B** its already fitted affine
joint LEACE (`E_pca_leace`); **C**16D base bottleneck (`C_bottleneck`); **D**16D
protected bottleneck (`D_protected`). Keep frozen26D rich neural/tree banks,
unchanged and erased, plus fitting priors/exposed controls. No source or bank
retraining and **no final eraser refit**. Verify original maps, saved release
arrays, fitting rows, selection protocol, standardizers and prediction hashes
before reusing reference results. Missing necessary objects cause an explicit
block or documented regeneration; never silently substitute a map.

## Fixed representation training

Only representation-fitting PCA rows, the three source binaries, and SEX/RAC1P
enter fitting. PCA standardization mean/population SD is fitted on these rows;
constant scales become1. Source-validation inputs/binaries may produce diagnostic
curves only. MIG/JWMNP are absent from the training API and reconstruction inputs.
They cannot select architecture, epoch, weight, dimension, or any release map.

Mapper32→64 ReLU→16 linear; three independent linear binary-logit heads;
decoder16→64 ReLU→32 linear reconstructs standardized PCA. Two adversaries,
16→64 ReLU→32 ReLU→K logits, K=2/9. Release only deterministic mapper output,
float32,16 coordinates/64bytes. No skip, raw PCA, heads, decoder, adversary scores,
original bank probabilities, or LEACE is appended/applied to C/D.
Reconstruction is generic retention, and can reward attribute information; it
does not guarantee transfer and may conflict with protection.

Adam .001, default betas(.9,.999),eps1e−8, no weight decay/clipping; batch256.
Base loss = equal-weight mean of three masked binary cross-entropies + .1 times
row/coordinate-mean reconstruction MSE. A batch with no valid labels for a source
contributes differentiable zero for that task, retaining denominator3; log any
such batch and support. Undefined/zero fitting attribute entropy is an explicit
numerical/data failure, not a silently dropped protection term.

Warm one mapper/heads/decoder60 full epochs on base loss. Initialize common
adversaries, train20 epochs on the frozen warm mapper. Clone exact mapper,
source heads, decoder **and Adam states** into C/D, and exact warmed adversaries
and their Adam states. Continue80 full epochs with identical shuffled minibatches.
For each mapper minibatch, do3 adversary updates on that same detached current
release, plain mean of separately masked SEX/RAC1P cross-entropies. Then do one
mapper/heads/decoder update. C's adversaries are observers with zero protection
gradient. D's mapper objective adds **−.1 mean(CE_SEX/H_SEX, CE_RAC1P/H_RAC1P)**,
where H is empirical entropy on valid representation-fitting labels, no smoothing,
fixed thereafter. Freeze adversary parameters while backpropagating through their
inputs; source heads/decoder get only intended base gradients. Zero gradients
between phases. Individual attribute losses, source losses, reconstruction,
gradient norms, actual steps/row and valid-label exposure, schedules, and clone
hashes are recorded. Missing attribute batch losses are differentiable zero with
denominator2; fitting support/undefined entropy still checked.

Seeds are fixed offsets from training seed: model1270000+100*s;
warm/continuation schedules1280000+100*s (+phase offsets recorded in training
metadata); adversary initialization1290000+100*s. No seed chosen by outcomes.
Save true pre-update initialization, common warm states, continuation epoch0,
and **fixed epoch80 final states**. Intermediate diagnostics every5 epochs/final
are never selection candidates. Numerical failure preserves evidence and stops
that arm; no substitute early checkpoint. Frozen final maps/standardizers and
representative outputs are hashed before any downstream fitting and after audit.

## Final utility and independent audit

Same2048 valid downstream-fitting rows per binary task, exact prior subset seeds
1230000+100*s+j, task order residence/commute/income/civilian/public. Heads:
logistic C1,lbfgs500,tol1e−4; MLP64/32ReLU,Adam.001,batch256,40epochs, validation
epoch0/every5/final; seed1250000+100*s+j. Minimum downstream-validation log loss,
lexicographic candidate tie and earliest MLP epoch. Fit standardizers/intercepts
only on head-fitting rows. All reserved labels first enter heads **after release
freeze**. Reuse unchanged reference predictions only after exact input/fitting/
preprocessing/protocol checks. C/D heads freshly fit on exact final16D coordinates.

Independent attackers match the preceding pilot exactly: logistic; two120-epoch
64/32 MLPs; HistGradientBoosting150iterations,max15leaves,lr.1,L2=1,noearlystop,
minleaf20/5. Attacker cap4096; original subset seeds1240000+100*s+j; basefitseed
1260000+100*s+j, MLPsecond+10000; same fitting/validation pools and train-only
standardizers. Keep every candidate and curve. No transferred adversary is used
on reference banks/PCA; their existing independently fitted evidence is reused.

For each learned arm/attribute, evaluate saved final training adversary unchanged,
then fit **one120-epoch catch-up candidate**, starting its exact saved weights,
reset Adam.001,batch256, seed1300000+100*s+j, validation epoch0/every5/final,
minimum validation loss/earliest epoch. This is two catch-up trajectories per
arm(total4 per seed), not two restarts per attribute. Use a direct-input path
with identity standardizer so inherited input/output conventions remain exact;
verify prediction equality before the first update. Adversary weights are never
loaded into an incompatible dimensionality. Catch-up has warm20 plus continuation
240 equivalent fitting exposures before the added120 attacker-pool epochs;
report exact counts and unique pools, not an equal-exposure claim versus fresh
MLPs. Saved original weights and mapper remain immutable throughout catch-up.

Primary attack minimizes validation log loss among five independent candidates
plus catch-up for learned arms; saved-before-catch-up is diagnostic only. Also
report C/D using **only the matched five independent candidates**, per-family
selections, and diagnostic validation-AUROC selection among the same loss-selected
checkpoints. Full-schema undefined AUROC stays undefined. Save selections before
opening development evaluation. Never choose epochs/restarts/attacks on that pool.

Reuse fitting priors/exposed-one-hot control evidence only under identical data
and candidate recipes. Keep all nine race categories, per-pool/per-class support,
all control failures, and lack of coverage. Race code4 has no attacker-fitting
or validation support; a complete all-attribute success cannot be asserted.
No negative gain/undefined score/unsuccessful finite attack is a privacy guarantee.

## Frozen descriptive comparisons and reporting

Primary is paired **D-minus-C**, each task/attribute/seed, both full-primary and
matched-independent audit sets. Compare both learned arms to originalPCA,
PCA+LEACE and all four bank references, every arm reported without release selection.
Carry margins unchanged, **using originalPCA as parent for both learned arms**:

1. Each of three source-task losses <= originalPCA + .01nats.
2. With b=better unprotected rich-bank residential loss and H=b−PCA_loss>0,
   retain b−learned_loss >= H/2. Otherwise undefined.
3. Signed attack gain=fit-prior loss−selected attack loss. If originalPCA gain>0,
   report whether learned gain<=half that gain; coverage gates any policy pass.
4. Learned feature versus each bank: residence loss at least .01 lower and each
   attribute gain no more than .005 higher. Publish numerical inequalities even
   where race coverage limits an all-policy assessment.

Apply separately on validation and development evaluation, never average away
failed constraints. Coverage requires all categories in attackfit,attackval,
evaluation and positive recall in validation-selected exposed control; separately
flag training support. Three-valued logic preserves known failures and undefined
cases. No toy R² floor. Commute remains a weak-task diagnostic.
Unweighted natural log loss primary; probabilities clipped1e−12/renormalized.
AUROC, balanced accuracy, prevalence, all per-class metrics/support and signed
gains; PWGTP sensitivity uses identical predictions with no weighted fitting or
selection. Means/sample SDs descriptive only: seeds share the cohort. Separate
residence-versus-SEX/RAC1P plots, training curves, saved/caught-up comparisons.

## Execution and publication boundary

Freeze protocol/config/execution closure before scientific fitting. Focused tests
and miniature artificial pipeline first. Time one complete seed with references,
both arms/heads/independent/catch-up attacks, scoring/serialization. On M4 Pro CPU,
one numerical thread, total target1800s. If three-seed estimate exceeds it, retain
the complete seed without shrinking fixed training or audit budgets. No new sweep.
Publish actual source dependencies, compact metrics/selections/support/training
diagnostics/provenance/verification/reproduction/plots. Raw records, cached releases,
fitted objects and duplicate archives remain local with hashes and regeneration
instructions. Preserve historical execution hashes rather than replacing them
with publication commit. Commit/push only research branch, no PR/force/main change.
