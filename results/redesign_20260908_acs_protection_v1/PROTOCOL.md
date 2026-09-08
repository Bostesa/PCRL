# ACS simple-erasure feasibility — frozen protocol, 2026-09-08

**DEVELOPMENT EVALUATION.** The previous ACS test outcomes informed this
question. We reuse the identical people and original test pools without claiming
an untouched confirmation. Within this run, fitting, validation selection, and
evaluation remain separate. No unused households are recruited; they remain
available for a later frozen confirmation.

Question: does joint affine attribute erasure preserve residential-transfer
headroom of reusable features while reducing recorded SEX/RAC1P recoverability,
compared with equally treated prediction banks? This is a single-recipient
precursor, without PCRL, purpose conditioning, coalitions, new encoder training,
task search, or method novelty. The hypothetical one-time recipient may learn
income >$50,000, civilian employed and at work, public coverage, same residence
one year ago, and commute >20 minutes where defined. SEX/RAC1P recovery is the
empirical restriction. Authorized outcomes can themselves correlate with these
attributes. Public ACS is a simulation, not a confidentiality claim against
public-record linkage or arbitrary future inference.

## Immutable parent data and source objects

Start at `e765ced246be0f1c8d5bf8a131ccae5e576e9bf2`. Reuse the
[original protocol](../redesign_20260907_acs_transfer_v1/PROTOCOL.md),
[schema](../../docs/ACS_2018_SCHEMA_NOTES.md), raw CSV SHA256, 30,000-person
whole-household sample, seed 1200000, and split seeds 1210000+[0,1,2]. Adults
19–34 with positive person weights; ten original allowed covariates, source-only
preprocessing/encoder/banks/PCA, fixed class schemas and missing masks unchanged.
Source fields PINCP/ESR/PUBCOV and reserved MIG/JWMNP, their direct encodings,
IDs, weights, SEX/RAC1P remain excluded from encoder inputs.

Keep the seven original household-disjoint pools: representation fitting,
source validation, downstream fitting/validation, attacker fitting/validation,
and original test. Preserve exact row ordering and hashes. No source model is
refitted if verified saved objects exist. Cached parent development releases
are checked against original hashes; original selected models regenerate only
the necessary evaluation outputs and must match their historical hashes.
Any missing required parent artifact is a blocker or an explicitly recorded
regeneration using the original recipe, never an unreported replacement.

Income >50000 uses valid signed PINCP; zero/negative amounts are valid. ESR==1
means civilian employed **and at work**, not all employment. PUBCOV==1 means
public insurance. MIG==1 means same residence one year ago. JWMNP >20 uses only
valid integer times 1…200; missing/inapplicable rows are masked. Military
commuters with valid times remain included. Commute eligibility is known through
the common fitting/evaluation membership, never a model input; that membership
can itself disclose information. Attribute audits measure the numerical release
on the common cohort, not extra eligibility lists. All other missing labels are
masked, never converted to negatives. Original SEX 1…2 and RAC1P 1…9 retained.

## Ten primary arms, two references, one erasure recipe

Parents: A three binary source probabilities (3); B rich neural bank (26);
C rich tree bank (26); D compressed learned features (26); E original PCA (32).
Report each unchanged parent and exactly one fully fitted joint LEACE version,
with original coordinate dimension. No partial erasure, strength/dimension grid,
post-outcome map selection, or original probabilities concatenated back in.
F full allowed preprocessed covariates and fitting-prior distributions remain
references. Erased bank coordinates are unrestricted real features; refit heads,
never interpret their coordinates as calibrated probabilities.

Fit each eraser on **representation-fitting rows only**, using concatenated
fixed-schema 2+9 one-hot attribute targets. Use rows with both attributes valid;
publish per-attribute and complete-case support, retain all 11 columns including
absent categories, and flag gaps. This is the encoder-fitting population reused
for empirical calibration, not an independent calibration guarantee. Reserved
task labels never enter eraser fitting or any release-map/dimension selection.

Use installed `concept-erasure` 0.2.4 LEACE, affine mean preservation, float64
centered sample covariance (n−1 denominator), no ridge, shrinkage, or covariance
trace constraint, cross-covariance SVD tolerance 1e−10. To prevent float32
probability-simplex roundoff from becoming a whitened signal, first retain input
covariance eigenvectors with eigenvalue >1e−10 times the largest eigenvalue.
Fit the library eraser in that numerical support and lift to the original space.
Truncate lifted map singular values <=1e−10*max(1, largest singular value).
A fully removed map is exactly zero, giving a constant fitted mean. These fixed
numerical rank tolerances apply identically to all parents and are not protection
strengths. Record discarded variance, matrix spectra, covariance residuals and
affine-map numerical checks. Application is `(x−mean) @ P.T + mean` in float64,
then float32 storage. No coordinate scaling is claimed as information removal.

Freeze/hash each complete map and all releases before fitting heads. Report
stored dimension/bytes, centered numerical rank at the same covariance tolerance,
and spectral effective rank. Hash maps and source state before/after all fitting
and evaluation. Covariance diagnostics are empirical fitting-set statistics,
separate from validation/development predictive metrics; they certify neither
classification accuracy nor nonlinear privacy.

The diagnostic residual threshold is 1e−10 + 1e−8 times the pre-erasure maximum
absolute cross-covariance. The raw numerical inequality and schema coverage are
reported separately; an absent/constant concept column cannot pass the combined
covariance flag. All 11 indicators are marginals concatenated, not an 18-category
SEX-by-race interaction target. The fit-sample unmodified-library map is compared
numerically with the fixed stabilized map only as a rounding diagnostic; it is
not another selected release or an outcome-dependent map choice.

## Matched independent heads and audits

Every release gets the original two downstream candidates for **all five** binary
tasks: logistic C=1, lbfgs max500 tol1e−4; 64/32 ReLU MLP, Adam .001, batch256,
40 epochs, no weighting/regularization/dropout, validation at epoch0/every5/final.
Select minimum downstream-validation log loss, ties by candidate name then
earliest epoch. Exactly 2,048 eligible downstream fitting labels (or all if fewer),
same examples/order/seeds across releases. Preserve original residential/commute
subset seeds 1230000+100*s+j and head seeds 1250000+100*s+j with j=0,1; append
the source binaries at j=2,3,4. Fit all standardizers/intercepts only on designated
head-fitting rows. Source binaries are probed after freezing too.

For each attribute/release, attacker fitting uses original cap4096 and subset
seed1240000+100*s+j, typically all ~3000 pool rows; all valid attacker-validation
rows. Five candidates, identical opportunities across ten arms, F, and exposed
controls: existing logistic; **two** 64/32 MLP initializations, **120** epochs
each with the same Adam/batch/validation rules; HistGradientBoosting **150**
iterations,15 leaves,lr.1,L2=1,no early stopping, minimum leaf **20 or 5**.
Base seed1260000+100*s+j; second MLP seed adds10000. Fit preprocessing on
attacker-fitting rows only. Training exposure, actual optimizer/boosting steps,
warnings, selected epochs, and validation curves are recorded. There is no
training adversary to transfer: all independent auditors fit the exact immutable
final map. No additional map refresh occurs.

Primary attacker minimizes validation log loss across all five candidates;
also select best per family. Report every restart/configuration and select a
validation-AUROC diagnostic among these same log-loss-selected checkpoints.
AUROC does not select a different epoch trajectory. If full-schema macro AUROC
is undefined, report that and separately label any observed-class macro statistic.
No evaluation-based attacker tuning. Fit-prior uses Laplace1/category. Exposed
one-hot controls undergo the identical five-candidate fitting procedure. Publish
every control's per-class failures, not just the best result.

All head/attacker objects and selections are saved before within-run evaluation
opens. Evaluate exact selected objects, also report every predeclared candidate.
Serialized validation/evaluation probabilities permit scoring replay without
retraining. Primary unweighted natural log loss clips probabilities at1e−12
then renormalizes. Report AUROC, balanced accuracy, accuracy, class prevalence,
all category support/recall/precision/F1 and undefined metrics. PWGTP sensitivity
uses identical predictions, never weighted fitting/selection. No official Census
estimates or survey/design-based uncertainty is claimed.

## Predeclared descriptive comparisons

On validation and DEVELOPMENT EVALUATION separately, for every seed and parent:

1. Report erased-minus-unprotected loss/other metrics for every task/attribute,
   selected and family-specific. Each of the three source binary losses must
   increase by at most **.01 nats** to meet its provisional preservation reference.
2. Let b be the better unprotected rich bank's residential loss on that seed and
   split. If H=b−parent_loss>0, retention means b−erased_loss >= **H/2**. Otherwise
   retention is undefined, never an invented pass or new task search.
3. Attribute attack gain is **prior_loss−attack_loss**, signed and reported.
   If parent gain>0 and coverage adequate, ask if erased gain <= **half** parent
   gain. Nonpositive baseline gains are undefined. Coverage requires all schema
   categories present in attacker fit, attacker validation, evaluated split,
   and positive recall for every category under the validation-selected exposed
   control on that split. Gaps/failures make the assessment limited. Also flag
   eraser-fitting gaps. No all-attribute-policy pass with incomplete coverage.
4. For each learned/PCA feature arm versus each rich-bank arm, compare residential
   loss at least **.01** lower while each attribute's gain is at most **.005**
   higher. Publish the scalar inequalities even if coverage makes an all-policy
   assessment limited. These are descriptive margins, not privacy budgets,
   statistical noninferiority, or release-selection rules.

Report criteria separately and jointly with three-valued true/false/undefined
logic; known failures remain failures even with other undefined conditions.
Commute remains reported without new headroom requirements. Report all seeds,
means/sample SDs and paired differences; seeds share the cohort and are not
independent population samples. Two plots show residential loss (lower better)
versus SEX or RAC1P attack loss (higher means less measured recovery), with
unprotected/erased pairs and priors. Neither audit failure nor covariance success
establishes privacy. A's task-output gain is a practical reference, not a lower
bound or automatic acceptable threshold.

## Bounded execution and publication

Freeze this protocol/configuration/execution source hashes before fitting new
models. Targeted numerical/boundary/comparison tests and a miniature fixture
precede one complete seed. Time loading, erasure, heads, all attacks/controls,
evaluation and integrity; estimate two remaining seeds before expansion.
One numerical thread on existing M4 Pro CPU; target total experiment <1800s.
If estimate exceeds budget retain the complete one-seed pilot. Do not weaken
audits or competitors after results, expand grids, or use paid resources.

Keep source dependencies, compact per-candidate/class metrics, support/split
hashes, maps' spectra/covariance diagnostics, selections/curves, paired results,
plots, runtime/verification/reproduction and source identities on GitHub. Raw
records, cached arrays, maps and fitted estimators remain local with SHA256 and
regeneration instructions. Preserve original provenance and historical bytes.
Commit/push only this work to `ablations-facct-2026-07-24`, no PR/force/main change.
