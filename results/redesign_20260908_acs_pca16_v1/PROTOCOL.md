# Frozen PCA16 diagnostic — 2026-09-08 UTC

Question: does the literal first16-coordinate slice of the existing PCA32
release retain more residential-transfer utility than the learned C16/D16
bottlenecks? This is a fixed dimensionality control, without a new protection
method or representation fit. The original PCA32 remains the comparison parent.

This protocol and [configuration](config.json) are hashed with the execution
dependencies before fitting. There is one new release and no selection of its
components, dimension, map, or source model. No synthetic experiments repeat.

## Fixed data and map

Use the existing California2018 public ACS cohort, 30,000 people, seeds0/1/2,
original household pools, masks, task labels and fitting rows. The full
[data, schema, support and task definitions](../redesign_20260907_acs_transfer_v1/PROTOCOL.md)
and [latest support/exposed-control limitations](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md)
remain in force. No raw data, labels, masks, cohorts or unused households change.
These are DEVELOPMENT EVALUATION households that already informed the research
direction, not a new untouched confirmation sample. Seeds share one cohort.

For each seed, load the original `release_maps.joblib` PCA and the cached
`release_E_pca.npz` from the LEACE stage. Verify their recorded file hashes,
32 component rows, original mean, no whitening, descending explained variance,
and bitwise replay of cached PCA32 from original preprocessed covariates. Freeze
the first16 columns in their original order as immutable float32 arrays:
`PCA16 = original_PCA32[:, :16]`, 16 coordinates, 64 bytes per record. Slice before
any head-specific standardization. No fitting, whitening, coordinate search,
rotation, rescaling, eraser or nonlinear mapper is introduced. Release hashes
are saved before downstream fitting; evaluation contents are accessed only
after all head and audit choices are saved.

The first16 components account for approximately90.1–90.2% of preprocessed
fitting variance; this identity diagnostic does not use task outcomes. Task
labels never select the components. The original map hashes and component-row
identities are saved per seed. Original source provenance remains unchanged.

## Identical fitting and scoring

Fit fresh logistic and 64/32 ReLU MLP utility heads for same residence one year
ago (`MIG==1`), valid commute duration over20 minutes, income over$50,000,
civilian employed **and at work** (`ESR==1`), and public coverage. All original
eligibility/missing-label masks apply. Every head gets the identical2048
eligible fitting examples used previously. The MLP runs40 epochs; Adam.001,
batch256. Logistic is C1, lbfgs, max500 iterations. Validation log loss selects
checkpoints at epoch0/every5/final and the head family, with original tie rules.

For each recorded attribute SEX(2 classes) and RAC1P(all9), fit the same five
independent candidates: logistic; two120-epoch64/32 MLPs; HistGradientBoosting
with min-leaf20 or5,150 iterations,15 leaves, learning-rate.1 and L2=1, no early
stopping. Other existing settings remain unchanged. The4096 fitting cap means
all approximately3000 eligible attacker-fitting examples are used, exactly
matching the prior indices. All standardizers use only their fitting rows.
Validation log loss selects candidates and MLP checkpoints. Save each family,
restart, learning curve, update count, exposure, selection and AUROC diagnostic.
No PCA16 saved/training adversary exists; none is invented or transferred.

Reuse unchanged PCA32, PCA32+LEACE, C16, D16, rich neural/tree banks and their
erased variants, plus prior and exposed controls. Verify release, fitting-row,
preprocessing/recipe and prediction identities; do not refit their models or
audits. For C/D, primary audit rows are the existing `independent_selected`
choices among these same five candidates. Show previous catch-up-inclusive
choices separately, with their additional inherited260 passes of representation
training exposure plus120 fitting epochs disclosed. Larger candidate pools can
select different validation winners and need not improve development scores.
Use the original saved selections; never minimize across development outcomes.

Unweighted natural log loss is primary, with signed attack gain = fitting-prior
loss minus attack loss. Keep negative gains. Report accuracy, AUROC, balanced
accuracy, class supports and PWGTP-weighted sensitivity on identical predictions.
These are not official survey estimates. Missing race code4 fitting/validation
support and exposed-control limitations persist: unsupported/undefined categories
cannot count as protected; all9 categories remain. No universal privacy claim.

## Frozen analysis and budget

Central paired scores are PCA16 minus PCA32, C16 and D16 for every task and
attribute, on validation and DEVELOPMENT EVALUATION separately. Include all
three seeds and descriptive mean/sample SD; do not treat seeds as independent
population samples. Other original interfaces are contextual references.

Carry forward source losses at most.01 nats above **original PCA32**, for each
source task separately. Residential retention is at least half the positive
original-PCA32 advantage over the better unprotected rich bank, separately per
seed and split; no positive headroom means undefined. These are unchanged
descriptive references, not statistical tests or an all-policy admission gate.
Release maps are never selected using them. Report every task and attribute.

Better PCA16 than C16 would show that this simpler16-coordinate interface
preserves more usable residential signal under these heads. Coordinate count
alone would not explain the learned bottleneck loss. Worse PCA16 would concern
this specific truncation, not every16-dimensional representation. Equal stored
dimensions do not imply equal information; this experiment cannot uniquely
separate training objectives, optimization, geometry and dimensionality. Neither
outcome establishes PCRL novelty. A baseline need not satisfy the whole policy
before developing a method to address its measured weakness.

Run focused slicing, boundary, frozen-state, selection and paired-score checks,
including an artificial miniature20-candidate pipeline. Then time one complete
scientific seed, including data/map verification, all heads/auditors and scoring,
on the existing M4 Pro with one numerical thread. Expand unchanged to seeds1/2
only if the measured three-seed estimate fits600 seconds. If not, retain the
completed seed; never weaken the audit or choose a shorter favorable budget.
No new method, broad sweep or historical experiment rerun follows this diagnostic.
