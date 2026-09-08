# ACS task-identity transfer: frozen protocol, 2026-09-07

This is a controlled scientific benchmark on public California 2018 ACS person
records. A hypothetical owner makes one fixed per-record release. A downstream
researcher can fit new supervised heads, but cannot retrain or query the owner's
encoder. Every interface receives identical downstream labels and fitting
privileges. A trusted service is a deployment alternative, not an admission
gate for this experiment. Public-record linkage confidentiality is not claimed.
No PCRL, LoRA, erasure, adversarial encoder objective, or privacy threshold is used.
No historical synthetic audit is repeated.

The question is whether source-trained reusable features preserve useful
information for **two fixed withheld task identities** beyond competitive source
prediction banks. It is cross-task transfer on contemporaneous survey attributes,
not longitudinal forecasting. Models, tasks and grids will not be changed in
response to final performance. Starting publication commit:
`5f162ab37cc9e1caa391e6c964711fa85cc86a14` on `ablations-facct-2026-07-24`.

## Data, codes and release inputs

Use `data/folktables/2018/1-Year/psam_p06.csv`, **not** the processed parquet.
Verify codes against [Census 2018 documentation](https://www.census.gov/programs-surveys/acs/microdata/documentation/2018.html)
and [Folktables source](https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py).
The schema review and data hashes are recorded alongside the run.

Common cohort: integer age 19–34 inclusive and positive PWGTP. This common
young-adult population supports socioeconomic, residential and commuting tasks;
it adapts, and does not reproduce all canonical Folktables population filters.
Person deduplication uses SERIALNO/SPORDER; conflicting duplicates fail.
Grouping uses SERIALNO, including group-quarter records as represented in PUMS.
All cohort members of a selected group remain together. No protected labels
are constructed or inferred from appearance.

| Role | Field, valid codes and task definition | Missing/eligibility policy |
|---|---|---|
| Source income | PINCP, signed total past-12-month personal income, −19998…4209995; binary `>50000`, plus at most eight distribution bins | Missing/out-of-range excluded from this head; valid zero/negative income retained. Nominal recorded amounts, no ADJINC transformation. Seven quantiles fit on representation-training income only, repeated edges merged. |
| Source labor force | ESR 1…6; categories retained; binary `ESR==1` means **civilian employed and at work** | Missing/invalid masked; category 2 is employed but absent, not the binary positive. |
| Source public coverage | PUBCOV 1=yes, 2=no | Missing/invalid masked, not made negative. |
| Withheld residential mobility | MIG 1=same residence, 2=abroad, 3=different US/PR residence; binary `MIG==1` is **same residence one year ago** | Only valid 1…3; missing masked. |
| Withheld commute duration | JWMNP valid integer 1…200; binary `>20` minutes | Only valid commute values; missing/inapplicable nonworkers and home workers excluded, including from fitting. Valid armed-forces commuters retained. All interfaces receive the same eligible examples; learner thereby knows commuting eligibility, but no eligibility bit or actual time is an input. |
| Attribute audit | SEX original codes 1,2; RAC1P original categories 1…9, unreduced | Invalid/missing masked. Category labels, every pool's supports and per-class metrics published. No unsupported category silently counted as success. |

Input allowlist (10 fields): numeric AGEP, WKHP; categorical SCHL, MAR, RELP,
CIT, DIS, DEAR, DEYE, DREM. Numeric median/mean/SD fit only representation-training
rows; explicit numeric missing flags. Categorical levels discovered only there;
separate missing/invalid and unseen-valid slots. Float32 dense releases.
Original WKHP is usual hours in the past year, a permitted correlated predictor;
its missingness is retained. No independence or temporal precedence is claimed.

Exclude all five task fields, task recodes, income/insurance components,
current-work answer fields, migration geography, commuting fields, allocation
flags, identifiers, all weights, SEX/RAC1P and their recodes. FER is excluded:
its applicability exposes SEX in this cohort. MIGSP/MIGPUMA applicability exposes
same residence. COW is conservatively excluded because some codes/missingness
directly determine an ESR category. NATIVITY is redundant with CIT and omitted.
Only the allowlist is passed to preprocessing; labels and masks are separate.

## Samples, access and selection boundaries

One deterministic whole-household random prefix, sampling seed 1200000, capped
at 30,000 cohort persons, before models. Same sampled population for all seeds.
Split seeds 1210000+[0,1,2] shuffle sorted household IDs. Rounded cumulative
household fractions define seven pools: representation fit 35%, source validation
10%, downstream fit 15%, downstream validation 10%, attacker fit 10%, attacker
validation 10%, final test 10%. Publish row/group hashes and pool/task counts.
Person counts vary with household sizes; no split regenerated on outcomes.

Schema/support checks may examine label validity and counts in every pool.
Final-test outcomes cannot enter fitting, preprocessing, checkpoint selection or
head/attacker selection. No test predictions are scored until **all** selections
for that seed are durably saved and hashed. Test evaluation uses exactly those
objects, with integrity hashes before and after. No test-based release selection.

Source fitting APIs accept only income, ESR, PUBCOV-derived labels. Withheld MIG
and JWMNP labels never enter encoder/bank fitting, source-validation loss,
auxiliary losses, compression or reconstruction (none used). After release
freezing, designated downstream pools provide those labels. Each task uses a
fixed deterministic subset of at most **2,048 valid downstream-fitting rows**;
there is one labeled-data budget. Every release uses exactly those same rows.
Attackers use at most 4,096 valid attacker-fitting rows (normally the entire
roughly 3,000-row pool). All validation rows with valid labels are used.

## Frozen interfaces and fitting budgets

Shared encoder: allowed-input dimension →64 ReLU→64 ReLU→32 linear features,
five heads: income binary (2 probabilities), income bins (≤8), ESR (6), PUBCOV
(2), joint binary state (8). Joint state is `4*income + 2*(ESR==1) + (PUBCOV==1)`
on complete source labels only. All eight states must be observed in source fit
and source validation to release joint probabilities; otherwise both rich banks
omit the joint output and report limited support. Rare states are not relabeled.
Each source head has a fixed schema, per-head missing masks, and equal weight
in mean cross entropy. Missing head minibatches contribute zero with denominator
five. Two Adam learning rates {.001,.003}, batch 256, 60 epochs each; true epoch-0
and all-epoch source-validation loss select the checkpoint/configuration. Both
configs have identical initialization and batch schedule per seed. No dropout/BN.

| Interface | Content and fitting |
|---|---|
| A_binary_bank | Three binary source probabilities from the selected shared encoder |
| B_rich_bank | All five source-head probability vectors, supported joint included; 26 dimensions when eight bins supported |
| C_tree_bank | Same source probability schema, independently fitted HistGradientBoostingClassifier heads on allowed preprocessed inputs. Two settings: 15/31 leaves, 150 iterations, learning rate .1, L2=1, min leaf20, no early stopping. Select the complete bank by equal-head source-validation cross entropy. |
| D_features | The identical selected shared encoder's 32 frozen coordinates |
| E_pca | PCA of allowed preprocessed inputs, dimension min(32,input rank bound), fit on representation-training inputs only |
| F_covariates | Full allowed preprocessed inputs, an information-rich reference |
| D_compressed | If 32 exceeds the richer B/C bank's dimension, PCA of D to that dimension, fit on representation-training features only; no held-out labels |

Store each interface's dimension, float precision, bytes/record, fitting time
and timed inference. Probability-vector simplex redundancies count toward stored
bytes; equal stored dimensions are not equal intrinsic capacity. F is a deployment
reference, not a matched-capacity claim. A/B/D share their source training cost.

Downstream candidates for each task/interface: logistic regression C=1,
max_iter=500; one 64→32 ReLU MLP, Adam .001, batch256, 40 full epochs, checkpoint
selection by validation log loss at epoch0/every5/final. One initialization per
family, same schedules/seeds across interfaces. Fit release-coordinate
standardizers only on downstream-fitting rows. Select family by validation log
loss; report each candidate and selected result. Do not class-reweight training.

Attribute attackers: independent logistic/MLP under the same definitions and
one HistGB (150 iterations, 15 leaves, learning rate .1, L2=1, min leaf20,
early_stopping=False). Fit preprocessing on attacker-fitting examples. Select
strongest by attacker-validation log loss; report every family and all categories.
Prior-only fitting and deliberately exposed one-hot SEX/RAC1P releases are
controls, with actual fitting through all three attacker families. Exposed
controls are inaccessible-label diagnostics, not competitors. A prior-only
transfer head is also reported. No continued adversary exists: these unprotected
encoders have never trained adversaries, and all auditors are freshly fitted to
the exact final frozen interface.

## Metrics, interpretation and runtime

Primary metrics: unweighted held-out log loss (nats, lower better), AUROC,
balanced accuracy, accuracy and prevalence for each task. Attribute audits report
log loss, accuracy, per-class one-vs-rest AUROC/precision/recall/support and macro
metrics with explicit missing-support flags. Probabilities clipped at 1e−12 then
renormalized for log loss. Fit-class gaps are retained in a fixed category schema.
Undefined metrics are null, not zero. No linear covariance statistic substitutes
for predictive audit performance.

PWGTP-weighted sensitivity repeats final metrics on identical predictions;
neither fitting nor model selection is weighted. Target population is this sampled
benchmark cohort. These are not official survey estimates. Three-seed sample SDs
are descriptive; overlapping cohort draws are not independent survey replicates
or design-based uncertainty.

Primary paired comparisons: D minus B and D minus C on **each** held-out task,
under identical validation-selected downstream protocol. Report all seed pairs,
means/sample SDs, and within-family head results. A 0.01-nat reduction is a
predeclared practical effect reference, not a significance test or privacy gate.
PCA/F distinguish learning from simply retaining more information. Feature gains
justify further study only. Bank ties support a simpler interface for these two
tasks, not all future tasks. Weak D with strong F implicates encoder limitations;
weak F makes this transfer test uninformative. Attribute recoverability is measured,
not an exclusion criterion. There is no protection or coalition claim in this run.

Use local CPU, one numerical thread, three seeds. First run focused unit tests
and a miniature artificial-data plumbing check (no scientific outcomes). Then
time a complete seed including loading/preprocessing/source fits/heads/audits/test.
Estimate remaining runtime before expansion; target <2 hours. If needed, reduce
a common declared budget **before final testing**, retain superseded artifacts
and record amendment; never weaken only a competitive baseline/auditor. No extra
grid, changed targets, paid resources, dependency additions or multi-day jobs.

Artifacts: config, schema notes, protocol-freeze hashes, support/split records,
selection-before-test files, per-seed/family/class metrics, paired results, plot,
source and local fitted-artifact manifests, runtime and reproduction commands.
Raw records, checkpoints, arrays and fitted estimators remain local with hashes.
Compact evidence and actual source dependencies will be committed and pushed to
the existing research branch; main and historical result bytes stay unchanged.
