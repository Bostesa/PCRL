# PCRL redesign review index

**Latest addition, 2026-09-08 UTC:** the [learned two-purpose coalition study](../results/redesign_20260908_acs_coalition_v1/RESEARCH_DECISION.md) completed all18 F/P × I/Iplus/J paired systems across three seeds. Feature J lowers mean coalition SEX recovery beyond ordinary and stronger-local protection, with residential costs. Race conclusions depend on attack scope; prediction J has no robust joint advantage over Iplus. Residence improves over coordinated predictions, but commute gains are small and direct E remains better on both reserved tasks. Original utility margins and race-support limits remain. See the [dated appendix](#appendix--2026-09-08-learned-two-purpose-coalition-study), [full table](../results/redesign_20260908_acs_coalition_v1/TABLE.md), and [one next experiment](../results/redesign_20260908_acs_coalition_v1/NEXT_DESIGN.md).

**Previous addition, 2026-09-08 UTC:** the [restricted-input, fair source-control, and exact coordination study](../results/redesign_20260908_acs_restricted_inputs_v1/RESEARCH_DECISION.md) is complete. Restriction lowers mean independent attribute recovery but costs source utility; E's pooled race counterexample prevents a general protection claim. Source-only banks explain the practical source gains under comparable fitting exposure, while representations retain a residential advantage. The exact coordinated prediction mechanism helps one issuance but reverses against independent noise after four fresh issuances. See the [dated appendix](#appendix--2026-09-08-restricted-inputs-source-controls-and-exact-coordination), [all results](../results/redesign_20260908_acs_restricted_inputs_v1/TABLE.md), and [one proposed next study](../results/redesign_20260908_acs_restricted_inputs_v1/PURPOSE_COORDINATION_DESIGN.md).

**Previous addition, 2026-09-08 UTC:** the [selective-preservation and reconstruction study](../results/redesign_20260908_acs_selective_preservation_v1/RESEARCH_DECISION.md) completed30 new learned releases, six reused raw-teacher releases and six direct teachers. Real-label erasure improves mean residential loss over shuffled erasure, but its apparent attribute-specific benefit weakens under catch-up. Removing reconstruction lowers recovery with a residential cost; all learned source margins pass. See the [dated appendix](#appendix--2026-09-08-selective-preservation-and-reconstruction), [full comparison](../results/redesign_20260908_acs_selective_preservation_v1/TABLE.md) and [one proposed next design](../results/redesign_20260908_acs_selective_preservation_v1/NEXT_DESIGN.md).

**Previous addition, 2026-09-08 UTC:** the [bounded PCA16 preservation study](../results/redesign_20260908_acs_preservation_v1/OVERNIGHT_RESEARCH_DECISION.md) completed all eight conditions across three seeds and the full360-epoch audit extension. Preservation improves mean residential transfer under both weights, while strong persistent preservation retains teacher-like leakage and sharply reduces the measured C/D difference. Warmup-only preservation offers a limited favorable tradeoff, with source costs and weighted/seed qualifications. See the [dated appendix](#appendix--2026-09-08-pca16-preservation-and-nested-stronger-audits).

**Previous addition, 2026-09-08 UTC:** the [PCA16-initialized matched C/D experiment](../results/redesign_20260908_acs_pca16_init_v1/RESEARCH_DECISION.md) did not establish a better final tradeoff. Source utility remains preserved, final protected residential performance ties historical D, and weighted sensitivity reverses the common-warmup mean change. See the [dated appendix](#appendix--2026-09-08-pca16-initialized-matched-cd).

**Previous addition, 2026-09-08 UTC:** the [fixed PCA16 control](../results/redesign_20260908_acs_pca16_v1/RESEARCH_DECISION.md) retains more residential utility than C16/D16 on the primary mean comparison, while original PCA32 remains better. The advantage shrinks under person weighting, and attributes remain recoverable. See the [dated PCA16 appendix](#appendix--2026-09-08-fixed-pca16-control). No new representation or protection method was fitted.

**Previous addition, 2026-09-08:** the [fixed nonlinear ACS bottleneck pilot](../results/redesign_20260908_acs_bottleneck_v1/RESEARCH_DECISION.md) is complete. Matched protection modestly reduces SEX recovery, but neither learned arm meets the residential/attribute tradeoff. Catch-up reduces the apparent race benefit. See the [dated appendix](#appendix--2026-09-08-fixed-nonlinear-acs-bottleneck). No PCRL advantage is established.

**Earlier addition, 2026-09-08:** the bounded ACS joint-erasure feasibility screen
is complete. Start with its [research decision](../results/redesign_20260908_acs_protection_v1/RESEARCH_DECISION.md)
and the [dated appendix below](#appendix--2026-09-08-acs-protection-feasibility).
PCA retained some residential transfer, but the declared source-utility and
attribute tradeoff was not met. This is development evaluation, not confirmation
or evidence of PCRL efficacy.

**Later update, 2026-09-07:** the defined ACS transfer experiment is now complete.
See the [dated appendix below](#appendix--2026-09-07-acs-transfer-completed) and
[research decision](../results/redesign_20260907_acs_transfer_v1/RESEARCH_DECISION.md).
The following original redesign summary remains historical.

**Research decision: redefine the problem before further method development.**
Prediction-only release passed the strengthened fixed-task audits for all three
seeds. Full representations remained leaky. No PCRL method advantage has been
established. These are bounded empirical results, not universal privacy guarantees.

Start with the [research report](../PCRL_NEXT_STAGE_RESEARCH_REPORT.md), then the
[implementation and experiment status](../PCRL_REDESIGN_STATUS.md),
[application selection](PCRL_APPLICATION_SELECTION.md), and
[focused primary-source comparison](PCRL_PRIOR_WORK.md). The older headline
experiments elsewhere in this repository are historical; this publication does
not revalidate them. No real-data representation-method pilot was admitted.

## What changed in the implementation

- Fixed absent/constant-class scoring: explicit class schemas, support and
  variance masks; undefined scores cannot pass privacy checks or drive dual
  updates. Inspect [training scores](../pcrl/training/losses.py),
  [aggregate and dominant-axis scores](../pcrl/purposes/verification.py),
  [certificates/reporting](../pcrl/evaluation/certificates.py), and
  [regressions](../tests/test_scoring_support.py).
- Used each constraint's own threshold in selection and reporting, including
  concatenated releases. Saved genuine initialization checkpoints with zero
  optimizer updates and consistent selected states. Inspect the
  [V2 trainer](../pcrl/training/v2_trainer.py),
  [constraint optimizer](../pcrl/training/proxy_lagrangian.py), and
  [training regressions](../tests/test_redesign_training.py).
- Made random/pretrained initialization explicit, preserved evaluation behavior
  for frozen backbone BatchNorm/dropout, and added separate per-purpose erasure
  while retaining shared-union erasure. Inspect the
  [canonical runner](../experiments/run_v2_dataset.py),
  [encoder](../pcrl/models/encoder.py), and [LoRA/erasure code](../pcrl/models/lora.py).
- Retired the false universal R²-to-classification-accuracy guarantee and its
  dependent nonlinear certificate. The exact 20-observation example has affine
  least-squares R² = 0 and threshold accuracy = 90%. See the
  [mathematical correction](ACCURACY_CERTIFICATE_RETIREMENT.md) and
  [regression example](../tests/test_accuracy_bound.py). Valid covariance and
  least-squares statements remain distinct from classification accuracy.

## Experiments and evidence

All tables retain failed constraints and negative predictive R². A negative
predictive score is not negative information. Each directory contains its
original protocol, configuration, selection/split records, raw per-seed metrics,
provenance and available verification logs. The
[publication manifest](../results/redesign_publication_20260907/artifact_manifest.json)
identifies exactly which original artifacts are included or remain local.

| Stage | Question and observation | Evidence | Runner |
|---|---|---|---|
| Gaussian purpose conflict | Separate LEACE retained each purpose's task; shared-union erasure destroyed both. Restricted PCRL tied separate LEACE. The approximate-leakage composition example does not refute exact zero-covariance composition. | [Protocol](../results/redesign_20260907_gaussian_v1/PROTOCOL.md), [table](../results/redesign_20260907_gaussian_v1/TABLE.md), [summary](../results/redesign_20260907_gaussian_v1/summary.json), [composition](../results/redesign_20260907_gaussian_v1/composition_stress.json) | [Gaussian runner](../experiments/run_redesign_conflict.py) |
| Nonlinear upstream adaptation | Task-pretrained features had almost no erasure-related utility loss. Linear ridge-R² protection before erasure gave no feasible advantage over task-only adaptation. | [Protocol](../results/redesign_20260907_nonlinear_upstream_v1/PROTOCOL.md), [table](../results/redesign_20260907_nonlinear_upstream_v1/TABLE.md), [analysis](../results/redesign_20260907_nonlinear_upstream_v1/ANALYSIS.md) | [Upstream runner](../experiments/run_nonlinear_conflict.py) |
| Nonlinear release protection | Compared simple releases, task-only adaptation, fixed adversarial penalties and standard projected dual weights at individual task floors of .99. Prediction-only passed; no full-representation method passed all protection criteria. | [Protocol](../results/redesign_20260907_nonlinear_release_v1/PROTOCOL.md), [table](../results/redesign_20260907_nonlinear_release_v1/TABLE.md), [analysis](../results/redesign_20260907_nonlinear_release_v1/ANALYSIS.md), [selections](../results/redesign_20260907_nonlinear_release_v1/selected_configurations.json) | [Release runner](../experiments/run_nonlinear_release.py) |
| Frozen-adversary diagnostic | Continued saved adversaries recovered more leakage than the previous independent auditors on all 30 test target/checkpoint pairs. The final eraser refresh contributed a transfer mismatch; catch-up does not uniquely identify moving representations as the cause. | [Protocol](../results/redesign_20260907_frozen_adversary_v1/PROTOCOL.md), [table](../results/redesign_20260907_frozen_adversary_v1/TABLE.md), [analysis](../results/redesign_20260907_frozen_adversary_v1/ANALYSIS.md), [paired targets](../results/redesign_20260907_frozen_adversary_v1/paired_target_metrics.json) | [Diagnostic runner](../experiments/run_frozen_release_diagnostic.py) |
| Strengthened prediction-only audit | Fresh linear, longer-budget MLP and boosted-tree attacks covered all five forbidden relationships. Scalar releases passed all three seeds; the frozen full-representation positive control remained leaky. | [Protocol](../results/redesign_20260907_prediction_audit_v1/PROTOCOL.md), [table](../results/redesign_20260907_prediction_audit_v1/TABLE.md), [analysis](../results/redesign_20260907_prediction_audit_v1/ANALYSIS.md), [every target/family](../results/redesign_20260907_prediction_audit_v1/PER_TARGET.csv), [plot](../results/redesign_20260907_prediction_audit_v1/audit_scores.png), [independent verification](../results/redesign_20260907_prediction_audit_v1/INDEPENDENT_REVIEW.md) | [Audit runner](../experiments/run_prediction_audit.py) |
| Application screen | HAR, Diabetes and ACS did not establish a coherent, measured need for richer releases over predictions, a competitive prediction bank or trusted computation. Development-only policy checks were run; no method comparison or final real-data test was run. | [Protocol](../results/redesign_20260907_application_screen_v1/PROTOCOL.md), [table](../results/redesign_20260907_application_screen_v1/TABLE.md), [analysis](../results/redesign_20260907_application_screen_v1/ANALYSIS.md), [metrics](../results/redesign_20260907_application_screen_v1/metrics.json), [decision](../results/redesign_20260907_application_screen_v1/selection_record.json) | [Screen runner](../experiments/screen_pcrl_applications.py) |

## Where to inspect the experimental boundaries

The synthetic policy is fixed: P1 predicts U and prohibits V/S; P2 predicts V
and prohibits U/S. Combined access permits U/V and prohibits only S. Inspect
`PROHIBITED`, `THRESHOLDS`, `make_data`, `generator_parameters`, and
`calibrate_and_apply` in the [nonlinear runner](../experiments/run_nonlinear_conflict.py).
The Gaussian generator is a separate sanity reference, not the nonlinear map.

| Review concern | Exact implementation and checks |
|---|---|
| Genuine task pretraining, identical starts and upstream gradients | [Pretraining/adaptation helpers](../experiments/nonlinear_conflict_training.py), [tests](../tests/test_nonlinear_conflict.py) |
| Fixed penalty versus projected dual adversarial objective; training eraser refresh and gradient handling | [Nonlinear release training](../experiments/nonlinear_release_training.py), [matched budgets/selection tests](../tests/test_nonlinear_release.py) |
| Frozen final encoder/eraser; original preprocessing; continued versus fresh adversaries | [Frozen loader and integrity checks](../experiments/run_frozen_release_diagnostic.py), [adversary fitting helpers](../experiments/frozen_release_adversaries.py), [tests](../tests/test_frozen_release_diagnostic.py) |
| Independent affine/MLP fits and evaluation-set centering for predictive R² | [Original probes](../experiments/nonlinear_conflict_probes.py) |
| Stronger scalar-compatible MLP/tree attacks, exposure counts, validation selection | [Strengthened attackers](../experiments/prediction_release_attackers.py), [attacker tests](../tests/test_prediction_release_attackers.py) |
| Separate fitting/calibration/attacker/validation/test roles; saved selection before fresh test | [Audit runner](../experiments/run_prediction_audit.py), [boundary tests](../tests/test_prediction_audit.py); original `seed_*/split_manifest*.json` and `selection_before_test.json` in each result directory |
| Real-data development-only access and subject-aware task-output reference | [Screen source](../experiments/screen_pcrl_applications.py), [screen tests](../tests/test_application_screen.py) |
| Reporting without retraining or reselection | [Upstream report](../scripts/summarize_nonlinear_conflict.py), [release report](../scripts/summarize_nonlinear_release.py), [diagnostic report](../scripts/summarize_frozen_release_diagnostic.py), [prediction audit report](../scripts/summarize_prediction_audit.py) |

The complete helper modules are published alongside the runners. Their imports
also use existing `pcrl` package modules; copying just a runner is insufficient.

## Application decision and remaining question

HAR's proposed coarse-activity purpose intentionally reveals information about
fine activity, so prohibiting all fine-activity information is contradictory.
A learned coarse prediction exposed this in the subject-aware development
check. Its existing six-class task-score bank also covers the proposed
coarsenings; a meaningful independent future task family was not established.
Diabetes lacked a defensible task-timing/interface specification and independent
authorized transfer family. ACS has a plausible flexible-query motivation, but
the recipients, permitted task family and reason a bank or trusted query service
would be insufficient remain unspecified. These findings do not establish that
all reusable representations are unnecessary.

Before the proposed ACS experiment, specify the data owner and recipients;
required/permitted/forbidden labels; permissible coalitions; genuinely meaningful
held-out task identities; household/person-aware example splits; the trusted
service alternative; a competitive score bank; and utility/leakage criteria that
account for task-output correlations. Then compare that bank with frozen
unprotected features for authorized transfer and task-output leakage **before**
adding PCRL optimization. No ACS experiment was run in this publication.

Unsupported claims include universal privacy from unsuccessful attacks, a
classification-accuracy certificate from R², novel protection from LoRA/multiple
purposes/standard dual optimization alone, an advantage over matched strong
controls, a demonstrated real-data transfer need, and a publication-level
novelty claim. The [prior-work matrix](PCRL_PRIOR_WORK.md) is focused, not exhaustive.

## Availability, provenance and reproduction

See [publication/reproduction notes](PCRL_PUBLICATION.md) for the exact boundary
between committed evidence and local-only artifacts, environment versions,
dependency order and safe fresh-run instructions. Raw datasets, model and
attacker checkpoints, eraser arrays, cached releases, optimizer/batch schedules,
environments, credentials and duplicate source archives are not published.
The omission manifest records local paths, byte sizes and SHA256 hashes. There
is no claimed public checkpoint download: exact saved-checkpoint replay requires
the local artifacts; otherwise regenerate the synthetic stages in dependency
order. Numerical reproduction is distinct from byte-identical checkpoint replay.

Historical starting commits, source hashes, protocol freezes, metrics and logs
are unchanged. The publication commit packages that evidence; it is not the
commit at which the experiments were originally executed. Packaging checks are
recorded separately in [publication validation](../results/redesign_publication_20260907/VALIDATION.md).

## Appendix — 2026-09-07: ACS transfer completed

The earlier application screen is preserved, with a dated admission amendment.
A defined one-time release is sufficient to test transfer; trusted services remain
deployment alternatives rather than empirical admission gates. This controlled
public-data benchmark makes no Census deployment, public-record linkage privacy,
PCRL protection or novelty claim. No synthetic audit was repeated.

On30,000 sampled California2018 ACS adults, source-only representations improved
same-residence transfer over rich neural and tree banks in all three seeds.
Mean paired log-loss gains were .011767 and .018594 nats, respectively. Plain
PCA was stronger still. Commute transfer was weak, with banks and learned features
practically tied. Recorded sex/race remained recoverable; rare race-category
support was incomplete and the nine-class schema was retained.

Start with the [ACS research decision](../results/redesign_20260907_acs_transfer_v1/RESEARCH_DECISION.md),
[complete tables](../results/redesign_20260907_acs_transfer_v1/TABLE.md),
[paired/weighted analysis](../results/redesign_20260907_acs_transfer_v1/ANALYSIS.md),
[transfer plot](../results/redesign_20260907_acs_transfer_v1/transfer.png),
and [frozen protocol](../results/redesign_20260907_acs_transfer_v1/PROTOCOL.md).
[Schema definitions](ACS_2018_SCHEMA_NOTES.md) verify Census codes, eligibility,
input exclusions and direct-answer aliases.

Review [raw-cohort/split/mask preprocessing](../experiments/acs_transfer_data.py),
[source-only encoder and banks](../experiments/acs_transfer_models.py),
[downstream and audit heads](../experiments/acs_transfer_heads.py), and the
[runner's frozen-release and saved-selection test gate](../experiments/run_acs_transfer.py).
[Reporting](../scripts/summarize_acs_transfer.py) reads saved metrics without
refitting. Each seed has source selection, preprocessing, selection-before-test,
raw metrics and omitted-artifact hashes.

See [verification](../results/redesign_20260907_acs_transfer_v1/VERIFICATION.md),
[score replay](../results/redesign_20260907_acs_transfer_v1/SCORE_REPLAY.json),
[reproduction/availability](../results/redesign_20260907_acs_transfer_v1/REPRODUCTION.md)
and the [compact-evidence manifest](../results/redesign_20260907_acs_transfer_v1/publication_manifest.json).
Source, tests and compact evidence are published; raw data, fitted models and
cached arrays stay local. Total three-seed experiment process wall was158.614s
on one CPU numerical thread, Apple M4 Pro.

This adds evidence for information-retaining interfaces on one reserved task,
with PCA as a strong simple alternative. It supports a bounded baseline-led
protection-feasibility study after explicit purpose/utility/leakage criteria,
not a reason to prioritize PCRL optimization. No PCRL method advantage has
been established; the prior synthetic prediction-only success and full-feature
leakage results remain unchanged.

## Appendix — 2026-09-08: ACS protection feasibility

Starting from `e765ced246be0f1c8d5bf8a131ccae5e576e9bf2`, the new screen reused
the exact source models, cohort, masks and household pools. Five parents—three
probabilities, rich neural/tree banks, compressed learned features and PCA—were
each released unchanged and after one joint SEX2/RAC1P9 LEACE fit. Full covariates,
fitting priors and exposed-label controls remain references. Source/eraser maps
were frozen before matched heads for all five authorized binary tasks and five
independent audit candidates per attribute. The original test households are
explicitly **DEVELOPMENT EVALUATION**; unused households remain unused.

The [research decision](../results/redesign_20260908_acs_protection_v1/RESEARCH_DECISION.md)
reports partial PCA residential retention, failures of the all-three-source
.01-nat preservation reference for every erased parent/seed, residual nonlinear
attribute recovery, and race-support limitations. No feature-versus-bank pair
meets all three descriptive utility/recoverability inequalities; very close
misses are disclosed. This is not a purpose-conditioned or coalition experiment,
and no PCRL, LoRA, adversarial encoder, task search or novel method was added.

- [Frozen protocol](../results/redesign_20260908_acs_protection_v1/PROTOCOL.md),
  [configuration](../results/redesign_20260908_acs_protection_v1/config.json),
  [tables](../results/redesign_20260908_acs_protection_v1/TABLE.md),
  [paired/weighted analysis](../results/redesign_20260908_acs_protection_v1/ANALYSIS.md),
  [support and every exposed-control failure](../results/redesign_20260908_acs_protection_v1/SUPPORT.md).
- [SEX tradeoff](../results/redesign_20260908_acs_protection_v1/tradeoff_SEX.png),
  [RAC1P tradeoff](../results/redesign_20260908_acs_protection_v1/tradeoff_RAC1P.png),
  [all candidates](../results/redesign_20260908_acs_protection_v1/PER_TARGET.csv),
  [all categories](../results/redesign_20260908_acs_protection_v1/PER_CLASS.csv),
  [fixed-margin decisions](../results/redesign_20260908_acs_protection_v1/criteria.json).
- [Runner and access boundaries](../experiments/run_acs_protection.py),
  [fixed-schema affine erasure](../experiments/acs_protection_maps.py),
  [stronger independent audits](../experiments/acs_protection_audits.py),
  [reporting without refits](../scripts/summarize_acs_protection.py).
  Original [data](../experiments/acs_transfer_data.py),
  [source models](../experiments/acs_transfer_models.py), and
  [heads/scoring](../experiments/acs_transfer_heads.py) are unchanged dependencies.
- [Validation](../results/redesign_20260908_acs_protection_v1/VALIDATION.md),
  [map replay](../results/redesign_20260908_acs_protection_v1/INDEPENDENT_VERIFICATION.json),
  [score replay](../results/redesign_20260908_acs_protection_v1/SCORE_REPLAY.json),
  [original task reproduction](../results/redesign_20260908_acs_protection_v1/PARENT_TASK_REPLAY.json),
  [runtime](../results/redesign_20260908_acs_protection_v1/runtime.json),
  [reproduction and omitted objects](../results/redesign_20260908_acs_protection_v1/REPRODUCTION.md).

The three-seed experiment took173.334s on the existing M4 Pro CPU. All15 maps
passed raw numerical covariance checks; covariance is not a classification
certificate, and seed2's absent eraser-fitting race category invalidates its
coverage-aware flag. No complete race-policy assessment is possible in any seed.
PCA remains the strongest residential-transfer parent and the neural bank a
strong task-serving alternative. The next methodological question is standard
utility-aware nonlinear protection feasibility, before prioritizing a PCRL-specific
mechanism. No such next experiment was started.

## Appendix — 2026-09-08: fixed nonlinear ACS bottleneck

The [fixed utility-aware nonlinear pilot](../results/redesign_20260908_acs_bottleneck_v1/RESEARCH_DECISION.md)
starts from `948169361c38fa5d37657fd45c5ab45c84f1fef5` and reuses exact PCA,
PCA+LEACE, rich neural/tree bank and control evidence. Matched16D arms share
60-epoch source/reconstruction warmup,20-epoch adversary warmup, exact model/Adam
clones and80 continuation epochs; only D applies the fixed nonlinear protection
penalty. Final release audits include fresh logistic/MLP/tree candidates and
saved-adversary catch-up in original coordinates. No purpose conditioning,
coalition, LoRA, new eraser, new data, task search, or PCRL mechanism was added.

D improves measured SEX attack loss over C by .010078nats on development,
with source utility preserved in every seed. The race difference is only.002580
with catch-up, compared with.012064 under independent-only audits. Both arms
retain half original PCA residential headroom in1/3 development seeds and0/3
validation seeds. No primary feature-bank numerical comparison passes all
margins. Race coverage remains inadequate for a complete all-attribute claim.
These are small fixed-design development findings, not novelty or impossibility.

- [Frozen protocol](../results/redesign_20260908_acs_bottleneck_v1/PROTOCOL.md),
  [configuration](../results/redesign_20260908_acs_bottleneck_v1/config.json),
  [tables](../results/redesign_20260908_acs_bottleneck_v1/TABLE.md),
  [paired/weighted analysis](../results/redesign_20260908_acs_bottleneck_v1/ANALYSIS.md),
  [support](../results/redesign_20260908_acs_bottleneck_v1/SUPPORT.md).
- [All target/candidate scores](../results/redesign_20260908_acs_bottleneck_v1/PER_TARGET.csv),
  [paired differences](../results/redesign_20260908_acs_bottleneck_v1/PAIRED.csv),
  [saved/fresh/catch-up audit](../results/redesign_20260908_acs_bottleneck_v1/CATCHUP.csv),
  [SEX tradeoff](../results/redesign_20260908_acs_bottleneck_v1/tradeoff_SEX.png),
  [race tradeoff](../results/redesign_20260908_acs_bottleneck_v1/tradeoff_RAC1P.png),
  [training curves](../results/redesign_20260908_acs_bottleneck_v1/training_curves.png).
- [Runner and reference/label boundaries](../experiments/run_acs_bottleneck.py),
  [training phases and gradients](../experiments/acs_bottleneck_training.py),
  [coordinate-faithful catch-up](../experiments/acs_bottleneck_catchup.py),
  [reporting without refits](../scripts/summarize_acs_bottleneck.py).
- [Validation](../results/redesign_20260908_acs_bottleneck_v1/VALIDATION.md),
  [state replay](../results/redesign_20260908_acs_bottleneck_v1/INDEPENDENT_VERIFICATION.json),
  [score replay](../results/redesign_20260908_acs_bottleneck_v1/SCORE_REPLAY.json),
  [runtime](../results/redesign_20260908_acs_bottleneck_v1/runtime.json),
  [reproduction and local-only artifacts](../results/redesign_20260908_acs_bottleneck_v1/REPRODUCTION.md).

The three-seed scientific run took88.731s on the existing M4 Pro CPU with one
numerical thread.23 focused tests passed;2,220 score sets and exact frozen
releases replayed successfully. All411 reused reference records are unchanged.
Compact source/evidence are published; raw records, fitted objects and arrays
stay local with hashes. The single proposed next check is a fixed16-coordinate
PCA control to distinguish compression from source-focused training; it was not
run. No basis for advancing PCRL-specific protection has been established.

## Appendix — 2026-09-08: fixed PCA16 control

The [PCA16 decision](../results/redesign_20260908_acs_pca16_v1/RESEARCH_DECISION.md)
adds exactly the first 16 original PCA32 coordinates, with no new PCA fit,
whitening, rotation, erasure or learned representation. Original component
ordering, output and fitting-row identities were verified. All other releases,
models, predictions and selections are reused unchanged.

Development residential log loss is .495062 ± .002095 (PCA16), .487411 ± .005595
(PCA32), .500295 ± .003986 (C16), .501592 ± .002660 (D16). PCA16 meets all original
PCA source-loss margins and retains half residential headroom in all three
development seeds, but only two validation seeds. The weighted advantage over
C/D is much smaller; recorded attributes remain recoverable and race coverage
is incomplete. Equal dimensions do not imply equal information or isolate the
cause of the learned mappers' loss. This establishes no PCRL novelty or privacy.

Inspect [the fixed protocol](../results/redesign_20260908_acs_pca16_v1/PROTOCOL.md),
[all task/seed results](../results/redesign_20260908_acs_pca16_v1/TABLE.md),
[paired and weighted analysis](../results/redesign_20260908_acs_pca16_v1/ANALYSIS.md),
[exact slicing and fitting boundaries](../experiments/run_acs_pca16.py),
[primary independent versus historical catch-up reporting](../scripts/summarize_acs_pca16.py),
[new-only score replay](../scripts/verify_acs_pca16.py),
[validation](../results/redesign_20260908_acs_pca16_v1/VALIDATION.md) and
[reproduction/local artifacts](../results/redesign_20260908_acs_pca16_v1/REPRODUCTION.md).
Ten focused tests passed; 120 new probability sets replayed bitwise. Three seeds
took 19.663 seconds on the existing M4 Pro, one numerical thread. No historical
model was retrained. The recommended next change is PCA16-exact initialization
of the otherwise unchanged matched C/D mapper; it has not been run.

## Appendix — 2026-09-08: PCA16-initialized matched C/D

Only the existing mapper's initialization changed. Signed positive/negative PCA32
units analytically undo saved fitting-only standardization and reproduce PCA16
within4.77e-7 coordinate error. Source heads/decoder, adversary initialization,
model capacity, objectives, schedules and example access remain identical to the
historical recipe. Frozen I/W snapshots diagnose initialization/common warmup;
C_init/D_init are parallel final continuations. All representation training
finished before fitting any reserved-task head, with no snapshot selection.

Mean unweighted residential loss I/W/C_init/D_init is
.495062/.498230/.499489/.501559. Final D practically ties historical D (.501592).
Common warmup's +.003168 unweighted change becomes −.002566 under PWGTP; source
utility improves and passes the original-PCA32+.01 reference throughout. Final
C/D retain half residential headroom in only1/3 development seeds. Independent
D−C race attack-loss gain .022542 shrinks to .010327 with catch-up and reverses
in seed0. Race support remains incomplete; no joint policy or PCRL advantage
is established. These are DEVELOPMENT EVALUATION, not confirmation.

- [Decision](../results/redesign_20260908_acs_pca16_init_v1/RESEARCH_DECISION.md),
  [stage/final tables](../results/redesign_20260908_acs_pca16_init_v1/TABLE.md),
  [per-seed/weighted analysis](../results/redesign_20260908_acs_pca16_init_v1/ANALYSIS.md),
  [stage plot](../results/redesign_20260908_acs_pca16_init_v1/residence_stages.png),
  [frozen protocol](../results/redesign_20260908_acs_pca16_init_v1/PROTOCOL.md).
- [Optional initialization/snapshot training](../experiments/acs_bottleneck_training.py),
  [extended runner and label/selection boundaries](../experiments/run_acs_bottleneck.py),
  [unchanged catch-up](../experiments/acs_bottleneck_catchup.py),
  [read-only reporting](../scripts/summarize_acs_pca16_init.py).
- [Validation](../results/redesign_20260908_acs_pca16_init_v1/VALIDATION.md),
  [new-only model/score replay](../results/redesign_20260908_acs_pca16_init_v1/SCORE_REPLAY.json),
  [reproduction/local-artifact limits](../results/redesign_20260908_acs_pca16_init_v1/REPRODUCTION.md),
  [runtime](../results/redesign_20260908_acs_pca16_init_v1/runtime.json).

Scientific process wall was91.787s for all three seeds on M4 Pro CPU, one
numerical thread. Historical models and audits were not refitted. Old execution
hashes remain intact; current source adds an option and is frozen separately.
Raw records, fitted models/Adam states and caches remain local with hashes.
The one proposed change is a PCA16-output preservation penalty during common
warmup. It has not been run; baseline failure is not a prohibition on developing
methods to address the demonstrated tradeoff.

## Appendix — 2026-09-08: PCA16 preservation and nested stronger audits

The [overnight decision](../results/redesign_20260908_acs_preservation_v1/OVERNIGHT_RESEARCH_DECISION.md) tests beta.1/1 × warmup-only/persistent × C/D, seeds0/1/2, from the same PCA16 initialization. Each beta/seed shares one60-epoch base and20-epoch adversary warmup followed by four exact full-state/Adam forks and80 continuation epochs. Raw immutable PCA16 targets and historical fitting scales define an additional preservation loss; reconstruction remains weight.1. No reserved task labels enter representation fitting. Historical beta0 and simple references are reused without representation or utility-head retraining.

All eight final means improve residence versus their corresponding beta0 arm under unweighted and PWGTP scoring. All source margins remain satisfied, but civilian-at-work/public-coverage losses rise. Beta1 persistent D reaches residence.492583/.472076 (unweighted/PWGTP), versus D_init.501559/.474214, while increasing both attribute gains. Its matched D−C gains are tiny, and affine teacher reconstruction error is.008816 versus prior1.000353. Warmup-only beta1 D instead improves residential means at essentially tied unweighted pooled recovery; weighted recovery and individual seeds qualify that favorable comparison. No unweighted development joint feature/bank numerical margin passes; race support remains incomplete.

All216 non-control fresh/catch-up trajectories keep their best validation checkpoint from epochs5–80;360 epochs change no release scores, rankings or feasibility flags. Exposed controls were extended too. Coordinate movement and affine teacher reconstruction reverse some release orderings, so movement cannot stand in for information loss. This is DEVELOPMENT EVALUATION and a standard training-objective study, not a privacy/novelty claim.

- [Full primary/weighted tables](../results/redesign_20260908_acs_preservation_v1/TABLE.md), [paired analysis](../results/redesign_20260908_acs_preservation_v1/ANALYSIS.md), [SEX plot](../results/redesign_20260908_acs_preservation_v1/tradeoff_SEX.png), [race plot](../results/redesign_20260908_acs_preservation_v1/tradeoff_RAC1P.png), [stage utility](../results/redesign_20260908_acs_preservation_v1/utility_stages.png), [coordinate/affine plot](../results/redesign_20260908_acs_preservation_v1/preservation_affine.png).
- [Nested budget results](../results/redesign_20260908_acs_preservation_v1/AUDIT_BUDGET.md), [support/controls](../results/redesign_20260908_acs_preservation_v1/SUPPORT.md), [executed matrix](../results/redesign_20260908_acs_preservation_v1/EXECUTED_MATRIX.json), [protocol](../results/redesign_20260908_acs_preservation_v1/PROTOCOL.md), [runtime](../results/redesign_20260908_acs_preservation_v1/runtime.json).
- [Study runner](../experiments/run_acs_preservation.py), [training support](../experiments/acs_bottleneck_training.py), [stronger audit runner](../experiments/run_acs_preservation_extended.py), [read-only reporting](../scripts/summarize_acs_preservation.py), [validation](../results/redesign_20260908_acs_preservation_v1/VALIDATION.md), [reproduction/local artifacts](../results/redesign_20260908_acs_preservation_v1/REPRODUCTION.md).

Scientific unit/audit wall was672.27s (11.20min), including a1.15s pre-fit manifest-compatibility failure. [The explicit amendment](../results/redesign_20260908_acs_preservation_v1/OPERATIONAL_RECOVERY.md) preserves original source hashes; no scientific result was invalidated or retrained. All core/extended work completed within the60-minute compute and4-hour total ceilings. Exactly one next experiment is proposed: distillation to an attribute-residualized fixed teacher under matched audits. It was not launched.


## Appendix — 2026-09-08: selective preservation and reconstruction

From reviewed commit `e1691955c330d4782ccd600d171ae9571f23fdb5`, the separately frozen R/E/S teacher × reconstruction.1/0 × matched C/D study is complete across three seeds. Six raw-teacher/beta1-persistent finals and compatible historical evidence were reused;30 new finals and six static E/S releases were fitted. The paired-label shuffle was frozen before either LEACE fit. E/S retained matching ranks7,7,8 and normalized distortions.5625,.5625,.5; no favorable shuffle was selected.

All12 learned conditions preserve the original source-task allowances. E has better mean residence than S in allfour matched comparisons under both weights, but more unweighted pooled SEX recovery in allfour. In E-C, removing reconstruction lowers pooled SEX/race gains by.007613/.006864 unweighted in every seed; residence worsens.002427. D adds modest recovery benefit at rho.1, but its pooled SEX effect at rho0 is essentially absent and changes sign under PWGTP. Learning greatly improves direct E's source utility, while restoring attribute recovery and some linearly recoverable erased structure. Race support remains unassessable; no full protection or coordination claim follows.

Scientific full-process wall was678.83s (11.31minutes), one M4 Pro numerical thread. All204 new fresh/catch-up trajectories completed360 epochs with nested120 checkpoints; selected predictions/rankings did not change between budgets. Independent replay passed1,470 candidate records,5,880 score dictionaries and2,940 prediction sets; separate training replay passed420 release arrays and105 gradient diagnostics exactly. No operational failure, scientific invalidation, historical model regeneration or next experiment occurred.

Start with [decision](../results/redesign_20260908_acs_selective_preservation_v1/RESEARCH_DECISION.md), [protocol](../results/redesign_20260908_acs_selective_preservation_v1/PROTOCOL.md), [executed matrix](../results/redesign_20260908_acs_selective_preservation_v1/EXECUTED_MATRIX.json), [table](../results/redesign_20260908_acs_selective_preservation_v1/TABLE.md), [analysis](../results/redesign_20260908_acs_selective_preservation_v1/ANALYSIS.md), [teachers](../results/redesign_20260908_acs_selective_preservation_v1/TEACHERS.md), [mechanism](../results/redesign_20260908_acs_selective_preservation_v1/MECHANISM.md), [audit budgets](../results/redesign_20260908_acs_selective_preservation_v1/AUDIT_BUDGET.md), [validation](../results/redesign_20260908_acs_selective_preservation_v1/VALIDATION.md), and [reproduction](../results/redesign_20260908_acs_selective_preservation_v1/REPRODUCTION.md). Earlier decisions and proposed-next-step statements above retain their historical scope.

## Appendix — 2026-09-08 restricted inputs, source controls, and exact coordination

Completed from reviewed commit `6a2d1d06e115004704361fd6d10d271fb9e9a853`: 24 matched E/S × full/restricted × C/D final models and six source-only probability banks. Exact frozen teachers, permutations, source states, PCA preprocessing, cohort and historical evidence were reused without refitting. The separately named 48→64→16 interface preserves historical runner contracts. All finite units and nested 120/360 audits completed in 9.49 minutes of scientific work. Original and unrelated files are preserved.

Restricted E-C residence is .505432 unweighted / .482860 PWGTP, versus its source bank .521901 / .497591. Banks nevertheless have better mean source prediction and much smaller attribute gains. Every restricted model fails the original joint source margins in every seed under both weights. E-C pooled race gain increases under restriction (.074529 to .077666; weighted .067691 to .069562), despite lower independent recovery. Real-label E has better mean residence than S, but worse restricted pooled race recovery. D supplies no robust joint improvement. All 456 composed witnesses reproduce teacher-only postprocessing exactly; recovery relative to small probes does not establish information creation. Full race assessment remains unassessable because code 4 lacks fitting/validation support.

The standalone rational calculation verifies a single-pair coalition improvement from 5/8 to 1/2 at equal 3/4 authorized accuracy. Four fresh pairs reverse the ranking: coordinated 107/128 versus independent 377/512, with authorized accuracy 27/32 in both. Cached identical pairs add no information in this model. Exact prediction controls, full-history enumeration and the utility-implied error bound are elementary illustrative evidence, not a novelty or deployment privacy claim.

Start with the [decision](../results/redesign_20260908_acs_restricted_inputs_v1/RESEARCH_DECISION.md), [analysis](../results/redesign_20260908_acs_restricted_inputs_v1/ANALYSIS.md), [protocol](../results/redesign_20260908_acs_restricted_inputs_v1/PROTOCOL.md), [matrix](../results/redesign_20260908_acs_restricted_inputs_v1/EXECUTED_MATRIX.json), [input boundary](../results/redesign_20260908_acs_restricted_inputs_v1/INPUT_BOUNDARY.md), [source controls](../results/redesign_20260908_acs_restricted_inputs_v1/SOURCE_BUDGET_CONTROL.md), [composed attacks](../results/redesign_20260908_acs_restricted_inputs_v1/COMPOSED_ATTACKS.md), [mechanism](../results/redesign_20260908_acs_restricted_inputs_v1/MECHANISM.md), [audit budget](../results/redesign_20260908_acs_restricted_inputs_v1/AUDIT_BUDGET.md), [exact calculation](../results/redesign_20260908_acs_restricted_inputs_v1/PURPOSE_COORDINATION_EXACT.md), [validation](../results/redesign_20260908_acs_restricted_inputs_v1/VALIDATION.md), and [reproduction](../results/redesign_20260908_acs_restricted_inputs_v1/REPRODUCTION.md). The [one next design](../results/redesign_20260908_acs_restricted_inputs_v1/PURPOSE_COORDINATION_DESIGN.md) compares separate and coalition-trained purpose releases against coordinated prediction-only and direct-teacher controls, with reserved-task and repeated-access requirements. It is proposed, not executed. Earlier conclusions and proposed-next-step statements remain historical evidence.


## Appendix — 2026-09-08 learned two-purpose coalition study

From reviewed commit `dcab4e16f5b9a094c67a5505a22dc2bbd6d6d286`, all18 paired systems (36 purpose-branch mappers) and their complete frozen-release audits are finished. The new protocol explicitly amends the historical proposal with the Iplus sensitive-only local control, every singleton candidate inherited into coalition pools, public source-prediction compositions, and separate residence/commute decisions. Both purposes use the original PCA32 input; F releases16 coordinates each and P releases2+1 source probabilities. The fixed E control, original PCA preprocessing, cohort and compatible historical predictions are reused without refitting.

Under the expanded catch-up360 pool, F-J lowers mean coalition SEX gain versus F-I by.013469 unweighted/.011413 PWGTP and versus F-Iplus by.005827/.002704. The latter unweighted difference favors J in allthree seeds; its weighted sign reverses in one. Race is less stable: the pooled J−Iplus gain difference is−.003364/−.004679, but the standard independent difference is+.010059/+.008066 and is unfavorable in every seed. Prediction J's SEX differences are small and its race gain exceeds Iplus in every seed. Individual opposing-task recovery remains substantial; a coalition SEX benefit does not satisfy the full policy.

F-J residence improves over P-J by.013650/.011420 mean nats, with favorable signs in allthree seeds. The.01 reference is met in only2/3 unweighted and1/3 weighted seeds, and F-J adds.012700/.011563 mean coalition race gain. Its commute advantage over P-J is only.001000/.000821, below.01 in every seed. Direct E has better mean residence and commute, but fails the joint source allowance; F-J passes that original allowance in2/3 seeds under both weights and retains half original-PCA32 residential headroom in none. No full feature/policy or race-support pass follows.

All new fresh/catch-up paths reach360 with nested120 checkpoints. Singleton projections, native-head compositions and deterministic duplicate histories are retained as separate evidence. Longer fitting changes64 selected role/scope endpoints; better validation selection can worsen development loss. Original code4 race support remains unassessable, and inherited observer exposure is separately disclosed. These are DEVELOPMENT EVALUATION households and descriptive three-fit variation.

Conservative scientific full-process wall was1883.536s (31.39minutes), including the preserved operational loader failure and one bounded recovery. Completed models and auditors were loaded without retraining; original scientific source identities and historical conclusions remain intact. See [decision](../results/redesign_20260908_acs_coalition_v1/RESEARCH_DECISION.md), [protocol/amendment](../results/redesign_20260908_acs_coalition_v1/PROTOCOL.md), [matrix](../results/redesign_20260908_acs_coalition_v1/EXECUTED_MATRIX.json), [counts](../results/redesign_20260908_acs_coalition_v1/FITTING_COUNTS.json), [analysis](../results/redesign_20260908_acs_coalition_v1/ANALYSIS.md), [protection control](../results/redesign_20260908_acs_coalition_v1/PROTECTION_CONTROL.md), [coalition audit](../results/redesign_20260908_acs_coalition_v1/COALITION_AUDIT.md), [access policy](../results/redesign_20260908_acs_coalition_v1/ACCESS_SCOPE.md), [validation](../results/redesign_20260908_acs_coalition_v1/VALIDATION.md), [runtime](../results/redesign_20260908_acs_coalition_v1/runtime.json), and [reproduction/local objects](../results/redesign_20260908_acs_coalition_v1/REPRODUCTION.md). The next method experiment is recommended only; it has not been launched. Earlier proposed-next-step statements remain historical.


## 2026-09-09 UTC — coalition strength and comparable utility

The fixed strength comparison from reviewed commit `9461fe29c4f7db337ea5dc20e295dc8d02c3d9e2` is complete: 54 unique paired systems, comprising 36 new continuations and 18 reused ordinary-I/beta.1 systems. Both F/P interfaces and Iplus/J families have the same five strengths, including one shared zero anchor per interface/seed. Every new continuation loads its exact historical model/observer/Adam/schedule fork; no warmup, teacher, historical representation, prior or exposed control is refitted. All 54 functions were frozen before any new reserved-task heads or audits.

F-J(.1) retains lower mean expanded-catch-up SEX recovery than every local strength under both weightings, but the grid does not establish a source-feasible, closely matched feature advantage at the primary .001-nat per-task tolerance. No fixed system passes all three original source allowances in all three seeds. There is no qualifying close F source-only or source-plus-residence SEX comparison under primary unweighted scoring; the sole directional source-plus-residence improvement is J(.025) versus the ordinary-I anchor in one seed. Adding commute removes that primary directional qualification. These are inadequate utility overlap and source-feasibility findings, not proof that coordination cannot help.

A strong counterexample is F-J(.1) versus F-Iplus(.2): mean residence differs by only +.000177 unweighted/−.000107 PWGTP, yet per-task/per-seed checks exclude the pair. Pooled coalition SEX favors J by .008161/.004576, while independent-only SEX slightly favors Iplus; pooled B/race gain is .020291/.019164 higher under J. F-J(.1) retains residence capability beyond P-J(.1), by .013650/.011420, with the .01 reference attained in only 2/3 unweighted and 1/3 weighted seeds. It has no .01 commute advantage. All individual targets, weaker local comparators, signed negative audit gains and support failures remain visible. RAC1P code4 still lacks independent fitting/validation support; there is no full race or all-target certificate.

Scientific process wall was 3,622.195 seconds (60.37 minutes), with no scientific failure, recovery, skipped system or added sweep. New fitting includes 1,188 fresh MLP360 trajectories, 324 own-observer catch-ups, 1,782 static auditors and 360 utility candidates. Full independent replay verifies 20,664 prediction arrays and 41,328 score dictionaries with maximum discrepancy 6.66e-16, plus exact model/access/selection and nested-checkpoint checks. All results remain DEVELOPMENT EVALUATION on the repeatedly studied cohort; three-seed SDs are descriptive. The proposed next experiment targets source-utility feasibility with guarded protection updates and matched local/coalition controls. It is recommended only; no refreshed-version or other follow-up experiment was launched.

See [decision](../results/redesign_20260909_acs_coalition_strength_v1/RESEARCH_DECISION.md), [full table](../results/redesign_20260909_acs_coalition_strength_v1/TABLE.md), [all fixed utility matches](../results/redesign_20260909_acs_coalition_strength_v1/MATCHING_ANALYSIS.md), [protocol](../results/redesign_20260909_acs_coalition_strength_v1/PROTOCOL.md), [audit findings](../results/redesign_20260909_acs_coalition_strength_v1/AUDIT_FINDINGS.md), [matrix/counts](../results/redesign_20260909_acs_coalition_strength_v1/FITTING_COUNTS.json), [validation](../results/redesign_20260909_acs_coalition_strength_v1/VALIDATION.md), [runtime](../results/redesign_20260909_acs_coalition_strength_v1/runtime.json), [reproduction](../results/redesign_20260909_acs_coalition_strength_v1/REPRODUCTION.md), and [one next design](../results/redesign_20260909_acs_coalition_strength_v1/NEXT_DESIGN.md). Earlier protocols and conclusions remain historical evidence.


## 2026-09-09 UTC — source feasibility and guarded protection

The [source-guard study](../results/redesign_20260909_acs_source_guard_v1/RESEARCH_DECISION.md) is complete from reviewed commit `02a069213e095ed9e8a890c4cbd92e9ef84edb41`: 21 unique new forward continuations (42 branch mappers), 24 interface/audit systems and 24 hash-bound primary references. Three source-only forward trajectories are shared by F/P with separate observer histories. All 48 primary releases were frozen before new reserved-label fitting. Historical source/observer prefixes, maps, cohorts and reference fits remain unchanged.

The guard does not generally repair source feasibility or establish a useful coalition advantage over both fixed local controls. Source-only F-T passes all three original source allowances in every seed under unweighted validation/development scoring; weighted development fails seed-1 public coverage by .000958 nats beyond the allowance. P-T passes development in 2/3 seeds under each weighting. No guarded condition passes development in all three seeds under either weighting. Both primary F G-J/local comparisons have 0/3 close and 0/3 directional source-plus-residence eligibility at delta=.001.

G-J minus G-L025 pooled coalition SEX gain is +.001881 unweighted /−.001004 PWGTP; versus G-L20 it is −.005481/−.008562. The latter favorable pooled comparison reverses under independent audits (+.006229/+.002448), and pooled recipient-B race recovery is higher under G-J by .010705/.006949. Guarding F-J itself improves mean residence by .002618/.000532 and reduces coalition race gain by .006904/.007247, while increasing SEX gain by .006876/.003475. These are specific exchanges, not a general protection advance. Full race assessment remains unassessable because RAC1P code4 lacks independent fitting/validation support.

All 60,000 guard transitions satisfy the declared ideal projection and stored-float32 error checks. At fixed diagnostic points, accepted native task loss improves versus the full proposal in 204/270 checks, ties in 52 and worsens in 14; this local effect does not imply probe feasibility. The T all-step diagnostic omission was repaired by separately frozen exact reconstruction of its existing trajectories, with no replacement models and its cost included. Scientific processes took 2,680.219 seconds (44.67 minutes), with no scientific failure or operational recovery. New fitting includes 792 fresh MLP360 paths, 216 own-observer catch-ups, 1,188 static auditors and 240 utility candidates. Independent replay verifies 13,776 prediction arrays and 27,552 score dictionaries (maximum discrepancy 6.66e-16); 52 focused tests pass.

See [table and per-seed evidence](../results/redesign_20260909_acs_source_guard_v1/TABLE.md), [source feasibility](../results/redesign_20260909_acs_source_guard_v1/SOURCE_FEASIBILITY.md), [fixed utility matching](../results/redesign_20260909_acs_source_guard_v1/MATCHING_ANALYSIS.md), [guard diagnostics](../results/redesign_20260909_acs_source_guard_v1/GUARD_DIAGNOSTICS.md), [audits](../results/redesign_20260909_acs_source_guard_v1/AUDIT_FINDINGS.md), [protocol](../results/redesign_20260909_acs_source_guard_v1/PROTOCOL.md), [matrix/counts](../results/redesign_20260909_acs_source_guard_v1/FITTING_COUNTS.json), [validation](../results/redesign_20260909_acs_source_guard_v1/VALIDATION.md), [runtime](../results/redesign_20260909_acs_source_guard_v1/runtime.json), and [reproduction](../results/redesign_20260909_acs_source_guard_v1/REPRODUCTION.md). The [one next design](../results/redesign_20260909_acs_source_guard_v1/NEXT_DESIGN.md) holds source-prediction functions fixed and tests an auxiliary residential feature channel against local/coalition and simple controls; it is proposed, not executed. These remain DEVELOPMENT EVALUATION results. All earlier protocols, findings and proposed-next-step statements retain their historical scope.


## Appendix — 2026-09-10 UTC: fixed source predictions and an auxiliary A channel

The [completed study](../results/redesign_20260909_acs_fixed_predictions_v1/RESEARCH_DECISION.md) preserves all nine original source-predictor vectors exactly and passes all three legacy source-readout allowances in every seed/split/weight. All18 systems were evaluated after the global freeze:12 learned A-only continuations plus H/static-E controls. J improves residence over H by .021528/.019516 nats (unweighted/PWGTP), with mean additional coalition SEX gain .001109/.000740. It improves mean residence and coalition recovery over both local controls, but no primary .001 close match qualifies, weighted directional lower-SEX comparisons fail, and A race recovery worsens versus L20. The original residential half-headroom criterion still fails for J in every seed. This is conditional development evidence, not an overall coordination or privacy certificate.

See [full table](../results/redesign_20260909_acs_fixed_predictions_v1/TABLE.md), [anchor parity](../results/redesign_20260909_acs_fixed_predictions_v1/ANCHOR_PARITY.md), [matching](../results/redesign_20260909_acs_fixed_predictions_v1/MATCHING_ANALYSIS.md), [validation](../results/redesign_20260909_acs_fixed_predictions_v1/VALIDATION.md), [reproduction](../results/redesign_20260909_acs_fixed_predictions_v1/REPRODUCTION.md), and [one proposed independent evaluation](../results/redesign_20260909_acs_fixed_predictions_v1/NEXT_DESIGN.md). No follow-up was launched; historical protocols and conclusions above remain historical.
