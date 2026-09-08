# PCRL redesign review index

**Latest addition, 2026-09-08:** the [fixed nonlinear ACS bottleneck pilot](../results/redesign_20260908_acs_bottleneck_v1/RESEARCH_DECISION.md) is complete. Matched protection modestly reduces SEX recovery, but neither learned arm meets the residential/attribute tradeoff. Catch-up reduces the apparent race benefit. See the [dated appendix](#appendix--2026-09-08-fixed-nonlinear-acs-bottleneck). No PCRL advantage is established.

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
