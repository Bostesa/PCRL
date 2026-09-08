# PCRL next-stage research report — September 7, 2026

## Executive summary and decision

**Recommendation: redefine the research problem before developing PCRL as a
method.** The stronger audit leaves scalar prediction release empirically
sufficient for the two fixed synthetic tasks. A known-leaky full representation
fails the same audit. None of the three screened real-data applications currently
provides both a defensible reusable-release requirement and a meaningful reserved
task family. Following the authorized stop condition, no real-data representation
method comparison was launched. This is a completed negative application screen,
not a claim that reusable private representations are never useful.

The fixed-task benchmark is closed. No representation, generator, head or eraser
was retrained or redesigned, and no grid was expanded. No PCRL protection advantage
or research novelty has been established. Stronger implementation and auditing
are useful research infrastructure, not a method contribution.

## What was executed

Read the repository instructions, current status, and available protocols,
analyses, tables and relevant source for all four preceding redesign stages.
The working tree and historical checkpoints/results were preserved on branch
`ablations-facct-2026-07-24`, commit
`15dbcc3c3338f6707e7a0b3901d738d99651a77d`. No AGENTS.md exists in this checkout
or its ancestors. The preservation inventory covers2662 historical result files.
No PR, push, dependency installation, paid resource, model download, or
interruption of another process occurred.

New work comprises a frozen scalar-release audit with fresh MLP and boosted-tree
attackers; focused tests; a development-only real-data policy/output diagnostic;
and a primary-source prior-work/application review. Artifacts and commands are
linked below. The previous E method was standard nonlinear adversarial training
with projected dual weights; this stage does not rename it as a new mechanism.

## Stronger prediction-only audit

All three saved task-only encoders, native scalar heads and original observation
standardizers were reused unchanged. The exact previous validation predictions
replayed. Each forbidden relationship receives affine OLS, a fresh32×32 ReLU MLP
with two1800-update restarts, and a fixed200-iteration histogram gradient-boosted
tree regressor. Scalar input dimensions are1,1,2; no incompatible full-feature
adversary weights are loaded into them.

Each MLP restart has460800 row presentations: more than the successful continued
adversary's900 original+640 additional updates, or394240 presentations per
trajectory. This does not equalize attack strength: fresh models use2048 distinct
attacker-fitting examples, while inherited models also saw3072 other training
examples; input dimensions and output sharing differ. Trees see200 full fitting
passes per target. All actual schedules, checkpoints, selected steps and target
identities are saved.

Attackers fit only attacker-fitting data. Independent affine/MLP task probes fit
only authorized representation-training examples. Validation selects MLP
checkpoints/restarts and the strongest family per target. Test RNG seeds
1100004/1100104/1100204 were recorded before fitting; each fresh4096-row test was
generated only after saved selection. All family results are reported, including
families not selected. Reused validation makes this an exploratory follow-up.

Fresh-test R² below; individual MLP leakage is the worst of P1→V,S and P2→U,S.
Feasibility additionally checks **every target for all three families**, and
both separate task floors. Mean ± sample SD is descriptive across three seeds.

| Release | Seed | Task U, MLP probe | Task V, MLP probe | Worst individual MLP leakage | Combined S MLP leakage | Validation / test feasible |
|---|---|---:|---:|---:|---:|---|
| Prediction-only | 0 | .998005 | .996804 | .000228 | −.002245 | Yes / Yes |
| Prediction-only | 1 | .998146 | .997708 | .000075 | −.002852 | Yes / Yes |
| Prediction-only | 2 | .997953 | .996046 | −.000912 | −.005660 | Yes / Yes |
| Prediction-only | Mean ± SD | .998035 ± .000100 | .996853 ± .000832 | −.000203 ± .000619 | −.003586 ± .001822 | 3/3 / 3/3 |
| Full E .01, unchanged positive control | 2 only | .990444 | .992069 | .349013 | .352950 | No / No |

Native scalar task R² is .997853±.000101 for U and .996745±.000874 for V;
each also exceeds.99 in every seed. Scalar worst-individual affine/tree R² means
are −.000576±.000328 / −.032221±.007508; combined S means are
−.002246±.001531 / −.063554±.017054. Negative R² means worse predictive squared
error than the evaluation-mean baseline, not negative information. The trees
overfit noise on oracle/scalar outputs; their negative scores do not strengthen
a universal privacy claim.

Controls behave as expected within the declared budgets. Oracle MLP forbidden
scores range −.006001 to .000325. Every exposed-target relationship, including S,
has MLP R² .999811–.999948, tree R² .996258–.998124 and affine R²1. The full
reference's fresh trees recover P2→U at.195354 and combined S at.125327, so both
nonlinear families detect violations. Its compatible saved continued adversaries
also still recover leakage; all five targets are reported separately in
[the audit analysis](results/redesign_20260907_prediction_audit_v1/ANALYSIS.md).

Prediction-only therefore remains empirically sufficient for these fixed tasks
under the tested attackers and thresholds. Unsuccessful attacks are bounded
evidence, not guarantees. The coalition is intentionally permitted U and V;
only its S inference is a privacy constraint. The earlier final-eraser mismatch
cannot hide the current attacks, because every attack sees the exact frozen
final release and no eraser is refitted.

The scalar interface uses one float32 prediction per purpose:4 bytes,8 combined.
The full reference has8 coordinates per purpose,16 combined; its float64 cached
release uses64/128 bytes. Different release content and dimensions prevent a
capacity-controlled method claim. Scalar extraction for4096 examples took
.002569–.003114 seconds locally. Per-arm fitting/inference costs and paired
prediction-minus-oracle statistics are in the saved tables/JSON; no F/G method
differences were estimated in this audit.

## Application screen and concrete counterevidence

Exactly three candidates were investigated. Access and runtime were not the
main blockers. Detailed ownership/recipient hypotheses, required/permitted/
forbidden information, attacker knowledge, coalition policies, simple solutions,
falsification criteria and data provenance are in
[the application-selection document](docs/PCRL_APPLICATION_SELECTION.md).

| Candidate | Evidence and decisive blocker | Decision |
|---|---|---|
| UCI HAR wearable activity analysis | Real activity/participant labels exist, but the second repository task is a coarsening of the first. It authorizes active/sedentary while forbidding activity. The same six-label task family is served by a six-way score bank; independent future health outcomes are absent. | No representation pilot; diagnose the policy conflict. |
| Diabetes encounter research | Genuine encounter outcomes exist. Current finite tasks fit a compact prediction bank; prospective target timing, a meaningful reserved outcome family and an external-learning requirement are unspecified. Task labels already correlate with forbidden demographics. | No representation pilot. |
| ACS microdata research | Official sources support genuine custom research queries, making this the strongest reuse lead. The permitted query/task family and reason existing microdata/query tools or a sufficient bank cannot serve it are missing. Public row linkage also changes the threat model. | First application to reconsider after specification; not selected now. |

A scientific pilot need not have a signed deployment contract. It does need a
clear interface and meaningful tasks whose labels do not influence encoder
selection. The present rejection rests on these missing ingredients and policy
conflicts, not merely the lack of a named stakeholder. These datasets are not
proved unsuitable for every future question.

One small learned-output diagnostic makes the HAR conflict concrete. Using only
official training participants, a fixed StandardScaler+logistic model predicts
active/sedentary. Disjoint groups contain13 task-fit,4 attacker-fit and4
development subjects (4697/1361/1294 windows). The scaler uses only task-fit rows.
Development task accuracy is.999227. An attacker fitted to its actual hard output
predicts the six-way activity at.363215 accuracy versus a fitted-prior.198609,
with.681384 nats of log-loss improvement. A fixed20-bin probability attack gives
the same accuracy and.681398 nats improvement. All six activity classes have
support. This uses recorded activity labels with a transparent coarsening;
it does not invent a new operational endpoint.

The oracle-label check independently shows the same logical problem: knowing the
coarse activity bit narrows the fine activity possibilities. Diabetes development
labels also expose associations—for example diagnosis-group release improves
gender accuracy from.522842 to.543620. These descriptive results are neither
universal bounds for noisy outputs nor medical conclusions. The earlier oracle
checks are explicitly retrospective; the learned HAR reference protocol was
written before its execution. No real-data final test was opened.

Absolute leakage and additional leakage beyond authorized outputs must be
separate criteria. A possible empirical incremental score is the held-out attack
log-loss improvement from (representation, authorized outputs) relative to
authorized outputs alone, using the same fitting/selection budgets. It is not
conditional mutual information or a guarantee merely because of its name.
Absolute attack performance must still be reported. No such revised policy was
substituted for a failed absolute policy in this stage.

## Prior work and what remains unresolved

[The focused matrix](docs/PCRL_PRIOR_WORK.md) verifies the five requested starting
papers and directly relevant adversarial/transfer/application methods using
primary papers and available official repositories. LEACE's affine/convex-loss
scope does not revive the retired threshold-accuracy certificate. SPLINCE already
preserves task covariance under stated conditions and explains refitted-head
equivalences. Least-privilege and perfect-erasure limits use materially different
assumptions from the finite Gaussian pilot. Robust privatization already treats
multiple possibly unknown tasks in its common-release information-theoretic
model. Their conclusions cannot be imported without their assumptions.

LAFTR studies held-out task labels; FFVAE and compositional graph filters support
flexible sensitive-attribute policies. Wearable transformation work motivates
external model compatibility. Elazar–Goldberg already documents demographic
recovery by fresh attackers despite weak training adversaries. These precedents
make a claim based only on multiple purposes, LoRA, dual optimization or the
observed audit failure untenable. This was a focused check, not an exhaustive
proof of non-novelty.

The strongest evidence against stopping method development is that these works
do establish meaningful reusable-release research interfaces, and distinct
recipient coalitions need not be equivalent to a single common release. Also,
scalar success here benefits from independent task/protected latents. Those are
reasons to define a better application, not evidence that the current PCRL
mechanism addresses it. No measured real-data transfer, competitive-bank failure,
matched F/G advantage, or new coalition benefit exists in this stage.

## Research decision and the experiment that could change it

Do not continue optimizing this toy benchmark or develop a method paper around
the current evidence. Use its scalar outputs for the fixed tasks, retain the
frozen-final-release audit as required evaluation practice, and redefine the
application/policy before another PCRL training run. The observed security
failure is useful but currently too narrow and too close to known adversarial
auditing failures to justify a broader security-analysis paper claim.

**Single next experiment:** after specifying one bounded ACS research task/query
family and which attributes its recipients may infer, freeze genuine task
holdouts and compare an adequate prediction bank with frozen unprotected features
on authorized transfer and task-output leakage, with person/household-aware
example separation. Do this necessity test before PCRL optimization. A meaningful
transfer gain that a competitive bank/trusted interface cannot deliver, at a
coherent output-aware privacy policy, would most change the present decision.
If the bank suffices or authorized outputs already defeat the proposed policy,
stop or revise that application. The task family and interface requirement are
prerequisites still missing, not promises of an experiment already validated.

## Evidence, checks and runtime

The strengthened audit took **52.24 seconds process wall** (21.01 first seed,
31.23 remaining seeds),48.777 seconds inside seed computations. The measured
first-seed estimate was recorded before expansion; no budget reduction was
needed. The full application screen, including data loading and the learned
reference, took5.073 seconds internally and approximately6.074 seconds command
wall; the learned fit and attackers took.06777 seconds. These are local
single-thread CPU measurements on Apple M4 Pro,14 cores,24GiB RAM. Preliminary
exploratory checks, source review and reporting are excluded from experiment
runtime. No real-data method seed was admitted, so there is no invented pilot
runtime or F/G comparison.

Eight focused scalar-audit tests passed in2.39 seconds; seven application-screen
tests passed in1.05 seconds. An initial application test fixture failed because
its subprocess mock also affected platform metadata; the fixture was corrected,
with no experiment/source change or invalidated run. Independent review replayed
380 audit scores exactly and checked all100 MLP trajectories and70 selected
checkpoints. All2662 historical result files,12 executed dependencies and existing
source work remain unchanged except the authorized status append. No entire-suite
audit repetition was needed. All original results remain historical evidence.

* [Audit protocol](results/redesign_20260907_prediction_audit_v1/PROTOCOL.md),
  [table](results/redesign_20260907_prediction_audit_v1/TABLE.md),
  [all target/family scores](results/redesign_20260907_prediction_audit_v1/PER_TARGET.csv),
  [plot](results/redesign_20260907_prediction_audit_v1/audit_scores.png),
  [paired metrics](results/redesign_20260907_prediction_audit_v1/paired_comparisons.json).
* [Application-screen protocol](results/redesign_20260907_application_screen_v1/PROTOCOL.md),
  [raw development metrics](results/redesign_20260907_application_screen_v1/metrics.json),
  [selection rationale](docs/PCRL_APPLICATION_SELECTION.md),
  [prior-work matrix](docs/PCRL_PRIOR_WORK.md).
* Code: experiments/run_prediction_audit.py, prediction_release_attackers.py,
  screen_pcrl_applications.py; scripts/summarize_prediction_audit.py. Protocols
  and logs contain reproduction commands. Fresh directories contain source/
  configuration snapshots, split/checkpoint identities, selection records,
  raw scores, schedules, runtimes and preservation checks.

## Amendment — 2026-09-07: defined ACS transfer stage authorized

The earlier screen and results above are preserved. The scientific admission
rule is amended: a defined one-time release with equal downstream fitting
privileges for banks and features suffices to test transfer. Proving every
trusted service inadequate is not a prerequisite; services remain deployment
alternatives. The [ACS transfer protocol](results/redesign_20260907_acs_transfer_v1/PROTOCOL.md)
fixes two genuinely withheld task identities and strong prediction-bank controls.
This stage measures unprotected transfer and recorded-attribute recoverability;
it adds no PCRL optimization, privacy guarantee or novelty claim.

The ACS stage is now complete: [research decision and compact results](results/redesign_20260907_acs_transfer_v1/RESEARCH_DECISION.md).
Learned features improved same-residence log loss over the rich neural/tree banks
by .011767±.006259 / .018594±.008287 nats in paired three-seed results, while
PCA outperformed the encoder. Commute features and banks tied at weak predictive
performance. Sex/race recoverability remains measurable, with incomplete rare
race-category coverage. A small baseline-led protection-feasibility study is
justified; PCRL optimization and novelty are not established. The complete run
took158.614s process wall on local CPU; no frozen task or search budget changed.
