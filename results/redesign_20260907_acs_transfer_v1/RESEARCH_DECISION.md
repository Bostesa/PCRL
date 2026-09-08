# ACS transfer result and research decision — 2026-09-07

**Reusable information helped residential transfer beyond both competitive
prediction banks. Plain PCA was stronger than the learned encoder. Commute
transfer showed no meaningful feature advantage. This supports a small,
baseline-led protection-feasibility study, not a PCRL method claim.**

The original application screen remains historical evidence. Its admission rule
has been amended: a defined one-time interface is enough to investigate transfer;
proving every trusted service inadequate is not a scientific prerequisite.
Public ACS records simulate a release setting. No Census deployment,
public-record linkage protection, longitudinal prediction, erasure or adversarial
encoder training is claimed here.

The protocol was frozen before ACS model comparisons. All three seeds used the
same 30,000-person household-selected young-adult cohort, disjoint household
pools within each seed, ten allowed covariates, source-only encoder/bank
supervision, and 2,048 eligible downstream fitting labels per reserved task.
Each interface received logistic and MLP heads with identical fitting examples,
schedules and validation-selection opportunities. Test was opened after every
head/attacker selection was saved. No grid or task changed after outcomes.

## Task-by-task result

Unweighted final log loss, nats; lower is better. Mean ± sample SD over three
split/training seeds. The same people can appear in different roles across
seeds; SD is descriptive, not independent survey uncertainty.

| Release | Dimensions / bytes per person | Same residence one year ago | Commute >20 minutes |
|---|---:|---:|---:|
| A: three binary probabilities | 3 / 12 | .522428 ± .002233 | .682161 ± .003200 |
| B: rich neural bank | 26 / 104 | .510529 ± .003504 | .681481 ± .002771 |
| C: rich tree bank | 26 / 104 | .517356 ± .002162 | .681863 ± .004760 |
| D: learned features | 32 / 128 | .498762 ± .006353 | .681786 ± .003885 |
| D compressed without task labels | 26 / 104 | .498699 ± .007746 | .680676 ± .001939 |
| E: PCA | 32 / 128 | .487411 ± .005595 | .687570 ± .008575 |
| F: allowed covariates | 75–77 / 300–308 | .490646 ± .006126 | .687624 ± .009124 |
| Fitting-prior reference | constant distribution | .536538 ± .005614 | .693188 ± .000380 |

All stored interfaces use float32. Full input dimensions vary because categorical
levels are discovered from each seed's representation-fitting pool. Probability
simplex redundancy means equal stored dimensions do not imply equal intrinsic
capacity. The compression is a predeclared dimension control, not a new method
chosen after looking at transfer results.

The primary paired log-loss differences are D minus comparator:

| Task | Comparator | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| Residence | B neural bank | −.004596 | −.014570 | −.016133 | −.011767 ± .006259 |
| Residence | C tree bank | −.012076 | −.015786 | −.027920 | −.018594 ± .008287 |
| Commute | B neural bank | +.003569 | −.000571 | −.002082 | +.000305 ± .002926 |
| Commute | C tree bank | −.000888 | +.000857 | −.000199 | −.000077 ± .000879 |

Residence AUROC is .686661 ± .020837 for D, .658019 ± .008926 for B,
.640546 ± .011257 for C, .711771 ± .020772 for PCA and .714582 ± .009658
for full inputs. D improves log loss over both banks in all three seeds; its
mean advantage over B exceeds the predeclared .01-nat practical reference,
but seed 0 does not. This is preliminary evidence, not a significance claim.
The matched-family comparisons in [ANALYSIS.md](ANALYSIS.md) and
[summary.json](summary.json) check that the result is not reported solely by
switching head classes.

PCA beats D for residence in every seed (D minus PCA .011351 ± .007609 nats).
The full input reference also wins on average, though not every seed. Thus the
finding is retained information useful for a reserved task; **a benefit from
the learned feature method over a simple information-retaining map is absent**.
The 26-dimensional compression retains D's average residence result, so the
bank comparison is not explained merely by 32 versus 26 stored coordinates.

Commute AUROC is only .575420 ± .008918 for D, .573761 ± .006289 for B,
.576170 ± .015167 for C and .564627 ± .028973 for full inputs. Utility is
slightly above the prior, but weak under the fixed covariates/head budget.
Banks and D are practically tied. This task provides little evidence for a
richer release; weak full-input performance limits the test's informativeness,
not a proof that commuting is intrinsically unpredictable.

Residence denominators are 2,982 / 3,015 / 2,979, with positive prevalence
.778672 / .769818 / .769721. Valid-commuter denominators are 1,971 / 2,022 /
1,955; prevalence .513445 / .506924 / .520205. Missing commute values are masked,
never negatives. All interfaces know the same commute-eligible population.
Eligibility membership can itself reveal worker status and associated attributes.
The main attribute audits cover the full sampled cohort's numerical releases;
they do not measure extra disclosure from an eligibility list or public linkage.
Final balanced accuracies, every candidate family, supports and weighted
sensitivity are retained in [TABLE.md](TABLE.md), [PER_TARGET.csv](PER_TARGET.csv)
and [PER_CLASS.csv](PER_CLASS.csv). Person weighting changes magnitudes, and is
reported on identical predictions without changing selection; no official
population estimates or design-based uncertainty are asserted.

## Attribute recoverability and limitations

These releases were deliberately unprotected. The table uses the family chosen
by attacker-validation log loss, not the best final-test result.

| Interface | SEX AUROC | SEX log loss | Nine-category RAC1P log loss | RAC1P accuracy |
|---|---:|---:|---:|---:|
| B neural bank | .629099 ± .016754 | .665486 ± .008452 | 1.241842 ± .030393 | .567332 ± .011008 |
| C tree bank | .604928 ± .004403 | .678170 ± .001198 | 1.259984 ± .018882 | .566535 ± .007320 |
| D features | .631907 ± .010057 | .660765 ± .002065 | 1.228515 ± .032005 | .579006 ± .009465 |
| E PCA | .654075 ± .017012 | .654491 ± .010551 | 1.203275 ± .021351 | .583040 ± .009165 |
| F covariates | .649137 ± .017295 | .661307 ± .014482 | 1.202185 ± .028538 | .586052 ± .010599 |
| Fitting-prior | .500000 ± .000000 | .692687 ± .000573 | 1.290043 ± .025603 | .566884 ± .010702 |

Log-loss gains show attribute information even where majority-class accuracy
barely changes. SEX and RAC1P remain the recorded schemas (2 and 9 categories).
Rare race classes have absent or very small support in some fitting/validation/
test pools. Full-nine-class macro AUROC and balanced accuracy are undefined
where support is missing, and their three-seed aggregates are not rescued by
silently dropping that seed. Per-class values and observed-class summaries are
separately named. These limitations particularly affect rare-category conclusions.

Exposed-label controls reach SEX accuracy 1 in every seed. Selected RAC1P control
accuracy is 1 / 1 / .999664; the missed seed-2 observation belongs to an
unsupported fitting category. This diagnoses support rather than universal
attacker competence. The exposed tree auditor also has zero recall for RAC1P
code 5 in every seed, with only 2–5 fitting examples; this family is not competent
on every rare category within its fixed budget. All three attack families and their successes/failures
are retained. No privacy criterion, coalition claim, or comparison to an
adversarial encoder has been made.

## Cost, checks and next experiment

Complete experiment process wall time: **158.614 seconds (2.64 minutes)** on an
Apple M4 Pro, 24 GiB RAM, CPU, one numerical thread. Seed 0 took 53.266 seconds;
the estimate of 106.532 seconds remaining was recorded before expansion; the
remaining two took 105.348 seconds. No budget reduction, new dependency, download
of the dataset, paid resource or interruption of another job was needed.
Source encoder search took 9.585 seconds total, source trees 94.156 seconds,
downstream heads plus audits/controls 34.757 seconds. Source A/B/D share encoder
training. Schema preparation took 1.053 seconds and the artificial full-pipeline
check 1.249 seconds. Per-release extraction timings appear in the main table;
they exclude the common covariate preprocessing and downstream-head inference.

Thirty focused regressions passed in 2.26 seconds. Real-run frozen-state,
household, source-only task-access, equal fitting-exposure, selection, and score
replay records accompany the results. Historical results were preserved. Local
checkpoints/cached releases are hashed but omitted from GitHub; see
[reproduction and availability](REPRODUCTION.md).

**Recommendation:** proceed only to a bounded, baseline-led protection-feasibility
experiment; do not prioritize a PCRL optimizer. Fix an explicit recipient policy
and output-aware utility/leakage criteria, retain these task identities, and
compare separate simple erasure of PCA and the 26-dimensional learned release
against the competitive bank and unprotected references. Fit and audit each exact
final release, with nonlinearity and rare-class coverage reported. The question
that would most change the decision is whether any such protected reusable
release retains the observed residential-transfer advantage at comparable
measured attribute recoverability. Commute should remain the declared weak-task
check rather than being replaced to manufacture headroom.

The strongest evidence against prioritizing protection methods now is that PCA
already beats the trained encoder on the informative task, the second task has
little headroom, and rare-class audit power is limited. The result justifies
examining a concrete transfer/privacy tradeoff; it does not establish that PCRL
solves it, that this public-data setting needs confidential deployment, or that
reusable features are necessary for all future tasks.
