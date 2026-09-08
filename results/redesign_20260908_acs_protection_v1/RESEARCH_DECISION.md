# ACS protection-feasibility decision — 2026-09-08

**Simple joint LEACE retained some residential-transfer value, especially for
PCA, but did not meet the declared utility/recoverability tradeoff. Every erased
parent failed to preserve all three authorized source tasks within .01 nats.
No feature arm met the descriptive utility-plus-attribute margins against a rich
bank in any seed, on validation or development evaluation. PCA remains the
strongest residential-transfer parent; this is not evidence of PCRL efficacy.**

These are **DEVELOPMENT EVALUATION** results on the original ACS test households.
Their previous outcomes informed the research direction. They are not a new
confirmation, and the three seeds reuse a cohort rather than independent
population samples. All maps and budgets were frozen before fitting this run;
within-run head/attacker selection used validation only. No encoder was trained,
eraser refreshed after auditing, release selected, target changed, or grid expanded.

## Main result

Mean ± sample SD over three seeds; natural log loss in nats. Task loss should
be lower. Attribute attack loss should be higher for less measured recoverability.
Each attacker is chosen by validation log loss from logistic, two 120-epoch MLP
initializations, and two fixed 150-iteration boosted-tree configurations.

| Parent / joint LEACE | Residence loss | SEX attack loss | RAC1P attack loss |
|---|---:|---:|---:|
| Three probabilities | .522428 ± .002233 | .689054 ± .002075 | 1.252065 ± .028860 |
| Three probabilities + LEACE | .536387 ± .005802 | .692777 ± .000772 | 1.289187 ± .025087 |
| Rich neural bank | .510529 ± .003504 | .664770 ± .008175 | 1.243173 ± .032479 |
| Rich neural bank + LEACE | .515247 ± .005386 | .671515 ± .007842 | 1.268265 ± .035414 |
| Rich tree bank | .517356 ± .002162 | .676694 ± .002856 | 1.259321 ± .019889 |
| Rich tree bank + LEACE | .532475 ± .005421 | .685465 ± .002333 | 1.290130 ± .014079 |
| Compressed learned features | .498699 ± .007746 | .654781 ± .005798 | 1.219621 ± .035568 |
| Compressed features + LEACE | .509683 ± .004454 | .675361 ± .004584 | 1.269058 ± .032043 |
| PCA | .487411 ± .005595 | .655218 ± .011326 | 1.200399 ± .022578 |
| PCA + LEACE | .496275 ± .004211 | .678120 ± .008021 | 1.235046 ± .027316 |
| Full allowed covariates | .490646 ± .006126 | .658396 ± .011732 | 1.200819 ± .026578 |
| Fitting prior | .536538 ± .005614 | .692687 ± .000573 | 1.290043 ± .025603 |

All nine race categories remain in overall loss. Category support is inadequate
for a complete race assessment; this table does not silently turn that limitation
into protection. Erased bank coordinates are real features and were probed with
fresh heads, not treated as probabilities. All original parent reserved-task
scores reproduced exactly under the same head procedure.

## What survived, and what failed

PCA's erased residence losses are **.492908 / .494921 / .500997** for seeds0/1/2.
Its paired losses relative to the unprotected parent rise by
**.000505 / .013558 / .012530**. It retains **96.96% / 59.13% / 35.94%** of its
positive advantage over the better unprotected rich bank. Thus the half-headroom
reference holds in **2/3 development seeds, 1/3 validation seeds**. The compressed
learned release retains **7.43% / 49.72% / −40.60%**, meeting the reference in
**0/3** on either split. Its seed1 miss is only about .00005 nats beyond the
half-headroom boundary; these provisional margins should not turn such a small
difference into a robust scientific distinction.

The more consequential failure is authorized source utility. For erased PCA,
paired development losses for income / civilian at work / public coverage are:

| Seed | Income increase | Civilian-at-work increase | Public-coverage increase |
|---|---:|---:|---:|
| 0 | .016304 | .046966 | .018250 |
| 1 | .016552 | .026848 | .020676 |
| 2 | .009552 | .027018 | .011143 |

The .01-nat reference fails for at least one source task for every parent and
seed, on both validation and development evaluation. This does **not** mean
every individual source margin fails: PCA seed2 income, for example, passes.
The three-coordinate bank is erased to a constant in every seed, with severe
source-utility loss. Its near-prior audits are therefore not a useful solution
to the authorized task policy.

Erasure reduced some recoverability. For PCA, mean SEX prior-relative attack
gain falls from .037469 to .014566 nats; it is halved in all three development
seeds, but only two validation seeds. Mean race gain falls from .089644 to
.054996; the numerical halving inequality fails in all three development seeds.
Race assessments remain coverage-limited regardless of that inequality.
Compressed features numerically halve SEX gain in all three development seeds
and substantially reduce race gain, while sacrificing transfer/source utility.

For every feature/PCA arm versus every unprotected/erased rich-bank arm,
we checked the predeclared .01-nat residential advantage and at most .005-nat
extra prior-relative gain for **each** attribute. **None of 96 seed/split/arm
comparisons meets all three numerical inequalities**, even before the race
coverage restriction. One erased-feature versus erased-neural-bank comparison
(seed1 development) meets both attribute inequalities and misses the utility
margin by only .00004064 nats; the result is not a dominance or impossibility
claim. The PCA-erased mean residential advantage over the
unprotected neural bank is about .01425 nats, but its mean race attack gain is
about .00813 nats higher; means do not conceal the individual failed comparisons.

Commute remains weak: PCA loss .687570 becomes .691074, compressed features
.680676 becomes .683998, versus prior .693188. All per-family/seed metrics and
weighted sensitivity are retained; no new commute requirement or replacement
task was introduced.

## Covariance, nonlinear recovery, and coverage

All 15 fitted maps meet the fixed numerical covariance residual tolerance in
float64 and in the actual float32 release; maximum residuals are 1.77e−15 and
5.83e−10 respectively. This is an empirical fitting-set
statement. Nonlinear attackers still recover attributes from the erased richer
releases. No zero-covariance-to-classification certificate is used. Numerical
rank stabilization removes float32 simplex roundoff before whitening, under
one fixed recipe; it is not a tuned erasure strength.

Recorded RAC1P code4, Alaska Native alone, has zero attacker-fitting and
attacker-validation examples in every seed, and only one development example
in seed2. It is also absent from seed2 eraser fitting. Consequently that seed's
coverage-aware covariance flags correctly remain false despite small residuals.
Full-nine-class validation AUROC is undefined, so no full-schema AUROC-selected
race attacker exists; separately named observed-class/per-class diagnostics are
reported. No arm can be labeled as satisfying an all-attribute policy.

The exposed-label controls show that the stronger MLPs and smaller-leaf trees
can recover the supported categories. The unsupported seed2 race observation
is still missed. The minimum-leaf20 tree has zero recall for supported RAC1P
code5 on validation and development evaluation in every seed; the leaf5 control
corrects that failure. These per-candidate failures are retained rather than hidden behind the selected
control. Lack of support is not evidence of protection. These attacks say nothing
about confidentiality against public-record linkage, additional eligibility
lists, arbitrary future tasks, or recipient coalitions.

## Cost, validation, and research decision

Total experiment process wall: **173.334 seconds (2.89 minutes)** on Apple M4 Pro,
24GiB, CPU, one numerical thread. Seed0 took58.022s; the unchanged remaining
estimate was116.045s and actual remaining time115.312s. Source models were reused;
eraser fitting/freezing took.372s, utility fitting15.961s, and audits plus controls
125.966s in aggregate internal phase counters. No model/data download, paid
resource, budget reduction, or unrelated-job interruption occurred.

Thirty-two distinct focused tests passed across the pipeline/reporting suites.
Independent artifact replay verified105 exact map
applications and1,455 new binary hashes. Independent score replay verified711
candidate records and2,844 score sets with maximum numerical discrepancy
6.66e−16. The unchanged parent comparison reproduced234 historical metric
dictionaries exactly. See [validation](VALIDATION.md), [runtime](runtime.json),
[protocol](PROTOCOL.md), [full tables](TABLE.md), [analysis](ANALYSIS.md),
[support](SUPPORT.md), and [reproduction/availability](REPRODUCTION.md).

**Decision:** keep PCA as the strongest simple residential-transfer baseline, with its
unprotected and fully erased variants defining an observed tradeoff. Keep the
rich neural bank as the stronger task-serving bank. This run does not justify
claiming a feasible protected release, replacing simple controls with PCRL, or
claiming novelty. The strongest counterevidence to abandoning erasure entirely
is PCA's surviving residential benefit and reduced SEX recovery; the strongest
counterevidence to calling it successful is the source-utility loss, residual
race recovery, missing category support, and weaker validation retention.

**One recommended next experiment:** compare a single predeclared utility-aware
nonlinear protection baseline on the fixed PCA interface against these frozen
PCA/LEACE and rich-bank controls under the unchanged five-task policy, with
explicit rare-category audit limits. Release-map fitting would use only the
original source families and audited attributes; reserved-task labels would
remain confined to heads after release freezing. The question is whether it can preserve the
three source tasks while retaining residential headroom and reducing the residual
attribute gain; a standard baseline must establish that feasibility before a
PCRL-specific mechanism or untouched-household confirmation is prioritized.
