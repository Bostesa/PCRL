# PCA16-initialized C/D: research decision — 2026-09-08 UTC

**PCA16 initialization did not establish a better final utility/protection
tradeoff under the unchanged schedule.** It starts at the intended PCA16
interface, preserves the three source-task references, and gives small source
improvements over the historical bottlenecks. Its final protected residential
loss practically ties historical D16, while measured SEX recovery is higher.
Unweighted residential loss first appears during common warmup, before protection,
and additional loss appears during continuation. Person weighting changes that
stage interpretation substantially.

This is DEVELOPMENT EVALUATION on the same cohort and household pools. Three
seeds provide descriptive variation, not population uncertainty or confirmation.
Only mapper initialization changed; no PCRL mechanism, new protection strength,
eraser, task, stopping rule or sweep was added. Snapshots were diagnosed after
all representation training finished; none was chosen instead of the fixed final.

## Stage utility and final recoverability

Natural log loss, mean ± sample SD. Lower task loss is better; higher attack loss
means less recoverability under the named audit. Attribute columns use the same
five independent validation-selected candidates throughout. I/W were not audited.
C_init and D_init are **parallel continuations of W**, not sequential stages.

| Snapshot | Residence | Income >$50k | Civilian employed and at work | Public coverage | SEX attack | RAC1P attack |
|---|---:|---:|---:|---:|---:|---:|
| I: zero update | .495062 ± .002095 | .294408 ± .007520 | .290035 ± .007223 | .505704 ± .008631 | not audited | not audited |
| W: common 60 epochs | .498230 ± .004660 | .286573 ± .009795 | .276222 ± .005653 | .493155 ± .012802 | not audited | not audited |
| C_init: final task-only | .499489 ± .001184 | .289622 ± .011301 | .277046 ± .007070 | .495113 ± .011102 | .655859 ± .007702 | 1.218044 ± .025271 |
| D_init: final protected | .501559 ± .002914 | .290818 ± .010429 | .276961 ± .008175 | .494527 ± .010908 | .663448 ± .006426 | 1.240585 ± .021967 |

PCA16 parity passed at the declared atol=rtol=1e-5. Maximum coordinate error was
**4.7684e-7**; fitting RMS was 2.2096e-8 / 2.3927e-8 / 2.2643e-8, and source-validation
RMS 2.2400e-8 / 2.3385e-8 / 2.2932e-8. All I heads were freshly fitted: selected
unweighted validation/development log-loss differences from PCA16 were below
3e-9, not assumed exactly zero. I retained true zero Adam steps. All original
non-mapper tensors and preprocessing buffers matched, unused readout connections
received gradients and subsequently changed, and W survived adversary warmup
unchanged. [Parity/state replay](SCORE_REPLAY.json) and per-seed training records
separate these implementation checks from utility measurements.

| Residential contrast, left minus right | Validation | Development | Development PWGTP |
|---|---:|---:|---:|
| W − I | +.001756 ± .012352 | +.003168 ± .002616 | −.002566 ± .004596 |
| C_init − W | +.004629 ± .002596 | +.001259 ± .005228 | +.000597 ± .004951 |
| D_init − C_init | +.001282 ± .000800 | +.002070 ± .002047 | +.001638 ± .002671 |
| C_init − historical C16 | −.001080 ± .003377 | −.000805 ± .002869 | −.002492 ± .007906 |
| D_init − historical D16 | −.002321 ± .003215 | −.000033 ± .000492 | −.001893 ± .003541 |

Development W−I is positive in all three seeds (.003739/.000313/.005451).
C_init−W varies (−.001815/+.007296/−.001703); D_init−C_init is positive in all
three (.000529/.001288/.004392). On validation, warmup instead improves seeds0/1
and loses .015468 in seed2; task-only continuation loses utility in every seed.
Thus common warmup accounts for much of the **mean unweighted** loss before C,
but the evidence does not isolate an objective, optimizer or geometric cause.
All task/family/seed contrasts are retained in [ANALYSIS.md](ANALYSIS.md) and
[PAIRED.csv](PAIRED.csv); [the plot](residence_stages.png) shows the heterogeneity.

Every I/W/C_init/D_init source task stays within original PCA32+.01 separately
for each seed on validation and development. The final C/D source gains over I
are substantial compared with their residential loss; these interfaces are not
universally worse. Commute remains weak: I .685527, W .684743, C_init .685948,
D_init .685954 versus prior .693188. It was not replaced or used to train/select
the mapper. All five tasks appear separately in [TABLE.md](TABLE.md).

## Weighted sensitivity and strongest counterevidence

PWGTP uses identical predictions selected by unweighted validation. It **reverses
the mean warmup result** and limits a claim that training broadly damages transfer:

| Snapshot | Development weighted residence |
|---|---:|
| I | .474544 ± .009318 |
| W | .471979 ± .010379 |
| C_init | .472576 ± .007503 |
| D_init | .474214 ± .005709 |

Weighted W−I improves in2/3 seeds; C_init and D_init improve over historical C/D
on average, albeit with variation exceeding these small means. Original PCA32
remains better at .464514±.011720. These are a sensitivity analysis on sampled
people, not official survey estimates or design-based confidence intervals.
This counterevidence prevents presenting the unweighted stage pattern as a
robust Census-population result or proof that warmup is intrinsically harmful.

## Protection, catch-up and unchanged margins

Compared with matched C_init, D_init raises independent attack loss by
**.007589±.002074 SEX** and **.022542±.003754 RAC1P** on development, while residence
loses .002070±.002047. These are utility/recoverability tradeoffs, not dominance.
Signed prior-relative attack gains remain positive: C_init .036828 SEX/.071999
race; D_init .029239/.049457. Original PCA32 gains are .037469/.089644. Negative
gains in individual candidate records remain signed; they are not negative
information or evidence of protection.

Against historical C16, C_init attack loss is lower by .002679 SEX/.010597 race
on average: the small residential improvement comes with **more recovery**.
Against historical D16, D_init residence is essentially tied (−.000033 nats),
SEX loss is .003643 lower and race loss differs by only −.000119. Initialization
alone therefore supplies no established final protection advantage.

The saved training adversaries remain weaker than final attacks. All12 catch-up
trajectories improve development loss over their own saved weights. Catch-up is
validation-selected for5/12 final arm/attribute/seed comparisons; the other seven
use fresh MLPs. Catch-up inherited260 representation-fitting passes before120
attacker-fitting epochs with reset Adam and exact direct-input coordinates.
That exposure is distinct from the fresh independent candidates.

Inclusive development SEX/race losses are C_init **.655859/1.221850**, D_init
**.667260/1.232176**. D−C's apparent race benefit shrinks from .022542 to
**.010327±.018311**, and reverses in seed0 to **−.010654**. Inclusive selection
need not improve every development score: its winners use validation only.
The saved, fresh and caught-up candidates, all families/restarts and exposure
counts remain separate in [ANALYSIS.md](ANALYSIS.md), [FITTING.csv](FITTING.csv)
and [CURVES.csv](CURVES.csv). Better attacks change the assessment, not the release.

The original PCA32 remains the parent for every descriptive reference:

- Source preservation passes all four stages in all seeds/splits.
- Half residential headroom: I validation2/3, development3/3; W1/3 and2/3;
  C_init1/3 and1/3; D_init0/3 and1/3. Final initialization did not improve the
  development count over historical C/D.
- Under the matched independent audit, neither final arm halves positive PCA32
  SEX gain in any seed on either split. Inclusive D_init does so only in seed1
  development; that seed fails residential retention. Race halving remains
  unassessable because coverage is incomplete. The joint reference fails every
  final arm/seed/split; undefined race coverage never creates a pass.
- No feature-versus-bank pair satisfies all numerical comparison margins on
  development. On validation only D_init versus the unprotected neural bank in
  seed1 meets the numerical margins (both audit scopes); race coverage still
  prevents an all-attribute assessment. [All inequalities](bank_comparisons.json)
  are reported without choosing a release from them.

Original PCA32 remains the strongest residential interface (.487411 mean),
PCA16 the stronger simple16-coordinate transfer reference (.495062), and PCA32
+LEACE has better residence/SEX loss than either final arm but worse source
utility. Rich neural/tree banks remain competitive source interfaces with less
attribute recovery under the independent audits; final features transfer better on average but do not meet
the declared joint bank margins. No single interface dominates every target.

Race code4, Alaska Native alone, has no attacker-fitting/validation support;
it is absent from development seeds0/1 and has one example in seed2. All9
categories, per-pool supports and [original exposed-control failures](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md)
are retained. A complete all-race or all-attribute claim is unavailable. No
unsuccessful finite attack establishes universal privacy or classification bounds.

## Execution and decision

The fixed three-seed experiment took **91.787s** on the existing Apple M4 Pro CPU,
24GiB, one numerical thread. Seed0 took30.884s; the remaining estimate was61.768s
before expansion, actual60.903s. Training took41.190s, utility heads5.304s,
independent audits18.058s, catch-up5.807s; process wall includes reference checks,
scoring and serialization. No configuration change, budget reduction, scientific
rerun, historical refit or historical artifact regeneration occurred. Total implementation,
analysis and publication time is recorded separately in [runtime.json](runtime.json).

Focused checks and an artificial miniature pipeline passed. Independent replay
verified204 new candidate records,816 score dictionaries,408 bitwise saved-model
prediction sets and84 bitwise release applications, with maximum score discrepancy
4.44e-16. All615 reused reference records remained unchanged. See
[validation](VALIDATION.md), [protocol](PROTOCOL.md), [reproduction](REPRODUCTION.md)
and per-seed local-artifact hashes. Raw records, fitted objects and caches remain
local; compact review evidence is published with its actual source dependencies.

**Decision:** preserve this negative initialization result and the source/weighted
counterevidence. It neither justifies a PCRL advantage nor rules out a method
addressing the measured transfer/protection conflict. A fixed baseline's failure
is not an admission gate for method development, and this experiment does not
uniquely explain why its objectives produce these stage changes.

**One recommended next method change:** add a fixed PCA16-output preservation
penalty during common base warmup, leaving the rest of this matched comparison
unchanged. This targets the first observed unweighted transfer loss without
using residential labels in representation fitting; weighted/source tradeoffs
must remain explicit. That change has not been run.
