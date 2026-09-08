# PCA16 control: research decision — 2026-09-08 UTC

**Yes, on the primary unweighted comparison: the fixed first16 PCA coordinates
retain more residential-transfer utility on average than C16/D16.** PCA32 still
does better. PCA16 preserves all three source tasks within the original-PCA32
.01-nat allowance in every seed, without fitting a new representation. This
supports keeping PCA16 as a stronger compact transfer baseline; coordinate count
alone is not a sufficient explanation for the learned bottlenecks' loss.

These are DEVELOPMENT EVALUATION results on the same households and three seeds
sharing a cohort, not untouched confirmation or a significance claim. No PCRL
method, representation training, eraser, component search or new task was run.

## Primary comparison

Natural log loss, mean ± sample SD. Lower task loss is better; higher attribute
attack loss indicates less recoverability under that finite audit. Every attribute
audit uses its validation-selected winner from the same five independent candidates.

| Release | Float32 coordinates / bytes | Residence | SEX attack | RAC1P attack |
|---|---:|---:|---:|---:|
| PCA32 | 32 / 128 | .487411 ± .005595 | .655218 ± .011326 | 1.200399 ± .022578 |
| PCA16 | 16 / 64 | .495062 ± .002095 | .659948 ± .004457 | 1.212077 ± .020218 |
| C16, source + reconstruction | 16 / 64 | .500295 ± .003986 | .658537 ± .006823 | 1.228641 ± .023749 |
| D16, also adversarial protection | 16 / 64 | .501592 ± .002660 | .667091 ± .008388 | 1.240704 ± .017866 |
| PCA32 + existing LEACE | 32 / 128 | .496275 ± .004211 | .678120 ± .008021 | 1.235046 ± .027316 |
| Rich neural bank | 26 / 104 | .510529 ± .003504 | .664770 ± .008175 | 1.243173 ± .032479 |
| Rich tree bank | 26 / 104 | .517356 ± .002162 | .676694 ± .002856 | 1.259321 ± .019889 |

PCA16 improves residence over C16 by .005232 ± .005723 nats and D16 by
.006530 ± .003566 on development. On validation, improvements are .007465 ±
.010905 and .009988 ± .014497. All signs and seed variation remain visible:

| Seed | Validation PCA16−C16 | Validation PCA16−D16 | Development PCA16−PCA32 | Development PCA16−C16 | Development PCA16−D16 |
|---|---:|---:|---:|---:|---:|
| 0 | +.000993 | +.000254 | +.003797 | +.000223 | −.002536 |
| 1 | −.003615 | −.003643 | +.011281 | −.011190 | −.009395 |
| 2 | −.019773 | −.026576 | +.007875 | −.004730 | −.007659 |

PCA16 wins over C16 in2/3 seeds on both splits; it wins over D16 in2/3 validation
and3/3 development seeds. Its loss relative to PCA32 is .007651 ± .003747 nats
on development. This specific truncation does discard useful residential signal.
All residence heads selected for PCA16 are logistic; its MLP and every other
candidate remain reported in [the candidate scores](PER_TARGET.csv).

## Source utility and residential headroom

PCA16 development source losses are income .294408 ± .007520, civilian employed
and at work .290035 ± .007223, and public coverage .505704 ± .008631. All three
are within the original PCA32+.01 reference separately in every seed on both
splits. C/D have better mean source losses, particularly civilian-at-work and
public coverage; PCA16 does not dominate the learned interfaces across tasks.
Commute remains weak: PCA16 .685527 ± .003931, C16 .684432 ± .005534, D16 .685018
± .004689, versus fitting prior .693188 ± .000380.

PCA16 retains at least half of original PCA32's positive residence advantage
over the better unprotected rich bank in3/3 development seeds, with retained
fractions **.771543 / .659917 / .597370**. Validation is2/3: **.381145 / .522390 /
1.087794**; seed0 fails. C/D previously retained half in1/3 development and0/3
validation seeds. The parent, banks, margins and denominators are unchanged.
These are descriptive references, not privacy budgets or an admission condition
for future method work. Full task/seed results, including erased banks, are in
[TABLE.md](TABLE.md) and [criteria.json](criteria.json). Reused helper fields
named `erased_log_loss` in that JSON denote the compared PCA16 arm, which has
**no erasure**.

## Recoverability and strongest counterevidence

PCA16's signed prior-relative development attack gains are **.032739 ± .003939
(SEX)** and **.077966 ± .006712 (RAC1P)**, compared with PCA32 .037469/.089644,
C16 .034149/.061402, and D16 .025596/.049338 under matched independent audits.
PCA16 has more measured race recoverability than either learned bottleneck and
more measured SEX recoverability than D16 on average. It is a useful compact
transfer control, not a protected release or a uniformly better tradeoff.

The **PWGTP sensitivity substantially weakens the transfer advantage**: using
the same unweighted-selected predictions, development PCA16−C16 residence loss
is only **−.000523 ± .009670**, and PCA16−D16 **−.001563 ± .006129**; PCA16−PCA32
is **+.010031 ± .002498**. The primary benchmark targets sampled cohort members
unweighted. These weighted results prevent claiming a robust survey-population
advantage. Ordinary seed SDs are not design-based uncertainty.

The prior C/D catch-up-inclusive audit is retained separately, without changing
its saved validation selections. Its development SEX/RAC1P losses are C16
.658537/1.230281 and D16 .668615/1.232861. The additional catch-up candidates inherited260
representation-fitting passes before120 attacker-fitting epochs. Catch-up
recovered more than the saved observers and reduced D's apparent race benefit
in the earlier study; it is a stronger diagnostic opportunity, not matched fresh
exposure, and validation selection need not improve every development score.
PCA16 has no saved adversary and received none.

Race code4 (Alaska Native alone) is absent from attacker fitting/validation for
every seed and from development seeds0/1; seed2 development has one such record.
Its protection is unassessable. All9 classes and original exposed-control
failures remain in the [support record](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md).
No absent class or undefined score counts as protected; no universal privacy
or complete race assessment follows from these attacks.

## What the diagnostic resolves, and its limit

The first16 original PCA coordinates retained more usable residential signal
under the tested heads than the learned16D mapper on the primary mean comparison.
Equal stored dimension therefore does not make dimensionality alone a sufficient
account of C16's loss. This comparison cannot uniquely separate objectives,
optimization, geometry and information content, and it does not prove every16D
representation must retain this signal. No method novelty has been established.

No original model or audit was retrained. Exact component/order/map identities,
all21 slices, fitting rows, standardizers, immutable arrays and saved-before-test
selections passed [focused and replay checks](VALIDATION.md). Ten focused tests
passed; all120 new saved-model prediction sets replayed bitwise and240 score
dictionaries agreed within2.22e−16. The complete three-seed experiment took
**19.663 seconds** on the existing M4 Pro CPU, one numerical thread; seed0 took
7.324 seconds and projected21.973 seconds before expansion. Implementation,
analysis time through the recorded publication-preparation cutoff is separate
in [runtime.json](runtime.json); final commit/push verification follows that cutoff.

**One recommended method change:** initialize the existing16D learned mapper
to reproduce frozen PCA16 exactly, then test the otherwise unchanged matched
C/D training schedules. This would test whether starting from the stronger
compact interface avoids the source-only transfer loss. It would not guarantee
that later training preserves it. Do not run that comparison in this diagnostic.
