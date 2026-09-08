# Fixed utility-aware ACS bottleneck: research decision — 2026-09-08

**D gives a modest measured SEX-recoverability improvement over matched C while
preserving source utility, but it does not achieve the declared overall tradeoff.
Catch-up substantially reduces the apparent race benefit. Both learned arms lose
most of original PCA's residential headroom in most seeds. Keep the simple
references; do not advance a PCRL-specific claim.**

This is a fixed standard adversarial baseline on **DEVELOPMENT EVALUATION**
households. It does not establish novelty, purpose conditioning, coalition
protection, universal privacy, or impossibility for other designs. Public ACS
records simulate a release, not confidential Census deployment. Three seeds
share a cohort; their sample SDs are descriptive, not population uncertainty.

## Matched C versus D

Unweighted development log loss, natural-log nats; mean ± sample SD over three
seeds. Lower task loss is better. Higher attribute attack loss indicates less
recoverability under the selected audit. Attacks are selected on validation,
including catch-up where applicable; no development score selects an attacker.

| Outcome | C: source + reconstruction | D: also protection | Paired D−C |
|---|---:|---:|---:|
| Income >50k | .291447 ± .010328 | .292097 ± .011459 | +.000649 ± .001421 |
| Civilian employed and at work | .279831 ± .007417 | .279840 ± .007691 | +.000009 ± .000297 |
| Public coverage | .495148 ± .012124 | .495805 ± .015940 | +.000657 ± .003869 |
| Same residence one year ago | .500295 ± .003986 | .501592 ± .002660 | +.001298 ± .002680 |
| Commute >20min, eligible rows | .684432 ± .005534 | .685018 ± .004689 | +.000586 ± .000996 |
| SEX attack | .658537 ± .006823 | .668615 ± .007089 | +.010078 ± .004332 |
| RAC1P attack | 1.230281 ± .022703 | 1.232861 ± .023236 | +.002580 ± .001821 |

Primary per-seed differences remain visible:

| Seed | Residence D−C | SEX attack D−C | RAC1P attack D−C | C source tasks preserved | D source tasks preserved |
|---|---:|---:|---:|---|---|
| 0 | +.002759 | +.005320 | +.000481 | all three | all three |
| 1 | −.001795 | +.013796 | +.003519 | all three | all three |
| 2 | +.002929 | +.011119 | +.003739 | all three | all three |

The source preservation reference is **original PCA +.01 nats**, separately for
all three tasks, never matched C as a substitute parent. C/D each pass all three
in all three seeds on both validation and development. D's differences from C
are small; the largest source loss increase is .004858(public coverage,seed0).
D reduces measured SEX recovery in all three seeds under both the primary and
matched-independent audits. This is a small tradeoff improvement, not a tie or
a demonstrated dominating release: residence worsens in two seeds, and no
all-policy or primary feature-versus-bank margin comparison succeeds.
Mean signed prior-relative gains are SEX .037469(original PCA), .034149(C),
.024071(D), and RAC1P .089644 / .059761 / .057181 respectively. These are
predictive loss gains, not quantities of information removed.

## Residential headroom and simple references

| Release | Dimension / float32 bytes | Residence loss | SEX attack loss | RAC1P attack loss |
|---|---:|---:|---:|---:|
| Original PCA | 32 / 128 | .487411 ± .005595 | .655218 ± .011326 | 1.200399 ± .022578 |
| PCA + existing LEACE | 32 / 128 | .496275 ± .004211 | .678120 ± .008021 | 1.235046 ± .027316 |
| Rich neural bank | 26 / 104 | .510529 ± .003504 | .664770 ± .008175 | 1.243173 ± .032479 |
| Erased neural bank | 26 / 104 | .515247 ± .005386 | .671515 ± .007842 | 1.268265 ± .035414 |
| Rich tree bank | 26 / 104 | .517356 ± .002162 | .676694 ± .002856 | 1.259321 ± .019889 |
| Erased tree bank | 26 / 104 | .532475 ± .005421 | .685465 ± .002333 | 1.290130 ± .014079 |
| C bottleneck | 16 / 64 | .500295 ± .003986 | .658537 ± .006823 | 1.230281 ± .022703 |
| D protected bottleneck | 16 / 64 | .501592 ± .002660 | .668615 ± .007089 | 1.232861 ± .023236 |
| Fitting prior | no record-specific features | .536538 ± .005614 | .692687 ± .000573 | 1.290043 ± .025603 |

Residential outcomes never entered mapper training, reconstruction, stopping,
architecture or weight selection. Some transfer survives, but both C/D retain
half of original PCA's advantage over the better unprotected rich bank in only
**1/3 development seeds, 0/3 validation seeds**. D retains **61.90% / 37.67% /
20.58%** on development, **39.76% / 34.88% / −18.70%** on validation. The negative
last value means worse residential validation loss than the better bank.
C already fails this reference without any protective gradient. D's mean
residence cost relative to C is only .001298; the larger loss occurs in the
shared compression/training design. This run cannot separate dimensionality
from source-focused training as its cause.
The margins are provisional descriptive references, not statistical or justified
privacy thresholds; small misses do not establish robust scientific distinctions.

No learned arm versus any unprotected/erased rich bank meets all the .01-nat
residential advantage and .005-nat attribute-gain excess inequalities under
the primary audits: **0/48 seed/split comparisons**. The neural bank remains a
strong simple task-serving alternative; original PCA remains strongest on
residence. PCA+LEACE has better mean residence and audit losses than D but loses
source utility. Thus none simply replaces all others. The learned interfaces
are smaller than the banks/PCA; only C/D have matched capacity and training.
The reused richer banks also retain their original source-family supervision;
the new mapper receives only the three specified source binaries plus allowed
PCA reconstruction and fitting attributes. No bank was weakened or retrained.

## Catch-up materially changes the interpretation

With only the matched five fresh independent candidates, development D−C
attribute attack-loss differences are **+.008554 ± .002957(SEX)** and
**+.012064 ± .009089(RAC1P)**. Adding saved-weight catch-up changes these to
+.010078 and +.002580. Race differences by seed shrink from
**+.021614 / +.003519 / +.011058** to **+.000481 / +.003519 / +.003739**.
Catch-up removes the single otherwise favorable numerical bank comparison on
validation(D seed0 versus unprotected tree bank), and removes the independent-only
race-halving inequality in development seed2. Both remain race-coverage limited
regardless. No primary race-halving inequality passes on either split.

All12 catch-up fits improve development loss over their own saved training
adversary, but only2/12 beat the validation-selected fresh independent candidate
on development. The saved training adversaries understate recoverability; simply
reporting their final losses would be misleading. Primary catch-up was selected
in4/12 arm/attribute cases. A validation winner can be worse on development than
another candidate; every score is retained, and we do not reselect on development.
[Catch-up results](CATCHUP.csv) and [curves](catchup_curves.png) distinguish these
conditions and show that further fitting may overfit after an early selected epoch.

The saved adversaries had20 warm epochs plus240 equivalent continuation passes
on representation-fitting examples before120 new attacker-fitting epochs, with
Adam reset. Fresh independent MLPs have only the latter120 epochs and fitting-only
standardization. Catch-up preserves direct16D coordinates using an identity map;
initial predictions match exactly. This is stronger inherited-exposure auditing,
not an equal-exposure initialization comparison. Improved attacks do not improve
protection: the release remains unchanged. Catch-up success alone cannot isolate
moving representations from added fitting, overfitting or input conditioning.

## Limits and counterevidence

The strongest evidence favoring this fixed baseline is source preservation and
consistent SEX-recovery reduction with small D−C utility costs. The strongest
counterevidence to overall feasibility is shared residential loss, negligible
remaining primary race advantage, and no primary bank-margin success. D never
halves original PCA's positive SEX gain on development; it does so in only
validation seed2. Original-PCA gains are not zero-information baselines.

RAC1P code4(Alaska Native alone) is absent from attacker fitting and validation
in every seed, and has one development example in seed2. All nine categories
remain in loss and per-class reporting. The unsupported exposed control fails;
minimum-leaf20 trees also miss a supported rare category, while other tested
controls recover supported categories. No complete race or all-attribute success
can be assessed. Full-schema validation AUROC remains undefined. No near-prior
loss, negative signed attack gain, or finite attack failure becomes a guarantee.
Person-weighted sensitivity uses identical selected predictions; it is not an
official survey estimate. Commute remains weak and was not replaced.

## Execution, verification and decision

Fixed60-epoch common source/reconstruction warmup;20-epoch common adversary warmup;
exact model and both Adam-state clones;80 fixed continuation epochs. C observes
with the same adversary schedule; only D applies the normalized negative-CE
protection gradient. No LEACE follows either mapper. Source/reconstruction and
protection gradients reach the mapper; protection leaves source heads/decoder
without gradients. Reconstruction can retain attribute information and is no
transfer guarantee. Every final release and saved selection remained unchanged.

The three scientific seed processes took **88.731s(1.48min)** on Apple M4 Pro,
24GiB, CPU, one numerical thread. Seed0 was30.652s; remaining estimate61.305s,
actual58.078s. Mapper/adversary representation training took42.198s; fresh final
audits17.799s, catch-up5.731s, utility heads2.708s. Reference verification and
serialization are included in process wall. No budget reduction, paid resource,
new model/data download, reference refit, invalid scientific run or extra sweep.
Implementation, tests, analysis and publication wall time is separately recorded
in [runtime](runtime.json), rather than being presented as experiment time.

**23 focused tests passed**. Independent replay checked42 bitwise learned release
applications,321 new artifact hashes, exact model/optimizer forks and counters,
555 candidate records,2,220 score dictionaries, and288 saved-model prediction
sets. Maximum score discrepancy6.66e−16. All411 reused reference records remain
identical. [Validation](VALIDATION.md), [protocol](PROTOCOL.md), [tables](TABLE.md),
[analysis](ANALYSIS.md), [support](SUPPORT.md), and [reproduction](REPRODUCTION.md)
contain complete per-seed/family/weighted evidence and local-artifact limits.

**Decision:** preliminary feasibility for a modest SEX tradeoff only; insufficient
basis for purpose-specific protection or a PCRL mechanism. Preserve the negative
residential/race findings and the stronger simple references. A negative result
for this fixed16D design and budget is not an impossibility result.

**Exactly one recommended next experiment:** evaluate a fixed first16-coordinate
PCA release from the existing map, with the unchanged five-task heads and final
audits, against the saved C16/D16/PCA32 references. This simple matched-dimension
control would help distinguish compression loss from source-focused nonlinear
training before adding protection complexity. It has not been run.
