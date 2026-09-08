# Selective preservation: prespecified contrasts

DEVELOPMENT EVALUATION on the original households and cohort. Unweighted loss is primary; PWGTP uses the SAME unweighted-validation-selected predictions. Lower task losses and signed prior-relative attack gains are better under the named finite audit. Three shared-cohort seed SDs are descriptive.

## Findings from the complete fixed matrix

**Real versus raw teacher.** E reduces mean measured recovery relative to R, while its residence loss increases by 0.006873–0.009489 unweighted across the four matched conditions. All 72/72 final seed/weight records meet the three source-loss allowances. This is a utility/recovery tradeoff, not dominance or privacy.

**Real versus permuted teacher.** E has MORE pooled SEX recovery than S in 4/4 unweighted matched-condition means. Race differences and seed signs vary. E improves mean residence relative to S, but the attribute-specific protection interpretation is weak after catch-up. E/S projection ranks and original-scaled distortion match within1e-6 in every seed; the realized control is not explained by unequal compression on those two measures.

**Reconstructing original PCA32.** Removing reconstruction from E-C changes pooled SEX/race attack LOSS by 0.007613 / 0.007531 and 0.006864 / 0.006609 (unweighted / PWGTP), reducing recovery. Residence loss changes by 0.002427 / 0.002623, and civilian-at-work loss by 0.005178 / 0.004661. R and S also respond; reconstruction pressure is not uniquely a real-erasure phenomenon.

**Protection gradient.** E/rho.1 D−C raises pooled SEX attack loss by 0.005231 / 0.004066, with residence change -0.000055 / 0.000108. After removing reconstruction, its SEX difference is 0.000045 / -0.000012; the incremental effect depends on the training objective.

**Learned versus direct teacher and stronger audits.** All four E students improve mean utility on all five tasks under both weights over direct E, while adding recovery. For E/rho0 D, pooled SEX/race gains are 0.024669 / 0.020418 and 0.071319 / 0.064367, versus direct E 0.008736 / 0.005874 and 0.048108 / 0.042453. Catch-up can reverse comparisons despite unchanged120/360 checkpoints; see [budget/exposure analysis](AUDIT_BUDGET.md). Person weighting is shown beside each primary contrast.

**Recoverable structure and design implication.** E students retain E accurately and recover part of qE that direct E cannot reconstruct affinely. This persists at rho0. Removed-component recoverability does not track protected-label gains monotonically, and qE is not pure sensitive information. The full-PCA32 input route and source objectives remain candidate paths for reintroduction. [Mechanism evidence](MECHANISM.md) supports testing that route explicitly; [the single next design](NEXT_DESIGN.md) is a proposal, not an additional run or a solution to purpose-specific combined-access coordination.

E−R compares true-label teacher erasure with raw teacher preservation. E−S compares real-attribute erasure with one fixed paired-label permutation; any realized rank/distortion mismatch would limit that control. rho0−rho.1 intervenes on reconstruction of original PCA32. D−C changes only the real protection gradient within a full-state fork. E-C still depends on real attributes through its fixed teacher.

Positive task-loss contrasts lose utility; positive attack-loss contrasts reduce measured recovery. Neither teacher construction nor any release is selected using these outcomes.

The standalone saved_adversary candidate is excluded, but the same predictions at epoch0 remain eligible within catch-up. Three new seed1 SEX trajectories (E/rho.1/D, E/rho0/C and E/rho0/D) select epoch0 and win pooled selection. Thus a pooled/fresh reversal can reflect inherited representation-fitting exposure without useful additional optimization. [Audit-budget evidence](AUDIT_BUDGET.md) separates this from any effect of longer fitting.

[Every paired seed score](PAIRED.csv), [all paired aggregates](PAIRED_AGGREGATE.csv), [unchanged criteria](criteria.json), [bank comparisons](bank_comparisons.json), [budget sensitivity](AUDIT_BUDGET.csv), [saved/fresh/catch-up](CATCHUP.csv).

## Controlled final contrasts

Cells are mean paired loss differences, unweighted / PWGTP. Audit columns use360. For attributes, the signed attack-gain difference is the NEGATIVE of this loss difference and is also explicit in PAIRED.csv. All five tasks, every seed, validation and both budgets remain in the linked CSVs; no result is selected for this table.

| Left − right | Residence U / W | SEX independent U / W | RAC1P independent U / W | SEX pooled U / W | RAC1P pooled U / W |
| --- | --- | --- | --- | --- | --- |
| E, rho=.1, C − R, rho=.1, C | 0.008741 / 0.005826 | 0.009038 / 0.009188 | 0.021423 / 0.023856 | 0.002654 / 0.003263 | 0.001042 / 0.004671 |
| E, rho=.1, C − S, rho=.1, C | -0.003539 / -0.003101 | -0.000233 / 0.000233 | 0.012144 / 0.010359 | -0.003515 / -0.003520 | 0.001812 / 0.001653 |
| E, rho=.1, D − R, rho=.1, D | 0.008390 / 0.005645 | 0.016934 / 0.016361 | 0.023198 / 0.025565 | 0.007362 / 0.007081 | 0.002253 / 0.005847 |
| E, rho=.1, D − S, rho=.1, D | -0.004091 / -0.003272 | 0.004779 / 0.005089 | 0.010834 / 0.009892 | -0.000858 / -0.002664 | -0.000801 / 0.000176 |
| E, rho=0, C − R, rho=0, C | 0.009489 / 0.006889 | 0.016496 / 0.016053 | 0.022163 / 0.021070 | 0.008658 / 0.010233 | 0.007022 / 0.009129 |
| E, rho=0, C − S, rho=0, C | -0.001317 / -0.001471 | 0.005740 / 0.005900 | 0.008526 / 0.004110 | -0.001566 / 0.001013 | 0.002844 / 0.002878 |
| E, rho=0, D − R, rho=0, D | 0.006873 / 0.005432 | 0.017774 / 0.017107 | 0.025297 / 0.024159 | 0.008099 / 0.009867 | 0.008300 / 0.010314 |
| E, rho=0, D − S, rho=0, D | -0.001984 / -0.001448 | 0.006512 / 0.006213 | 0.009515 / 0.005507 | -0.003660 / -0.000542 | 0.002476 / 0.002142 |
| R, rho=0, C − R, rho=.1, C | 0.001679 / 0.001560 | 0.001609 / 0.000561 | 0.000884 / 0.002150 | 0.001609 / 0.000561 | 0.000884 / 0.002150 |
| R, rho=0, D − R, rho=.1, D | 0.003636 / 0.002872 | 0.001690 / 0.000667 | 0.000405 / 0.001859 | 0.001690 / 0.000667 | 0.000405 / 0.001859 |
| E, rho=0, C − E, rho=.1, C | 0.002427 / 0.002623 | 0.009067 / 0.007426 | 0.001624 / -0.000635 | 0.007613 / 0.007531 | 0.006864 / 0.006609 |
| E, rho=0, D − E, rho=.1, D | 0.002119 / 0.002659 | 0.002530 / 0.001413 | 0.002504 / 0.000453 | 0.002428 / 0.003453 | 0.006451 / 0.006326 |
| S, rho=0, C − S, rho=.1, C | 0.000206 / 0.000994 | 0.003095 / 0.001759 | 0.005242 / 0.005614 | 0.005664 / 0.002998 | 0.005832 / 0.005384 |
| S, rho=0, D − S, rho=.1, D | 0.000012 / 0.000835 | 0.000797 / 0.000289 | 0.003824 / 0.004838 | 0.005229 / 0.001331 | 0.003174 / 0.004360 |
| R, rho=.1, D − R, rho=.1, C | 0.000296 / 0.000289 | 0.000523 / 0.000247 | 0.000549 / 0.000556 | 0.000523 / 0.000247 | 0.000549 / 0.000556 |
| R, rho=0, D − R, rho=0, C | 0.002252 / 0.001601 | 0.000604 / 0.000354 | 0.000070 / 0.000265 | 0.000604 / 0.000354 | 0.000070 / 0.000265 |
| E, rho=.1, D − E, rho=.1, C | -0.000055 / 0.000108 | 0.008419 / 0.007420 | 0.002324 / 0.002265 | 0.005231 / 0.004066 | 0.001760 / 0.001732 |
| E, rho=0, D − E, rho=0, C | -0.000363 / 0.000144 | 0.001882 / 0.001407 | 0.003204 / 0.003354 | 0.000045 / -0.000012 | 0.001347 / 0.001449 |
| S, rho=.1, D − S, rho=.1, C | 0.000497 / 0.000278 | 0.003408 / 0.002564 | 0.003634 / 0.002733 | 0.002574 / 0.003209 | 0.004373 / 0.003209 |
| S, rho=0, D − S, rho=0, C | 0.000303 / 0.000120 | 0.001110 / 0.001094 | 0.002215 / 0.001956 | 0.002139 / 0.001542 | 0.001715 / 0.002185 |

## Reconstruction interactions

Each value is (E/rho0−E/rho.1)−(reference/rho0−reference/rho.1), using paired seeds. A positive attack-loss interaction means removing reconstruction reduced E recovery more than reference recovery. A positive utility interaction means a larger E utility cost. Cells are unweighted / PWGTP; full five-task, both-budget, validation and mean±SD results are in [INTERACTIONS.csv](INTERACTIONS.csv) and [INTERACTIONS_AGGREGATE.csv](INTERACTIONS_AGGREGATE.csv).

| Reference / arm | Residence U / W | SEX independent U / W | RAC1P independent U / W | SEX pooled U / W | RAC1P pooled U / W |
| --- | --- | --- | --- | --- | --- |
| R / C | 0.000748 / 0.001063 | 0.007458 / 0.006865 | 0.000740 / -0.002786 | 0.006004 / 0.006970 | 0.005980 / 0.004458 |
| R / D | -0.001516 / -0.000213 | 0.000840 / 0.000745 | 0.002099 / -0.001406 | 0.000738 / 0.002785 | 0.006046 / 0.004467 |
| S / C | 0.002221 / 0.001630 | 0.005972 / 0.005667 | -0.003618 / -0.006249 | 0.001949 / 0.004533 | 0.001032 / 0.001224 |
| S / D | 0.002107 / 0.001824 | 0.001733 / 0.001124 | -0.001320 / -0.004384 | -0.002802 / 0.002122 | 0.003277 / 0.001966 |

## New learned release versus its direct teacher

Independent audits are matched by fitting pool and candidate budget. Pooled learned audits add inherited observer exposure; the static teacher has no catch-up.

| Learned − own teacher | Same residence U / W | Commute >20 min U / W | Income >$50k U / W | Civilian at work U / W | Public coverage U / W | SEX independent U / W | RAC1P independent U / W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E, rho=.1, C − E: real-attribute teacher | -0.007357 / -0.008549 | -0.002505 / -0.003053 | -0.068020 / -0.069389 | -0.128262 / -0.120668 | -0.032699 / -0.030126 | -0.017207 / -0.016137 | -0.011041 / -0.010786 |
| E, rho=.1, D − E: real-attribute teacher | -0.007412 / -0.008442 | -0.002825 / -0.003294 | -0.068180 / -0.069641 | -0.128424 / -0.120571 | -0.031867 / -0.029111 | -0.008788 / -0.008717 | -0.008718 / -0.008521 |
| E, rho=0, C − E: real-attribute teacher | -0.004930 / -0.005926 | -0.002747 / -0.001565 | -0.067282 / -0.069622 | -0.123084 / -0.116008 | -0.032891 / -0.030517 | -0.008140 / -0.008711 | -0.009418 / -0.011422 |
| E, rho=0, D − E: real-attribute teacher | -0.005293 / -0.005782 | -0.002840 / -0.001397 | -0.066799 / -0.068932 | -0.126906 / -0.119137 | -0.032138 / -0.030281 | -0.006258 / -0.007304 | -0.006214 / -0.008068 |
| S, rho=.1, C − S: permuted-label teacher | -0.008513 / -0.007793 | -0.002230 / 0.000597 | -0.048716 / -0.049023 | -0.095744 / -0.088189 | -0.029721 / -0.026847 | -0.009499 / -0.007988 | -0.016230 / -0.014808 |
| S, rho=.1, D − S: permuted-label teacher | -0.008016 / -0.007515 | -0.002803 / -0.000216 | -0.048623 / -0.048667 | -0.094930 / -0.087090 | -0.029673 / -0.027679 | -0.006092 / -0.005424 | -0.012597 / -0.012075 |
| S, rho=0, C − S: permuted-label teacher | -0.008307 / -0.006800 | -0.002694 / -0.001025 | -0.047248 / -0.047330 | -0.095508 / -0.088383 | -0.031517 / -0.031143 | -0.006404 / -0.006229 | -0.010989 / -0.009194 |
| S, rho=0, D − S: permuted-label teacher | -0.008004 / -0.006680 | -0.002629 / -0.001039 | -0.046947 / -0.046637 | -0.093440 / -0.085656 | -0.030593 / -0.029988 | -0.005294 / -0.005135 | -0.008773 / -0.007238 |

Stage W−I, final−W, beta0 comparisons and all source-task costs are complete in PAIRED.csv/PAIRED_AGGREGATE.csv. Stage utility losses and source margins are shown in TABLE.md and utility_stages.png. No I/W audit is inferred from its target teacher.

## Margins and support

Every source loss permits +.01 nats relative to original PCA32. Residence must retain half positive PCA32 headroom over the stronger unprotected bank. Nonpositive denominators are undefined. Attribute halving requires positive PCA32 gain and complete category/control coverage. Bank comparisons retain the .01 utility advantage and .005 maximum extra attribute gain. These are descriptive margins, not privacy budgets or noninferiority tests. No known failure is hidden by another undefined criterion. Historical helper keys containing "erased" name the compared release; they do not imply a final eraser was applied.

All nine race codes remain. Code4 is absent from attacker fitting/validation for every seed. Missing support is unassessable; signed gains are never clipped. All support and exposed-control failures remain in PER_CLASS.csv and SUPPORT.csv. I/W receive no new attribute audit.
