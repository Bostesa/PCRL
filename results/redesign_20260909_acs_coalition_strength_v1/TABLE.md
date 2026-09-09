# Fixed coalition-strength comparison

The full 54-system matrix is complete: 18 historical systems and 36 new continuations.

These are development-evaluation losses and signed prior-relative attack gains, in nats. Lower utility loss and lower attack gain are better. Each value is mean ± sample SD across the same three cohort-sharing seeds; the SD is descriptive. Unweighted validation selects each predictor once; PWGTP scores those same predictions. Display rounding does not enter any comparison; full saved values and the separate 1e−12 roundoff rule are used. The tables use the expanded catch-up-inclusive 360-epoch audit. Other scopes and both budgets remain in [the complete selected results](PER_SEED.csv) and [aggregates](AGGREGATE.csv).

Source feasibility requires **each** of the three independent source heads to be within .01 nats of its original same-seed, same-weight PCA32 head. It is not a source average, native-head comparison or statistical noninferiority test. The primary comparison is F, SEX, residential-transfer utility, δ=.001, unweighted. Every fixed J/local coefficient pair is reported; no coefficient or release is selected for use.

β changes only the additional sensitive-attribute penalty. The ordinary individual protection term remains at −.1. Iplus applies the extra local sum M_A+M_B; J applies the coalition mean M_AB. At β=0 they are the same ordinary I system. F releases two feature vectors; P releases the matched A two-source and B one-source probabilities. The forward architecture, fitting data, source-loss weights and total fitting capacity are matched.

## Unweighted

| System | Source pass | Income | Employment | Coverage | Residence | Commute | AB SEX gain | AB RAC1P gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F I (shared β=0) | 2/3 | 0.29283 ± 0.01052 | 0.27711 ± 0.00831 | 0.51497 ± 0.01077 | 0.51163 ± 0.01246 | 0.69049 ± 0.00137 | 0.01887 ± 0.00922 | 0.04483 ± 0.00388 |
| F Iplus β=0.025 | 2/3 | 0.29212 ± 0.01014 | 0.27752 ± 0.00749 | 0.51368 ± 0.01014 | 0.50875 ± 0.00934 | 0.69053 ± 0.00071 | 0.01048 ± 0.01194 | 0.04614 ± 0.00504 |
| F J β=0.025 | 2/3 | 0.29219 ± 0.01069 | 0.27800 ± 0.00663 | 0.51142 ± 0.00872 | 0.51311 ± 0.01143 | 0.69473 ± 0.00365 | 0.01174 ± 0.00540 | 0.04205 ± 0.00364 |
| F Iplus β=0.05 | 2/3 | 0.29284 ± 0.01036 | 0.27844 ± 0.00848 | 0.51192 ± 0.00988 | 0.51755 ± 0.00974 | 0.69071 ± 0.00350 | 0.01274 ± 0.00817 | 0.04548 ± 0.00789 |
| F J β=0.05 | 2/3 | 0.29255 ± 0.01112 | 0.27813 ± 0.00844 | 0.51686 ± 0.01022 | 0.51515 ± 0.01613 | 0.69266 ± 0.00460 | 0.01055 ± 0.00506 | 0.03538 ± 0.00687 |
| F Iplus β=0.1 | 2/3 | 0.29305 ± 0.01193 | 0.27870 ± 0.00761 | 0.51240 ± 0.00837 | 0.51361 ± 0.00517 | 0.69065 ± 0.00030 | 0.01123 ± 0.00546 | 0.04164 ± 0.00543 |
| F J β=0.1 | 2/3 | 0.29382 ± 0.01208 | 0.27760 ± 0.00680 | 0.51325 ± 0.00851 | 0.51743 ± 0.00961 | 0.69174 ± 0.00252 | 0.00541 ± 0.00272 | 0.03828 ± 0.00762 |
| F Iplus β=0.2 | 1/3 | 0.29292 ± 0.00982 | 0.27904 ± 0.00857 | 0.51701 ± 0.00201 | 0.51725 ± 0.00646 | 0.69252 ± 0.00332 | 0.01357 ± 0.00500 | 0.04375 ± 0.00738 |
| F J β=0.2 | 1/3 | 0.29553 ± 0.01060 | 0.27843 ± 0.00710 | 0.51328 ± 0.00666 | 0.51606 ± 0.01040 | 0.69060 ± 0.00332 | 0.01255 ± 0.00425 | 0.02441 ± 0.00407 |
| P I (shared β=0) | 2/3 | 0.29559 ± 0.01529 | 0.28662 ± 0.00372 | 0.51381 ± 0.00825 | 0.53068 ± 0.00352 | 0.69298 ± 0.00068 | 0.00437 ± 0.00259 | 0.02583 ± 0.00222 |
| P Iplus β=0.025 | 2/3 | 0.29592 ± 0.01544 | 0.28688 ± 0.00357 | 0.51405 ± 0.00805 | 0.53074 ± 0.00373 | 0.69281 ± 0.00089 | -0.00303 ± 0.01306 | 0.02560 ± 0.00123 |
| P J β=0.025 | 2/3 | 0.29590 ± 0.01541 | 0.28707 ± 0.00391 | 0.51393 ± 0.00839 | 0.53108 ± 0.00325 | 0.69273 ± 0.00080 | 0.00378 ± 0.00281 | 0.02550 ± 0.00203 |
| P Iplus β=0.05 | 1/3 | 0.29596 ± 0.01544 | 0.28683 ± 0.00367 | 0.51418 ± 0.00812 | 0.53114 ± 0.00335 | 0.69307 ± 0.00075 | 0.00380 ± 0.00209 | 0.02589 ± 0.00048 |
| P J β=0.05 | 2/3 | 0.29598 ± 0.01541 | 0.28722 ± 0.00399 | 0.51394 ± 0.00868 | 0.53063 ± 0.00394 | 0.69276 ± 0.00087 | 0.00343 ± 0.00276 | 0.02620 ± 0.00094 |
| P Iplus β=0.1 | 1/3 | 0.29628 ± 0.01511 | 0.28704 ± 0.00390 | 0.51449 ± 0.00834 | 0.53097 ± 0.00374 | 0.69298 ± 0.00095 | 0.00337 ± 0.00253 | 0.02501 ± 0.00082 |
| P J β=0.1 | 1/3 | 0.29519 ± 0.01415 | 0.28758 ± 0.00367 | 0.51425 ± 0.00835 | 0.53108 ± 0.00333 | 0.69274 ± 0.00065 | 0.00319 ± 0.00253 | 0.02557 ± 0.00093 |
| P Iplus β=0.2 | 1/3 | 0.29749 ± 0.01433 | 0.28749 ± 0.00393 | 0.51526 ± 0.00822 | 0.53144 ± 0.00382 | 0.69292 ± 0.00074 | 0.00302 ± 0.00232 | 0.02308 ± 0.00031 |
| P J β=0.2 | 1/3 | 0.29640 ± 0.01376 | 0.28801 ± 0.00397 | 0.51475 ± 0.00839 | 0.53167 ± 0.00387 | 0.69299 ± 0.00073 | 0.00193 ± 0.00255 | 0.02365 ± 0.00095 |
| E fixed teacher (context) | 0/3 | 0.35666 ± 0.01755 | 0.41451 ± 0.05529 | 0.52916 ± 0.02563 | 0.50838 ± 0.00204 | 0.68724 ± 0.00406 | 0.00820 ± 0.00419 | 0.05271 ± 0.00259 |

## PWGTP

| System | Source pass | Income | Employment | Coverage | Residence | Commute | AB SEX gain | AB RAC1P gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F I (shared β=0) | 2/3 | 0.30396 ± 0.00855 | 0.26653 ± 0.01443 | 0.52529 ± 0.00773 | 0.48905 ± 0.01211 | 0.69088 ± 0.00132 | 0.01557 ± 0.00553 | 0.04414 ± 0.00467 |
| F Iplus β=0.025 | 2/3 | 0.30332 ± 0.00897 | 0.26709 ± 0.01379 | 0.52330 ± 0.00870 | 0.48598 ± 0.00494 | 0.68963 ± 0.00067 | 0.00594 ± 0.00868 | 0.04549 ± 0.00504 |
| F J β=0.025 | 2/3 | 0.30494 ± 0.00802 | 0.26744 ± 0.01243 | 0.52164 ± 0.00666 | 0.48993 ± 0.00607 | 0.69338 ± 0.00517 | 0.00925 ± 0.00290 | 0.04520 ± 0.00781 |
| F Iplus β=0.05 | 2/3 | 0.30424 ± 0.00881 | 0.26854 ± 0.01391 | 0.52172 ± 0.00860 | 0.49508 ± 0.00484 | 0.69073 ± 0.00368 | 0.01161 ± 0.00606 | 0.04453 ± 0.00594 |
| F J β=0.05 | 2/3 | 0.30482 ± 0.00879 | 0.26760 ± 0.01401 | 0.52566 ± 0.00787 | 0.49074 ± 0.01077 | 0.69094 ± 0.00378 | 0.00809 ± 0.00732 | 0.03501 ± 0.00855 |
| F Iplus β=0.1 | 2/3 | 0.30568 ± 0.01012 | 0.26807 ± 0.01252 | 0.52193 ± 0.00971 | 0.49214 ± 0.00296 | 0.69039 ± 0.00017 | 0.00686 ± 0.00358 | 0.04251 ± 0.00064 |
| F J β=0.1 | 2/3 | 0.30511 ± 0.01166 | 0.26694 ± 0.01203 | 0.52325 ± 0.00884 | 0.49342 ± 0.00597 | 0.69132 ± 0.00216 | 0.00415 ± 0.00100 | 0.03783 ± 0.01153 |
| F Iplus β=0.2 | 1/3 | 0.30695 ± 0.00812 | 0.26912 ± 0.01393 | 0.52617 ± 0.00159 | 0.49353 ± 0.00299 | 0.69225 ± 0.00387 | 0.00873 ± 0.00564 | 0.03921 ± 0.00553 |
| F J β=0.2 | 2/3 | 0.30707 ± 0.00866 | 0.26847 ± 0.01352 | 0.52425 ± 0.00770 | 0.49296 ± 0.00945 | 0.69030 ± 0.00403 | 0.01111 ± 0.00170 | 0.02150 ± 0.00136 |
| P I (shared β=0) | 1/3 | 0.30783 ± 0.01618 | 0.27627 ± 0.00894 | 0.52439 ± 0.00699 | 0.50462 ± 0.00599 | 0.69253 ± 0.00068 | 0.00205 ± 0.00544 | 0.02652 ± 0.00558 |
| P Iplus β=0.025 | 1/3 | 0.30815 ± 0.01634 | 0.27652 ± 0.00900 | 0.52461 ± 0.00682 | 0.50471 ± 0.00588 | 0.69221 ± 0.00106 | -0.00605 ± 0.01374 | 0.02627 ± 0.00447 |
| P J β=0.025 | 1/3 | 0.30819 ± 0.01612 | 0.27668 ± 0.00913 | 0.52453 ± 0.00722 | 0.50497 ± 0.00611 | 0.69217 ± 0.00113 | 0.00278 ± 0.00455 | 0.02629 ± 0.00573 |
| P Iplus β=0.05 | 1/3 | 0.30844 ± 0.01626 | 0.27658 ± 0.00898 | 0.52495 ± 0.00704 | 0.50508 ± 0.00616 | 0.69266 ± 0.00069 | 0.00203 ± 0.00455 | 0.02634 ± 0.00355 |
| P J β=0.05 | 1/3 | 0.30826 ± 0.01611 | 0.27674 ± 0.00948 | 0.52449 ± 0.00763 | 0.50452 ± 0.00547 | 0.69216 ± 0.00114 | 0.00216 ± 0.00487 | 0.02696 ± 0.00353 |
| P Iplus β=0.1 | 1/3 | 0.30884 ± 0.01583 | 0.27666 ± 0.00921 | 0.52511 ± 0.00740 | 0.50493 ± 0.00591 | 0.69235 ± 0.00112 | 0.00177 ± 0.00559 | 0.02536 ± 0.00343 |
| P J β=0.1 | 2/3 | 0.30726 ± 0.01413 | 0.27709 ± 0.00919 | 0.52490 ± 0.00733 | 0.50484 ± 0.00591 | 0.69214 ± 0.00100 | 0.00136 ± 0.00519 | 0.02627 ± 0.00344 |
| P Iplus β=0.2 | 1/3 | 0.31006 ± 0.01441 | 0.27704 ± 0.00907 | 0.52586 ± 0.00744 | 0.50489 ± 0.00601 | 0.69239 ± 0.00098 | 0.00145 ± 0.00424 | 0.02359 ± 0.00363 |
| P J β=0.2 | 2/3 | 0.30886 ± 0.01383 | 0.27747 ± 0.00964 | 0.52547 ± 0.00736 | 0.50524 ± 0.00575 | 0.69251 ± 0.00057 | 0.00157 ± 0.00404 | 0.02398 ± 0.00301 |
| E fixed teacher (context) | 0/3 | 0.36938 ± 0.01938 | 0.39921 ± 0.05715 | 0.53528 ± 0.02223 | 0.48616 ± 0.00584 | 0.68969 ± 0.00264 | 0.00449 ± 0.00353 | 0.04623 ± 0.00430 |

## Fixed pairs in the primary utility panel

The direct E teacher is a reused context control with a different fitting history. Its reserved-task utility and source-floor failures are both retained; it is not an additional coefficient-grid system.

Cells below give the number of the **same fixed pair’s three seeds** qualifying under close matching / directional comparison. Both systems must pass all source floors and J must have strictly lower AB SEX gain. Close matching requires every absolute utility difference ≤.001; directional comparison permits utility improvements of any size while limiting every deterioration to .001. This distinction prevents calling distant utility points matched.

**F; unweighted**

| J β / Iplus β | 0 | 0.025 | 0.05 | 0.1 | 0.2 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.025 | 0/3 / 1/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.05 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.1 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.2 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |

**F; PWGTP**

| J β / Iplus β | 0 | 0.025 | 0.05 | 0.1 | 0.2 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.025 | 0/3 / 1/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.05 | 0/3 / 1/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.1 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 1/3 | 0/3 / 0/3 |
| 0.2 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |

**P; unweighted**

| J β / Iplus β | 0 | 0.025 | 0.05 | 0.1 | 0.2 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0/3 / 0/3 | 1/3 / 1/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.025 | 1/3 / 1/3 | 1/3 / 1/3 | 1/3 / 1/3 | 1/3 / 1/3 | 0/3 / 0/3 |
| 0.05 | 1/3 / 1/3 | 1/3 / 1/3 | 0/3 / 1/3 | 1/3 / 1/3 | 0/3 / 0/3 |
| 0.1 | 0/3 / 0/3 | 0/3 / 0/3 | 1/3 / 1/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.2 | 0/3 / 0/3 | 0/3 / 0/3 | 1/3 / 1/3 | 1/3 / 1/3 | 0/3 / 1/3 |

**P; PWGTP**

| J β / Iplus β | 0 | 0.025 | 0.05 | 0.1 | 0.2 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.025 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.05 | 1/3 / 1/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |
| 0.1 | 0/3 / 0/3 | 0/3 / 0/3 | 1/3 / 1/3 | 1/3 / 1/3 | 0/3 / 0/3 |
| 0.2 | 0/3 / 0/3 | 0/3 / 0/3 | 1/3 / 1/3 | 0/3 / 0/3 | 1/3 / 1/3 |

The zero entries in the family grid name the same physical I anchor. Their self-comparison cannot show a strict improvement. No seed-specific choice, interpolation, or average restricted to qualifying seeds is used. Race comparisons are aggregate numerical descriptions: census code 4 is absent from independent attacker fitting and attacker validation. All nine categories and pool-specific observer exposure remain reported, and full race assessment is unassessable.

[All panels, deltas and audit-scope comparisons](MATCHING_ANALYSIS.md) · [Native source heads](NATIVE_SOURCE.md) · [Audit budgets and exposure](AUDIT_FINDINGS.md) · [Contextual references](CONTEXTUAL_REFERENCES.md) · [Frozen numerical rules](comparison_rules.json)

[Complete individual A/B forbidden-target tables](INDIVIDUAL_TARGETS.md) gives every condition under both weightings, including the opposing reserved targets.
