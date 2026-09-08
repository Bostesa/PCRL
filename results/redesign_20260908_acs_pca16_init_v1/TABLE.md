# PCA16-initialized bottleneck — stage evidence

These are development results on the original households, not untouched confirmation. I is the initialized mapper, W the common warm state, C_init the final task-only continuation, and D_init the final protected continuation. Each stage has independent utility probes after all training finishes. I/W have no attribute audits. Primary attribute comparisons use matched five-candidate independent audits; catch-up-inclusive selections remain separate. No saved selection is recomputed from development outcomes. Means ± sample SD describe seeds sharing a cohort.

Completed seeds: 0, 1, 2. [Analysis](ANALYSIS.md), [paired metrics](PAIRED.csv), [new candidate scores](PER_TARGET.csv), [new classes](PER_CLASS.csv), [new fitting curves](CURVES.csv), [PCA32 criteria](criteria.json), [bank comparisons](bank_comparisons.json).

## Validation: primary log loss, mean ± sample SD

| Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen PCA16 | 0.505908 ± 0.006331 | 0.678627 ± 0.002608 | 0.306442 ± 0.003365 | 0.302483 ± 0.007853 | 0.489936 ± 0.002588 | 0.657737 ± 0.006500 | 1.210469 ± 0.010444 |
| I initialized mapper | 0.505908 ± 0.006331 | 0.678627 ± 0.002608 | 0.306442 ± 0.003365 | 0.302483 ± 0.007853 | 0.489936 ± 0.002588 | not audited | not audited |
| W after common warm-up | 0.507664 ± 0.013582 | 0.674423 ± 0.000333 | 0.293998 ± 0.005360 | 0.290146 ± 0.008622 | 0.477323 ± 0.005013 | not audited | not audited |
| C initialized task-only | 0.512293 ± 0.012022 | 0.674835 ± 0.001025 | 0.295535 ± 0.005603 | 0.292678 ± 0.007942 | 0.480097 ± 0.004741 | 0.657862 ± 0.006334 | 1.218840 ± 0.008378 |
| D initialized protected | 0.513576 ± 0.011465 | 0.674507 ± 0.000928 | 0.296035 ± 0.005526 | 0.292438 ± 0.007575 | 0.479007 ± 0.006864 | 0.668242 ± 0.005119 | 1.239168 ± 0.002268 |
| Historical C task-only | 0.513373 ± 0.009356 | 0.676067 ± 0.001161 | 0.294968 ± 0.006757 | 0.294330 ± 0.006975 | 0.483496 ± 0.004615 | 0.664912 ± 0.004464 | 1.226887 ± 0.005408 |
| Historical D protected | 0.515896 ± 0.012953 | 0.675791 ± 0.001662 | 0.295692 ± 0.004684 | 0.293134 ± 0.006315 | 0.481045 ± 0.007482 | 0.673014 ± 0.003678 | 1.239003 ± 0.009375 |
| Original PCA32 | 0.500004 ± 0.007773 | 0.678247 ± 0.001214 | 0.306979 ± 0.001135 | 0.304701 ± 0.008147 | 0.486786 ± 0.001949 | 0.654803 ± 0.005947 | 1.208832 ± 0.010693 |
| PCA32 + LEACE | 0.510395 ± 0.006065 | 0.682046 ± 0.002754 | 0.321661 ± 0.007824 | 0.340376 ± 0.018177 | 0.507168 ± 0.005879 | 0.673654 ± 0.007783 | 1.240505 ± 0.014155 |
| Rich neural bank | 0.519075 ± 0.007071 | 0.672589 ± 0.001273 | 0.287795 ± 0.003673 | 0.287739 ± 0.008515 | 0.474939 ± 0.004183 | 0.660420 ± 0.001992 | 1.240366 ± 0.005523 |
| Neural bank + LEACE | 0.524121 ± 0.011871 | 0.677601 ± 0.002403 | 0.325812 ± 0.008173 | 0.324002 ± 0.006732 | 0.496884 ± 0.009311 | 0.668293 ± 0.004692 | 1.264294 ± 0.019221 |
| Rich tree bank | 0.527357 ± 0.011294 | 0.674715 ± 0.000318 | 0.287529 ± 0.002755 | 0.291640 ± 0.012045 | 0.475793 ± 0.006442 | 0.676885 ± 0.005214 | 1.252472 ± 0.008746 |
| Tree bank + LEACE | 0.536665 ± 0.010436 | 0.681729 ± 0.004078 | 0.349898 ± 0.001542 | 0.357083 ± 0.003590 | 0.500226 ± 0.012284 | 0.684147 ± 0.001832 | 1.287390 ± 0.004448 |
| Fitting prior | 0.543056 ± 0.013563 | 0.693473 ± 0.000409 | 0.480116 ± 0.006864 | 0.626051 ± 0.011224 | 0.553401 ± 0.010565 | 0.692388 ± 0.000413 | 1.283497 ± 0.005963 |

## Validation: every seed, new stages

| Seed | Stage | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | I initialized mapper | 0.513083 | 0.680875 | 0.307815 | 0.311123 | 0.492435 | not audited | not audited |
| 0 | W after common warm-up | 0.511382 | 0.674117 | 0.290602 | 0.300101 | 0.478455 | not audited | not audited |
| 0 | C initialized task-only | 0.513358 | 0.673689 | 0.292323 | 0.301848 | 0.481873 | 0.653812 | 1.226727 |
| 0 | D initialized protected | 0.513945 | 0.673437 | 0.292151 | 0.301162 | 0.483077 | 0.666909 | 1.237714 |
| 1 | I initialized mapper | 0.501109 | 0.679238 | 0.308903 | 0.295780 | 0.490107 | not audited | not audited |
| 1 | W after common warm-up | 0.492610 | 0.674372 | 0.291215 | 0.285326 | 0.481673 | not audited | not audited |
| 1 | C initialized task-only | 0.499774 | 0.675662 | 0.292278 | 0.288062 | 0.483693 | 0.654613 | 1.219746 |
| 1 | D initialized protected | 0.501931 | 0.675106 | 0.293592 | 0.287525 | 0.482862 | 0.663921 | 1.241781 |
| 2 | I initialized mapper | 0.503531 | 0.675767 | 0.302608 | 0.300546 | 0.487267 | not audited | not audited |
| 2 | W after common warm-up | 0.518999 | 0.674778 | 0.300178 | 0.285011 | 0.471840 | not audited | not audited |
| 2 | C initialized task-only | 0.523747 | 0.675155 | 0.302004 | 0.288124 | 0.474724 | 0.665161 | 1.210046 |
| 2 | D initialized protected | 0.524851 | 0.674977 | 0.302362 | 0.288628 | 0.471082 | 0.673895 | 1.238008 |

## Validation: residence stage differences

Positive differences lose utility. Tiny nonzero differences are shown in scientific notation. These comparisons localize observed changes; they do not select a stage.

| Contrast (left − right) | Mean ± sample SD |
| --- | --- |
| I initialized mapper − Frozen PCA16 | -3.147e-10 ± 1.337e-10 |
| W after common warm-up − I initialized mapper | 0.001756 ± 0.012352 |
| C initialized task-only − W after common warm-up | 0.004629 ± 0.002596 |
| D initialized protected − C initialized task-only | 0.001282 ± 0.000800 |
| C initialized task-only − Historical C task-only | -0.001080 ± 0.003377 |
| D initialized protected − Historical D protected | -0.002321 ± 0.003215 |

## Development evaluation: primary log loss, mean ± sample SD

| Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen PCA16 | 0.495062 ± 0.002095 | 0.685527 ± 0.003931 | 0.294408 ± 0.007520 | 0.290035 ± 0.007223 | 0.505704 ± 0.008631 | 0.659948 ± 0.004457 | 1.212077 ± 0.020218 |
| I initialized mapper | 0.495062 ± 0.002095 | 0.685527 ± 0.003931 | 0.294408 ± 0.007520 | 0.290035 ± 0.007223 | 0.505704 ± 0.008631 | not audited | not audited |
| W after common warm-up | 0.498230 ± 0.004660 | 0.684743 ± 0.004745 | 0.286573 ± 0.009795 | 0.276222 ± 0.005653 | 0.493155 ± 0.012802 | not audited | not audited |
| C initialized task-only | 0.499489 ± 0.001184 | 0.685948 ± 0.003371 | 0.289622 ± 0.011301 | 0.277046 ± 0.007070 | 0.495113 ± 0.011102 | 0.655859 ± 0.007702 | 1.218044 ± 0.025271 |
| D initialized protected | 0.501559 ± 0.002914 | 0.685954 ± 0.003734 | 0.290818 ± 0.010429 | 0.276961 ± 0.008175 | 0.494527 ± 0.010908 | 0.663448 ± 0.006426 | 1.240585 ± 0.021967 |
| Historical C task-only | 0.500295 ± 0.003986 | 0.684432 ± 0.005534 | 0.291447 ± 0.010328 | 0.279831 ± 0.007417 | 0.495148 ± 0.012124 | 0.658537 ± 0.006823 | 1.228641 ± 0.023749 |
| Historical D protected | 0.501592 ± 0.002660 | 0.685018 ± 0.004689 | 0.292097 ± 0.011459 | 0.279840 ± 0.007691 | 0.495805 ± 0.015940 | 0.667091 ± 0.008388 | 1.240704 ± 0.017866 |
| Original PCA32 | 0.487411 ± 0.005595 | 0.687570 ± 0.008575 | 0.298578 ± 0.008781 | 0.288407 ± 0.005005 | 0.504623 ± 0.011550 | 0.655218 ± 0.011326 | 1.200399 ± 0.022578 |
| PCA32 + LEACE | 0.496275 ± 0.004211 | 0.691074 ± 0.007655 | 0.312714 ± 0.009209 | 0.322018 ± 0.009292 | 0.521313 ± 0.013445 | 0.678120 ± 0.008021 | 1.235046 ± 0.027316 |
| Rich neural bank | 0.510529 ± 0.003504 | 0.681481 ± 0.002771 | 0.278421 ± 0.007137 | 0.270941 ± 0.011699 | 0.487067 ± 0.012910 | 0.664770 ± 0.008175 | 1.243173 ± 0.032479 |
| Neural bank + LEACE | 0.515247 ± 0.005386 | 0.687252 ± 0.002053 | 0.318360 ± 0.002766 | 0.315928 ± 0.017206 | 0.505838 ± 0.008451 | 0.671515 ± 0.007842 | 1.268265 ± 0.035414 |
| Rich tree bank | 0.517356 ± 0.002162 | 0.681863 ± 0.004760 | 0.279997 ± 0.009591 | 0.274990 ± 0.006996 | 0.488114 ± 0.014003 | 0.676694 ± 0.002856 | 1.259321 ± 0.019889 |
| Tree bank + LEACE | 0.532475 ± 0.005421 | 0.689332 ± 0.003410 | 0.327893 ± 0.006511 | 0.330752 ± 0.012562 | 0.512242 ± 0.008722 | 0.685465 ± 0.002333 | 1.290130 ± 0.014079 |
| Fitting prior | 0.536538 ± 0.005614 | 0.693188 ± 0.000380 | 0.481821 ± 0.006349 | 0.632113 ± 0.006230 | 0.565686 ± 0.009423 | 0.692687 ± 0.000573 | 1.290043 ± 0.025603 |

## Development evaluation: every seed, new stages

| Seed | Stage | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | I initialized mapper | 0.496200 | 0.689505 | 0.299625 | 0.286524 | 0.515307 | not audited | not audited |
| 0 | W after common warm-up | 0.499940 | 0.689023 | 0.293513 | 0.271932 | 0.507910 | not audited | not audited |
| 0 | C initialized task-only | 0.498125 | 0.687898 | 0.297984 | 0.272594 | 0.507824 | 0.658248 | 1.220709 |
| 0 | D initialized protected | 0.498654 | 0.687739 | 0.298257 | 0.272050 | 0.506932 | 0.667038 | 1.240987 |
| 1 | I initialized mapper | 0.492644 | 0.681644 | 0.285788 | 0.285238 | 0.498593 | not audited | not audited |
| 1 | W after common warm-up | 0.492957 | 0.679641 | 0.275369 | 0.274106 | 0.486548 | not audited | not audited |
| 1 | C initialized task-only | 0.500253 | 0.682055 | 0.276765 | 0.273346 | 0.490199 | 0.647246 | 1.241876 |
| 1 | D initialized protected | 0.501541 | 0.681662 | 0.278897 | 0.272433 | 0.490213 | 0.656029 | 1.262349 |
| 2 | I initialized mapper | 0.496342 | 0.685431 | 0.297810 | 0.298343 | 0.503211 | not audited | not audited |
| 2 | W after common warm-up | 0.501793 | 0.685566 | 0.290839 | 0.282628 | 0.485006 | not audited | not audited |
| 2 | C initialized task-only | 0.500091 | 0.687892 | 0.294116 | 0.285199 | 0.487317 | 0.662083 | 1.191545 |
| 2 | D initialized protected | 0.504482 | 0.688460 | 0.295301 | 0.286398 | 0.486437 | 0.667278 | 1.218420 |

## Development evaluation: residence stage differences

Positive differences lose utility. Tiny nonzero differences are shown in scientific notation. These comparisons localize observed changes; they do not select a stage.

| Contrast (left − right) | Mean ± sample SD |
| --- | --- |
| I initialized mapper − Frozen PCA16 | -1.940e-11 ± 2.920e-10 |
| W after common warm-up − I initialized mapper | 0.003168 ± 0.002616 |
| C initialized task-only − W after common warm-up | 0.001259 ± 0.005228 |
| D initialized protected − C initialized task-only | 0.002070 ± 0.002047 |
| C initialized task-only − Historical C task-only | -0.000805 ± 0.002869 |
| D initialized protected − Historical D protected | -0.000033 ± 0.000492 |

## Catch-up-inclusive audit sensitivity

Only additional catch-up candidates inherit representation-training exposure before fitting on the attacker pool. The inclusive selector may still choose a fresh independent candidate. Saved training adversaries are diagnostic-only.

| Split | Release | SEX independent | SEX inclusive | RAC1P independent | RAC1P inclusive |
| --- | --- | --- | --- | --- | --- |
| validation | C initialized task-only | 0.657862 ± 0.006334 | 0.657862 ± 0.006334 | 1.218840 ± 0.008378 | 1.217819 ± 0.007012 |
| validation | D initialized protected | 0.668242 ± 0.005119 | 0.667478 ± 0.006153 | 1.239168 ± 0.002268 | 1.227287 ± 0.012265 |
| validation | Historical C task-only | 0.664912 ± 0.004464 | 0.664912 ± 0.004464 | 1.226887 ± 0.005408 | 1.225577 ± 0.003344 |
| validation | Historical D protected | 0.673014 ± 0.003678 | 0.672774 ± 0.003827 | 1.239003 ± 0.009375 | 1.229728 ± 0.005403 |
| development evaluation | C initialized task-only | 0.655859 ± 0.007702 | 0.655859 ± 0.007702 | 1.218044 ± 0.025271 | 1.221850 ± 0.026693 |
| development evaluation | D initialized protected | 0.663448 ± 0.006426 | 0.667260 ± 0.000214 | 1.240585 ± 0.021967 | 1.232176 ± 0.024698 |
| development evaluation | Historical C task-only | 0.658537 ± 0.006823 | 0.658537 ± 0.006823 | 1.228641 ± 0.023749 | 1.230281 ± 0.022703 |
| development evaluation | Historical D protected | 0.667091 ± 0.008388 | 0.668615 ± 0.007089 | 1.240704 ± 0.017866 | 1.232861 ± 0.023236 |

## Original-PCA32 source and residence references

Each source loss permits at most +.01 nats relative to original PCA32. Residence retains half the positive PCA32 advantage over the stronger unprotected rich bank. Nonpositive headroom stays undefined. These descriptive references are not an all-attribute admission gate.

| Seed | Split | Stage | Income Δ | Work Δ | Coverage Δ | Source all | Residence retained | Residence half |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | I initialized mapper | 0.001060 | -0.002974 | 0.003619 | pass | 0.381145 | fail |
| 0 | validation | W after common warm-up | -0.016153 | -0.013996 | -0.010361 | pass | 0.491679 | fail |
| 0 | validation | C initialized task-only | -0.014432 | -0.012249 | -0.006944 | pass | 0.363262 | fail |
| 0 | validation | D initialized protected | -0.014604 | -0.012935 | -0.005739 | pass | 0.325126 | fail |
| 0 | development evaluation | I initialized mapper | -0.007273 | 0.001769 | -0.002614 | pass | 0.771543 | pass |
| 0 | development evaluation | W after common warm-up | -0.013385 | -0.012823 | -0.010012 | pass | 0.546574 | pass |
| 0 | development evaluation | C initialized task-only | -0.008914 | -0.012161 | -0.010097 | pass | 0.655752 | pass |
| 0 | development evaluation | D initialized protected | -0.008641 | -0.012705 | -0.010989 | pass | 0.623951 | pass |
| 1 | validation | I initialized mapper | 0.002931 | -0.003808 | 0.003493 | pass | 0.522390 | pass |
| 1 | validation | W after common warm-up | -0.014758 | -0.014261 | -0.004941 | pass | 0.927482 | pass |
| 1 | validation | C initialized task-only | -0.013694 | -0.011526 | -0.002920 | pass | 0.586034 | pass |
| 1 | validation | D initialized protected | -0.012381 | -0.012063 | -0.003752 | pass | 0.483237 | fail |
| 1 | development evaluation | I initialized mapper | -0.003612 | -0.001115 | 0.001490 | pass | 0.659917 | pass |
| 1 | development evaluation | W after common warm-up | -0.014031 | -0.012248 | -0.010556 | pass | 0.650485 | pass |
| 1 | development evaluation | C initialized task-only | -0.012635 | -0.013007 | -0.006905 | pass | 0.430540 | fail |
| 1 | development evaluation | D initialized protected | -0.010502 | -0.013921 | -0.006891 | pass | 0.391699 | fail |
| 2 | validation | I initialized mapper | -0.005601 | 0.000126 | 0.002338 | pass | 1.087794 | pass |
| 2 | validation | W after common warm-up | -0.008031 | -0.015409 | -0.013089 | pass | 0.345821 | fail |
| 2 | validation | C initialized task-only | -0.006204 | -0.012295 | -0.010204 | pass | 0.118075 | fail |
| 2 | validation | D initialized protected | -0.005847 | -0.011792 | -0.013847 | pass | 0.065138 | fail |
| 2 | development evaluation | I initialized mapper | -0.001627 | 0.004231 | 0.004367 | pass | 0.597370 | pass |
| 2 | development evaluation | W after common warm-up | -0.008599 | -0.011485 | -0.013839 | pass | 0.318705 | fail |
| 2 | development evaluation | C initialized task-only | -0.005322 | -0.008914 | -0.011527 | pass | 0.405748 | fail |
| 2 | development evaluation | D initialized protected | -0.004137 | -0.007714 | -0.012408 | pass | 0.181224 | fail |

Full race-schema coverage is incomplete. No absent class, unaudited stage, or unsuccessful finite attacker counts as protection. Historical scores and flags remain unchanged.

## Final-stage descriptive policy criteria

The original PCA32 is the parent for source preservation, residential retention, and attribute-gain halving. Halving requires positive parent gain and complete audit/exposed-control coverage. Known failures remain failures when another criterion is undefined. Both saved audit candidate sets are shown separately.

| Seed | Split | Final release | Audit selector | Source all | Residence half | SEX half gain | RAC1P half gain | Joint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | C initialized task-only | primary | pass | fail | fail | undefined | fail |
| 0 | validation | D initialized protected | primary | pass | fail | fail | undefined | fail |
| 0 | validation | C initialized task-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | validation | D initialized protected | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | development evaluation | C initialized task-only | primary | pass | pass | fail | undefined | fail |
| 0 | development evaluation | D initialized protected | primary | pass | pass | fail | undefined | fail |
| 0 | development evaluation | C initialized task-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development evaluation | D initialized protected | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | C initialized task-only | primary | pass | pass | fail | undefined | fail |
| 1 | validation | D initialized protected | primary | pass | fail | fail | undefined | fail |
| 1 | validation | C initialized task-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | D initialized protected | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | development evaluation | C initialized task-only | primary | pass | fail | fail | undefined | fail |
| 1 | development evaluation | D initialized protected | primary | pass | fail | fail | undefined | fail |
| 1 | development evaluation | C initialized task-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | development evaluation | D initialized protected | catchup_inclusive | pass | fail | pass | undefined | fail |
| 2 | validation | C initialized task-only | primary | pass | fail | fail | undefined | fail |
| 2 | validation | D initialized protected | primary | pass | fail | fail | undefined | fail |
| 2 | validation | C initialized task-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | validation | D initialized protected | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development evaluation | C initialized task-only | primary | pass | fail | fail | undefined | fail |
| 2 | development evaluation | D initialized protected | primary | pass | fail | fail | undefined | fail |
| 2 | development evaluation | C initialized task-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development evaluation | D initialized protected | catchup_inclusive | pass | fail | fail | undefined | fail |
