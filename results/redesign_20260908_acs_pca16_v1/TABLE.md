# PCA16 dimension diagnostic — development evidence

PCA16 is the fixed first16-column slice of the original PCA32 release, before head-specific standardization. All original test households are development evaluation. No fitting or selection is performed by this report. The primary comparison uses matched five-candidate independent auditors for every release: original `independent_selected` for C/D and saved `selected` for PCA16 and other references. Historical catch-up-inclusive C/D choices are shown separately; original flags and files remain unchanged. Natural log loss is primary; means ± sample SD describe seeds sharing a cohort.

Completed seeds: 0, 1, 2. [Analysis](ANALYSIS.md), [new PCA16 candidates](PER_TARGET.csv), [new class scores](PER_CLASS.csv), [paired scores](PAIRED.csv), [historical support/exposed controls](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md), [historical tables](../redesign_20260908_acs_bottleneck_v1/TABLE.md).

## Validation: primary log loss, mean ± sample SD

| Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original PCA32 | 0.500004 ± 0.007773 | 0.678247 ± 0.001214 | 0.306979 ± 0.001135 | 0.304701 ± 0.008147 | 0.486786 ± 0.001949 | 0.654803 ± 0.005947 | 1.208832 ± 0.010693 |
| PCA16 | 0.505908 ± 0.006331 | 0.678627 ± 0.002608 | 0.306442 ± 0.003365 | 0.302483 ± 0.007853 | 0.489936 ± 0.002588 | 0.657737 ± 0.006500 | 1.210469 ± 0.010444 |
| PCA32 + LEACE | 0.510395 ± 0.006065 | 0.682046 ± 0.002754 | 0.321661 ± 0.007824 | 0.340376 ± 0.018177 | 0.507168 ± 0.005879 | 0.673654 ± 0.007783 | 1.240505 ± 0.014155 |
| C task-only bottleneck (16D) | 0.513373 ± 0.009356 | 0.676067 ± 0.001161 | 0.294968 ± 0.006757 | 0.294330 ± 0.006975 | 0.483496 ± 0.004615 | 0.664912 ± 0.004464 | 1.226887 ± 0.005408 |
| D protected bottleneck (16D) | 0.515896 ± 0.012953 | 0.675791 ± 0.001662 | 0.295692 ± 0.004684 | 0.293134 ± 0.006315 | 0.481045 ± 0.007482 | 0.673014 ± 0.003678 | 1.239003 ± 0.009375 |
| Rich neural bank (26D) | 0.519075 ± 0.007071 | 0.672589 ± 0.001273 | 0.287795 ± 0.003673 | 0.287739 ± 0.008515 | 0.474939 ± 0.004183 | 0.660420 ± 0.001992 | 1.240366 ± 0.005523 |
| Neural bank + LEACE | 0.524121 ± 0.011871 | 0.677601 ± 0.002403 | 0.325812 ± 0.008173 | 0.324002 ± 0.006732 | 0.496884 ± 0.009311 | 0.668293 ± 0.004692 | 1.264294 ± 0.019221 |
| Rich tree bank (26D) | 0.527357 ± 0.011294 | 0.674715 ± 0.000318 | 0.287529 ± 0.002755 | 0.291640 ± 0.012045 | 0.475793 ± 0.006442 | 0.676885 ± 0.005214 | 1.252472 ± 0.008746 |
| Tree bank + LEACE | 0.536665 ± 0.010436 | 0.681729 ± 0.004078 | 0.349898 ± 0.001542 | 0.357083 ± 0.003590 | 0.500226 ± 0.012284 | 0.684147 ± 0.001832 | 1.287390 ± 0.004448 |
| Fitting prior | 0.543056 ± 0.013563 | 0.693473 ± 0.000409 | 0.480116 ± 0.006864 | 0.626051 ± 0.011224 | 0.553401 ± 0.010565 | 0.692388 ± 0.000413 | 1.283497 ± 0.005963 |

## Validation: every seed

| Seed | Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Original PCA32 | 0.503561 | 0.679332 | 0.306755 | 0.314097 | 0.488816 | 0.650941 | 1.215295 |
| 0 | PCA16 | 0.513083 | 0.680875 | 0.307815 | 0.311123 | 0.492435 | 0.650922 | 1.221820 |
| 0 | PCA32 + LEACE | 0.509199 | 0.684627 | 0.315523 | 0.354080 | 0.508074 | 0.664679 | 1.255602 |
| 0 | C task-only bottleneck (16D) | 0.512090 | 0.675485 | 0.288718 | 0.302383 | 0.485586 | 0.660592 | 1.232791 |
| 0 | D protected bottleneck (16D) | 0.512829 | 0.675579 | 0.291594 | 0.300332 | 0.485553 | 0.670170 | 1.248881 |
| 0 | Rich neural bank (26D) | 0.518947 | 0.673423 | 0.284420 | 0.297523 | 0.478954 | 0.658227 | 1.241035 |
| 0 | Neural bank + LEACE | 0.527689 | 0.675735 | 0.323419 | 0.321948 | 0.499947 | 0.664084 | 1.266142 |
| 0 | Rich tree bank (26D) | 0.528466 | 0.675030 | 0.290585 | 0.305389 | 0.479301 | 0.670867 | 1.247913 |
| 0 | Tree bank + LEACE | 0.539884 | 0.683840 | 0.349465 | 0.358462 | 0.497781 | 0.682637 | 1.282730 |
| 0 | Fitting prior | 0.545019 | 0.693946 | 0.478159 | 0.633326 | 0.548909 | 0.692864 | 1.277010 |
| 1 | Original PCA32 | 0.491088 | 0.678473 | 0.305972 | 0.299588 | 0.486613 | 0.651816 | 1.214711 |
| 1 | PCA16 | 0.501109 | 0.679238 | 0.308903 | 0.295780 | 0.490107 | 0.658419 | 1.208321 |
| 1 | PCA32 + LEACE | 0.505018 | 0.682365 | 0.330471 | 0.347294 | 0.512541 | 0.677728 | 1.238382 |
| 1 | C task-only bottleneck (16D) | 0.504724 | 0.677404 | 0.294048 | 0.290394 | 0.486697 | 0.664639 | 1.225698 |
| 1 | D protected bottleneck (16D) | 0.504752 | 0.677550 | 0.294684 | 0.290544 | 0.485173 | 0.671706 | 1.230229 |
| 1 | Rich neural bank (26D) | 0.512069 | 0.671124 | 0.287258 | 0.282005 | 0.475259 | 0.660915 | 1.245524 |
| 1 | Neural bank + LEACE | 0.510875 | 0.680313 | 0.334914 | 0.331521 | 0.504277 | 0.667441 | 1.282524 |
| 1 | Rich tree bank (26D) | 0.515549 | 0.674723 | 0.285235 | 0.282948 | 0.479719 | 0.680036 | 1.246947 |
| 1 | Tree bank + LEACE | 0.524998 | 0.684318 | 0.351610 | 0.359779 | 0.513549 | 0.683620 | 1.287851 |
| 1 | Fitting prior | 0.528618 | 0.693237 | 0.474443 | 0.631703 | 0.565470 | 0.692158 | 1.288740 |
| 2 | Original PCA32 | 0.505361 | 0.676937 | 0.308208 | 0.300420 | 0.484929 | 0.661652 | 1.196489 |
| 2 | PCA16 | 0.503531 | 0.675767 | 0.302608 | 0.300546 | 0.487267 | 0.663869 | 1.201267 |
| 2 | PCA32 + LEACE | 0.516969 | 0.679147 | 0.318990 | 0.319756 | 0.500888 | 0.678555 | 1.227531 |
| 2 | C task-only bottleneck (16D) | 0.523304 | 0.675312 | 0.302138 | 0.290211 | 0.478206 | 0.669507 | 1.222173 |
| 2 | D protected bottleneck (16D) | 0.530107 | 0.674245 | 0.300797 | 0.288526 | 0.472408 | 0.677168 | 1.237899 |
| 2 | Rich neural bank (26D) | 0.526209 | 0.673222 | 0.291707 | 0.283689 | 0.470606 | 0.662117 | 1.234539 |
| 2 | Neural bank + LEACE | 0.533798 | 0.676756 | 0.319102 | 0.318536 | 0.486428 | 0.673352 | 1.244216 |
| 2 | Rich tree bank (26D) | 0.538056 | 0.674393 | 0.286767 | 0.286584 | 0.468358 | 0.679753 | 1.262555 |
| 2 | Tree bank + LEACE | 0.545113 | 0.677028 | 0.348619 | 0.353008 | 0.489348 | 0.686185 | 1.291590 |
| 2 | Fitting prior | 0.555530 | 0.693237 | 0.487746 | 0.613124 | 0.545825 | 0.692140 | 1.284742 |

## Development evaluation: primary log loss, mean ± sample SD

| Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original PCA32 | 0.487411 ± 0.005595 | 0.687570 ± 0.008575 | 0.298578 ± 0.008781 | 0.288407 ± 0.005005 | 0.504623 ± 0.011550 | 0.655218 ± 0.011326 | 1.200399 ± 0.022578 |
| PCA16 | 0.495062 ± 0.002095 | 0.685527 ± 0.003931 | 0.294408 ± 0.007520 | 0.290035 ± 0.007223 | 0.505704 ± 0.008631 | 0.659948 ± 0.004457 | 1.212077 ± 0.020218 |
| PCA32 + LEACE | 0.496275 ± 0.004211 | 0.691074 ± 0.007655 | 0.312714 ± 0.009209 | 0.322018 ± 0.009292 | 0.521313 ± 0.013445 | 0.678120 ± 0.008021 | 1.235046 ± 0.027316 |
| C task-only bottleneck (16D) | 0.500295 ± 0.003986 | 0.684432 ± 0.005534 | 0.291447 ± 0.010328 | 0.279831 ± 0.007417 | 0.495148 ± 0.012124 | 0.658537 ± 0.006823 | 1.228641 ± 0.023749 |
| D protected bottleneck (16D) | 0.501592 ± 0.002660 | 0.685018 ± 0.004689 | 0.292097 ± 0.011459 | 0.279840 ± 0.007691 | 0.495805 ± 0.015940 | 0.667091 ± 0.008388 | 1.240704 ± 0.017866 |
| Rich neural bank (26D) | 0.510529 ± 0.003504 | 0.681481 ± 0.002771 | 0.278421 ± 0.007137 | 0.270941 ± 0.011699 | 0.487067 ± 0.012910 | 0.664770 ± 0.008175 | 1.243173 ± 0.032479 |
| Neural bank + LEACE | 0.515247 ± 0.005386 | 0.687252 ± 0.002053 | 0.318360 ± 0.002766 | 0.315928 ± 0.017206 | 0.505838 ± 0.008451 | 0.671515 ± 0.007842 | 1.268265 ± 0.035414 |
| Rich tree bank (26D) | 0.517356 ± 0.002162 | 0.681863 ± 0.004760 | 0.279997 ± 0.009591 | 0.274990 ± 0.006996 | 0.488114 ± 0.014003 | 0.676694 ± 0.002856 | 1.259321 ± 0.019889 |
| Tree bank + LEACE | 0.532475 ± 0.005421 | 0.689332 ± 0.003410 | 0.327893 ± 0.006511 | 0.330752 ± 0.012562 | 0.512242 ± 0.008722 | 0.685465 ± 0.002333 | 1.290130 ± 0.014079 |
| Fitting prior | 0.536538 ± 0.005614 | 0.693188 ± 0.000380 | 0.481821 ± 0.006349 | 0.632113 ± 0.006230 | 0.565686 ± 0.009423 | 0.692687 ± 0.000573 | 1.290043 ± 0.025603 |

## Development evaluation: every seed

| Seed | Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Original PCA32 | 0.492403 | 0.693730 | 0.306898 | 0.284755 | 0.517922 | 0.659577 | 1.198246 |
| 0 | PCA16 | 0.496200 | 0.689505 | 0.299625 | 0.286524 | 0.515307 | 0.663449 | 1.213965 |
| 0 | PCA32 + LEACE | 0.492908 | 0.695010 | 0.323202 | 0.331721 | 0.536172 | 0.682553 | 1.232618 |
| 0 | C task-only bottleneck (16D) | 0.495977 | 0.690660 | 0.297596 | 0.272128 | 0.508977 | 0.658094 | 1.217126 |
| 0 | D protected bottleneck (16D) | 0.498737 | 0.690141 | 0.299793 | 0.271794 | 0.513835 | 0.663414 | 1.238740 |
| 0 | Rich neural bank (26D) | 0.509025 | 0.682088 | 0.279399 | 0.263572 | 0.501255 | 0.672497 | 1.246794 |
| 0 | Neural bank + LEACE | 0.520427 | 0.688489 | 0.319184 | 0.296084 | 0.510988 | 0.679537 | 1.276777 |
| 0 | Rich tree bank (26D) | 0.516504 | 0.686544 | 0.286501 | 0.274014 | 0.504268 | 0.679542 | 1.260125 |
| 0 | Tree bank + LEACE | 0.526266 | 0.692341 | 0.325916 | 0.318856 | 0.522272 | 0.688154 | 1.294031 |
| 0 | Fitting prior | 0.530056 | 0.692786 | 0.474987 | 0.631280 | 0.575990 | 0.692902 | 1.287228 |
| 1 | Original PCA32 | 0.481364 | 0.677776 | 0.289400 | 0.286354 | 0.497104 | 0.642360 | 1.223976 |
| 1 | PCA16 | 0.492644 | 0.681644 | 0.285788 | 0.285238 | 0.498593 | 0.654931 | 1.231284 |
| 1 | PCA32 + LEACE | 0.494921 | 0.682252 | 0.305952 | 0.313202 | 0.517779 | 0.668861 | 1.263496 |
| 1 | C task-only bottleneck (16D) | 0.503835 | 0.680077 | 0.279523 | 0.280441 | 0.490124 | 0.651947 | 1.255952 |
| 1 | D protected bottleneck (16D) | 0.502039 | 0.680938 | 0.278927 | 0.280607 | 0.489999 | 0.661170 | 1.259471 |
| 1 | Rich neural bank (26D) | 0.514534 | 0.678456 | 0.270845 | 0.264820 | 0.483938 | 0.656210 | 1.273689 |
| 1 | Neural bank + LEACE | 0.515636 | 0.684882 | 0.315276 | 0.326692 | 0.510442 | 0.663867 | 1.298647 |
| 1 | Rich tree bank (26D) | 0.515749 | 0.677029 | 0.268982 | 0.268533 | 0.480627 | 0.676709 | 1.278795 |
| 1 | Tree bank + LEACE | 0.536267 | 0.685628 | 0.322601 | 0.329510 | 0.508013 | 0.683979 | 1.301846 |
| 1 | Fitting prior | 0.539879 | 0.693238 | 0.482941 | 0.626341 | 0.557509 | 0.692037 | 1.316936 |
| 2 | Original PCA32 | 0.488467 | 0.691204 | 0.299437 | 0.294113 | 0.498845 | 0.663717 | 1.178974 |
| 2 | PCA16 | 0.496342 | 0.685431 | 0.297810 | 0.298343 | 0.503211 | 0.661463 | 1.190981 |
| 2 | PCA32 + LEACE | 0.500997 | 0.695960 | 0.308989 | 0.321130 | 0.509988 | 0.682947 | 1.209025 |
| 2 | C task-only bottleneck (16D) | 0.501072 | 0.682560 | 0.297223 | 0.286924 | 0.486343 | 0.665571 | 1.212843 |
| 2 | D protected bottleneck (16D) | 0.504001 | 0.683976 | 0.297570 | 0.287119 | 0.483582 | 0.676690 | 1.223901 |
| 2 | Rich neural bank (26D) | 0.508027 | 0.683898 | 0.285018 | 0.284431 | 0.476010 | 0.665602 | 1.209035 |
| 2 | Neural bank + LEACE | 0.509677 | 0.688385 | 0.320621 | 0.325008 | 0.496085 | 0.671140 | 1.229371 |
| 2 | Rich tree bank (26D) | 0.519814 | 0.682015 | 0.284508 | 0.282424 | 0.479446 | 0.673830 | 1.239042 |
| 2 | Tree bank + LEACE | 0.534893 | 0.690026 | 0.335164 | 0.343888 | 0.506440 | 0.684262 | 1.274512 |
| 2 | Fitting prior | 0.539678 | 0.693541 | 0.487535 | 0.638718 | 0.563559 | 0.693121 | 1.265963 |

## Original-PCA source and residence references

Each source loss permits at most +.01 nats relative to original PCA32. Residence retains half of the positive original-PCA advantage over the stronger unprotected rich bank. Nonpositive headroom is undefined. These are descriptive dimension-diagnostic references, not all-attribute policy admission checks.

| Seed | Split | Income difference | Work difference | Coverage difference | Source all within margin | Residence retained fraction | Residence half-headroom |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | 0.001060 | -0.002974 | 0.003619 | pass | 0.381145 | fail |
| 0 | development evaluation | -0.007273 | 0.001769 | -0.002614 | pass | 0.771543 | pass |
| 1 | validation | 0.002931 | -0.003808 | 0.003493 | pass | 0.522390 | pass |
| 1 | development evaluation | -0.003612 | -0.001115 | 0.001490 | pass | 0.659917 | pass |
| 2 | validation | -0.005601 | 0.000126 | 0.002338 | pass | 1.087794 | pass |
| 2 | development evaluation | -0.001627 | 0.004231 | 0.004367 | pass | 0.597370 | pass |

## Historical catch-up-inclusive C/D audit sensitivity

These preserve the previous stage’s different candidate set; they do not replace the matched independent primary table. Each historical catch-up trajectory inherited260 representation-fitting passes and added120 attacker-fitting epochs. Fresh independent candidates have no inherited training history, so these are different lifetime budgets.

| Split | Method | SEX independent | SEX historical inclusive | RAC1P independent | RAC1P historical inclusive |
| --- | --- | --- | --- | --- | --- |
| validation | C task-only bottleneck (16D) | 0.664912 ± 0.004464 | 0.664912 ± 0.004464 | 1.226887 ± 0.005408 | 1.225577 ± 0.003344 |
| validation | D protected bottleneck (16D) | 0.673014 ± 0.003678 | 0.672774 ± 0.003827 | 1.239003 ± 0.009375 | 1.229728 ± 0.005403 |
| development evaluation | C task-only bottleneck (16D) | 0.658537 ± 0.006823 | 0.658537 ± 0.006823 | 1.228641 ± 0.023749 | 1.230281 ± 0.022703 |
| development evaluation | D protected bottleneck (16D) | 0.667091 ± 0.008388 | 0.668615 ± 0.007089 | 1.240704 ± 0.017866 | 1.232861 ± 0.023236 |

Race-category coverage remains incomplete. Lower finite-auditor gains are not a privacy guarantee. Every new candidate, weighted sensitivity, and undefined full-schema statistic is preserved.
