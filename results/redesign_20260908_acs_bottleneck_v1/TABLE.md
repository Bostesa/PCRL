# ACS bottleneck development evidence

This report reads saved metrics and selections only. The original test households informed this research direction; they are development evidence. Means ± sample SD describe seeds sharing the same cohort. Natural log loss is primary: lower for utility, higher for less measured attribute recovery. The primary C/D auditor is the saved validation-log-loss winner among five fresh candidates and catch-up; the saved training adversary is diagnostic-only. Reference audit scores are reused from the prior frozen stage.

Completed seeds: 0, 1, 2. [Analysis](ANALYSIS.md), [coverage](SUPPORT.md), [all candidates](PER_TARGET.csv), [all classes](PER_CLASS.csv), [paired scores](PAIRED.csv), [catch-up diagnostics](CATCHUP.csv).

## Validation: primary log loss, mean ± sample SD

| Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original PCA | 0.500004 ± 0.007773 | 0.678247 ± 0.001214 | 0.306979 ± 0.001135 | 0.304701 ± 0.008147 | 0.486786 ± 0.001949 | 0.654803 ± 0.005947 | 1.208832 ± 0.010693 |
| PCA + LEACE | 0.510395 ± 0.006065 | 0.682046 ± 0.002754 | 0.321661 ± 0.007824 | 0.340376 ± 0.018177 | 0.507168 ± 0.005879 | 0.673654 ± 0.007783 | 1.240505 ± 0.014155 |
| Rich neural bank | 0.519075 ± 0.007071 | 0.672589 ± 0.001273 | 0.287795 ± 0.003673 | 0.287739 ± 0.008515 | 0.474939 ± 0.004183 | 0.660420 ± 0.001992 | 1.240366 ± 0.005523 |
| Neural bank + LEACE | 0.524121 ± 0.011871 | 0.677601 ± 0.002403 | 0.325812 ± 0.008173 | 0.324002 ± 0.006732 | 0.496884 ± 0.009311 | 0.668293 ± 0.004692 | 1.264294 ± 0.019221 |
| Rich tree bank | 0.527357 ± 0.011294 | 0.674715 ± 0.000318 | 0.287529 ± 0.002755 | 0.291640 ± 0.012045 | 0.475793 ± 0.006442 | 0.676885 ± 0.005214 | 1.252472 ± 0.008746 |
| Tree bank + LEACE | 0.536665 ± 0.010436 | 0.681729 ± 0.004078 | 0.349898 ± 0.001542 | 0.357083 ± 0.003590 | 0.500226 ± 0.012284 | 0.684147 ± 0.001832 | 1.287390 ± 0.004448 |
| C task-only bottleneck | 0.513373 ± 0.009356 | 0.676067 ± 0.001161 | 0.294968 ± 0.006757 | 0.294330 ± 0.006975 | 0.483496 ± 0.004615 | 0.664912 ± 0.004464 | 1.225577 ± 0.003344 |
| D protected bottleneck | 0.515896 ± 0.012953 | 0.675791 ± 0.001662 | 0.295692 ± 0.004684 | 0.293134 ± 0.006315 | 0.481045 ± 0.007482 | 0.672774 ± 0.003827 | 1.229728 ± 0.005403 |
| Fitting prior | 0.543056 ± 0.013563 | 0.693473 ± 0.000409 | 0.480116 ± 0.006864 | 0.626051 ± 0.011224 | 0.553401 ± 0.010565 | 0.692388 ± 0.000413 | 1.283497 ± 0.005963 |

## Validation: every seed

| Seed | Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Original PCA | 0.503561 | 0.679332 | 0.306755 | 0.314097 | 0.488816 | 0.650941 | 1.215295 |
| 0 | PCA + LEACE | 0.509199 | 0.684627 | 0.315523 | 0.354080 | 0.508074 | 0.664679 | 1.255602 |
| 0 | Rich neural bank | 0.518947 | 0.673423 | 0.284420 | 0.297523 | 0.478954 | 0.658227 | 1.241035 |
| 0 | Neural bank + LEACE | 0.527689 | 0.675735 | 0.323419 | 0.321948 | 0.499947 | 0.664084 | 1.266142 |
| 0 | Rich tree bank | 0.528466 | 0.675030 | 0.290585 | 0.305389 | 0.479301 | 0.670867 | 1.247913 |
| 0 | Tree bank + LEACE | 0.539884 | 0.683840 | 0.349465 | 0.358462 | 0.497781 | 0.682637 | 1.282730 |
| 0 | C task-only bottleneck | 0.512090 | 0.675485 | 0.288718 | 0.302383 | 0.485586 | 0.660592 | 1.228859 |
| 0 | D protected bottleneck | 0.512829 | 0.675579 | 0.291594 | 0.300332 | 0.485553 | 0.670170 | 1.234862 |
| 0 | Fitting prior | 0.545019 | 0.693946 | 0.478159 | 0.633326 | 0.548909 | 0.692864 | 1.277010 |
| 1 | Original PCA | 0.491088 | 0.678473 | 0.305972 | 0.299588 | 0.486613 | 0.651816 | 1.214711 |
| 1 | PCA + LEACE | 0.505018 | 0.682365 | 0.330471 | 0.347294 | 0.512541 | 0.677728 | 1.238382 |
| 1 | Rich neural bank | 0.512069 | 0.671124 | 0.287258 | 0.282005 | 0.475259 | 0.660915 | 1.245524 |
| 1 | Neural bank + LEACE | 0.510875 | 0.680313 | 0.334914 | 0.331521 | 0.504277 | 0.667441 | 1.282524 |
| 1 | Rich tree bank | 0.515549 | 0.674723 | 0.285235 | 0.282948 | 0.479719 | 0.680036 | 1.246947 |
| 1 | Tree bank + LEACE | 0.524998 | 0.684318 | 0.351610 | 0.359779 | 0.513549 | 0.683620 | 1.287851 |
| 1 | C task-only bottleneck | 0.504724 | 0.677404 | 0.294048 | 0.290394 | 0.486697 | 0.664639 | 1.225698 |
| 1 | D protected bottleneck | 0.504752 | 0.677550 | 0.294684 | 0.290544 | 0.485173 | 0.670983 | 1.230229 |
| 1 | Fitting prior | 0.528618 | 0.693237 | 0.474443 | 0.631703 | 0.565470 | 0.692158 | 1.288740 |
| 2 | Original PCA | 0.505361 | 0.676937 | 0.308208 | 0.300420 | 0.484929 | 0.661652 | 1.196489 |
| 2 | PCA + LEACE | 0.516969 | 0.679147 | 0.318990 | 0.319756 | 0.500888 | 0.678555 | 1.227531 |
| 2 | Rich neural bank | 0.526209 | 0.673222 | 0.291707 | 0.283689 | 0.470606 | 0.662117 | 1.234539 |
| 2 | Neural bank + LEACE | 0.533798 | 0.676756 | 0.319102 | 0.318536 | 0.486428 | 0.673352 | 1.244216 |
| 2 | Rich tree bank | 0.538056 | 0.674393 | 0.286767 | 0.286584 | 0.468358 | 0.679753 | 1.262555 |
| 2 | Tree bank + LEACE | 0.545113 | 0.677028 | 0.348619 | 0.353008 | 0.489348 | 0.686185 | 1.291590 |
| 2 | C task-only bottleneck | 0.523304 | 0.675312 | 0.302138 | 0.290211 | 0.478206 | 0.669507 | 1.222173 |
| 2 | D protected bottleneck | 0.530107 | 0.674245 | 0.300797 | 0.288526 | 0.472408 | 0.677168 | 1.224092 |
| 2 | Fitting prior | 0.555530 | 0.693237 | 0.487746 | 0.613124 | 0.545825 | 0.692140 | 1.284742 |

## Validation: PCA-parent criteria

PCA means the unchanged original PCA parent for both learned methods. Source tasks each permit at most +.01 nats. Residence must retain half the original PCA advantage over the stronger unprotected rich bank, if that advantage is positive. Attribute gain must halve relative to original PCA, when positive and adequately supported. Race coverage limits cannot become a pass. A known failure dominates an undefined condition.

| Seed | Method | Source all | Residence retention | SEX half gain | RAC1P half gain | Joint |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | pass | fail | fail | undefined | fail |
| 0 | D protected bottleneck | pass | fail | fail | undefined | fail |
| 1 | C task-only bottleneck | pass | fail | fail | undefined | fail |
| 1 | D protected bottleneck | pass | fail | fail | undefined | fail |
| 2 | C task-only bottleneck | pass | fail | fail | undefined | fail |
| 2 | D protected bottleneck | pass | fail | pass | undefined | fail |

## Development evaluation: primary log loss, mean ± sample SD

| Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original PCA | 0.487411 ± 0.005595 | 0.687570 ± 0.008575 | 0.298578 ± 0.008781 | 0.288407 ± 0.005005 | 0.504623 ± 0.011550 | 0.655218 ± 0.011326 | 1.200399 ± 0.022578 |
| PCA + LEACE | 0.496275 ± 0.004211 | 0.691074 ± 0.007655 | 0.312714 ± 0.009209 | 0.322018 ± 0.009292 | 0.521313 ± 0.013445 | 0.678120 ± 0.008021 | 1.235046 ± 0.027316 |
| Rich neural bank | 0.510529 ± 0.003504 | 0.681481 ± 0.002771 | 0.278421 ± 0.007137 | 0.270941 ± 0.011699 | 0.487067 ± 0.012910 | 0.664770 ± 0.008175 | 1.243173 ± 0.032479 |
| Neural bank + LEACE | 0.515247 ± 0.005386 | 0.687252 ± 0.002053 | 0.318360 ± 0.002766 | 0.315928 ± 0.017206 | 0.505838 ± 0.008451 | 0.671515 ± 0.007842 | 1.268265 ± 0.035414 |
| Rich tree bank | 0.517356 ± 0.002162 | 0.681863 ± 0.004760 | 0.279997 ± 0.009591 | 0.274990 ± 0.006996 | 0.488114 ± 0.014003 | 0.676694 ± 0.002856 | 1.259321 ± 0.019889 |
| Tree bank + LEACE | 0.532475 ± 0.005421 | 0.689332 ± 0.003410 | 0.327893 ± 0.006511 | 0.330752 ± 0.012562 | 0.512242 ± 0.008722 | 0.685465 ± 0.002333 | 1.290130 ± 0.014079 |
| C task-only bottleneck | 0.500295 ± 0.003986 | 0.684432 ± 0.005534 | 0.291447 ± 0.010328 | 0.279831 ± 0.007417 | 0.495148 ± 0.012124 | 0.658537 ± 0.006823 | 1.230281 ± 0.022703 |
| D protected bottleneck | 0.501592 ± 0.002660 | 0.685018 ± 0.004689 | 0.292097 ± 0.011459 | 0.279840 ± 0.007691 | 0.495805 ± 0.015940 | 0.668615 ± 0.007089 | 1.232861 ± 0.023236 |
| Fitting prior | 0.536538 ± 0.005614 | 0.693188 ± 0.000380 | 0.481821 ± 0.006349 | 0.632113 ± 0.006230 | 0.565686 ± 0.009423 | 0.692687 ± 0.000573 | 1.290043 ± 0.025603 |

## Development evaluation: every seed

| Seed | Release | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Original PCA | 0.492403 | 0.693730 | 0.306898 | 0.284755 | 0.517922 | 0.659577 | 1.198246 |
| 0 | PCA + LEACE | 0.492908 | 0.695010 | 0.323202 | 0.331721 | 0.536172 | 0.682553 | 1.232618 |
| 0 | Rich neural bank | 0.509025 | 0.682088 | 0.279399 | 0.263572 | 0.501255 | 0.672497 | 1.246794 |
| 0 | Neural bank + LEACE | 0.520427 | 0.688489 | 0.319184 | 0.296084 | 0.510988 | 0.679537 | 1.276777 |
| 0 | Rich tree bank | 0.516504 | 0.686544 | 0.286501 | 0.274014 | 0.504268 | 0.679542 | 1.260125 |
| 0 | Tree bank + LEACE | 0.526266 | 0.692341 | 0.325916 | 0.318856 | 0.522272 | 0.688154 | 1.294031 |
| 0 | C task-only bottleneck | 0.495977 | 0.690660 | 0.297596 | 0.272128 | 0.508977 | 0.658094 | 1.222048 |
| 0 | D protected bottleneck | 0.498737 | 0.690141 | 0.299793 | 0.271794 | 0.513835 | 0.663414 | 1.222530 |
| 0 | Fitting prior | 0.530056 | 0.692786 | 0.474987 | 0.631280 | 0.575990 | 0.692902 | 1.287228 |
| 1 | Original PCA | 0.481364 | 0.677776 | 0.289400 | 0.286354 | 0.497104 | 0.642360 | 1.223976 |
| 1 | PCA + LEACE | 0.494921 | 0.682252 | 0.305952 | 0.313202 | 0.517779 | 0.668861 | 1.263496 |
| 1 | Rich neural bank | 0.514534 | 0.678456 | 0.270845 | 0.264820 | 0.483938 | 0.656210 | 1.273689 |
| 1 | Neural bank + LEACE | 0.515636 | 0.684882 | 0.315276 | 0.326692 | 0.510442 | 0.663867 | 1.298647 |
| 1 | Rich tree bank | 0.515749 | 0.677029 | 0.268982 | 0.268533 | 0.480627 | 0.676709 | 1.278795 |
| 1 | Tree bank + LEACE | 0.536267 | 0.685628 | 0.322601 | 0.329510 | 0.508013 | 0.683979 | 1.301846 |
| 1 | C task-only bottleneck | 0.503835 | 0.680077 | 0.279523 | 0.280441 | 0.490124 | 0.651947 | 1.255952 |
| 1 | D protected bottleneck | 0.502039 | 0.680938 | 0.278927 | 0.280607 | 0.489999 | 0.665743 | 1.259471 |
| 1 | Fitting prior | 0.539879 | 0.693238 | 0.482941 | 0.626341 | 0.557509 | 0.692037 | 1.316936 |
| 2 | Original PCA | 0.488467 | 0.691204 | 0.299437 | 0.294113 | 0.498845 | 0.663717 | 1.178974 |
| 2 | PCA + LEACE | 0.500997 | 0.695960 | 0.308989 | 0.321130 | 0.509988 | 0.682947 | 1.209025 |
| 2 | Rich neural bank | 0.508027 | 0.683898 | 0.285018 | 0.284431 | 0.476010 | 0.665602 | 1.209035 |
| 2 | Neural bank + LEACE | 0.509677 | 0.688385 | 0.320621 | 0.325008 | 0.496085 | 0.671140 | 1.229371 |
| 2 | Rich tree bank | 0.519814 | 0.682015 | 0.284508 | 0.282424 | 0.479446 | 0.673830 | 1.239042 |
| 2 | Tree bank + LEACE | 0.534893 | 0.690026 | 0.335164 | 0.343888 | 0.506440 | 0.684262 | 1.274512 |
| 2 | C task-only bottleneck | 0.501072 | 0.682560 | 0.297223 | 0.286924 | 0.486343 | 0.665571 | 1.212843 |
| 2 | D protected bottleneck | 0.504001 | 0.683976 | 0.297570 | 0.287119 | 0.483582 | 0.676690 | 1.216583 |
| 2 | Fitting prior | 0.539678 | 0.693541 | 0.487535 | 0.638718 | 0.563559 | 0.693121 | 1.265963 |

## Development evaluation: PCA-parent criteria

PCA means the unchanged original PCA parent for both learned methods. Source tasks each permit at most +.01 nats. Residence must retain half the original PCA advantage over the stronger unprotected rich bank, if that advantage is positive. Attribute gain must halve relative to original PCA, when positive and adequately supported. Race coverage limits cannot become a pass. A known failure dominates an undefined condition.

| Seed | Method | Source all | Residence retention | SEX half gain | RAC1P half gain | Joint |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | pass | pass | fail | undefined | fail |
| 0 | D protected bottleneck | pass | pass | fail | undefined | fail |
| 1 | C task-only bottleneck | pass | fail | fail | undefined | fail |
| 1 | D protected bottleneck | pass | fail | fail | undefined | fail |
| 2 | C task-only bottleneck | pass | fail | fail | undefined | fail |
| 2 | D protected bottleneck | pass | fail | fail | undefined | fail |

## Independent, catch-up, and saved-adversary losses

Primary selection uses validation only, so primary and independent development rankings may differ. These rows expose that difference instead of taking the development minimum.

| Seed | Split | Method | Target | Saved adversary | Fresh independent | Catch-up | Primary | Primary candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | C task-only bottleneck | SEX | 0.689282 | 0.660592 | 0.670226 | 0.660592 | mlp_1 |
| 0 | development evaluation | C task-only bottleneck | SEX | 0.714270 | 0.658094 | 0.690568 | 0.658094 | mlp_1 |
| 0 | validation | C task-only bottleneck | RAC1P | 1.246884 | 1.232791 | 1.228859 | 1.228859 | catchup |
| 0 | development evaluation | C task-only bottleneck | RAC1P | 1.242136 | 1.217126 | 1.222048 | 1.222048 | catchup |
| 0 | validation | D protected bottleneck | SEX | 0.686342 | 0.670170 | 0.671760 | 0.670170 | mlp_0 |
| 0 | development evaluation | D protected bottleneck | SEX | 0.692739 | 0.663414 | 0.674326 | 0.663414 | mlp_0 |
| 0 | validation | D protected bottleneck | RAC1P | 1.240599 | 1.248881 | 1.234862 | 1.234862 | catchup |
| 0 | development evaluation | D protected bottleneck | RAC1P | 1.239753 | 1.238740 | 1.222530 | 1.222530 | catchup |
| 1 | validation | C task-only bottleneck | SEX | 0.697376 | 0.664639 | 0.676066 | 0.664639 | mlp_0 |
| 1 | development evaluation | C task-only bottleneck | SEX | 0.705167 | 0.651947 | 0.676242 | 0.651947 | mlp_0 |
| 1 | validation | C task-only bottleneck | RAC1P | 1.272011 | 1.225698 | 1.254298 | 1.225698 | mlp_1 |
| 1 | development evaluation | C task-only bottleneck | RAC1P | 1.283757 | 1.255952 | 1.274365 | 1.255952 | mlp_1 |
| 1 | validation | D protected bottleneck | SEX | 0.677381 | 0.671706 | 0.670983 | 0.670983 | catchup |
| 1 | development evaluation | D protected bottleneck | SEX | 0.682599 | 0.661170 | 0.665743 | 0.665743 | catchup |
| 1 | validation | D protected bottleneck | RAC1P | 1.262664 | 1.230229 | 1.253591 | 1.230229 | mlp_1 |
| 1 | development evaluation | D protected bottleneck | RAC1P | 1.263552 | 1.259471 | 1.260534 | 1.259471 | mlp_1 |
| 2 | validation | C task-only bottleneck | SEX | 0.726591 | 0.669507 | 0.688539 | 0.669507 | mlp_1 |
| 2 | development evaluation | C task-only bottleneck | SEX | 0.701946 | 0.665571 | 0.689099 | 0.665571 | mlp_1 |
| 2 | validation | C task-only bottleneck | RAC1P | 1.247320 | 1.222173 | 1.230731 | 1.222173 | mlp_0 |
| 2 | development evaluation | C task-only bottleneck | RAC1P | 1.240078 | 1.212843 | 1.216735 | 1.212843 | mlp_0 |
| 2 | validation | D protected bottleneck | SEX | 0.695576 | 0.677168 | 0.680988 | 0.677168 | mlp_1 |
| 2 | development evaluation | D protected bottleneck | SEX | 0.690707 | 0.676690 | 0.682679 | 0.676690 | mlp_1 |
| 2 | validation | D protected bottleneck | RAC1P | 1.240308 | 1.237899 | 1.224092 | 1.224092 | catchup |
| 2 | development evaluation | D protected bottleneck | RAC1P | 1.233541 | 1.223901 | 1.216583 | 1.216583 | catchup |

Full weighted sensitivity, classification metrics, supported/unsupported category scores, and every candidate/restart are preserved in CSV/JSON. No covariance or accuracy privacy certificate is used.
