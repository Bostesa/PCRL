# ACS erasure feasibility — saved development evidence

The original test households informed this question; all development columns reuse them. Selections use only the designated validation pool. Natural log loss is primary. Lower task loss is better; higher attack loss means less recovery by this finite auditor set. Means ± sample SD are descriptive across shared-cohort seeds. Undefined SD with one completed seed is retained.

Completed seeds: 0, 1, 2. [Criteria and interpretation](ANALYSIS.md), [support and exposed controls](SUPPORT.md), [all candidates](PER_TARGET.csv), [all classes](PER_CLASS.csv), [paired scores](PAIRED.csv).

## Validation: primary log loss, mean ± sample SD

| Release | Residence | Commute | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- |
| A binary bank | 0.529969 ± 0.009524 | 0.674088 ± 0.001156 | 0.687015 ± 0.004844 | 1.250865 ± 0.003028 |
| A binary bank + LEACE | 0.542760 ± 0.013788 | 0.693340 ± 0.000316 | 0.692199 ± 0.000421 | 1.282822 ± 0.005755 |
| B rich neural bank | 0.519075 ± 0.007071 | 0.672589 ± 0.001273 | 0.660420 ± 0.001992 | 1.240366 ± 0.005523 |
| B rich neural bank + LEACE | 0.524121 ± 0.011871 | 0.677601 ± 0.002403 | 0.668293 ± 0.004692 | 1.264294 ± 0.019221 |
| C rich tree bank | 0.527357 ± 0.011294 | 0.674715 ± 0.000318 | 0.676885 ± 0.005214 | 1.252472 ± 0.008746 |
| C rich tree bank + LEACE | 0.536665 ± 0.010436 | 0.681729 ± 0.004078 | 0.684147 ± 0.001832 | 1.287390 ± 0.004448 |
| D compressed features | 0.507655 ± 0.007399 | 0.672725 ± 0.001254 | 0.656206 ± 0.006586 | 1.221156 ± 0.013415 |
| D compressed features + LEACE | 0.519327 ± 0.010028 | 0.677689 ± 0.003299 | 0.671924 ± 0.007009 | 1.271525 ± 0.007342 |
| E PCA features | 0.500004 ± 0.007773 | 0.678247 ± 0.001214 | 0.654803 ± 0.005947 | 1.208832 ± 0.010693 |
| E PCA features + LEACE | 0.510395 ± 0.006065 | 0.682046 ± 0.002754 | 0.673654 ± 0.007783 | 1.240505 ± 0.014155 |
| F full covariates | 0.500837 ± 0.008018 | 0.680333 ± 0.003601 | 0.657016 ± 0.008847 | 1.204822 ± 0.011114 |
| Fitting prior | 0.543056 ± 0.013563 | 0.693473 ± 0.000409 | 0.692388 ± 0.000413 | 1.283497 ± 0.005963 |

Source tasks use newly fitted, matched downstream heads on each frozen release.

| Release | Income >$50k | Civilian at work | Public coverage |
| --- | --- | --- | --- |
| A binary bank | 0.289275 ± 0.001297 | 0.291139 ± 0.007640 | 0.474439 ± 0.006254 |
| A binary bank + LEACE | 0.479912 ± 0.006918 | 0.626015 ± 0.011256 | 0.552866 ± 0.010412 |
| B rich neural bank | 0.287795 ± 0.003673 | 0.287739 ± 0.008515 | 0.474939 ± 0.004183 |
| B rich neural bank + LEACE | 0.325812 ± 0.008173 | 0.324002 ± 0.006732 | 0.496884 ± 0.009311 |
| C rich tree bank | 0.287529 ± 0.002755 | 0.291640 ± 0.012045 | 0.475793 ± 0.006442 |
| C rich tree bank + LEACE | 0.349898 ± 0.001542 | 0.357083 ± 0.003590 | 0.500226 ± 0.012284 |
| D compressed features | 0.291116 ± 0.001365 | 0.292921 ± 0.007257 | 0.478015 ± 0.004380 |
| D compressed features + LEACE | 0.320595 ± 0.008623 | 0.346264 ± 0.010923 | 0.505705 ± 0.006132 |
| E PCA features | 0.306979 ± 0.001135 | 0.304701 ± 0.008147 | 0.486786 ± 0.001949 |
| E PCA features + LEACE | 0.321661 ± 0.007824 | 0.340376 ± 0.018177 | 0.507168 ± 0.005879 |
| F full covariates | 0.315025 ± 0.005883 | 0.310709 ± 0.005425 | 0.495790 ± 0.004999 |
| Fitting prior | 0.480116 ± 0.006864 | 0.626051 ± 0.011224 | 0.553401 ± 0.010565 |

## Validation: every seed, primary log loss

| Seed | Release | Residence | Commute | Income | Civilian at work | Public coverage | SEX | RAC1P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | A binary bank | 0.531919 | 0.673642 | 0.288134 | 0.299923 | 0.479528 | 0.681624 | 1.250139 |
| 0 | A binary bank + LEACE | 0.544879 | 0.693705 | 0.477900 | 0.633293 | 0.548422 | 0.692639 | 1.276424 |
| 0 | B rich neural bank | 0.518947 | 0.673423 | 0.284420 | 0.297523 | 0.478954 | 0.658227 | 1.241035 |
| 0 | B rich neural bank + LEACE | 0.527689 | 0.675735 | 0.323419 | 0.321948 | 0.499947 | 0.664084 | 1.266142 |
| 0 | C rich tree bank | 0.528466 | 0.675030 | 0.290585 | 0.305389 | 0.479301 | 0.670867 | 1.247913 |
| 0 | C rich tree bank + LEACE | 0.539884 | 0.683840 | 0.349465 | 0.358462 | 0.497781 | 0.682637 | 1.282730 |
| 0 | D compressed features | 0.514308 | 0.672062 | 0.289869 | 0.301140 | 0.483053 | 0.648880 | 1.227929 |
| 0 | D compressed features + LEACE | 0.519853 | 0.673969 | 0.313805 | 0.345016 | 0.504787 | 0.663844 | 1.268831 |
| 0 | E PCA features | 0.503561 | 0.679332 | 0.306755 | 0.314097 | 0.488816 | 0.650941 | 1.215295 |
| 0 | E PCA features + LEACE | 0.509199 | 0.684627 | 0.315523 | 0.354080 | 0.508074 | 0.664679 | 1.255602 |
| 0 | F full covariates | 0.505868 | 0.683750 | 0.314921 | 0.316674 | 0.499913 | 0.647068 | 1.210924 |
| 0 | Fitting prior | 0.545019 | 0.693946 | 0.478159 | 0.633326 | 0.548909 | 0.692864 | 1.277010 |
| 1 | A binary bank | 0.519620 | 0.673221 | 0.290685 | 0.287451 | 0.476331 | 0.691003 | 1.248265 |
| 1 | A binary bank + LEACE | 0.528036 | 0.693160 | 0.474223 | 0.631700 | 0.564762 | 0.692155 | 1.287576 |
| 1 | B rich neural bank | 0.512069 | 0.671124 | 0.287258 | 0.282005 | 0.475259 | 0.660915 | 1.245524 |
| 1 | B rich neural bank + LEACE | 0.510875 | 0.680313 | 0.334914 | 0.331521 | 0.504277 | 0.667441 | 1.282524 |
| 1 | C rich tree bank | 0.515549 | 0.674723 | 0.285235 | 0.282948 | 0.479719 | 0.680036 | 1.246947 |
| 1 | C rich tree bank + LEACE | 0.524998 | 0.684318 | 0.351610 | 0.359779 | 0.513549 | 0.683620 | 1.287851 |
| 1 | D compressed features | 0.499687 | 0.671940 | 0.290904 | 0.287395 | 0.475885 | 0.658102 | 1.229834 |
| 1 | D compressed features + LEACE | 0.509046 | 0.680260 | 0.330298 | 0.357757 | 0.512244 | 0.676361 | 1.279833 |
| 1 | E PCA features | 0.491088 | 0.678473 | 0.305972 | 0.299588 | 0.486613 | 0.651816 | 1.214711 |
| 1 | E PCA features + LEACE | 0.505018 | 0.682365 | 0.330471 | 0.347294 | 0.512541 | 0.677728 | 1.238382 |
| 1 | F full covariates | 0.491591 | 0.676572 | 0.309195 | 0.306068 | 0.497228 | 0.659979 | 1.211549 |
| 1 | Fitting prior | 0.528618 | 0.693237 | 0.474443 | 0.631703 | 0.565470 | 0.692158 | 1.288740 |
| 2 | A binary bank | 0.538367 | 0.675401 | 0.289005 | 0.286041 | 0.467458 | 0.688418 | 1.254190 |
| 2 | A binary bank + LEACE | 0.555365 | 0.693156 | 0.487613 | 0.613050 | 0.545413 | 0.691802 | 1.284466 |
| 2 | B rich neural bank | 0.526209 | 0.673222 | 0.291707 | 0.283689 | 0.470606 | 0.662117 | 1.234539 |
| 2 | B rich neural bank + LEACE | 0.533798 | 0.676756 | 0.319102 | 0.318536 | 0.486428 | 0.673352 | 1.244216 |
| 2 | C rich tree bank | 0.538056 | 0.674393 | 0.286767 | 0.286584 | 0.468358 | 0.679753 | 1.262555 |
| 2 | C rich tree bank + LEACE | 0.545113 | 0.677028 | 0.348619 | 0.353008 | 0.489348 | 0.686185 | 1.291590 |
| 2 | D compressed features | 0.508970 | 0.674171 | 0.292574 | 0.290229 | 0.475108 | 0.661637 | 1.205704 |
| 2 | D compressed features + LEACE | 0.529081 | 0.678838 | 0.317684 | 0.336019 | 0.500084 | 0.675568 | 1.265910 |
| 2 | E PCA features | 0.505361 | 0.676937 | 0.308208 | 0.300420 | 0.484929 | 0.661652 | 1.196489 |
| 2 | E PCA features + LEACE | 0.516969 | 0.679147 | 0.318990 | 0.319756 | 0.500888 | 0.678555 | 1.227531 |
| 2 | F full covariates | 0.505051 | 0.680677 | 0.320959 | 0.309386 | 0.490229 | 0.664001 | 1.191994 |
| 2 | Fitting prior | 0.555530 | 0.693237 | 0.487746 | 0.613124 | 0.545825 | 0.692140 | 1.284742 |

## Validation: paired erased − parent log loss

Positive differences lose task utility but reduce measured attribute recovery. No differences are clipped.

| Parent | Residence | Commute | Income | Civilian at work | Public coverage | SEX | RAC1P |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A binary bank | 0.012791 ± 0.004294 | 0.019252 ± 0.001298 | 0.190637 ± 0.007572 | 0.334876 ± 0.008718 | 0.078426 ± 0.009777 | 0.005184 ± 0.005172 | 0.031957 ± 0.006673 |
| B rich neural bank | 0.005045 ± 0.005435 | 0.005012 ± 0.003669 | 0.038017 ± 0.010166 | 0.036262 ± 0.012606 | 0.021944 ± 0.006649 | 0.007873 ± 0.002931 | 0.023928 ± 0.013700 |
| C rich tree bank | 0.009308 ± 0.002184 | 0.007013 ± 0.003812 | 0.062369 ± 0.003774 | 0.065443 ± 0.011909 | 0.024433 ± 0.008234 | 0.007262 ± 0.004156 | 0.034919 ± 0.005936 |
| D compressed features | 0.011672 ± 0.007553 | 0.004965 ± 0.003217 | 0.029480 ± 0.008606 | 0.053343 ± 0.014770 | 0.027690 ± 0.007680 | 0.015718 ± 0.002260 | 0.050369 ± 0.009657 |
| E PCA features | 0.010392 ± 0.004278 | 0.003799 ± 0.001544 | 0.014682 ± 0.008560 | 0.035675 ± 0.014667 | 0.020382 ± 0.005078 | 0.018851 ± 0.006316 | 0.031674 ± 0.008336 |

## Validation: provisional criteria by seed

Undefined is not a pass. Known failures remain failures when another condition is undefined. Residential retention requires positive headroom over the stronger unprotected rich bank; attribute halving requires positive parent gain and full support/competent exposed control.

| Seed | Parent | Source all ≤+.01 | Residence retain half | SEX halve gain | RAC1P halve gain | Joint |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A binary bank | fail | undefined | pass | undefined | fail |
| 0 | B rich neural bank | fail | undefined | fail | undefined | fail |
| 0 | C rich tree bank | fail | undefined | pass | undefined | fail |
| 0 | D compressed features | fail | fail | fail | undefined | fail |
| 0 | E PCA features | fail | pass | fail | undefined | fail |
| 1 | A binary bank | fail | undefined | pass | undefined | fail |
| 1 | B rich neural bank | fail | undefined | fail | undefined | fail |
| 1 | C rich tree bank | fail | undefined | fail | undefined | fail |
| 1 | D compressed features | fail | fail | pass | undefined | fail |
| 1 | E PCA features | fail | fail | pass | undefined | fail |
| 2 | A binary bank | fail | undefined | pass | undefined | fail |
| 2 | B rich neural bank | fail | undefined | fail | undefined | fail |
| 2 | C rich tree bank | fail | undefined | pass | undefined | fail |
| 2 | D compressed features | fail | fail | fail | undefined | fail |
| 2 | E PCA features | fail | fail | pass | undefined | fail |

## Development evaluation: primary log loss, mean ± sample SD

| Release | Residence | Commute | SEX attack | RAC1P attack |
| --- | --- | --- | --- | --- |
| A binary bank | 0.522428 ± 0.002233 | 0.682161 ± 0.003200 | 0.689054 ± 0.002075 | 1.252065 ± 0.028860 |
| A binary bank + LEACE | 0.536387 ± 0.005802 | 0.693057 ± 0.000218 | 0.692777 ± 0.000772 | 1.289187 ± 0.025087 |
| B rich neural bank | 0.510529 ± 0.003504 | 0.681481 ± 0.002771 | 0.664770 ± 0.008175 | 1.243173 ± 0.032479 |
| B rich neural bank + LEACE | 0.515247 ± 0.005386 | 0.687252 ± 0.002053 | 0.671515 ± 0.007842 | 1.268265 ± 0.035414 |
| C rich tree bank | 0.517356 ± 0.002162 | 0.681863 ± 0.004760 | 0.676694 ± 0.002856 | 1.259321 ± 0.019889 |
| C rich tree bank + LEACE | 0.532475 ± 0.005421 | 0.689332 ± 0.003410 | 0.685465 ± 0.002333 | 1.290130 ± 0.014079 |
| D compressed features | 0.498699 ± 0.007746 | 0.680676 ± 0.001939 | 0.654781 ± 0.005798 | 1.219621 ± 0.035568 |
| D compressed features + LEACE | 0.509683 ± 0.004454 | 0.683998 ± 0.001091 | 0.675361 ± 0.004584 | 1.269058 ± 0.032043 |
| E PCA features | 0.487411 ± 0.005595 | 0.687570 ± 0.008575 | 0.655218 ± 0.011326 | 1.200399 ± 0.022578 |
| E PCA features + LEACE | 0.496275 ± 0.004211 | 0.691074 ± 0.007655 | 0.678120 ± 0.008021 | 1.235046 ± 0.027316 |
| F full covariates | 0.490646 ± 0.006126 | 0.687624 ± 0.009124 | 0.658396 ± 0.011732 | 1.200819 ± 0.026578 |
| Fitting prior | 0.536538 ± 0.005614 | 0.693188 ± 0.000380 | 0.692687 ± 0.000573 | 1.290043 ± 0.025603 |

Source tasks use newly fitted, matched downstream heads on each frozen release.

| Release | Income >$50k | Civilian at work | Public coverage |
| --- | --- | --- | --- |
| A binary bank | 0.279148 ± 0.008083 | 0.277283 ± 0.008673 | 0.487810 ± 0.012493 |
| A binary bank + LEACE | 0.481656 ± 0.006481 | 0.632253 ± 0.006504 | 0.565992 ± 0.009781 |
| B rich neural bank | 0.278421 ± 0.007137 | 0.270941 ± 0.011699 | 0.487067 ± 0.012910 |
| B rich neural bank + LEACE | 0.318360 ± 0.002766 | 0.315928 ± 0.017206 | 0.505838 ± 0.008451 |
| C rich tree bank | 0.279997 ± 0.009591 | 0.274990 ± 0.006996 | 0.488114 ± 0.014003 |
| C rich tree bank + LEACE | 0.327893 ± 0.006511 | 0.330752 ± 0.012562 | 0.512242 ± 0.008722 |
| D compressed features | 0.280999 ± 0.008037 | 0.275957 ± 0.009834 | 0.490342 ± 0.013956 |
| D compressed features + LEACE | 0.315014 ± 0.003041 | 0.326801 ± 0.014926 | 0.515717 ± 0.014644 |
| E PCA features | 0.298578 ± 0.008781 | 0.288407 ± 0.005005 | 0.504623 ± 0.011550 |
| E PCA features + LEACE | 0.312714 ± 0.009209 | 0.322018 ± 0.009292 | 0.521313 ± 0.013445 |
| F full covariates | 0.297187 ± 0.011155 | 0.295826 ± 0.008363 | 0.506998 ± 0.015027 |
| Fitting prior | 0.481821 ± 0.006349 | 0.632113 ± 0.006230 | 0.565686 ± 0.009423 |

## Development evaluation: every seed, primary log loss

| Seed | Release | Residence | Commute | Income | Civilian at work | Public coverage | SEX | RAC1P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | A binary bank | 0.520707 | 0.682579 | 0.281899 | 0.271765 | 0.502027 | 0.686788 | 1.248985 |
| 0 | A binary bank + LEACE | 0.529689 | 0.692807 | 0.474633 | 0.631305 | 0.576620 | 0.692735 | 1.286436 |
| 0 | B rich neural bank | 0.509025 | 0.682088 | 0.279399 | 0.263572 | 0.501255 | 0.672497 | 1.246794 |
| 0 | B rich neural bank + LEACE | 0.520427 | 0.688489 | 0.319184 | 0.296084 | 0.510988 | 0.679537 | 1.276777 |
| 0 | C rich tree bank | 0.516504 | 0.686544 | 0.286501 | 0.274014 | 0.504268 | 0.679542 | 1.260125 |
| 0 | C rich tree bank + LEACE | 0.526266 | 0.692341 | 0.325916 | 0.318856 | 0.522272 | 0.688154 | 1.294031 |
| 0 | D compressed features | 0.507242 | 0.681570 | 0.282783 | 0.265870 | 0.506408 | 0.658079 | 1.227791 |
| 0 | D compressed features + LEACE | 0.508892 | 0.685157 | 0.312531 | 0.311594 | 0.528654 | 0.676775 | 1.257406 |
| 0 | E PCA features | 0.492403 | 0.693730 | 0.306898 | 0.284755 | 0.517922 | 0.659577 | 1.198246 |
| 0 | E PCA features + LEACE | 0.492908 | 0.695010 | 0.323202 | 0.331721 | 0.536172 | 0.682553 | 1.232618 |
| 0 | F full covariates | 0.486863 | 0.696178 | 0.303013 | 0.289228 | 0.524316 | 0.660575 | 1.203703 |
| 0 | Fitting prior | 0.530056 | 0.692786 | 0.474987 | 0.631280 | 0.575990 | 0.692902 | 1.287228 |
| 1 | A binary bank | 0.521626 | 0.678773 | 0.270049 | 0.272805 | 0.482815 | 0.689514 | 1.282342 |
| 1 | A binary bank + LEACE | 0.539602 | 0.693160 | 0.482932 | 0.626275 | 0.557368 | 0.692026 | 1.315536 |
| 1 | B rich neural bank | 0.514534 | 0.678456 | 0.270845 | 0.264820 | 0.483938 | 0.656210 | 1.273689 |
| 1 | B rich neural bank + LEACE | 0.515636 | 0.684882 | 0.315276 | 0.326692 | 0.510442 | 0.663867 | 1.298647 |
| 1 | C rich tree bank | 0.515749 | 0.677029 | 0.268982 | 0.268533 | 0.480627 | 0.676709 | 1.278795 |
| 1 | C rich tree bank + LEACE | 0.536267 | 0.685628 | 0.322601 | 0.329510 | 0.508013 | 0.683979 | 1.301846 |
| 1 | D compressed features | 0.496720 | 0.678452 | 0.272220 | 0.276484 | 0.483403 | 0.648086 | 1.250392 |
| 1 | D compressed features + LEACE | 0.505677 | 0.683847 | 0.314104 | 0.327381 | 0.518679 | 0.670237 | 1.305297 |
| 1 | E PCA features | 0.481364 | 0.677776 | 0.289400 | 0.286354 | 0.497104 | 0.642360 | 1.223976 |
| 1 | E PCA features + LEACE | 0.494921 | 0.682252 | 0.305952 | 0.313202 | 0.517779 | 0.668861 | 1.263496 |
| 1 | F full covariates | 0.487362 | 0.678022 | 0.284326 | 0.293018 | 0.497415 | 0.645727 | 1.225838 |
| 1 | Fitting prior | 0.539879 | 0.693238 | 0.482941 | 0.626341 | 0.557509 | 0.692037 | 1.316936 |
| 2 | A binary bank | 0.524952 | 0.685132 | 0.285496 | 0.287281 | 0.478588 | 0.690860 | 1.224869 |
| 2 | A binary bank + LEACE | 0.539871 | 0.693205 | 0.487404 | 0.639179 | 0.563989 | 0.693570 | 1.265589 |
| 2 | B rich neural bank | 0.508027 | 0.683898 | 0.285018 | 0.284431 | 0.476010 | 0.665602 | 1.209035 |
| 2 | B rich neural bank + LEACE | 0.509677 | 0.688385 | 0.320621 | 0.325008 | 0.496085 | 0.671140 | 1.229371 |
| 2 | C rich tree bank | 0.519814 | 0.682015 | 0.284508 | 0.282424 | 0.479446 | 0.673830 | 1.239042 |
| 2 | C rich tree bank + LEACE | 0.534893 | 0.690026 | 0.335164 | 0.343888 | 0.506440 | 0.684262 | 1.274512 |
| 2 | D compressed features | 0.492134 | 0.682006 | 0.287993 | 0.285517 | 0.481216 | 0.658177 | 1.180678 |
| 2 | D compressed features + LEACE | 0.514479 | 0.682992 | 0.318406 | 0.341428 | 0.499819 | 0.679072 | 1.244472 |
| 2 | E PCA features | 0.488467 | 0.691204 | 0.299437 | 0.294113 | 0.498845 | 0.663717 | 1.178974 |
| 2 | E PCA features + LEACE | 0.500997 | 0.695960 | 0.308989 | 0.321130 | 0.509988 | 0.682947 | 1.209025 |
| 2 | F full covariates | 0.497714 | 0.688671 | 0.304223 | 0.305231 | 0.499262 | 0.668885 | 1.172917 |
| 2 | Fitting prior | 0.539678 | 0.693541 | 0.487535 | 0.638718 | 0.563559 | 0.693121 | 1.265963 |

## Development evaluation: paired erased − parent log loss

Positive differences lose task utility but reduce measured attribute recovery. No differences are clipped.

| Parent | Residence | Commute | Income | Civilian at work | Public coverage | SEX | RAC1P |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A binary bank | 0.013959 ± 0.004573 | 0.010896 ± 0.003210 | 0.202508 ± 0.010088 | 0.354970 ± 0.004036 | 0.078182 ± 0.006252 | 0.003723 ± 0.001929 | 0.037122 ± 0.003774 |
| B rich neural bank | 0.004718 ± 0.005795 | 0.005771 ± 0.001113 | 0.039940 ± 0.004415 | 0.044987 ± 0.015168 | 0.018771 ± 0.008461 | 0.006745 ± 0.001090 | 0.025093 ± 0.004825 |
| C rich tree bank | 0.015120 ± 0.005378 | 0.007469 ± 0.001478 | 0.047896 ± 0.007493 | 0.055761 ± 0.009459 | 0.024128 ± 0.005307 | 0.008771 ± 0.001588 | 0.030809 ± 0.006764 |
| D compressed features | 0.010984 ± 0.010495 | 0.003322 ± 0.002217 | 0.034015 ± 0.006823 | 0.050844 ± 0.005094 | 0.025375 ± 0.008766 | 0.020581 ± 0.001749 | 0.049438 ± 0.017733 |
| E PCA features | 0.008864 ± 0.007258 | 0.003504 ± 0.001931 | 0.014136 ± 0.003972 | 0.033611 ± 0.011567 | 0.016690 ± 0.004954 | 0.022902 ± 0.003636 | 0.034648 ± 0.004741 |

## Development evaluation: provisional criteria by seed

Undefined is not a pass. Known failures remain failures when another condition is undefined. Residential retention requires positive headroom over the stronger unprotected rich bank; attribute halving requires positive parent gain and full support/competent exposed control.

| Seed | Parent | Source all ≤+.01 | Residence retain half | SEX halve gain | RAC1P halve gain | Joint |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A binary bank | fail | undefined | pass | undefined | fail |
| 0 | B rich neural bank | fail | undefined | fail | undefined | fail |
| 0 | C rich tree bank | fail | undefined | pass | undefined | fail |
| 0 | D compressed features | fail | fail | pass | undefined | fail |
| 0 | E PCA features | fail | pass | pass | undefined | fail |
| 1 | A binary bank | fail | undefined | pass | undefined | fail |
| 1 | B rich neural bank | fail | undefined | fail | undefined | fail |
| 1 | C rich tree bank | fail | undefined | fail | undefined | fail |
| 1 | D compressed features | fail | fail | pass | undefined | fail |
| 1 | E PCA features | fail | pass | pass | undefined | fail |
| 2 | A binary bank | fail | undefined | pass | undefined | fail |
| 2 | B rich neural bank | fail | undefined | fail | undefined | fail |
| 2 | C rich tree bank | fail | undefined | pass | undefined | fail |
| 2 | D compressed features | fail | fail | pass | undefined | fail |
| 2 | E PCA features | fail | fail | pass | undefined | fail |

## Stored release properties

Dimensions and ranks are reported on representation-fitting rows. Rank uses the frozen relative covariance threshold; stored dimensions are preserved even where a map becomes constant.

| Seed | Release | Stored dim | Bytes/person | Numerical rank | Spectral effective rank | Float32 calibration max /cov/ |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A binary bank | 3 | 12 | 3 | 2.046048 | — |
| 0 | A binary bank + LEACE | 3 | 12 | 0 | 0.000000 | 0.000000 |
| 0 | B rich neural bank | 26 | 104 | 21 | 2.899903 | — |
| 0 | B rich neural bank + LEACE | 26 | 104 | 12 | 2.817621 | 0.000000 |
| 0 | C rich tree bank | 26 | 104 | 20 | 3.595203 | — |
| 0 | C rich tree bank + LEACE | 26 | 104 | 11 | 3.676746 | 0.000000 |
| 0 | D compressed features | 26 | 104 | 26 | 2.518573 | — |
| 0 | D compressed features + LEACE | 26 | 104 | 17 | 2.361305 | 0.000000 |
| 0 | E PCA features | 32 | 128 | 32 | 12.897874 | — |
| 0 | E PCA features + LEACE | 32 | 128 | 23 | 10.796682 | 0.000000 |
| 0 | F full covariates | 76 | 304 | 51 | 14.210020 | — |
| 1 | A binary bank | 3 | 12 | 3 | 2.002446 | — |
| 1 | A binary bank + LEACE | 3 | 12 | 0 | 0.000000 | 0.000000 |
| 1 | B rich neural bank | 26 | 104 | 21 | 2.856482 | — |
| 1 | B rich neural bank + LEACE | 26 | 104 | 12 | 3.036750 | 0.000000 |
| 1 | C rich tree bank | 26 | 104 | 20 | 3.559483 | — |
| 1 | C rich tree bank + LEACE | 26 | 104 | 11 | 3.826031 | 0.000000 |
| 1 | D compressed features | 26 | 104 | 26 | 2.307402 | — |
| 1 | D compressed features + LEACE | 26 | 104 | 17 | 2.633503 | 0.000000 |
| 1 | E PCA features | 32 | 128 | 32 | 12.979559 | — |
| 1 | E PCA features + LEACE | 32 | 128 | 23 | 10.702970 | 0.000000 |
| 1 | F full covariates | 75 | 300 | 50 | 14.306923 | — |
| 2 | A binary bank | 3 | 12 | 3 | 2.103749 | — |
| 2 | A binary bank + LEACE | 3 | 12 | 0 | 0.000000 | 0.000000 |
| 2 | B rich neural bank | 26 | 104 | 21 | 3.030961 | — |
| 2 | B rich neural bank + LEACE | 26 | 104 | 13 | 3.476742 | 0.000000 |
| 2 | C rich tree bank | 26 | 104 | 20 | 3.629941 | — |
| 2 | C rich tree bank + LEACE | 26 | 104 | 12 | 4.763100 | 0.000000 |
| 2 | D compressed features | 26 | 104 | 26 | 2.665252 | — |
| 2 | D compressed features + LEACE | 26 | 104 | 18 | 3.183893 | 0.000000 |
| 2 | E PCA features | 32 | 128 | 32 | 13.010537 | — |
| 2 | E PCA features + LEACE | 32 | 128 | 24 | 11.458818 | 0.000000 |
| 2 | F full covariates | 77 | 308 | 52 | 14.328403 | — |

Calibration covariance is a fitting-sample statistic. It provides no classification-accuracy or nonlinear-privacy guarantee.
