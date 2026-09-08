# PCA16 paired diagnostic analysis

This diagnostic adds only the fixed PCA16 interface. All older releases, heads, attackers, selections and scores are read unchanged. Primary C/D comparisons use their matched independent attackers; no development minimum is substituted. Sample SDs are descriptive across shared-cohort seeds.

[Main table](TABLE.md), [all paired metrics](PAIRED.csv), [new fitting accounting](FITTING.csv), [new curves](CURVES.csv), [new classes](PER_CLASS.csv), [exact source/residence criteria](criteria.json), [historical coverage and exposed controls](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md).

Positive PCA16-minus-reference loss differences lose utility or reduce measured attribute recovery. Negative differences retain their sign. All five tasks remain reported; no alternative task or attribute threshold is selected from this diagnostic.

## Primary paired log losses

| Split | Reference | Task / attribute | PCA16 − reference |
| --- | --- | --- | --- |
| development evaluation | C task-only bottleneck (16D) | RAC1P | -0.016564 ± 0.011692 |
| validation | C task-only bottleneck (16D) | RAC1P | -0.016418 ± 0.005037 |
| development evaluation | C task-only bottleneck (16D) | SEX | 0.001410 ± 0.004924 |
| validation | C task-only bottleneck (16D) | SEX | -0.007175 ± 0.002179 |
| development evaluation | C task-only bottleneck (16D) | Civilian at work | 0.010204 ± 0.004914 |
| validation | C task-only bottleneck (16D) | Civilian at work | 0.008153 ± 0.002526 |
| development evaluation | C task-only bottleneck (16D) | Commute >20 min | 0.001095 ± 0.002054 |
| validation | C task-only bottleneck (16D) | Commute >20 min | 0.002560 ± 0.002546 |
| development evaluation | C task-only bottleneck (16D) | Income >$50k | 0.002960 ± 0.002951 |
| validation | C task-only bottleneck (16D) | Income >$50k | 0.011474 ± 0.009763 |
| development evaluation | C task-only bottleneck (16D) | Public coverage | 0.010556 ± 0.005570 |
| validation | C task-only bottleneck (16D) | Public coverage | 0.006440 ± 0.002848 |
| development evaluation | C task-only bottleneck (16D) | Same residence | -0.005232 ± 0.005723 |
| validation | C task-only bottleneck (16D) | Same residence | -0.007465 ± 0.010905 |
| development evaluation | D protected bottleneck (16D) | RAC1P | -0.028628 ± 0.004091 |
| validation | D protected bottleneck (16D) | RAC1P | -0.028534 ± 0.007472 |
| development evaluation | D protected bottleneck (16D) | SEX | -0.007143 ± 0.007671 |
| validation | D protected bottleneck (16D) | SEX | -0.015277 ± 0.003438 |
| development evaluation | D protected bottleneck (16D) | Civilian at work | 0.010195 ± 0.005127 |
| validation | D protected bottleneck (16D) | Civilian at work | 0.009349 ± 0.003615 |
| development evaluation | D protected bottleneck (16D) | Commute >20 min | 0.000508 ± 0.001059 |
| validation | D protected bottleneck (16D) | Commute >20 min | 0.002835 ± 0.002132 |
| development evaluation | D protected bottleneck (16D) | Income >$50k | 0.002311 ± 0.003946 |
| validation | D protected bottleneck (16D) | Income >$50k | 0.010750 ± 0.007806 |
| development evaluation | D protected bottleneck (16D) | Public coverage | 0.009899 ± 0.009148 |
| validation | D protected bottleneck (16D) | Public coverage | 0.008891 ± 0.005259 |
| development evaluation | D protected bottleneck (16D) | Same residence | -0.006530 ± 0.003566 |
| validation | D protected bottleneck (16D) | Same residence | -0.009988 ± 0.014497 |
| development evaluation | Original PCA32 | RAC1P | 0.011678 ± 0.004215 |
| validation | Original PCA32 | RAC1P | 0.001637 ± 0.007007 |
| development evaluation | Original PCA32 | SEX | 0.004730 ± 0.007450 |
| validation | Original PCA32 | SEX | 0.002934 ± 0.003368 |
| development evaluation | Original PCA32 | Civilian at work | 0.001628 ± 0.002676 |
| validation | Original PCA32 | Civilian at work | -0.002218 ± 0.002073 |
| development evaluation | Original PCA32 | Commute >20 min | -0.002043 ± 0.005178 |
| validation | Original PCA32 | Commute >20 min | 0.000380 ± 0.001397 |
| development evaluation | Original PCA32 | Income >$50k | -0.004171 ± 0.002864 |
| validation | Original PCA32 | Income >$50k | -0.000537 ± 0.004484 |
| development evaluation | Original PCA32 | Public coverage | 0.001081 ± 0.003509 |
| validation | Original PCA32 | Public coverage | 0.003150 ± 0.000706 |
| development evaluation | Original PCA32 | Same residence | 0.007651 ± 0.003747 |
| validation | Original PCA32 | Same residence | 0.005904 ± 0.006703 |

## Matched-family paired log losses

Within-family choices are the previously saved validation choices. The MLP family comparison uses fresh independent candidates, excluding historical catch-up and saved adversaries.

| Split | Reference | Task / attribute | Family | PCA16 − reference |
| --- | --- | --- | --- | --- |
| development evaluation | C task-only bottleneck (16D) | RAC1P | histgb | -0.005219 ± 0.011303 |
| validation | C task-only bottleneck (16D) | RAC1P | histgb | -0.023009 ± 0.004937 |
| development evaluation | C task-only bottleneck (16D) | RAC1P | logistic | -0.026165 ± 0.011836 |
| validation | C task-only bottleneck (16D) | RAC1P | logistic | -0.029370 ± 0.007910 |
| development evaluation | C task-only bottleneck (16D) | RAC1P | mlp | -0.016564 ± 0.011692 |
| validation | C task-only bottleneck (16D) | RAC1P | mlp | -0.016418 ± 0.005037 |
| development evaluation | C task-only bottleneck (16D) | SEX | histgb | -0.000012 ± 0.007216 |
| validation | C task-only bottleneck (16D) | SEX | histgb | -0.005388 ± 0.006059 |
| development evaluation | C task-only bottleneck (16D) | SEX | logistic | -0.009460 ± 0.010034 |
| validation | C task-only bottleneck (16D) | SEX | logistic | -0.011138 ± 0.009486 |
| development evaluation | C task-only bottleneck (16D) | SEX | mlp | 0.001410 ± 0.004924 |
| validation | C task-only bottleneck (16D) | SEX | mlp | -0.007175 ± 0.002179 |
| development evaluation | C task-only bottleneck (16D) | Civilian at work | logistic | 0.013825 ± 0.002697 |
| validation | C task-only bottleneck (16D) | Civilian at work | logistic | 0.016455 ± 0.003226 |
| development evaluation | C task-only bottleneck (16D) | Civilian at work | mlp | 0.009847 ± 0.005507 |
| validation | C task-only bottleneck (16D) | Civilian at work | mlp | 0.007785 ± 0.003137 |
| development evaluation | C task-only bottleneck (16D) | Commute >20 min | logistic | 0.001137 ± 0.002029 |
| validation | C task-only bottleneck (16D) | Commute >20 min | logistic | 0.001411 ± 0.004883 |
| development evaluation | C task-only bottleneck (16D) | Commute >20 min | mlp | 0.001392 ± 0.002418 |
| validation | C task-only bottleneck (16D) | Commute >20 min | mlp | 0.003953 ± 0.001337 |
| development evaluation | C task-only bottleneck (16D) | Income >$50k | logistic | 0.006965 ± 0.003413 |
| validation | C task-only bottleneck (16D) | Income >$50k | logistic | 0.017403 ± 0.006728 |
| development evaluation | C task-only bottleneck (16D) | Income >$50k | mlp | 0.000576 ± 0.004348 |
| validation | C task-only bottleneck (16D) | Income >$50k | mlp | 0.007569 ± 0.008447 |
| development evaluation | C task-only bottleneck (16D) | Public coverage | logistic | 0.014717 ± 0.002641 |
| validation | C task-only bottleneck (16D) | Public coverage | logistic | 0.012347 ± 0.004228 |
| development evaluation | C task-only bottleneck (16D) | Public coverage | mlp | 0.007014 ± 0.003940 |
| validation | C task-only bottleneck (16D) | Public coverage | mlp | 0.005673 ± 0.004194 |
| development evaluation | C task-only bottleneck (16D) | Same residence | logistic | -0.017454 ± 0.010130 |
| validation | C task-only bottleneck (16D) | Same residence | logistic | -0.021991 ± 0.011467 |
| development evaluation | C task-only bottleneck (16D) | Same residence | mlp | -0.000771 ± 0.003959 |
| validation | C task-only bottleneck (16D) | Same residence | mlp | -0.005413 ± 0.010079 |
| development evaluation | D protected bottleneck (16D) | RAC1P | histgb | -0.017353 ± 0.013861 |
| validation | D protected bottleneck (16D) | RAC1P | histgb | -0.027903 ± 0.015367 |
| development evaluation | D protected bottleneck (16D) | RAC1P | logistic | -0.042512 ± 0.005473 |
| validation | D protected bottleneck (16D) | RAC1P | logistic | -0.043908 ± 0.010530 |
| development evaluation | D protected bottleneck (16D) | RAC1P | mlp | -0.028628 ± 0.004091 |
| validation | D protected bottleneck (16D) | RAC1P | mlp | -0.028534 ± 0.007472 |
| development evaluation | D protected bottleneck (16D) | SEX | histgb | -0.000163 ± 0.003366 |
| validation | D protected bottleneck (16D) | SEX | histgb | -0.007755 ± 0.001225 |
| development evaluation | D protected bottleneck (16D) | SEX | logistic | -0.012845 ± 0.009852 |
| validation | D protected bottleneck (16D) | SEX | logistic | -0.015523 ± 0.006687 |
| development evaluation | D protected bottleneck (16D) | SEX | mlp | -0.007143 ± 0.007671 |
| validation | D protected bottleneck (16D) | SEX | mlp | -0.015277 ± 0.003438 |
| development evaluation | D protected bottleneck (16D) | Civilian at work | logistic | 0.013450 ± 0.002274 |
| validation | D protected bottleneck (16D) | Civilian at work | logistic | 0.016610 ± 0.002958 |
| development evaluation | D protected bottleneck (16D) | Civilian at work | mlp | 0.010669 ± 0.004365 |
| validation | D protected bottleneck (16D) | Civilian at work | mlp | 0.009332 ± 0.003643 |
| development evaluation | D protected bottleneck (16D) | Commute >20 min | logistic | 0.000713 ± 0.001214 |
| validation | D protected bottleneck (16D) | Commute >20 min | logistic | 0.001141 ± 0.004961 |
| development evaluation | D protected bottleneck (16D) | Commute >20 min | mlp | 0.001153 ± 0.002070 |
| validation | D protected bottleneck (16D) | Commute >20 min | mlp | 0.003865 ± 0.001337 |
| development evaluation | D protected bottleneck (16D) | Income >$50k | logistic | 0.006316 ± 0.004515 |
| validation | D protected bottleneck (16D) | Income >$50k | logistic | 0.016679 ± 0.004888 |
| development evaluation | D protected bottleneck (16D) | Income >$50k | mlp | 0.001867 ± 0.003829 |
| validation | D protected bottleneck (16D) | Income >$50k | mlp | 0.008111 ± 0.008157 |
| development evaluation | D protected bottleneck (16D) | Public coverage | logistic | 0.014187 ± 0.004299 |
| validation | D protected bottleneck (16D) | Public coverage | logistic | 0.013400 ± 0.003805 |
| development evaluation | D protected bottleneck (16D) | Public coverage | mlp | 0.007960 ± 0.005880 |
| validation | D protected bottleneck (16D) | Public coverage | mlp | 0.008834 ± 0.006002 |
| development evaluation | D protected bottleneck (16D) | Same residence | logistic | -0.019248 ± 0.010843 |
| validation | D protected bottleneck (16D) | Same residence | logistic | -0.022278 ± 0.012664 |
| development evaluation | D protected bottleneck (16D) | Same residence | mlp | -0.002068 ± 0.001442 |
| validation | D protected bottleneck (16D) | Same residence | mlp | -0.007936 ± 0.013586 |
| development evaluation | Original PCA32 | RAC1P | histgb | -0.030288 ± 0.008075 |
| validation | Original PCA32 | RAC1P | histgb | -0.043869 ± 0.010277 |
| development evaluation | Original PCA32 | RAC1P | logistic | -0.008137 ± 0.004177 |
| validation | Original PCA32 | RAC1P | logistic | -0.017104 ± 0.006743 |
| development evaluation | Original PCA32 | RAC1P | mlp | 0.011678 ± 0.004215 |
| validation | Original PCA32 | RAC1P | mlp | 0.001637 ± 0.007007 |
| development evaluation | Original PCA32 | SEX | histgb | -0.003055 ± 0.008719 |
| validation | Original PCA32 | SEX | histgb | -0.000923 ± 0.005438 |
| development evaluation | Original PCA32 | SEX | logistic | 0.013219 ± 0.005437 |
| validation | Original PCA32 | SEX | logistic | 0.009995 ± 0.001574 |
| development evaluation | Original PCA32 | SEX | mlp | 0.004730 ± 0.007450 |
| validation | Original PCA32 | SEX | mlp | 0.002934 ± 0.003368 |
| development evaluation | Original PCA32 | Civilian at work | logistic | 0.004859 ± 0.001441 |
| validation | Original PCA32 | Civilian at work | logistic | 0.006513 ± 0.002602 |
| development evaluation | Original PCA32 | Civilian at work | mlp | 0.001636 ± 0.002676 |
| validation | Original PCA32 | Civilian at work | mlp | -0.003460 ± 0.003425 |
| development evaluation | Original PCA32 | Commute >20 min | logistic | 0.000393 ± 0.004168 |
| validation | Original PCA32 | Commute >20 min | logistic | 0.000269 ± 0.003083 |
| development evaluation | Original PCA32 | Commute >20 min | mlp | -0.004234 ± 0.004425 |
| validation | Original PCA32 | Commute >20 min | mlp | 0.000556 ± 0.001815 |
| development evaluation | Original PCA32 | Income >$50k | logistic | 0.002253 ± 0.000869 |
| validation | Original PCA32 | Income >$50k | logistic | 0.002659 ± 0.001472 |
| development evaluation | Original PCA32 | Income >$50k | mlp | -0.004171 ± 0.002864 |
| validation | Original PCA32 | Income >$50k | mlp | -0.000537 ± 0.004484 |
| development evaluation | Original PCA32 | Public coverage | logistic | 0.004206 ± 0.001707 |
| validation | Original PCA32 | Public coverage | logistic | 0.006796 ± 0.005077 |
| development evaluation | Original PCA32 | Public coverage | mlp | -0.001150 ± 0.002290 |
| validation | Original PCA32 | Public coverage | mlp | 0.003387 ± 0.000299 |
| development evaluation | Original PCA32 | Same residence | logistic | 0.008144 ± 0.003012 |
| validation | Original PCA32 | Same residence | logistic | 0.004998 ± 0.006128 |
| development evaluation | Original PCA32 | Same residence | mlp | 0.007300 ± 0.003280 |
| validation | Original PCA32 | Same residence | mlp | 0.006402 ± 0.008260 |

## Signed attribute gain and coverage

Gain is fitting-prior log loss minus attack log loss, paired within each seed before aggregation. Negative gains stay negative. Fixed-schema missing support and exposed-control failures limit race interpretation; no all-attribute protection pass is asserted.

| Release | Validation SEX gain | Validation RAC1P gain | Development SEX gain | Development RAC1P gain |
| --- | --- | --- | --- | --- |
| Original PCA32 | 0.037585 ± 0.006196 | 0.074665 ± 0.013280 | 0.037469 ± 0.010753 | 0.089644 ± 0.003041 |
| PCA16 | 0.034651 ± 0.006881 | 0.073028 ± 0.015524 | 0.032739 ± 0.003939 | 0.077966 ± 0.006712 |
| PCA32 + LEACE | 0.018733 ± 0.008196 | 0.042992 ± 0.019004 | 0.014566 ± 0.007457 | 0.054996 ± 0.001780 |
| C task-only bottleneck (16D) | 0.027475 ± 0.004820 | 0.056610 ± 0.010734 | 0.034149 ± 0.006296 | 0.061402 ± 0.008499 |
| D protected bottleneck (16D) | 0.019373 ± 0.003973 | 0.044494 ± 0.015327 | 0.025596 ± 0.007966 | 0.049338 ± 0.007737 |
| Rich neural bank (26D) | 0.031968 ± 0.002391 | 0.043131 ± 0.007114 | 0.027917 ± 0.007718 | 0.046870 ± 0.008823 |
| Neural bank + LEACE | 0.024095 ± 0.005025 | 0.019203 ± 0.018612 | 0.021172 ± 0.007435 | 0.021777 ± 0.013415 |
| Rich tree bank (26D) | 0.015502 ± 0.005626 | 0.031025 ± 0.009944 | 0.015993 ± 0.003021 | 0.030722 ± 0.006426 |
| Tree bank + LEACE | 0.008240 ± 0.002152 | -0.003893 ± 0.004179 | 0.007222 ± 0.002179 | -0.000087 ± 0.013173 |
| Fitting prior | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |

New PCA16 primary coverage checks use actual fitting support and the unchanged exposed-control evidence.

| Seed | Split | Attribute | Primary candidate | Complete | Limitations (Census codes) |
| --- | --- | --- | --- | --- | --- |
| 0 | validation | SEX | mlp_1 | True | {} |
| 0 | validation | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | SEX | mlp_1 | True | {} |
| 0 | development evaluation | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | SEX | mlp_0 | True | {} |
| 1 | validation | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | SEX | mlp_0 | True | {} |
| 1 | development evaluation | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | SEX | mlp_0 | True | {} |
| 2 | validation | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | SEX | mlp_0 | True | {} |
| 2 | development evaluation | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |

## PCA16 metrics and weighted sensitivity

PWGTP sensitivity uses the same predictions and unweighted validation selections. Full-schema balanced accuracy/macro AUROC remain undefined when a class is unsupported; observed-class alternatives are separately named in CSV. These are not official Census estimates.

| Split | Target | Log loss | Accuracy | Balanced accuracy | AUROC / macro |
| --- | --- | --- | --- | --- | --- |
| validation | Same residence | 0.505908 ± 0.006331 | 0.765377 ± 0.006931 | 0.524877 ± 0.003238 | 0.687057 ± 0.020849 |
| validation | Commute >20 min | 0.678627 ± 0.002608 | 0.569964 ± 0.002190 | 0.569832 ± 0.001477 | 0.597043 ± 0.005137 |
| validation | Income >$50k | 0.306442 ± 0.003365 | 0.858328 ± 0.005747 | 0.739622 ± 0.003007 | 0.893809 ± 0.004309 |
| validation | Civilian at work | 0.302483 ± 0.007853 | 0.890460 ± 0.004102 | 0.836909 ± 0.009167 | 0.889179 ± 0.007319 |
| validation | Public coverage | 0.489936 ± 0.002588 | 0.775890 ± 0.003880 | 0.585520 ± 0.011420 | 0.727345 ± 0.015712 |
| validation | SEX | 0.657737 ± 0.006500 | 0.605785 ± 0.010419 | 0.602037 ± 0.012566 | 0.640859 ± 0.019511 |
| validation | RAC1P | 1.210469 ± 0.010444 | 0.594276 ± 0.005458 | undefined (0/3) | undefined (0/3) |
| development evaluation | Same residence | 0.495062 ± 0.002095 | 0.773643 ± 0.005612 | 0.527815 ± 0.009284 | 0.695676 ± 0.013098 |
| development evaluation | Commute >20 min | 0.685527 ± 0.003931 | 0.552990 ± 0.011693 | 0.551746 ± 0.011898 | 0.575694 ± 0.007983 |
| development evaluation | Income >$50k | 0.294408 ± 0.007520 | 0.867742 ± 0.003666 | 0.747240 ± 0.010486 | 0.902085 ± 0.007348 |
| development evaluation | Civilian at work | 0.290035 ± 0.007223 | 0.895477 ± 0.008613 | 0.848779 ± 0.013394 | 0.901041 ± 0.001751 |
| development evaluation | Public coverage | 0.505704 ± 0.008631 | 0.762444 ± 0.011110 | 0.576967 ± 0.014823 | 0.720378 ± 0.003550 |
| development evaluation | SEX | 0.659948 ± 0.004457 | 0.597904 ± 0.005440 | 0.595533 ± 0.003577 | 0.641017 ± 0.003751 |
| development evaluation | RAC1P | 1.212077 ± 0.020218 | 0.586158 ± 0.011053 | undefined (1/3) | undefined (1/3) |
| validation person weighted | Same residence | 0.481576 ± 0.003475 | 0.788534 ± 0.013784 | 0.530411 ± 0.004041 | 0.690486 ± 0.020027 |
| validation person weighted | Commute >20 min | 0.676259 ± 0.002059 | 0.576107 ± 0.008525 | 0.575892 ± 0.007575 | 0.604967 ± 0.009330 |
| validation person weighted | Income >$50k | 0.305116 ± 0.008916 | 0.858658 ± 0.006103 | 0.726225 ± 0.000710 | 0.891569 ± 0.004304 |
| validation person weighted | Civilian at work | 0.299592 ± 0.003126 | 0.895407 ± 0.001444 | 0.838116 ± 0.008471 | 0.883374 ± 0.001962 |
| validation person weighted | Public coverage | 0.501413 ± 0.003969 | 0.765744 ± 0.004968 | 0.574148 ± 0.015527 | 0.716611 ± 0.016197 |
| validation person weighted | SEX | 0.657767 ± 0.011808 | 0.606129 ± 0.014730 | 0.601201 ± 0.016035 | 0.635767 ± 0.027719 |
| validation person weighted | RAC1P | 1.214683 ± 0.011943 | 0.589677 ± 0.002427 | undefined (0/3) | undefined (0/3) |
| development evaluation person weighted | Same residence | 0.474544 ± 0.009318 | 0.794866 ± 0.005974 | 0.530074 ± 0.012305 | 0.689241 ± 0.012124 |
| development evaluation person weighted | Commute >20 min | 0.687659 ± 0.003119 | 0.548268 ± 0.012592 | 0.546486 ± 0.013037 | 0.566763 ± 0.012276 |
| development evaluation person weighted | Income >$50k | 0.304320 ± 0.009139 | 0.863699 ± 0.004219 | 0.732448 ± 0.011290 | 0.894771 ± 0.009310 |
| development evaluation person weighted | Civilian at work | 0.278038 ± 0.010757 | 0.906525 ± 0.009699 | 0.853258 ± 0.017332 | 0.896057 ± 0.000812 |
| development evaluation person weighted | Public coverage | 0.511918 ± 0.010256 | 0.758593 ± 0.015417 | 0.573717 ± 0.010772 | 0.716729 ± 0.007794 |
| development evaluation person weighted | SEX | 0.661744 ± 0.001436 | 0.595757 ± 0.003970 | 0.590388 ± 0.006036 | 0.632089 ± 0.006105 |
| development evaluation person weighted | RAC1P | 1.212406 ± 0.037497 | 0.584039 ± 0.013400 | undefined (1/3) | undefined (1/3) |

No catch-up fitting, eraser, protection strength, new encoder, or task selection is introduced here. The purpose is to measure the fixed dimensionality control under the same scorer and independent audit budget.
