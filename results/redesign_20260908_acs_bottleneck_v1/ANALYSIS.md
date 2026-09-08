# ACS bottleneck analysis

Saved development evidence only; this report neither fits models nor reselects on development outcomes. C is the matched task-only training ablation; D adds the fixed protection objective. All parent-relative margins use original PCA. The original five-task policy and reserved-task boundary remain fixed. Descriptive seed SDs are not survey uncertainty or a statistical noninferiority test.

[Primary tables](TABLE.md), [support and controls](SUPPORT.md), [all candidates](PER_TARGET.csv), [classes](PER_CLASS.csv), [paired metrics](PAIRED.csv), [training curves](TRAINING_CURVES.csv), [attacker curves](CURVES.csv), [catch-up](CATCHUP.csv), [criteria](criteria.json), [bank comparisons](bank_comparisons.json), [all aggregates](summary.json).

The primary six-candidate audit and five-candidate independent audit are distinct. The saved adversary receives no primary/independent selector. A catch-up audit can lower validation loss and still have higher development loss; the persisted choice is retained. Prior-relative gains remain signed.

## Validation: exact PCA-parent comparisons

| Seed | Method | Audit selector | Source LL differences (income / work / coverage) | Residence headroom before → after | Residence retained | SEX gain before → after | RAC1P gain before → after | Joint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | selected | -0.018037 / -0.011713 / -0.003230 | 0.015386 → 0.006857 | 0.445678 | 0.041923 → 0.032272 | 0.061715 → 0.048151 | fail |
| 0 | D protected bottleneck | selected | -0.015162 / -0.013765 / -0.003263 | 0.015386 → 0.006118 | 0.397639 | 0.041923 → 0.022695 | 0.061715 → 0.042147 | fail |
| 0 | C task-only bottleneck | independent_selected | -0.018037 / -0.011713 / -0.003230 | 0.015386 → 0.006857 | 0.445678 | 0.041923 → 0.032272 | 0.061715 → 0.044218 | fail |
| 0 | D protected bottleneck | independent_selected | -0.015162 / -0.013765 / -0.003263 | 0.015386 → 0.006118 | 0.397639 | 0.041923 → 0.022695 | 0.061715 → 0.028129 | fail |
| 1 | C task-only bottleneck | selected | -0.011925 / -0.009193 / 0.000084 | 0.020981 → 0.007345 | 0.350081 | 0.040342 → 0.027520 | 0.074029 → 0.063042 | fail |
| 1 | D protected bottleneck | selected | -0.011288 / -0.009044 / -0.001440 | 0.020981 → 0.007318 | 0.348770 | 0.040342 → 0.021175 | 0.074029 → 0.058511 | fail |
| 1 | C task-only bottleneck | independent_selected | -0.011925 / -0.009193 / 0.000084 | 0.020981 → 0.007345 | 0.350081 | 0.040342 → 0.027520 | 0.074029 → 0.063042 | fail |
| 1 | D protected bottleneck | independent_selected | -0.011288 / -0.009044 / -0.001440 | 0.020981 → 0.007318 | 0.348770 | 0.040342 → 0.020453 | 0.074029 → 0.058511 | fail |
| 2 | C task-only bottleneck | selected | -0.006071 / -0.010208 / -0.006722 | 0.020847 → 0.002905 | 0.139335 | 0.030488 → 0.022633 | 0.088252 → 0.062569 | fail |
| 2 | D protected bottleneck | selected | -0.007411 / -0.011894 / -0.012520 | 0.020847 → -0.003898 | -0.186992 | 0.030488 → 0.014972 | 0.088252 → 0.060650 | fail |
| 2 | C task-only bottleneck | independent_selected | -0.006071 / -0.010208 / -0.006722 | 0.020847 → 0.002905 | 0.139335 | 0.030488 → 0.022633 | 0.088252 → 0.062569 | fail |
| 2 | D protected bottleneck | independent_selected | -0.007411 / -0.011894 / -0.012520 | 0.020847 → -0.003898 | -0.186992 | 0.030488 → 0.014972 | 0.088252 → 0.046843 | fail |

## Validation: rich-bank margins by seed

Utility requires .01 lower residence loss. Each attribute permits at most .005 extra signed gain. Numeric columns publish the raw inequalities even when coverage prevents policy assessment.

| Seed | Method | Bank | Selector | Residence LL difference | Utility | SEX numeric | RAC1P numeric | Numeric joint | Policy joint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | Rich neural bank | selected | -0.006857 | fail | pass | fail | fail | fail |
| 0 | C task-only bottleneck | Neural bank + LEACE | selected | -0.015599 | pass | pass | fail | fail | undefined |
| 0 | C task-only bottleneck | Rich tree bank | selected | -0.016376 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Tree bank + LEACE | selected | -0.027794 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Rich neural bank | selected | -0.006118 | fail | pass | fail | fail | fail |
| 0 | D protected bottleneck | Neural bank + LEACE | selected | -0.014860 | pass | pass | fail | fail | undefined |
| 0 | D protected bottleneck | Rich tree bank | selected | -0.015637 | pass | pass | fail | fail | undefined |
| 0 | D protected bottleneck | Tree bank + LEACE | selected | -0.027055 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Rich neural bank | independent_selected | -0.006857 | fail | pass | fail | fail | fail |
| 0 | C task-only bottleneck | Neural bank + LEACE | independent_selected | -0.015599 | pass | pass | fail | fail | undefined |
| 0 | C task-only bottleneck | Rich tree bank | independent_selected | -0.016376 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Tree bank + LEACE | independent_selected | -0.027794 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Rich neural bank | independent_selected | -0.006118 | fail | pass | pass | fail | fail |
| 0 | D protected bottleneck | Neural bank + LEACE | independent_selected | -0.014860 | pass | pass | fail | fail | undefined |
| 0 | D protected bottleneck | Rich tree bank | independent_selected | -0.015637 | pass | pass | pass | pass | undefined |
| 0 | D protected bottleneck | Tree bank + LEACE | independent_selected | -0.027055 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Rich neural bank | selected | -0.007345 | fail | pass | fail | fail | fail |
| 1 | C task-only bottleneck | Neural bank + LEACE | selected | -0.006151 | fail | pass | fail | fail | fail |
| 1 | C task-only bottleneck | Rich tree bank | selected | -0.010824 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Tree bank + LEACE | selected | -0.020274 | pass | fail | fail | fail | fail |
| 1 | D protected bottleneck | Rich neural bank | selected | -0.007318 | fail | pass | fail | fail | fail |
| 1 | D protected bottleneck | Neural bank + LEACE | selected | -0.006123 | fail | pass | fail | fail | fail |
| 1 | D protected bottleneck | Rich tree bank | selected | -0.010797 | pass | fail | fail | fail | fail |
| 1 | D protected bottleneck | Tree bank + LEACE | selected | -0.020246 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Rich neural bank | independent_selected | -0.007345 | fail | pass | fail | fail | fail |
| 1 | C task-only bottleneck | Neural bank + LEACE | independent_selected | -0.006151 | fail | pass | fail | fail | fail |
| 1 | C task-only bottleneck | Rich tree bank | independent_selected | -0.010824 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Tree bank + LEACE | independent_selected | -0.020274 | pass | fail | fail | fail | fail |
| 1 | D protected bottleneck | Rich neural bank | independent_selected | -0.007318 | fail | pass | fail | fail | fail |
| 1 | D protected bottleneck | Neural bank + LEACE | independent_selected | -0.006123 | fail | pass | fail | fail | fail |
| 1 | D protected bottleneck | Rich tree bank | independent_selected | -0.010797 | pass | fail | fail | fail | fail |
| 1 | D protected bottleneck | Tree bank + LEACE | independent_selected | -0.020246 | pass | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Rich neural bank | selected | -0.002905 | fail | pass | fail | fail | fail |
| 2 | C task-only bottleneck | Neural bank + LEACE | selected | -0.010494 | pass | pass | fail | fail | undefined |
| 2 | C task-only bottleneck | Rich tree bank | selected | -0.014752 | pass | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Tree bank + LEACE | selected | -0.021809 | pass | fail | fail | fail | fail |
| 2 | D protected bottleneck | Rich neural bank | selected | 0.003898 | fail | pass | fail | fail | fail |
| 2 | D protected bottleneck | Neural bank + LEACE | selected | -0.003691 | fail | pass | fail | fail | fail |
| 2 | D protected bottleneck | Rich tree bank | selected | -0.007949 | fail | pass | fail | fail | fail |
| 2 | D protected bottleneck | Tree bank + LEACE | selected | -0.015005 | pass | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Rich neural bank | independent_selected | -0.002905 | fail | pass | fail | fail | fail |
| 2 | C task-only bottleneck | Neural bank + LEACE | independent_selected | -0.010494 | pass | pass | fail | fail | undefined |
| 2 | C task-only bottleneck | Rich tree bank | independent_selected | -0.014752 | pass | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Tree bank + LEACE | independent_selected | -0.021809 | pass | fail | fail | fail | fail |
| 2 | D protected bottleneck | Rich neural bank | independent_selected | 0.003898 | fail | pass | pass | fail | fail |
| 2 | D protected bottleneck | Neural bank + LEACE | independent_selected | -0.003691 | fail | pass | fail | fail | fail |
| 2 | D protected bottleneck | Rich tree bank | independent_selected | -0.007949 | fail | pass | fail | fail | fail |
| 2 | D protected bottleneck | Tree bank + LEACE | independent_selected | -0.015005 | pass | fail | fail | fail | fail |

## Validation: signed gains and independent audit sensitivity

| Release | Selector | SEX gain | RAC1P gain | SEX attack LL | RAC1P attack LL |
| --- | --- | --- | --- | --- | --- |
| Original PCA | selected | 0.037585 ± 0.006196 | 0.074665 ± 0.013280 | 0.654803 ± 0.005947 | 1.208832 ± 0.010693 |
| Original PCA | independent_selected | 0.037585 ± 0.006196 | 0.074665 ± 0.013280 | 0.654803 ± 0.005947 | 1.208832 ± 0.010693 |
| PCA + LEACE | selected | 0.018733 ± 0.008196 | 0.042992 ± 0.019004 | 0.673654 ± 0.007783 | 1.240505 ± 0.014155 |
| PCA + LEACE | independent_selected | 0.018733 ± 0.008196 | 0.042992 ± 0.019004 | 0.673654 ± 0.007783 | 1.240505 ± 0.014155 |
| Rich neural bank | selected | 0.031968 ± 0.002391 | 0.043131 ± 0.007114 | 0.660420 ± 0.001992 | 1.240366 ± 0.005523 |
| Rich neural bank | independent_selected | 0.031968 ± 0.002391 | 0.043131 ± 0.007114 | 0.660420 ± 0.001992 | 1.240366 ± 0.005523 |
| Neural bank + LEACE | selected | 0.024095 ± 0.005025 | 0.019203 ± 0.018612 | 0.668293 ± 0.004692 | 1.264294 ± 0.019221 |
| Neural bank + LEACE | independent_selected | 0.024095 ± 0.005025 | 0.019203 ± 0.018612 | 0.668293 ± 0.004692 | 1.264294 ± 0.019221 |
| Rich tree bank | selected | 0.015502 ± 0.005626 | 0.031025 ± 0.009944 | 0.676885 ± 0.005214 | 1.252472 ± 0.008746 |
| Rich tree bank | independent_selected | 0.015502 ± 0.005626 | 0.031025 ± 0.009944 | 0.676885 ± 0.005214 | 1.252472 ± 0.008746 |
| Tree bank + LEACE | selected | 0.008240 ± 0.002152 | -0.003893 ± 0.004179 | 0.684147 ± 0.001832 | 1.287390 ± 0.004448 |
| Tree bank + LEACE | independent_selected | 0.008240 ± 0.002152 | -0.003893 ± 0.004179 | 0.684147 ± 0.001832 | 1.287390 ± 0.004448 |
| C task-only bottleneck | selected | 0.027475 ± 0.004820 | 0.057920 ± 0.008464 | 0.664912 ± 0.004464 | 1.225577 ± 0.003344 |
| C task-only bottleneck | independent_selected | 0.027475 ± 0.004820 | 0.056610 ± 0.010734 | 0.664912 ± 0.004464 | 1.226887 ± 0.005408 |
| D protected bottleneck | selected | 0.019614 ± 0.004091 | 0.053769 ± 0.010122 | 0.672774 ± 0.003827 | 1.229728 ± 0.005403 |
| D protected bottleneck | independent_selected | 0.019373 ± 0.003973 | 0.044494 ± 0.015327 | 0.673014 ± 0.003678 | 1.239003 ± 0.009375 |
| Fitting prior | selected | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.692388 ± 0.000413 | 1.283497 ± 0.005963 |
| Fitting prior | independent_selected | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.692388 ± 0.000413 | 1.283497 ± 0.005963 |

## Development evaluation: exact PCA-parent comparisons

| Seed | Method | Audit selector | Source LL differences (income / work / coverage) | Residence headroom before → after | Residence retained | SEX gain before → after | RAC1P gain before → after | Joint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | selected | -0.009302 / -0.012627 / -0.008945 | 0.016622 → 0.013048 | 0.784964 | 0.033326 → 0.034809 | 0.088982 → 0.065180 | fail |
| 0 | D protected bottleneck | selected | -0.007105 / -0.012961 / -0.004087 | 0.016622 → 0.010288 | 0.618956 | 0.033326 → 0.029489 | 0.088982 → 0.064698 | fail |
| 0 | C task-only bottleneck | independent_selected | -0.009302 / -0.012627 / -0.008945 | 0.016622 → 0.013048 | 0.784964 | 0.033326 → 0.034809 | 0.088982 → 0.070102 | fail |
| 0 | D protected bottleneck | independent_selected | -0.007105 / -0.012961 / -0.004087 | 0.016622 → 0.010288 | 0.618956 | 0.033326 → 0.029489 | 0.088982 → 0.048488 | fail |
| 1 | C task-only bottleneck | selected | -0.009876 / -0.005913 / -0.006980 | 0.033170 → 0.010699 | 0.322558 | 0.049677 → 0.040090 | 0.092961 → 0.060984 | fail |
| 1 | D protected bottleneck | selected | -0.010473 / -0.005747 / -0.007105 | 0.033170 → 0.012494 | 0.376680 | 0.049677 → 0.026294 | 0.092961 → 0.057465 | fail |
| 1 | C task-only bottleneck | independent_selected | -0.009876 / -0.005913 / -0.006980 | 0.033170 → 0.010699 | 0.322558 | 0.049677 → 0.040090 | 0.092961 → 0.060984 | fail |
| 1 | D protected bottleneck | independent_selected | -0.010473 / -0.005747 / -0.007105 | 0.033170 → 0.012494 | 0.376680 | 0.049677 → 0.030867 | 0.092961 → 0.057465 | fail |
| 2 | C task-only bottleneck | selected | -0.002215 / -0.007189 / -0.012501 | 0.019560 → 0.006955 | 0.355572 | 0.029404 → 0.027550 | 0.086989 → 0.053120 | fail |
| 2 | D protected bottleneck | selected | -0.001867 / -0.006994 / -0.015262 | 0.019560 → 0.004026 | 0.205828 | 0.029404 → 0.016431 | 0.086989 → 0.049380 | fail |
| 2 | C task-only bottleneck | independent_selected | -0.002215 / -0.007189 / -0.012501 | 0.019560 → 0.006955 | 0.355572 | 0.029404 → 0.027550 | 0.086989 → 0.053120 | fail |
| 2 | D protected bottleneck | independent_selected | -0.001867 / -0.006994 / -0.015262 | 0.019560 → 0.004026 | 0.205828 | 0.029404 → 0.016431 | 0.086989 → 0.042062 | fail |

## Development evaluation: rich-bank margins by seed

Utility requires .01 lower residence loss. Each attribute permits at most .005 extra signed gain. Numeric columns publish the raw inequalities even when coverage prevents policy assessment.

| Seed | Method | Bank | Selector | Residence LL difference | Utility | SEX numeric | RAC1P numeric | Numeric joint | Policy joint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | Rich neural bank | selected | -0.013048 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Neural bank + LEACE | selected | -0.024450 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Rich tree bank | selected | -0.020527 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Tree bank + LEACE | selected | -0.030288 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Rich neural bank | selected | -0.010288 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Neural bank + LEACE | selected | -0.021691 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Rich tree bank | selected | -0.017767 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Tree bank + LEACE | selected | -0.027529 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Rich neural bank | independent_selected | -0.013048 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Neural bank + LEACE | independent_selected | -0.024450 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Rich tree bank | independent_selected | -0.020527 | pass | fail | fail | fail | fail |
| 0 | C task-only bottleneck | Tree bank + LEACE | independent_selected | -0.030288 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Rich neural bank | independent_selected | -0.010288 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Neural bank + LEACE | independent_selected | -0.021691 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Rich tree bank | independent_selected | -0.017767 | pass | fail | fail | fail | fail |
| 0 | D protected bottleneck | Tree bank + LEACE | independent_selected | -0.027529 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Rich neural bank | selected | -0.010699 | pass | pass | fail | fail | undefined |
| 1 | C task-only bottleneck | Neural bank + LEACE | selected | -0.011802 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Rich tree bank | selected | -0.011915 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Tree bank + LEACE | selected | -0.032433 | pass | fail | fail | fail | fail |
| 1 | D protected bottleneck | Rich neural bank | selected | -0.012494 | pass | pass | fail | fail | undefined |
| 1 | D protected bottleneck | Neural bank + LEACE | selected | -0.013597 | pass | pass | fail | fail | undefined |
| 1 | D protected bottleneck | Rich tree bank | selected | -0.013710 | pass | fail | fail | fail | fail |
| 1 | D protected bottleneck | Tree bank + LEACE | selected | -0.034228 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Rich neural bank | independent_selected | -0.010699 | pass | pass | fail | fail | undefined |
| 1 | C task-only bottleneck | Neural bank + LEACE | independent_selected | -0.011802 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Rich tree bank | independent_selected | -0.011915 | pass | fail | fail | fail | fail |
| 1 | C task-only bottleneck | Tree bank + LEACE | independent_selected | -0.032433 | pass | fail | fail | fail | fail |
| 1 | D protected bottleneck | Rich neural bank | independent_selected | -0.012494 | pass | pass | fail | fail | undefined |
| 1 | D protected bottleneck | Neural bank + LEACE | independent_selected | -0.013597 | pass | pass | fail | fail | undefined |
| 1 | D protected bottleneck | Rich tree bank | independent_selected | -0.013710 | pass | fail | fail | fail | fail |
| 1 | D protected bottleneck | Tree bank + LEACE | independent_selected | -0.034228 | pass | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Rich neural bank | selected | -0.006955 | fail | pass | pass | fail | fail |
| 2 | C task-only bottleneck | Neural bank + LEACE | selected | -0.008605 | fail | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Rich tree bank | selected | -0.018742 | pass | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Tree bank + LEACE | selected | -0.033821 | pass | fail | fail | fail | fail |
| 2 | D protected bottleneck | Rich neural bank | selected | -0.004026 | fail | pass | pass | fail | fail |
| 2 | D protected bottleneck | Neural bank + LEACE | selected | -0.005676 | fail | pass | fail | fail | fail |
| 2 | D protected bottleneck | Rich tree bank | selected | -0.015813 | pass | pass | fail | fail | undefined |
| 2 | D protected bottleneck | Tree bank + LEACE | selected | -0.030892 | pass | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Rich neural bank | independent_selected | -0.006955 | fail | pass | pass | fail | fail |
| 2 | C task-only bottleneck | Neural bank + LEACE | independent_selected | -0.008605 | fail | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Rich tree bank | independent_selected | -0.018742 | pass | fail | fail | fail | fail |
| 2 | C task-only bottleneck | Tree bank + LEACE | independent_selected | -0.033821 | pass | fail | fail | fail | fail |
| 2 | D protected bottleneck | Rich neural bank | independent_selected | -0.004026 | fail | pass | pass | fail | fail |
| 2 | D protected bottleneck | Neural bank + LEACE | independent_selected | -0.005676 | fail | pass | fail | fail | fail |
| 2 | D protected bottleneck | Rich tree bank | independent_selected | -0.015813 | pass | pass | fail | fail | undefined |
| 2 | D protected bottleneck | Tree bank + LEACE | independent_selected | -0.030892 | pass | fail | fail | fail | fail |

## Development evaluation: signed gains and independent audit sensitivity

| Release | Selector | SEX gain | RAC1P gain | SEX attack LL | RAC1P attack LL |
| --- | --- | --- | --- | --- | --- |
| Original PCA | selected | 0.037469 ± 0.010753 | 0.089644 ± 0.003041 | 0.655218 ± 0.011326 | 1.200399 ± 0.022578 |
| Original PCA | independent_selected | 0.037469 ± 0.010753 | 0.089644 ± 0.003041 | 0.655218 ± 0.011326 | 1.200399 ± 0.022578 |
| PCA + LEACE | selected | 0.014566 ± 0.007457 | 0.054996 ± 0.001780 | 0.678120 ± 0.008021 | 1.235046 ± 0.027316 |
| PCA + LEACE | independent_selected | 0.014566 ± 0.007457 | 0.054996 ± 0.001780 | 0.678120 ± 0.008021 | 1.235046 ± 0.027316 |
| Rich neural bank | selected | 0.027917 ± 0.007718 | 0.046870 ± 0.008823 | 0.664770 ± 0.008175 | 1.243173 ± 0.032479 |
| Rich neural bank | independent_selected | 0.027917 ± 0.007718 | 0.046870 ± 0.008823 | 0.664770 ± 0.008175 | 1.243173 ± 0.032479 |
| Neural bank + LEACE | selected | 0.021172 ± 0.007435 | 0.021777 ± 0.013415 | 0.671515 ± 0.007842 | 1.268265 ± 0.035414 |
| Neural bank + LEACE | independent_selected | 0.021172 ± 0.007435 | 0.021777 ± 0.013415 | 0.671515 ± 0.007842 | 1.268265 ± 0.035414 |
| Rich tree bank | selected | 0.015993 ± 0.003021 | 0.030722 ± 0.006426 | 0.676694 ± 0.002856 | 1.259321 ± 0.019889 |
| Rich tree bank | independent_selected | 0.015993 ± 0.003021 | 0.030722 ± 0.006426 | 0.676694 ± 0.002856 | 1.259321 ± 0.019889 |
| Tree bank + LEACE | selected | 0.007222 ± 0.002179 | -0.000087 ± 0.013173 | 0.685465 ± 0.002333 | 1.290130 ± 0.014079 |
| Tree bank + LEACE | independent_selected | 0.007222 ± 0.002179 | -0.000087 ± 0.013173 | 0.685465 ± 0.002333 | 1.290130 ± 0.014079 |
| C task-only bottleneck | selected | 0.034149 ± 0.006296 | 0.059761 ± 0.006122 | 0.658537 ± 0.006823 | 1.230281 ± 0.022703 |
| C task-only bottleneck | independent_selected | 0.034149 ± 0.006296 | 0.061402 ± 0.008499 | 0.658537 ± 0.006823 | 1.228641 ± 0.023749 |
| D protected bottleneck | selected | 0.024071 ± 0.006807 | 0.057181 ± 0.007663 | 0.668615 ± 0.007089 | 1.232861 ± 0.023236 |
| D protected bottleneck | independent_selected | 0.025596 ± 0.007966 | 0.049338 ± 0.007737 | 0.667091 ± 0.008388 | 1.240704 ± 0.017866 |
| Fitting prior | selected | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.692687 ± 0.000573 | 1.290043 ± 0.025603 |
| Fitting prior | independent_selected | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.692687 ± 0.000573 | 1.290043 ± 0.025603 |

## Paired D − C: every task and attribute

Positive loss differences reduce task utility and reduce measured attribute recovery. Within-family choices are saved validation selections. Catch-up and training-adversary pairs are diagnostics.

| Split | Role | Target | Selector | D − C log loss |
| --- | --- | --- | --- | --- |
| development evaluation | audit | RAC1P | auc_selected | undefined (0/3) |
| validation | audit | RAC1P | auc_selected | undefined (0/3) |
| development evaluation | audit | RAC1P | candidate:catchup | -0.004501 ± 0.008087 |
| validation | audit | RAC1P | candidate:catchup | -0.000447 ± 0.006325 |
| development evaluation | audit | RAC1P | candidate:saved_adversary | -0.009708 ± 0.009325 |
| validation | audit | RAC1P | candidate:saved_adversary | -0.007548 ± 0.001600 |
| development evaluation | audit | RAC1P | family:histgb | 0.012133 ± 0.006427 |
| validation | audit | RAC1P | family:histgb | 0.004894 ± 0.011649 |
| development evaluation | audit | RAC1P | family:logistic | 0.016347 ± 0.009349 |
| validation | audit | RAC1P | family:logistic | 0.014538 ± 0.003086 |
| development evaluation | audit | RAC1P | family:mlp | 0.012064 ± 0.009089 |
| validation | audit | RAC1P | family:mlp | 0.012115 ± 0.006571 |
| development evaluation | audit | RAC1P | independent_selected | 0.012064 ± 0.009089 |
| validation | audit | RAC1P | independent_selected | 0.012115 ± 0.006571 |
| development evaluation | audit | RAC1P | selected | 0.002580 ± 0.001821 |
| validation | audit | RAC1P | selected | 0.004151 ± 0.002069 |
| development evaluation | audit | SEX | auc_selected | -0.011054 ± 0.004935 |
| validation | audit | SEX | auc_selected | -0.003700 ± 0.004698 |
| development evaluation | audit | SEX | candidate:catchup | -0.011054 ± 0.004935 |
| validation | audit | SEX | candidate:catchup | -0.003700 ± 0.004698 |
| development evaluation | audit | SEX | candidate:saved_adversary | -0.018446 ± 0.006262 |
| validation | audit | SEX | candidate:saved_adversary | -0.017983 ± 0.014145 |
| development evaluation | audit | SEX | family:histgb | 0.000150 ± 0.005884 |
| validation | audit | SEX | family:histgb | 0.002366 ± 0.006563 |
| development evaluation | audit | SEX | family:logistic | 0.003385 ± 0.002039 |
| validation | audit | SEX | family:logistic | 0.004385 ± 0.002921 |
| development evaluation | audit | SEX | family:mlp | 0.008554 ± 0.002957 |
| validation | audit | SEX | family:mlp | 0.008102 ± 0.001312 |
| development evaluation | audit | SEX | independent_selected | 0.008554 ± 0.002957 |
| validation | audit | SEX | independent_selected | 0.008102 ± 0.001312 |
| development evaluation | audit | SEX | selected | 0.010078 ± 0.004332 |
| validation | audit | SEX | selected | 0.007861 ± 0.001626 |
| development evaluation | transfer | Civilian at work | family:logistic | 0.000375 ± 0.000515 |
| validation | transfer | Civilian at work | family:logistic | -0.000155 ± 0.000286 |
| development evaluation | transfer | Civilian at work | family:mlp | -0.000822 ± 0.001331 |
| validation | transfer | Civilian at work | family:mlp | -0.001547 ± 0.000586 |
| development evaluation | transfer | Civilian at work | selected | 0.000009 ± 0.000297 |
| validation | transfer | Civilian at work | selected | -0.001196 ± 0.001179 |
| development evaluation | transfer | Commute >20 min | family:logistic | 0.000424 ± 0.000821 |
| validation | transfer | Commute >20 min | family:logistic | 0.000270 ± 0.000460 |
| development evaluation | transfer | Commute >20 min | family:mlp | 0.000238 ± 0.001584 |
| validation | transfer | Commute >20 min | family:mlp | 0.000088 ± 0.001127 |
| development evaluation | transfer | Commute >20 min | selected | 0.000586 ± 0.000996 |
| validation | transfer | Commute >20 min | selected | -0.000275 ± 0.000686 |
| development evaluation | transfer | Income >$50k | family:logistic | 0.000649 ± 0.001421 |
| validation | transfer | Income >$50k | family:logistic | 0.000724 ± 0.002109 |
| development evaluation | transfer | Income >$50k | family:mlp | -0.001291 ± 0.000812 |
| validation | transfer | Income >$50k | family:mlp | -0.000543 ± 0.001062 |
| development evaluation | transfer | Income >$50k | selected | 0.000649 ± 0.001421 |
| validation | transfer | Income >$50k | selected | 0.000724 ± 0.002109 |
| development evaluation | transfer | Public coverage | family:logistic | 0.000530 ± 0.001695 |
| validation | transfer | Public coverage | family:logistic | -0.001053 ± 0.001028 |
| development evaluation | transfer | Public coverage | family:mlp | -0.000946 ± 0.001957 |
| validation | transfer | Public coverage | family:mlp | -0.003162 ± 0.002426 |
| development evaluation | transfer | Public coverage | selected | 0.000657 ± 0.003869 |
| validation | transfer | Public coverage | selected | -0.002452 ± 0.002993 |
| development evaluation | transfer | Same residence | family:logistic | 0.001794 ± 0.000893 |
| validation | transfer | Same residence | family:logistic | 0.000287 ± 0.001858 |
| development evaluation | transfer | Same residence | family:mlp | 0.001298 ± 0.002680 |
| validation | transfer | Same residence | family:mlp | 0.002523 ± 0.003723 |
| development evaluation | transfer | Same residence | selected | 0.001298 ± 0.002680 |
| validation | transfer | Same residence | selected | 0.002523 ± 0.003723 |

## Every primary metric and person-weighted sensitivity

PWGTP sensitivity uses identical predictions and the same unweighted validation selection. Full-schema macro AUROC/balanced accuracy remain undefined with absent categories; observed-class alternatives are separately named in CSV/JSON.

### Validation

| Release | Target | Log loss | Accuracy | Balanced accuracy | AUROC / macro |
| --- | --- | --- | --- | --- | --- |
| Original PCA | Same residence | 0.500004 ± 0.007773 | 0.762939 ± 0.008261 | 0.542018 ± 0.017551 | 0.701969 ± 0.010921 |
| PCA + LEACE | Same residence | 0.510395 ± 0.006065 | 0.762712 ± 0.017876 | 0.528593 ± 0.024804 | 0.681074 ± 0.019487 |
| Rich neural bank | Same residence | 0.519075 ± 0.007071 | 0.764694 ± 0.009667 | 0.521165 ± 0.013749 | 0.655130 ± 0.018799 |
| Neural bank + LEACE | Same residence | 0.524121 ± 0.011871 | 0.759556 ± 0.015473 | 0.528652 ± 0.013821 | 0.648229 ± 0.012116 |
| Rich tree bank | Same residence | 0.527357 ± 0.011294 | 0.764589 ± 0.008397 | 0.522086 ± 0.009816 | 0.631480 ± 0.003059 |
| Tree bank + LEACE | Same residence | 0.536665 ± 0.010436 | 0.765556 ± 0.011967 | 0.511482 ± 0.009696 | 0.595714 ± 0.029867 |
| C task-only bottleneck | Same residence | 0.513373 ± 0.009356 | 0.764264 ± 0.006242 | 0.550601 ± 0.007539 | 0.675814 ± 0.010471 |
| D protected bottleneck | Same residence | 0.515896 ± 0.012953 | 0.763939 ± 0.005298 | 0.549108 ± 0.005753 | 0.670761 ± 0.008228 |
| Fitting prior | Same residence | 0.543056 ± 0.013563 | 0.767125 ± 0.011791 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Commute >20 min | 0.678247 ± 0.001214 | 0.562874 ± 0.009805 | 0.562933 ± 0.008787 | 0.597813 ± 0.004481 |
| PCA + LEACE | Commute >20 min | 0.682046 ± 0.002754 | 0.556198 ± 0.013725 | 0.555807 ± 0.012564 | 0.586970 ± 0.006161 |
| Rich neural bank | Commute >20 min | 0.672589 ± 0.001273 | 0.575425 ± 0.012785 | 0.575357 ± 0.013969 | 0.601379 ± 0.005423 |
| Neural bank + LEACE | Commute >20 min | 0.677601 ± 0.002403 | 0.567554 ± 0.007619 | 0.566827 ± 0.009262 | 0.594016 ± 0.010923 |
| Rich tree bank | Commute >20 min | 0.674715 ± 0.000318 | 0.577948 ± 0.009375 | 0.577735 ± 0.008321 | 0.597245 ± 0.004073 |
| Tree bank + LEACE | Commute >20 min | 0.681729 ± 0.004078 | 0.562003 ± 0.004512 | 0.561492 ± 0.004356 | 0.579154 ± 0.010891 |
| C task-only bottleneck | Commute >20 min | 0.676067 ± 0.001161 | 0.569491 ± 0.002176 | 0.569218 ± 0.001900 | 0.601378 ± 0.005153 |
| D protected bottleneck | Commute >20 min | 0.675791 ± 0.001662 | 0.569669 ± 0.005876 | 0.569619 ± 0.004715 | 0.602155 ± 0.001084 |
| Fitting prior | Commute >20 min | 0.693473 ± 0.000409 | 0.498491 ± 0.007846 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Income >$50k | 0.306979 ± 0.001135 | 0.861110 ± 0.004169 | 0.739244 ± 0.013304 | 0.892729 ± 0.002250 |
| PCA + LEACE | Income >$50k | 0.321661 ± 0.007824 | 0.857266 ± 0.007348 | 0.721022 ± 0.022545 | 0.880103 ± 0.007249 |
| Rich neural bank | Income >$50k | 0.287795 ± 0.003673 | 0.869607 ± 0.003277 | 0.757771 ± 0.007062 | 0.906083 ± 0.002400 |
| Neural bank + LEACE | Income >$50k | 0.325812 ± 0.008173 | 0.861798 ± 0.007897 | 0.717911 ± 0.012551 | 0.880294 ± 0.007775 |
| Rich tree bank | Income >$50k | 0.287529 ± 0.002755 | 0.868828 ± 0.005019 | 0.762732 ± 0.006977 | 0.906394 ± 0.002837 |
| Tree bank + LEACE | Income >$50k | 0.349898 ± 0.001542 | 0.862424 ± 0.001318 | 0.712003 ± 0.022514 | 0.863688 ± 0.006191 |
| C task-only bottleneck | Income >$50k | 0.294968 ± 0.006757 | 0.863328 ± 0.004403 | 0.742004 ± 0.009243 | 0.901703 ± 0.003554 |
| D protected bottleneck | Income >$50k | 0.295692 ± 0.004684 | 0.861316 ± 0.001934 | 0.734523 ± 0.000681 | 0.901131 ± 0.002773 |
| Fitting prior | Income >$50k | 0.480116 ± 0.006864 | 0.814343 ± 0.004627 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Civilian at work | 0.304701 ± 0.008147 | 0.893245 ± 0.001131 | 0.843135 ± 0.009587 | 0.882927 ± 0.014472 |
| PCA + LEACE | Civilian at work | 0.340376 ± 0.018177 | 0.877661 ± 0.004579 | 0.827889 ± 0.002491 | 0.873151 ± 0.011497 |
| Rich neural bank | Civilian at work | 0.287739 ± 0.008515 | 0.896638 ± 0.005680 | 0.845545 ± 0.006635 | 0.897266 ± 0.004108 |
| Neural bank + LEACE | Civilian at work | 0.324002 ± 0.006732 | 0.885173 ± 0.003532 | 0.836344 ± 0.005271 | 0.884465 ± 0.004486 |
| Rich tree bank | Civilian at work | 0.291640 ± 0.012045 | 0.897521 ± 0.005865 | 0.850720 ± 0.007364 | 0.893995 ± 0.005203 |
| Tree bank + LEACE | Civilian at work | 0.357083 ± 0.003590 | 0.877763 ± 0.004634 | 0.831404 ± 0.003109 | 0.871054 ± 0.007781 |
| C task-only bottleneck | Civilian at work | 0.294330 ± 0.006975 | 0.893906 ± 0.002249 | 0.843793 ± 0.007410 | 0.893229 ± 0.010089 |
| D protected bottleneck | Civilian at work | 0.293134 ± 0.006315 | 0.893578 ± 0.001977 | 0.843713 ± 0.007599 | 0.895536 ± 0.008464 |
| Fitting prior | Civilian at work | 0.626051 ± 0.011224 | 0.680710 ± 0.014514 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Public coverage | 0.486786 ± 0.001949 | 0.777425 ± 0.005462 | 0.604812 ± 0.004748 | 0.733542 ± 0.011445 |
| PCA + LEACE | Public coverage | 0.507168 ± 0.005879 | 0.765523 ± 0.005726 | 0.563137 ± 0.005691 | 0.693619 ± 0.009387 |
| Rich neural bank | Public coverage | 0.474939 ± 0.004183 | 0.781864 ± 0.004343 | 0.599104 ± 0.007900 | 0.747591 ± 0.014252 |
| Neural bank + LEACE | Public coverage | 0.496884 ± 0.009311 | 0.773556 ± 0.006520 | 0.573080 ± 0.006634 | 0.719445 ± 0.012195 |
| Rich tree bank | Public coverage | 0.475793 ± 0.006442 | 0.783526 ± 0.005850 | 0.596045 ± 0.005813 | 0.745787 ± 0.010396 |
| Tree bank + LEACE | Public coverage | 0.500226 ± 0.012284 | 0.773382 ± 0.010906 | 0.568768 ± 0.022240 | 0.709004 ± 0.004646 |
| C task-only bottleneck | Public coverage | 0.483496 ± 0.004615 | 0.778132 ± 0.009715 | 0.588999 ± 0.015439 | 0.738064 ± 0.012038 |
| D protected bottleneck | Public coverage | 0.481045 ± 0.007482 | 0.783107 ± 0.010372 | 0.591696 ± 0.012639 | 0.740179 ± 0.014393 |
| Fitting prior | Public coverage | 0.553401 ± 0.010565 | 0.758354 ± 0.009240 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | SEX | 0.654803 ± 0.005947 | 0.603099 ± 0.005095 | 0.603043 ± 0.004971 | 0.651093 ± 0.012383 |
| PCA + LEACE | SEX | 0.673654 ± 0.007783 | 0.582222 ± 0.013368 | 0.580621 ± 0.013472 | 0.615305 ± 0.020757 |
| Rich neural bank | SEX | 0.660420 ± 0.001992 | 0.599140 ± 0.007804 | 0.595625 ± 0.009181 | 0.638610 ± 0.011911 |
| Neural bank + LEACE | SEX | 0.668293 ± 0.004692 | 0.583159 ± 0.012872 | 0.580134 ± 0.015470 | 0.620989 ± 0.020690 |
| Rich tree bank | SEX | 0.676885 ± 0.005214 | 0.575092 ± 0.013754 | 0.573832 ± 0.015100 | 0.602916 ± 0.019332 |
| Tree bank + LEACE | SEX | 0.684147 ± 0.001832 | 0.550048 ± 0.007453 | 0.546871 ± 0.010501 | 0.569673 ± 0.013326 |
| C task-only bottleneck | SEX | 0.664912 ± 0.004464 | 0.591855 ± 0.003140 | 0.589712 ± 0.002232 | 0.633236 ± 0.009819 |
| D protected bottleneck | SEX | 0.672774 ± 0.003827 | 0.585223 ± 0.014029 | 0.583864 ± 0.015454 | 0.626467 ± 0.016911 |
| Fitting prior | SEX | 0.692388 ± 0.000413 | 0.521547 ± 0.004979 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | RAC1P | 1.208832 ± 0.010693 | 0.597492 ± 0.005048 | undefined (0/3) | undefined (0/3) |
| PCA + LEACE | RAC1P | 1.240505 ± 0.014155 | 0.591167 ± 0.001316 | undefined (0/3) | undefined (0/3) |
| Rich neural bank | RAC1P | 1.240366 ± 0.005523 | 0.579299 ± 0.002261 | undefined (0/3) | undefined (0/3) |
| Neural bank + LEACE | RAC1P | 1.264294 ± 0.019221 | 0.580705 ± 0.003860 | undefined (0/3) | undefined (0/3) |
| Rich tree bank | RAC1P | 1.252472 ± 0.008746 | 0.576680 ± 0.005543 | undefined (0/3) | undefined (0/3) |
| Tree bank + LEACE | RAC1P | 1.287390 ± 0.004448 | 0.575624 ± 0.009504 | undefined (0/3) | undefined (0/3) |
| C task-only bottleneck | RAC1P | 1.225577 ± 0.003344 | 0.590444 ± 0.006888 | undefined (0/3) | undefined (0/3) |
| D protected bottleneck | RAC1P | 1.229728 ± 0.005403 | 0.582638 ± 0.004449 | undefined (0/3) | undefined (0/3) |
| Fitting prior | RAC1P | 1.283497 ± 0.005963 | 0.576796 ± 0.005769 | undefined (0/3) | undefined (0/3) |

### Development evaluation

| Release | Target | Log loss | Accuracy | Balanced accuracy | AUROC / macro |
| --- | --- | --- | --- | --- | --- |
| Original PCA | Same residence | 0.487411 ± 0.005595 | 0.776310 ± 0.003900 | 0.555757 ± 0.023928 | 0.711771 ± 0.020772 |
| PCA + LEACE | Same residence | 0.496275 ± 0.004211 | 0.773966 ± 0.008160 | 0.536240 ± 0.031349 | 0.696406 ± 0.009465 |
| Rich neural bank | Same residence | 0.510529 ± 0.003504 | 0.772401 ± 0.002171 | 0.523410 ± 0.011293 | 0.658019 ± 0.008926 |
| Neural bank + LEACE | Same residence | 0.515247 ± 0.005386 | 0.767495 ± 0.008779 | 0.535355 ± 0.019337 | 0.650251 ± 0.016585 |
| Rich tree bank | Same residence | 0.517356 ± 0.002162 | 0.768826 ± 0.003650 | 0.517177 ± 0.007308 | 0.640546 ± 0.011257 |
| Tree bank + LEACE | Same residence | 0.532475 ± 0.005421 | 0.770054 ± 0.005673 | 0.507281 ± 0.007759 | 0.581692 ± 0.030874 |
| C task-only bottleneck | Same residence | 0.500295 ± 0.003986 | 0.772399 ± 0.005249 | 0.552132 ± 0.004744 | 0.686137 ± 0.007066 |
| D protected bottleneck | Same residence | 0.501592 ± 0.002660 | 0.773744 ± 0.004978 | 0.552153 ± 0.005544 | 0.684058 ± 0.002061 |
| Fitting prior | Same residence | 0.536538 ± 0.005614 | 0.772737 ± 0.005140 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Commute >20 min | 0.687570 ± 0.008575 | 0.557127 ± 0.017298 | 0.554809 ± 0.018354 | 0.573197 ± 0.020073 |
| PCA + LEACE | Commute >20 min | 0.691074 ± 0.007655 | 0.553783 ± 0.002520 | 0.551986 ± 0.002683 | 0.566762 ± 0.013256 |
| Rich neural bank | Commute >20 min | 0.681481 ± 0.002771 | 0.558207 ± 0.011498 | 0.554658 ± 0.012277 | 0.573761 ± 0.006289 |
| Neural bank + LEACE | Commute >20 min | 0.687252 ± 0.002053 | 0.547506 ± 0.008595 | 0.544327 ± 0.010468 | 0.562233 ± 0.002760 |
| Rich tree bank | Commute >20 min | 0.681863 ± 0.004760 | 0.557214 ± 0.010000 | 0.554120 ± 0.010375 | 0.576170 ± 0.015167 |
| Tree bank + LEACE | Commute >20 min | 0.689332 ± 0.003410 | 0.548733 ± 0.002761 | 0.545477 ± 0.003057 | 0.562703 ± 0.011599 |
| C task-only bottleneck | Commute >20 min | 0.684432 ± 0.005534 | 0.557587 ± 0.006960 | 0.554647 ± 0.007581 | 0.577767 ± 0.009327 |
| D protected bottleneck | Commute >20 min | 0.685018 ± 0.004689 | 0.554250 ± 0.007362 | 0.552399 ± 0.005559 | 0.575423 ± 0.008056 |
| Fitting prior | Commute >20 min | 0.693188 ± 0.000380 | 0.500055 ± 0.017845 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Income >$50k | 0.298578 ± 0.008781 | 0.866966 ± 0.002910 | 0.742142 ± 0.003935 | 0.899561 ± 0.008644 |
| PCA + LEACE | Income >$50k | 0.312714 ± 0.009209 | 0.864292 ± 0.005630 | 0.727299 ± 0.006489 | 0.887339 ± 0.011278 |
| Rich neural bank | Income >$50k | 0.278421 ± 0.007137 | 0.877636 ± 0.008565 | 0.763357 ± 0.018508 | 0.913522 ± 0.004691 |
| Neural bank + LEACE | Income >$50k | 0.318360 ± 0.002766 | 0.862506 ± 0.003626 | 0.714007 ± 0.028528 | 0.886492 ± 0.007655 |
| Rich tree bank | Income >$50k | 0.279997 ± 0.009591 | 0.876531 ± 0.006834 | 0.765900 ± 0.014067 | 0.914331 ± 0.004924 |
| Tree bank + LEACE | Income >$50k | 0.327893 ± 0.006511 | 0.861044 ± 0.007212 | 0.704422 ± 0.004794 | 0.878466 ± 0.000529 |
| C task-only bottleneck | Income >$50k | 0.291447 ± 0.010328 | 0.867941 ± 0.009060 | 0.738922 ± 0.017129 | 0.903716 ± 0.008779 |
| D protected bottleneck | Income >$50k | 0.292097 ± 0.011459 | 0.868278 ± 0.008439 | 0.737498 ± 0.016975 | 0.903172 ± 0.009782 |
| Fitting prior | Income >$50k | 0.481821 ± 0.006349 | 0.813170 ± 0.004358 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Civilian at work | 0.288407 ± 0.005005 | 0.899731 ± 0.002401 | 0.856164 ± 0.005436 | 0.895990 ± 0.006882 |
| PCA + LEACE | Civilian at work | 0.322018 ± 0.009292 | 0.887259 ± 0.001427 | 0.842734 ± 0.000483 | 0.885500 ± 0.007824 |
| Rich neural bank | Civilian at work | 0.270941 ± 0.011699 | 0.903620 ± 0.006152 | 0.858975 ± 0.008981 | 0.912559 ± 0.001712 |
| Neural bank + LEACE | Civilian at work | 0.315928 ± 0.017206 | 0.892274 ± 0.006027 | 0.849316 ± 0.010483 | 0.889992 ± 0.007343 |
| Rich tree bank | Civilian at work | 0.274990 ± 0.006996 | 0.901386 ± 0.004608 | 0.860087 ± 0.006345 | 0.910541 ± 0.002846 |
| Tree bank + LEACE | Civilian at work | 0.330752 ± 0.012562 | 0.890604 ± 0.002724 | 0.849710 ± 0.004559 | 0.888007 ± 0.006331 |
| C task-only bottleneck | Civilian at work | 0.279831 ± 0.007417 | 0.902296 ± 0.005462 | 0.859317 ± 0.008036 | 0.902455 ± 0.000885 |
| D protected bottleneck | Civilian at work | 0.279840 ± 0.007691 | 0.901960 ± 0.004748 | 0.859056 ± 0.007897 | 0.903337 ± 0.000728 |
| Fitting prior | Civilian at work | 0.632113 ± 0.006230 | 0.673544 ± 0.007647 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Public coverage | 0.504623 ± 0.011550 | 0.761098 ± 0.015362 | 0.592374 ± 0.008203 | 0.725580 ± 0.007502 |
| PCA + LEACE | Public coverage | 0.521313 ± 0.013445 | 0.757683 ± 0.008749 | 0.562303 ± 0.011321 | 0.694414 ± 0.014609 |
| Rich neural bank | Public coverage | 0.487067 ± 0.012910 | 0.768462 ± 0.008781 | 0.590619 ± 0.014917 | 0.751094 ± 0.011565 |
| Neural bank + LEACE | Public coverage | 0.505838 ± 0.008451 | 0.762576 ± 0.009052 | 0.567414 ± 0.014835 | 0.725310 ± 0.013636 |
| Rich tree bank | Public coverage | 0.488114 ± 0.014003 | 0.771700 ± 0.010292 | 0.588936 ± 0.011428 | 0.746444 ± 0.009312 |
| Tree bank + LEACE | Public coverage | 0.512242 ± 0.008722 | 0.763690 ± 0.005372 | 0.564810 ± 0.021578 | 0.708940 ± 0.007912 |
| C task-only bottleneck | Public coverage | 0.495148 ± 0.012124 | 0.765003 ± 0.010303 | 0.583188 ± 0.011294 | 0.738597 ± 0.011972 |
| D protected bottleneck | Public coverage | 0.495805 ± 0.015940 | 0.766459 ± 0.008305 | 0.581455 ± 0.009802 | 0.736620 ± 0.014397 |
| Fitting prior | Public coverage | 0.565686 ± 0.009423 | 0.746965 ± 0.008732 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | SEX | 0.655218 ± 0.011326 | 0.602646 ± 0.016481 | 0.602140 ± 0.015169 | 0.653168 ± 0.018182 |
| PCA + LEACE | SEX | 0.678120 ± 0.008021 | 0.574061 ± 0.009086 | 0.572712 ± 0.008572 | 0.609186 ± 0.011824 |
| Rich neural bank | SEX | 0.664770 ± 0.008175 | 0.591286 ± 0.015338 | 0.589147 ± 0.013817 | 0.630714 ± 0.015185 |
| Neural bank + LEACE | SEX | 0.671515 ± 0.007842 | 0.582824 ± 0.014720 | 0.581193 ± 0.012467 | 0.617251 ± 0.011868 |
| Rich tree bank | SEX | 0.676694 ± 0.002856 | 0.575438 ± 0.003589 | 0.574628 ± 0.005117 | 0.609244 ± 0.003641 |
| Tree bank + LEACE | SEX | 0.685465 ± 0.002333 | 0.545618 ± 0.014823 | 0.543604 ± 0.011699 | 0.570866 ± 0.011219 |
| C task-only bottleneck | SEX | 0.658537 ± 0.006823 | 0.595229 ± 0.005637 | 0.593270 ± 0.003127 | 0.641639 ± 0.010758 |
| D protected bottleneck | SEX | 0.668615 ± 0.007089 | 0.591072 ± 0.014640 | 0.590043 ± 0.014499 | 0.632664 ± 0.017880 |
| Fitting prior | SEX | 0.692687 ± 0.000573 | 0.515005 ± 0.008432 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | RAC1P | 1.200399 ± 0.022578 | 0.584495 ± 0.010801 | undefined (1/3) | undefined (1/3) |
| PCA + LEACE | RAC1P | 1.235046 ± 0.027316 | 0.578140 ± 0.010818 | undefined (1/3) | undefined (1/3) |
| Rich neural bank | RAC1P | 1.243173 ± 0.032479 | 0.567332 ± 0.011427 | undefined (1/3) | undefined (1/3) |
| Neural bank + LEACE | RAC1P | 1.268265 ± 0.035414 | 0.571014 ± 0.015822 | undefined (1/3) | undefined (1/3) |
| Rich tree bank | RAC1P | 1.259321 ± 0.019889 | 0.567207 ± 0.008447 | undefined (1/3) | undefined (1/3) |
| Tree bank + LEACE | RAC1P | 1.290130 ± 0.014079 | 0.563067 ± 0.004561 | undefined (1/3) | undefined (1/3) |
| C task-only bottleneck | RAC1P | 1.230281 ± 0.022703 | 0.578466 ± 0.011410 | undefined (1/3) | undefined (1/3) |
| D protected bottleneck | RAC1P | 1.232861 ± 0.023236 | 0.574457 ± 0.010116 | undefined (1/3) | undefined (1/3) |
| Fitting prior | RAC1P | 1.290043 ± 0.025603 | 0.566884 ± 0.010702 | undefined (1/3) | undefined (1/3) |

### Validation person weighted

| Release | Target | Log loss | Accuracy | Balanced accuracy | AUROC / macro |
| --- | --- | --- | --- | --- | --- |
| Original PCA | Same residence | 0.475720 ± 0.012149 | 0.787201 ± 0.012720 | 0.541851 ± 0.014671 | 0.702784 ± 0.007865 |
| PCA + LEACE | Same residence | 0.485121 ± 0.010198 | 0.783190 ± 0.020014 | 0.530371 ± 0.026308 | 0.684352 ± 0.022762 |
| Rich neural bank | Same residence | 0.492122 ± 0.014357 | 0.786987 ± 0.012848 | 0.515021 ± 0.008379 | 0.659271 ± 0.017019 |
| Neural bank + LEACE | Same residence | 0.496954 ± 0.016750 | 0.782265 ± 0.014899 | 0.523743 ± 0.013218 | 0.649013 ± 0.014497 |
| Rich tree bank | Same residence | 0.500790 ± 0.017486 | 0.785226 ± 0.013520 | 0.516934 ± 0.008287 | 0.635650 ± 0.009540 |
| Tree bank + LEACE | Same residence | 0.511091 ± 0.012408 | 0.785611 ± 0.013531 | 0.507820 ± 0.007069 | 0.594711 ± 0.041453 |
| C task-only bottleneck | Same residence | 0.489103 ± 0.017502 | 0.787144 ± 0.010979 | 0.548115 ± 0.004650 | 0.674317 ± 0.012734 |
| D protected bottleneck | Same residence | 0.491083 ± 0.018843 | 0.786581 ± 0.011544 | 0.547876 ± 0.009239 | 0.669298 ± 0.012282 |
| Fitting prior | Same residence | 0.518076 ± 0.015817 | 0.788923 ± 0.014129 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Commute >20 min | 0.677199 ± 0.002434 | 0.567832 ± 0.010347 | 0.567876 ± 0.008933 | 0.604147 ± 0.010035 |
| PCA + LEACE | Commute >20 min | 0.681596 ± 0.003300 | 0.561704 ± 0.020953 | 0.561013 ± 0.019145 | 0.590639 ± 0.008967 |
| Rich neural bank | Commute >20 min | 0.673348 ± 0.002310 | 0.578031 ± 0.015786 | 0.577709 ± 0.014808 | 0.603130 ± 0.011095 |
| Neural bank + LEACE | Commute >20 min | 0.678599 ± 0.002211 | 0.565812 ± 0.004340 | 0.565065 ± 0.003843 | 0.592966 ± 0.006845 |
| Rich tree bank | Commute >20 min | 0.675934 ± 0.000926 | 0.576583 ± 0.009089 | 0.575860 ± 0.006141 | 0.595829 ± 0.005026 |
| Tree bank + LEACE | Commute >20 min | 0.682969 ± 0.003493 | 0.556888 ± 0.002032 | 0.556367 ± 0.002745 | 0.577485 ± 0.009912 |
| C task-only bottleneck | Commute >20 min | 0.676243 ± 0.001416 | 0.569573 ± 0.009231 | 0.569254 ± 0.007322 | 0.603846 ± 0.011912 |
| D protected bottleneck | Commute >20 min | 0.675660 ± 0.001303 | 0.573310 ± 0.009164 | 0.572724 ± 0.007970 | 0.604792 ± 0.008265 |
| Fitting prior | Commute >20 min | 0.693288 ± 0.000509 | 0.503233 ± 0.009610 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Income >$50k | 0.302738 ± 0.011410 | 0.862912 ± 0.007021 | 0.728142 ± 0.012646 | 0.892567 ± 0.004584 |
| PCA + LEACE | Income >$50k | 0.317523 ± 0.016240 | 0.856648 ± 0.009105 | 0.703789 ± 0.018944 | 0.879239 ± 0.009346 |
| Rich neural bank | Income >$50k | 0.285877 ± 0.014648 | 0.871308 ± 0.003448 | 0.744947 ± 0.004237 | 0.904977 ± 0.006627 |
| Neural bank + LEACE | Income >$50k | 0.323649 ± 0.016605 | 0.863361 ± 0.012294 | 0.705196 ± 0.016943 | 0.878725 ± 0.007380 |
| Rich tree bank | Income >$50k | 0.283950 ± 0.007828 | 0.871867 ± 0.004859 | 0.753269 ± 0.006630 | 0.906458 ± 0.001634 |
| Tree bank + LEACE | Income >$50k | 0.339976 ± 0.007487 | 0.866879 ± 0.005038 | 0.706668 ± 0.022652 | 0.865367 ± 0.007868 |
| C task-only bottleneck | Income >$50k | 0.292216 ± 0.017934 | 0.864731 ± 0.009273 | 0.729970 ± 0.011627 | 0.901094 ± 0.007776 |
| D protected bottleneck | Income >$50k | 0.292776 ± 0.015818 | 0.863072 ± 0.004335 | 0.723766 ± 0.005609 | 0.900646 ± 0.006315 |
| Fitting prior | Income >$50k | 0.473620 ± 0.013674 | 0.818897 ± 0.009406 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Civilian at work | 0.303016 ± 0.002813 | 0.897782 ± 0.001721 | 0.841927 ± 0.010451 | 0.875482 ± 0.008546 |
| PCA + LEACE | Civilian at work | 0.338874 ± 0.015496 | 0.881028 ± 0.004887 | 0.825933 ± 0.002612 | 0.867142 ± 0.009099 |
| Rich neural bank | Civilian at work | 0.288589 ± 0.004723 | 0.899059 ± 0.002601 | 0.843587 ± 0.005276 | 0.891958 ± 0.006287 |
| Neural bank + LEACE | Civilian at work | 0.328837 ± 0.019095 | 0.886258 ± 0.004568 | 0.832103 ± 0.009132 | 0.877028 ± 0.003985 |
| Rich tree bank | Civilian at work | 0.291939 ± 0.006334 | 0.897845 ± 0.003083 | 0.845554 ± 0.006811 | 0.888622 ± 0.002097 |
| Tree bank + LEACE | Civilian at work | 0.356920 ± 0.005188 | 0.880649 ± 0.002374 | 0.829279 ± 0.006150 | 0.864895 ± 0.002856 |
| C task-only bottleneck | Civilian at work | 0.293327 ± 0.004619 | 0.895871 ± 0.002674 | 0.840671 ± 0.006907 | 0.887398 ± 0.008309 |
| D protected bottleneck | Civilian at work | 0.292275 ± 0.004879 | 0.895814 ± 0.002499 | 0.841027 ± 0.007589 | 0.889427 ± 0.007666 |
| Fitting prior | Civilian at work | 0.619460 ± 0.010784 | 0.689583 ± 0.013587 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Public coverage | 0.499454 ± 0.002775 | 0.770753 ± 0.003666 | 0.599369 ± 0.015880 | 0.720548 ± 0.016067 |
| PCA + LEACE | Public coverage | 0.516150 ± 0.001136 | 0.760141 ± 0.006681 | 0.562917 ± 0.016387 | 0.684082 ± 0.014464 |
| Rich neural bank | Public coverage | 0.486450 ± 0.005845 | 0.772760 ± 0.002951 | 0.591368 ± 0.006593 | 0.737237 ± 0.017977 |
| Neural bank + LEACE | Public coverage | 0.506618 ± 0.008300 | 0.764758 ± 0.005818 | 0.566560 ± 0.005310 | 0.712876 ± 0.014658 |
| Rich tree bank | Public coverage | 0.487827 ± 0.006838 | 0.775362 ± 0.007856 | 0.588738 ± 0.009457 | 0.736584 ± 0.010726 |
| Tree bank + LEACE | Public coverage | 0.511325 ± 0.006928 | 0.766021 ± 0.009241 | 0.561682 ± 0.017328 | 0.700509 ± 0.008886 |
| C task-only bottleneck | Public coverage | 0.499234 ± 0.008883 | 0.765636 ± 0.009967 | 0.575709 ± 0.017217 | 0.725789 ± 0.014267 |
| D protected bottleneck | Public coverage | 0.496640 ± 0.011953 | 0.770518 ± 0.011723 | 0.579736 ± 0.018262 | 0.726215 ± 0.019300 |
| Fitting prior | Public coverage | 0.559243 ± 0.011605 | 0.753146 ± 0.009966 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | SEX | 0.657087 ± 0.011125 | 0.599948 ± 0.011035 | 0.599380 ± 0.010914 | 0.644458 ± 0.022837 |
| PCA + LEACE | SEX | 0.678143 ± 0.012822 | 0.576863 ± 0.023751 | 0.574333 ± 0.023265 | 0.605278 ± 0.029932 |
| Rich neural bank | SEX | 0.661484 ± 0.010108 | 0.596452 ± 0.015991 | 0.592291 ± 0.016979 | 0.631908 ± 0.024826 |
| Neural bank + LEACE | SEX | 0.668700 ± 0.010633 | 0.581868 ± 0.020293 | 0.578804 ± 0.022840 | 0.617003 ± 0.029167 |
| Rich tree bank | SEX | 0.677006 ± 0.010934 | 0.573192 ± 0.024452 | 0.571733 ± 0.026203 | 0.597409 ± 0.030979 |
| Tree bank + LEACE | SEX | 0.683393 ± 0.004255 | 0.555984 ± 0.012600 | 0.553050 ± 0.015875 | 0.571405 ± 0.019719 |
| C task-only bottleneck | SEX | 0.665412 ± 0.010743 | 0.589058 ± 0.008505 | 0.586589 ± 0.011287 | 0.630831 ± 0.021720 |
| D protected bottleneck | SEX | 0.675159 ± 0.008769 | 0.580561 ± 0.014716 | 0.578823 ± 0.015196 | 0.619584 ± 0.018483 |
| Fitting prior | SEX | 0.692325 ± 0.000243 | 0.526064 ± 0.007349 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | RAC1P | 1.214821 ± 0.012640 | 0.593446 ± 0.006613 | undefined (0/3) | undefined (0/3) |
| PCA + LEACE | RAC1P | 1.240136 ± 0.017379 | 0.587786 ± 0.001156 | undefined (0/3) | undefined (0/3) |
| Rich neural bank | RAC1P | 1.243811 ± 0.007761 | 0.578428 ± 0.005340 | undefined (0/3) | undefined (0/3) |
| Neural bank + LEACE | RAC1P | 1.266158 ± 0.021202 | 0.579618 ± 0.007375 | undefined (0/3) | undefined (0/3) |
| Rich tree bank | RAC1P | 1.254550 ± 0.009227 | 0.574619 ± 0.003513 | undefined (0/3) | undefined (0/3) |
| Tree bank + LEACE | RAC1P | 1.287516 ± 0.004167 | 0.572891 ± 0.004649 | undefined (0/3) | undefined (0/3) |
| C task-only bottleneck | RAC1P | 1.230299 ± 0.007099 | 0.587114 ± 0.004464 | undefined (0/3) | undefined (0/3) |
| D protected bottleneck | RAC1P | 1.236325 ± 0.007697 | 0.578126 ± 0.004742 | undefined (0/3) | undefined (0/3) |
| Fitting prior | RAC1P | 1.284705 ± 0.004481 | 0.575154 ± 0.003044 | undefined (0/3) | undefined (0/3) |

### Development evaluation person weighted

| Release | Target | Log loss | Accuracy | Balanced accuracy | AUROC / macro |
| --- | --- | --- | --- | --- | --- |
| Original PCA | Same residence | 0.464514 ± 0.011720 | 0.795434 ± 0.004143 | 0.549055 ± 0.021186 | 0.707653 ± 0.021239 |
| PCA + LEACE | Same residence | 0.472787 ± 0.006598 | 0.795030 ± 0.005725 | 0.534146 ± 0.029653 | 0.692862 ± 0.009024 |
| Rich neural bank | Same residence | 0.485311 ± 0.004346 | 0.793823 ± 0.005484 | 0.516351 ± 0.007916 | 0.656173 ± 0.003438 |
| Neural bank + LEACE | Same residence | 0.491478 ± 0.005218 | 0.788950 ± 0.008304 | 0.527058 ± 0.017551 | 0.644798 ± 0.012976 |
| Rich tree bank | Same residence | 0.490844 ± 0.006835 | 0.791587 ± 0.004323 | 0.511795 ± 0.007485 | 0.638580 ± 0.009314 |
| Tree bank + LEACE | Same residence | 0.505749 ± 0.002469 | 0.793977 ± 0.003669 | 0.507121 ± 0.008280 | 0.576700 ± 0.038546 |
| C task-only bottleneck | Same residence | 0.475068 ± 0.002146 | 0.792043 ± 0.002174 | 0.540638 ± 0.010072 | 0.684574 ± 0.012945 |
| D protected bottleneck | Same residence | 0.476107 ± 0.003473 | 0.792582 ± 0.003446 | 0.539783 ± 0.012043 | 0.683418 ± 0.007191 |
| Fitting prior | Same residence | 0.510614 ± 0.003348 | 0.795226 ± 0.002907 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Commute >20 min | 0.690615 ± 0.007991 | 0.549921 ± 0.015707 | 0.546679 ± 0.017127 | 0.563494 ± 0.020653 |
| PCA + LEACE | Commute >20 min | 0.692154 ± 0.007944 | 0.552752 ± 0.006577 | 0.549786 ± 0.007021 | 0.561384 ± 0.017757 |
| Rich neural bank | Commute >20 min | 0.684738 ± 0.004633 | 0.554099 ± 0.014895 | 0.548928 ± 0.015497 | 0.565415 ± 0.007056 |
| Neural bank + LEACE | Commute >20 min | 0.688564 ± 0.004083 | 0.542711 ± 0.013195 | 0.537430 ± 0.016042 | 0.559579 ± 0.008938 |
| Rich tree bank | Commute >20 min | 0.685216 ± 0.005507 | 0.556772 ± 0.017162 | 0.552002 ± 0.016944 | 0.571381 ± 0.020921 |
| Tree bank + LEACE | Commute >20 min | 0.692684 ± 0.004875 | 0.543019 ± 0.002202 | 0.538021 ± 0.000533 | 0.555770 ± 0.009800 |
| C task-only bottleneck | Commute >20 min | 0.686297 ± 0.001931 | 0.556604 ± 0.008077 | 0.552277 ± 0.008839 | 0.574496 ± 0.003508 |
| D protected bottleneck | Commute >20 min | 0.686496 ± 0.002314 | 0.555234 ± 0.015568 | 0.552685 ± 0.013299 | 0.572587 ± 0.006769 |
| Fitting prior | Commute >20 min | 0.693108 ± 0.000489 | 0.499653 ± 0.023173 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Income >$50k | 0.310568 ± 0.008791 | 0.862559 ± 0.005658 | 0.727316 ± 0.008792 | 0.890815 ± 0.010209 |
| PCA + LEACE | Income >$50k | 0.330002 ± 0.007493 | 0.857409 ± 0.007850 | 0.708945 ± 0.010853 | 0.872954 ± 0.012382 |
| Rich neural bank | Income >$50k | 0.289414 ± 0.009718 | 0.874057 ± 0.012088 | 0.749783 ± 0.020539 | 0.906656 ± 0.006721 |
| Neural bank + LEACE | Income >$50k | 0.332625 ± 0.006531 | 0.858072 ± 0.005572 | 0.700598 ± 0.029093 | 0.876651 ± 0.008751 |
| Rich tree bank | Income >$50k | 0.290793 ± 0.012727 | 0.870805 ± 0.011587 | 0.749609 ± 0.020073 | 0.906942 ± 0.008461 |
| Tree bank + LEACE | Income >$50k | 0.343861 ± 0.023211 | 0.857306 ± 0.011027 | 0.693363 ± 0.009488 | 0.866072 ± 0.008285 |
| C task-only bottleneck | Income >$50k | 0.303143 ± 0.012336 | 0.864019 ± 0.010746 | 0.724730 ± 0.017164 | 0.894910 ± 0.011110 |
| D protected bottleneck | Income >$50k | 0.303279 ± 0.013486 | 0.864448 ± 0.010206 | 0.725112 ± 0.016266 | 0.894521 ± 0.012364 |
| Fitting prior | Income >$50k | 0.485039 ± 0.010056 | 0.811006 ± 0.006873 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Civilian at work | 0.277256 ± 0.012455 | 0.908927 ± 0.007070 | 0.857215 ± 0.012616 | 0.891243 ± 0.006392 |
| PCA + LEACE | Civilian at work | 0.309628 ± 0.007200 | 0.897170 ± 0.004286 | 0.844816 ± 0.008094 | 0.879959 ± 0.005011 |
| Rich neural bank | Civilian at work | 0.262001 ± 0.016606 | 0.910833 ± 0.009072 | 0.858988 ± 0.015456 | 0.907799 ± 0.006383 |
| Neural bank + LEACE | Civilian at work | 0.302972 ± 0.022398 | 0.899875 ± 0.008912 | 0.849931 ± 0.015241 | 0.886100 ± 0.011713 |
| Rich tree bank | Civilian at work | 0.265649 ± 0.011761 | 0.908382 ± 0.005844 | 0.859548 ± 0.011892 | 0.906389 ± 0.006331 |
| Tree bank + LEACE | Civilian at work | 0.314541 ± 0.010777 | 0.897800 ± 0.004367 | 0.850316 ± 0.009101 | 0.890037 ± 0.005392 |
| C task-only bottleneck | Civilian at work | 0.270632 ± 0.013288 | 0.911120 ± 0.008190 | 0.860394 ± 0.013666 | 0.895899 ± 0.004840 |
| D protected bottleneck | Civilian at work | 0.270341 ± 0.013232 | 0.910506 ± 0.007180 | 0.859772 ± 0.013134 | 0.897567 ± 0.004001 |
| Fitting prior | Civilian at work | 0.615531 ± 0.001928 | 0.695344 ± 0.002759 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | Public coverage | 0.514111 ± 0.013080 | 0.756315 ± 0.020133 | 0.587080 ± 0.000597 | 0.717274 ± 0.000213 |
| PCA + LEACE | Public coverage | 0.530149 ± 0.012554 | 0.753727 ± 0.008904 | 0.560666 ± 0.007629 | 0.686946 ± 0.005262 |
| Rich neural bank | Public coverage | 0.492893 ± 0.015925 | 0.761706 ± 0.013324 | 0.584394 ± 0.009932 | 0.748352 ± 0.010642 |
| Neural bank + LEACE | Public coverage | 0.509596 ± 0.009402 | 0.759346 ± 0.009371 | 0.564335 ± 0.014410 | 0.723698 ± 0.010707 |
| Rich tree bank | Public coverage | 0.496851 ± 0.017326 | 0.764329 ± 0.016480 | 0.581647 ± 0.002420 | 0.739338 ± 0.007977 |
| Tree bank + LEACE | Public coverage | 0.519100 ± 0.011282 | 0.757123 ± 0.009809 | 0.558640 ± 0.012004 | 0.703999 ± 0.003630 |
| C task-only bottleneck | Public coverage | 0.505044 ± 0.011269 | 0.759750 ± 0.011238 | 0.575329 ± 0.010467 | 0.728451 ± 0.005776 |
| D protected bottleneck | Public coverage | 0.504960 ± 0.015929 | 0.758596 ± 0.013056 | 0.570553 ± 0.004998 | 0.728115 ± 0.009632 |
| Fitting prior | Public coverage | 0.570347 ± 0.011212 | 0.742781 ± 0.010502 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | SEX | 0.660511 ± 0.009960 | 0.596418 ± 0.016796 | 0.594673 ± 0.014498 | 0.641142 ± 0.016744 |
| PCA + LEACE | SEX | 0.685390 ± 0.005541 | 0.562481 ± 0.011337 | 0.559113 ± 0.012404 | 0.591691 ± 0.007248 |
| Rich neural bank | SEX | 0.667123 ± 0.006632 | 0.589656 ± 0.005683 | 0.584507 ± 0.004189 | 0.622860 ± 0.011377 |
| Neural bank + LEACE | SEX | 0.674760 ± 0.006309 | 0.578620 ± 0.014303 | 0.574717 ± 0.011749 | 0.608014 ± 0.008957 |
| Rich tree bank | SEX | 0.679560 ± 0.002528 | 0.571883 ± 0.006879 | 0.569471 ± 0.009240 | 0.599470 ± 0.008687 |
| Tree bank + LEACE | SEX | 0.687218 ± 0.001967 | 0.542221 ± 0.013830 | 0.537454 ± 0.008742 | 0.562674 ± 0.007645 |
| C task-only bottleneck | SEX | 0.662334 ± 0.005647 | 0.589761 ± 0.001365 | 0.586081 ± 0.005392 | 0.631634 ± 0.010102 |
| D protected bottleneck | SEX | 0.673340 ± 0.006577 | 0.586564 ± 0.009452 | 0.584006 ± 0.009773 | 0.622400 ± 0.013897 |
| Fitting prior | SEX | 0.692113 ± 0.000765 | 0.525434 ± 0.006520 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| Original PCA | RAC1P | 1.203587 ± 0.039868 | 0.576852 ± 0.021465 | undefined (1/3) | undefined (1/3) |
| PCA + LEACE | RAC1P | 1.234764 ± 0.044637 | 0.573252 ± 0.019014 | undefined (1/3) | undefined (1/3) |
| Rich neural bank | RAC1P | 1.241852 ± 0.043678 | 0.569605 ± 0.013015 | undefined (1/3) | undefined (1/3) |
| Neural bank + LEACE | RAC1P | 1.266959 ± 0.044732 | 0.573922 ± 0.014999 | undefined (1/3) | undefined (1/3) |
| Rich tree bank | RAC1P | 1.255603 ± 0.037122 | 0.568781 ± 0.010068 | undefined (1/3) | undefined (1/3) |
| Tree bank + LEACE | RAC1P | 1.285837 ± 0.033122 | 0.566322 ± 0.008037 | undefined (1/3) | undefined (1/3) |
| C task-only bottleneck | RAC1P | 1.234828 ± 0.047500 | 0.572517 ± 0.018162 | undefined (1/3) | undefined (1/3) |
| D protected bottleneck | RAC1P | 1.238074 ± 0.047608 | 0.572623 ± 0.014922 | undefined (1/3) | undefined (1/3) |
| Fitting prior | RAC1P | 1.287242 ± 0.039105 | 0.568724 ± 0.012209 | undefined (1/3) | undefined (1/3) |

## Scope

The five-task policy is unchanged. Missing race support is an assessment limit, not protection. Task-only C versus protected D tests the contribution of this standard protection objective under the fixed recipe. Residual recovery and utility losses must be considered together. No PCRL novelty, universal privacy, coalition protection, official Census estimate, or untouched-household confirmation follows from these measurements.

## Training phases and exposure accounting

Training curves are diagnostics on representation-fitting and source-validation rows. They do not select a checkpoint, and they do not contain reserved-task labels. C/D use the fixed final continuation epoch. Step and exposure totals below include shared warm-up. Shared work is inherited by both arms; totals are model histories, not duplicated runtime claims. Each adversary update processes both attribute networks on the same minibatch. [Full curves](TRAINING_CURVES.csv) and [training accounting](TRAINING.csv) retain phase-specific details.

| Seed | Method | Mapper steps | Adversary steps before catch-up | Mapper row exposures | Adversary row exposures before catch-up | Final epoch | Exact shared fork |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | 5880 | 10920 | 1471820 | 2733380 | 80 | True |
| 0 | D protected bottleneck | 5880 | 10920 | 1471820 | 2733380 | 80 | True |
| 1 | C task-only bottleneck | 5740 | 10660 | 1459920 | 2711280 | 80 | True |
| 1 | D protected bottleneck | 5740 | 10660 | 1459920 | 2711280 | 80 | True |
| 2 | C task-only bottleneck | 5880 | 10920 | 1477140 | 2743260 | 80 | True |
| 2 | D protected bottleneck | 5880 | 10920 | 1477140 | 2743260 | 80 | True |

Catch-up resets Adam and adds attacker-pool training after the inherited representation-pool history. The saved adversary performs zero new fitting updates. Its provided attacker-fit arrays serve fidelity/scoring checks, not an additional fitting phase. Fresh independent MLPs have no inherited warm-up history.

| Seed | Method | Target | Candidate | New fitting rows | New updates | New row exposures | Selected epoch | Identity-input fidelity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | SEX | catchup | 2993 | 1440 | 359160 | 5 | True |
| 0 | C task-only bottleneck | SEX | saved_adversary | 0 | 0 | 0 | 0 | True |
| 0 | C task-only bottleneck | RAC1P | catchup | 2993 | 1440 | 359160 | 5 | True |
| 0 | C task-only bottleneck | RAC1P | saved_adversary | 0 | 0 | 0 | 0 | True |
| 0 | D protected bottleneck | SEX | catchup | 2993 | 1440 | 359160 | 15 | True |
| 0 | D protected bottleneck | SEX | saved_adversary | 0 | 0 | 0 | 0 | True |
| 0 | D protected bottleneck | RAC1P | catchup | 2993 | 1440 | 359160 | 5 | True |
| 0 | D protected bottleneck | RAC1P | saved_adversary | 0 | 0 | 0 | 0 | True |
| 1 | C task-only bottleneck | SEX | catchup | 3021 | 1440 | 362520 | 15 | True |
| 1 | C task-only bottleneck | SEX | saved_adversary | 0 | 0 | 0 | 0 | True |
| 1 | C task-only bottleneck | RAC1P | catchup | 3021 | 1440 | 362520 | 5 | True |
| 1 | C task-only bottleneck | RAC1P | saved_adversary | 0 | 0 | 0 | 0 | True |
| 1 | D protected bottleneck | SEX | catchup | 3021 | 1440 | 362520 | 5 | True |
| 1 | D protected bottleneck | SEX | saved_adversary | 0 | 0 | 0 | 0 | True |
| 1 | D protected bottleneck | RAC1P | catchup | 3021 | 1440 | 362520 | 5 | True |
| 1 | D protected bottleneck | RAC1P | saved_adversary | 0 | 0 | 0 | 0 | True |
| 2 | C task-only bottleneck | SEX | catchup | 2941 | 1440 | 352920 | 15 | True |
| 2 | C task-only bottleneck | SEX | saved_adversary | 0 | 0 | 0 | 0 | True |
| 2 | C task-only bottleneck | RAC1P | catchup | 2941 | 1440 | 352920 | 5 | True |
| 2 | C task-only bottleneck | RAC1P | saved_adversary | 0 | 0 | 0 | 0 | True |
| 2 | D protected bottleneck | SEX | catchup | 2941 | 1440 | 352920 | 10 | True |
| 2 | D protected bottleneck | SEX | saved_adversary | 0 | 0 | 0 | 0 | True |
| 2 | D protected bottleneck | RAC1P | catchup | 2941 | 1440 | 352920 | 5 | True |
| 2 | D protected bottleneck | RAC1P | saved_adversary | 0 | 0 | 0 | 0 | True |

## Fixed-final training diagnostics

These source losses are from the jointly trained source heads, separate from the independently fitted utility heads in the primary tables. Reconstruction is standardized-PCA MSE. Attribute cross-entropy here is from the current training adversary on fitting rows, not a held-out privacy result.

| Seed | Method | Fit source CE | Source-validation CE | Fit reconstruction MSE | Fit SEX adversary CE | Fit RAC1P adversary CE | Normalized mean adversary CE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | 0.316139 | 0.358147 | 0.071648 | 0.577172 | 1.091410 | 0.837724 |
| 0 | D protected bottleneck | 0.316319 | 0.357313 | 0.088182 | 0.600167 | 1.140549 | 0.873283 |
| 1 | C task-only bottleneck | 0.313917 | 0.369842 | 0.071576 | 0.572700 | 1.081849 | 0.837635 |
| 1 | D protected bottleneck | 0.314147 | 0.369222 | 0.077955 | 0.590473 | 1.109589 | 0.861342 |
| 2 | C task-only bottleneck | 0.316433 | 0.365053 | 0.068361 | 0.564883 | 1.084656 | 0.826217 |
| 2 | D protected bottleneck | 0.316237 | 0.365278 | 0.079193 | 0.599935 | 1.117650 | 0.864250 |

## Training support and gradient checks

Training support is reported separately from attacker support. Inherited exposure to a rare category does not replace missing support in the fixed independent attack/validation pools. Gradient norms are measured on the first shared continuation minibatch; they are not a record of every optimizer update.

| Seed | Target | Training support | Missing rows | Schema complete | Fitting entropy |
| --- | --- | --- | --- | --- | --- |
| 0 | SEX | [5391, 5122] | 0 | True | 0.692820 |
| 0 | RAC1P | [5989, 555, 90, 1, 32, 1674, 49, 1527, 596] | 0 | True | 1.295640 |
| 1 | SEX | [5351, 5077] | 0 | True | 0.692802 |
| 1 | RAC1P | [5993, 529, 64, 1, 25, 1626, 35, 1536, 619] | 0 | True | 1.274823 |
| 2 | SEX | [5458, 5093] | 0 | True | 0.692549 |
| 2 | RAC1P | [5990, 581, 78, 0, 28, 1672, 58, 1562, 582] | 0 | False | 1.296232 |

The potential protection gradient measures the common objective path; C applies zero protection gradient and only D applies the potential term.

| Seed | Method | Base mapper gradient L2 | Potential protection gradient L2 | Applied protection gradient L2 | Source mapper gradient L2 | Weighted reconstruction mapper gradient L2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | C task-only bottleneck | 0.119453 | 0.020207 | 0.000000 | 0.119278 | 0.013297 |
| 0 | D protected bottleneck | 0.119453 | 0.020207 | 0.020207 | 0.119278 | 0.013297 |
| 1 | C task-only bottleneck | 0.145755 | 0.019336 | 0.000000 | 0.145560 | 0.010285 |
| 1 | D protected bottleneck | 0.145755 | 0.019336 | 0.019336 | 0.145560 | 0.010285 |
| 2 | C task-only bottleneck | 0.121834 | 0.023055 | 0.000000 | 0.121173 | 0.012300 |
| 2 | D protected bottleneck | 0.121834 | 0.023055 | 0.023055 | 0.121173 | 0.012300 |

## Recorded internal runtime

| Seed | Total s | Reference load s | Training s | Reuse verification s | Utility s | Fresh audit s | Catch-up s | Evaluation s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 28.973812 | 1.601216 | 14.440832 | 2.115934 | 0.899887 | 5.966245 | 1.938594 | 1.977979 |
| 1 | 27.996490 | 1.582648 | 13.818857 | 1.987717 | 0.920310 | 5.847242 | 1.903180 | 1.900919 |
| 2 | 28.312405 | 1.651406 | 13.937933 | 2.000015 | 0.887981 | 5.985997 | 1.889593 | 1.927014 |

Sum of internal seed runtimes: 85.282708 seconds. The orchestrator records process wall time separately. All states and historical references remain frozen during auditing.
