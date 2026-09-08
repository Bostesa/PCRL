# PCA16 initialization: paired stage analysis

All comparisons use saved post-freeze heads and audit choices. Primary is the matched independent audit for both initialized and historical learned releases. Utility comparisons retain every task. The stage contrasts separate initialization, shared warm-up, task-only continuation, and added protection. C_init and D_init are separate continuations from W; D_init is not trained from C_init. They are descriptive differences, not an outcome-driven checkpoint-selection rule.

[Main table](TABLE.md), [all paired metrics](PAIRED.csv), [new fit accounting](FITTING.csv), [new classes](PER_CLASS.csv), [new curves](CURVES.csv), [full aggregates](summary.json).

## Every stage contrast, per-seed task log loss

| Seed | Split | Left − right | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | I initialized mapper − Frozen PCA16 | -3.462e-10 | 4.357e-10 | 8.511e-10 | 1.172e-09 | 2.840e-09 |
| 0 | validation | W after common warm-up − I initialized mapper | -0.001701 | -0.006758 | -0.017213 | -0.011022 | -0.013979 |
| 0 | validation | C initialized task-only − W after common warm-up | 0.001976 | -0.000429 | 0.001721 | 0.001747 | 0.003417 |
| 0 | validation | D initialized protected − C initialized task-only | 0.000587 | -0.000252 | -0.000172 | -0.000686 | 0.001204 |
| 0 | validation | C initialized task-only − Historical C task-only | 0.001268 | -0.001796 | 0.003605 | -0.000535 | -0.003713 |
| 0 | validation | D initialized protected − Historical D protected | 0.001116 | -0.002142 | 0.000557 | 0.000830 | -0.002476 |
| 1 | validation | I initialized mapper − Frozen PCA16 | -4.298e-10 | -1.129e-10 | -5.054e-10 | 4.990e-11 | -7.002e-10 |
| 1 | validation | W after common warm-up − I initialized mapper | -0.008499 | -0.004866 | -0.017688 | -0.010454 | -0.008434 |
| 1 | validation | C initialized task-only − W after common warm-up | 0.007164 | 0.001290 | 0.001063 | 0.002735 | 0.002021 |
| 1 | validation | D initialized protected − C initialized task-only | 0.002157 | -0.000557 | 0.001314 | -0.000537 | -0.000832 |
| 1 | validation | C initialized task-only − Historical C task-only | -0.004951 | -0.001741 | -0.001770 | -0.002332 | -0.003004 |
| 1 | validation | D initialized protected − Historical D protected | -0.002821 | -0.002444 | -0.001093 | -0.003019 | -0.002312 |
| 2 | validation | I initialized mapper − Frozen PCA16 | -1.680e-10 | -8.059e-11 | -7.728e-10 | 5.927e-11 | -2.003e-10 |
| 2 | validation | W after common warm-up − I initialized mapper | 0.015468 | -0.000989 | -0.002430 | -0.015535 | -0.015427 |
| 2 | validation | C initialized task-only − W after common warm-up | 0.004748 | 0.000376 | 0.001827 | 0.003113 | 0.002885 |
| 2 | validation | D initialized protected − C initialized task-only | 0.001104 | -0.000178 | 0.000357 | 0.000503 | -0.003642 |
| 2 | validation | C initialized task-only − Historical C task-only | 0.000443 | -0.000157 | -0.000133 | -0.002087 | -0.003482 |
| 2 | validation | D initialized protected − Historical D protected | -0.005256 | 0.000732 | 0.001564 | 0.000102 | -0.001326 |
| 0 | development evaluation | I initialized mapper − Frozen PCA16 | -1.145e-10 | 2.112e-10 | 1.313e-10 | -6.522e-10 | 7.256e-10 |
| 0 | development evaluation | W after common warm-up − I initialized mapper | 0.003739 | -0.000482 | -0.006112 | -0.014592 | -0.007397 |
| 0 | development evaluation | C initialized task-only − W after common warm-up | -0.001815 | -0.001125 | 0.004471 | 0.000661 | -0.000086 |
| 0 | development evaluation | D initialized protected − C initialized task-only | 0.000529 | -0.000158 | 0.000273 | -0.000543 | -0.000892 |
| 0 | development evaluation | C initialized task-only − Historical C task-only | 0.002148 | -0.002762 | 0.000388 | 0.000466 | -0.001152 |
| 0 | development evaluation | D initialized protected − Historical D protected | -0.000083 | -0.002402 | -0.001536 | 0.000256 | -0.006902 |
| 1 | development evaluation | I initialized mapper − Frozen PCA16 | 3.083e-10 | -2.762e-10 | -8.532e-10 | -1.095e-09 | -9.412e-10 |
| 1 | development evaluation | W after common warm-up − I initialized mapper | 0.000313 | -0.002004 | -0.010419 | -0.011133 | -0.012045 |
| 1 | development evaluation | C initialized task-only − W after common warm-up | 0.007296 | 0.002415 | 0.001396 | -0.000759 | 0.003651 |
| 1 | development evaluation | D initialized protected − C initialized task-only | 0.001288 | -0.000393 | 0.002132 | -0.000913 | 0.000014 |
| 1 | development evaluation | C initialized task-only − Historical C task-only | -0.003582 | 0.001979 | -0.002758 | -0.007095 | 0.000075 |
| 1 | development evaluation | D initialized protected − Historical D protected | -0.000498 | 0.000724 | -0.000030 | -0.008174 | 0.000214 |
| 2 | development evaluation | I initialized mapper − Frozen PCA16 | -2.520e-10 | 3.417e-10 | 5.410e-10 | 1.497e-09 | 3.784e-10 |
| 2 | development evaluation | W after common warm-up − I initialized mapper | 0.005451 | 0.000135 | -0.006972 | -0.015716 | -0.018206 |
| 2 | development evaluation | C initialized task-only − W after common warm-up | -0.001703 | 0.002326 | 0.003277 | 0.002571 | 0.002311 |
| 2 | development evaluation | D initialized protected − C initialized task-only | 0.004392 | 0.000568 | 0.001185 | 0.001199 | -0.000880 |
| 2 | development evaluation | C initialized task-only − Historical C task-only | -0.000981 | 0.005332 | -0.003107 | -0.001725 | 0.000974 |
| 2 | development evaluation | D initialized protected − Historical D protected | 0.000481 | 0.004484 | -0.002270 | -0.000721 | 0.002855 |

## All task/family stage contrasts, mean ± sample SD

Within-family selections use their designated validation pool. Fresh MLP family comparisons exclude catch-up.

| Split | Left − right | Target | Selector | Log-loss difference |
| --- | --- | --- | --- | --- |
| development evaluation | I initialized mapper − Frozen PCA16 | Civilian at work | family:logistic | 6.025e-10 ± 6.776e-10 |
| validation | I initialized mapper − Frozen PCA16 | Civilian at work | family:logistic | 3.480e-11 ± 7.687e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Civilian at work | family:mlp | -8.313e-11 ± 1.387e-09 |
| validation | I initialized mapper − Frozen PCA16 | Civilian at work | family:mlp | 4.269e-10 ± 6.450e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Civilian at work | primary | -8.313e-11 ± 1.387e-09 |
| validation | I initialized mapper − Frozen PCA16 | Civilian at work | primary | 4.269e-10 ± 6.450e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Commute >20 min | family:logistic | -7.305e-11 ± 3.592e-10 |
| validation | I initialized mapper − Frozen PCA16 | Commute >20 min | family:logistic | -4.698e-11 ± 8.773e-11 |
| development evaluation | I initialized mapper − Frozen PCA16 | Commute >20 min | family:mlp | 3.526e-10 ± 2.732e-10 |
| validation | I initialized mapper − Frozen PCA16 | Commute >20 min | family:mlp | -1.607e-10 ± 1.114e-09 |
| development evaluation | I initialized mapper − Frozen PCA16 | Commute >20 min | primary | 9.223e-11 ± 3.256e-10 |
| validation | I initialized mapper − Frozen PCA16 | Commute >20 min | primary | 8.073e-11 ± 3.079e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Income >$50k | family:logistic | 4.690e-11 ± 6.703e-10 |
| validation | I initialized mapper − Frozen PCA16 | Income >$50k | family:logistic | 5.028e-10 ± 4.537e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Income >$50k | family:mlp | -6.026e-11 ± 7.166e-10 |
| validation | I initialized mapper − Frozen PCA16 | Income >$50k | family:mlp | -1.424e-10 ± 8.707e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Income >$50k | primary | -6.026e-11 ± 7.166e-10 |
| validation | I initialized mapper − Frozen PCA16 | Income >$50k | primary | -1.424e-10 ± 8.707e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Public coverage | family:logistic | 2.113e-10 ± 2.141e-10 |
| validation | I initialized mapper − Frozen PCA16 | Public coverage | family:logistic | 1.622e-10 ± 3.140e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Public coverage | family:mlp | -1.609e-10 ± 8.384e-10 |
| validation | I initialized mapper − Frozen PCA16 | Public coverage | family:mlp | 1.276e-09 ± 1.806e-09 |
| development evaluation | I initialized mapper − Frozen PCA16 | Public coverage | primary | 5.428e-11 ± 8.794e-10 |
| validation | I initialized mapper − Frozen PCA16 | Public coverage | primary | 6.464e-10 ± 1.916e-09 |
| development evaluation | I initialized mapper − Frozen PCA16 | Same residence | family:logistic | -1.940e-11 ± 2.920e-10 |
| validation | I initialized mapper − Frozen PCA16 | Same residence | family:logistic | -3.147e-10 ± 1.337e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Same residence | family:mlp | -5.554e-10 ± 3.096e-10 |
| validation | I initialized mapper − Frozen PCA16 | Same residence | family:mlp | -4.791e-10 ± 7.753e-10 |
| development evaluation | I initialized mapper − Frozen PCA16 | Same residence | primary | -1.940e-11 ± 2.920e-10 |
| validation | I initialized mapper − Frozen PCA16 | Same residence | primary | -3.147e-10 ± 1.337e-10 |
| development evaluation | D initialized protected − Historical D protected | RAC1P | catchup_inclusive | -0.000685 ± 0.001484 |
| validation | D initialized protected − Historical D protected | RAC1P | catchup_inclusive | -0.002440 ± 0.011152 |
| development evaluation | D initialized protected − Historical D protected | RAC1P | family:histgb | -0.000444 ± 0.015460 |
| validation | D initialized protected − Historical D protected | RAC1P | family:histgb | 0.006502 ± 0.012578 |
| development evaluation | D initialized protected − Historical D protected | RAC1P | family:logistic | 0.000032 ± 0.010460 |
| validation | D initialized protected − Historical D protected | RAC1P | family:logistic | -0.002405 ± 0.001395 |
| development evaluation | D initialized protected − Historical D protected | RAC1P | family:mlp | -0.000119 ± 0.004655 |
| validation | D initialized protected − Historical D protected | RAC1P | family:mlp | 0.000165 ± 0.011359 |
| development evaluation | D initialized protected − Historical D protected | RAC1P | primary | -0.000119 ± 0.004655 |
| validation | D initialized protected − Historical D protected | RAC1P | primary | 0.000165 ± 0.011359 |
| development evaluation | D initialized protected − Historical D protected | SEX | catchup_inclusive | -0.001356 ± 0.007042 |
| validation | D initialized protected − Historical D protected | SEX | catchup_inclusive | -0.005296 ± 0.003515 |
| development evaluation | D initialized protected − Historical D protected | SEX | family:histgb | -0.001789 ± 0.002996 |
| validation | D initialized protected − Historical D protected | SEX | family:histgb | -0.002846 ± 0.002985 |
| development evaluation | D initialized protected − Historical D protected | SEX | family:logistic | -0.008027 ± 0.016252 |
| validation | D initialized protected − Historical D protected | SEX | family:logistic | -0.005730 ± 0.012271 |
| development evaluation | D initialized protected − Historical D protected | SEX | family:mlp | -0.003643 ± 0.006646 |
| validation | D initialized protected − Historical D protected | SEX | family:mlp | -0.004773 ± 0.002609 |
| development evaluation | D initialized protected − Historical D protected | SEX | primary | -0.003643 ± 0.006646 |
| validation | D initialized protected − Historical D protected | SEX | primary | -0.004773 ± 0.002609 |
| development evaluation | D initialized protected − Historical D protected | Civilian at work | family:logistic | -0.002656 ± 0.002688 |
| validation | D initialized protected − Historical D protected | Civilian at work | family:logistic | -0.002281 ± 0.002080 |
| development evaluation | D initialized protected − Historical D protected | Civilian at work | family:mlp | -0.002406 ± 0.003796 |
| validation | D initialized protected − Historical D protected | Civilian at work | family:mlp | -0.000712 ± 0.002074 |
| development evaluation | D initialized protected − Historical D protected | Civilian at work | primary | -0.002879 ± 0.004611 |
| validation | D initialized protected − Historical D protected | Civilian at work | primary | -0.000696 ± 0.002045 |
| development evaluation | D initialized protected − Historical D protected | Commute >20 min | family:logistic | 0.001260 ± 0.003328 |
| validation | D initialized protected − Historical D protected | Commute >20 min | family:logistic | -0.002752 ± 0.000530 |
| development evaluation | D initialized protected − Historical D protected | Commute >20 min | family:mlp | 0.002381 ± 0.001696 |
| validation | D initialized protected − Historical D protected | Commute >20 min | family:mlp | 0.000370 ± 0.003890 |
| development evaluation | D initialized protected − Historical D protected | Commute >20 min | primary | 0.000935 ± 0.003448 |
| validation | D initialized protected − Historical D protected | Commute >20 min | primary | -0.001285 ± 0.001753 |
| development evaluation | D initialized protected − Historical D protected | Income >$50k | family:logistic | -0.001278 ± 0.001142 |
| validation | D initialized protected − Historical D protected | Income >$50k | family:logistic | 0.000343 ± 0.001342 |
| development evaluation | D initialized protected − Historical D protected | Income >$50k | family:mlp | -0.000047 ± 0.002310 |
| validation | D initialized protected − Historical D protected | Income >$50k | family:mlp | 0.000365 ± 0.004150 |
| development evaluation | D initialized protected − Historical D protected | Income >$50k | primary | -0.001278 ± 0.001142 |
| validation | D initialized protected − Historical D protected | Income >$50k | primary | 0.000343 ± 0.001342 |
| development evaluation | D initialized protected − Historical D protected | Public coverage | family:logistic | -0.001435 ± 0.002655 |
| validation | D initialized protected − Historical D protected | Public coverage | family:logistic | -0.003960 ± 0.002581 |
| development evaluation | D initialized protected − Historical D protected | Public coverage | family:mlp | -0.001140 ± 0.005178 |
| validation | D initialized protected − Historical D protected | Public coverage | family:mlp | -0.001015 ± 0.000541 |
| development evaluation | D initialized protected − Historical D protected | Public coverage | primary | -0.001278 ± 0.005046 |
| validation | D initialized protected − Historical D protected | Public coverage | primary | -0.002038 ± 0.000622 |
| development evaluation | D initialized protected − Historical D protected | Same residence | family:logistic | 0.000016 ± 0.002916 |
| validation | D initialized protected − Historical D protected | Same residence | family:logistic | -0.001101 ± 0.004705 |
| development evaluation | D initialized protected − Historical D protected | Same residence | family:mlp | -0.000033 ± 0.000492 |
| validation | D initialized protected − Historical D protected | Same residence | family:mlp | -0.002321 ± 0.003215 |
| development evaluation | D initialized protected − Historical D protected | Same residence | primary | -0.000033 ± 0.000492 |
| validation | D initialized protected − Historical D protected | Same residence | primary | -0.002321 ± 0.003215 |
| development evaluation | D initialized protected − C initialized task-only | RAC1P | catchup_inclusive | 0.010327 ± 0.018311 |
| validation | D initialized protected − C initialized task-only | RAC1P | catchup_inclusive | 0.009468 ± 0.010194 |
| development evaluation | D initialized protected − C initialized task-only | RAC1P | family:histgb | 0.011385 ± 0.016879 |
| validation | D initialized protected − C initialized task-only | RAC1P | family:histgb | 0.013568 ± 0.005002 |
| development evaluation | D initialized protected − C initialized task-only | RAC1P | family:logistic | 0.025504 ± 0.005357 |
| validation | D initialized protected − C initialized task-only | RAC1P | family:logistic | 0.023841 ± 0.007145 |
| development evaluation | D initialized protected − C initialized task-only | RAC1P | family:mlp | 0.022542 ± 0.003754 |
| validation | D initialized protected − C initialized task-only | RAC1P | family:mlp | 0.020328 ± 0.008615 |
| development evaluation | D initialized protected − C initialized task-only | RAC1P | primary | 0.022542 ± 0.003754 |
| validation | D initialized protected − C initialized task-only | RAC1P | primary | 0.020328 ± 0.008615 |
| development evaluation | D initialized protected − C initialized task-only | SEX | catchup_inclusive | 0.011401 ± 0.007845 |
| validation | D initialized protected − C initialized task-only | SEX | catchup_inclusive | 0.009615 ± 0.003135 |
| development evaluation | D initialized protected − C initialized task-only | SEX | family:histgb | 0.000695 ± 0.002550 |
| validation | D initialized protected − C initialized task-only | SEX | family:histgb | 0.006203 ± 0.004703 |
| development evaluation | D initialized protected − C initialized task-only | SEX | family:logistic | 0.008917 ± 0.004359 |
| validation | D initialized protected − C initialized task-only | SEX | family:logistic | 0.011844 ± 0.003214 |
| development evaluation | D initialized protected − C initialized task-only | SEX | family:mlp | 0.007589 ± 0.002074 |
| validation | D initialized protected − C initialized task-only | SEX | family:mlp | 0.010380 ± 0.002371 |
| development evaluation | D initialized protected − C initialized task-only | SEX | primary | 0.007589 ± 0.002074 |
| validation | D initialized protected − C initialized task-only | SEX | primary | 0.010380 ± 0.002371 |
| development evaluation | D initialized protected − C initialized task-only | Civilian at work | family:logistic | -0.000033 ± 0.001425 |
| validation | D initialized protected − C initialized task-only | Civilian at work | family:logistic | 0.000209 ± 0.000178 |
| development evaluation | D initialized protected − C initialized task-only | Civilian at work | family:mlp | -0.000086 ± 0.001128 |
| validation | D initialized protected − C initialized task-only | Civilian at work | family:mlp | -0.000240 ± 0.000648 |
| development evaluation | D initialized protected − C initialized task-only | Civilian at work | primary | -0.000086 ± 0.001128 |
| validation | D initialized protected − C initialized task-only | Civilian at work | primary | -0.000240 ± 0.000648 |
| development evaluation | D initialized protected − C initialized task-only | Commute >20 min | family:logistic | 0.000078 ± 0.000424 |
| validation | D initialized protected − C initialized task-only | Commute >20 min | family:logistic | -0.000312 ± 0.000173 |
| development evaluation | D initialized protected − C initialized task-only | Commute >20 min | family:mlp | -0.000461 ± 0.001289 |
| validation | D initialized protected − C initialized task-only | Commute >20 min | family:mlp | -0.000085 ± 0.000974 |
| development evaluation | D initialized protected − C initialized task-only | Commute >20 min | primary | 0.000005 ± 0.000501 |
| validation | D initialized protected − C initialized task-only | Commute >20 min | primary | -0.000329 ± 0.000201 |
| development evaluation | D initialized protected − C initialized task-only | Income >$50k | family:logistic | 0.000342 ± 0.000810 |
| validation | D initialized protected − C initialized task-only | Income >$50k | family:logistic | 0.000014 ± 0.000298 |
| development evaluation | D initialized protected − C initialized task-only | Income >$50k | family:mlp | 0.000249 ± 0.000940 |
| validation | D initialized protected − C initialized task-only | Income >$50k | family:mlp | -0.000293 ± 0.001721 |
| development evaluation | D initialized protected − C initialized task-only | Income >$50k | primary | 0.001197 ± 0.000929 |
| validation | D initialized protected − C initialized task-only | Income >$50k | primary | 0.000500 ± 0.000753 |
| development evaluation | D initialized protected − C initialized task-only | Public coverage | family:logistic | -0.000454 ± 0.000449 |
| validation | D initialized protected − C initialized task-only | Public coverage | family:logistic | -0.001423 ± 0.001991 |
| development evaluation | D initialized protected − C initialized task-only | Public coverage | family:mlp | -0.001632 ± 0.001900 |
| validation | D initialized protected − C initialized task-only | Public coverage | family:mlp | -0.001216 ± 0.003782 |
| development evaluation | D initialized protected − C initialized task-only | Public coverage | primary | -0.000586 ± 0.000520 |
| validation | D initialized protected − C initialized task-only | Public coverage | primary | -0.001090 ± 0.002433 |
| development evaluation | D initialized protected − C initialized task-only | Same residence | family:logistic | -0.002186 ± 0.008752 |
| validation | D initialized protected − C initialized task-only | Same residence | family:logistic | 0.000456 ± 0.004124 |
| development evaluation | D initialized protected − C initialized task-only | Same residence | family:mlp | 0.002070 ± 0.002047 |
| validation | D initialized protected − C initialized task-only | Same residence | family:mlp | 0.001282 ± 0.000800 |
| development evaluation | D initialized protected − C initialized task-only | Same residence | primary | 0.002070 ± 0.002047 |
| validation | D initialized protected − C initialized task-only | Same residence | primary | 0.001282 ± 0.000800 |
| development evaluation | C initialized task-only − W after common warm-up | Civilian at work | family:logistic | 0.002338 ± 0.000560 |
| validation | C initialized task-only − W after common warm-up | Civilian at work | family:logistic | 0.003017 ± 0.000879 |
| development evaluation | C initialized task-only − W after common warm-up | Civilian at work | family:mlp | 0.001176 ± 0.001222 |
| validation | C initialized task-only − W after common warm-up | Civilian at work | family:mlp | 0.002490 ± 0.000691 |
| development evaluation | C initialized task-only − W after common warm-up | Civilian at work | primary | 0.000824 ± 0.001671 |
| validation | C initialized task-only − W after common warm-up | Civilian at work | primary | 0.002532 ± 0.000705 |
| development evaluation | C initialized task-only − W after common warm-up | Commute >20 min | family:logistic | 0.000519 ± 0.001731 |
| validation | C initialized task-only − W after common warm-up | Commute >20 min | family:logistic | 0.000888 ± 0.001633 |
| development evaluation | C initialized task-only − W after common warm-up | Commute >20 min | family:mlp | 0.001773 ± 0.001443 |
| validation | C initialized task-only − W after common warm-up | Commute >20 min | family:mlp | 0.000791 ± 0.000817 |
| development evaluation | C initialized task-only − W after common warm-up | Commute >20 min | primary | 0.001205 ± 0.002019 |
| validation | C initialized task-only − W after common warm-up | Commute >20 min | primary | 0.000413 ± 0.000860 |
| development evaluation | C initialized task-only − W after common warm-up | Income >$50k | family:logistic | 0.003441 ± 0.000959 |
| validation | C initialized task-only − W after common warm-up | Income >$50k | family:logistic | 0.001784 ± 0.000055 |
| development evaluation | C initialized task-only − W after common warm-up | Income >$50k | family:mlp | 0.002340 ± 0.001923 |
| validation | C initialized task-only − W after common warm-up | Income >$50k | family:mlp | 0.000916 ± 0.000133 |
| development evaluation | C initialized task-only − W after common warm-up | Income >$50k | primary | 0.003048 ± 0.001550 |
| validation | C initialized task-only − W after common warm-up | Income >$50k | primary | 0.001537 ± 0.000414 |
| development evaluation | C initialized task-only − W after common warm-up | Public coverage | family:logistic | 0.003118 ± 0.000710 |
| validation | C initialized task-only − W after common warm-up | Public coverage | family:logistic | 0.002945 ± 0.000956 |
| development evaluation | C initialized task-only − W after common warm-up | Public coverage | family:mlp | 0.001501 ± 0.001595 |
| validation | C initialized task-only − W after common warm-up | Public coverage | family:mlp | 0.002790 ± 0.001411 |
| development evaluation | C initialized task-only − W after common warm-up | Public coverage | primary | 0.001959 ± 0.001893 |
| validation | C initialized task-only − W after common warm-up | Public coverage | primary | 0.002774 ± 0.000705 |
| development evaluation | C initialized task-only − W after common warm-up | Same residence | family:logistic | 0.004620 ± 0.005612 |
| validation | C initialized task-only − W after common warm-up | Same residence | family:logistic | 0.006724 ± 0.003232 |
| development evaluation | C initialized task-only − W after common warm-up | Same residence | family:mlp | 0.001259 ± 0.005228 |
| validation | C initialized task-only − W after common warm-up | Same residence | family:mlp | 0.004629 ± 0.002596 |
| development evaluation | C initialized task-only − W after common warm-up | Same residence | primary | 0.001259 ± 0.005228 |
| validation | C initialized task-only − W after common warm-up | Same residence | primary | 0.004629 ± 0.002596 |
| development evaluation | C initialized task-only − Historical C task-only | RAC1P | catchup_inclusive | -0.008432 ± 0.016432 |
| validation | C initialized task-only − Historical C task-only | RAC1P | catchup_inclusive | -0.007757 ± 0.003804 |
| development evaluation | C initialized task-only − Historical C task-only | RAC1P | family:histgb | 0.000304 ± 0.019346 |
| validation | C initialized task-only − Historical C task-only | RAC1P | family:histgb | -0.002173 ± 0.016588 |
| development evaluation | C initialized task-only − Historical C task-only | RAC1P | family:logistic | -0.009125 ± 0.019753 |
| validation | C initialized task-only − Historical C task-only | RAC1P | family:logistic | -0.011707 ± 0.005624 |
| development evaluation | C initialized task-only − Historical C task-only | RAC1P | family:mlp | -0.010597 ± 0.012800 |
| validation | C initialized task-only − Historical C task-only | RAC1P | family:mlp | -0.008048 ± 0.003534 |
| development evaluation | C initialized task-only − Historical C task-only | RAC1P | primary | -0.010597 ± 0.012800 |
| validation | C initialized task-only − Historical C task-only | RAC1P | primary | -0.008048 ± 0.003534 |
| development evaluation | C initialized task-only − Historical C task-only | SEX | catchup_inclusive | -0.002679 ± 0.002527 |
| validation | C initialized task-only − Historical C task-only | SEX | catchup_inclusive | -0.007050 ± 0.002850 |
| development evaluation | C initialized task-only − Historical C task-only | SEX | family:histgb | -0.002333 ± 0.005465 |
| validation | C initialized task-only − Historical C task-only | SEX | family:histgb | -0.006683 ± 0.007778 |
| development evaluation | C initialized task-only − Historical C task-only | SEX | family:logistic | -0.013559 ± 0.014399 |
| validation | C initialized task-only − Historical C task-only | SEX | family:logistic | -0.013190 ± 0.011883 |
| development evaluation | C initialized task-only − Historical C task-only | SEX | family:mlp | -0.002679 ± 0.002527 |
| validation | C initialized task-only − Historical C task-only | SEX | family:mlp | -0.007050 ± 0.002850 |
| development evaluation | C initialized task-only − Historical C task-only | SEX | primary | -0.002679 ± 0.002527 |
| validation | C initialized task-only − Historical C task-only | SEX | primary | -0.007050 ± 0.002850 |
| development evaluation | C initialized task-only − Historical C task-only | Civilian at work | family:logistic | -0.002248 ± 0.001756 |
| validation | C initialized task-only − Historical C task-only | Civilian at work | family:logistic | -0.002645 ± 0.002191 |
| development evaluation | C initialized task-only − Historical C task-only | Civilian at work | family:mlp | -0.003142 ± 0.004488 |
| validation | C initialized task-only − Historical C task-only | Civilian at work | family:mlp | -0.002019 ± 0.001452 |
| development evaluation | C initialized task-only − Historical C task-only | Civilian at work | primary | -0.002785 ± 0.003890 |
| validation | C initialized task-only − Historical C task-only | Civilian at work | primary | -0.001652 ± 0.000974 |
| development evaluation | C initialized task-only − Historical C task-only | Commute >20 min | family:logistic | 0.001605 ± 0.003822 |
| validation | C initialized task-only − Historical C task-only | Commute >20 min | family:logistic | -0.002170 ± 0.000715 |
| development evaluation | C initialized task-only − Historical C task-only | Commute >20 min | family:mlp | 0.003080 ± 0.003802 |
| validation | C initialized task-only − Historical C task-only | Commute >20 min | family:mlp | 0.000543 ± 0.003421 |
| development evaluation | C initialized task-only − Historical C task-only | Commute >20 min | primary | 0.001516 ± 0.004067 |
| validation | C initialized task-only − Historical C task-only | Commute >20 min | primary | -0.001231 ± 0.000931 |
| development evaluation | C initialized task-only − Historical C task-only | Income >$50k | family:logistic | -0.000971 ± 0.001872 |
| validation | C initialized task-only − Historical C task-only | Income >$50k | family:logistic | 0.001053 ± 0.002212 |
| development evaluation | C initialized task-only − Historical C task-only | Income >$50k | family:mlp | -0.001587 ± 0.002511 |
| validation | C initialized task-only − Historical C task-only | Income >$50k | family:mlp | 0.000116 ± 0.004664 |
| development evaluation | C initialized task-only − Historical C task-only | Income >$50k | primary | -0.001826 ± 0.001925 |
| validation | C initialized task-only − Historical C task-only | Income >$50k | primary | 0.000567 ± 0.002755 |
| development evaluation | C initialized task-only − Historical C task-only | Public coverage | family:logistic | -0.000452 ± 0.000951 |
| validation | C initialized task-only − Historical C task-only | Public coverage | family:logistic | -0.003591 ± 0.001275 |
| development evaluation | C initialized task-only − Historical C task-only | Public coverage | family:mlp | -0.000454 ± 0.004867 |
| validation | C initialized task-only − Historical C task-only | Public coverage | family:mlp | -0.002960 ± 0.001847 |
| development evaluation | C initialized task-only − Historical C task-only | Public coverage | primary | -0.000035 ± 0.001067 |
| validation | C initialized task-only − Historical C task-only | Public coverage | primary | -0.003400 ± 0.000362 |
| development evaluation | C initialized task-only − Historical C task-only | Same residence | family:logistic | 0.003997 ± 0.005907 |
| validation | C initialized task-only − Historical C task-only | Same residence | family:logistic | -0.001270 ± 0.002401 |
| development evaluation | C initialized task-only − Historical C task-only | Same residence | family:mlp | -0.000805 ± 0.002869 |
| validation | C initialized task-only − Historical C task-only | Same residence | family:mlp | -0.001080 ± 0.003377 |
| development evaluation | C initialized task-only − Historical C task-only | Same residence | primary | -0.000805 ± 0.002869 |
| validation | C initialized task-only − Historical C task-only | Same residence | primary | -0.001080 ± 0.003377 |
| development evaluation | W after common warm-up − I initialized mapper | Civilian at work | family:logistic | -0.018412 ± 0.001429 |
| validation | W after common warm-up − I initialized mapper | Civilian at work | family:logistic | -0.022117 ± 0.002101 |
| development evaluation | W after common warm-up − I initialized mapper | Civilian at work | family:mlp | -0.014165 ± 0.001802 |
| validation | W after common warm-up − I initialized mapper | Civilian at work | family:mlp | -0.012295 ± 0.002827 |
| development evaluation | W after common warm-up − I initialized mapper | Civilian at work | primary | -0.013814 ± 0.002389 |
| validation | W after common warm-up − I initialized mapper | Civilian at work | primary | -0.012337 ± 0.002784 |
| development evaluation | W after common warm-up − I initialized mapper | Commute >20 min | family:logistic | -0.000050 ± 0.000372 |
| validation | W after common warm-up − I initialized mapper | Commute >20 min | family:logistic | -0.004469 ± 0.003662 |
| development evaluation | W after common warm-up − I initialized mapper | Commute >20 min | family:mlp | -0.000084 ± 0.000443 |
| validation | W after common warm-up − I initialized mapper | Commute >20 min | family:mlp | -0.004201 ± 0.002978 |
| development evaluation | W after common warm-up − I initialized mapper | Commute >20 min | primary | -0.000784 ± 0.001101 |
| validation | W after common warm-up − I initialized mapper | Commute >20 min | primary | -0.004204 ± 0.002941 |
| development evaluation | W after common warm-up − I initialized mapper | Income >$50k | family:logistic | -0.011378 ± 0.002138 |
| validation | W after common warm-up − I initialized mapper | Income >$50k | family:logistic | -0.018134 ± 0.005716 |
| development evaluation | W after common warm-up − I initialized mapper | Income >$50k | family:mlp | -0.004502 ± 0.005391 |
| validation | W after common warm-up − I initialized mapper | Income >$50k | family:mlp | -0.008368 ± 0.011417 |
| development evaluation | W after common warm-up − I initialized mapper | Income >$50k | primary | -0.007834 ± 0.002279 |
| validation | W after common warm-up − I initialized mapper | Income >$50k | primary | -0.012444 ± 0.008675 |
| development evaluation | W after common warm-up − I initialized mapper | Public coverage | family:logistic | -0.018287 ± 0.001710 |
| validation | W after common warm-up − I initialized mapper | Public coverage | family:logistic | -0.018883 ± 0.003017 |
| development evaluation | W after common warm-up − I initialized mapper | Public coverage | family:mlp | -0.008970 ± 0.003158 |
| validation | W after common warm-up − I initialized mapper | Public coverage | family:mlp | -0.011423 ± 0.004952 |
| development evaluation | W after common warm-up − I initialized mapper | Public coverage | primary | -0.012549 ± 0.005422 |
| validation | W after common warm-up − I initialized mapper | Public coverage | primary | -0.012614 ± 0.003691 |
| development evaluation | W after common warm-up − I initialized mapper | Same residence | family:logistic | 0.016831 ± 0.004793 |
| validation | W after common warm-up − I initialized mapper | Same residence | family:logistic | 0.013996 ± 0.008508 |
| development evaluation | W after common warm-up − I initialized mapper | Same residence | family:mlp | -0.001294 ± 0.003849 |
| validation | W after common warm-up − I initialized mapper | Same residence | family:mlp | -0.000296 ± 0.011160 |
| development evaluation | W after common warm-up − I initialized mapper | Same residence | primary | 0.003168 ± 0.002616 |
| validation | W after common warm-up − I initialized mapper | Same residence | primary | 0.001756 ± 0.012352 |

## Person-weighted sensitivity: primary paired log losses

PWGTP weights change scoring only; fitting and validation selection remain unweighted. These are sensitivity calculations, not official Census estimates or design-based uncertainty.

| Split | Left − right | Target | Log-loss difference |
| --- | --- | --- | --- |
| development evaluation person weighted | I initialized mapper − Frozen PCA16 | Civilian at work | -7.746e-11 ± 1.023e-09 |
| validation person weighted | I initialized mapper − Frozen PCA16 | Civilian at work | 9.226e-10 ± 4.542e-10 |
| development evaluation person weighted | I initialized mapper − Frozen PCA16 | Commute >20 min | 3.139e-10 ± 6.338e-10 |
| validation person weighted | I initialized mapper − Frozen PCA16 | Commute >20 min | 2.425e-10 ± 3.296e-10 |
| development evaluation person weighted | I initialized mapper − Frozen PCA16 | Income >$50k | -6.298e-13 ± 1.391e-09 |
| validation person weighted | I initialized mapper − Frozen PCA16 | Income >$50k | -3.010e-10 ± 1.011e-09 |
| development evaluation person weighted | I initialized mapper − Frozen PCA16 | Public coverage | 4.880e-10 ± 3.241e-10 |
| validation person weighted | I initialized mapper − Frozen PCA16 | Public coverage | 4.407e-10 ± 2.079e-09 |
| development evaluation person weighted | I initialized mapper − Frozen PCA16 | Same residence | 3.071e-11 ± 3.614e-10 |
| validation person weighted | I initialized mapper − Frozen PCA16 | Same residence | -1.667e-10 ± 3.519e-10 |
| development evaluation person weighted | D initialized protected − Historical D protected | RAC1P | 0.001197 ± 0.008660 |
| validation person weighted | D initialized protected − Historical D protected | RAC1P | 0.000566 ± 0.004827 |
| development evaluation person weighted | D initialized protected − Historical D protected | SEX | -0.001157 ± 0.005396 |
| validation person weighted | D initialized protected − Historical D protected | SEX | -0.001569 ± 0.008084 |
| development evaluation person weighted | D initialized protected − Historical D protected | Civilian at work | -0.002194 ± 0.002942 |
| validation person weighted | D initialized protected − Historical D protected | Civilian at work | -0.001161 ± 0.000669 |
| development evaluation person weighted | D initialized protected − Historical D protected | Commute >20 min | 0.001762 ± 0.003797 |
| validation person weighted | D initialized protected − Historical D protected | Commute >20 min | -0.000875 ± 0.002313 |
| development evaluation person weighted | D initialized protected − Historical D protected | Income >$50k | -0.001746 ± 0.002737 |
| validation person weighted | D initialized protected − Historical D protected | Income >$50k | -0.001379 ± 0.001235 |
| development evaluation person weighted | D initialized protected − Historical D protected | Public coverage | -0.002031 ± 0.005809 |
| validation person weighted | D initialized protected − Historical D protected | Public coverage | -0.002694 ± 0.002963 |
| development evaluation person weighted | D initialized protected − Historical D protected | Same residence | -0.001893 ± 0.003541 |
| validation person weighted | D initialized protected − Historical D protected | Same residence | -0.003629 ± 0.003813 |
| development evaluation person weighted | D initialized protected − C initialized task-only | RAC1P | 0.023056 ± 0.008633 |
| validation person weighted | D initialized protected − C initialized task-only | RAC1P | 0.022369 ± 0.003711 |
| development evaluation person weighted | D initialized protected − C initialized task-only | SEX | 0.008772 ± 0.002403 |
| validation person weighted | D initialized protected − C initialized task-only | SEX | 0.011953 ± 0.001541 |
| development evaluation person weighted | D initialized protected − C initialized task-only | Civilian at work | -0.000190 ± 0.000879 |
| validation person weighted | D initialized protected − C initialized task-only | Civilian at work | -0.000403 ± 0.000764 |
| development evaluation person weighted | D initialized protected − C initialized task-only | Commute >20 min | -0.000794 ± 0.000933 |
| validation person weighted | D initialized protected − C initialized task-only | Commute >20 min | -0.000652 ± 0.000904 |
| development evaluation person weighted | D initialized protected − C initialized task-only | Income >$50k | 0.000764 ± 0.000362 |
| validation person weighted | D initialized protected − C initialized task-only | Income >$50k | 0.000202 ± 0.000594 |
| development evaluation person weighted | D initialized protected − C initialized task-only | Public coverage | -0.000440 ± 0.000722 |
| validation person weighted | D initialized protected − C initialized task-only | Public coverage | 0.000039 ± 0.004284 |
| development evaluation person weighted | D initialized protected − C initialized task-only | Same residence | 0.001638 ± 0.002671 |
| validation person weighted | D initialized protected − C initialized task-only | Same residence | 0.000333 ± 0.002231 |
| development evaluation person weighted | C initialized task-only − W after common warm-up | Civilian at work | 0.001185 ± 0.001008 |
| validation person weighted | C initialized task-only − W after common warm-up | Civilian at work | 0.002106 ± 0.001022 |
| development evaluation person weighted | C initialized task-only − W after common warm-up | Commute >20 min | 0.001455 ± 0.002872 |
| validation person weighted | C initialized task-only − W after common warm-up | Commute >20 min | 0.000444 ± 0.000653 |
| development evaluation person weighted | C initialized task-only − W after common warm-up | Income >$50k | 0.003050 ± 0.002180 |
| validation person weighted | C initialized task-only − W after common warm-up | Income >$50k | 0.001256 ± 0.001623 |
| development evaluation person weighted | C initialized task-only − W after common warm-up | Public coverage | 0.003846 ± 0.002308 |
| validation person weighted | C initialized task-only − W after common warm-up | Public coverage | 0.002327 ± 0.000874 |
| development evaluation person weighted | C initialized task-only − W after common warm-up | Same residence | 0.000597 ± 0.004951 |
| validation person weighted | C initialized task-only − W after common warm-up | Same residence | 0.005792 ± 0.002570 |
| development evaluation person weighted | C initialized task-only − Historical C task-only | RAC1P | -0.012813 ± 0.013129 |
| validation person weighted | C initialized task-only − Historical C task-only | RAC1P | -0.007799 ± 0.001447 |
| development evaluation person weighted | C initialized task-only − Historical C task-only | SEX | -0.002529 ± 0.001589 |
| validation person weighted | C initialized task-only − Historical C task-only | SEX | -0.004378 ± 0.007810 |
| development evaluation person weighted | C initialized task-only − Historical C task-only | Civilian at work | -0.002295 ± 0.002846 |
| validation person weighted | C initialized task-only − Historical C task-only | Civilian at work | -0.001811 ± 0.001102 |
| development evaluation person weighted | C initialized task-only − Historical C task-only | Commute >20 min | 0.002756 ± 0.003819 |
| validation person weighted | C initialized task-only − Historical C task-only | Commute >20 min | -0.000806 ± 0.001076 |
| development evaluation person weighted | C initialized task-only − Historical C task-only | Income >$50k | -0.002374 ± 0.002306 |
| validation person weighted | C initialized task-only − Historical C task-only | Income >$50k | -0.001020 ± 0.001874 |
| development evaluation person weighted | C initialized task-only − Historical C task-only | Public coverage | -0.001674 ± 0.001204 |
| validation person weighted | C initialized task-only − Historical C task-only | Public coverage | -0.005327 ± 0.003892 |
| development evaluation person weighted | C initialized task-only − Historical C task-only | Same residence | -0.002492 ± 0.007906 |
| validation person weighted | C initialized task-only − Historical C task-only | Same residence | -0.001982 ± 0.001633 |
| development evaluation person weighted | W after common warm-up − I initialized mapper | Civilian at work | -0.010886 ± 0.003542 |
| validation person weighted | W after common warm-up − I initialized mapper | Civilian at work | -0.010182 ± 0.003007 |
| development evaluation person weighted | W after common warm-up − I initialized mapper | Commute >20 min | -0.000061 ± 0.000804 |
| validation person weighted | W after common warm-up − I initialized mapper | Commute >20 min | -0.001266 ± 0.001776 |
| development evaluation person weighted | W after common warm-up − I initialized mapper | Income >$50k | -0.006601 ± 0.001191 |
| validation person weighted | W after common warm-up − I initialized mapper | Income >$50k | -0.015176 ± 0.009201 |
| development evaluation person weighted | W after common warm-up − I initialized mapper | Public coverage | -0.012394 ± 0.004097 |
| validation person weighted | W after common warm-up − I initialized mapper | Public coverage | -0.009833 ± 0.001014 |
| development evaluation person weighted | W after common warm-up − I initialized mapper | Same residence | -0.002566 ± 0.004596 |
| validation person weighted | W after common warm-up − I initialized mapper | Same residence | -0.000246 ± 0.014847 |

## Signed attribute gains

Gain is fitting-prior log loss minus attack log loss, paired per seed before aggregation. Negative gains remain negative. I/W are not audited. A scalar gain does not resolve missing category support.

| Release | Selector | Validation SEX | Validation RAC1P | Development SEX | Development RAC1P |
| --- | --- | --- | --- | --- | --- |
| Frozen PCA16 | primary | 0.034651 ± 0.006881 | 0.073028 ± 0.015524 | 0.032739 ± 0.003939 | 0.077966 ± 0.006712 |
| C initialized task-only | primary | 0.034526 ± 0.006579 | 0.064657 ± 0.012771 | 0.036828 ± 0.007129 | 0.071999 ± 0.004757 |
| C initialized task-only | catchup_inclusive | 0.034526 ± 0.006579 | 0.065678 ± 0.011056 | 0.036828 ± 0.007129 | 0.068193 ± 0.011342 |
| D initialized protected | primary | 0.024146 ± 0.005236 | 0.044329 ± 0.004361 | 0.029239 ± 0.005862 | 0.049457 ± 0.004490 |
| D initialized protected | catchup_inclusive | 0.024910 ± 0.006209 | 0.056210 ± 0.010670 | 0.025427 ± 0.000740 | 0.057866 ± 0.007308 |
| Historical C task-only | primary | 0.027475 ± 0.004820 | 0.056610 ± 0.010734 | 0.034149 ± 0.006296 | 0.061402 ± 0.008499 |
| Historical C task-only | catchup_inclusive | 0.027475 ± 0.004820 | 0.057920 ± 0.008464 | 0.034149 ± 0.006296 | 0.059761 ± 0.006122 |
| Historical D protected | primary | 0.019373 ± 0.003973 | 0.044494 ± 0.015327 | 0.025596 ± 0.007966 | 0.049338 ± 0.007737 |
| Historical D protected | catchup_inclusive | 0.019614 ± 0.004091 | 0.053769 ± 0.010122 | 0.024071 ± 0.006807 | 0.057181 ± 0.007663 |
| Original PCA32 | primary | 0.037585 ± 0.006196 | 0.074665 ± 0.013280 | 0.037469 ± 0.010753 | 0.089644 ± 0.003041 |
| PCA32 + LEACE | primary | 0.018733 ± 0.008196 | 0.042992 ± 0.019004 | 0.014566 ± 0.007457 | 0.054996 ± 0.001780 |

## Final-stage audit coverage

The fixed two-class SEX/nine-class RAC1P schema, actual attacker-fit support, validation/evaluation support and validation-selected exposed-control recalls are required. Missing race categories remain limitations.

| Seed | Split | Release | Selector | Target | Candidate | Complete | Limitations (Census codes) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | C initialized task-only | primary | SEX | mlp_1 | True | {} |
| 0 | validation | C initialized task-only | primary | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | C initialized task-only | catchup_inclusive | SEX | mlp_1 | True | {} |
| 0 | validation | C initialized task-only | catchup_inclusive | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | D initialized protected | primary | SEX | mlp_1 | True | {} |
| 0 | validation | D initialized protected | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | D initialized protected | catchup_inclusive | SEX | mlp_1 | True | {} |
| 0 | validation | D initialized protected | catchup_inclusive | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | C initialized task-only | primary | SEX | mlp_1 | True | {} |
| 0 | development evaluation | C initialized task-only | primary | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | C initialized task-only | catchup_inclusive | SEX | mlp_1 | True | {} |
| 0 | development evaluation | C initialized task-only | catchup_inclusive | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | D initialized protected | primary | SEX | mlp_1 | True | {} |
| 0 | development evaluation | D initialized protected | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | D initialized protected | catchup_inclusive | SEX | mlp_1 | True | {} |
| 0 | development evaluation | D initialized protected | catchup_inclusive | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | C initialized task-only | primary | SEX | mlp_0 | True | {} |
| 1 | validation | C initialized task-only | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | C initialized task-only | catchup_inclusive | SEX | mlp_0 | True | {} |
| 1 | validation | C initialized task-only | catchup_inclusive | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | D initialized protected | primary | SEX | mlp_0 | True | {} |
| 1 | validation | D initialized protected | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | D initialized protected | catchup_inclusive | SEX | catchup | True | {} |
| 1 | validation | D initialized protected | catchup_inclusive | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | C initialized task-only | primary | SEX | mlp_0 | True | {} |
| 1 | development evaluation | C initialized task-only | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | C initialized task-only | catchup_inclusive | SEX | mlp_0 | True | {} |
| 1 | development evaluation | C initialized task-only | catchup_inclusive | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | D initialized protected | primary | SEX | mlp_0 | True | {} |
| 1 | development evaluation | D initialized protected | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | D initialized protected | catchup_inclusive | SEX | catchup | True | {} |
| 1 | development evaluation | D initialized protected | catchup_inclusive | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | C initialized task-only | primary | SEX | mlp_0 | True | {} |
| 2 | validation | C initialized task-only | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | C initialized task-only | catchup_inclusive | SEX | mlp_0 | True | {} |
| 2 | validation | C initialized task-only | catchup_inclusive | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | D initialized protected | primary | SEX | mlp_1 | True | {} |
| 2 | validation | D initialized protected | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | D initialized protected | catchup_inclusive | SEX | mlp_1 | True | {} |
| 2 | validation | D initialized protected | catchup_inclusive | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | C initialized task-only | primary | SEX | mlp_0 | True | {} |
| 2 | development evaluation | C initialized task-only | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | C initialized task-only | catchup_inclusive | SEX | mlp_0 | True | {} |
| 2 | development evaluation | C initialized task-only | catchup_inclusive | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | D initialized protected | primary | SEX | mlp_1 | True | {} |
| 2 | development evaluation | D initialized protected | primary | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | D initialized protected | catchup_inclusive | SEX | mlp_1 | True | {} |
| 2 | development evaluation | D initialized protected | catchup_inclusive | RAC1P | catchup | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |

## Attribute halving versus original PCA32

Signed gain is reported even when a coverage limit prevents fractional interpretation or a policy pass. Numeric halvings are shown separately from coverage-gated assessments.

| Seed | Split | Final release | Selector | Attribute | Parent gain | Final gain | Numeric halving | Coverage-gated halving |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | C initialized task-only | primary | SEX | 0.041923 | 0.039052 | fail | fail |
| 0 | validation | C initialized task-only | primary | RAC1P | 0.061715 | 0.050283 | fail | undefined |
| 0 | validation | D initialized protected | primary | SEX | 0.041923 | 0.025955 | fail | fail |
| 0 | validation | D initialized protected | primary | RAC1P | 0.061715 | 0.039296 | fail | undefined |
| 0 | validation | C initialized task-only | catchup_inclusive | SEX | 0.041923 | 0.039052 | fail | fail |
| 0 | validation | C initialized task-only | catchup_inclusive | RAC1P | 0.061715 | 0.053343 | fail | undefined |
| 0 | validation | D initialized protected | catchup_inclusive | SEX | 0.041923 | 0.025955 | fail | fail |
| 0 | validation | D initialized protected | catchup_inclusive | RAC1P | 0.061715 | 0.052174 | fail | undefined |
| 0 | development evaluation | C initialized task-only | primary | SEX | 0.033326 | 0.034655 | fail | fail |
| 0 | development evaluation | C initialized task-only | primary | RAC1P | 0.088982 | 0.066519 | fail | undefined |
| 0 | development evaluation | D initialized protected | primary | SEX | 0.033326 | 0.025864 | fail | fail |
| 0 | development evaluation | D initialized protected | primary | RAC1P | 0.088982 | 0.046241 | fail | undefined |
| 0 | development evaluation | C initialized task-only | catchup_inclusive | SEX | 0.033326 | 0.034655 | fail | fail |
| 0 | development evaluation | C initialized task-only | catchup_inclusive | RAC1P | 0.088982 | 0.055101 | fail | undefined |
| 0 | development evaluation | D initialized protected | catchup_inclusive | SEX | 0.033326 | 0.025864 | fail | fail |
| 0 | development evaluation | D initialized protected | catchup_inclusive | RAC1P | 0.088982 | 0.065755 | fail | undefined |
| 1 | validation | C initialized task-only | primary | SEX | 0.040342 | 0.037546 | fail | fail |
| 1 | validation | C initialized task-only | primary | RAC1P | 0.074029 | 0.068993 | fail | undefined |
| 1 | validation | D initialized protected | primary | SEX | 0.040342 | 0.028237 | fail | fail |
| 1 | validation | D initialized protected | primary | RAC1P | 0.074029 | 0.046959 | fail | undefined |
| 1 | validation | C initialized task-only | catchup_inclusive | SEX | 0.040342 | 0.037546 | fail | fail |
| 1 | validation | C initialized task-only | catchup_inclusive | RAC1P | 0.074029 | 0.068993 | fail | undefined |
| 1 | validation | D initialized protected | catchup_inclusive | SEX | 0.040342 | 0.030530 | fail | fail |
| 1 | validation | D initialized protected | catchup_inclusive | RAC1P | 0.074029 | 0.048147 | fail | undefined |
| 1 | development evaluation | C initialized task-only | primary | SEX | 0.049677 | 0.044791 | fail | fail |
| 1 | development evaluation | C initialized task-only | primary | RAC1P | 0.092961 | 0.075060 | fail | undefined |
| 1 | development evaluation | D initialized protected | primary | SEX | 0.049677 | 0.036008 | fail | fail |
| 1 | development evaluation | D initialized protected | primary | RAC1P | 0.092961 | 0.054587 | fail | undefined |
| 1 | development evaluation | C initialized task-only | catchup_inclusive | SEX | 0.049677 | 0.044791 | fail | fail |
| 1 | development evaluation | C initialized task-only | catchup_inclusive | RAC1P | 0.092961 | 0.075060 | fail | undefined |
| 1 | development evaluation | D initialized protected | catchup_inclusive | SEX | 0.049677 | 0.024573 | pass | pass |
| 1 | development evaluation | D initialized protected | catchup_inclusive | RAC1P | 0.092961 | 0.056516 | fail | undefined |
| 2 | validation | C initialized task-only | primary | SEX | 0.030488 | 0.026979 | fail | fail |
| 2 | validation | C initialized task-only | primary | RAC1P | 0.088252 | 0.074696 | fail | undefined |
| 2 | validation | D initialized protected | primary | SEX | 0.030488 | 0.018245 | fail | fail |
| 2 | validation | D initialized protected | primary | RAC1P | 0.088252 | 0.046734 | fail | undefined |
| 2 | validation | C initialized task-only | catchup_inclusive | SEX | 0.030488 | 0.026979 | fail | fail |
| 2 | validation | C initialized task-only | catchup_inclusive | RAC1P | 0.088252 | 0.074696 | fail | undefined |
| 2 | validation | D initialized protected | catchup_inclusive | SEX | 0.030488 | 0.018245 | fail | fail |
| 2 | validation | D initialized protected | catchup_inclusive | RAC1P | 0.088252 | 0.068308 | fail | undefined |
| 2 | development evaluation | C initialized task-only | primary | SEX | 0.029404 | 0.031038 | fail | fail |
| 2 | development evaluation | C initialized task-only | primary | RAC1P | 0.086989 | 0.074418 | fail | undefined |
| 2 | development evaluation | D initialized protected | primary | SEX | 0.029404 | 0.025844 | fail | fail |
| 2 | development evaluation | D initialized protected | primary | RAC1P | 0.086989 | 0.047543 | fail | undefined |
| 2 | development evaluation | C initialized task-only | catchup_inclusive | SEX | 0.029404 | 0.031038 | fail | fail |
| 2 | development evaluation | C initialized task-only | catchup_inclusive | RAC1P | 0.086989 | 0.074418 | fail | undefined |
| 2 | development evaluation | D initialized protected | catchup_inclusive | SEX | 0.029404 | 0.025844 | fail | fail |
| 2 | development evaluation | D initialized protected | catchup_inclusive | RAC1P | 0.086989 | 0.051328 | fail | undefined |

## Final releases versus all four bank references

Residence must be at least .01 nats lower while each attribute gain is at most .005 nats higher. Numeric inequalities are published even when race coverage limits the joint interpretation. These descriptive references do not select or reject future method work.

| Seed | Split | Final release | Bank | Selector | Residence Δ | Utility | SEX numeric | RAC1P numeric | Numeric joint | Joint with coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | C initialized task-only | Rich neural bank | primary | -0.005589 | fail | pass | fail | fail | fail |
| 0 | validation | C initialized task-only | Neural bank + LEACE | primary | -0.014331 | pass | fail | fail | fail | fail |
| 0 | validation | C initialized task-only | Rich tree bank | primary | -0.015108 | pass | fail | fail | fail | fail |
| 0 | validation | C initialized task-only | Tree bank + LEACE | primary | -0.026526 | pass | fail | fail | fail | fail |
| 0 | validation | D initialized protected | Rich neural bank | primary | -0.005002 | fail | pass | pass | fail | fail |
| 0 | validation | D initialized protected | Neural bank + LEACE | primary | -0.013744 | pass | pass | fail | fail | undefined |
| 0 | validation | D initialized protected | Rich tree bank | primary | -0.014521 | pass | pass | fail | fail | undefined |
| 0 | validation | D initialized protected | Tree bank + LEACE | primary | -0.025939 | pass | fail | fail | fail | fail |
| 0 | validation | C initialized task-only | Rich neural bank | catchup_inclusive | -0.005589 | fail | pass | fail | fail | fail |
| 0 | validation | C initialized task-only | Neural bank + LEACE | catchup_inclusive | -0.014331 | pass | fail | fail | fail | fail |
| 0 | validation | C initialized task-only | Rich tree bank | catchup_inclusive | -0.015108 | pass | fail | fail | fail | fail |
| 0 | validation | C initialized task-only | Tree bank + LEACE | catchup_inclusive | -0.026526 | pass | fail | fail | fail | fail |
| 0 | validation | D initialized protected | Rich neural bank | catchup_inclusive | -0.005002 | fail | pass | fail | fail | fail |
| 0 | validation | D initialized protected | Neural bank + LEACE | catchup_inclusive | -0.013744 | pass | pass | fail | fail | undefined |
| 0 | validation | D initialized protected | Rich tree bank | catchup_inclusive | -0.014521 | pass | pass | fail | fail | undefined |
| 0 | validation | D initialized protected | Tree bank + LEACE | catchup_inclusive | -0.025939 | pass | fail | fail | fail | fail |
| 0 | development evaluation | C initialized task-only | Rich neural bank | primary | -0.010900 | pass | fail | fail | fail | fail |
| 0 | development evaluation | C initialized task-only | Neural bank + LEACE | primary | -0.022302 | pass | fail | fail | fail | fail |
| 0 | development evaluation | C initialized task-only | Rich tree bank | primary | -0.018379 | pass | fail | fail | fail | fail |
| 0 | development evaluation | C initialized task-only | Tree bank + LEACE | primary | -0.028141 | pass | fail | fail | fail | fail |
| 0 | development evaluation | D initialized protected | Rich neural bank | primary | -0.010371 | pass | fail | fail | fail | fail |
| 0 | development evaluation | D initialized protected | Neural bank + LEACE | primary | -0.021774 | pass | fail | fail | fail | fail |
| 0 | development evaluation | D initialized protected | Rich tree bank | primary | -0.017850 | pass | fail | fail | fail | fail |
| 0 | development evaluation | D initialized protected | Tree bank + LEACE | primary | -0.027612 | pass | fail | fail | fail | fail |
| 0 | development evaluation | C initialized task-only | Rich neural bank | catchup_inclusive | -0.010900 | pass | fail | fail | fail | fail |
| 0 | development evaluation | C initialized task-only | Neural bank + LEACE | catchup_inclusive | -0.022302 | pass | fail | fail | fail | fail |
| 0 | development evaluation | C initialized task-only | Rich tree bank | catchup_inclusive | -0.018379 | pass | fail | fail | fail | fail |
| 0 | development evaluation | C initialized task-only | Tree bank + LEACE | catchup_inclusive | -0.028141 | pass | fail | fail | fail | fail |
| 0 | development evaluation | D initialized protected | Rich neural bank | catchup_inclusive | -0.010371 | pass | fail | fail | fail | fail |
| 0 | development evaluation | D initialized protected | Neural bank + LEACE | catchup_inclusive | -0.021774 | pass | fail | fail | fail | fail |
| 0 | development evaluation | D initialized protected | Rich tree bank | catchup_inclusive | -0.017850 | pass | fail | fail | fail | fail |
| 0 | development evaluation | D initialized protected | Tree bank + LEACE | catchup_inclusive | -0.027612 | pass | fail | fail | fail | fail |
| 1 | validation | C initialized task-only | Rich neural bank | primary | -0.012296 | pass | fail | fail | fail | fail |
| 1 | validation | C initialized task-only | Neural bank + LEACE | primary | -0.011101 | pass | fail | fail | fail | fail |
| 1 | validation | C initialized task-only | Rich tree bank | primary | -0.015775 | pass | fail | fail | fail | fail |
| 1 | validation | C initialized task-only | Tree bank + LEACE | primary | -0.025224 | pass | fail | fail | fail | fail |
| 1 | validation | D initialized protected | Rich neural bank | primary | -0.010139 | pass | pass | pass | pass | undefined |
| 1 | validation | D initialized protected | Neural bank + LEACE | primary | -0.008944 | fail | pass | fail | fail | fail |
| 1 | validation | D initialized protected | Rich tree bank | primary | -0.013618 | pass | fail | fail | fail | fail |
| 1 | validation | D initialized protected | Tree bank + LEACE | primary | -0.023067 | pass | fail | fail | fail | fail |
| 1 | validation | C initialized task-only | Rich neural bank | catchup_inclusive | -0.012296 | pass | fail | fail | fail | fail |
| 1 | validation | C initialized task-only | Neural bank + LEACE | catchup_inclusive | -0.011101 | pass | fail | fail | fail | fail |
| 1 | validation | C initialized task-only | Rich tree bank | catchup_inclusive | -0.015775 | pass | fail | fail | fail | fail |
| 1 | validation | C initialized task-only | Tree bank + LEACE | catchup_inclusive | -0.025224 | pass | fail | fail | fail | fail |
| 1 | validation | D initialized protected | Rich neural bank | catchup_inclusive | -0.010139 | pass | pass | pass | pass | undefined |
| 1 | validation | D initialized protected | Neural bank + LEACE | catchup_inclusive | -0.008944 | fail | fail | fail | fail | fail |
| 1 | validation | D initialized protected | Rich tree bank | catchup_inclusive | -0.013618 | pass | fail | fail | fail | fail |
| 1 | validation | D initialized protected | Tree bank + LEACE | catchup_inclusive | -0.023067 | pass | fail | fail | fail | fail |
| 1 | development evaluation | C initialized task-only | Rich neural bank | primary | -0.014281 | pass | fail | fail | fail | fail |
| 1 | development evaluation | C initialized task-only | Neural bank + LEACE | primary | -0.015383 | pass | fail | fail | fail | fail |
| 1 | development evaluation | C initialized task-only | Rich tree bank | primary | -0.015496 | pass | fail | fail | fail | fail |
| 1 | development evaluation | C initialized task-only | Tree bank + LEACE | primary | -0.036015 | pass | fail | fail | fail | fail |
| 1 | development evaluation | D initialized protected | Rich neural bank | primary | -0.012993 | pass | pass | fail | fail | undefined |
| 1 | development evaluation | D initialized protected | Neural bank + LEACE | primary | -0.014095 | pass | fail | fail | fail | fail |
| 1 | development evaluation | D initialized protected | Rich tree bank | primary | -0.014208 | pass | fail | fail | fail | fail |
| 1 | development evaluation | D initialized protected | Tree bank + LEACE | primary | -0.034726 | pass | fail | fail | fail | fail |
| 1 | development evaluation | C initialized task-only | Rich neural bank | catchup_inclusive | -0.014281 | pass | fail | fail | fail | fail |
| 1 | development evaluation | C initialized task-only | Neural bank + LEACE | catchup_inclusive | -0.015383 | pass | fail | fail | fail | fail |
| 1 | development evaluation | C initialized task-only | Rich tree bank | catchup_inclusive | -0.015496 | pass | fail | fail | fail | fail |
| 1 | development evaluation | C initialized task-only | Tree bank + LEACE | catchup_inclusive | -0.036015 | pass | fail | fail | fail | fail |
| 1 | development evaluation | D initialized protected | Rich neural bank | catchup_inclusive | -0.012993 | pass | pass | fail | fail | undefined |
| 1 | development evaluation | D initialized protected | Neural bank + LEACE | catchup_inclusive | -0.014095 | pass | pass | fail | fail | undefined |
| 1 | development evaluation | D initialized protected | Rich tree bank | catchup_inclusive | -0.014208 | pass | fail | fail | fail | fail |
| 1 | development evaluation | D initialized protected | Tree bank + LEACE | catchup_inclusive | -0.034726 | pass | fail | fail | fail | fail |
| 2 | validation | C initialized task-only | Rich neural bank | primary | -0.002462 | fail | pass | fail | fail | fail |
| 2 | validation | C initialized task-only | Neural bank + LEACE | primary | -0.010051 | pass | fail | fail | fail | fail |
| 2 | validation | C initialized task-only | Rich tree bank | primary | -0.014308 | pass | fail | fail | fail | fail |
| 2 | validation | C initialized task-only | Tree bank + LEACE | primary | -0.021365 | pass | fail | fail | fail | fail |
| 2 | validation | D initialized protected | Rich neural bank | primary | -0.001358 | fail | pass | pass | fail | fail |
| 2 | validation | D initialized protected | Neural bank + LEACE | primary | -0.008947 | fail | pass | fail | fail | fail |
| 2 | validation | D initialized protected | Rich tree bank | primary | -0.013205 | pass | fail | fail | fail | fail |
| 2 | validation | D initialized protected | Tree bank + LEACE | primary | -0.020262 | pass | fail | fail | fail | fail |
| 2 | validation | C initialized task-only | Rich neural bank | catchup_inclusive | -0.002462 | fail | pass | fail | fail | fail |
| 2 | validation | C initialized task-only | Neural bank + LEACE | catchup_inclusive | -0.010051 | pass | fail | fail | fail | fail |
| 2 | validation | C initialized task-only | Rich tree bank | catchup_inclusive | -0.014308 | pass | fail | fail | fail | fail |
| 2 | validation | C initialized task-only | Tree bank + LEACE | catchup_inclusive | -0.021365 | pass | fail | fail | fail | fail |
| 2 | validation | D initialized protected | Rich neural bank | catchup_inclusive | -0.001358 | fail | pass | fail | fail | fail |
| 2 | validation | D initialized protected | Neural bank + LEACE | catchup_inclusive | -0.008947 | fail | pass | fail | fail | fail |
| 2 | validation | D initialized protected | Rich tree bank | catchup_inclusive | -0.013205 | pass | fail | fail | fail | fail |
| 2 | validation | D initialized protected | Tree bank + LEACE | catchup_inclusive | -0.020262 | pass | fail | fail | fail | fail |
| 2 | development evaluation | C initialized task-only | Rich neural bank | primary | -0.007936 | fail | pass | fail | fail | fail |
| 2 | development evaluation | C initialized task-only | Neural bank + LEACE | primary | -0.009586 | fail | fail | fail | fail | fail |
| 2 | development evaluation | C initialized task-only | Rich tree bank | primary | -0.019724 | pass | fail | fail | fail | fail |
| 2 | development evaluation | C initialized task-only | Tree bank + LEACE | primary | -0.034803 | pass | fail | fail | fail | fail |
| 2 | development evaluation | D initialized protected | Rich neural bank | primary | -0.003545 | fail | pass | pass | fail | fail |
| 2 | development evaluation | D initialized protected | Neural bank + LEACE | primary | -0.005194 | fail | pass | fail | fail | fail |
| 2 | development evaluation | D initialized protected | Rich tree bank | primary | -0.015332 | pass | fail | fail | fail | fail |
| 2 | development evaluation | D initialized protected | Tree bank + LEACE | primary | -0.030411 | pass | fail | fail | fail | fail |
| 2 | development evaluation | C initialized task-only | Rich neural bank | catchup_inclusive | -0.007936 | fail | pass | fail | fail | fail |
| 2 | development evaluation | C initialized task-only | Neural bank + LEACE | catchup_inclusive | -0.009586 | fail | fail | fail | fail | fail |
| 2 | development evaluation | C initialized task-only | Rich tree bank | catchup_inclusive | -0.019724 | pass | fail | fail | fail | fail |
| 2 | development evaluation | C initialized task-only | Tree bank + LEACE | catchup_inclusive | -0.034803 | pass | fail | fail | fail | fail |
| 2 | development evaluation | D initialized protected | Rich neural bank | catchup_inclusive | -0.003545 | fail | pass | pass | fail | fail |
| 2 | development evaluation | D initialized protected | Neural bank + LEACE | catchup_inclusive | -0.005194 | fail | pass | fail | fail | fail |
| 2 | development evaluation | D initialized protected | Rich tree bank | catchup_inclusive | -0.015332 | pass | fail | fail | fail | fail |
| 2 | development evaluation | D initialized protected | Tree bank + LEACE | catchup_inclusive | -0.030411 | pass | fail | fail | fail | fail |

## New-stage classification metrics

Full-schema balanced accuracy/macro AUROC remain undefined if a category is unsupported. Per-class and observed-class alternatives are explicitly named in CSV.

| Split | Stage | Target | Log loss | Accuracy | Balanced accuracy | AUROC / macro |
| --- | --- | --- | --- | --- | --- | --- |
| validation | I initialized mapper | Same residence | 0.505908 ± 0.006331 | 0.765377 ± 0.006931 | 0.524877 ± 0.003238 | 0.687057 ± 0.020849 |
| validation | I initialized mapper | Commute >20 min | 0.678627 ± 0.002608 | 0.569964 ± 0.002190 | 0.569832 ± 0.001477 | 0.597043 ± 0.005137 |
| validation | I initialized mapper | Income >$50k | 0.306442 ± 0.003365 | 0.858328 ± 0.005747 | 0.739622 ± 0.003007 | 0.893809 ± 0.004309 |
| validation | I initialized mapper | Civilian at work | 0.302483 ± 0.007853 | 0.890460 ± 0.004102 | 0.836909 ± 0.009167 | 0.889179 ± 0.007319 |
| validation | I initialized mapper | Public coverage | 0.489936 ± 0.002588 | 0.775890 ± 0.003880 | 0.585520 ± 0.011420 | 0.727345 ± 0.015712 |
| validation | W after common warm-up | Same residence | 0.507664 ± 0.013582 | 0.764577 ± 0.013812 | 0.544640 ± 0.023126 | 0.687117 ± 0.002045 |
| validation | W after common warm-up | Commute >20 min | 0.674423 ± 0.000333 | 0.568819 ± 0.002641 | 0.568633 ± 0.001401 | 0.601007 ± 0.003445 |
| validation | W after common warm-up | Income >$50k | 0.293998 ± 0.005360 | 0.864413 ± 0.002223 | 0.747254 ± 0.001135 | 0.902329 ± 0.001859 |
| validation | W after common warm-up | Civilian at work | 0.290146 ± 0.008622 | 0.895322 ± 0.003985 | 0.843732 ± 0.008864 | 0.895949 ± 0.007679 |
| validation | W after common warm-up | Public coverage | 0.477323 ± 0.005013 | 0.783524 ± 0.009329 | 0.598406 ± 0.011674 | 0.744113 ± 0.009587 |
| validation | C initialized task-only | Same residence | 0.512293 ± 0.012022 | 0.764263 ± 0.013212 | 0.545917 ± 0.033258 | 0.679586 ± 0.005792 |
| validation | C initialized task-only | Commute >20 min | 0.674835 ± 0.001025 | 0.567884 ± 0.004038 | 0.567791 ± 0.004356 | 0.601456 ± 0.002826 |
| validation | C initialized task-only | Income >$50k | 0.295535 ± 0.005603 | 0.864741 ± 0.002591 | 0.742173 ± 0.002318 | 0.901689 ± 0.002072 |
| validation | C initialized task-only | Civilian at work | 0.292678 ± 0.007942 | 0.894999 ± 0.002848 | 0.843497 ± 0.008334 | 0.893444 ± 0.007656 |
| validation | C initialized task-only | Public coverage | 0.480097 ± 0.004741 | 0.781348 ± 0.010284 | 0.587760 ± 0.013003 | 0.739984 ± 0.010520 |
| validation | C initialized task-only | SEX | 0.657862 ± 0.006334 | 0.597217 ± 0.015187 | 0.595348 ± 0.016254 | 0.643349 ± 0.013848 |
| validation | C initialized task-only | RAC1P | 1.218840 ± 0.008378 | 0.592728 ± 0.003414 | undefined (0/3) | undefined (0/3) |
| validation | D initialized protected | Same residence | 0.513576 ± 0.011465 | 0.758947 ± 0.014381 | 0.538028 ± 0.025378 | 0.677514 ± 0.006740 |
| validation | D initialized protected | Commute >20 min | 0.674507 ± 0.000928 | 0.570380 ± 0.010007 | 0.570257 ± 0.009292 | 0.603249 ± 0.003616 |
| validation | D initialized protected | Income >$50k | 0.296035 ± 0.005526 | 0.862552 ± 0.004884 | 0.739852 ± 0.005122 | 0.901058 ± 0.002539 |
| validation | D initialized protected | Civilian at work | 0.292438 ± 0.007575 | 0.894784 ± 0.001775 | 0.843189 ± 0.006731 | 0.893181 ± 0.007011 |
| validation | D initialized protected | Public coverage | 0.479007 ± 0.006864 | 0.781344 ± 0.009435 | 0.588626 ± 0.010937 | 0.743195 ± 0.011072 |
| validation | D initialized protected | SEX | 0.668242 ± 0.005119 | 0.585271 ± 0.009028 | 0.583319 ± 0.009859 | 0.622804 ± 0.008559 |
| validation | D initialized protected | RAC1P | 1.239168 ± 0.002268 | 0.583213 ± 0.006015 | undefined (0/3) | undefined (0/3) |
| development evaluation | I initialized mapper | Same residence | 0.495062 ± 0.002095 | 0.773643 ± 0.005612 | 0.527815 ± 0.009284 | 0.695676 ± 0.013098 |
| development evaluation | I initialized mapper | Commute >20 min | 0.685527 ± 0.003931 | 0.552990 ± 0.011693 | 0.551746 ± 0.011898 | 0.575694 ± 0.007983 |
| development evaluation | I initialized mapper | Income >$50k | 0.294408 ± 0.007520 | 0.867742 ± 0.003666 | 0.747240 ± 0.010486 | 0.902085 ± 0.007348 |
| development evaluation | I initialized mapper | Civilian at work | 0.290035 ± 0.007223 | 0.895477 ± 0.008613 | 0.848779 ± 0.013394 | 0.901041 ± 0.001751 |
| development evaluation | I initialized mapper | Public coverage | 0.505704 ± 0.008631 | 0.762444 ± 0.011110 | 0.576967 ± 0.014823 | 0.720378 ± 0.003550 |
| development evaluation | W after common warm-up | Same residence | 0.498230 ± 0.004660 | 0.773283 ± 0.002443 | 0.546921 ± 0.026226 | 0.691954 ± 0.015998 |
| development evaluation | W after common warm-up | Commute >20 min | 0.684743 ± 0.004745 | 0.554525 ± 0.010935 | 0.552042 ± 0.011104 | 0.574441 ± 0.012927 |
| development evaluation | W after common warm-up | Income >$50k | 0.286573 ± 0.009795 | 0.871305 ± 0.004356 | 0.748474 ± 0.018096 | 0.907430 ± 0.008417 |
| development evaluation | W after common warm-up | Civilian at work | 0.276222 ± 0.005653 | 0.902400 ± 0.006326 | 0.858303 ± 0.008681 | 0.905176 ± 0.002538 |
| development evaluation | W after common warm-up | Public coverage | 0.493155 ± 0.012802 | 0.767459 ± 0.010763 | 0.589901 ± 0.011222 | 0.740932 ± 0.010495 |
| development evaluation | C initialized task-only | Same residence | 0.499489 ± 0.001184 | 0.773405 ± 0.003787 | 0.550725 ± 0.031639 | 0.688643 ± 0.007789 |
| development evaluation | C initialized task-only | Commute >20 min | 0.685948 ± 0.003371 | 0.553523 ± 0.010136 | 0.551056 ± 0.010195 | 0.572546 ± 0.009714 |
| development evaluation | C initialized task-only | Income >$50k | 0.289622 ± 0.011301 | 0.869747 ± 0.003985 | 0.743016 ± 0.013786 | 0.905308 ± 0.009828 |
| development evaluation | C initialized task-only | Civilian at work | 0.277046 ± 0.007070 | 0.903610 ± 0.008395 | 0.858879 ± 0.009289 | 0.904364 ± 0.002018 |
| development evaluation | C initialized task-only | Public coverage | 0.495113 ± 0.011102 | 0.763691 ± 0.005618 | 0.578582 ± 0.015929 | 0.739065 ± 0.010638 |
| development evaluation | C initialized task-only | SEX | 0.655859 ± 0.007702 | 0.602232 ± 0.010606 | 0.600736 ± 0.009617 | 0.649065 ± 0.011309 |
| development evaluation | C initialized task-only | RAC1P | 1.218044 ± 0.025271 | 0.580024 ± 0.012251 | undefined (1/3) | undefined (1/3) |
| development evaluation | D initialized protected | Same residence | 0.501559 ± 0.002914 | 0.774406 ± 0.001927 | 0.551510 ± 0.023445 | 0.684627 ± 0.005118 |
| development evaluation | D initialized protected | Commute >20 min | 0.685954 ± 0.003734 | 0.550821 ± 0.010230 | 0.548561 ± 0.010443 | 0.571989 ± 0.011035 |
| development evaluation | D initialized protected | Income >$50k | 0.290818 ± 0.010429 | 0.866733 ± 0.005129 | 0.739284 ± 0.016742 | 0.904626 ± 0.008882 |
| development evaluation | D initialized protected | Civilian at work | 0.276961 ± 0.008175 | 0.904060 ± 0.007326 | 0.858854 ± 0.008048 | 0.904276 ± 0.001361 |
| development evaluation | D initialized protected | Public coverage | 0.494527 ± 0.010908 | 0.765479 ± 0.007381 | 0.581096 ± 0.018323 | 0.740094 ± 0.010600 |
| development evaluation | D initialized protected | SEX | 0.663448 ± 0.006426 | 0.593874 ± 0.009738 | 0.592327 ± 0.007703 | 0.632174 ± 0.010954 |
| development evaluation | D initialized protected | RAC1P | 1.240585 ± 0.021967 | 0.571125 ± 0.013508 | undefined (1/3) | undefined (1/3) |
| validation person weighted | I initialized mapper | Same residence | 0.481576 ± 0.003475 | 0.788534 ± 0.013784 | 0.530411 ± 0.004041 | 0.690486 ± 0.020027 |
| validation person weighted | I initialized mapper | Commute >20 min | 0.676259 ± 0.002059 | 0.576107 ± 0.008525 | 0.575892 ± 0.007575 | 0.604967 ± 0.009330 |
| validation person weighted | I initialized mapper | Income >$50k | 0.305116 ± 0.008916 | 0.858658 ± 0.006103 | 0.726225 ± 0.000710 | 0.891569 ± 0.004304 |
| validation person weighted | I initialized mapper | Civilian at work | 0.299592 ± 0.003126 | 0.895407 ± 0.001444 | 0.838116 ± 0.008471 | 0.883374 ± 0.001962 |
| validation person weighted | I initialized mapper | Public coverage | 0.501413 ± 0.003969 | 0.765744 ± 0.004968 | 0.574148 ± 0.015527 | 0.716611 ± 0.016197 |
| validation person weighted | W after common warm-up | Same residence | 0.481330 ± 0.018290 | 0.785059 ± 0.019679 | 0.537548 ± 0.020711 | 0.691130 ± 0.008553 |
| validation person weighted | W after common warm-up | Commute >20 min | 0.674993 ± 0.002171 | 0.569940 ± 0.005865 | 0.569489 ± 0.003564 | 0.600898 ± 0.007575 |
| validation person weighted | W after common warm-up | Income >$50k | 0.289940 ± 0.015094 | 0.864562 ± 0.006011 | 0.733800 ± 0.007275 | 0.902558 ± 0.006011 |
| validation person weighted | W after common warm-up | Civilian at work | 0.289410 ± 0.005562 | 0.897333 ± 0.000455 | 0.840806 ± 0.007806 | 0.890482 ± 0.007405 |
| validation person weighted | W after common warm-up | Public coverage | 0.491580 ± 0.004982 | 0.776048 ± 0.008912 | 0.591009 ± 0.012226 | 0.730417 ± 0.013069 |
| validation person weighted | C initialized task-only | Same residence | 0.487121 ± 0.019048 | 0.785415 ± 0.021299 | 0.541125 ± 0.031985 | 0.679612 ± 0.012045 |
| validation person weighted | C initialized task-only | Commute >20 min | 0.675437 ± 0.002486 | 0.569604 ± 0.007950 | 0.569245 ± 0.007440 | 0.600956 ± 0.008592 |
| validation person weighted | C initialized task-only | Income >$50k | 0.291196 ± 0.016550 | 0.866072 ± 0.005402 | 0.729716 ± 0.011524 | 0.902017 ± 0.007382 |
| validation person weighted | C initialized task-only | Civilian at work | 0.291516 ± 0.005633 | 0.897397 ± 0.001500 | 0.841067 ± 0.008436 | 0.887755 ± 0.008068 |
| validation person weighted | C initialized task-only | Public coverage | 0.493907 ± 0.005221 | 0.771947 ± 0.011108 | 0.576236 ± 0.018071 | 0.726337 ± 0.013421 |
| validation person weighted | C initialized task-only | SEX | 0.661034 ± 0.012503 | 0.596257 ± 0.019787 | 0.593827 ± 0.020575 | 0.637468 ± 0.022573 |
| validation person weighted | C initialized task-only | RAC1P | 1.222536 ± 0.007978 | 0.588013 ± 0.004146 | undefined (0/3) | undefined (0/3) |
| validation person weighted | D initialized protected | Same residence | 0.487454 ± 0.016863 | 0.782150 ± 0.019689 | 0.533773 ± 0.031854 | 0.679044 ± 0.013268 |
| validation person weighted | D initialized protected | Commute >20 min | 0.674785 ± 0.003389 | 0.574644 ± 0.014889 | 0.574206 ± 0.013725 | 0.603426 ± 0.011814 |
| validation person weighted | D initialized protected | Income >$50k | 0.291398 ± 0.016674 | 0.865201 ± 0.007039 | 0.727944 ± 0.007631 | 0.901727 ± 0.007540 |
| validation person weighted | D initialized protected | Civilian at work | 0.291113 ± 0.004869 | 0.897443 ± 0.001383 | 0.841242 ± 0.008384 | 0.887530 ± 0.007481 |
| validation person weighted | D initialized protected | Public coverage | 0.493946 ± 0.009242 | 0.771596 ± 0.008227 | 0.574195 ± 0.010697 | 0.729594 ± 0.014727 |
| validation person weighted | D initialized protected | SEX | 0.672988 ± 0.012112 | 0.579953 ± 0.017686 | 0.578255 ± 0.018763 | 0.612936 ± 0.022667 |
| validation person weighted | D initialized protected | RAC1P | 1.244905 ± 0.009788 | 0.578244 ± 0.005026 | undefined (0/3) | undefined (0/3) |
| development evaluation person weighted | I initialized mapper | Same residence | 0.474544 ± 0.009318 | 0.794866 ± 0.005974 | 0.530074 ± 0.012305 | 0.689241 ± 0.012124 |
| development evaluation person weighted | I initialized mapper | Commute >20 min | 0.687659 ± 0.003119 | 0.548268 ± 0.012592 | 0.546486 ± 0.013037 | 0.566763 ± 0.012276 |
| development evaluation person weighted | I initialized mapper | Income >$50k | 0.304320 ± 0.009139 | 0.863699 ± 0.004219 | 0.732448 ± 0.011290 | 0.894771 ± 0.009310 |
| development evaluation person weighted | I initialized mapper | Civilian at work | 0.278038 ± 0.010757 | 0.906525 ± 0.009699 | 0.853258 ± 0.017332 | 0.896057 ± 0.000812 |
| development evaluation person weighted | I initialized mapper | Public coverage | 0.511918 ± 0.010256 | 0.758593 ± 0.015417 | 0.573717 ± 0.010772 | 0.716729 ± 0.007794 |
| development evaluation person weighted | W after common warm-up | Same residence | 0.471979 ± 0.010379 | 0.795309 ± 0.004980 | 0.538464 ± 0.025303 | 0.690335 ± 0.017489 |
| development evaluation person weighted | W after common warm-up | Commute >20 min | 0.687598 ± 0.003855 | 0.548987 ± 0.015005 | 0.545241 ± 0.015092 | 0.565688 ± 0.011836 |
| development evaluation person weighted | W after common warm-up | Income >$50k | 0.297719 ± 0.008409 | 0.867148 ± 0.004151 | 0.732475 ± 0.016868 | 0.899592 ± 0.008858 |
| development evaluation person weighted | W after common warm-up | Civilian at work | 0.267152 ± 0.011518 | 0.910366 ± 0.009689 | 0.858483 ± 0.014750 | 0.900486 ± 0.001806 |
| development evaluation person weighted | W after common warm-up | Public coverage | 0.499523 ± 0.013558 | 0.762002 ± 0.014555 | 0.580887 ± 0.008322 | 0.735455 ± 0.004528 |
| development evaluation person weighted | C initialized task-only | Same residence | 0.472576 ± 0.007503 | 0.797328 ± 0.003220 | 0.544849 ± 0.032508 | 0.687840 ± 0.015295 |
| development evaluation person weighted | C initialized task-only | Commute >20 min | 0.689052 ± 0.003516 | 0.549808 ± 0.011380 | 0.546279 ± 0.011261 | 0.563889 ± 0.011209 |
| development evaluation person weighted | C initialized task-only | Income >$50k | 0.300769 ± 0.010408 | 0.865382 ± 0.005719 | 0.727636 ± 0.014831 | 0.897295 ± 0.010880 |
| development evaluation person weighted | C initialized task-only | Civilian at work | 0.268338 ± 0.012371 | 0.911072 ± 0.010915 | 0.859344 ± 0.015444 | 0.899072 ± 0.002017 |
| development evaluation person weighted | C initialized task-only | Public coverage | 0.503369 ± 0.011680 | 0.757512 ± 0.008541 | 0.568504 ± 0.011104 | 0.731674 ± 0.004426 |
| development evaluation person weighted | C initialized task-only | SEX | 0.659805 ± 0.005992 | 0.600834 ± 0.012790 | 0.597126 ± 0.013607 | 0.641105 ± 0.012415 |
| development evaluation person weighted | C initialized task-only | RAC1P | 1.220618 ± 0.042998 | 0.575745 ± 0.012983 | undefined (1/3) | undefined (1/3) |
| development evaluation person weighted | D initialized protected | Same residence | 0.474214 ± 0.005709 | 0.796939 ± 0.001446 | 0.542339 ± 0.025039 | 0.684770 ± 0.007993 |
| development evaluation person weighted | D initialized protected | Commute >20 min | 0.688258 ± 0.004412 | 0.547970 ± 0.007747 | 0.544723 ± 0.007534 | 0.566463 ± 0.015288 |
| development evaluation person weighted | D initialized protected | Income >$50k | 0.301533 ± 0.010749 | 0.862907 ± 0.006267 | 0.724020 ± 0.015153 | 0.896447 ± 0.010242 |
| development evaluation person weighted | D initialized protected | Civilian at work | 0.268148 ± 0.013236 | 0.911165 ± 0.010041 | 0.858923 ± 0.014415 | 0.898750 ± 0.003664 |
| development evaluation person weighted | D initialized protected | Public coverage | 0.502929 ± 0.012359 | 0.759399 ± 0.010590 | 0.569385 ± 0.013047 | 0.733729 ± 0.004651 |
| development evaluation person weighted | D initialized protected | SEX | 0.668576 ± 0.004859 | 0.588096 ± 0.009440 | 0.584439 ± 0.009522 | 0.620406 ± 0.010972 |
| development evaluation person weighted | D initialized protected | RAC1P | 1.243674 ± 0.036200 | 0.570728 ± 0.013244 | undefined (1/3) | undefined (1/3) |

All initialization identities, training metadata, inherited exposures, original references and runtime remain in the per-seed evidence linked by summary.json. No privacy, novelty, or population claim follows from this stage comparison. Every release map remains fixed before its downstream heads are fitted.
