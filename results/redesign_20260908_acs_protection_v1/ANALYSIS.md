# ACS erasure feasibility analysis

This report summarizes frozen outputs without refitting or selecting on development outcomes. The provisional comparisons are descriptive empirical checks, not privacy budgets or statistical noninferiority tests. Validation and development evaluation are assessed independently, seed by seed. These seeds share the same cohort, so the SD is not population uncertainty.

[Primary tables](TABLE.md), [coverage and exposed controls](SUPPORT.md), [all scores](PER_TARGET.csv), [per-class scores](PER_CLASS.csv), [training curves](CURVES.csv), [fit accounting](FITTING.csv), [paired comparisons](PAIRED.csv), [exact criteria](criteria.json), [feature/bank comparisons](feature_comparisons.json), [all means and SDs](summary.json).

Attack gain is prior log loss minus attack log loss, retaining its sign. A nonpositive parent gain makes fractional reduction undefined. Incomplete fit/validation/evaluation class support or a zero/undefined class recall for the validation-selected exposed control limits the attribute assessment; it cannot pass. Raw scalar inequalities are still published. Eraser calibration coverage gaps are flagged separately.

## Provisional criterion counts

Cells are pass / fail / undefined across completed seeds.

| Split | Parent | Source all | Residence retention | SEX halving | RAC1P halving | Joint |
| --- | --- | --- | --- | --- | --- | --- |
| validation | A binary bank | 0 / 3 / 0 | 0 / 0 / 3 | 3 / 0 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| validation | B rich neural bank | 0 / 3 / 0 | 0 / 0 / 3 | 0 / 3 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| validation | C rich tree bank | 0 / 3 / 0 | 0 / 0 / 3 | 2 / 1 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| validation | D compressed features | 0 / 3 / 0 | 0 / 3 / 0 | 1 / 2 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| validation | E PCA features | 0 / 3 / 0 | 1 / 2 / 0 | 2 / 1 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| development evaluation | A binary bank | 0 / 3 / 0 | 0 / 0 / 3 | 3 / 0 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| development evaluation | B rich neural bank | 0 / 3 / 0 | 0 / 0 / 3 | 0 / 3 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| development evaluation | C rich tree bank | 0 / 3 / 0 | 0 / 0 / 3 | 2 / 1 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| development evaluation | D compressed features | 0 / 3 / 0 | 0 / 3 / 0 | 3 / 0 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |
| development evaluation | E PCA features | 0 / 3 / 0 | 2 / 1 / 0 | 3 / 0 / 0 | 0 / 0 / 3 | 0 / 3 / 0 |

## Signed attribute gain, mean ± sample SD

Each seed pairs the selected attack with that seed’s fitting-prior loss before aggregation. Negative gains remain negative. Gains do not imply a protection pass when coverage is limited.

| Release | Validation SEX | Validation RAC1P | Development SEX | Development RAC1P |
| --- | --- | --- | --- | --- |
| A binary bank | 0.005373 ± 0.005241 | 0.032632 ± 0.007037 | 0.003633 ± 0.002153 | 0.037977 ± 0.003258 |
| A binary bank + LEACE | 0.000189 ± 0.000171 | 0.000675 ± 0.000451 | -0.000090 ± 0.000320 | 0.000856 ± 0.000516 |
| B rich neural bank | 0.031968 ± 0.002391 | 0.043131 ± 0.007114 | 0.027917 ± 0.007718 | 0.046870 ± 0.008823 |
| B rich neural bank + LEACE | 0.024095 ± 0.005025 | 0.019203 ± 0.018612 | 0.021172 ± 0.007435 | 0.021777 ± 0.013415 |
| C rich tree bank | 0.015502 ± 0.005626 | 0.031025 ± 0.009944 | 0.015993 ± 0.003021 | 0.030722 ± 0.006426 |
| C rich tree bank + LEACE | 0.008240 ± 0.002152 | -0.003893 ± 0.004179 | 0.007222 ± 0.002179 | -0.000087 ± 0.013173 |
| D compressed features | 0.036181 ± 0.006987 | 0.062341 ± 0.015271 | 0.037906 ± 0.005235 | 0.070422 ± 0.013353 |
| D compressed features + LEACE | 0.020463 ± 0.007421 | 0.011972 ± 0.005952 | 0.017325 ± 0.004011 | 0.020984 ± 0.009102 |
| E PCA features | 0.037585 ± 0.006196 | 0.074665 ± 0.013280 | 0.037469 ± 0.010753 | 0.089644 ± 0.003041 |
| E PCA features + LEACE | 0.018733 ± 0.008196 | 0.042992 ± 0.019004 | 0.014566 ± 0.007457 | 0.054996 ± 0.001780 |
| F full covariates | 0.035371 ± 0.009252 | 0.078675 ± 0.013393 | 0.034291 ± 0.011167 | 0.089223 ± 0.005030 |
| Fitting prior | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |

## Compressed features versus PCA: explicit paired contrasts

These contrasts use validation-selected primary heads and attacks. Each cell is the mean ± sample SD of compressed-feature minus PCA log loss paired within seed. Lower residence loss is better; higher attack loss means less measured recovery. This comparison has no additional provisional threshold.

| Split | Compressed arm | PCA arm | Residence LL difference | SEX attack LL difference | RAC1P attack LL difference |
| --- | --- | --- | --- | --- | --- |
| validation | D compressed features | E PCA features | 0.007651 ± 0.003662 | 0.001404 ± 0.004351 | 0.012324 ± 0.002966 |
| validation | D compressed features | E PCA features + LEACE | -0.002741 ± 0.006928 | -0.017448 ± 0.001967 | -0.019350 ± 0.009800 |
| validation | D compressed features + LEACE | E PCA features | 0.019323 ± 0.003898 | 0.017121 ± 0.006449 | 0.062693 ± 0.008216 |
| validation | D compressed features + LEACE | E PCA features + LEACE | 0.008931 ± 0.004308 | -0.001730 ± 0.001121 | 0.031020 ± 0.015483 |
| development evaluation | D compressed features | E PCA features | 0.011287 ± 0.006604 | -0.000437 ± 0.005707 | 0.019222 ± 0.015251 |
| development evaluation | D compressed features | E PCA features + LEACE | 0.002423 ± 0.011611 | -0.023340 ± 0.002226 | -0.015426 ± 0.011931 |
| development evaluation | D compressed features + LEACE | E PCA features | 0.022271 ± 0.005079 | 0.020144 ± 0.006761 | 0.068660 ± 0.011414 |
| development evaluation | D compressed features + LEACE | E PCA features + LEACE | 0.013407 ± 0.002615 | -0.002759 ± 0.003706 | 0.034012 ± 0.008597 |

## Validation: headroom and signed attribute gains

| Seed | Parent | Residence H before | Residence H after | Residence retained | SEX gain before → after | SEX retained | RAC1P gain before → after | RAC1P retained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | A binary bank | -0.012971 | -0.025931 | undefined | 0.011240 → 0.000225 | 0.020025 | 0.026870 → 0.000585 | undefined |
| 0 | B rich neural bank | 0.000000 | -0.008742 | undefined | 0.034637 → 0.028780 | 0.830900 | 0.035974 → 0.010868 | undefined |
| 0 | C rich tree bank | -0.009519 | -0.020936 | undefined | 0.021997 → 0.010227 | 0.464932 | 0.029097 → -0.005720 | undefined |
| 0 | D compressed features | 0.004640 | -0.000906 | -0.195189 | 0.043984 → 0.029021 | 0.659798 | 0.049080 → 0.008178 | undefined |
| 0 | E PCA features | 0.015386 | 0.009749 | 0.633588 | 0.041923 → 0.028185 | 0.672296 | 0.061715 → 0.021407 | undefined |
| 1 | A binary bank | -0.007551 | -0.015966 | undefined | 0.001155 → 0.000003 | 0.002760 | 0.040474 → 0.001164 | undefined |
| 1 | B rich neural bank | 0.000000 | 0.001194 | undefined | 0.031243 → 0.024717 | 0.791111 | 0.043216 → 0.006216 | undefined |
| 1 | C rich tree bank | -0.003479 | -0.012929 | undefined | 0.012122 → 0.008539 | 0.704403 | 0.041793 → 0.000888 | undefined |
| 1 | D compressed features | 0.012383 | 0.003023 | 0.244157 | 0.034056 → 0.015797 | 0.463861 | 0.058906 → 0.008907 | undefined |
| 1 | E PCA features | 0.020981 | 0.007051 | 0.336082 | 0.040342 → 0.014430 | 0.357688 | 0.074029 → 0.050357 | undefined |
| 2 | A binary bank | -0.012158 | -0.029157 | undefined | 0.003722 → 0.000339 | 0.090936 | 0.030551 → 0.000276 | undefined |
| 2 | B rich neural bank | 0.000000 | -0.007589 | undefined | 0.030023 → 0.018788 | 0.625798 | 0.050203 → 0.040526 | undefined |
| 2 | C rich tree bank | -0.011847 | -0.018904 | undefined | 0.012387 → 0.005955 | 0.480732 | 0.022186 → -0.006848 | undefined |
| 2 | D compressed features | 0.017239 | -0.002872 | -0.166603 | 0.030503 → 0.016572 | 0.543290 | 0.079038 → 0.018832 | undefined |
| 2 | E PCA features | 0.020847 | 0.009239 | 0.443193 | 0.030488 → 0.013585 | 0.445578 | 0.088252 → 0.057211 | undefined |

## Validation: every feature/bank margin

Count cells are pass / fail / undefined. Numeric attribute and joint columns disregard coverage only to show the scalar inequalities; the final policy column retains coverage limitations. Utility means residence loss at least .01 lower; each attribute permits at most .005 extra signed gain.

| Feature | Bank | Utility | SEX numeric | RAC1P numeric | Joint numeric | Joint with coverage |
| --- | --- | --- | --- | --- | --- | --- |
| D compressed features | B rich neural bank | 2 / 1 / 0 | 2 / 1 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 1 / 2 |
| D compressed features | B rich neural bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features | C rich tree bank | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features | C rich tree bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features + LEACE | B rich neural bank | 0 / 3 / 0 | 3 / 0 / 0 | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features + LEACE | B rich neural bank + LEACE | 0 / 3 / 0 | 3 / 0 / 0 | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features + LEACE | C rich tree bank | 0 / 3 / 0 | 2 / 1 / 0 | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features + LEACE | C rich tree bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features | B rich neural bank | 3 / 0 / 0 | 1 / 2 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 2 / 1 |
| E PCA features | B rich neural bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features | C rich tree bank | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features | C rich tree bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features + LEACE | B rich neural bank | 0 / 3 / 0 | 3 / 0 / 0 | 1 / 2 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features + LEACE | B rich neural bank + LEACE | 2 / 1 / 0 | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 1 / 2 |
| E PCA features + LEACE | C rich tree bank | 3 / 0 / 0 | 2 / 1 / 0 | 1 / 2 / 0 | 0 / 3 / 0 | 0 / 1 / 2 |
| E PCA features + LEACE | C rich tree bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |

## Development evaluation: headroom and signed attribute gains

| Seed | Parent | Residence H before | Residence H after | Residence retained | SEX gain before → after | SEX retained | RAC1P gain before → after | RAC1P retained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | A binary bank | -0.011682 | -0.020664 | undefined | 0.006114 → 0.000167 | 0.027368 | 0.038244 → 0.000792 | undefined |
| 0 | B rich neural bank | 0.000000 | -0.011403 | undefined | 0.020405 → 0.013365 | 0.654986 | 0.040435 → 0.010451 | undefined |
| 0 | C rich tree bank | -0.007479 | -0.017241 | undefined | 0.013360 → 0.004748 | 0.355413 | 0.027104 → -0.006803 | undefined |
| 0 | D compressed features | 0.001783 | 0.000133 | 0.074347 | 0.034823 → 0.016127 | 0.463119 | 0.059437 → 0.029822 | undefined |
| 0 | E PCA features | 0.016622 | 0.016117 | 0.969626 | 0.033326 → 0.010349 | 0.310549 | 0.088982 → 0.054610 | undefined |
| 1 | A binary bank | -0.007093 | -0.025068 | undefined | 0.002522 → 0.000010 | 0.004109 | 0.034594 → 0.001400 | undefined |
| 1 | B rich neural bank | 0.000000 | -0.001102 | undefined | 0.035827 → 0.028170 | 0.786269 | 0.043247 → 0.018289 | undefined |
| 1 | C rich tree bank | -0.001215 | -0.021734 | undefined | 0.015327 → 0.008058 | 0.525709 | 0.038141 → 0.015090 | undefined |
| 1 | D compressed features | 0.017814 | 0.008857 | 0.497198 | 0.043951 → 0.021799 | 0.495989 | 0.066544 → 0.011640 | undefined |
| 1 | E PCA features | 0.033170 | 0.019612 | 0.591269 | 0.049677 → 0.023176 | 0.466536 | 0.092961 → 0.053441 | undefined |
| 2 | A binary bank | -0.016925 | -0.031844 | undefined | 0.002261 → -0.000448 | -0.198270 | 0.041094 → 0.000374 | undefined |
| 2 | B rich neural bank | 0.000000 | -0.001650 | undefined | 0.027520 → 0.021981 | 0.798738 | 0.056928 → 0.036592 | undefined |
| 2 | C rich tree bank | -0.011787 | -0.026866 | undefined | 0.019292 → 0.008859 | 0.459209 | 0.026921 → -0.008548 | undefined |
| 2 | D compressed features | 0.015893 | -0.006452 | -0.405952 | 0.034944 → 0.014049 | 0.402055 | 0.085285 → 0.021491 | undefined |
| 2 | E PCA features | 0.019560 | 0.007030 | 0.359408 | 0.029404 → 0.010174 | 0.346009 | 0.086989 → 0.056938 | undefined |

## Development evaluation: every feature/bank margin

Count cells are pass / fail / undefined. Numeric attribute and joint columns disregard coverage only to show the scalar inequalities; the final policy column retains coverage limitations. Utility means residence loss at least .01 lower; each attribute permits at most .005 extra signed gain.

| Feature | Bank | Utility | SEX numeric | RAC1P numeric | Joint numeric | Joint with coverage |
| --- | --- | --- | --- | --- | --- | --- |
| D compressed features | B rich neural bank | 2 / 1 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features | B rich neural bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features | C rich tree bank | 2 / 1 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features | C rich tree bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features + LEACE | B rich neural bank | 0 / 3 / 0 | 3 / 0 / 0 | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features + LEACE | B rich neural bank + LEACE | 1 / 2 / 0 | 3 / 0 / 0 | 2 / 1 / 0 | 0 / 3 / 0 | 0 / 2 / 1 |
| D compressed features + LEACE | C rich tree bank | 1 / 2 / 0 | 2 / 1 / 0 | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| D compressed features + LEACE | C rich tree bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 1 / 2 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features | B rich neural bank | 3 / 0 / 0 | 1 / 2 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 2 / 1 |
| E PCA features | B rich neural bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features | C rich tree bank | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features | C rich tree bank + LEACE | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| E PCA features + LEACE | B rich neural bank | 2 / 1 / 0 | 3 / 0 / 0 | 1 / 2 / 0 | 0 / 3 / 0 | 0 / 1 / 2 |
| E PCA features + LEACE | B rich neural bank + LEACE | 2 / 1 / 0 | 3 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 1 / 2 |
| E PCA features + LEACE | C rich tree bank | 3 / 0 / 0 | 2 / 1 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 1 / 2 |
| E PCA features + LEACE | C rich tree bank + LEACE | 3 / 0 / 0 | 1 / 2 / 0 | 0 / 3 / 0 | 0 / 3 / 0 | 0 / 2 / 1 |

## All primary scoring metrics, mean ± sample SD

Full-schema macro AUROC and balanced accuracy stay undefined if a category is unsupported. Observed-class alternatives remain explicitly labeled in CSV/JSON. PWGTP sensitivity uses the same predictions and unweighted validation selections; it is not a Census population estimate.

### Validation

| Release | Target | Log loss | Accuracy | Balanced accuracy | AUROC (binary) / macro AUROC (RAC1P) |
| --- | --- | --- | --- | --- | --- |
| A binary bank | Same residence | 0.529969 ± 0.009524 | 0.767244 ± 0.011700 | 0.506176 ± 0.006692 | 0.613183 ± 0.011361 |
| A binary bank + LEACE | Same residence | 0.542760 ± 0.013788 | 0.767125 ± 0.011791 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Same residence | 0.519075 ± 0.007071 | 0.764694 ± 0.009667 | 0.521165 ± 0.013749 | 0.655130 ± 0.018799 |
| B rich neural bank + LEACE | Same residence | 0.524121 ± 0.011871 | 0.759556 ± 0.015473 | 0.528652 ± 0.013821 | 0.648229 ± 0.012116 |
| C rich tree bank | Same residence | 0.527357 ± 0.011294 | 0.764589 ± 0.008397 | 0.522086 ± 0.009816 | 0.631480 ± 0.003059 |
| C rich tree bank + LEACE | Same residence | 0.536665 ± 0.010436 | 0.765556 ± 0.011967 | 0.511482 ± 0.009696 | 0.595714 ± 0.029867 |
| D compressed features | Same residence | 0.507655 ± 0.007399 | 0.762578 ± 0.009162 | 0.542666 ± 0.015359 | 0.685022 ± 0.017534 |
| D compressed features + LEACE | Same residence | 0.519327 ± 0.010028 | 0.762384 ± 0.005399 | 0.539408 ± 0.018550 | 0.660448 ± 0.003657 |
| E PCA features | Same residence | 0.500004 ± 0.007773 | 0.762939 ± 0.008261 | 0.542018 ± 0.017551 | 0.701969 ± 0.010921 |
| E PCA features + LEACE | Same residence | 0.510395 ± 0.006065 | 0.762712 ± 0.017876 | 0.528593 ± 0.024804 | 0.681074 ± 0.019487 |
| F full covariates | Same residence | 0.500837 ± 0.008018 | 0.766028 ± 0.011870 | 0.522628 ± 0.035820 | 0.700282 ± 0.017473 |
| Fitting prior | Same residence | 0.543056 ± 0.013563 | 0.767125 ± 0.011791 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Commute >20 min | 0.674088 ± 0.001156 | 0.570236 ± 0.012193 | 0.570255 ± 0.011177 | 0.598596 ± 0.003650 |
| A binary bank + LEACE | Commute >20 min | 0.693340 ± 0.000316 | 0.498491 ± 0.007846 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Commute >20 min | 0.672589 ± 0.001273 | 0.575425 ± 0.012785 | 0.575357 ± 0.013969 | 0.601379 ± 0.005423 |
| B rich neural bank + LEACE | Commute >20 min | 0.677601 ± 0.002403 | 0.567554 ± 0.007619 | 0.566827 ± 0.009262 | 0.594016 ± 0.010923 |
| C rich tree bank | Commute >20 min | 0.674715 ± 0.000318 | 0.577948 ± 0.009375 | 0.577735 ± 0.008321 | 0.597245 ± 0.004073 |
| C rich tree bank + LEACE | Commute >20 min | 0.681729 ± 0.004078 | 0.562003 ± 0.004512 | 0.561492 ± 0.004356 | 0.579154 ± 0.010891 |
| D compressed features | Commute >20 min | 0.672725 ± 0.001254 | 0.578275 ± 0.008370 | 0.578282 ± 0.007711 | 0.605543 ± 0.003626 |
| D compressed features + LEACE | Commute >20 min | 0.677689 ± 0.003299 | 0.562354 ± 0.008297 | 0.561978 ± 0.009124 | 0.588243 ± 0.009207 |
| E PCA features | Commute >20 min | 0.678247 ± 0.001214 | 0.562874 ± 0.009805 | 0.562933 ± 0.008787 | 0.597813 ± 0.004481 |
| E PCA features + LEACE | Commute >20 min | 0.682046 ± 0.002754 | 0.556198 ± 0.013725 | 0.555807 ± 0.012564 | 0.586970 ± 0.006161 |
| F full covariates | Commute >20 min | 0.680333 ± 0.003601 | 0.566266 ± 0.007658 | 0.566119 ± 0.006595 | 0.592446 ± 0.010819 |
| Fitting prior | Commute >20 min | 0.693473 ± 0.000409 | 0.498491 ± 0.007846 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Income >$50k | 0.289275 ± 0.001297 | 0.868723 ± 0.002647 | 0.760593 ± 0.012434 | 0.906298 ± 0.001662 |
| A binary bank + LEACE | Income >$50k | 0.479912 ± 0.006918 | 0.814343 ± 0.004627 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Income >$50k | 0.287795 ± 0.003673 | 0.869607 ± 0.003277 | 0.757771 ± 0.007062 | 0.906083 ± 0.002400 |
| B rich neural bank + LEACE | Income >$50k | 0.325812 ± 0.008173 | 0.861798 ± 0.007897 | 0.717911 ± 0.012551 | 0.880294 ± 0.007775 |
| C rich tree bank | Income >$50k | 0.287529 ± 0.002755 | 0.868828 ± 0.005019 | 0.762732 ± 0.006977 | 0.906394 ± 0.002837 |
| C rich tree bank + LEACE | Income >$50k | 0.349898 ± 0.001542 | 0.862424 ± 0.001318 | 0.712003 ± 0.022514 | 0.863688 ± 0.006191 |
| D compressed features | Income >$50k | 0.291116 ± 0.001365 | 0.869516 ± 0.004995 | 0.758803 ± 0.004539 | 0.904349 ± 0.001786 |
| D compressed features + LEACE | Income >$50k | 0.320595 ± 0.008623 | 0.861313 ± 0.009319 | 0.729229 ± 0.007258 | 0.880294 ± 0.006435 |
| E PCA features | Income >$50k | 0.306979 ± 0.001135 | 0.861110 ± 0.004169 | 0.739244 ± 0.013304 | 0.892729 ± 0.002250 |
| E PCA features + LEACE | Income >$50k | 0.321661 ± 0.007824 | 0.857266 ± 0.007348 | 0.721022 ± 0.022545 | 0.880103 ± 0.007249 |
| F full covariates | Income >$50k | 0.315025 ± 0.005883 | 0.863758 ± 0.003620 | 0.744399 ± 0.008058 | 0.890787 ± 0.003292 |
| Fitting prior | Income >$50k | 0.480116 ± 0.006864 | 0.814343 ± 0.004627 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Civilian at work | 0.291139 ± 0.007640 | 0.897082 ± 0.005484 | 0.848153 ± 0.006036 | 0.897060 ± 0.002190 |
| A binary bank + LEACE | Civilian at work | 0.626015 ± 0.011256 | 0.680710 ± 0.014514 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Civilian at work | 0.287739 ± 0.008515 | 0.896638 ± 0.005680 | 0.845545 ± 0.006635 | 0.897266 ± 0.004108 |
| B rich neural bank + LEACE | Civilian at work | 0.324002 ± 0.006732 | 0.885173 ± 0.003532 | 0.836344 ± 0.005271 | 0.884465 ± 0.004486 |
| C rich tree bank | Civilian at work | 0.291640 ± 0.012045 | 0.897521 ± 0.005865 | 0.850720 ± 0.007364 | 0.893995 ± 0.005203 |
| C rich tree bank + LEACE | Civilian at work | 0.357083 ± 0.003590 | 0.877763 ± 0.004634 | 0.831404 ± 0.003109 | 0.871054 ± 0.007781 |
| D compressed features | Civilian at work | 0.292921 ± 0.007257 | 0.894222 ± 0.004371 | 0.844523 ± 0.006695 | 0.893728 ± 0.005947 |
| D compressed features + LEACE | Civilian at work | 0.346264 ± 0.010923 | 0.875359 ± 0.003469 | 0.824334 ± 0.008105 | 0.874055 ± 0.006127 |
| E PCA features | Civilian at work | 0.304701 ± 0.008147 | 0.893245 ± 0.001131 | 0.843135 ± 0.009587 | 0.882927 ± 0.014472 |
| E PCA features + LEACE | Civilian at work | 0.340376 ± 0.018177 | 0.877661 ± 0.004579 | 0.827889 ± 0.002491 | 0.873151 ± 0.011497 |
| F full covariates | Civilian at work | 0.310709 ± 0.005425 | 0.888735 ± 0.004600 | 0.837084 ± 0.014514 | 0.878972 ± 0.011914 |
| Fitting prior | Civilian at work | 0.626051 ± 0.011224 | 0.680710 ± 0.014514 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Public coverage | 0.474439 ± 0.006254 | 0.786510 ± 0.002567 | 0.610252 ± 0.005027 | 0.748734 ± 0.014748 |
| A binary bank + LEACE | Public coverage | 0.552866 ± 0.010412 | 0.758354 ± 0.009240 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Public coverage | 0.474939 ± 0.004183 | 0.781864 ± 0.004343 | 0.599104 ± 0.007900 | 0.747591 ± 0.014252 |
| B rich neural bank + LEACE | Public coverage | 0.496884 ± 0.009311 | 0.773556 ± 0.006520 | 0.573080 ± 0.006634 | 0.719445 ± 0.012195 |
| C rich tree bank | Public coverage | 0.475793 ± 0.006442 | 0.783526 ± 0.005850 | 0.596045 ± 0.005813 | 0.745787 ± 0.010396 |
| C rich tree bank + LEACE | Public coverage | 0.500226 ± 0.012284 | 0.773382 ± 0.010906 | 0.568768 ± 0.022240 | 0.709004 ± 0.004646 |
| D compressed features | Public coverage | 0.478015 ± 0.004380 | 0.781959 ± 0.001639 | 0.594778 ± 0.008152 | 0.744229 ± 0.017884 |
| D compressed features + LEACE | Public coverage | 0.505705 ± 0.006132 | 0.777191 ± 0.005850 | 0.578368 ± 0.003593 | 0.697792 ± 0.005185 |
| E PCA features | Public coverage | 0.486786 ± 0.001949 | 0.777425 ± 0.005462 | 0.604812 ± 0.004748 | 0.733542 ± 0.011445 |
| E PCA features + LEACE | Public coverage | 0.507168 ± 0.005879 | 0.765523 ± 0.005726 | 0.563137 ± 0.005691 | 0.693619 ± 0.009387 |
| F full covariates | Public coverage | 0.495790 ± 0.004999 | 0.772358 ± 0.002685 | 0.578152 ± 0.009927 | 0.722483 ± 0.015481 |
| Fitting prior | Public coverage | 0.553401 ± 0.010565 | 0.758354 ± 0.009240 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | SEX | 0.687015 ± 0.004844 | 0.542811 ± 0.012760 | 0.534940 ± 0.016869 | 0.557586 ± 0.025170 |
| A binary bank + LEACE | SEX | 0.692199 ± 0.000421 | 0.521547 ± 0.004979 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | SEX | 0.660420 ± 0.001992 | 0.599140 ± 0.007804 | 0.595625 ± 0.009181 | 0.638610 ± 0.011911 |
| B rich neural bank + LEACE | SEX | 0.668293 ± 0.004692 | 0.583159 ± 0.012872 | 0.580134 ± 0.015470 | 0.620989 ± 0.020690 |
| C rich tree bank | SEX | 0.676885 ± 0.005214 | 0.575092 ± 0.013754 | 0.573832 ± 0.015100 | 0.602916 ± 0.019332 |
| C rich tree bank + LEACE | SEX | 0.684147 ± 0.001832 | 0.550048 ± 0.007453 | 0.546871 ± 0.010501 | 0.569673 ± 0.013326 |
| D compressed features | SEX | 0.656206 ± 0.006586 | 0.599367 ± 0.013200 | 0.597316 ± 0.014646 | 0.643876 ± 0.014772 |
| D compressed features + LEACE | SEX | 0.671924 ± 0.007009 | 0.575789 ± 0.016194 | 0.573778 ± 0.017393 | 0.609785 ± 0.020431 |
| E PCA features | SEX | 0.654803 ± 0.005947 | 0.603099 ± 0.005095 | 0.603043 ± 0.004971 | 0.651093 ± 0.012383 |
| E PCA features + LEACE | SEX | 0.673654 ± 0.007783 | 0.582222 ± 0.013368 | 0.580621 ± 0.013472 | 0.615305 ± 0.020757 |
| F full covariates | SEX | 0.657016 ± 0.008847 | 0.597390 ± 0.011929 | 0.595892 ± 0.012170 | 0.650253 ± 0.014047 |
| Fitting prior | SEX | 0.692388 ± 0.000413 | 0.521547 ± 0.004979 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | RAC1P | 1.250865 ± 0.003028 | 0.576904 ± 0.005592 | undefined (0/3) | undefined (0/3) |
| A binary bank + LEACE | RAC1P | 1.282822 ± 0.005755 | 0.576796 ± 0.005769 | undefined (0/3) | undefined (0/3) |
| B rich neural bank | RAC1P | 1.240366 ± 0.005523 | 0.579299 ± 0.002261 | undefined (0/3) | undefined (0/3) |
| B rich neural bank + LEACE | RAC1P | 1.264294 ± 0.019221 | 0.580705 ± 0.003860 | undefined (0/3) | undefined (0/3) |
| C rich tree bank | RAC1P | 1.252472 ± 0.008746 | 0.576680 ± 0.005543 | undefined (0/3) | undefined (0/3) |
| C rich tree bank + LEACE | RAC1P | 1.287390 ± 0.004448 | 0.575624 ± 0.009504 | undefined (0/3) | undefined (0/3) |
| D compressed features | RAC1P | 1.221156 ± 0.013415 | 0.591030 ± 0.005120 | undefined (0/3) | undefined (0/3) |
| D compressed features + LEACE | RAC1P | 1.271525 ± 0.007342 | 0.584435 ± 0.005384 | undefined (0/3) | undefined (0/3) |
| E PCA features | RAC1P | 1.208832 ± 0.010693 | 0.597492 ± 0.005048 | undefined (0/3) | undefined (0/3) |
| E PCA features + LEACE | RAC1P | 1.240505 ± 0.014155 | 0.591167 ± 0.001316 | undefined (0/3) | undefined (0/3) |
| F full covariates | RAC1P | 1.204822 ± 0.011114 | 0.595283 ± 0.004998 | undefined (0/3) | undefined (0/3) |
| Fitting prior | RAC1P | 1.283497 ± 0.005963 | 0.576796 ± 0.005769 | undefined (0/3) | undefined (0/3) |

### Development evaluation

| Release | Target | Log loss | Accuracy | Balanced accuracy | AUROC (binary) / macro AUROC (RAC1P) |
| --- | --- | --- | --- | --- | --- |
| A binary bank | Same residence | 0.522428 ± 0.002233 | 0.773854 ± 0.005922 | 0.507292 ± 0.006374 | 0.611981 ± 0.012767 |
| A binary bank + LEACE | Same residence | 0.536387 ± 0.005802 | 0.772737 ± 0.005140 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Same residence | 0.510529 ± 0.003504 | 0.772401 ± 0.002171 | 0.523410 ± 0.011293 | 0.658019 ± 0.008926 |
| B rich neural bank + LEACE | Same residence | 0.515247 ± 0.005386 | 0.767495 ± 0.008779 | 0.535355 ± 0.019337 | 0.650251 ± 0.016585 |
| C rich tree bank | Same residence | 0.517356 ± 0.002162 | 0.768826 ± 0.003650 | 0.517177 ± 0.007308 | 0.640546 ± 0.011257 |
| C rich tree bank + LEACE | Same residence | 0.532475 ± 0.005421 | 0.770054 ± 0.005673 | 0.507281 ± 0.007759 | 0.581692 ± 0.030874 |
| D compressed features | Same residence | 0.498699 ± 0.007746 | 0.769505 ± 0.007574 | 0.547159 ± 0.023352 | 0.689148 ± 0.024289 |
| D compressed features + LEACE | Same residence | 0.509683 ± 0.004454 | 0.769628 ± 0.005248 | 0.542787 ± 0.016240 | 0.669952 ± 0.011734 |
| E PCA features | Same residence | 0.487411 ± 0.005595 | 0.776310 ± 0.003900 | 0.555757 ± 0.023928 | 0.711771 ± 0.020772 |
| E PCA features + LEACE | Same residence | 0.496275 ± 0.004211 | 0.773966 ± 0.008160 | 0.536240 ± 0.031349 | 0.696406 ± 0.009465 |
| F full covariates | Same residence | 0.490646 ± 0.006126 | 0.775875 ± 0.006399 | 0.530494 ± 0.049595 | 0.714582 ± 0.009658 |
| Fitting prior | Same residence | 0.536538 ± 0.005614 | 0.772737 ± 0.005140 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Commute >20 min | 0.682161 ± 0.003200 | 0.558367 ± 0.014013 | 0.558048 ± 0.011174 | 0.573651 ± 0.009530 |
| A binary bank + LEACE | Commute >20 min | 0.693057 ± 0.000218 | 0.500055 ± 0.017845 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Commute >20 min | 0.681481 ± 0.002771 | 0.558207 ± 0.011498 | 0.554658 ± 0.012277 | 0.573761 ± 0.006289 |
| B rich neural bank + LEACE | Commute >20 min | 0.687252 ± 0.002053 | 0.547506 ± 0.008595 | 0.544327 ± 0.010468 | 0.562233 ± 0.002760 |
| C rich tree bank | Commute >20 min | 0.681863 ± 0.004760 | 0.557214 ± 0.010000 | 0.554120 ± 0.010375 | 0.576170 ± 0.015167 |
| C rich tree bank + LEACE | Commute >20 min | 0.689332 ± 0.003410 | 0.548733 ± 0.002761 | 0.545477 ± 0.003057 | 0.562703 ± 0.011599 |
| D compressed features | Commute >20 min | 0.680676 ± 0.001939 | 0.554801 ± 0.002325 | 0.552330 ± 0.001319 | 0.579379 ± 0.005597 |
| D compressed features + LEACE | Commute >20 min | 0.683998 ± 0.001091 | 0.551528 ± 0.016141 | 0.548784 ± 0.015117 | 0.570443 ± 0.009669 |
| E PCA features | Commute >20 min | 0.687570 ± 0.008575 | 0.557127 ± 0.017298 | 0.554809 ± 0.018354 | 0.573197 ± 0.020073 |
| E PCA features + LEACE | Commute >20 min | 0.691074 ± 0.007655 | 0.553783 ± 0.002520 | 0.551986 ± 0.002683 | 0.566762 ± 0.013256 |
| F full covariates | Commute >20 min | 0.687624 ± 0.009124 | 0.546444 ± 0.011860 | 0.544628 ± 0.012319 | 0.564627 ± 0.028973 |
| Fitting prior | Commute >20 min | 0.693188 ± 0.000380 | 0.500055 ± 0.017845 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Income >$50k | 0.279148 ± 0.008083 | 0.876743 ± 0.009132 | 0.768178 ± 0.020549 | 0.914401 ± 0.004853 |
| A binary bank + LEACE | Income >$50k | 0.481656 ± 0.006481 | 0.813170 ± 0.004358 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Income >$50k | 0.278421 ± 0.007137 | 0.877636 ± 0.008565 | 0.763357 ± 0.018508 | 0.913522 ± 0.004691 |
| B rich neural bank + LEACE | Income >$50k | 0.318360 ± 0.002766 | 0.862506 ± 0.003626 | 0.714007 ± 0.028528 | 0.886492 ± 0.007655 |
| C rich tree bank | Income >$50k | 0.279997 ± 0.009591 | 0.876531 ± 0.006834 | 0.765900 ± 0.014067 | 0.914331 ± 0.004924 |
| C rich tree bank + LEACE | Income >$50k | 0.327893 ± 0.006511 | 0.861044 ± 0.007212 | 0.704422 ± 0.004794 | 0.878466 ± 0.000529 |
| D compressed features | Income >$50k | 0.280999 ± 0.008037 | 0.875633 ± 0.008034 | 0.763054 ± 0.021260 | 0.912043 ± 0.005217 |
| D compressed features + LEACE | Income >$50k | 0.315014 ± 0.003041 | 0.861970 ± 0.004040 | 0.725394 ± 0.010427 | 0.884596 ± 0.002016 |
| E PCA features | Income >$50k | 0.298578 ± 0.008781 | 0.866966 ± 0.002910 | 0.742142 ± 0.003935 | 0.899561 ± 0.008644 |
| E PCA features + LEACE | Income >$50k | 0.312714 ± 0.009209 | 0.864292 ± 0.005630 | 0.727299 ± 0.006489 | 0.887339 ± 0.011278 |
| F full covariates | Income >$50k | 0.297187 ± 0.011155 | 0.868186 ± 0.004151 | 0.745329 ± 0.008136 | 0.901134 ± 0.009704 |
| Fitting prior | Income >$50k | 0.481821 ± 0.006349 | 0.813170 ± 0.004358 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Civilian at work | 0.277283 ± 0.008673 | 0.903172 ± 0.004572 | 0.859854 ± 0.006754 | 0.910489 ± 0.003699 |
| A binary bank + LEACE | Civilian at work | 0.632253 ± 0.006504 | 0.673544 ± 0.007647 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Civilian at work | 0.270941 ± 0.011699 | 0.903620 ± 0.006152 | 0.858975 ± 0.008981 | 0.912559 ± 0.001712 |
| B rich neural bank + LEACE | Civilian at work | 0.315928 ± 0.017206 | 0.892274 ± 0.006027 | 0.849316 ± 0.010483 | 0.889992 ± 0.007343 |
| C rich tree bank | Civilian at work | 0.274990 ± 0.006996 | 0.901386 ± 0.004608 | 0.860087 ± 0.006345 | 0.910541 ± 0.002846 |
| C rich tree bank + LEACE | Civilian at work | 0.330752 ± 0.012562 | 0.890604 ± 0.002724 | 0.849710 ± 0.004559 | 0.888007 ± 0.006331 |
| D compressed features | Civilian at work | 0.275957 ± 0.009834 | 0.900735 ± 0.006778 | 0.858045 ± 0.009995 | 0.909540 ± 0.003023 |
| D compressed features + LEACE | Civilian at work | 0.326801 ± 0.014926 | 0.884248 ± 0.008657 | 0.841175 ± 0.013261 | 0.889951 ± 0.008438 |
| E PCA features | Civilian at work | 0.288407 ± 0.005005 | 0.899731 ± 0.002401 | 0.856164 ± 0.005436 | 0.895990 ± 0.006882 |
| E PCA features + LEACE | Civilian at work | 0.322018 ± 0.009292 | 0.887259 ± 0.001427 | 0.842734 ± 0.000483 | 0.885500 ± 0.007824 |
| F full covariates | Civilian at work | 0.295826 ± 0.008363 | 0.895713 ± 0.007432 | 0.850778 ± 0.010471 | 0.890359 ± 0.003166 |
| Fitting prior | Civilian at work | 0.632113 ± 0.006230 | 0.673544 ± 0.007647 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Public coverage | 0.487810 ± 0.012493 | 0.767229 ± 0.010453 | 0.593142 ± 0.006724 | 0.749360 ± 0.009380 |
| A binary bank + LEACE | Public coverage | 0.565992 ± 0.009781 | 0.746965 ± 0.008732 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Public coverage | 0.487067 ± 0.012910 | 0.768462 ± 0.008781 | 0.590619 ± 0.014917 | 0.751094 ± 0.011565 |
| B rich neural bank + LEACE | Public coverage | 0.505838 ± 0.008451 | 0.762576 ± 0.009052 | 0.567414 ± 0.014835 | 0.725310 ± 0.013636 |
| C rich tree bank | Public coverage | 0.488114 ± 0.014003 | 0.771700 ± 0.010292 | 0.588936 ± 0.011428 | 0.746444 ± 0.009312 |
| C rich tree bank + LEACE | Public coverage | 0.512242 ± 0.008722 | 0.763690 ± 0.005372 | 0.564810 ± 0.021578 | 0.708940 ± 0.007912 |
| D compressed features | Public coverage | 0.490342 ± 0.013956 | 0.768139 ± 0.006822 | 0.589285 ± 0.016878 | 0.745495 ± 0.010023 |
| D compressed features + LEACE | Public coverage | 0.515717 ± 0.014644 | 0.762680 ± 0.011130 | 0.573415 ± 0.014445 | 0.703447 ± 0.023685 |
| E PCA features | Public coverage | 0.504623 ± 0.011550 | 0.761098 ± 0.015362 | 0.592374 ± 0.008203 | 0.725580 ± 0.007502 |
| E PCA features + LEACE | Public coverage | 0.521313 ± 0.013445 | 0.757683 ± 0.008749 | 0.562303 ± 0.011321 | 0.694414 ± 0.014609 |
| F full covariates | Public coverage | 0.506998 ± 0.015027 | 0.758867 ± 0.011711 | 0.570705 ± 0.004788 | 0.727279 ± 0.013625 |
| Fitting prior | Public coverage | 0.565686 ± 0.009423 | 0.746965 ± 0.008732 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | SEX | 0.689054 ± 0.002075 | 0.538523 ± 0.010263 | 0.532844 ± 0.009073 | 0.554052 ± 0.008742 |
| A binary bank + LEACE | SEX | 0.692777 ± 0.000772 | 0.515005 ± 0.008432 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | SEX | 0.664770 ± 0.008175 | 0.591286 ± 0.015338 | 0.589147 ± 0.013817 | 0.630714 ± 0.015185 |
| B rich neural bank + LEACE | SEX | 0.671515 ± 0.007842 | 0.582824 ± 0.014720 | 0.581193 ± 0.012467 | 0.617251 ± 0.011868 |
| C rich tree bank | SEX | 0.676694 ± 0.002856 | 0.575438 ± 0.003589 | 0.574628 ± 0.005117 | 0.609244 ± 0.003641 |
| C rich tree bank + LEACE | SEX | 0.685465 ± 0.002333 | 0.545618 ± 0.014823 | 0.543604 ± 0.011699 | 0.570866 ± 0.011219 |
| D compressed features | SEX | 0.654781 ± 0.005798 | 0.602569 ± 0.010358 | 0.601294 ± 0.008748 | 0.648554 ± 0.010889 |
| D compressed features + LEACE | SEX | 0.675361 ± 0.004584 | 0.567908 ± 0.013539 | 0.566438 ± 0.012392 | 0.604554 ± 0.013242 |
| E PCA features | SEX | 0.655218 ± 0.011326 | 0.602646 ± 0.016481 | 0.602140 ± 0.015169 | 0.653168 ± 0.018182 |
| E PCA features + LEACE | SEX | 0.678120 ± 0.008021 | 0.574061 ± 0.009086 | 0.572712 ± 0.008572 | 0.609186 ± 0.011824 |
| F full covariates | SEX | 0.658396 ± 0.011732 | 0.607263 ± 0.007177 | 0.606392 ± 0.006870 | 0.651493 ± 0.014715 |
| Fitting prior | SEX | 0.692687 ± 0.000573 | 0.515005 ± 0.008432 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | RAC1P | 1.252065 ± 0.028860 | 0.566884 ± 0.010702 | undefined (1/3) | undefined (1/3) |
| A binary bank + LEACE | RAC1P | 1.289187 ± 0.025087 | 0.566884 ± 0.010702 | undefined (1/3) | undefined (1/3) |
| B rich neural bank | RAC1P | 1.243173 ± 0.032479 | 0.567332 ± 0.011427 | undefined (1/3) | undefined (1/3) |
| B rich neural bank + LEACE | RAC1P | 1.268265 ± 0.035414 | 0.571014 ± 0.015822 | undefined (1/3) | undefined (1/3) |
| C rich tree bank | RAC1P | 1.259321 ± 0.019889 | 0.567207 ± 0.008447 | undefined (1/3) | undefined (1/3) |
| C rich tree bank + LEACE | RAC1P | 1.290130 ± 0.014079 | 0.563067 ± 0.004561 | undefined (1/3) | undefined (1/3) |
| D compressed features | RAC1P | 1.219621 ± 0.035568 | 0.579352 ± 0.011089 | undefined (1/3) | undefined (1/3) |
| D compressed features + LEACE | RAC1P | 1.269058 ± 0.032043 | 0.572345 ± 0.010458 | undefined (1/3) | undefined (1/3) |
| E PCA features | RAC1P | 1.200399 ± 0.022578 | 0.584495 ± 0.010801 | undefined (1/3) | undefined (1/3) |
| E PCA features + LEACE | RAC1P | 1.235046 ± 0.027316 | 0.578140 ± 0.010818 | undefined (1/3) | undefined (1/3) |
| F full covariates | RAC1P | 1.200819 ± 0.026578 | 0.586605 ± 0.009726 | undefined (1/3) | undefined (1/3) |
| Fitting prior | RAC1P | 1.290043 ± 0.025603 | 0.566884 ± 0.010702 | undefined (1/3) | undefined (1/3) |

### Validation person weighted

| Release | Target | Log loss | Accuracy | Balanced accuracy | AUROC (binary) / macro AUROC (RAC1P) |
| --- | --- | --- | --- | --- | --- |
| A binary bank | Same residence | 0.504014 ± 0.014465 | 0.788013 ± 0.016022 | 0.504018 ± 0.003549 | 0.618802 ± 0.007899 |
| A binary bank + LEACE | Same residence | 0.517557 ± 0.016566 | 0.788923 ± 0.014129 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Same residence | 0.492122 ± 0.014357 | 0.786987 ± 0.012848 | 0.515021 ± 0.008379 | 0.659271 ± 0.017019 |
| B rich neural bank + LEACE | Same residence | 0.496954 ± 0.016750 | 0.782265 ± 0.014899 | 0.523743 ± 0.013218 | 0.649013 ± 0.014497 |
| C rich tree bank | Same residence | 0.500790 ± 0.017486 | 0.785226 ± 0.013520 | 0.516934 ± 0.008287 | 0.635650 ± 0.009540 |
| C rich tree bank + LEACE | Same residence | 0.511091 ± 0.012408 | 0.785611 ± 0.013531 | 0.507820 ± 0.007069 | 0.594711 ± 0.041453 |
| D compressed features | Same residence | 0.481279 ± 0.010474 | 0.784550 ± 0.011788 | 0.535812 ± 0.010017 | 0.690276 ± 0.011370 |
| D compressed features + LEACE | Same residence | 0.491828 ± 0.015999 | 0.784700 ± 0.010854 | 0.537775 ± 0.016960 | 0.664617 ± 0.005161 |
| E PCA features | Same residence | 0.475720 ± 0.012149 | 0.787201 ± 0.012720 | 0.541851 ± 0.014671 | 0.702784 ± 0.007865 |
| E PCA features + LEACE | Same residence | 0.485121 ± 0.010198 | 0.783190 ± 0.020014 | 0.530371 ± 0.026308 | 0.684352 ± 0.022762 |
| F full covariates | Same residence | 0.477421 ± 0.012920 | 0.788536 ± 0.012793 | 0.521737 ± 0.034664 | 0.700199 ± 0.013060 |
| Fitting prior | Same residence | 0.518076 ± 0.015817 | 0.788923 ± 0.014129 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Commute >20 min | 0.675915 ± 0.002623 | 0.573594 ± 0.018372 | 0.572693 ± 0.017306 | 0.594446 ± 0.010031 |
| A binary bank + LEACE | Commute >20 min | 0.693214 ± 0.000394 | 0.503233 ± 0.009610 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Commute >20 min | 0.673348 ± 0.002310 | 0.578031 ± 0.015786 | 0.577709 ± 0.014808 | 0.603130 ± 0.011095 |
| B rich neural bank + LEACE | Commute >20 min | 0.678599 ± 0.002211 | 0.565812 ± 0.004340 | 0.565065 ± 0.003843 | 0.592966 ± 0.006845 |
| C rich tree bank | Commute >20 min | 0.675934 ± 0.000926 | 0.576583 ± 0.009089 | 0.575860 ± 0.006141 | 0.595829 ± 0.005026 |
| C rich tree bank + LEACE | Commute >20 min | 0.682969 ± 0.003493 | 0.556888 ± 0.002032 | 0.556367 ± 0.002745 | 0.577485 ± 0.009912 |
| D compressed features | Commute >20 min | 0.673164 ± 0.003565 | 0.580510 ± 0.013068 | 0.580182 ± 0.010744 | 0.607508 ± 0.010890 |
| D compressed features + LEACE | Commute >20 min | 0.676502 ± 0.001965 | 0.568085 ± 0.008462 | 0.567593 ± 0.007358 | 0.595405 ± 0.006637 |
| E PCA features | Commute >20 min | 0.677199 ± 0.002434 | 0.567832 ± 0.010347 | 0.567876 ± 0.008933 | 0.604147 ± 0.010035 |
| E PCA features + LEACE | Commute >20 min | 0.681596 ± 0.003300 | 0.561704 ± 0.020953 | 0.561013 ± 0.019145 | 0.590639 ± 0.008967 |
| F full covariates | Commute >20 min | 0.680491 ± 0.005421 | 0.566694 ± 0.009390 | 0.566264 ± 0.008406 | 0.592777 ± 0.016156 |
| Fitting prior | Commute >20 min | 0.693288 ± 0.000509 | 0.503233 ± 0.009610 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Income >$50k | 0.286160 ± 0.010453 | 0.871469 ± 0.004882 | 0.747904 ± 0.016882 | 0.905341 ± 0.004157 |
| A binary bank + LEACE | Income >$50k | 0.473215 ± 0.013944 | 0.818897 ± 0.009406 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Income >$50k | 0.285877 ± 0.014648 | 0.871308 ± 0.003448 | 0.744947 ± 0.004237 | 0.904977 ± 0.006627 |
| B rich neural bank + LEACE | Income >$50k | 0.323649 ± 0.016605 | 0.863361 ± 0.012294 | 0.705196 ± 0.016943 | 0.878725 ± 0.007380 |
| C rich tree bank | Income >$50k | 0.283950 ± 0.007828 | 0.871867 ± 0.004859 | 0.753269 ± 0.006630 | 0.906458 ± 0.001634 |
| C rich tree bank + LEACE | Income >$50k | 0.339976 ± 0.007487 | 0.866879 ± 0.005038 | 0.706668 ± 0.022652 | 0.865367 ± 0.007868 |
| D compressed features | Income >$50k | 0.289914 ± 0.014824 | 0.869849 ± 0.006234 | 0.744273 ± 0.003395 | 0.902216 ± 0.006844 |
| D compressed features + LEACE | Income >$50k | 0.316664 ± 0.020248 | 0.865026 ± 0.006588 | 0.726797 ± 0.006125 | 0.879389 ± 0.010372 |
| E PCA features | Income >$50k | 0.302738 ± 0.011410 | 0.862912 ± 0.007021 | 0.728142 ± 0.012646 | 0.892567 ± 0.004584 |
| E PCA features + LEACE | Income >$50k | 0.317523 ± 0.016240 | 0.856648 ± 0.009105 | 0.703789 ± 0.018944 | 0.879239 ± 0.009346 |
| F full covariates | Income >$50k | 0.312348 ± 0.017191 | 0.865536 ± 0.005141 | 0.733370 ± 0.009775 | 0.889700 ± 0.004907 |
| Fitting prior | Income >$50k | 0.473620 ± 0.013674 | 0.818897 ± 0.009406 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Civilian at work | 0.290427 ± 0.003592 | 0.898907 ± 0.002989 | 0.844369 ± 0.006389 | 0.891138 ± 0.005005 |
| A binary bank + LEACE | Civilian at work | 0.619386 ± 0.010912 | 0.689583 ± 0.013587 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Civilian at work | 0.288589 ± 0.004723 | 0.899059 ± 0.002601 | 0.843587 ± 0.005276 | 0.891958 ± 0.006287 |
| B rich neural bank + LEACE | Civilian at work | 0.328837 ± 0.019095 | 0.886258 ± 0.004568 | 0.832103 ± 0.009132 | 0.877028 ± 0.003985 |
| C rich tree bank | Civilian at work | 0.291939 ± 0.006334 | 0.897845 ± 0.003083 | 0.845554 ± 0.006811 | 0.888622 ± 0.002097 |
| C rich tree bank + LEACE | Civilian at work | 0.356920 ± 0.005188 | 0.880649 ± 0.002374 | 0.829279 ± 0.006150 | 0.864895 ± 0.002856 |
| D compressed features | Civilian at work | 0.294220 ± 0.004719 | 0.895776 ± 0.003240 | 0.841213 ± 0.006013 | 0.887017 ± 0.004457 |
| D compressed features + LEACE | Civilian at work | 0.340604 ± 0.008493 | 0.880533 ± 0.003392 | 0.824747 ± 0.011610 | 0.870679 ± 0.006853 |
| E PCA features | Civilian at work | 0.303016 ± 0.002813 | 0.897782 ± 0.001721 | 0.841927 ± 0.010451 | 0.875482 ± 0.008546 |
| E PCA features + LEACE | Civilian at work | 0.338874 ± 0.015496 | 0.881028 ± 0.004887 | 0.825933 ± 0.002612 | 0.867142 ± 0.009099 |
| F full covariates | Civilian at work | 0.307378 ± 0.002073 | 0.894695 ± 0.004299 | 0.837295 ± 0.015142 | 0.871987 ± 0.008019 |
| Fitting prior | Civilian at work | 0.619460 ± 0.010784 | 0.689583 ± 0.013587 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Public coverage | 0.485578 ± 0.006392 | 0.779128 ± 0.001115 | 0.604119 ± 0.005993 | 0.739554 ± 0.016294 |
| A binary bank + LEACE | Public coverage | 0.558709 ± 0.011085 | 0.753146 ± 0.009966 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Public coverage | 0.486450 ± 0.005845 | 0.772760 ± 0.002951 | 0.591368 ± 0.006593 | 0.737237 ± 0.017977 |
| B rich neural bank + LEACE | Public coverage | 0.506618 ± 0.008300 | 0.764758 ± 0.005818 | 0.566560 ± 0.005310 | 0.712876 ± 0.014658 |
| C rich tree bank | Public coverage | 0.487827 ± 0.006838 | 0.775362 ± 0.007856 | 0.588738 ± 0.009457 | 0.736584 ± 0.010726 |
| C rich tree bank + LEACE | Public coverage | 0.511325 ± 0.006928 | 0.766021 ± 0.009241 | 0.561682 ± 0.017328 | 0.700509 ± 0.008886 |
| D compressed features | Public coverage | 0.489901 ± 0.007663 | 0.773345 ± 0.004789 | 0.586943 ± 0.007079 | 0.735371 ± 0.023440 |
| D compressed features + LEACE | Public coverage | 0.513711 ± 0.007251 | 0.768809 ± 0.008847 | 0.575249 ± 0.003656 | 0.696529 ± 0.007839 |
| E PCA features | Public coverage | 0.499454 ± 0.002775 | 0.770753 ± 0.003666 | 0.599369 ± 0.015880 | 0.720548 ± 0.016067 |
| E PCA features + LEACE | Public coverage | 0.516150 ± 0.001136 | 0.760141 ± 0.006681 | 0.562917 ± 0.016387 | 0.684082 ± 0.014464 |
| F full covariates | Public coverage | 0.509641 ± 0.003762 | 0.765260 ± 0.006138 | 0.571600 ± 0.020095 | 0.708534 ± 0.019983 |
| Fitting prior | Public coverage | 0.559243 ± 0.011605 | 0.753146 ± 0.009966 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | SEX | 0.685098 ± 0.006491 | 0.548892 ± 0.012610 | 0.539999 ± 0.014184 | 0.564218 ± 0.027557 |
| A binary bank + LEACE | SEX | 0.692023 ± 0.000324 | 0.526064 ± 0.007349 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | SEX | 0.661484 ± 0.010108 | 0.596452 ± 0.015991 | 0.592291 ± 0.016979 | 0.631908 ± 0.024826 |
| B rich neural bank + LEACE | SEX | 0.668700 ± 0.010633 | 0.581868 ± 0.020293 | 0.578804 ± 0.022840 | 0.617003 ± 0.029167 |
| C rich tree bank | SEX | 0.677006 ± 0.010934 | 0.573192 ± 0.024452 | 0.571733 ± 0.026203 | 0.597409 ± 0.030979 |
| C rich tree bank + LEACE | SEX | 0.683393 ± 0.004255 | 0.555984 ± 0.012600 | 0.553050 ± 0.015875 | 0.571405 ± 0.019719 |
| D compressed features | SEX | 0.659445 ± 0.011498 | 0.595049 ± 0.014311 | 0.592420 ± 0.015458 | 0.635372 ± 0.020385 |
| D compressed features + LEACE | SEX | 0.674298 ± 0.011054 | 0.571301 ± 0.019847 | 0.569147 ± 0.020942 | 0.602846 ± 0.027377 |
| E PCA features | SEX | 0.657087 ± 0.011125 | 0.599948 ± 0.011035 | 0.599380 ± 0.010914 | 0.644458 ± 0.022837 |
| E PCA features + LEACE | SEX | 0.678143 ± 0.012822 | 0.576863 ± 0.023751 | 0.574333 ± 0.023265 | 0.605278 ± 0.029932 |
| F full covariates | SEX | 0.660264 ± 0.014432 | 0.599339 ± 0.017450 | 0.597046 ± 0.017509 | 0.643374 ± 0.022743 |
| Fitting prior | SEX | 0.692325 ± 0.000243 | 0.526064 ± 0.007349 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | RAC1P | 1.249060 ± 0.005635 | 0.575175 ± 0.003031 | undefined (0/3) | undefined (0/3) |
| A binary bank + LEACE | RAC1P | 1.284628 ± 0.004193 | 0.575154 ± 0.003044 | undefined (0/3) | undefined (0/3) |
| B rich neural bank | RAC1P | 1.243811 ± 0.007761 | 0.578428 ± 0.005340 | undefined (0/3) | undefined (0/3) |
| B rich neural bank + LEACE | RAC1P | 1.266158 ± 0.021202 | 0.579618 ± 0.007375 | undefined (0/3) | undefined (0/3) |
| C rich tree bank | RAC1P | 1.254550 ± 0.009227 | 0.574619 ± 0.003513 | undefined (0/3) | undefined (0/3) |
| C rich tree bank + LEACE | RAC1P | 1.287516 ± 0.004167 | 0.572891 ± 0.004649 | undefined (0/3) | undefined (0/3) |
| D compressed features | RAC1P | 1.226986 ± 0.011573 | 0.589685 ± 0.004565 | undefined (0/3) | undefined (0/3) |
| D compressed features + LEACE | RAC1P | 1.267874 ± 0.008437 | 0.580809 ± 0.003200 | undefined (0/3) | undefined (0/3) |
| E PCA features | RAC1P | 1.214821 ± 0.012640 | 0.593446 ± 0.006613 | undefined (0/3) | undefined (0/3) |
| E PCA features + LEACE | RAC1P | 1.240136 ± 0.017379 | 0.587786 ± 0.001156 | undefined (0/3) | undefined (0/3) |
| F full covariates | RAC1P | 1.209533 ± 0.009107 | 0.589472 ± 0.008548 | undefined (0/3) | undefined (0/3) |
| Fitting prior | RAC1P | 1.284705 ± 0.004481 | 0.575154 ± 0.003044 | undefined (0/3) | undefined (0/3) |

### Development evaluation person weighted

| Release | Target | Log loss | Accuracy | Balanced accuracy | AUROC (binary) / macro AUROC (RAC1P) |
| --- | --- | --- | --- | --- | --- |
| A binary bank | Same residence | 0.495632 ± 0.006228 | 0.796335 ± 0.002110 | 0.505865 ± 0.004738 | 0.614530 ± 0.010598 |
| A binary bank + LEACE | Same residence | 0.510300 ± 0.003858 | 0.795226 ± 0.002907 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Same residence | 0.485311 ± 0.004346 | 0.793823 ± 0.005484 | 0.516351 ± 0.007916 | 0.656173 ± 0.003438 |
| B rich neural bank + LEACE | Same residence | 0.491478 ± 0.005218 | 0.788950 ± 0.008304 | 0.527058 ± 0.017551 | 0.644798 ± 0.012976 |
| C rich tree bank | Same residence | 0.490844 ± 0.006835 | 0.791587 ± 0.004323 | 0.511795 ± 0.007485 | 0.638580 ± 0.009314 |
| C rich tree bank + LEACE | Same residence | 0.505749 ± 0.002469 | 0.793977 ± 0.003669 | 0.507121 ± 0.008280 | 0.576700 ± 0.038546 |
| D compressed features | Same residence | 0.474088 ± 0.014636 | 0.791800 ± 0.009404 | 0.540296 ± 0.020239 | 0.687438 ± 0.025072 |
| D compressed features + LEACE | Same residence | 0.486891 ± 0.009380 | 0.789976 ± 0.001529 | 0.535752 ± 0.017261 | 0.665120 ± 0.011999 |
| E PCA features | Same residence | 0.464514 ± 0.011720 | 0.795434 ± 0.004143 | 0.549055 ± 0.021186 | 0.707653 ± 0.021239 |
| E PCA features + LEACE | Same residence | 0.472787 ± 0.006598 | 0.795030 ± 0.005725 | 0.534146 ± 0.029653 | 0.692862 ± 0.009024 |
| F full covariates | Same residence | 0.467068 ± 0.010009 | 0.796063 ± 0.002380 | 0.525631 ± 0.042475 | 0.712215 ± 0.011533 |
| Fitting prior | Same residence | 0.510614 ± 0.003348 | 0.795226 ± 0.002907 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Commute >20 min | 0.686282 ± 0.004945 | 0.553153 ± 0.016381 | 0.552812 ± 0.011873 | 0.564848 ± 0.009108 |
| A binary bank + LEACE | Commute >20 min | 0.692961 ± 0.000236 | 0.499653 ± 0.023173 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Commute >20 min | 0.684738 ± 0.004633 | 0.554099 ± 0.014895 | 0.548928 ± 0.015497 | 0.565415 ± 0.007056 |
| B rich neural bank + LEACE | Commute >20 min | 0.688564 ± 0.004083 | 0.542711 ± 0.013195 | 0.537430 ± 0.016042 | 0.559579 ± 0.008938 |
| C rich tree bank | Commute >20 min | 0.685216 ± 0.005507 | 0.556772 ± 0.017162 | 0.552002 ± 0.016944 | 0.571381 ± 0.020921 |
| C rich tree bank + LEACE | Commute >20 min | 0.692684 ± 0.004875 | 0.543019 ± 0.002202 | 0.538021 ± 0.000533 | 0.555770 ± 0.009800 |
| D compressed features | Commute >20 min | 0.684140 ± 0.001715 | 0.547603 ± 0.005980 | 0.543983 ± 0.005328 | 0.570105 ± 0.004255 |
| D compressed features + LEACE | Commute >20 min | 0.687249 ± 0.000729 | 0.545498 ± 0.010598 | 0.541118 ± 0.008605 | 0.561498 ± 0.003789 |
| E PCA features | Commute >20 min | 0.690615 ± 0.007991 | 0.549921 ± 0.015707 | 0.546679 ± 0.017127 | 0.563494 ± 0.020653 |
| E PCA features + LEACE | Commute >20 min | 0.692154 ± 0.007944 | 0.552752 ± 0.006577 | 0.549786 ± 0.007021 | 0.561384 ± 0.017757 |
| F full covariates | Commute >20 min | 0.692963 ± 0.009965 | 0.538021 ± 0.016067 | 0.535500 ± 0.016096 | 0.550398 ± 0.029719 |
| Fitting prior | Commute >20 min | 0.693108 ± 0.000489 | 0.499653 ± 0.023173 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Income >$50k | 0.290776 ± 0.009773 | 0.872336 ± 0.012107 | 0.753708 ± 0.021331 | 0.907551 ± 0.006220 |
| A binary bank + LEACE | Income >$50k | 0.484887 ± 0.010219 | 0.811006 ± 0.006873 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Income >$50k | 0.289414 ± 0.009718 | 0.874057 ± 0.012088 | 0.749783 ± 0.020539 | 0.906656 ± 0.006721 |
| B rich neural bank + LEACE | Income >$50k | 0.332625 ± 0.006531 | 0.858072 ± 0.005572 | 0.700598 ± 0.029093 | 0.876651 ± 0.008751 |
| C rich tree bank | Income >$50k | 0.290793 ± 0.012727 | 0.870805 ± 0.011587 | 0.749609 ± 0.020073 | 0.906942 ± 0.008461 |
| C rich tree bank + LEACE | Income >$50k | 0.343861 ± 0.023211 | 0.857306 ± 0.011027 | 0.693363 ± 0.009488 | 0.866072 ± 0.008285 |
| D compressed features | Income >$50k | 0.291523 ± 0.009301 | 0.871439 ± 0.010632 | 0.747410 ± 0.023298 | 0.904943 ± 0.006791 |
| D compressed features + LEACE | Income >$50k | 0.329022 ± 0.008877 | 0.855380 ± 0.006417 | 0.709930 ± 0.007986 | 0.873545 ± 0.003714 |
| E PCA features | Income >$50k | 0.310568 ± 0.008791 | 0.862559 ± 0.005658 | 0.727316 ± 0.008792 | 0.890815 ± 0.010209 |
| E PCA features + LEACE | Income >$50k | 0.330002 ± 0.007493 | 0.857409 ± 0.007850 | 0.708945 ± 0.010853 | 0.872954 ± 0.012382 |
| F full covariates | Income >$50k | 0.308338 ± 0.011964 | 0.862805 ± 0.006691 | 0.728799 ± 0.008203 | 0.893006 ± 0.012298 |
| Fitting prior | Income >$50k | 0.485039 ± 0.010056 | 0.811006 ± 0.006873 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Civilian at work | 0.267716 ± 0.015720 | 0.910322 ± 0.008003 | 0.859390 ± 0.014273 | 0.904191 ± 0.005109 |
| A binary bank + LEACE | Civilian at work | 0.615646 ± 0.001558 | 0.695344 ± 0.002759 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Civilian at work | 0.262001 ± 0.016606 | 0.910833 ± 0.009072 | 0.858988 ± 0.015456 | 0.907799 ± 0.006383 |
| B rich neural bank + LEACE | Civilian at work | 0.302972 ± 0.022398 | 0.899875 ± 0.008912 | 0.849931 ± 0.015241 | 0.886100 ± 0.011713 |
| C rich tree bank | Civilian at work | 0.265649 ± 0.011761 | 0.908382 ± 0.005844 | 0.859548 ± 0.011892 | 0.906389 ± 0.006331 |
| C rich tree bank + LEACE | Civilian at work | 0.314541 ± 0.010777 | 0.897800 ± 0.004367 | 0.850316 ± 0.009101 | 0.890037 ± 0.005392 |
| D compressed features | Civilian at work | 0.267737 ± 0.016342 | 0.908390 ± 0.010131 | 0.858196 ± 0.017141 | 0.903591 ± 0.008164 |
| D compressed features + LEACE | Civilian at work | 0.311292 ± 0.019782 | 0.894009 ± 0.010198 | 0.844220 ± 0.017344 | 0.888475 ± 0.014458 |
| E PCA features | Civilian at work | 0.277256 ± 0.012455 | 0.908927 ± 0.007070 | 0.857215 ± 0.012616 | 0.891243 ± 0.006392 |
| E PCA features + LEACE | Civilian at work | 0.309628 ± 0.007200 | 0.897170 ± 0.004286 | 0.844816 ± 0.008094 | 0.879959 ± 0.005011 |
| F full covariates | Civilian at work | 0.284740 ± 0.017131 | 0.906249 ± 0.010977 | 0.853099 ± 0.017538 | 0.883089 ± 0.007316 |
| Fitting prior | Civilian at work | 0.615531 ± 0.001928 | 0.695344 ± 0.002759 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | Public coverage | 0.493601 ± 0.015575 | 0.762051 ± 0.015959 | 0.589236 ± 0.001608 | 0.746572 ± 0.007952 |
| A binary bank + LEACE | Public coverage | 0.570704 ± 0.011804 | 0.742781 ± 0.010502 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | Public coverage | 0.492893 ± 0.015925 | 0.761706 ± 0.013324 | 0.584394 ± 0.009932 | 0.748352 ± 0.010642 |
| B rich neural bank + LEACE | Public coverage | 0.509596 ± 0.009402 | 0.759346 ± 0.009371 | 0.564335 ± 0.014410 | 0.723698 ± 0.010707 |
| C rich tree bank | Public coverage | 0.496851 ± 0.017326 | 0.764329 ± 0.016480 | 0.581647 ± 0.002420 | 0.739338 ± 0.007977 |
| C rich tree bank + LEACE | Public coverage | 0.519100 ± 0.011282 | 0.757123 ± 0.009809 | 0.558640 ± 0.012004 | 0.703999 ± 0.003630 |
| D compressed features | Public coverage | 0.495541 ± 0.015807 | 0.762510 ± 0.010711 | 0.584183 ± 0.010953 | 0.743013 ± 0.008588 |
| D compressed features + LEACE | Public coverage | 0.519516 ± 0.010780 | 0.759735 ± 0.009922 | 0.572352 ± 0.011028 | 0.705822 ± 0.015082 |
| E PCA features | Public coverage | 0.514111 ± 0.013080 | 0.756315 ± 0.020133 | 0.587080 ± 0.000597 | 0.717274 ± 0.000213 |
| E PCA features + LEACE | Public coverage | 0.530149 ± 0.012554 | 0.753727 ± 0.008904 | 0.560666 ± 0.007629 | 0.686946 ± 0.005262 |
| F full covariates | Public coverage | 0.515807 ± 0.019251 | 0.753422 ± 0.015092 | 0.563532 ± 0.001831 | 0.720174 ± 0.011026 |
| Fitting prior | Public coverage | 0.570347 ± 0.011212 | 0.742781 ± 0.010502 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | SEX | 0.687423 ± 0.001654 | 0.542643 ± 0.007894 | 0.531892 ± 0.007610 | 0.556125 ± 0.009770 |
| A binary bank + LEACE | SEX | 0.691903 ± 0.000618 | 0.525434 ± 0.006520 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| B rich neural bank | SEX | 0.667123 ± 0.006632 | 0.589656 ± 0.005683 | 0.584507 ± 0.004189 | 0.622860 ± 0.011377 |
| B rich neural bank + LEACE | SEX | 0.674760 ± 0.006309 | 0.578620 ± 0.014303 | 0.574717 ± 0.011749 | 0.608014 ± 0.008957 |
| C rich tree bank | SEX | 0.679560 ± 0.002528 | 0.571883 ± 0.006879 | 0.569471 ± 0.009240 | 0.599470 ± 0.008687 |
| C rich tree bank + LEACE | SEX | 0.687218 ± 0.001967 | 0.542221 ± 0.013830 | 0.537454 ± 0.008742 | 0.562674 ± 0.007645 |
| D compressed features | SEX | 0.659384 ± 0.004651 | 0.593358 ± 0.004843 | 0.590021 ± 0.002527 | 0.633993 ± 0.010476 |
| D compressed features + LEACE | SEX | 0.680021 ± 0.002458 | 0.560967 ± 0.008895 | 0.557860 ± 0.010154 | 0.587602 ± 0.012412 |
| E PCA features | SEX | 0.660511 ± 0.009960 | 0.596418 ± 0.016796 | 0.594673 ± 0.014498 | 0.641142 ± 0.016744 |
| E PCA features + LEACE | SEX | 0.685390 ± 0.005541 | 0.562481 ± 0.011337 | 0.559113 ± 0.012404 | 0.591691 ± 0.007248 |
| F full covariates | SEX | 0.664560 ± 0.013672 | 0.597206 ± 0.011048 | 0.594939 ± 0.011234 | 0.636894 ± 0.018052 |
| Fitting prior | SEX | 0.692113 ± 0.000765 | 0.525434 ± 0.006520 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |
| A binary bank | RAC1P | 1.248031 ± 0.045210 | 0.568724 ± 0.012209 | undefined (1/3) | undefined (1/3) |
| A binary bank + LEACE | RAC1P | 1.286011 ± 0.037928 | 0.568724 ± 0.012209 | undefined (1/3) | undefined (1/3) |
| B rich neural bank | RAC1P | 1.241852 ± 0.043678 | 0.569605 ± 0.013015 | undefined (1/3) | undefined (1/3) |
| B rich neural bank + LEACE | RAC1P | 1.266959 ± 0.044732 | 0.573922 ± 0.014999 | undefined (1/3) | undefined (1/3) |
| C rich tree bank | RAC1P | 1.255603 ± 0.037122 | 0.568781 ± 0.010068 | undefined (1/3) | undefined (1/3) |
| C rich tree bank + LEACE | RAC1P | 1.285837 ± 0.033122 | 0.566322 ± 0.008037 | undefined (1/3) | undefined (1/3) |
| D compressed features | RAC1P | 1.220797 ± 0.047922 | 0.578614 ± 0.013891 | undefined (1/3) | undefined (1/3) |
| D compressed features + LEACE | RAC1P | 1.268174 ± 0.047870 | 0.571157 ± 0.014259 | undefined (1/3) | undefined (1/3) |
| E PCA features | RAC1P | 1.203587 ± 0.039868 | 0.576852 ± 0.021465 | undefined (1/3) | undefined (1/3) |
| E PCA features + LEACE | RAC1P | 1.234764 ± 0.044637 | 0.573252 ± 0.019014 | undefined (1/3) | undefined (1/3) |
| F full covariates | RAC1P | 1.205614 ± 0.041461 | 0.579008 ± 0.016008 | undefined (1/3) | undefined (1/3) |
| Fitting prior | RAC1P | 1.287242 ± 0.039105 | 0.568724 ± 0.012209 | undefined (1/3) | undefined (1/3) |

## Matched-family paired log losses

These are within-family validation selections, separate from the primary minimum-log-loss candidate across families. Every restart/configuration and its validation-AUROC diagnostic selection appears in the raw candidate CSV. Paired CSV/JSON retain all erased-parent task/attribute metrics and person-weighted sensitivity. Feature/bank and compressed/PCA cross-comparisons retain primary residence/attribute contrasts; every other raw candidate score remains available separately.

### Validation

| Parent | Role | Family | Target | Erased − parent log loss |
| --- | --- | --- | --- | --- |
| A binary bank | transfer | logistic | Same residence | 0.006327 ± 0.001880 |
| A binary bank | transfer | logistic | Commute >20 min | 0.018244 ± 0.001213 |
| A binary bank | transfer | logistic | Income >$50k | 0.184758 ± 0.010356 |
| A binary bank | transfer | logistic | Civilian at work | 0.331771 ± 0.008758 |
| A binary bank | transfer | logistic | Public coverage | 0.078109 ± 0.009172 |
| A binary bank | transfer | mlp | Same residence | 0.012791 ± 0.004294 |
| A binary bank | transfer | mlp | Commute >20 min | 0.019252 ± 0.001298 |
| A binary bank | transfer | mlp | Income >$50k | 0.190637 ± 0.007572 |
| A binary bank | transfer | mlp | Civilian at work | 0.334876 ± 0.008718 |
| A binary bank | transfer | mlp | Public coverage | 0.078426 ± 0.009777 |
| A binary bank | audit | logistic | SEX | -0.000237 ± 0.001466 |
| A binary bank | audit | logistic | RAC1P | 0.023238 ± 0.007066 |
| A binary bank | audit | mlp | SEX | 0.005184 ± 0.005172 |
| A binary bank | audit | mlp | RAC1P | 0.031957 ± 0.006673 |
| A binary bank | audit | histgb | SEX | 0.000355 ± 0.010955 |
| A binary bank | audit | histgb | RAC1P | -0.120561 ± 0.017913 |
| B rich neural bank | transfer | logistic | Same residence | 0.013873 ± 0.005594 |
| B rich neural bank | transfer | logistic | Commute >20 min | 0.004354 ± 0.007168 |
| B rich neural bank | transfer | logistic | Income >$50k | 0.130229 ± 0.020197 |
| B rich neural bank | transfer | logistic | Civilian at work | 0.232819 ± 0.035640 |
| B rich neural bank | transfer | logistic | Public coverage | 0.069385 ± 0.006301 |
| B rich neural bank | transfer | mlp | Same residence | 0.004941 ± 0.005364 |
| B rich neural bank | transfer | mlp | Commute >20 min | 0.005314 ± 0.003360 |
| B rich neural bank | transfer | mlp | Income >$50k | 0.037290 ± 0.011318 |
| B rich neural bank | transfer | mlp | Civilian at work | 0.036262 ± 0.012606 |
| B rich neural bank | transfer | mlp | Public coverage | 0.021682 ± 0.006384 |
| B rich neural bank | audit | logistic | SEX | 0.022983 ± 0.003968 |
| B rich neural bank | audit | logistic | RAC1P | 0.047169 ± 0.006421 |
| B rich neural bank | audit | mlp | SEX | 0.007873 ± 0.002931 |
| B rich neural bank | audit | mlp | RAC1P | 0.023928 ± 0.013700 |
| B rich neural bank | audit | histgb | SEX | 0.012780 ± 0.008031 |
| B rich neural bank | audit | histgb | RAC1P | 0.016543 ± 0.015213 |
| C rich tree bank | transfer | logistic | Same residence | 0.007155 ± 0.002717 |
| C rich tree bank | transfer | logistic | Commute >20 min | 0.004579 ± 0.004395 |
| C rich tree bank | transfer | logistic | Income >$50k | 0.120182 ± 0.019379 |
| C rich tree bank | transfer | logistic | Civilian at work | 0.240170 ± 0.044156 |
| C rich tree bank | transfer | logistic | Public coverage | 0.065970 ± 0.007121 |
| C rich tree bank | transfer | mlp | Same residence | 0.010844 ± 0.003535 |
| C rich tree bank | transfer | mlp | Commute >20 min | 0.007652 ± 0.004550 |
| C rich tree bank | transfer | mlp | Income >$50k | 0.059253 ± 0.006325 |
| C rich tree bank | transfer | mlp | Civilian at work | 0.065443 ± 0.011909 |
| C rich tree bank | transfer | mlp | Public coverage | 0.023813 ± 0.007934 |
| C rich tree bank | audit | logistic | SEX | 0.009868 ± 0.003746 |
| C rich tree bank | audit | logistic | RAC1P | 0.027868 ± 0.006894 |
| C rich tree bank | audit | mlp | SEX | 0.007262 ± 0.004156 |
| C rich tree bank | audit | mlp | RAC1P | 0.034919 ± 0.005936 |
| C rich tree bank | audit | histgb | SEX | 0.014631 ± 0.004501 |
| C rich tree bank | audit | histgb | RAC1P | 0.032879 ± 0.015866 |
| D compressed features | transfer | logistic | Same residence | 0.013996 ± 0.007486 |
| D compressed features | transfer | logistic | Commute >20 min | 0.003429 ± 0.002893 |
| D compressed features | transfer | logistic | Income >$50k | 0.131130 ± 0.002939 |
| D compressed features | transfer | logistic | Civilian at work | 0.189102 ± 0.032385 |
| D compressed features | transfer | logistic | Public coverage | 0.059188 ± 0.006664 |
| D compressed features | transfer | mlp | Same residence | 0.010659 ± 0.008059 |
| D compressed features | transfer | mlp | Commute >20 min | 0.004965 ± 0.003217 |
| D compressed features | transfer | mlp | Income >$50k | 0.023157 ± 0.005146 |
| D compressed features | transfer | mlp | Civilian at work | 0.041686 ± 0.016089 |
| D compressed features | transfer | mlp | Public coverage | 0.023533 ± 0.007076 |
| D compressed features | audit | logistic | SEX | 0.031902 ± 0.002236 |
| D compressed features | audit | logistic | RAC1P | 0.072487 ± 0.005537 |
| D compressed features | audit | mlp | SEX | 0.015718 ± 0.002260 |
| D compressed features | audit | mlp | RAC1P | 0.050369 ± 0.009657 |
| D compressed features | audit | histgb | SEX | 0.014388 ± 0.010358 |
| D compressed features | audit | histgb | RAC1P | 0.008950 ± 0.008006 |
| E PCA features | transfer | logistic | Same residence | 0.020123 ± 0.006683 |
| E PCA features | transfer | logistic | Commute >20 min | 0.005472 ± 0.004484 |
| E PCA features | transfer | logistic | Income >$50k | 0.122099 ± 0.009773 |
| E PCA features | transfer | logistic | Civilian at work | 0.177381 ± 0.022621 |
| E PCA features | transfer | logistic | Public coverage | 0.051616 ± 0.011508 |
| E PCA features | transfer | mlp | Same residence | 0.008837 ± 0.004364 |
| E PCA features | transfer | mlp | Commute >20 min | 0.002537 ± 0.001580 |
| E PCA features | transfer | mlp | Income >$50k | 0.014682 ± 0.008560 |
| E PCA features | transfer | mlp | Civilian at work | 0.034434 ± 0.014273 |
| E PCA features | transfer | mlp | Public coverage | 0.020382 ± 0.005078 |
| E PCA features | audit | logistic | SEX | 0.035757 ± 0.003917 |
| E PCA features | audit | logistic | RAC1P | 0.082368 ± 0.012655 |
| E PCA features | audit | mlp | SEX | 0.018851 ± 0.006316 |
| E PCA features | audit | mlp | RAC1P | 0.031674 ± 0.008336 |
| E PCA features | audit | histgb | SEX | 0.015760 ± 0.000610 |
| E PCA features | audit | histgb | RAC1P | 0.020369 ± 0.006160 |

### Development evaluation

| Parent | Role | Family | Target | Erased − parent log loss |
| --- | --- | --- | --- | --- |
| A binary bank | transfer | logistic | Same residence | 0.006457 ± 0.003827 |
| A binary bank | transfer | logistic | Commute >20 min | 0.010121 ± 0.002668 |
| A binary bank | transfer | logistic | Income >$50k | 0.197109 ± 0.010838 |
| A binary bank | transfer | logistic | Civilian at work | 0.351311 ± 0.002056 |
| A binary bank | transfer | logistic | Public coverage | 0.078435 ± 0.004740 |
| A binary bank | transfer | mlp | Same residence | 0.013959 ± 0.004573 |
| A binary bank | transfer | mlp | Commute >20 min | 0.010896 ± 0.003210 |
| A binary bank | transfer | mlp | Income >$50k | 0.202508 ± 0.010088 |
| A binary bank | transfer | mlp | Civilian at work | 0.354970 ± 0.004036 |
| A binary bank | transfer | mlp | Public coverage | 0.078182 ± 0.006252 |
| A binary bank | audit | logistic | SEX | 0.000579 ± 0.000346 |
| A binary bank | audit | logistic | RAC1P | 0.029680 ± 0.003790 |
| A binary bank | audit | mlp | SEX | 0.003723 ± 0.001929 |
| A binary bank | audit | mlp | RAC1P | 0.037122 ± 0.003774 |
| A binary bank | audit | histgb | SEX | -0.002210 ± 0.006380 |
| A binary bank | audit | histgb | RAC1P | -0.106169 ± 0.015771 |
| B rich neural bank | transfer | logistic | Same residence | 0.014716 ± 0.002813 |
| B rich neural bank | transfer | logistic | Commute >20 min | 0.004604 ± 0.005838 |
| B rich neural bank | transfer | logistic | Income >$50k | 0.146662 ± 0.028635 |
| B rich neural bank | transfer | logistic | Civilian at work | 0.241754 ± 0.025713 |
| B rich neural bank | transfer | logistic | Public coverage | 0.065711 ± 0.007367 |
| B rich neural bank | transfer | mlp | Same residence | 0.005475 ± 0.005323 |
| B rich neural bank | transfer | mlp | Commute >20 min | 0.004761 ± 0.001545 |
| B rich neural bank | transfer | mlp | Income >$50k | 0.040623 ± 0.003466 |
| B rich neural bank | transfer | mlp | Civilian at work | 0.044987 ± 0.015168 |
| B rich neural bank | transfer | mlp | Public coverage | 0.018906 ± 0.011301 |
| B rich neural bank | audit | logistic | SEX | 0.024389 ± 0.006086 |
| B rich neural bank | audit | logistic | RAC1P | 0.052016 ± 0.003998 |
| B rich neural bank | audit | mlp | SEX | 0.006745 ± 0.001090 |
| B rich neural bank | audit | mlp | RAC1P | 0.025093 ± 0.004825 |
| B rich neural bank | audit | histgb | SEX | 0.000690 ± 0.008034 |
| B rich neural bank | audit | histgb | RAC1P | 0.034958 ± 0.003279 |
| C rich tree bank | transfer | logistic | Same residence | 0.012098 ± 0.005699 |
| C rich tree bank | transfer | logistic | Commute >20 min | 0.004883 ± 0.002891 |
| C rich tree bank | transfer | logistic | Income >$50k | 0.125292 ± 0.029366 |
| C rich tree bank | transfer | logistic | Civilian at work | 0.250171 ± 0.055211 |
| C rich tree bank | transfer | logistic | Public coverage | 0.068190 ± 0.005672 |
| C rich tree bank | transfer | mlp | Same residence | 0.013532 ± 0.003283 |
| C rich tree bank | transfer | mlp | Commute >20 min | 0.008567 ± 0.003085 |
| C rich tree bank | transfer | mlp | Income >$50k | 0.046552 ± 0.008760 |
| C rich tree bank | transfer | mlp | Civilian at work | 0.055761 ± 0.009459 |
| C rich tree bank | transfer | mlp | Public coverage | 0.023434 ± 0.004523 |
| C rich tree bank | audit | logistic | SEX | 0.011133 ± 0.000833 |
| C rich tree bank | audit | logistic | RAC1P | 0.031209 ± 0.000134 |
| C rich tree bank | audit | mlp | SEX | 0.008771 ± 0.001588 |
| C rich tree bank | audit | mlp | RAC1P | 0.030809 ± 0.006764 |
| C rich tree bank | audit | histgb | SEX | 0.019803 ± 0.007600 |
| C rich tree bank | audit | histgb | RAC1P | 0.046983 ± 0.012236 |
| D compressed features | transfer | logistic | Same residence | 0.016210 ± 0.007081 |
| D compressed features | transfer | logistic | Commute >20 min | 0.003234 ± 0.004257 |
| D compressed features | transfer | logistic | Income >$50k | 0.137869 ± 0.006130 |
| D compressed features | transfer | logistic | Civilian at work | 0.193651 ± 0.016546 |
| D compressed features | transfer | logistic | Public coverage | 0.057247 ± 0.001542 |
| D compressed features | transfer | mlp | Same residence | 0.010694 ± 0.006842 |
| D compressed features | transfer | mlp | Commute >20 min | 0.003322 ± 0.002217 |
| D compressed features | transfer | mlp | Income >$50k | 0.026584 ± 0.005824 |
| D compressed features | transfer | mlp | Civilian at work | 0.039569 ± 0.006958 |
| D compressed features | transfer | mlp | Public coverage | 0.021947 ± 0.008209 |
| D compressed features | audit | logistic | SEX | 0.032753 ± 0.006365 |
| D compressed features | audit | logistic | RAC1P | 0.077860 ± 0.012785 |
| D compressed features | audit | mlp | SEX | 0.020581 ± 0.001749 |
| D compressed features | audit | mlp | RAC1P | 0.049438 ± 0.017733 |
| D compressed features | audit | histgb | SEX | 0.013459 ± 0.007255 |
| D compressed features | audit | histgb | RAC1P | 0.009573 ± 0.007701 |
| E PCA features | transfer | logistic | Same residence | 0.017615 ± 0.010774 |
| E PCA features | transfer | logistic | Commute >20 min | 0.003417 ± 0.005937 |
| E PCA features | transfer | logistic | Income >$50k | 0.136435 ± 0.008012 |
| E PCA features | transfer | logistic | Civilian at work | 0.206642 ± 0.007899 |
| E PCA features | transfer | logistic | Public coverage | 0.052800 ± 0.002290 |
| E PCA features | transfer | mlp | Same residence | 0.004051 ± 0.003466 |
| E PCA features | transfer | mlp | Commute >20 min | 0.001516 ± 0.004048 |
| E PCA features | transfer | mlp | Income >$50k | 0.014136 ± 0.003972 |
| E PCA features | transfer | mlp | Civilian at work | 0.033619 ± 0.011581 |
| E PCA features | transfer | mlp | Public coverage | 0.016690 ± 0.004954 |
| E PCA features | audit | logistic | SEX | 0.038352 ± 0.003755 |
| E PCA features | audit | logistic | RAC1P | 0.099434 ± 0.010457 |
| E PCA features | audit | mlp | SEX | 0.022902 ± 0.003636 |
| E PCA features | audit | mlp | RAC1P | 0.034648 ± 0.004741 |
| E PCA features | audit | histgb | SEX | 0.007935 ± 0.008080 |
| E PCA features | audit | histgb | RAC1P | 0.004634 ± 0.010037 |

## Runtime and invariants

| Seed | Total s | Parent load s | Erasure s | Heads s | Audits s | Controls s | Evaluation s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 56.626288 | 1.315480 | 0.107390 | 5.376826 | 39.982001 | 2.125575 | 7.679106 |
| 1 | 55.922648 | 1.488532 | 0.153312 | 5.464489 | 38.995810 | 2.188888 | 7.580829 |
| 2 | 57.194772 | 1.295719 | 0.110809 | 5.119300 | 40.610380 | 2.063289 | 7.956387 |

Sum of recorded internal runtimes: 169.743708 seconds. Wall-clock execution timing is recorded by the orchestrator separately. All loaded per-seed integrity flags are retained in summary.json, including source/map/output hashes and selection-before-evaluation times.

No source model, map, task, or protection strength is selected by these comparisons. Empirical covariance removal and bounded-auditor failure do not establish privacy. A task-probability bank is a practical reference, not an information lower bound. Commute eligibility itself may reveal information beyond the audited numerical release.
