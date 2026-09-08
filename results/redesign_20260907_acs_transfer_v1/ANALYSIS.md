# ACS transfer evidence summary

This report summarizes the frozen scientific comparison. See [RESEARCH_DECISION.md](RESEARCH_DECISION.md) for the research interpretation and next-step decision. No release is selected using final test outcomes. The comparison concerns two specified withheld task identities on contemporaneous survey records; it does not establish reusable utility for arbitrary tasks, longitudinal prediction, privacy protection, or PCRL novelty.

## Paired selected-head comparisons

Differences are D_features minus the named comparator on identical test examples. Negative log-loss differences favor D; positive AUROC/accuracy differences favor D. The predeclared −0.01-nat reference is descriptive, not a significance test. Fitting and selection remain unweighted.

| Task | Comparison | Seed | D / comparator head | Δ log loss | Δ AUROC | Δ balanced acc. | Δ accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| commute_over20 | D_features minus B_rich_bank | 0 | mlp / mlp | 0.003569 | -0.005542 | 0.000186 | -0.001015 |
| commute_over20 | D_features minus C_tree_bank | 0 | mlp / mlp | -0.000888 | 0.002577 | 0.004417 | 0.004059 |
| commute_over20 | D_features minus E_pca | 0 | mlp / logistic | -0.008073 | -0.000231 | 0.004777 | 0.005074 |
| commute_over20 | D_features minus F_covariates | 0 | mlp / mlp | -0.010522 | 0.027336 | 0.017844 | 0.018772 |
| same_residence | D_features minus B_rich_bank | 0 | mlp / mlp | -0.004596 | 0.012481 | 0.023086 | 0.003018 |
| same_residence | D_features minus C_tree_bank | 0 | mlp / mlp | -0.012076 | 0.034905 | 0.021243 | 0.002683 |
| same_residence | D_features minus E_pca | 0 | mlp / mlp | 0.012025 | -0.025516 | 0.010103 | -0.003689 |
| same_residence | D_features minus F_covariates | 0 | mlp / mlp | 0.017565 | -0.040919 | 0.042749 | -0.004359 |
| commute_over20 | D_features minus B_rich_bank | 1 | mlp / mlp | -0.000571 | 0.001624 | -0.005612 | -0.005440 |
| commute_over20 | D_features minus C_tree_bank | 1 | mlp / mlp | 0.000857 | -0.010094 | -0.003834 | -0.003956 |
| commute_over20 | D_features minus E_pca | 1 | mlp / logistic | 0.000109 | -0.013362 | -0.013351 | -0.011869 |
| commute_over20 | D_features minus F_covariates | 1 | mlp / mlp | -0.000136 | -0.012957 | 0.003809 | 0.004946 |
| same_residence | D_features minus B_rich_bank | 1 | logistic / mlp | -0.014570 | 0.038116 | -0.000512 | 0.002322 |
| same_residence | D_features minus C_tree_bank | 1 | logistic / mlp | -0.015786 | 0.042723 | -0.006216 | 0.003648 |
| same_residence | D_features minus E_pca | 1 | logistic / logistic | 0.018600 | -0.035934 | -0.038242 | 0.000995 |
| same_residence | D_features minus F_covariates | 1 | logistic / mlp | 0.012601 | -0.029990 | 0.010881 | 0.004312 |
| commute_over20 | D_features minus B_rich_bank | 2 | mlp / mlp | -0.002082 | 0.008896 | 0.004026 | 0.007161 |
| commute_over20 | D_features minus C_tree_bank | 2 | mlp / mlp | -0.000199 | 0.005268 | -0.000368 | 0.003581 |
| commute_over20 | D_features minus E_pca | 2 | mlp / mlp | -0.009388 | 0.020262 | 0.006722 | 0.010742 |
| commute_over20 | D_features minus F_covariates | 2 | mlp / mlp | -0.006855 | 0.018000 | 0.007038 | 0.012276 |
| same_residence | D_features minus B_rich_bank | 2 | logistic / logistic | -0.016133 | 0.035329 | 0.016271 | 0.000671 |
| same_residence | D_features minus C_tree_bank | 2 | logistic / mlp | -0.027920 | 0.060716 | 0.042516 | 0.010406 |
| same_residence | D_features minus E_pca | 2 | logistic / logistic | 0.003427 | -0.013879 | -0.030057 | -0.003021 |
| same_residence | D_features minus F_covariates | 2 | logistic / logistic | -0.005820 | -0.012854 | -0.036037 | -0.004364 |

### Paired means ± sample SD

| Task | Comparison | Δ log loss | Δ AUROC | Δ balanced acc. | Δ accuracy |
| --- | --- | --- | --- | --- | --- |
| commute_over20 | D_features minus B_rich_bank | 0.000305 ± 0.002926 | 0.001660 ± 0.007219 | -0.000467 ± 0.004852 | 0.000235 ± 0.006393 |
| commute_over20 | D_features minus C_tree_bank | -0.000077 ± 0.000879 | -0.000750 ± 0.008204 | 0.000072 ± 0.004143 | 0.001228 ± 0.004496 |
| commute_over20 | D_features minus E_pca | -0.005784 ± 0.005146 | 0.002223 ± 0.016946 | -0.000617 ± 0.011071 | 0.001315 ± 0.011765 |
| commute_over20 | D_features minus F_covariates | -0.005838 ± 0.005267 | 0.010793 ± 0.021091 | 0.009564 ± 0.007350 | 0.011998 ± 0.006917 |
| same_residence | D_features minus B_rich_bank | -0.011767 ± 0.006259 | 0.028642 ± 0.014065 | 0.012948 ± 0.012145 | 0.002004 ± 0.001205 |
| same_residence | D_features minus C_tree_bank | -0.018594 ± 0.008287 | 0.046115 ± 0.013235 | 0.019181 ± 0.024431 | 0.005579 ± 0.004208 |
| same_residence | D_features minus E_pca | 0.011351 ± 0.007609 | -0.025110 ± 0.011033 | -0.019399 ± 0.025875 | -0.001905 ± 0.002534 |
| same_residence | D_features minus F_covariates | 0.008115 ± 0.012321 | -0.027921 ± 0.014146 | 0.005864 ± 0.039632 | -0.001471 ± 0.005008 |

### Matched-family transfer checks

Both members of each pair use the named family, with each model checkpoint still selected on its own validation data. All per-seed matched-family differences are retained in summary.json.

| Task | Comparison | Head family | Δ log loss | Δ AUROC |
| --- | --- | --- | --- | --- |
| commute_over20 | D_features minus B_rich_bank | logistic | -0.004892 ± 0.006408 | 0.011399 ± 0.019528 |
| commute_over20 | D_features minus B_rich_bank | mlp | 0.000305 ± 0.002926 | 0.001660 ± 0.007219 |
| commute_over20 | D_features minus C_tree_bank | logistic | -0.004651 ± 0.002671 | 0.007134 ± 0.013938 |
| commute_over20 | D_features minus C_tree_bank | mlp | -0.000077 ± 0.000879 | -0.000750 ± 0.008204 |
| commute_over20 | D_features minus E_pca | logistic | -0.005391 ± 0.004717 | 0.005592 ± 0.016151 |
| commute_over20 | D_features minus E_pca | mlp | -0.007772 ± 0.005590 | 0.005650 ± 0.013455 |
| commute_over20 | D_features minus F_covariates | logistic | -0.022335 ± 0.021145 | 0.009873 ± 0.019553 |
| commute_over20 | D_features minus F_covariates | mlp | -0.005838 ± 0.005267 | 0.010793 ± 0.021091 |
| same_residence | D_features minus B_rich_bank | logistic | -0.011641 ± 0.006721 | 0.028373 ± 0.014824 |
| same_residence | D_features minus B_rich_bank | mlp | -0.005880 ± 0.002607 | 0.013157 ± 0.004680 |
| same_residence | D_features minus C_tree_bank | logistic | -0.022359 ± 0.009412 | 0.061499 ± 0.020472 |
| same_residence | D_features minus C_tree_bank | mlp | -0.013464 ± 0.008861 | 0.033405 ± 0.019240 |
| same_residence | D_features minus E_pca | logistic | 0.013669 ± 0.008872 | -0.034454 ± 0.019876 |
| same_residence | D_features minus E_pca | mlp | 0.011668 ± 0.011394 | -0.031424 ± 0.022960 |
| same_residence | D_features minus F_covariates | logistic | 0.007306 ± 0.011550 | -0.037004 ± 0.023073 |
| same_residence | D_features minus F_covariates | mlp | 0.013826 ± 0.011516 | -0.037939 ± 0.022950 |

## PWGTP sensitivity on unchanged selected predictions

Weighted scores are sensitivity analyses of this sampled benchmark cohort, not official survey estimates. No weighted validation score exists because selection was unweighted.

| Role | Task | Release | Weighted LL | Weighted AUROC | Weighted balanced acc. | Weighted accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| audit | RAC1P | A_binary_bank | 1.252286 ± 0.042261 | undefined (1/3 defined) | undefined (1/3 defined) | 0.568724 ± 0.012209 |
| audit | SEX | A_binary_bank | 0.694848 ± 0.010946 | 0.549831 ± 0.004323 | 0.529035 ± 0.008620 | 0.540435 ± 0.003357 |
| audit | RAC1P | B_rich_bank | 1.240958 ± 0.044928 | undefined (1/3 defined) | undefined (1/3 defined) | 0.569081 ± 0.012518 |
| audit | SEX | B_rich_bank | 0.668015 ± 0.007100 | 0.621255 ± 0.012386 | 0.583703 ± 0.004686 | 0.587491 ± 0.007594 |
| audit | RAC1P | C_tree_bank | 1.256237 ± 0.036773 | undefined (1/3 defined) | undefined (1/3 defined) | 0.567340 ± 0.009299 |
| audit | SEX | C_tree_bank | 0.680926 ± 0.000926 | 0.594346 ± 0.008085 | 0.567900 ± 0.007819 | 0.569933 ± 0.005685 |
| audit | RAC1P | D_compressed | 1.221852 ± 0.046836 | undefined (1/3 defined) | undefined (1/3 defined) | 0.578051 ± 0.013203 |
| audit | SEX | D_compressed | 0.660440 ± 0.003892 | 0.631996 ± 0.008517 | 0.587980 ± 0.001008 | 0.590924 ± 0.001979 |
| audit | RAC1P | D_features | 1.229414 ± 0.045707 | undefined (1/3 defined) | undefined (1/3 defined) | 0.580193 ± 0.009286 |
| audit | SEX | D_features | 0.663661 ± 0.001668 | 0.622314 ± 0.014664 | 0.577271 ± 0.015959 | 0.580895 ± 0.012171 |
| audit | RAC1P | E_pca | 1.206228 ± 0.038253 | undefined (1/3 defined) | undefined (1/3 defined) | 0.576189 ± 0.017992 |
| audit | SEX | E_pca | 0.659754 ± 0.008941 | 0.641881 ± 0.015603 | 0.593488 ± 0.016182 | 0.595287 ± 0.018271 |
| audit | RAC1P | F_covariates | 1.208150 ± 0.045849 | undefined (1/3 defined) | undefined (1/3 defined) | 0.579021 ± 0.015985 |
| audit | SEX | F_covariates | 0.668340 ± 0.017514 | 0.634811 ± 0.021677 | 0.600212 ± 0.012216 | 0.602564 ± 0.010615 |
| audit | RAC1P | exposed | 0.001464 ± 0.000400 | undefined (1/3 defined) | undefined (1/3 defined) | 0.999921 ± 0.000137 |
| audit | SEX | exposed | 0.000094 ± 0.000013 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| audit | RAC1P | prior | 1.287242 ± 0.039105 | undefined (1/3 defined) | undefined (1/3 defined) | 0.568724 ± 0.012209 |
| audit | SEX | prior | 0.692113 ± 0.000765 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.525434 ± 0.006520 |
| transfer | commute_over20 | A_binary_bank | 0.686282 ± 0.004945 | 0.564848 ± 0.009108 | 0.552812 ± 0.011873 | 0.553153 ± 0.016381 |
| transfer | same_residence | A_binary_bank | 0.495632 ± 0.006228 | 0.614530 ± 0.010598 | 0.505865 ± 0.004738 | 0.796335 ± 0.002110 |
| transfer | commute_over20 | B_rich_bank | 0.684738 ± 0.004633 | 0.565415 ± 0.007056 | 0.548928 ± 0.015497 | 0.554099 ± 0.014895 |
| transfer | same_residence | B_rich_bank | 0.485311 ± 0.004346 | 0.656173 ± 0.003438 | 0.516351 ± 0.007916 | 0.793823 ± 0.005484 |
| transfer | commute_over20 | C_tree_bank | 0.685216 ± 0.005507 | 0.571381 ± 0.020921 | 0.552002 ± 0.016944 | 0.556772 ± 0.017162 |
| transfer | same_residence | C_tree_bank | 0.490844 ± 0.006835 | 0.638580 ± 0.009314 | 0.511795 ± 0.007485 | 0.791587 ± 0.004323 |
| transfer | commute_over20 | D_compressed | 0.684140 ± 0.001715 | 0.570105 ± 0.004255 | 0.543983 ± 0.005328 | 0.547603 ± 0.005980 |
| transfer | same_residence | D_compressed | 0.474088 ± 0.014636 | 0.687438 ± 0.025072 | 0.540296 ± 0.020239 | 0.791800 ± 0.009404 |
| transfer | commute_over20 | D_features | 0.684998 ± 0.002874 | 0.568151 ± 0.003202 | 0.550311 ± 0.010371 | 0.556823 ± 0.007558 |
| transfer | same_residence | D_features | 0.474978 ± 0.009784 | 0.682416 ± 0.019221 | 0.530229 ± 0.021297 | 0.796363 ± 0.003224 |
| transfer | commute_over20 | E_pca | 0.690615 ± 0.007991 | 0.563494 ± 0.020653 | 0.546679 ± 0.017127 | 0.549921 ± 0.015707 |
| transfer | same_residence | E_pca | 0.464514 ± 0.011720 | 0.707653 ± 0.021239 | 0.549055 ± 0.021186 | 0.795434 ± 0.004143 |
| transfer | commute_over20 | F_covariates | 0.692963 ± 0.009965 | 0.550398 ± 0.029719 | 0.535500 ± 0.016096 | 0.538021 ± 0.016067 |
| transfer | same_residence | F_covariates | 0.467068 ± 0.010009 | 0.712215 ± 0.011533 | 0.525631 ± 0.042475 | 0.796063 ± 0.002380 |
| transfer | commute_over20 | prior | 0.693108 ± 0.000489 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.499653 ± 0.023173 |
| transfer | same_residence | prior | 0.510614 ± 0.003348 | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 | 0.795226 ± 0.002907 |

## Coverage, controls, and interpretation limits

PER_TARGET.csv retains every candidate family, its saved selected flag, fitting support/fallback, unweighted validation and test metrics, weighted test sensitivity, prevalence and missing counts. PER_CLASS.csv retains every fixed-schema class, including absent classes. Observed-class summaries have distinct names and do not replace undefined full-schema balanced accuracy or macro AUROC.

Prior controls use only designated fitting-label frequencies with the declared smoothing. Exposed SEX/RAC1P one-hot controls test attacker competence; they are diagnostics with label access. Raw covariates and PCA are information-retention references. Probability-vector simplex redundancy means matching stored dimensions does not match intrinsic capacity.

The three splits reuse a common sampled cohort, so their sample SD is descriptive rather than independent-sample or survey-design uncertainty. Age, hours, and retained covariates may correlate with source tasks and audited attributes. Attribute recoverability is measured without a protection objective or privacy pass/fail threshold.

Completed seeds: [0, 1, 2]. Declared seeds: [0, 1, 2]. Missing completed seed records: [].
Candidate prior fallbacks recorded: 0. Source/release/selection integrity passed for every completed seed: True.

Undefined aggregates are not averaged away: summary.json records n_defined and n_seeds; a mean is null if any contributing seed is undefined. A one-seed sample SD is undefined.

## Runtime and provenance

Sum of recorded per-seed total runtime: 154.854754 seconds. Process wall timings, schema preparation, and artificial plumbing-check costs are separate artifacts. Configuration and exact input file hashes are copied into summary.json; raw records and fitted objects are never opened by this script.
