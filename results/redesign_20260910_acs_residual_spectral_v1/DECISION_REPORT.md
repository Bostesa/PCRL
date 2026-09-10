# Residual spectral development decision

Status: **complete_development_matrix**; 42/42 evaluation units. This is the already studied California 2018 cohort, not independent confirmation. All 24 maps were frozen before downstream evaluation. Spectral auxiliaries are float64; historical neural float32 auxiliaries are promoted losslessly. Wires are float64 and coordinate counts are matched where r=16, but underlying numerical precision is not matched. Anchors are never requantized. No privacy guarantee follows from these empirical attacks or finite moments.

Residual preservation added residence capability beyond H: 8/8 spectral recipes exceed the .01 gain threshold in every seed and both weightings. It did not establish a robust method advance over frozen J: no recipe passes the full source-and-sensitive vector requirements, no recipe is nominated, and coalition conditioning fails the fixed coordination criterion.
The exact source-service probability vectors remain preserved on every release. Failures below concern separately fitted source probes versus the legacy PCA32 allowance; they do not mean the immutable source service changed. [PER_SEED.csv.gz](PUBLICATION_EVIDENCE/PER_SEED.csv.gz) provides the complete per-seed, endpoint, scope, budget and weighting table.

## 1. Did residual preservation add useful capability beyond H?

The table reports mean ± sample SD over available original seeds; complete results require all three, and partial values show their n. The .01 improvement over H and original PCA/rich-bank half-headroom are distinct criteria; source allowance requires each original PCA source-probe loss + .01, checked separately on validation and development.

| Arm | Unweighted residence gain vs H | PWGTP residence gain vs H | .01 in every seed/weight |
|---|---:|---:|---|
| H | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | reference |
| E | 0.025273 ± 0.005724 | 0.023320 ± 0.003824 | reference |
| A0 | 0.031798 ± 0.004500 | 0.028307 ± 0.004035 | reference |
| L025 | 0.018098 ± 0.002184 | 0.016422 ± 0.002260 | reference |
| L20 | 0.017847 ± 0.006742 | 0.015357 ± 0.010347 | reference |
| J | 0.021528 ± 0.006914 | 0.019516 ± 0.004526 | reference |
| spectral_S0 | 0.031920 ± 0.004781 | 0.028355 ± 0.004030 | True |
| spectral_M025 | 0.028023 ± 0.006324 | 0.026402 ± 0.005597 | True |
| spectral_M1 | 0.030618 ± 0.001396 | 0.028370 ± 0.004170 | True |
| spectral_L025 | 0.029405 ± 0.001039 | 0.025414 ± 0.002623 | True |
| spectral_L1 | 0.028786 ± 0.001330 | 0.026037 ± 0.002909 | True |
| spectral_C025 | 0.030400 ± 0.002206 | 0.026442 ± 0.003211 | True |
| spectral_C1 | 0.026480 ± 0.003776 | 0.024189 ± 0.003587 | True |
| spectral_L2 | 0.028537 ± 0.005089 | 0.025998 ± 0.000926 | True |

Separate legacy criteria (development headroom count and source preservation on both validation and development):

| Arm | Weight | Source pass seeds | Original half-headroom pass seeds |
|---|---|---:|---:|
| H | unweighted | 3/3 | 0/3 |
| H | person_weighted | 3/3 | 0/3 |
| E | unweighted | 3/3 | 0/3 |
| E | person_weighted | 3/3 | 0/3 |
| A0 | unweighted | 3/3 | 2/3 |
| A0 | person_weighted | 3/3 | 0/3 |
| L025 | unweighted | 3/3 | 0/3 |
| L025 | person_weighted | 3/3 | 0/3 |
| L20 | unweighted | 3/3 | 1/3 |
| L20 | person_weighted | 3/3 | 0/3 |
| J | unweighted | 3/3 | 0/3 |
| J | person_weighted | 3/3 | 0/3 |
| spectral_S0 | unweighted | 0/3 | 3/3 |
| spectral_S0 | person_weighted | 1/3 | 0/3 |
| spectral_M025 | unweighted | 1/3 | 1/3 |
| spectral_M025 | person_weighted | 1/3 | 0/3 |
| spectral_M1 | unweighted | 0/3 | 1/3 |
| spectral_M1 | person_weighted | 1/3 | 1/3 |
| spectral_L025 | unweighted | 0/3 | 1/3 |
| spectral_L025 | person_weighted | 1/3 | 0/3 |
| spectral_L1 | unweighted | 0/3 | 1/3 |
| spectral_L1 | person_weighted | 1/3 | 0/3 |
| spectral_C025 | unweighted | 1/3 | 2/3 |
| spectral_C025 | person_weighted | 1/3 | 0/3 |
| spectral_C1 | unweighted | 1/3 | 0/3 |
| spectral_C1 | person_weighted | 1/3 | 0/3 |
| spectral_L2 | unweighted | 0/3 | 1/3 |
| spectral_L2 | person_weighted | 1/3 | 0/3 |

Original half-headroom results are in [ORIGINAL_CRITERIA.csv.gz](PUBLICATION_EVIDENCE/ORIGINAL_CRITERIA.csv.gz); no H denominator replaces PCA32. Source failures cannot cancel across tasks, seeds or weighting.

## 2. Did a spectral method improve on J and simpler alternatives?

Fixed-protocol nomination: **none**. Qualifying recipes: []. No mean-based or per-seed recipe selection is used.
All eight recipes versus J/H/E/A0, historical J versus controls, and fixed main comparisons are in [PAIRED.csv.gz](PUBLICATION_EVIDENCE/PAIRED.csv.gz) and [PAIRED_AGGREGATE.csv.gz](PUBLICATION_EVIDENCE/PAIRED_AGGREGATE.csv.gz). Close matches preserve all four original panels and deltas; directional vectors retain each A/B/AB SEX/race role separately.

## 3. Did coalition conditioning beat equal-setting and equal-mass local controls?

Fixed C1 versus L1 and C1 versus L2 coordination criterion: **False**. The same all-seed/both-weight vector rule and one common strict endpoint apply to each comparison; C025 versus L025 is a separately reported sensitivity comparison.
[DIRECTIONAL_VECTORS.csv.gz](PUBLICATION_EVIDENCE/DIRECTIONAL_VECTORS.csv.gz) exposes local race differences directly. A coalition gain does not compensate for worse local race. [DECISION.json](DECISION.json) records every threshold and failing component.

## 4. Did it survive nonlinear attacks, both weightings and local race?

The common kernel expanded catch-up360 scope is used for the decision, including bounded Gaussian RFF ridge attackers and all legal historical ancestors. Original no-kernel scopes and independent-only scopes remain separate. A spectral method has no historical saved observer; unequal inherited history is disclosed. Every candidate, including every fixed ancestor, has its own score row; no development-best ancestor is substituted. Absent-class recall/AUROC and full-schema protection remain unassessable. Finite observed-distribution log-loss comparisons still follow the prospective numerical vector rule; full class coverage is reported separately, never added as an absolute nomination gate.

| Arm | Weight | A/race additional recovery | AB/race additional recovery | AB/SEX additional recovery |
|---|---|---:|---:|---:|
| J | unweighted | 0.007143 ± 0.003456 | 0.005694 ± 0.005086 | 0.001109 ± 0.005129 |
| J | person_weighted | 0.006181 ± 0.006481 | 0.005948 ± 0.005155 | 0.000740 ± 0.003001 |
| spectral_S0 | unweighted | 0.065454 ± 0.005515 | 0.049962 ± 0.009829 | 0.023691 ± 0.004114 |
| spectral_S0 | person_weighted | 0.062679 ± 0.013727 | 0.048871 ± 0.014913 | 0.021312 ± 0.007737 |
| spectral_M025 | unweighted | 0.048061 ± 0.004049 | 0.035729 ± 0.009375 | 0.019583 ± 0.003712 |
| spectral_M025 | person_weighted | 0.045752 ± 0.009388 | 0.030525 ± 0.017516 | 0.019006 ± 0.006764 |
| spectral_M1 | unweighted | 0.047420 ± 0.006559 | 0.034065 ± 0.007084 | 0.019091 ± 0.003421 |
| spectral_M1 | person_weighted | 0.045245 ± 0.015400 | 0.028405 ± 0.014742 | 0.017148 ± 0.007828 |
| spectral_L025 | unweighted | 0.053387 ± 0.006705 | 0.045322 ± 0.010535 | 0.019428 ± 0.003574 |
| spectral_L025 | person_weighted | 0.051254 ± 0.013400 | 0.041414 ± 0.015406 | 0.018257 ± 0.005938 |
| spectral_L1 | unweighted | 0.038106 ± 0.004238 | 0.037382 ± 0.011288 | 0.014934 ± 0.002698 |
| spectral_L1 | person_weighted | 0.035617 ± 0.010272 | 0.033436 ± 0.015684 | 0.013883 ± 0.006198 |
| spectral_C025 | unweighted | 0.050201 ± 0.002962 | 0.040170 ± 0.013526 | 0.016855 ± 0.005606 |
| spectral_C025 | person_weighted | 0.047085 ± 0.009704 | 0.036035 ± 0.016526 | 0.016473 ± 0.007487 |
| spectral_C1 | unweighted | 0.027448 ± 0.007386 | 0.023115 ± 0.003852 | 0.009784 ± 0.002386 |
| spectral_C1 | person_weighted | 0.023584 ± 0.015901 | 0.018414 ± 0.012540 | 0.007853 ± 0.005308 |
| spectral_L2 | unweighted | 0.032757 ± 0.005312 | 0.023688 ± 0.008103 | 0.010103 ± 0.002978 |
| spectral_L2 | person_weighted | 0.029118 ± 0.012579 | 0.018838 ± 0.012986 | 0.008766 ± 0.006025 |

Validation-selected attacks sometimes generalize worse than the H attack. Negative signed additional recovery is retained exactly, without clipping or development reselection. Counts below cover all eleven forbidden roles per seed and condition; H is excluded from its own comparisons. Parenthesized counts are below -1e-12, separating numerical roundoff from strictly negative values.

| Scope | Budget | Weight | Spectral negative / total (below -1e-12) | Historical augmented negative / total (below -1e-12) |
|---|---:|---|---:|---:|
| standard_independent | 120 | unweighted | 0/264 (0) | 8/165 (8) |
| standard_independent | 120 | person_weighted | 1/264 (1) | 8/165 (8) |
| standard_independent | 360 | unweighted | 0/264 (0) | 8/165 (8) |
| standard_independent | 360 | person_weighted | 1/264 (1) | 9/165 (9) |
| expanded_independent | 120 | unweighted | 0/264 (0) | 12/165 (12) |
| expanded_independent | 120 | person_weighted | 1/264 (1) | 15/165 (15) |
| expanded_independent | 360 | unweighted | 0/264 (0) | 12/165 (12) |
| expanded_independent | 360 | person_weighted | 1/264 (1) | 15/165 (15) |
| expanded_catchup | 120 | unweighted | 0/264 (0) | 9/165 (9) |
| expanded_catchup | 120 | person_weighted | 1/264 (1) | 18/165 (18) |
| expanded_catchup | 360 | unweighted | 0/264 (0) | 9/165 (9) |
| expanded_catchup | 360 | person_weighted | 1/264 (1) | 18/165 (18) |
| kernel_standard_independent | 120 | unweighted | 0/264 (0) | 9/165 (9) |
| kernel_standard_independent | 120 | person_weighted | 1/264 (1) | 7/165 (7) |
| kernel_standard_independent | 360 | unweighted | 0/264 (0) | 9/165 (9) |
| kernel_standard_independent | 360 | person_weighted | 1/264 (1) | 8/165 (8) |
| kernel_expanded_independent | 120 | unweighted | 0/264 (0) | 13/165 (13) |
| kernel_expanded_independent | 120 | person_weighted | 1/264 (1) | 14/165 (14) |
| kernel_expanded_independent | 360 | unweighted | 0/264 (0) | 13/165 (13) |
| kernel_expanded_independent | 360 | person_weighted | 1/264 (1) | 14/165 (14) |
| kernel_expanded_catchup | 120 | unweighted | 0/264 (0) | 9/165 (9) |
| kernel_expanded_catchup | 120 | person_weighted | 1/264 (1) | 15/165 (15) |
| kernel_expanded_catchup | 360 | unweighted | 0/264 (0) | 9/165 (9) |
| kernel_expanded_catchup | 360 | person_weighted | 1/264 (1) | 15/165 (15) |

[NEGATIVE_INCREMENT_COUNTS.csv.gz](PUBLICATION_EVIDENCE/NEGATIVE_INCREMENT_COUNTS.csv.gz) also gives endpoint-specific negative counts. These are descriptive generalization differences of frozen validation selections, not evidence that removing information improves an optimal attacker.

Training versus heldout residual moments use the same fixed training attribute-trace normalization and fixed local/coalition denominators 3/2. Training uses grouped OOF nuisances; heldout uses the equal ensemble of three frozen nuisance models, without refitting. Values are finite-surrogate diagnostics, separate from the empirical recovery table; a low average does not establish privacy, and unknown/zero training normalizers stay explicit.

| Arm | Role | Train normalized moment | Development normalized moment |
|---|---|---:|---:|
| spectral_S0 | local | 0.390602 (n=3) | 0.445599 (n=3) |
| spectral_S0 | coalition | 0.263729 (n=3) | 0.322119 (n=3) |
| spectral_C1 | local | 0.040716 (n=3) | 0.152830 (n=3) |
| spectral_C1 | coalition | 0.039318 (n=3) | 0.162326 (n=3) |
| spectral_L1 | local | 0.065465 (n=3) | 0.180266 (n=3) |
| spectral_L1 | coalition | 0.101797 (n=3) | 0.209469 (n=3) |
| spectral_L2 | local | 0.033252 (n=3) | 0.150330 (n=3) |
| spectral_L2 | coalition | 0.075518 (n=3) | 0.190725 (n=3) |

[SURROGATE_DIAGNOSTICS.csv.gz](PUBLICATION_EVIDENCE/SURROGATE_DIAGNOSTICS.csv.gz) reports every arm, class-support limitations, attribute and fixed role mean for representation fitting, source validation and development, with raw projected norms and separately labeled empirical recovery.
C1 versus spectral_L1: vector nonworsening=False; residence strict in all six seed/weight pairs=False; fixed sensitive endpoints strict in all six=['A/RAC1P', 'AB/SEX']. Each comparator must pass independently; the strict endpoint may differ between comparators.
C1 versus spectral_L2: vector nonworsening=False; residence strict in all six seed/weight pairs=False; fixed sensitive endpoints strict in all six=[]. Each comparator must pass independently; the strict endpoint may differ between comparators.

Fixed coordination comparisons, mean paired difference (left minus right; lower is better for every displayed component), common kernel expanded catch-up360:

| Left / right | Weight | Residence loss | AB/SEX recovery | AB/race recovery | A/race recovery |
|---|---|---:|---:|---:|---:|
| spectral_C1 / spectral_L1 | unweighted | 0.002306 (n=3) | -0.005150 (n=3) | -0.014267 (n=3) | -0.010658 (n=3) |
| spectral_C1 / spectral_L1 | person_weighted | 0.001849 (n=3) | -0.006030 (n=3) | -0.015022 (n=3) | -0.012033 (n=3) |
| spectral_C1 / spectral_L2 | unweighted | 0.002057 (n=3) | -0.000319 (n=3) | -0.000573 (n=3) | -0.005309 (n=3) |
| spectral_C1 / spectral_L2 | person_weighted | 0.001810 (n=3) | -0.000913 (n=3) | -0.000423 (n=3) | -0.005534 (n=3) |
| spectral_C025 / spectral_L025 | unweighted | -0.000995 (n=3) | -0.002573 (n=3) | -0.005153 (n=3) | -0.003186 (n=3) |
| spectral_C025 / spectral_L025 | person_weighted | -0.001028 (n=3) | -0.001784 (n=3) | -0.005378 (n=3) | -0.004169 (n=3) |

The original historical J residence gains versus H were approximately .021528 unweighted and .019516 PWGTP; its additional AB/SEX recovery was approximately .001109 and .000740, respectively. These values belong to the original audit context. Recomputed original versus common-kernel comparisons follow:

| Weight | J original expanded catch-up AB/SEX | J common kernel expanded catch-up AB/SEX |
|---|---:|---:|
| unweighted | 0.001109 ± 0.005129 | 0.001109 ± 0.005129 |
| person_weighted | 0.000740 ± 0.003001 | 0.000740 ± 0.003001 |

[PER_CLASS.csv.gz](PUBLICATION_EVIDENCE/PER_CLASS.csv.gz) and [ALL_CANDIDATES.csv.gz](PUBLICATION_EVIDENCE/ALL_CANDIDATES.csv.gz) retain fixed class support, recall/AUC availability and complete candidate scores. The original and kernel scopes are not pooled.

## 5. Did actual withholding explain the trade-off?

H remains visible. A label-independent Bernoulli branch is visible to A and AB and fixed per person/release with the same draw for A/AB. B remains exactly the original coverage vector with no branch indicator; routed B predictors are identical to H. Each p=0,.25,.5,.75,1 has routed saved H/augmented predictors; expected per-person log losses are mixed before averaging, never probabilities or channels. Utility is an average over people with unequal branches. No p is selected.
[WITHHOLDING.csv.gz](PUBLICATION_EVIDENCE/WITHHOLDING.csv.gz) gives all five utility and eleven forbidden roles, both budgets, six scopes, both weightings and both splits; sampled routing and score replay appear in [WITHHOLDING_VALIDATION.json](WITHHOLDING_VALIDATION.json). Aligned person-loss vectors and branch uniforms are retained in local-only compressed archives whose hashes are listed there; individual rows are not published. [WITHHOLDING_COMPARISONS.csv.gz](PUBLICATION_EVIDENCE/WITHHOLDING_COMPARISONS.csv.gz) gives componentwise spectral comparisons at every fixed p.

- Spectral → withholding, spectral_S0: 0/120 seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.
  Fixed withholding mechanisms that directionally dominate spectral_S0 in all three seeds and both weightings: []. Every listed p remains descriptive; none is nominated.
- Spectral → withholding, spectral_M025: 0/120 seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.
  Fixed withholding mechanisms that directionally dominate spectral_M025 in all three seeds and both weightings: []. Every listed p remains descriptive; none is nominated.
- Spectral → withholding, spectral_M1: 0/120 seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.
  Fixed withholding mechanisms that directionally dominate spectral_M1 in all three seeds and both weightings: []. Every listed p remains descriptive; none is nominated.
- Spectral → withholding, spectral_L025: 0/120 seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.
  Fixed withholding mechanisms that directionally dominate spectral_L025 in all three seeds and both weightings: []. Every listed p remains descriptive; none is nominated.
- Spectral → withholding, spectral_L1: 0/120 seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.
  Fixed withholding mechanisms that directionally dominate spectral_L1 in all three seeds and both weightings: []. Every listed p remains descriptive; none is nominated.
- Spectral → withholding, spectral_C025: 0/120 seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.
  Fixed withholding mechanisms that directionally dominate spectral_C025 in all three seeds and both weightings: []. Every listed p remains descriptive; none is nominated.
- Spectral → withholding, spectral_C1: 0/120 seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.
  Fixed withholding mechanisms that directionally dominate spectral_C1 in all three seeds and both weightings: []. Every listed p remains descriptive; none is nominated.
- Spectral → withholding, spectral_L2: 0/120 seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.
  Fixed withholding mechanisms that directionally dominate spectral_L2 in all three seeds and both weightings: []. Every listed p remains descriptive; none is nominated.
Plain result under the full source-feasible five-utility/six-sensitive-role directional rule at delta=.001: spectral dominates withholding in 0/960 individual fixed comparisons; withholding dominates spectral in 6/960. No fixed withholding mechanism dominates a spectral recipe across all seeds and both weightings. The curves therefore do not establish a robust full-vector advantage in either direction; higher residence utility and lower sensitive recovery remain separate components.

Every per-seed trade-off figure has four sensitive endpoints, all spectral arms and all withholding curves; PNG and PDF variants are listed in [FIGURES.json](FIGURES.json).

## 6. What limitation remains, and what one next study resolves it?

No recipe meets the fixed nomination rule. Source failures: ['spectral_S0', 'spectral_M025', 'spectral_M1', 'spectral_L025', 'spectral_L1', 'spectral_C025', 'spectral_C1', 'spectral_L2']; insufficient all-seed residence gains: []; failed J vector/strict comparison: ['spectral_S0', 'spectral_M025', 'spectral_M1', 'spectral_L025', 'spectral_L1', 'spectral_C025', 'spectral_C1', 'spectral_L2']. These are separate limitations, not averaged away.
One next study is an explicitly labeled limitation study on the admitted California 2017 cohort: preregister the already frozen C1, L2, J and H systems, frozen plus fresh attacks, and fixed H-defined strata; use disjoint household fitting, selection and final partitions to test whether residual-moment blind spots and residence gains transport. This is not a new recipe nomination or a successful-method confirmation, does not expand the grid, and is not launched here. The present 2018 scores remain reused development evidence; reshuffling them would not create independent confirmation.

Concrete diagnostic evidence for that limitation:
- spectral_C1: normalized local moment mean train=0.04071568013150385, development=0.1528299490680231; additional local-race recovery unweighted=0.027448 ± 0.007386, PWGTP=0.023584 ± 0.015901. The finite-surrogate and attack columns measure different quantities and are not interchangeable guarantees.
- spectral_L2: normalized local moment mean train=0.033252127788184586, development=0.15033044764057726; additional local-race recovery unweighted=0.032757 ± 0.005312, PWGTP=0.029118 ± 0.012579. The finite-surrogate and attack columns measure different quantities and are not interchangeable guarantees.

Finite trace optimality, exact-source LP feasibility, and empirical attack resistance are different claims. The mathematical counterexamples show why zero first moments are insufficient for unrestricted conditional privacy. [MATHEMATICS.md](MATHEMATICS.md) and [NUMERICAL_SUMMARY.json](NUMERICAL_SUMMARY.json) contain the derivation and synthetic checks. Means/sample SD are descriptive; three seeds are not independent population replications. Independent pre-report review aligned the decision implementation with the prospective finite-log-loss rule: full-schema class support remains a separate limitation, not an additional nomination criterion. No individual rows are published in these aggregate tables.
