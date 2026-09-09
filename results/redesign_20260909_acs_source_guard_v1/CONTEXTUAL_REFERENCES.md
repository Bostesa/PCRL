# Contextual reference losses

These historical releases do not implement the new A/B policy and cannot replace its matched F/P comparisons. Rich and source-only banks differ in purpose routing, source-loss weights, interfaces and inherited exposure; the new objective uses source weights .25/.25/.5. Full original source identities remain in [CONTEXTUAL.csv](CONTEXTUAL.csv). Original PCA32 is the fixed parent for existing margins.

Each cell is mean ± sample SD over three cohort-sharing seeds. U and PWGTP score the same original validation-selected predictions. Utility losses are lower-is-better; **attacker log losses are higher when that selected predictor recovers less**. These are raw losses, not the gains in the primary decision table. Attack rows use historical independent selections; their actual budget is shown explicitly. No new fits, fresh holdout, matched two-purpose routing, or complete race support is implied.

## Unweighted development evaluation

| Reference | Income | Employment | Coverage | Residence | Commute |
| --- | --- | --- | --- | --- | --- |
| Original PCA32 | 0.298578 ± 0.008781 | 0.288407 ± 0.005005 | 0.504623 ± 0.011550 | 0.487411 ± 0.005595 | 0.687570 ± 0.008575 |
| Original PCA16 | 0.294408 ± 0.007520 | 0.290035 ± 0.007223 | 0.505704 ± 0.008631 | 0.495062 ± 0.002095 | 0.685527 ± 0.003931 |
| PCA32 + LEACE | 0.312714 ± 0.009209 | 0.322018 ± 0.009292 | 0.521313 ± 0.013445 | 0.496275 ± 0.004211 | 0.691074 ± 0.007655 |
| Rich neural bank | 0.278421 ± 0.007137 | 0.270941 ± 0.011699 | 0.487067 ± 0.012910 | 0.510529 ± 0.003504 | 0.681481 ± 0.002771 |
| Rich tree bank | 0.279997 ± 0.009591 | 0.274990 ± 0.006996 | 0.488114 ± 0.014003 | 0.517356 ± 0.002162 | 0.681863 ± 0.004760 |
| E source-only bank | 0.320373 ± 0.012973 | 0.350527 ± 0.021481 | 0.504200 ± 0.016816 | 0.521901 ± 0.002726 | 0.686863 ± 0.003923 |
| Sham source-only bank | 0.307972 ± 0.004674 | 0.326115 ± 0.007231 | 0.505659 ± 0.011927 | 0.526645 ± 0.004278 | 0.684814 ± 0.003661 |

| Reference | SEX attack log loss | RAC1P attack log loss | Actual epochs SEX / RAC1P |
| --- | --- | --- | --- |
| Original PCA32 | 0.655218 ± 0.011326 | 1.200399 ± 0.022578 | 120 / 120 |
| Original PCA16 | 0.659948 ± 0.004457 | 1.212077 ± 0.020218 | 360 / 360 |
| PCA32 + LEACE | 0.678120 ± 0.008021 | 1.235046 ± 0.027316 | 120 / 120 |
| Rich neural bank | 0.664770 ± 0.008175 | 1.243173 ± 0.032479 | 360 / 360 |
| Rich tree bank | 0.676694 ± 0.002856 | 1.259321 ± 0.019889 | 360 / 360 |
| E source-only bank | 0.686329 ± 0.002479 | 1.264819 ± 0.032789 | 360 / 360 |
| Sham source-only bank | 0.682769 ± 0.006032 | 1.270108 ± 0.030825 | 360 / 360 |

## PWGTP development evaluation

| Reference | Income | Employment | Coverage | Residence | Commute |
| --- | --- | --- | --- | --- | --- |
| Original PCA32 | 0.310568 ± 0.008791 | 0.277256 ± 0.012455 | 0.514111 ± 0.013080 | 0.464514 ± 0.011720 | 0.690615 ± 0.007991 |
| Original PCA16 | 0.304320 ± 0.009139 | 0.278038 ± 0.010757 | 0.511918 ± 0.010256 | 0.474544 ± 0.009318 | 0.687659 ± 0.003119 |
| PCA32 + LEACE | 0.330002 ± 0.007493 | 0.309628 ± 0.007200 | 0.530149 ± 0.012554 | 0.472787 ± 0.006598 | 0.692154 ± 0.007944 |
| Rich neural bank | 0.289414 ± 0.009718 | 0.262001 ± 0.016606 | 0.492893 ± 0.015925 | 0.485311 ± 0.004346 | 0.684738 ± 0.004633 |
| Rich tree bank | 0.290793 ± 0.012727 | 0.265649 ± 0.011761 | 0.496851 ± 0.017326 | 0.490844 ± 0.006835 | 0.685216 ± 0.005507 |
| E source-only bank | 0.330090 ± 0.008391 | 0.332763 ± 0.015387 | 0.510623 ± 0.016492 | 0.497591 ± 0.005694 | 0.690143 ± 0.006111 |
| Sham source-only bank | 0.321943 ± 0.004059 | 0.307991 ± 0.008263 | 0.511866 ± 0.012992 | 0.500173 ± 0.007289 | 0.686255 ± 0.003885 |

| Reference | SEX attack log loss | RAC1P attack log loss | Actual epochs SEX / RAC1P |
| --- | --- | --- | --- |
| Original PCA32 | 0.660511 ± 0.009960 | 1.203587 ± 0.039868 | 120 / 120 |
| Original PCA16 | 0.661744 ± 0.001436 | 1.212406 ± 0.037497 | 360 / 360 |
| PCA32 + LEACE | 0.685390 ± 0.005541 | 1.234764 ± 0.044637 | 120 / 120 |
| Rich neural bank | 0.667123 ± 0.006632 | 1.241852 ± 0.043678 | 360 / 360 |
| Rich tree bank | 0.679560 ± 0.002528 | 1.255603 ± 0.037122 | 360 / 360 |
| E source-only bank | 0.686088 ± 0.002616 | 1.262973 ± 0.047558 | 360 / 360 |
| Sham source-only bank | 0.682131 ± 0.004533 | 1.267222 ± 0.048918 | 360 / 360 |

[All raw reference aggregates](CONTEXTUAL_AGGREGATE.csv) retain additional single-interface arms and original audit scopes. [Every contextual paired seed contrast](CONTEXTUAL_PAIRED.csv.gz) and [paired means/SDs](CONTEXTUAL_PAIRED_AGGREGATE.csv.gz) preserve the actual historical budget and scope. They are descriptive comparisons, not causal replacements or statistical noninferiority tests.
