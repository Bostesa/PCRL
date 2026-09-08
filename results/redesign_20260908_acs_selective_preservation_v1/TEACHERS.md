# Direct teacher evidence

R is the original raw PCA16. E fits joint SEX2/RAC1P9 LEACE on real representation-fitting labels. S uses one frozen permutation of complete-case SEX/RAC1P pairs, with joint counts and missing masks preserved. The concatenated concept schema has11 columns, not18 intersections. All observers, D gradients and audits use real labels. No teacher is updated toward a student.

Original raw-PCA16 fitting scales are used throughout; no erased-teacher whitening or residual-variance normalization. E/S are attribute-informed targets and contain no reserved-task labels. S need not match E in realized rank or distortion. A sham contrast cannot automatically identify attribute specificity independently of those geometric differences.

[Full teacher geometry/maps/permutation evidence](TEACHER_GEOMETRY.csv), [direct matching errors](DIRECT_MATCHING.csv), [all teacher candidates](PER_TARGET.csv).

| Seed | Teacher | Projection retained rank | Movement MSE | Joint counts preserved | Masks preserved |
| --- | --- | --- | --- | --- | --- |
| 0 | R | 16 | 0.000000 | True | True |
| 0 | E | 7 | 0.562500 | True | True |
| 0 | S | 7 | 0.562500 | True | True |
| 1 | R | 16 | 0.000000 | True | True |
| 1 | E | 7 | 0.562500 | True | True |
| 1 | S | 7 | 0.562500 | True | True |
| 2 | R | 16 | 0.000000 | True | True |
| 2 | E | 8 | 0.500000 | True | True |
| 2 | S | 8 | 0.500000 | True | True |

## development evaluation

| Teacher | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX independent360 | RAC1P independent360 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R: raw PCA16 | 0.495062 ± 0.002095 | 0.685527 ± 0.003931 | 0.294408 ± 0.007520 | 0.290035 ± 0.007223 | 0.505704 ± 0.008631 | 0.659948 ± 0.004457 | 1.212077 ± 0.020218 |
| E: real-attribute teacher | 0.508385 ± 0.002038 | 0.687243 ± 0.004055 | 0.356662 ± 0.017555 | 0.414513 ± 0.055286 | 0.529162 ± 0.025627 | 0.683950 ± 0.002900 | 1.241935 ± 0.026872 |
| S: permuted-label teacher | 0.513080 ± 0.003917 | 0.687644 ± 0.003623 | 0.337047 ± 0.029978 | 0.381357 ± 0.018584 | 0.524608 ± 0.012703 | 0.676475 ± 0.008809 | 1.234979 ± 0.027421 |

## development evaluation person weighted

| Teacher | Same residence | Commute >20 min | Income >$50k | Civilian at work | Public coverage | SEX independent360 | RAC1P independent360 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R: raw PCA16 | 0.474544 ± 0.009318 | 0.687659 ± 0.003119 | 0.304320 ± 0.009139 | 0.278038 ± 0.010757 | 0.511918 ± 0.010256 | 0.661744 ± 0.001436 | 1.212406 ± 0.037497 |
| E: real-attribute teacher | 0.486163 ± 0.005844 | 0.689689 ± 0.002638 | 0.369383 ± 0.019375 | 0.399205 ± 0.057150 | 0.535277 ± 0.022231 | 0.686239 ± 0.003189 | 1.244789 ± 0.040376 |
| S: permuted-label teacher | 0.488508 ± 0.011704 | 0.689102 ± 0.002258 | 0.349703 ± 0.028876 | 0.364471 ± 0.017891 | 0.532308 ± 0.012062 | 0.677857 ± 0.008042 | 1.238451 ± 0.051109 |

Direct teachers are independent interfaces. Their audits are not attributed to I/W or to learned students. Small fitting covariance with labels is not a nonlinear recovery guarantee.
