# Per-anchor heterogeneity (anchors are not independent samples)

Absolute expected log loss of the validation-selected predictor on the final pool, per anchor. The primary estimand is the equal mean of the three anchors.

| Family | Role | Weighting | Anchor 0 | Anchor 1 | Anchor 2 | Mean |
|---|---|---|---:|---:|---:|---:|
| H | attack:A/SEX | unweighted | 0.66794 | 0.66894 | 0.67191 | 0.66960 |
| H | attack:A/SEX | PWGTP | 0.67302 | 0.67315 | 0.67639 | 0.67419 |
| H | attack:A/RAC1P | unweighted | 1.24338 | 1.25298 | 1.24531 | 1.24723 |
| H | attack:A/RAC1P | PWGTP | 1.25336 | 1.25970 | 1.25513 | 1.25606 |
| H | attack:AB/SEX | unweighted | 0.66151 | 0.66373 | 0.66649 | 0.66391 |
| H | attack:AB/SEX | PWGTP | 0.66595 | 0.66726 | 0.67057 | 0.66792 |
| H | attack:AB/RAC1P | unweighted | 1.22236 | 1.23430 | 1.22840 | 1.22835 |
| H | attack:AB/RAC1P | PWGTP | 1.23569 | 1.24302 | 1.23986 | 1.23952 |
| H | utility:A/same_residence | unweighted | 0.51749 | 0.51675 | 0.51835 | 0.51753 |
| H | utility:A/same_residence | PWGTP | 0.51431 | 0.51122 | 0.51356 | 0.51303 |
| J | attack:A/SEX | unweighted | 0.65116 | 0.65329 | 0.65173 | 0.65206 |
| J | attack:A/SEX | PWGTP | 0.65643 | 0.65629 | 0.65704 | 0.65659 |
| J | attack:A/RAC1P | unweighted | 1.20082 | 1.20445 | 1.20540 | 1.20356 |
| J | attack:A/RAC1P | PWGTP | 1.21077 | 1.21224 | 1.21734 | 1.21345 |
| J | attack:AB/SEX | unweighted | 0.64913 | 0.65492 | 0.64960 | 0.65122 |
| J | attack:AB/SEX | PWGTP | 0.65421 | 0.65677 | 0.65437 | 0.65511 |
| J | attack:AB/RAC1P | unweighted | 1.19873 | 1.20445 | 1.20032 | 1.20116 |
| J | attack:AB/RAC1P | PWGTP | 1.20681 | 1.21224 | 1.20951 | 1.20952 |
| J | utility:A/same_residence | unweighted | 0.49465 | 0.49314 | 0.49584 | 0.49454 |
| J | utility:A/same_residence | PWGTP | 0.49077 | 0.49049 | 0.49339 | 0.49155 |
| Q | attack:A/SEX | unweighted | 0.66280 | 0.66617 | 0.66572 | 0.66490 |
| Q | attack:A/SEX | PWGTP | 0.66824 | 0.67085 | 0.67035 | 0.66981 |
| Q | attack:A/RAC1P | unweighted | 1.24045 | 1.24523 | 1.24019 | 1.24196 |
| Q | attack:A/RAC1P | PWGTP | 1.24945 | 1.25264 | 1.24798 | 1.25003 |
| Q | attack:AB/SEX | unweighted | 0.65711 | 0.66218 | 0.66098 | 0.66009 |
| Q | attack:AB/SEX | PWGTP | 0.66199 | 0.66599 | 0.66550 | 0.66449 |
| Q | attack:AB/RAC1P | unweighted | 1.22102 | 1.23776 | 1.22481 | 1.22786 |
| Q | attack:AB/RAC1P | PWGTP | 1.23306 | 1.24652 | 1.23510 | 1.23823 |
| Q | utility:A/same_residence | unweighted | 0.49045 | 0.49080 | 0.49257 | 0.49127 |
| Q | utility:A/same_residence | PWGTP | 0.48698 | 0.48714 | 0.48854 | 0.48756 |
| D17 | attack:A/SEX | unweighted | 0.66165 | 0.66616 | 0.66564 | 0.66449 |
| D17 | attack:A/SEX | PWGTP | 0.66669 | 0.66989 | 0.66953 | 0.66870 |
| D17 | attack:A/RAC1P | unweighted | 1.23902 | 1.24518 | 1.24006 | 1.24142 |
| D17 | attack:A/RAC1P | PWGTP | 1.24884 | 1.25324 | 1.24869 | 1.25026 |
| D17 | attack:AB/SEX | unweighted | 0.65813 | 0.66206 | 0.66059 | 0.66026 |
| D17 | attack:AB/SEX | PWGTP | 0.66347 | 0.66560 | 0.66411 | 0.66439 |
| D17 | attack:AB/RAC1P | unweighted | 1.21978 | 1.23091 | 1.22666 | 1.22578 |
| D17 | attack:AB/RAC1P | PWGTP | 1.23271 | 1.24213 | 1.23762 | 1.23749 |
| D17 | utility:A/same_residence | unweighted | 0.48852 | 0.48922 | 0.49165 | 0.48980 |
| D17 | utility:A/same_residence | PWGTP | 0.48581 | 0.48572 | 0.48770 | 0.48641 |
| D33 | attack:A/SEX | unweighted | 0.66232 | 0.66622 | 0.66504 | 0.66453 |
| D33 | attack:A/SEX | PWGTP | 0.66752 | 0.66988 | 0.66908 | 0.66882 |
| D33 | attack:A/RAC1P | unweighted | 1.23888 | 1.24660 | 1.23892 | 1.24147 |
| D33 | attack:A/RAC1P | PWGTP | 1.24910 | 1.25387 | 1.24836 | 1.25044 |
| D33 | attack:AB/SEX | unweighted | 0.65869 | 0.66298 | 0.66128 | 0.66098 |
| D33 | attack:AB/SEX | PWGTP | 0.66349 | 0.66676 | 0.66536 | 0.66520 |
| D33 | attack:AB/RAC1P | unweighted | 1.22236 | 1.23108 | 1.22627 | 1.22657 |
| D33 | attack:AB/RAC1P | PWGTP | 1.23569 | 1.24144 | 1.23744 | 1.23819 |
| D33 | utility:A/same_residence | unweighted | 0.48863 | 0.48853 | 0.49132 | 0.48949 |
| D33 | utility:A/same_residence | PWGTP | 0.48583 | 0.48529 | 0.48771 | 0.48628 |
| C | attack:A/SEX | unweighted | 0.65693 | 0.66205 | 0.66204 | 0.66034 |
| C | attack:A/SEX | PWGTP | 0.66302 | 0.66703 | 0.66616 | 0.66540 |
| C | attack:A/RAC1P | unweighted | 1.23789 | 1.24158 | 1.23283 | 1.23743 |
| C | attack:A/RAC1P | PWGTP | 1.24865 | 1.25118 | 1.24285 | 1.24756 |
| C | attack:AB/SEX | unweighted | 0.65417 | 0.65743 | 0.65680 | 0.65613 |
| C | attack:AB/SEX | PWGTP | 0.65924 | 0.66248 | 0.66138 | 0.66103 |
| C | attack:AB/RAC1P | unweighted | 1.22049 | 1.22946 | 1.21920 | 1.22305 |
| C | attack:AB/RAC1P | PWGTP | 1.23392 | 1.23811 | 1.23128 | 1.23444 |
| C | utility:A/same_residence | unweighted | 0.48588 | 0.48621 | 0.48855 | 0.48688 |
| C | utility:A/same_residence | PWGTP | 0.48344 | 0.48332 | 0.48452 | 0.48376 |
| E | attack:A/SEX | unweighted | 0.63747 | 0.63734 | 0.63635 | 0.63706 |
| E | attack:A/SEX | PWGTP | 0.64210 | 0.64280 | 0.64217 | 0.64236 |
| E | attack:A/RAC1P | unweighted | 1.17459 | 1.17612 | 1.17165 | 1.17412 |
| E | attack:A/RAC1P | PWGTP | 1.18569 | 1.18689 | 1.17934 | 1.18397 |
| E | attack:AB/SEX | unweighted | 0.63747 | 0.63762 | 0.63635 | 0.63715 |
| E | attack:AB/SEX | PWGTP | 0.64210 | 0.64311 | 0.64217 | 0.64246 |
| E | attack:AB/RAC1P | unweighted | 1.17459 | 1.17520 | 1.17165 | 1.17381 |
| E | attack:AB/RAC1P | PWGTP | 1.18569 | 1.18867 | 1.17934 | 1.18457 |
| E | utility:A/same_residence | unweighted | 0.47837 | 0.48079 | 0.48430 | 0.48115 |
| E | utility:A/same_residence | PWGTP | 0.47735 | 0.47859 | 0.48345 | 0.47980 |
| S | attack:A/SEX | unweighted | 0.63760 | 0.63865 | 0.63733 | 0.63786 |
| S | attack:A/SEX | PWGTP | 0.64261 | 0.64361 | 0.64253 | 0.64292 |
| S | attack:A/RAC1P | unweighted | 1.17186 | 1.17524 | 1.17696 | 1.17469 |
| S | attack:A/RAC1P | PWGTP | 1.18351 | 1.18598 | 1.18708 | 1.18552 |
| S | attack:AB/SEX | unweighted | 0.63765 | 0.63832 | 0.63828 | 0.63808 |
| S | attack:AB/SEX | PWGTP | 0.64289 | 0.64348 | 0.64311 | 0.64316 |
| S | attack:AB/RAC1P | unweighted | 1.17183 | 1.17526 | 1.17696 | 1.17468 |
| S | attack:AB/RAC1P | PWGTP | 1.18270 | 1.18848 | 1.18708 | 1.18608 |
| S | utility:A/same_residence | unweighted | 0.47952 | 0.48103 | 0.48007 | 0.48021 |
| S | utility:A/same_residence | PWGTP | 0.47756 | 0.47935 | 0.47907 | 0.47866 |
| RR75 | attack:A/SEX | unweighted | 0.66489 | 0.66824 | 0.66906 | 0.66740 |
| RR75 | attack:A/SEX | PWGTP | 0.66996 | 0.67200 | 0.67310 | 0.67168 |
| RR75 | attack:A/RAC1P | unweighted | 1.24092 | 1.24819 | 1.24608 | 1.24506 |
| RR75 | attack:A/RAC1P | PWGTP | 1.25109 | 1.25466 | 1.25503 | 1.25359 |
| RR75 | attack:AB/SEX | unweighted | 0.66036 | 0.66311 | 0.66429 | 0.66259 |
| RR75 | attack:AB/SEX | PWGTP | 0.66487 | 0.66661 | 0.66827 | 0.66659 |
| RR75 | attack:AB/RAC1P | unweighted | 1.21997 | 1.23265 | 1.22693 | 1.22651 |
| RR75 | attack:AB/RAC1P | PWGTP | 1.23302 | 1.24300 | 1.23768 | 1.23790 |
| RR75 | utility:A/same_residence | unweighted | 0.50218 | 0.50186 | 0.50474 | 0.50293 |
| RR75 | utility:A/same_residence | PWGTP | 0.49870 | 0.49736 | 0.50012 | 0.49872 |
| W75 | attack:A/SEX | unweighted | 0.66505 | 0.66839 | 0.66717 | 0.66687 |
| W75 | attack:A/SEX | PWGTP | 0.66991 | 0.67208 | 0.67149 | 0.67116 |
| W75 | attack:A/RAC1P | unweighted | 1.25119 | 1.24797 | 1.24184 | 1.24700 |
| W75 | attack:A/RAC1P | PWGTP | 1.25821 | 1.25552 | 1.25073 | 1.25482 |
| W75 | attack:AB/SEX | unweighted | 0.65979 | 0.66326 | 0.66277 | 0.66194 |
| W75 | attack:AB/SEX | PWGTP | 0.66463 | 0.66681 | 0.66675 | 0.66606 |
| W75 | attack:AB/RAC1P | unweighted | 1.22073 | 1.23180 | 1.22721 | 1.22658 |
| W75 | attack:AB/RAC1P | PWGTP | 1.23419 | 1.24203 | 1.23832 | 1.23818 |
| W75 | utility:A/same_residence | unweighted | 0.49701 | 0.49819 | 0.49979 | 0.49833 |
| W75 | utility:A/same_residence | PWGTP | 0.49420 | 0.49397 | 0.49512 | 0.49443 |

## Primary clause estimates by anchor

### Q (T0_L_0.01_a17): **FAIL** — 8/10 clauses pass

| Clause | Weighting | Estimate | SE | One-sided 97.5% UB | Threshold | Status | Per-anchor estimates |
|---|---|---:|---:|---:|---:|---|---|
| utility:A/same_residence | unweighted | -0.00327 | 0.00090 | -0.00151 | -0.003 | unresolved | -0.00419, -0.00234, -0.00327 |
| utility:A/same_residence | PWGTP | -0.00400 | 0.00108 | -0.00187 | -0.003 | unresolved | -0.00378, -0.00335, -0.00485 |
| attack:A/SEX | unweighted | -0.01284 | 0.00100 | -0.01089 | +0.001 | passed | -0.01165, -0.01288, -0.01399 |
| attack:A/SEX | PWGTP | -0.01322 | 0.00124 | -0.01080 | +0.001 | passed | -0.01181, -0.01455, -0.01331 |
| attack:A/RAC1P | unweighted | -0.03840 | 0.00238 | -0.03375 | +0.001 | passed | -0.03963, -0.04078, -0.03479 |
| attack:A/RAC1P | PWGTP | -0.03657 | 0.00282 | -0.03105 | +0.001 | passed | -0.03868, -0.04040, -0.03064 |
| attack:AB/SEX | unweighted | -0.00887 | 0.00083 | -0.00724 | +0.001 | passed | -0.00798, -0.00726, -0.01138 |
| attack:AB/SEX | PWGTP | -0.00938 | 0.00103 | -0.00736 | +0.001 | passed | -0.00778, -0.00922, -0.01113 |
| attack:AB/RAC1P | unweighted | -0.02670 | 0.00197 | -0.02284 | +0.001 | passed | -0.02229, -0.03331, -0.02450 |
| attack:AB/RAC1P | PWGTP | -0.02870 | 0.00240 | -0.02400 | +0.001 | passed | -0.02625, -0.03428, -0.02559 |

### D17 (T0_U_unconstrained_a17): **FAIL** — 8/10 clauses pass

| Clause | Weighting | Estimate | SE | One-sided 97.5% UB | Threshold | Status | Per-anchor estimates |
|---|---|---:|---:|---:|---:|---|---|
| utility:A/same_residence | unweighted | -0.00474 | 0.00093 | -0.00291 | -0.003 | unresolved | -0.00612, -0.00391, -0.00419 |
| utility:A/same_residence | PWGTP | -0.00514 | 0.00112 | -0.00295 | -0.003 | unresolved | -0.00496, -0.00477, -0.00569 |
| attack:A/SEX | unweighted | -0.01243 | 0.00101 | -0.01046 | +0.001 | passed | -0.01050, -0.01287, -0.01391 |
| attack:A/SEX | PWGTP | -0.01211 | 0.00124 | -0.00968 | +0.001 | passed | -0.01026, -0.01360, -0.01248 |
| attack:A/RAC1P | unweighted | -0.03786 | 0.00239 | -0.03319 | +0.001 | passed | -0.03820, -0.04073, -0.03466 |
| attack:A/RAC1P | PWGTP | -0.03681 | 0.00285 | -0.03122 | +0.001 | passed | -0.03806, -0.04100, -0.03135 |
| attack:AB/SEX | unweighted | -0.00904 | 0.00085 | -0.00738 | +0.001 | passed | -0.00900, -0.00715, -0.01098 |
| attack:AB/SEX | PWGTP | -0.00928 | 0.00105 | -0.00723 | +0.001 | passed | -0.00927, -0.00884, -0.00974 |
| attack:AB/RAC1P | unweighted | -0.02462 | 0.00202 | -0.02065 | +0.001 | passed | -0.02105, -0.02646, -0.02634 |
| attack:AB/RAC1P | PWGTP | -0.02797 | 0.00245 | -0.02316 | +0.001 | passed | -0.02591, -0.02989, -0.02811 |

