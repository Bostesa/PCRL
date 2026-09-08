# Two-purpose development comparison

Complete three-seed matrix.

All losses/gains are nats. Lower utility loss and lower signed prior-relative attack gain are better. Mean ± sample SD describes three fits on the same cohort; partial summaries explicitly show available n/3. They are not population uncertainty. PWGTP scores the same unweighted-validation-selected predictions. Test households remain DEVELOPMENT EVALUATION.

F receives 16 features per purpose; P receives two A source probabilities and one B probability. Iplus adds sensitive-only local pressure; J adds coalition pressure. Read [policy](PURPOSE_POLICY.md) and [access scopes](ACCESS_SCOPE.md) before interpreting J−I or a pooled score.

[Every selected seed/role](PER_SEED.csv), [all candidate scores](PER_CANDIDATE.csv.gz), [all classes](PER_CLASS.csv.gz), [fixed paired contrasts](PAIRED.csv), [lineage/exposure](CANDIDATE_LINEAGE.json.gz). The four large derived exports use deterministic gzip; [format and hashes](COMPACT_EXPORTS.json).

## development evaluation

| Condition | Income A | Employment A | Coverage B | Residence A | Commute B | All-source margins /3 |
| --- | --- | --- | --- | --- | --- | --- |
| F_I | 0.292829 ± 0.010525 | 0.277106 ± 0.008313 | 0.514969 ± 0.010772 | 0.511633 ± 0.012463 | 0.690494 ± 0.001369 | 2 |
| F_Iplus | 0.293047 ± 0.011932 | 0.278696 ± 0.007613 | 0.512398 ± 0.008367 | 0.513606 ± 0.005169 | 0.690654 ± 0.000297 | 2 |
| F_J | 0.293822 ± 0.012080 | 0.277598 ± 0.006802 | 0.513253 ± 0.008510 | 0.517426 ± 0.009607 | 0.691736 ± 0.002518 | 2 |
| P_I | 0.295594 ± 0.015288 | 0.286624 ± 0.003718 | 0.513815 ± 0.008253 | 0.530677 ± 0.003523 | 0.692977 ± 0.000684 | 2 |
| P_Iplus | 0.296280 ± 0.015106 | 0.287044 ± 0.003901 | 0.514492 ± 0.008345 | 0.530966 ± 0.003744 | 0.692983 ± 0.000946 | 1 |
| P_J | 0.295192 ± 0.014148 | 0.287580 ± 0.003673 | 0.514249 ± 0.008354 | 0.531076 ± 0.003332 | 0.692736 ± 0.000646 | 1 |
| E | 0.356662 ± 0.017555 | 0.414513 ± 0.055286 | 0.529162 ± 0.025627 | 0.508385 ± 0.002038 | 0.687243 ± 0.004055 | 0 |

### Coalition sensitive recovery: standard_independent, 360 epochs

| Condition | AB SEX gain | AB RAC1P gain |
| --- | --- | --- |
| F_I | 0.017370 ± 0.007317 | 0.033230 ± 0.013592 |
| F_Iplus | 0.009686 ± 0.006072 | 0.024297 ± 0.004144 |
| F_J | 0.006035 ± 0.002192 | 0.034356 ± 0.001120 |
| P_I | 0.002154 ± 0.002248 | 0.021380 ± 0.002831 |
| P_Iplus | 0.001259 ± 0.001883 | 0.018948 ± 0.002857 |
| P_J | 0.001107 ± 0.002009 | 0.019954 ± 0.003307 |
| E | 0.008202 ± 0.004191 | 0.052713 ± 0.002590 |

### Coalition sensitive recovery: expanded_independent, 360 epochs

| Condition | AB SEX gain | AB RAC1P gain |
| --- | --- | --- |
| F_I | 0.017370 ± 0.007317 | 0.033230 ± 0.013592 |
| F_Iplus | 0.009686 ± 0.006072 | 0.022860 ± 0.003197 |
| F_J | 0.006035 ± 0.002192 | 0.027590 ± 0.010600 |
| P_I | 0.002154 ± 0.002248 | 0.021380 ± 0.002831 |
| P_Iplus | 0.001259 ± 0.001883 | 0.018948 ± 0.002857 |
| P_J | 0.001107 ± 0.002009 | 0.019954 ± 0.003307 |
| E | 0.008202 ± 0.004191 | 0.052713 ± 0.002590 |

### Coalition sensitive recovery: expanded_catchup, 360 epochs

| Condition | AB SEX gain | AB RAC1P gain |
| --- | --- | --- |
| F_I | 0.018875 ± 0.009222 | 0.044829 ± 0.003882 |
| F_Iplus | 0.011233 ± 0.005457 | 0.041639 ± 0.005431 |
| F_J | 0.005406 ± 0.002719 | 0.038275 ± 0.007616 |
| P_I | 0.004370 ± 0.002592 | 0.025834 ± 0.002216 |
| P_Iplus | 0.003370 ± 0.002534 | 0.025008 ± 0.000824 |
| P_J | 0.003195 ± 0.002535 | 0.025575 ± 0.000930 |
| E | 0.008202 ± 0.004191 | 0.052713 ± 0.002590 |

### Individual forbidden recovery, expanded catch-up scope

Opposing task gains have no predeclared pass threshold. Reserved opposing targets have no catch-up. E has no saved observer; its expanded-catchup selector contains the available independent candidates only.

| Recipient A | public_coverage | commute_over20 | SEX | RAC1P |
| --- | --- | --- | --- | --- |
| F_I | 0.047785 ± 0.001350 | 0.008244 ± 0.005229 | 0.010617 ± 0.007247 | 0.026872 ± 0.007506 |
| F_Iplus | 0.051338 ± 0.003720 | 0.008335 ± 0.006436 | 0.007193 ± 0.006033 | 0.019909 ± 0.009483 |
| F_J | 0.053493 ± 0.007476 | 0.008648 ± 0.005423 | 0.004952 ± 0.001948 | 0.023368 ± 0.004594 |
| P_I | 0.046236 ± 0.001708 | 0.009263 ± 0.002693 | 0.001797 ± 0.000833 | 0.016582 ± 0.001657 |
| P_Iplus | 0.045793 ± 0.001742 | 0.009259 ± 0.002570 | 0.000542 ± 0.000734 | 0.016161 ± 0.001971 |
| P_J | 0.046108 ± 0.002136 | 0.009473 ± 0.002653 | 0.000342 ± 0.000396 | 0.016892 ± 0.002186 |
| E | 0.044869 ± 0.019184 | 0.010662 ± 0.004483 | 0.008736 ± 0.002942 | 0.048108 ± 0.003174 |

| Recipient B | income_binary | civilian_at_work | same_residence | SEX | RAC1P |
| --- | --- | --- | --- | --- | --- |
| F_I | 0.130463 ± 0.014990 | 0.163487 ± 0.009147 | 0.030539 ± 0.008568 | 0.008845 ± 0.006072 | 0.047265 ± 0.004631 |
| F_Iplus | 0.118557 ± 0.012657 | 0.161641 ± 0.001848 | 0.031087 ± 0.003246 | 0.002328 ± 0.002790 | 0.022252 ± 0.003752 |
| F_J | 0.130739 ± 0.016249 | 0.153785 ± 0.014955 | 0.028881 ± 0.002661 | 0.002721 ± 0.005496 | 0.040293 ± 0.005586 |
| P_I | 0.076417 ± 0.010217 | 0.068181 ± 0.012849 | 0.008400 ± 0.001458 | -0.000281 ± 0.000868 | 0.020697 ± 0.001839 |
| P_Iplus | 0.075960 ± 0.007339 | 0.062531 ± 0.008783 | 0.008000 ± 0.001584 | -0.000130 ± 0.000495 | 0.018895 ± 0.001823 |
| P_J | 0.076581 ± 0.007294 | 0.062172 ± 0.006611 | 0.007976 ± 0.001397 | -0.000404 ± 0.000327 | 0.019541 ± 0.002004 |
| E | 0.139492 ± 0.016332 | 0.264337 ± 0.017908 | 0.038140 ± 0.006577 | 0.007849 ± 0.002765 | 0.051448 ± 0.003941 |

### Fixed native source heads

These heads receive representation-fitting source supervision. Independent utility probes above use their separate 2,048-label budget. No better-of selection is made.

| Condition | Income A | Employment A | Coverage B |
| --- | --- | --- | --- |
| F_I | 0.297600 ± 0.013540 | 0.303456 ± 0.006723 | 0.560736 ± 0.017982 |
| F_Iplus | 0.298262 ± 0.014075 | 0.305044 ± 0.006971 | 0.560690 ± 0.017924 |
| F_J | 0.298025 ± 0.014587 | 0.305389 ± 0.007586 | 0.561887 ± 0.016682 |
| P_I | 0.298540 ± 0.014160 | 0.304009 ± 0.005669 | 0.559202 ± 0.017397 |
| P_Iplus | 0.299306 ± 0.013993 | 0.304276 ± 0.005569 | 0.560749 ± 0.017142 |
| P_J | 0.299117 ± 0.014387 | 0.304609 ± 0.006340 | 0.560286 ± 0.017339 |

## development evaluation person weighted

| Condition | Income A | Employment A | Coverage B | Residence A | Commute B | All-source margins /3 |
| --- | --- | --- | --- | --- | --- | --- |
| F_I | 0.303956 ± 0.008553 | 0.266532 ± 0.014430 | 0.525288 ± 0.007727 | 0.489051 ± 0.012110 | 0.690882 ± 0.001321 | 2 |
| F_Iplus | 0.305684 ± 0.010123 | 0.268072 ± 0.012518 | 0.521926 ± 0.009709 | 0.492142 ± 0.002963 | 0.690389 ± 0.000168 | 2 |
| F_J | 0.305111 ± 0.011657 | 0.266940 ± 0.012029 | 0.523252 ± 0.008837 | 0.493421 ± 0.005975 | 0.691324 ± 0.002160 | 2 |
| P_I | 0.307826 ± 0.016175 | 0.276268 ± 0.008942 | 0.524390 ± 0.006987 | 0.504623 ± 0.005992 | 0.692534 ± 0.000684 | 1 |
| P_Iplus | 0.308836 ± 0.015834 | 0.276661 ± 0.009208 | 0.525109 ± 0.007399 | 0.504927 ± 0.005907 | 0.692348 ± 0.001124 | 1 |
| P_J | 0.307259 ± 0.014126 | 0.277091 ± 0.009185 | 0.524897 ± 0.007331 | 0.504841 ± 0.005910 | 0.692145 ± 0.000998 | 2 |
| E | 0.369383 ± 0.019375 | 0.399205 ± 0.057150 | 0.535277 ± 0.022231 | 0.486163 ± 0.005844 | 0.689689 ± 0.002638 | 0 |

### Coalition sensitive recovery: standard_independent, 360 epochs

| Condition | AB SEX gain | AB RAC1P gain |
| --- | --- | --- |
| F_I | 0.013376 ± 0.003394 | 0.033983 ± 0.010207 |
| F_Iplus | 0.006293 ± 0.003248 | 0.023470 ± 0.004547 |
| F_J | 0.004919 ± 0.001892 | 0.031536 ± 0.006976 |
| P_I | 0.000806 ± 0.004532 | 0.022305 ± 0.000525 |
| P_Iplus | 0.000045 ± 0.003777 | 0.019429 ± 0.003012 |
| P_J | -0.000132 ± 0.003763 | 0.020436 ± 0.001222 |
| E | 0.004494 ± 0.003528 | 0.046232 ± 0.004299 |

### Coalition sensitive recovery: expanded_independent, 360 epochs

| Condition | AB SEX gain | AB RAC1P gain |
| --- | --- | --- |
| F_I | 0.013376 ± 0.003394 | 0.033983 ± 0.010207 |
| F_Iplus | 0.006293 ± 0.003248 | 0.023309 ± 0.004713 |
| F_J | 0.004919 ± 0.001892 | 0.026551 ± 0.008851 |
| P_I | 0.000806 ± 0.004532 | 0.022305 ± 0.000525 |
| P_Iplus | 0.000045 ± 0.003777 | 0.019429 ± 0.003012 |
| P_J | -0.000132 ± 0.003763 | 0.020436 ± 0.001222 |
| E | 0.004494 ± 0.003528 | 0.046232 ± 0.004299 |

### Coalition sensitive recovery: expanded_catchup, 360 epochs

| Condition | AB SEX gain | AB RAC1P gain |
| --- | --- | --- |
| F_I | 0.015566 ± 0.005530 | 0.044139 ± 0.004671 |
| F_Iplus | 0.006857 ± 0.003579 | 0.042509 ± 0.000642 |
| F_J | 0.004153 ± 0.000997 | 0.037830 ± 0.011535 |
| P_I | 0.002050 ± 0.005445 | 0.026523 ± 0.005584 |
| P_Iplus | 0.001766 ± 0.005587 | 0.025362 ± 0.003435 |
| P_J | 0.001363 ± 0.005188 | 0.026267 ± 0.003441 |
| E | 0.004494 ± 0.003528 | 0.046232 ± 0.004299 |

### Individual forbidden recovery, expanded catch-up scope

Opposing task gains have no predeclared pass threshold. Reserved opposing targets have no catch-up. E has no saved observer; its expanded-catchup selector contains the available independent candidates only.

| Recipient A | public_coverage | commute_over20 | SEX | RAC1P |
| --- | --- | --- | --- | --- |
| F_I | 0.047465 ± 0.002164 | 0.005922 ± 0.003289 | 0.009474 ± 0.002309 | 0.028700 ± 0.001229 |
| F_Iplus | 0.053163 ± 0.002278 | 0.005769 ± 0.003401 | 0.007963 ± 0.002915 | 0.021405 ± 0.013459 |
| F_J | 0.054526 ± 0.007975 | 0.005854 ± 0.004212 | 0.003987 ± 0.001280 | 0.024287 ± 0.004395 |
| P_I | 0.048508 ± 0.002104 | 0.006063 ± 0.000925 | 0.002209 ± 0.002048 | 0.015649 ± 0.005560 |
| P_Iplus | 0.048211 ± 0.002425 | 0.005881 ± 0.000758 | 0.000898 ± 0.001937 | 0.016034 ± 0.003364 |
| P_J | 0.048468 ± 0.002564 | 0.006086 ± 0.000893 | 0.000766 ± 0.001447 | 0.017117 ± 0.003741 |
| E | 0.041815 ± 0.014813 | 0.009056 ± 0.001485 | 0.005874 ± 0.002864 | 0.042453 ± 0.002347 |

| Recipient B | income_binary | civilian_at_work | same_residence | SEX | RAC1P |
| --- | --- | --- | --- | --- | --- |
| F_I | 0.125878 ± 0.016014 | 0.165765 ± 0.015539 | 0.029113 ± 0.005472 | 0.006016 ± 0.005076 | 0.046069 ± 0.002429 |
| F_Iplus | 0.112648 ± 0.018788 | 0.161545 ± 0.006821 | 0.032119 ± 0.002198 | -0.001669 ± 0.001788 | 0.022495 ± 0.006781 |
| F_J | 0.122828 ± 0.020850 | 0.151835 ± 0.021523 | 0.028094 ± 0.002261 | 0.000361 ± 0.003462 | 0.038489 ± 0.010470 |
| P_I | 0.075335 ± 0.006662 | 0.068019 ± 0.022875 | 0.009974 ± 0.000319 | -0.000280 ± 0.000717 | 0.020456 ± 0.003230 |
| P_Iplus | 0.073962 ± 0.004810 | 0.064447 ± 0.018357 | 0.009515 ± 0.000748 | -0.000290 ± 0.000490 | 0.018426 ± 0.003028 |
| P_J | 0.075231 ± 0.005348 | 0.068251 ± 0.010493 | 0.009419 ± 0.000337 | -0.000167 ± 0.000189 | 0.019330 ± 0.002753 |
| E | 0.129976 ± 0.017649 | 0.267507 ± 0.019920 | 0.032935 ± 0.006210 | 0.004191 ± 0.002359 | 0.047475 ± 0.002250 |

### Fixed native source heads

These heads receive representation-fitting source supervision. Independent utility probes above use their separate 2,048-label budget. No better-of selection is made.

| Condition | Income A | Employment A | Coverage B |
| --- | --- | --- | --- |
| F_I | 0.308930 ± 0.015173 | 0.293513 ± 0.014718 | 0.568006 ± 0.012301 |
| F_Iplus | 0.310107 ± 0.015471 | 0.295520 ± 0.015841 | 0.567789 ± 0.011423 |
| F_J | 0.308903 ± 0.016813 | 0.295465 ± 0.016533 | 0.569115 ± 0.010366 |
| P_I | 0.310044 ± 0.015533 | 0.293876 ± 0.013182 | 0.567723 ± 0.010165 |
| P_Iplus | 0.310824 ± 0.015111 | 0.294579 ± 0.013306 | 0.569034 ± 0.010278 |
| P_J | 0.310766 ± 0.016067 | 0.294470 ± 0.013939 | 0.569294 ± 0.010331 |

## Context and limitations

Original PCA32 remains the parent for source and residential-retention denominators. Its unchanged historical audit is120 epochs, so parent-halving references are not advertised as a matched360 parent audit. Historical PCA16/banks and prior single-interface arms remain [contextual reference tables](CONTEXTUAL_REFERENCES.md), not causal replacements for a new paired system. Historical rich and current source-only banks differ in purpose routing, source-loss weights (this study uses .25/.25/.5), interfaces and inherited exposure; they cannot replace the matched F/P comparisons. All nine race categories remain reported; missing independent attacker-fitting/validation support leaves full race protection unassessable. Saved observers inherit a separate representation-fitting exposure pool, whose support is recorded independently. Independent failures and expanded candidate minima do not certify privacy.

The [feature capability criteria](feature_capability.json) evaluate residence and commute separately against P-J and direct E, with own-recipient and coalition sensitive-gain references. [All original-parent references](criteria.json) retain failed and undefined margins.
