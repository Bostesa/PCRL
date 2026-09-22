# Attack calibration

predetermined validation diagnostic; failure to demonstrate stress gain is not proof of privacy or attack optimality; all candidate selections remain frozen

Diagnostic: `stress_gain_demonstrated_on_informative_controls`.

Nonlinear upgrade diagnostic: `nonlinear_gain_demonstrated_on_informative_controls`.

added continued360 candidate versus the standard slate that already contains logistic, trees and fresh-initialized MLP120; nonlinear versus logistic gains reported separately

Positive gains below mean smaller validation CE. These are descriptive point comparisons, not a hypothesis test.

| Control | Role | Weighting | Selected recovery over H | MLP120 minus continued360 | Standard minus catchup | Logistic minus best tree |
| --- | --- | --- | --- | --- | --- | --- |
| H | attack:A/SEX | unweighted | unavailable | 0.00135479179 | 0.00135479179 | 0.00594308692 |
| H | attack:A/SEX | PWGTP | unavailable | 0.00132582758 | 0.00132582758 | -0.00238167595 |
| H | attack:A/SEX | balanced | unavailable | 0.00134030968 | 0.00134030968 | 0.00178070548 |
| H | attack:A/RAC1P | unweighted | unavailable | 0.0019087237 | 0.0019087237 | -0.058052389 |
| H | attack:A/RAC1P | PWGTP | unavailable | 0.00100631151 | 0.00100631151 | -0.0656200377 |
| H | attack:A/RAC1P | balanced | unavailable | 0.0014575176 | 0.0014575176 | -0.0618362133 |
| H | attack:AB/SEX | unweighted | unavailable | 0.000537754362 | 0.000537754362 | 0.00865156854 |
| H | attack:AB/SEX | PWGTP | unavailable | 0.000328775531 | 0.000328775531 | 0.00128364486 |
| H | attack:AB/SEX | balanced | unavailable | 0.000433264946 | 0.000433264946 | 0.0049676067 |
| H | attack:AB/RAC1P | unweighted | unavailable | 0 | 0 | -0.0551565153 |
| H | attack:AB/RAC1P | PWGTP | unavailable | 0 | 0 | -0.0631934891 |
| H | attack:AB/RAC1P | balanced | unavailable | 0 | 0 | -0.0591750022 |
| J | attack:A/SEX | unweighted | 0.0155737751 | 0 | 0 | 0.0136484221 |
| J | attack:A/SEX | PWGTP | 0.00822122487 | 0 | 0 | 0.00675990234 |
| J | attack:A/SEX | balanced | 0.0118975 | 0 | 0 | 0.0102041622 |
| J | attack:A/RAC1P | unweighted | 0.00741016756 | 0.000426142781 | 0.000426142781 | -0.0663245681 |
| J | attack:A/RAC1P | PWGTP | 0.004436265 | 0.000240987432 | 0.000240987432 | -0.0745771727 |
| J | attack:A/RAC1P | balanced | 0.00592321628 | 0.000333565107 | 0.000333565107 | -0.0704508704 |
| J | attack:AB/SEX | unweighted | 0.0140394433 | 0 | 0 | 0.0124132868 |
| J | attack:AB/SEX | PWGTP | 0.00729845264 | 0 | 0 | 0.00620152687 |
| J | attack:AB/SEX | balanced | 0.010668948 | 0 | 0 | 0.00930740685 |
| J | attack:AB/RAC1P | unweighted | 0.00811937812 | 0 | 0 | -0.0743502656 |
| J | attack:AB/RAC1P | PWGTP | 0.00522003757 | 0 | 0 | -0.0833449256 |
| J | attack:AB/RAC1P | balanced | 0.00666970784 | 0 | 0 | -0.0788475956 |
| continuous_task | attack:A/SEX | unweighted | 0.00826509937 | 0.000136701145 | 0.000136701145 | 0.0114832646 |
| continuous_task | attack:A/SEX | PWGTP | 0.00526724271 | 1.23629648e-05 | 1.23629648e-05 | 0.00257620922 |
| continuous_task | attack:A/SEX | balanced | 0.00676617104 | 7.45320549e-05 | 7.45320549e-05 | 0.0070297369 |
| continuous_task | attack:A/RAC1P | unweighted | 0.00789653635 | 0.00150030874 | 0.00150030874 | -0.057955664 |
| continuous_task | attack:A/RAC1P | PWGTP | 0.00813075466 | 0.00137251784 | 0.00137251784 | -0.0623497809 |
| continuous_task | attack:A/RAC1P | balanced | 0.0080136455 | 0.00143641329 | 0.00143641329 | -0.0601527224 |
| continuous_task | attack:AB/SEX | unweighted | 0.0113356451 | 0 | 0 | 0.0125394836 |
| continuous_task | attack:AB/SEX | PWGTP | 0.00636522354 | 0 | 0 | 0.00406585022 |
| continuous_task | attack:AB/SEX | balanced | 0.0088504343 | 0 | 0 | 0.00830266694 |
| continuous_task | attack:AB/RAC1P | unweighted | 0.00554887677 | 0 | 0 | -0.0583559185 |
| continuous_task | attack:AB/RAC1P | PWGTP | 0.00261464472 | 0 | 0 | -0.0618846937 |
| continuous_task | attack:AB/RAC1P | balanced | 0.00408176075 | 0 | 0 | -0.0601203061 |
| Trisk_code | attack:A/SEX | unweighted | 0.00946141387 | 0 | 0 | 0.00742991132 |
| Trisk_code | attack:A/SEX | PWGTP | 0.00554813449 | 0 | 0 | 0.00522369825 |
| Trisk_code | attack:A/SEX | balanced | 0.00750477418 | 0 | 0 | 0.00632680479 |
| Trisk_code | attack:A/RAC1P | unweighted | 0.00962629499 | 0 | 0 | -0.0116410519 |
| Trisk_code | attack:A/RAC1P | PWGTP | 0.00980205554 | 0 | 0 | -0.0117100636 |
| Trisk_code | attack:A/RAC1P | balanced | 0.00971417526 | 0 | 0 | -0.0116755577 |
| Trisk_code | attack:AB/SEX | unweighted | 0.00769322128 | 0 | 0 | 0.0101463976 |
| Trisk_code | attack:AB/SEX | PWGTP | 0.00434232017 | 0 | 0 | 0.00818149109 |
| Trisk_code | attack:AB/SEX | balanced | 0.00601777072 | 0 | 0 | 0.00916394436 |
| Trisk_code | attack:AB/RAC1P | unweighted | 0.00214870626 | 0 | 0 | 0.0133421964 |
| Trisk_code | attack:AB/RAC1P | PWGTP | 0.00269226885 | 0 | 0 | 0.015781741 |
| Trisk_code | attack:AB/RAC1P | balanced | 0.00242048756 | 0 | 0 | 0.0145619687 |
