# PCA16 preservation: fixed contrasts and descriptive criteria

Positive left-minus-right task losses lose utility; positive attribute losses reduce measured recovery. Every contrast uses saved unweighted-validation selections. No mapper snapshot, attacker, weighting or experiment was selected by development outcomes. All five tasks remain in the comparison.

[Primary table](TABLE.md), [per-seed paired scores](PAIRED.csv), [paired aggregates](PAIRED_AGGREGATE.csv), [source/residence and policy criteria](criteria.json), [all bank comparisons](bank_comparisons.json).

## Prespecified task and attribute contrasts

### Development evaluation

| Left − right | Target | Audit/head selector | Log-loss difference |
| --- | --- | --- | --- |
| W: historical beta=0 − I: shared initialization | Same residence | primary | 0.003168 ± 0.002616 |
| W: historical beta=0 − I: shared initialization | Commute >20 min | primary | -0.000784 ± 0.001101 |
| W: historical beta=0 − I: shared initialization | Income >$50k | primary | -0.007834 ± 0.002279 |
| W: historical beta=0 − I: shared initialization | Civilian at work | primary | -0.013814 ± 0.002389 |
| W: historical beta=0 − I: shared initialization | Public coverage | primary | -0.012549 ± 0.005422 |
| beta=.1 W − I: shared initialization | Same residence | primary | -0.000114 ± 0.000286 |
| beta=.1 W − I: shared initialization | Commute >20 min | primary | -0.000111 ± 0.000229 |
| beta=.1 W − I: shared initialization | Income >$50k | primary | -0.007621 ± 0.000515 |
| beta=.1 W − I: shared initialization | Civilian at work | primary | -0.010627 ± 0.004372 |
| beta=.1 W − I: shared initialization | Public coverage | primary | -0.011441 ± 0.005818 |
| beta=.1 W − W: historical beta=0 | Same residence | primary | -0.003281 ± 0.002855 |
| beta=.1 W − W: historical beta=0 | Commute >20 min | primary | 0.000673 ± 0.001311 |
| beta=.1 W − W: historical beta=0 | Income >$50k | primary | 0.000214 ± 0.002795 |
| beta=.1 W − W: historical beta=0 | Civilian at work | primary | 0.003187 ± 0.002548 |
| beta=.1 W − W: historical beta=0 | Public coverage | primary | 0.001108 ± 0.003073 |
| beta=.1 C warmup-only − beta=.1 W | Same residence | primary | -0.000882 ± 0.004738 |
| beta=.1 C warmup-only − beta=.1 W | Commute >20 min | primary | 0.000989 ± 0.002159 |
| beta=.1 C warmup-only − beta=.1 W | Income >$50k | primary | 0.003956 ± 0.002572 |
| beta=.1 C warmup-only − beta=.1 W | Civilian at work | primary | -0.000797 ± 0.005175 |
| beta=.1 C warmup-only − beta=.1 W | Public coverage | primary | 0.005639 ± 0.004293 |
| beta=.1 C warmup-only − beta=0 C_init | Same residence | primary | -0.005422 ± 0.004066 |
| beta=.1 C warmup-only − beta=0 C_init | Commute >20 min | primary | 0.000457 ± 0.000641 |
| beta=.1 C warmup-only − beta=0 C_init | Income >$50k | primary | 0.001121 ± 0.001762 |
| beta=.1 C warmup-only − beta=0 C_init | Civilian at work | primary | 0.001565 ± 0.001826 |
| beta=.1 C warmup-only − beta=0 C_init | Public coverage | primary | 0.004789 ± 0.003345 |
| beta=.1 C warmup-only − beta=0 C_init | SEX | primary | 0.003281 ± 0.002671 |
| beta=.1 C warmup-only − beta=0 C_init | SEX | catchup_inclusive | 0.003281 ± 0.002671 |
| beta=.1 C warmup-only − beta=0 C_init | RAC1P | primary | 0.008641 ± 0.005899 |
| beta=.1 C warmup-only − beta=0 C_init | RAC1P | catchup_inclusive | 0.004835 ± 0.011539 |
| beta=.1 D warmup-only − beta=.1 W | Same residence | primary | 0.001377 ± 0.004566 |
| beta=.1 D warmup-only − beta=.1 W | Commute >20 min | primary | 0.000878 ± 0.002935 |
| beta=.1 D warmup-only − beta=.1 W | Income >$50k | primary | 0.004456 ± 0.003678 |
| beta=.1 D warmup-only − beta=.1 W | Civilian at work | primary | 0.001108 ± 0.003871 |
| beta=.1 D warmup-only − beta=.1 W | Public coverage | primary | 0.003553 ± 0.004725 |
| beta=.1 D warmup-only − beta=0 D_init | Same residence | primary | -0.005234 ± 0.001032 |
| beta=.1 D warmup-only − beta=0 D_init | Commute >20 min | primary | 0.000340 ± 0.001262 |
| beta=.1 D warmup-only − beta=0 D_init | Income >$50k | primary | 0.000425 ± 0.002277 |
| beta=.1 D warmup-only − beta=0 D_init | Civilian at work | primary | 0.003556 ± 0.000168 |
| beta=.1 D warmup-only − beta=0 D_init | Public coverage | primary | 0.003289 ± 0.003823 |
| beta=.1 D warmup-only − beta=0 D_init | SEX | primary | 0.000476 ± 0.002794 |
| beta=.1 D warmup-only − beta=0 D_init | SEX | catchup_inclusive | -0.004461 ± 0.006670 |
| beta=.1 D warmup-only − beta=0 D_init | RAC1P | primary | -0.014588 ± 0.000814 |
| beta=.1 D warmup-only − beta=0 D_init | RAC1P | catchup_inclusive | -0.009397 ± 0.014570 |
| beta=.1 C persistent − beta=.1 W | Same residence | primary | -0.001045 ± 0.004772 |
| beta=.1 C persistent − beta=.1 W | Commute >20 min | primary | 0.000338 ± 0.002642 |
| beta=.1 C persistent − beta=.1 W | Income >$50k | primary | 0.004247 ± 0.001699 |
| beta=.1 C persistent − beta=.1 W | Civilian at work | primary | 0.004368 ± 0.001547 |
| beta=.1 C persistent − beta=.1 W | Public coverage | primary | 0.003872 ± 0.000778 |
| beta=.1 C persistent − beta=0 C_init | Same residence | primary | -0.005586 ± 0.001945 |
| beta=.1 C persistent − beta=0 C_init | Commute >20 min | primary | -0.000195 ± 0.000584 |
| beta=.1 C persistent − beta=0 C_init | Income >$50k | primary | 0.001412 ± 0.003157 |
| beta=.1 C persistent − beta=0 C_init | Civilian at work | primary | 0.006730 ± 0.001808 |
| beta=.1 C persistent − beta=0 C_init | Public coverage | primary | 0.003022 ± 0.002065 |
| beta=.1 C persistent − beta=0 C_init | SEX | primary | 0.004851 ± 0.004830 |
| beta=.1 C persistent − beta=0 C_init | SEX | catchup_inclusive | 0.004851 ± 0.004830 |
| beta=.1 C persistent − beta=0 C_init | RAC1P | primary | -0.005799 ± 0.007628 |
| beta=.1 C persistent − beta=0 C_init | RAC1P | catchup_inclusive | -0.007125 ± 0.007405 |
| beta=.1 D persistent − beta=.1 W | Same residence | primary | 0.000002 ± 0.006436 |
| beta=.1 D persistent − beta=.1 W | Commute >20 min | primary | 0.000167 ± 0.002346 |
| beta=.1 D persistent − beta=.1 W | Income >$50k | primary | 0.004120 ± 0.001734 |
| beta=.1 D persistent − beta=.1 W | Civilian at work | primary | 0.004340 ± 0.000485 |
| beta=.1 D persistent − beta=.1 W | Public coverage | primary | 0.003949 ± 0.001438 |
| beta=.1 D persistent − beta=0 D_init | Same residence | primary | -0.006608 ± 0.002819 |
| beta=.1 D persistent − beta=0 D_init | Commute >20 min | primary | -0.000371 ± 0.000800 |
| beta=.1 D persistent − beta=0 D_init | Income >$50k | primary | 0.000089 ± 0.001979 |
| beta=.1 D persistent − beta=0 D_init | Civilian at work | primary | 0.006787 ± 0.003588 |
| beta=.1 D persistent − beta=0 D_init | Public coverage | primary | 0.003685 ± 0.001289 |
| beta=.1 D persistent − beta=0 D_init | SEX | primary | -0.001897 ± 0.005713 |
| beta=.1 D persistent − beta=0 D_init | SEX | catchup_inclusive | -0.005709 ± 0.001838 |
| beta=.1 D persistent − beta=0 D_init | RAC1P | primary | -0.027713 ± 0.004992 |
| beta=.1 D persistent − beta=0 D_init | RAC1P | catchup_inclusive | -0.016998 ± 0.017440 |
| beta=.1 C persistent − beta=.1 C warmup-only | Same residence | primary | -0.000163 ± 0.004743 |
| beta=.1 C persistent − beta=.1 C warmup-only | Commute >20 min | primary | -0.000651 ± 0.001172 |
| beta=.1 C persistent − beta=.1 C warmup-only | Income >$50k | primary | 0.000291 ± 0.001518 |
| beta=.1 C persistent − beta=.1 C warmup-only | Civilian at work | primary | 0.005166 ± 0.003629 |
| beta=.1 C persistent − beta=.1 C warmup-only | Public coverage | primary | -0.001767 ± 0.005032 |
| beta=.1 C persistent − beta=.1 C warmup-only | SEX | primary | 0.001569 ± 0.002962 |
| beta=.1 C persistent − beta=.1 C warmup-only | SEX | catchup_inclusive | 0.001569 ± 0.002962 |
| beta=.1 C persistent − beta=.1 C warmup-only | RAC1P | primary | -0.014440 ± 0.013053 |
| beta=.1 C persistent − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.011961 ± 0.015515 |
| beta=.1 D persistent − beta=.1 D warmup-only | Same residence | primary | -0.001374 ± 0.001994 |
| beta=.1 D persistent − beta=.1 D warmup-only | Commute >20 min | primary | -0.000711 ± 0.002014 |
| beta=.1 D persistent − beta=.1 D warmup-only | Income >$50k | primary | -0.000336 ± 0.003225 |
| beta=.1 D persistent − beta=.1 D warmup-only | Civilian at work | primary | 0.003232 ± 0.003476 |
| beta=.1 D persistent − beta=.1 D warmup-only | Public coverage | primary | 0.000396 ± 0.004970 |
| beta=.1 D persistent − beta=.1 D warmup-only | SEX | primary | -0.002373 ± 0.003403 |
| beta=.1 D persistent − beta=.1 D warmup-only | SEX | catchup_inclusive | -0.001248 ± 0.005345 |
| beta=.1 D persistent − beta=.1 D warmup-only | RAC1P | primary | -0.013126 ± 0.004184 |
| beta=.1 D persistent − beta=.1 D warmup-only | RAC1P | catchup_inclusive | -0.007601 ± 0.006768 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Same residence | primary | 0.002258 ± 0.001514 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Commute >20 min | primary | -0.000111 ± 0.001030 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Income >$50k | primary | 0.000500 ± 0.002223 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Civilian at work | primary | 0.001906 ± 0.001718 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Public coverage | primary | -0.002086 ± 0.001672 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | SEX | primary | 0.004784 ± 0.001184 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | SEX | catchup_inclusive | 0.003659 ± 0.001596 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | RAC1P | primary | -0.000687 ± 0.008425 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.003905 ± 0.008589 |
| beta=.1 D persistent − beta=.1 C persistent | Same residence | primary | 0.001047 ± 0.003634 |
| beta=.1 D persistent − beta=.1 C persistent | Commute >20 min | primary | -0.000171 ± 0.000306 |
| beta=.1 D persistent − beta=.1 C persistent | Income >$50k | primary | -0.000126 ± 0.000297 |
| beta=.1 D persistent − beta=.1 C persistent | Civilian at work | primary | -0.000029 ± 0.001095 |
| beta=.1 D persistent − beta=.1 C persistent | Public coverage | primary | 0.000077 ± 0.001089 |
| beta=.1 D persistent − beta=.1 C persistent | SEX | primary | 0.000841 ± 0.001919 |
| beta=.1 D persistent − beta=.1 C persistent | SEX | catchup_inclusive | 0.000841 ± 0.001919 |
| beta=.1 D persistent − beta=.1 C persistent | RAC1P | primary | 0.000627 ± 0.000754 |
| beta=.1 D persistent − beta=.1 C persistent | RAC1P | catchup_inclusive | 0.000454 ± 0.001033 |
| beta=1 W − I: shared initialization | Same residence | primary | -0.001554 ± 0.000624 |
| beta=1 W − I: shared initialization | Commute >20 min | primary | -0.000786 ± 0.000315 |
| beta=1 W − I: shared initialization | Income >$50k | primary | -0.005007 ± 0.001682 |
| beta=1 W − I: shared initialization | Civilian at work | primary | -0.012391 ± 0.005228 |
| beta=1 W − I: shared initialization | Public coverage | primary | -0.005996 ± 0.004988 |
| beta=1 W − W: historical beta=0 | Same residence | primary | -0.004722 ± 0.002711 |
| beta=1 W − W: historical beta=0 | Commute >20 min | primary | -0.000002 ± 0.001376 |
| beta=1 W − W: historical beta=0 | Income >$50k | primary | 0.002828 ± 0.002825 |
| beta=1 W − W: historical beta=0 | Civilian at work | primary | 0.001423 ± 0.003159 |
| beta=1 W − W: historical beta=0 | Public coverage | primary | 0.006553 ± 0.001634 |
| beta=1 C warmup-only − beta=1 W | Same residence | primary | 0.000775 ± 0.001749 |
| beta=1 C warmup-only − beta=1 W | Commute >20 min | primary | -0.001209 ± 0.000306 |
| beta=1 C warmup-only − beta=1 W | Income >$50k | primary | 0.005655 ± 0.003205 |
| beta=1 C warmup-only − beta=1 W | Civilian at work | primary | 0.002788 ± 0.004472 |
| beta=1 C warmup-only − beta=1 W | Public coverage | primary | -0.001292 ± 0.003266 |
| beta=1 C warmup-only − beta=0 C_init | Same residence | primary | -0.005207 ± 0.004828 |
| beta=1 C warmup-only − beta=0 C_init | Commute >20 min | primary | -0.002416 ± 0.002306 |
| beta=1 C warmup-only − beta=0 C_init | Income >$50k | primary | 0.005434 ± 0.005571 |
| beta=1 C warmup-only − beta=0 C_init | Civilian at work | primary | 0.003386 ± 0.000594 |
| beta=1 C warmup-only − beta=0 C_init | Public coverage | primary | 0.003303 ± 0.003540 |
| beta=1 C warmup-only − beta=0 C_init | SEX | primary | 0.001984 ± 0.004996 |
| beta=1 C warmup-only − beta=0 C_init | SEX | catchup_inclusive | 0.001984 ± 0.004996 |
| beta=1 C warmup-only − beta=0 C_init | RAC1P | primary | 0.001590 ± 0.006668 |
| beta=1 C warmup-only − beta=0 C_init | RAC1P | catchup_inclusive | -0.002216 ± 0.009931 |
| beta=1 D warmup-only − beta=1 W | Same residence | primary | 0.003728 ± 0.003336 |
| beta=1 D warmup-only − beta=1 W | Commute >20 min | primary | -0.000806 ± 0.000814 |
| beta=1 D warmup-only − beta=1 W | Income >$50k | primary | 0.005376 ± 0.001500 |
| beta=1 D warmup-only − beta=1 W | Civilian at work | primary | 0.004384 ± 0.004582 |
| beta=1 D warmup-only − beta=1 W | Public coverage | primary | -0.002223 ± 0.001333 |
| beta=1 D warmup-only − beta=0 D_init | Same residence | primary | -0.004323 ± 0.005987 |
| beta=1 D warmup-only − beta=0 D_init | Commute >20 min | primary | -0.002019 ± 0.003066 |
| beta=1 D warmup-only − beta=0 D_init | Income >$50k | primary | 0.003959 ± 0.004505 |
| beta=1 D warmup-only − beta=0 D_init | Civilian at work | primary | 0.005068 ± 0.000463 |
| beta=1 D warmup-only − beta=0 D_init | Public coverage | primary | 0.002957 ± 0.000818 |
| beta=1 D warmup-only − beta=0 D_init | SEX | primary | 0.004044 ± 0.002898 |
| beta=1 D warmup-only − beta=0 D_init | SEX | catchup_inclusive | 0.000233 ± 0.005145 |
| beta=1 D warmup-only − beta=0 D_init | RAC1P | primary | -0.002619 ± 0.005365 |
| beta=1 D warmup-only − beta=0 D_init | RAC1P | catchup_inclusive | 0.000016 ± 0.012945 |
| beta=1 C persistent − beta=1 W | Same residence | primary | -0.001221 ± 0.000289 |
| beta=1 C persistent − beta=1 W | Commute >20 min | primary | 0.001164 ± 0.003225 |
| beta=1 C persistent − beta=1 W | Income >$50k | primary | 0.001284 ± 0.001293 |
| beta=1 C persistent − beta=1 W | Civilian at work | primary | 0.005938 ± 0.000931 |
| beta=1 C persistent − beta=1 W | Public coverage | primary | -0.001316 ± 0.001775 |
| beta=1 C persistent − beta=0 C_init | Same residence | primary | -0.007202 ± 0.003583 |
| beta=1 C persistent − beta=0 C_init | Commute >20 min | primary | -0.000043 ± 0.001023 |
| beta=1 C persistent − beta=0 C_init | Income >$50k | primary | 0.001064 ± 0.005214 |
| beta=1 C persistent − beta=0 C_init | Civilian at work | primary | 0.006536 ± 0.004258 |
| beta=1 C persistent − beta=0 C_init | Public coverage | primary | 0.003279 ± 0.002009 |
| beta=1 C persistent − beta=0 C_init | SEX | primary | 0.001847 ± 0.005187 |
| beta=1 C persistent − beta=0 C_init | SEX | catchup_inclusive | 0.001847 ± 0.005187 |
| beta=1 C persistent − beta=0 C_init | RAC1P | primary | -0.008574 ± 0.006428 |
| beta=1 C persistent − beta=0 C_init | RAC1P | catchup_inclusive | -0.012379 ± 0.010148 |
| beta=1 D persistent − beta=1 W | Same residence | primary | -0.000925 ± 0.000398 |
| beta=1 D persistent − beta=1 W | Commute >20 min | primary | 0.001192 ± 0.002867 |
| beta=1 D persistent − beta=1 W | Income >$50k | primary | 0.001618 ± 0.001298 |
| beta=1 D persistent − beta=1 W | Civilian at work | primary | 0.005299 ± 0.000794 |
| beta=1 D persistent − beta=1 W | Public coverage | primary | -0.000182 ± 0.000668 |
| beta=1 D persistent − beta=0 D_init | Same residence | primary | -0.008976 ± 0.004417 |
| beta=1 D persistent − beta=0 D_init | Commute >20 min | primary | -0.000021 ± 0.000188 |
| beta=1 D persistent − beta=0 D_init | Income >$50k | primary | 0.000200 ± 0.004429 |
| beta=1 D persistent − beta=0 D_init | Civilian at work | primary | 0.005983 ± 0.005204 |
| beta=1 D persistent − beta=0 D_init | Public coverage | primary | 0.004999 ± 0.000183 |
| beta=1 D persistent − beta=0 D_init | SEX | primary | -0.005220 ± 0.004015 |
| beta=1 D persistent − beta=0 D_init | SEX | catchup_inclusive | -0.009031 ± 0.002908 |
| beta=1 D persistent − beta=0 D_init | RAC1P | primary | -0.030567 ± 0.003218 |
| beta=1 D persistent − beta=0 D_init | RAC1P | catchup_inclusive | -0.022158 ± 0.011010 |
| beta=1 C persistent − beta=1 C warmup-only | Same residence | primary | -0.001996 ± 0.001484 |
| beta=1 C persistent − beta=1 C warmup-only | Commute >20 min | primary | 0.002373 ± 0.003287 |
| beta=1 C persistent − beta=1 C warmup-only | Income >$50k | primary | -0.004370 ± 0.002276 |
| beta=1 C persistent − beta=1 C warmup-only | Civilian at work | primary | 0.003150 ± 0.004373 |
| beta=1 C persistent − beta=1 C warmup-only | Public coverage | primary | -0.000024 ± 0.001545 |
| beta=1 C persistent − beta=1 C warmup-only | SEX | primary | -0.000137 ± 0.001719 |
| beta=1 C persistent − beta=1 C warmup-only | SEX | catchup_inclusive | -0.000137 ± 0.001719 |
| beta=1 C persistent − beta=1 C warmup-only | RAC1P | primary | -0.010163 ± 0.000663 |
| beta=1 C persistent − beta=1 C warmup-only | RAC1P | catchup_inclusive | -0.010163 ± 0.000663 |
| beta=1 D persistent − beta=1 D warmup-only | Same residence | primary | -0.004653 ± 0.003000 |
| beta=1 D persistent − beta=1 D warmup-only | Commute >20 min | primary | 0.001998 ± 0.003252 |
| beta=1 D persistent − beta=1 D warmup-only | Income >$50k | primary | -0.003759 ± 0.000271 |
| beta=1 D persistent − beta=1 D warmup-only | Civilian at work | primary | 0.000915 ± 0.004796 |
| beta=1 D persistent − beta=1 D warmup-only | Public coverage | primary | 0.002041 ± 0.000993 |
| beta=1 D persistent − beta=1 D warmup-only | SEX | primary | -0.009264 ± 0.003735 |
| beta=1 D persistent − beta=1 D warmup-only | SEX | catchup_inclusive | -0.009264 ± 0.003735 |
| beta=1 D persistent − beta=1 D warmup-only | RAC1P | primary | -0.027948 ± 0.007123 |
| beta=1 D persistent − beta=1 D warmup-only | RAC1P | catchup_inclusive | -0.022174 ± 0.010141 |
| beta=1 D warmup-only − beta=1 C warmup-only | Same residence | primary | 0.002954 ± 0.001622 |
| beta=1 D warmup-only − beta=1 C warmup-only | Commute >20 min | primary | 0.000403 ± 0.000519 |
| beta=1 D warmup-only − beta=1 C warmup-only | Income >$50k | primary | -0.000278 ± 0.002246 |
| beta=1 D warmup-only − beta=1 C warmup-only | Civilian at work | primary | 0.001596 ± 0.000572 |
| beta=1 D warmup-only − beta=1 C warmup-only | Public coverage | primary | -0.000932 ± 0.003939 |
| beta=1 D warmup-only − beta=1 C warmup-only | SEX | primary | 0.009650 ± 0.004143 |
| beta=1 D warmup-only − beta=1 C warmup-only | SEX | catchup_inclusive | 0.009650 ± 0.004143 |
| beta=1 D warmup-only − beta=1 C warmup-only | RAC1P | primary | 0.018333 ± 0.007053 |
| beta=1 D warmup-only − beta=1 C warmup-only | RAC1P | catchup_inclusive | 0.012559 ± 0.010344 |
| beta=1 D persistent − beta=1 C persistent | Same residence | primary | 0.000296 ± 0.000110 |
| beta=1 D persistent − beta=1 C persistent | Commute >20 min | primary | 0.000027 ± 0.000358 |
| beta=1 D persistent − beta=1 C persistent | Income >$50k | primary | 0.000334 ± 0.000249 |
| beta=1 D persistent − beta=1 C persistent | Civilian at work | primary | -0.000639 ± 0.000467 |
| beta=1 D persistent − beta=1 C persistent | Public coverage | primary | 0.001134 ± 0.001495 |
| beta=1 D persistent − beta=1 C persistent | SEX | primary | 0.000523 ± 0.000482 |
| beta=1 D persistent − beta=1 C persistent | SEX | catchup_inclusive | 0.000523 ± 0.000482 |
| beta=1 D persistent − beta=1 C persistent | RAC1P | primary | 0.000549 ± 0.000087 |
| beta=1 D persistent − beta=1 C persistent | RAC1P | catchup_inclusive | 0.000549 ± 0.000087 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Same residence | primary | 0.000216 ± 0.005498 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Commute >20 min | primary | -0.002873 ± 0.002358 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Income >$50k | primary | 0.004313 ± 0.003908 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Civilian at work | primary | 0.001821 ± 0.001733 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Public coverage | primary | -0.001486 ± 0.004704 |
| beta=1 C warmup-only − beta=.1 C warmup-only | SEX | primary | -0.001297 ± 0.004817 |
| beta=1 C warmup-only − beta=.1 C warmup-only | SEX | catchup_inclusive | -0.001297 ± 0.004817 |
| beta=1 C warmup-only − beta=.1 C warmup-only | RAC1P | primary | -0.007051 ± 0.011305 |
| beta=1 C warmup-only − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.007051 ± 0.011305 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Same residence | primary | 0.000911 ± 0.006559 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Commute >20 min | primary | -0.002359 ± 0.003640 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Income >$50k | primary | 0.003534 ± 0.005144 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Civilian at work | primary | 0.001512 ± 0.000300 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Public coverage | primary | -0.000331 ± 0.004058 |
| beta=1 D warmup-only − beta=.1 D warmup-only | SEX | primary | 0.003569 ± 0.000402 |
| beta=1 D warmup-only − beta=.1 D warmup-only | SEX | catchup_inclusive | 0.004693 ± 0.001785 |
| beta=1 D warmup-only − beta=.1 D warmup-only | RAC1P | primary | 0.011969 ± 0.005217 |
| beta=1 D warmup-only − beta=.1 D warmup-only | RAC1P | catchup_inclusive | 0.009413 ± 0.005325 |
| beta=1 C persistent − beta=.1 C persistent | Same residence | primary | -0.001616 ± 0.005524 |
| beta=1 C persistent − beta=.1 C persistent | Commute >20 min | primary | 0.000152 ± 0.001017 |
| beta=1 C persistent − beta=.1 C persistent | Income >$50k | primary | -0.000348 ± 0.002109 |
| beta=1 C persistent − beta=.1 C persistent | Civilian at work | primary | -0.000194 ± 0.002463 |
| beta=1 C persistent − beta=.1 C persistent | Public coverage | primary | 0.000257 ± 0.001562 |
| beta=1 C persistent − beta=.1 C persistent | SEX | primary | -0.003004 ± 0.001643 |
| beta=1 C persistent − beta=.1 C persistent | SEX | catchup_inclusive | -0.003004 ± 0.001643 |
| beta=1 C persistent − beta=.1 C persistent | RAC1P | primary | -0.002775 ± 0.003443 |
| beta=1 C persistent − beta=.1 C persistent | RAC1P | catchup_inclusive | -0.005254 ± 0.007634 |
| beta=1 D persistent − beta=.1 D persistent | Same residence | primary | -0.002368 ± 0.007236 |
| beta=1 D persistent − beta=.1 D persistent | Commute >20 min | primary | 0.000350 ± 0.000955 |
| beta=1 D persistent − beta=.1 D persistent | Income >$50k | primary | 0.000111 ± 0.002453 |
| beta=1 D persistent − beta=.1 D persistent | Civilian at work | primary | -0.000805 ± 0.001760 |
| beta=1 D persistent − beta=.1 D persistent | Public coverage | primary | 0.001314 ± 0.001375 |
| beta=1 D persistent − beta=.1 D persistent | SEX | primary | -0.003323 ± 0.002996 |
| beta=1 D persistent − beta=.1 D persistent | SEX | catchup_inclusive | -0.003323 ± 0.002996 |
| beta=1 D persistent − beta=.1 D persistent | RAC1P | primary | -0.002853 ± 0.002647 |
| beta=1 D persistent − beta=.1 D persistent | RAC1P | catchup_inclusive | -0.005160 ± 0.006583 |

### Development evaluation person weighted

| Left − right | Target | Audit/head selector | Log-loss difference |
| --- | --- | --- | --- |
| W: historical beta=0 − I: shared initialization | Same residence | primary | -0.002566 ± 0.004596 |
| W: historical beta=0 − I: shared initialization | Commute >20 min | primary | -0.000061 ± 0.000804 |
| W: historical beta=0 − I: shared initialization | Income >$50k | primary | -0.006601 ± 0.001191 |
| W: historical beta=0 − I: shared initialization | Civilian at work | primary | -0.010886 ± 0.003542 |
| W: historical beta=0 − I: shared initialization | Public coverage | primary | -0.012394 ± 0.004097 |
| beta=.1 W − I: shared initialization | Same residence | primary | -0.002985 ± 0.001453 |
| beta=.1 W − I: shared initialization | Commute >20 min | primary | 0.001564 ± 0.001280 |
| beta=.1 W − I: shared initialization | Income >$50k | primary | -0.006763 ± 0.000873 |
| beta=.1 W − I: shared initialization | Civilian at work | primary | -0.007496 ± 0.006902 |
| beta=.1 W − I: shared initialization | Public coverage | primary | -0.009980 ± 0.004105 |
| beta=.1 W − W: historical beta=0 | Same residence | primary | -0.000420 ± 0.004563 |
| beta=.1 W − W: historical beta=0 | Commute >20 min | primary | 0.001625 ± 0.000768 |
| beta=.1 W − W: historical beta=0 | Income >$50k | primary | -0.000162 ± 0.000327 |
| beta=.1 W − W: historical beta=0 | Civilian at work | primary | 0.003390 ± 0.003392 |
| beta=.1 W − W: historical beta=0 | Public coverage | primary | 0.002414 ± 0.002539 |
| beta=.1 C warmup-only − beta=.1 W | Same residence | primary | -0.000850 ± 0.007948 |
| beta=.1 C warmup-only − beta=.1 W | Commute >20 min | primary | 0.001062 ± 0.000551 |
| beta=.1 C warmup-only − beta=.1 W | Income >$50k | primary | 0.004706 ± 0.003199 |
| beta=.1 C warmup-only − beta=.1 W | Civilian at work | primary | -0.001690 ± 0.005497 |
| beta=.1 C warmup-only − beta=.1 W | Public coverage | primary | 0.006774 ± 0.000514 |
| beta=.1 C warmup-only − beta=0 C_init | Same residence | primary | -0.001867 ± 0.007769 |
| beta=.1 C warmup-only − beta=0 C_init | Commute >20 min | primary | 0.001233 ± 0.002088 |
| beta=.1 C warmup-only − beta=0 C_init | Income >$50k | primary | 0.001493 ± 0.002637 |
| beta=.1 C warmup-only − beta=0 C_init | Civilian at work | primary | 0.000515 ± 0.001657 |
| beta=.1 C warmup-only − beta=0 C_init | Public coverage | primary | 0.005342 ± 0.003417 |
| beta=.1 C warmup-only − beta=0 C_init | SEX | primary | 0.003984 ± 0.002360 |
| beta=.1 C warmup-only − beta=0 C_init | SEX | catchup_inclusive | 0.003984 ± 0.002360 |
| beta=.1 C warmup-only − beta=0 C_init | RAC1P | primary | 0.011466 ± 0.008389 |
| beta=.1 C warmup-only − beta=0 C_init | RAC1P | catchup_inclusive | 0.005329 ± 0.018114 |
| beta=.1 D warmup-only − beta=.1 W | Same residence | primary | 0.001769 ± 0.005263 |
| beta=.1 D warmup-only − beta=.1 W | Commute >20 min | primary | 0.001039 ± 0.000368 |
| beta=.1 D warmup-only − beta=.1 W | Income >$50k | primary | 0.005541 ± 0.001278 |
| beta=.1 D warmup-only − beta=.1 W | Civilian at work | primary | 0.000416 ± 0.004710 |
| beta=.1 D warmup-only − beta=.1 W | Public coverage | primary | 0.004747 ± 0.003125 |
| beta=.1 D warmup-only − beta=0 D_init | Same residence | primary | -0.000887 ± 0.002442 |
| beta=.1 D warmup-only − beta=0 D_init | Commute >20 min | primary | 0.002003 ± 0.002366 |
| beta=.1 D warmup-only − beta=0 D_init | Income >$50k | primary | 0.001565 ± 0.001955 |
| beta=.1 D warmup-only − beta=0 D_init | Civilian at work | primary | 0.002810 ± 0.000952 |
| beta=.1 D warmup-only − beta=0 D_init | Public coverage | primary | 0.003756 ± 0.000590 |
| beta=.1 D warmup-only − beta=0 D_init | SEX | primary | 0.000604 ± 0.003632 |
| beta=.1 D warmup-only − beta=0 D_init | SEX | catchup_inclusive | -0.005279 ± 0.007097 |
| beta=.1 D warmup-only − beta=0 D_init | RAC1P | primary | -0.013399 ± 0.003510 |
| beta=.1 D warmup-only − beta=0 D_init | RAC1P | catchup_inclusive | -0.009853 ± 0.011245 |
| beta=.1 C persistent − beta=.1 W | Same residence | primary | 0.000565 ± 0.004901 |
| beta=.1 C persistent − beta=.1 W | Commute >20 min | primary | -0.000182 ± 0.001807 |
| beta=.1 C persistent − beta=.1 W | Income >$50k | primary | 0.004483 ± 0.002786 |
| beta=.1 C persistent − beta=.1 W | Civilian at work | primary | 0.004647 ± 0.002221 |
| beta=.1 C persistent − beta=.1 W | Public coverage | primary | 0.005098 ± 0.002817 |
| beta=.1 C persistent − beta=0 C_init | Same residence | primary | -0.000452 ± 0.004430 |
| beta=.1 C persistent − beta=0 C_init | Commute >20 min | primary | -0.000012 ± 0.001679 |
| beta=.1 C persistent − beta=0 C_init | Income >$50k | primary | 0.001271 ± 0.001723 |
| beta=.1 C persistent − beta=0 C_init | Civilian at work | primary | 0.006851 ± 0.002564 |
| beta=.1 C persistent − beta=0 C_init | Public coverage | primary | 0.003667 ± 0.004160 |
| beta=.1 C persistent − beta=0 C_init | SEX | primary | 0.005716 ± 0.004293 |
| beta=.1 C persistent − beta=0 C_init | SEX | catchup_inclusive | 0.005716 ± 0.004293 |
| beta=.1 C persistent − beta=0 C_init | RAC1P | primary | -0.004316 ± 0.007696 |
| beta=.1 C persistent − beta=0 C_init | RAC1P | catchup_inclusive | -0.004334 ± 0.007697 |
| beta=.1 D persistent − beta=.1 W | Same residence | primary | 0.001867 ± 0.006393 |
| beta=.1 D persistent − beta=.1 W | Commute >20 min | primary | -0.000293 ± 0.001460 |
| beta=.1 D persistent − beta=.1 W | Income >$50k | primary | 0.003996 ± 0.002674 |
| beta=.1 D persistent − beta=.1 W | Civilian at work | primary | 0.004667 ± 0.001428 |
| beta=.1 D persistent − beta=.1 W | Public coverage | primary | 0.004973 ± 0.002813 |
| beta=.1 D persistent − beta=0 D_init | Same residence | primary | -0.000789 ± 0.003854 |
| beta=.1 D persistent − beta=0 D_init | Commute >20 min | primary | 0.000671 ± 0.001082 |
| beta=.1 D persistent − beta=0 D_init | Income >$50k | primary | 0.000020 ± 0.001448 |
| beta=.1 D persistent − beta=0 D_init | Civilian at work | primary | 0.007062 ± 0.003680 |
| beta=.1 D persistent − beta=0 D_init | Public coverage | primary | 0.003982 ± 0.003938 |
| beta=.1 D persistent − beta=0 D_init | SEX | primary | -0.002499 ± 0.006218 |
| beta=.1 D persistent − beta=0 D_init | SEX | catchup_inclusive | -0.008480 ± 0.005359 |
| beta=.1 D persistent − beta=0 D_init | RAC1P | primary | -0.027331 ± 0.001701 |
| beta=.1 D persistent − beta=0 D_init | RAC1P | catchup_inclusive | -0.015083 ± 0.017214 |
| beta=.1 C persistent − beta=.1 C warmup-only | Same residence | primary | 0.001414 ± 0.007059 |
| beta=.1 C persistent − beta=.1 C warmup-only | Commute >20 min | primary | -0.001244 ± 0.001311 |
| beta=.1 C persistent − beta=.1 C warmup-only | Income >$50k | primary | -0.000222 ± 0.001009 |
| beta=.1 C persistent − beta=.1 C warmup-only | Civilian at work | primary | 0.006336 ± 0.003819 |
| beta=.1 C persistent − beta=.1 C warmup-only | Public coverage | primary | -0.001675 ± 0.002980 |
| beta=.1 C persistent − beta=.1 C warmup-only | SEX | primary | 0.001732 ± 0.002374 |
| beta=.1 C persistent − beta=.1 C warmup-only | SEX | catchup_inclusive | 0.001732 ± 0.002374 |
| beta=.1 C persistent − beta=.1 C warmup-only | RAC1P | primary | -0.015781 ± 0.014180 |
| beta=.1 C persistent − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.009663 ± 0.021255 |
| beta=.1 D persistent − beta=.1 D warmup-only | Same residence | primary | 0.000098 ± 0.002055 |
| beta=.1 D persistent − beta=.1 D warmup-only | Commute >20 min | primary | -0.001332 ± 0.001456 |
| beta=.1 D persistent − beta=.1 D warmup-only | Income >$50k | primary | -0.001545 ± 0.002979 |
| beta=.1 D persistent − beta=.1 D warmup-only | Civilian at work | primary | 0.004252 ± 0.004231 |
| beta=.1 D persistent − beta=.1 D warmup-only | Public coverage | primary | 0.000226 ± 0.004502 |
| beta=.1 D persistent − beta=.1 D warmup-only | SEX | primary | -0.003103 ± 0.004794 |
| beta=.1 D persistent − beta=.1 D warmup-only | SEX | catchup_inclusive | -0.003201 ± 0.004711 |
| beta=.1 D persistent − beta=.1 D warmup-only | RAC1P | primary | -0.013932 ± 0.003611 |
| beta=.1 D persistent − beta=.1 D warmup-only | RAC1P | catchup_inclusive | -0.005230 ± 0.006186 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Same residence | primary | 0.002618 ± 0.002708 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Commute >20 min | primary | -0.000023 ± 0.000490 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Income >$50k | primary | 0.000835 ± 0.003902 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Civilian at work | primary | 0.002105 ± 0.001268 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Public coverage | primary | -0.002026 ± 0.003532 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | SEX | primary | 0.005391 ± 0.003230 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | SEX | catchup_inclusive | 0.005490 ± 0.003313 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | RAC1P | primary | -0.001809 ± 0.011596 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.005008 ± 0.014051 |
| beta=.1 D persistent − beta=.1 C persistent | Same residence | primary | 0.001302 ± 0.004208 |
| beta=.1 D persistent − beta=.1 C persistent | Commute >20 min | primary | -0.000111 ± 0.000347 |
| beta=.1 D persistent − beta=.1 C persistent | Income >$50k | primary | -0.000488 ± 0.000297 |
| beta=.1 D persistent − beta=.1 C persistent | Civilian at work | primary | 0.000020 ± 0.000971 |
| beta=.1 D persistent − beta=.1 C persistent | Public coverage | primary | -0.000125 ± 0.000907 |
| beta=.1 D persistent − beta=.1 C persistent | SEX | primary | 0.000556 ± 0.001747 |
| beta=.1 D persistent − beta=.1 C persistent | SEX | catchup_inclusive | 0.000556 ± 0.001747 |
| beta=.1 D persistent − beta=.1 C persistent | RAC1P | primary | 0.000041 ± 0.000609 |
| beta=.1 D persistent − beta=.1 C persistent | RAC1P | catchup_inclusive | -0.000576 ± 0.001225 |
| beta=1 W − I: shared initialization | Same residence | primary | -0.001339 ± 0.000891 |
| beta=1 W − I: shared initialization | Commute >20 min | primary | 0.000263 ± 0.000419 |
| beta=1 W − I: shared initialization | Income >$50k | primary | -0.004296 ± 0.001980 |
| beta=1 W − I: shared initialization | Civilian at work | primary | -0.008910 ± 0.007004 |
| beta=1 W − I: shared initialization | Public coverage | primary | -0.005602 ± 0.004451 |
| beta=1 W − W: historical beta=0 | Same residence | primary | 0.001227 ± 0.004866 |
| beta=1 W − W: historical beta=0 | Commute >20 min | primary | 0.000324 ± 0.000745 |
| beta=1 W − W: historical beta=0 | Income >$50k | primary | 0.002305 ± 0.002476 |
| beta=1 W − W: historical beta=0 | Civilian at work | primary | 0.001975 ± 0.003469 |
| beta=1 W − W: historical beta=0 | Public coverage | primary | 0.006792 ± 0.002130 |
| beta=1 C warmup-only − beta=1 W | Same residence | primary | -0.003956 ± 0.003032 |
| beta=1 C warmup-only − beta=1 W | Commute >20 min | primary | -0.000568 ± 0.002082 |
| beta=1 C warmup-only − beta=1 W | Income >$50k | primary | 0.006928 ± 0.005175 |
| beta=1 C warmup-only − beta=1 W | Civilian at work | primary | 0.002328 ± 0.005179 |
| beta=1 C warmup-only − beta=1 W | Public coverage | primary | -0.000066 ± 0.001445 |
| beta=1 C warmup-only − beta=0 C_init | Same residence | primary | -0.003327 ± 0.002587 |
| beta=1 C warmup-only − beta=0 C_init | Commute >20 min | primary | -0.001698 ± 0.003500 |
| beta=1 C warmup-only − beta=0 C_init | Income >$50k | primary | 0.006183 ± 0.004736 |
| beta=1 C warmup-only − beta=0 C_init | Civilian at work | primary | 0.003118 ± 0.001927 |
| beta=1 C warmup-only − beta=0 C_init | Public coverage | primary | 0.002880 ± 0.004313 |
| beta=1 C warmup-only − beta=0 C_init | SEX | primary | 0.002635 ± 0.005211 |
| beta=1 C warmup-only − beta=0 C_init | SEX | catchup_inclusive | 0.002635 ± 0.005211 |
| beta=1 C warmup-only − beta=0 C_init | RAC1P | primary | 0.003253 ± 0.006518 |
| beta=1 C warmup-only − beta=0 C_init | RAC1P | catchup_inclusive | -0.002884 ± 0.012047 |
| beta=1 D warmup-only − beta=1 W | Same residence | primary | -0.000263 ± 0.003684 |
| beta=1 D warmup-only − beta=1 W | Commute >20 min | primary | -0.000052 ± 0.002304 |
| beta=1 D warmup-only − beta=1 W | Income >$50k | primary | 0.007036 ± 0.005026 |
| beta=1 D warmup-only − beta=1 W | Civilian at work | primary | 0.003892 ± 0.005560 |
| beta=1 D warmup-only − beta=1 W | Public coverage | primary | -0.001408 ± 0.002318 |
| beta=1 D warmup-only − beta=0 D_init | Same residence | primary | -0.001272 ± 0.002299 |
| beta=1 D warmup-only − beta=0 D_init | Commute >20 min | primary | -0.000389 ± 0.004210 |
| beta=1 D warmup-only − beta=0 D_init | Income >$50k | primary | 0.005528 ± 0.005153 |
| beta=1 D warmup-only − beta=0 D_init | Civilian at work | primary | 0.004872 ± 0.002377 |
| beta=1 D warmup-only − beta=0 D_init | Public coverage | primary | 0.001977 ± 0.002547 |
| beta=1 D warmup-only − beta=0 D_init | SEX | primary | 0.004023 ± 0.002785 |
| beta=1 D warmup-only − beta=0 D_init | SEX | catchup_inclusive | -0.001958 ± 0.007574 |
| beta=1 D warmup-only − beta=0 D_init | RAC1P | primary | -0.006288 ± 0.006371 |
| beta=1 D warmup-only − beta=0 D_init | RAC1P | catchup_inclusive | -0.002154 ± 0.007784 |
| beta=1 C persistent − beta=1 W | Same residence | primary | -0.001418 ± 0.000386 |
| beta=1 C persistent − beta=1 W | Commute >20 min | primary | 0.001150 ± 0.004091 |
| beta=1 C persistent − beta=1 W | Income >$50k | primary | 0.001566 ± 0.000272 |
| beta=1 C persistent − beta=1 W | Civilian at work | primary | 0.006371 ± 0.001326 |
| beta=1 C persistent − beta=1 W | Public coverage | primary | -0.001569 ± 0.001961 |
| beta=1 C persistent − beta=0 C_init | Same residence | primary | -0.000788 ± 0.002669 |
| beta=1 C persistent − beta=0 C_init | Commute >20 min | primary | 0.000019 ± 0.002820 |
| beta=1 C persistent − beta=0 C_init | Income >$50k | primary | 0.000821 ± 0.001036 |
| beta=1 C persistent − beta=0 C_init | Civilian at work | primary | 0.007161 ± 0.004071 |
| beta=1 C persistent − beta=0 C_init | Public coverage | primary | 0.001377 ± 0.004983 |
| beta=1 C persistent − beta=0 C_init | SEX | primary | 0.001109 ± 0.005577 |
| beta=1 C persistent − beta=0 C_init | SEX | catchup_inclusive | 0.001109 ± 0.005577 |
| beta=1 C persistent − beta=0 C_init | RAC1P | primary | -0.010471 ± 0.006638 |
| beta=1 C persistent − beta=0 C_init | RAC1P | catchup_inclusive | -0.016608 ± 0.011419 |
| beta=1 D persistent − beta=1 W | Same residence | primary | -0.001129 ± 0.000376 |
| beta=1 D persistent − beta=1 W | Commute >20 min | primary | 0.001380 ± 0.003501 |
| beta=1 D persistent − beta=1 W | Income >$50k | primary | 0.002005 ± 0.000348 |
| beta=1 D persistent − beta=1 W | Civilian at work | primary | 0.005852 ± 0.001004 |
| beta=1 D persistent − beta=1 W | Public coverage | primary | -0.000020 ± 0.000675 |
| beta=1 D persistent − beta=0 D_init | Same residence | primary | -0.002138 ± 0.005078 |
| beta=1 D persistent − beta=0 D_init | Commute >20 min | primary | 0.001043 ± 0.001487 |
| beta=1 D persistent − beta=0 D_init | Income >$50k | primary | 0.000496 ± 0.001192 |
| beta=1 D persistent − beta=0 D_init | Civilian at work | primary | 0.006832 ± 0.003743 |
| beta=1 D persistent − beta=0 D_init | Public coverage | primary | 0.003366 ± 0.002880 |
| beta=1 D persistent − beta=0 D_init | SEX | primary | -0.007415 ± 0.005387 |
| beta=1 D persistent − beta=0 D_init | SEX | catchup_inclusive | -0.013396 ± 0.004987 |
| beta=1 D persistent − beta=0 D_init | RAC1P | primary | -0.032972 ± 0.003172 |
| beta=1 D persistent − beta=0 D_init | RAC1P | catchup_inclusive | -0.026226 ± 0.009443 |
| beta=1 C persistent − beta=1 C warmup-only | Same residence | primary | 0.002538 ± 0.002701 |
| beta=1 C persistent − beta=1 C warmup-only | Commute >20 min | primary | 0.001717 ± 0.005883 |
| beta=1 C persistent − beta=1 C warmup-only | Income >$50k | primary | -0.005362 ± 0.004961 |
| beta=1 C persistent − beta=1 C warmup-only | Civilian at work | primary | 0.004044 ± 0.005087 |
| beta=1 C persistent − beta=1 C warmup-only | Public coverage | primary | -0.001502 ± 0.000833 |
| beta=1 C persistent − beta=1 C warmup-only | SEX | primary | -0.001525 ± 0.004403 |
| beta=1 C persistent − beta=1 C warmup-only | SEX | catchup_inclusive | -0.001525 ± 0.004403 |
| beta=1 C persistent − beta=1 C warmup-only | RAC1P | primary | -0.013724 ± 0.000767 |
| beta=1 C persistent − beta=1 C warmup-only | RAC1P | catchup_inclusive | -0.013724 ± 0.000767 |
| beta=1 D persistent − beta=1 D warmup-only | Same residence | primary | -0.000866 ± 0.003514 |
| beta=1 D persistent − beta=1 D warmup-only | Commute >20 min | primary | 0.001432 ± 0.005373 |
| beta=1 D persistent − beta=1 D warmup-only | Income >$50k | primary | -0.005031 ± 0.004711 |
| beta=1 D persistent − beta=1 D warmup-only | Civilian at work | primary | 0.001960 ± 0.005435 |
| beta=1 D persistent − beta=1 D warmup-only | Public coverage | primary | 0.001389 ± 0.001779 |
| beta=1 D persistent − beta=1 D warmup-only | SEX | primary | -0.011438 ± 0.002607 |
| beta=1 D persistent − beta=1 D warmup-only | SEX | catchup_inclusive | -0.011438 ± 0.002607 |
| beta=1 D persistent − beta=1 D warmup-only | RAC1P | primary | -0.026684 ± 0.004469 |
| beta=1 D persistent − beta=1 D warmup-only | RAC1P | catchup_inclusive | -0.024072 ± 0.006821 |
| beta=1 D warmup-only − beta=1 C warmup-only | Same residence | primary | 0.003693 ± 0.000923 |
| beta=1 D warmup-only − beta=1 C warmup-only | Commute >20 min | primary | 0.000515 ± 0.000403 |
| beta=1 D warmup-only − beta=1 C warmup-only | Income >$50k | primary | 0.000108 ± 0.001100 |
| beta=1 D warmup-only − beta=1 C warmup-only | Civilian at work | primary | 0.001564 ± 0.001288 |
| beta=1 D warmup-only − beta=1 C warmup-only | Public coverage | primary | -0.001342 ± 0.002916 |
| beta=1 D warmup-only − beta=1 C warmup-only | SEX | primary | 0.010160 ± 0.002782 |
| beta=1 D warmup-only − beta=1 C warmup-only | SEX | catchup_inclusive | 0.010160 ± 0.002782 |
| beta=1 D warmup-only − beta=1 C warmup-only | RAC1P | primary | 0.013516 ± 0.004240 |
| beta=1 D warmup-only − beta=1 C warmup-only | RAC1P | catchup_inclusive | 0.010904 ± 0.006583 |
| beta=1 D persistent − beta=1 C persistent | Same residence | primary | 0.000289 ± 0.000134 |
| beta=1 D persistent − beta=1 C persistent | Commute >20 min | primary | 0.000230 ± 0.000591 |
| beta=1 D persistent − beta=1 C persistent | Income >$50k | primary | 0.000439 ± 0.000075 |
| beta=1 D persistent − beta=1 C persistent | Civilian at work | primary | -0.000519 ± 0.000403 |
| beta=1 D persistent − beta=1 C persistent | Public coverage | primary | 0.001549 ± 0.002071 |
| beta=1 D persistent − beta=1 C persistent | SEX | primary | 0.000247 ± 0.000505 |
| beta=1 D persistent − beta=1 C persistent | SEX | catchup_inclusive | 0.000247 ± 0.000505 |
| beta=1 D persistent − beta=1 C persistent | RAC1P | primary | 0.000556 ± 0.000092 |
| beta=1 D persistent − beta=1 C persistent | RAC1P | catchup_inclusive | 0.000556 ± 0.000092 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Same residence | primary | -0.001460 ± 0.007733 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Commute >20 min | primary | -0.002931 ± 0.003504 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Income >$50k | primary | 0.004690 ± 0.005686 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Civilian at work | primary | 0.002603 ± 0.001666 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Public coverage | primary | -0.002462 ± 0.000899 |
| beta=1 C warmup-only − beta=.1 C warmup-only | SEX | primary | -0.001349 ± 0.003754 |
| beta=1 C warmup-only − beta=.1 C warmup-only | SEX | catchup_inclusive | -0.001349 ± 0.003754 |
| beta=1 C warmup-only − beta=.1 C warmup-only | RAC1P | primary | -0.008213 ± 0.013533 |
| beta=1 C warmup-only − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.008213 ± 0.013533 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Same residence | primary | -0.000385 ± 0.004401 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Commute >20 min | primary | -0.002392 ± 0.003522 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Income >$50k | primary | 0.003963 ± 0.003768 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Civilian at work | primary | 0.002062 ± 0.002172 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Public coverage | primary | -0.001778 ± 0.003087 |
| beta=1 D warmup-only − beta=.1 D warmup-only | SEX | primary | 0.003419 ± 0.001679 |
| beta=1 D warmup-only − beta=.1 D warmup-only | SEX | catchup_inclusive | 0.003321 ± 0.001736 |
| beta=1 D warmup-only − beta=.1 D warmup-only | RAC1P | primary | 0.007112 ± 0.005529 |
| beta=1 D warmup-only − beta=.1 D warmup-only | RAC1P | catchup_inclusive | 0.007699 ± 0.006155 |
| beta=1 C persistent − beta=.1 C persistent | Same residence | primary | -0.000336 ± 0.006935 |
| beta=1 C persistent − beta=.1 C persistent | Commute >20 min | primary | 0.000031 ± 0.001486 |
| beta=1 C persistent − beta=.1 C persistent | Income >$50k | primary | -0.000450 ± 0.000740 |
| beta=1 C persistent − beta=.1 C persistent | Civilian at work | primary | 0.000310 ± 0.003830 |
| beta=1 C persistent − beta=.1 C persistent | Public coverage | primary | -0.002290 ± 0.003936 |
| beta=1 C persistent − beta=.1 C persistent | SEX | primary | -0.004607 ± 0.001980 |
| beta=1 C persistent − beta=.1 C persistent | SEX | catchup_inclusive | -0.004607 ± 0.001980 |
| beta=1 C persistent − beta=.1 C persistent | RAC1P | primary | -0.006156 ± 0.001931 |
| beta=1 C persistent − beta=.1 C persistent | RAC1P | catchup_inclusive | -0.012274 ± 0.009138 |
| beta=1 D persistent − beta=.1 D persistent | Same residence | primary | -0.001350 ± 0.008781 |
| beta=1 D persistent − beta=.1 D persistent | Commute >20 min | primary | 0.000372 ± 0.001220 |
| beta=1 D persistent − beta=.1 D persistent | Income >$50k | primary | 0.000476 ± 0.000581 |
| beta=1 D persistent − beta=.1 D persistent | Civilian at work | primary | -0.000230 ± 0.003180 |
| beta=1 D persistent − beta=.1 D persistent | Public coverage | primary | -0.000615 ± 0.002817 |
| beta=1 D persistent − beta=.1 D persistent | SEX | primary | -0.004916 ± 0.002316 |
| beta=1 D persistent − beta=.1 D persistent | SEX | catchup_inclusive | -0.004916 ± 0.002316 |
| beta=1 D persistent − beta=.1 D persistent | RAC1P | primary | -0.005641 ± 0.001700 |
| beta=1 D persistent − beta=.1 D persistent | RAC1P | catchup_inclusive | -0.011143 ± 0.007932 |

### Validation

| Left − right | Target | Audit/head selector | Log-loss difference |
| --- | --- | --- | --- |
| W: historical beta=0 − I: shared initialization | Same residence | primary | 0.001756 ± 0.012352 |
| W: historical beta=0 − I: shared initialization | Commute >20 min | primary | -0.004204 ± 0.002941 |
| W: historical beta=0 − I: shared initialization | Income >$50k | primary | -0.012444 ± 0.008675 |
| W: historical beta=0 − I: shared initialization | Civilian at work | primary | -0.012337 ± 0.002784 |
| W: historical beta=0 − I: shared initialization | Public coverage | primary | -0.012614 ± 0.003691 |
| beta=.1 W − I: shared initialization | Same residence | primary | -0.001060 ± 0.003962 |
| beta=.1 W − I: shared initialization | Commute >20 min | primary | -0.004948 ± 0.003820 |
| beta=.1 W − I: shared initialization | Income >$50k | primary | -0.010864 ± 0.007416 |
| beta=.1 W − I: shared initialization | Civilian at work | primary | -0.009937 ± 0.003117 |
| beta=.1 W − I: shared initialization | Public coverage | primary | -0.009842 ± 0.005108 |
| beta=.1 W − W: historical beta=0 | Same residence | primary | -0.002816 ± 0.012177 |
| beta=.1 W − W: historical beta=0 | Commute >20 min | primary | -0.000744 ± 0.001142 |
| beta=.1 W − W: historical beta=0 | Income >$50k | primary | 0.001580 ± 0.003948 |
| beta=.1 W − W: historical beta=0 | Civilian at work | primary | 0.002400 ± 0.002454 |
| beta=.1 W − W: historical beta=0 | Public coverage | primary | 0.002771 ± 0.003290 |
| beta=.1 C warmup-only − beta=.1 W | Same residence | primary | -0.000157 ± 0.005232 |
| beta=.1 C warmup-only − beta=.1 W | Commute >20 min | primary | 0.001941 ± 0.002541 |
| beta=.1 C warmup-only − beta=.1 W | Income >$50k | primary | 0.003345 ± 0.002583 |
| beta=.1 C warmup-only − beta=.1 W | Civilian at work | primary | 0.001621 ± 0.001497 |
| beta=.1 C warmup-only − beta=.1 W | Public coverage | primary | 0.003596 ± 0.002786 |
| beta=.1 C warmup-only − beta=0 C_init | Same residence | primary | -0.007602 ± 0.005887 |
| beta=.1 C warmup-only − beta=0 C_init | Commute >20 min | primary | 0.000784 ± 0.002609 |
| beta=.1 C warmup-only − beta=0 C_init | Income >$50k | primary | 0.003388 ± 0.002058 |
| beta=.1 C warmup-only − beta=0 C_init | Civilian at work | primary | 0.001489 ± 0.000693 |
| beta=.1 C warmup-only − beta=0 C_init | Public coverage | primary | 0.003594 ± 0.003918 |
| beta=.1 C warmup-only − beta=0 C_init | SEX | primary | 0.005338 ± 0.002558 |
| beta=.1 C warmup-only − beta=0 C_init | SEX | catchup_inclusive | 0.005338 ± 0.002558 |
| beta=.1 C warmup-only − beta=0 C_init | RAC1P | primary | 0.002979 ± 0.001132 |
| beta=.1 C warmup-only − beta=0 C_init | RAC1P | catchup_inclusive | 0.003999 ± 0.002657 |
| beta=.1 D warmup-only − beta=.1 W | Same residence | primary | 0.005788 ± 0.009565 |
| beta=.1 D warmup-only − beta=.1 W | Commute >20 min | primary | 0.002165 ± 0.003309 |
| beta=.1 D warmup-only − beta=.1 W | Income >$50k | primary | 0.003761 ± 0.003811 |
| beta=.1 D warmup-only − beta=.1 W | Civilian at work | primary | 0.002629 ± 0.000888 |
| beta=.1 D warmup-only − beta=.1 W | Public coverage | primary | 0.003614 ± 0.004143 |
| beta=.1 D warmup-only − beta=0 D_init | Same residence | primary | -0.002939 ± 0.003012 |
| beta=.1 D warmup-only − beta=0 D_init | Commute >20 min | primary | 0.001337 ± 0.003129 |
| beta=.1 D warmup-only − beta=0 D_init | Income >$50k | primary | 0.003305 ± 0.001957 |
| beta=.1 D warmup-only − beta=0 D_init | Civilian at work | primary | 0.002737 ± 0.001862 |
| beta=.1 D warmup-only − beta=0 D_init | Public coverage | primary | 0.004701 ± 0.004305 |
| beta=.1 D warmup-only − beta=0 D_init | SEX | primary | 0.001645 ± 0.004993 |
| beta=.1 D warmup-only − beta=0 D_init | SEX | catchup_inclusive | -0.000693 ± 0.002378 |
| beta=.1 D warmup-only − beta=0 D_init | RAC1P | primary | -0.008482 ± 0.005375 |
| beta=.1 D warmup-only − beta=0 D_init | RAC1P | catchup_inclusive | -0.003708 ± 0.006160 |
| beta=.1 C persistent − beta=.1 W | Same residence | primary | -0.000569 ± 0.001227 |
| beta=.1 C persistent − beta=.1 W | Commute >20 min | primary | 0.000265 ± 0.000394 |
| beta=.1 C persistent − beta=.1 W | Income >$50k | primary | 0.001867 ± 0.001711 |
| beta=.1 C persistent − beta=.1 W | Civilian at work | primary | 0.003689 ± 0.001114 |
| beta=.1 C persistent − beta=.1 W | Public coverage | primary | 0.003264 ± 0.001379 |
| beta=.1 C persistent − beta=0 C_init | Same residence | primary | -0.008014 ± 0.010697 |
| beta=.1 C persistent − beta=0 C_init | Commute >20 min | primary | -0.000891 ± 0.000225 |
| beta=.1 C persistent − beta=0 C_init | Income >$50k | primary | 0.001910 ± 0.003712 |
| beta=.1 C persistent − beta=0 C_init | Civilian at work | primary | 0.003556 ± 0.000739 |
| beta=.1 C persistent − beta=0 C_init | Public coverage | primary | 0.003261 ± 0.003379 |
| beta=.1 C persistent − beta=0 C_init | SEX | primary | 0.000302 ± 0.003978 |
| beta=.1 C persistent − beta=0 C_init | SEX | catchup_inclusive | 0.000302 ± 0.003978 |
| beta=.1 C persistent − beta=0 C_init | RAC1P | primary | -0.006067 ± 0.003751 |
| beta=.1 C persistent − beta=0 C_init | RAC1P | catchup_inclusive | -0.006704 ± 0.003020 |
| beta=.1 D persistent − beta=.1 W | Same residence | primary | -0.000478 ± 0.001313 |
| beta=.1 D persistent − beta=.1 W | Commute >20 min | primary | 0.000685 ± 0.000794 |
| beta=.1 D persistent − beta=.1 W | Income >$50k | primary | 0.002487 ± 0.001356 |
| beta=.1 D persistent − beta=.1 W | Civilian at work | primary | 0.003703 ± 0.000474 |
| beta=.1 D persistent − beta=.1 W | Public coverage | primary | 0.003198 ± 0.001686 |
| beta=.1 D persistent − beta=0 D_init | Same residence | primary | -0.009206 ± 0.009901 |
| beta=.1 D persistent − beta=0 D_init | Commute >20 min | primary | -0.000143 ± 0.000316 |
| beta=.1 D persistent − beta=0 D_init | Income >$50k | primary | 0.002031 ± 0.003407 |
| beta=.1 D persistent − beta=0 D_init | Civilian at work | primary | 0.003810 ± 0.001236 |
| beta=.1 D persistent − beta=0 D_init | Public coverage | primary | 0.004285 ± 0.001734 |
| beta=.1 D persistent − beta=0 D_init | SEX | primary | -0.008589 ± 0.006462 |
| beta=.1 D persistent − beta=0 D_init | SEX | catchup_inclusive | -0.007825 ± 0.007659 |
| beta=.1 D persistent − beta=0 D_init | RAC1P | primary | -0.024474 ± 0.011290 |
| beta=.1 D persistent − beta=0 D_init | RAC1P | catchup_inclusive | -0.014949 ± 0.012398 |
| beta=.1 C persistent − beta=.1 C warmup-only | Same residence | primary | -0.000412 ± 0.004812 |
| beta=.1 C persistent − beta=.1 C warmup-only | Commute >20 min | primary | -0.001676 ± 0.002714 |
| beta=.1 C persistent − beta=.1 C warmup-only | Income >$50k | primary | -0.001478 ± 0.002696 |
| beta=.1 C persistent − beta=.1 C warmup-only | Civilian at work | primary | 0.002068 ± 0.000769 |
| beta=.1 C persistent − beta=.1 C warmup-only | Public coverage | primary | -0.000332 ± 0.001436 |
| beta=.1 C persistent − beta=.1 C warmup-only | SEX | primary | -0.005036 ± 0.001484 |
| beta=.1 C persistent − beta=.1 C warmup-only | SEX | catchup_inclusive | -0.005036 ± 0.001484 |
| beta=.1 C persistent − beta=.1 C warmup-only | RAC1P | primary | -0.009046 ± 0.003922 |
| beta=.1 C persistent − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.010703 ± 0.003446 |
| beta=.1 D persistent − beta=.1 D warmup-only | Same residence | primary | -0.006266 ± 0.008592 |
| beta=.1 D persistent − beta=.1 D warmup-only | Commute >20 min | primary | -0.001480 ± 0.003215 |
| beta=.1 D persistent − beta=.1 D warmup-only | Income >$50k | primary | -0.001274 ± 0.002867 |
| beta=.1 D persistent − beta=.1 D warmup-only | Civilian at work | primary | 0.001073 ± 0.001057 |
| beta=.1 D persistent − beta=.1 D warmup-only | Public coverage | primary | -0.000416 ± 0.002586 |
| beta=.1 D persistent − beta=.1 D warmup-only | SEX | primary | -0.010235 ± 0.001470 |
| beta=.1 D persistent − beta=.1 D warmup-only | SEX | catchup_inclusive | -0.007132 ± 0.006724 |
| beta=.1 D persistent − beta=.1 D warmup-only | RAC1P | primary | -0.015992 ± 0.006297 |
| beta=.1 D persistent − beta=.1 D warmup-only | RAC1P | catchup_inclusive | -0.011242 ± 0.006312 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Same residence | primary | 0.005945 ± 0.004362 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Commute >20 min | primary | 0.000224 ± 0.000803 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Income >$50k | primary | 0.000416 ± 0.002644 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Civilian at work | primary | 0.001008 ± 0.001680 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Public coverage | primary | 0.000017 ± 0.001857 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | SEX | primary | 0.006687 ± 0.000923 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | SEX | catchup_inclusive | 0.003584 ± 0.004452 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | RAC1P | primary | 0.008867 ± 0.004275 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | RAC1P | catchup_inclusive | 0.001761 ± 0.006377 |
| beta=.1 D persistent − beta=.1 C persistent | Same residence | primary | 0.000091 ± 0.000448 |
| beta=.1 D persistent − beta=.1 C persistent | Commute >20 min | primary | 0.000420 ± 0.000502 |
| beta=.1 D persistent − beta=.1 C persistent | Income >$50k | primary | 0.000620 ± 0.000403 |
| beta=.1 D persistent − beta=.1 C persistent | Civilian at work | primary | 0.000014 ± 0.000680 |
| beta=.1 D persistent − beta=.1 C persistent | Public coverage | primary | -0.000066 ± 0.000328 |
| beta=.1 D persistent − beta=.1 C persistent | SEX | primary | 0.001489 ± 0.000890 |
| beta=.1 D persistent − beta=.1 C persistent | SEX | catchup_inclusive | 0.001489 ± 0.000890 |
| beta=.1 D persistent − beta=.1 C persistent | RAC1P | primary | 0.001921 ± 0.000482 |
| beta=.1 D persistent − beta=.1 C persistent | RAC1P | catchup_inclusive | 0.001222 ± 0.000884 |
| beta=1 W − I: shared initialization | Same residence | primary | -0.000608 ± 0.000034 |
| beta=1 W − I: shared initialization | Commute >20 min | primary | -0.003492 ± 0.001592 |
| beta=1 W − I: shared initialization | Income >$50k | primary | -0.007954 ± 0.003913 |
| beta=1 W − I: shared initialization | Civilian at work | primary | -0.009517 ± 0.003055 |
| beta=1 W − I: shared initialization | Public coverage | primary | -0.003970 ± 0.001542 |
| beta=1 W − W: historical beta=0 | Same residence | primary | -0.002364 ± 0.012340 |
| beta=1 W − W: historical beta=0 | Commute >20 min | primary | 0.000712 ± 0.001687 |
| beta=1 W − W: historical beta=0 | Income >$50k | primary | 0.004490 ± 0.005954 |
| beta=1 W − W: historical beta=0 | Civilian at work | primary | 0.002820 ± 0.002113 |
| beta=1 W − W: historical beta=0 | Public coverage | primary | 0.008643 ± 0.003672 |
| beta=1 C warmup-only − beta=1 W | Same residence | primary | -0.002600 ± 0.002448 |
| beta=1 C warmup-only − beta=1 W | Commute >20 min | primary | 0.000682 ± 0.002849 |
| beta=1 C warmup-only − beta=1 W | Income >$50k | primary | -0.002173 ± 0.002368 |
| beta=1 C warmup-only − beta=1 W | Civilian at work | primary | -0.000492 ± 0.004403 |
| beta=1 C warmup-only − beta=1 W | Public coverage | primary | -0.002213 ± 0.004951 |
| beta=1 C warmup-only − beta=0 C_init | Same residence | primary | -0.009593 ± 0.010788 |
| beta=1 C warmup-only − beta=0 C_init | Commute >20 min | primary | 0.000981 ± 0.004346 |
| beta=1 C warmup-only − beta=0 C_init | Income >$50k | primary | 0.000780 ± 0.004057 |
| beta=1 C warmup-only − beta=0 C_init | Civilian at work | primary | -0.000204 ± 0.003108 |
| beta=1 C warmup-only − beta=0 C_init | Public coverage | primary | 0.003656 ± 0.002323 |
| beta=1 C warmup-only − beta=0 C_init | SEX | primary | -0.001103 ± 0.004203 |
| beta=1 C warmup-only − beta=0 C_init | SEX | catchup_inclusive | -0.001103 ± 0.004203 |
| beta=1 C warmup-only − beta=0 C_init | RAC1P | primary | -0.002346 ± 0.005035 |
| beta=1 C warmup-only − beta=0 C_init | RAC1P | catchup_inclusive | -0.001326 ± 0.004521 |
| beta=1 D warmup-only − beta=1 W | Same residence | primary | 0.000793 ± 0.006390 |
| beta=1 D warmup-only − beta=1 W | Commute >20 min | primary | 0.001951 ± 0.004508 |
| beta=1 D warmup-only − beta=1 W | Income >$50k | primary | -0.000503 ± 0.003597 |
| beta=1 D warmup-only − beta=1 W | Civilian at work | primary | 0.002315 ± 0.004657 |
| beta=1 D warmup-only − beta=1 W | Public coverage | primary | -0.002236 ± 0.005413 |
| beta=1 D warmup-only − beta=0 D_init | Same residence | primary | -0.007482 ± 0.005444 |
| beta=1 D warmup-only − beta=0 D_init | Commute >20 min | primary | 0.002579 ± 0.005950 |
| beta=1 D warmup-only − beta=0 D_init | Income >$50k | primary | 0.001950 ± 0.002352 |
| beta=1 D warmup-only − beta=0 D_init | Civilian at work | primary | 0.002843 ± 0.004214 |
| beta=1 D warmup-only − beta=0 D_init | Public coverage | primary | 0.004723 ± 0.002889 |
| beta=1 D warmup-only − beta=0 D_init | SEX | primary | 0.000159 ± 0.011989 |
| beta=1 D warmup-only − beta=0 D_init | SEX | catchup_inclusive | 0.000924 ± 0.013068 |
| beta=1 D warmup-only − beta=0 D_init | RAC1P | primary | 0.001542 ± 0.003031 |
| beta=1 D warmup-only − beta=0 D_init | RAC1P | catchup_inclusive | -0.000053 ± 0.005991 |
| beta=1 C persistent − beta=1 W | Same residence | primary | -0.000891 ± 0.000578 |
| beta=1 C persistent − beta=1 W | Commute >20 min | primary | 0.000271 ± 0.001355 |
| beta=1 C persistent − beta=1 W | Income >$50k | primary | 0.000205 ± 0.001202 |
| beta=1 C persistent − beta=1 W | Civilian at work | primary | 0.003357 ± 0.003902 |
| beta=1 C persistent − beta=1 W | Public coverage | primary | 0.000472 ± 0.001264 |
| beta=1 C persistent − beta=0 C_init | Same residence | primary | -0.007884 ± 0.011423 |
| beta=1 C persistent − beta=0 C_init | Commute >20 min | primary | 0.000570 ± 0.002574 |
| beta=1 C persistent − beta=0 C_init | Income >$50k | primary | 0.003158 ± 0.006854 |
| beta=1 C persistent − beta=0 C_init | Civilian at work | primary | 0.003645 ± 0.003112 |
| beta=1 C persistent − beta=0 C_init | Public coverage | primary | 0.006341 ± 0.002687 |
| beta=1 C persistent − beta=0 C_init | SEX | primary | -0.002605 ± 0.002473 |
| beta=1 C persistent − beta=0 C_init | SEX | catchup_inclusive | -0.002605 ± 0.002473 |
| beta=1 C persistent − beta=0 C_init | RAC1P | primary | -0.010967 ± 0.002823 |
| beta=1 C persistent − beta=0 C_init | RAC1P | catchup_inclusive | -0.009947 ± 0.004481 |
| beta=1 D persistent − beta=1 W | Same residence | primary | -0.000781 ± 0.000436 |
| beta=1 D persistent − beta=1 W | Commute >20 min | primary | 0.000442 ± 0.001112 |
| beta=1 D persistent − beta=1 W | Income >$50k | primary | 0.000476 ± 0.001137 |
| beta=1 D persistent − beta=1 W | Civilian at work | primary | 0.002961 ± 0.003710 |
| beta=1 D persistent − beta=1 W | Public coverage | primary | 0.000449 ± 0.001087 |
| beta=1 D persistent − beta=0 D_init | Same residence | primary | -0.009056 ± 0.011372 |
| beta=1 D persistent − beta=0 D_init | Commute >20 min | primary | 0.001070 ± 0.002608 |
| beta=1 D persistent − beta=0 D_init | Income >$50k | primary | 0.002929 ± 0.006489 |
| beta=1 D persistent − beta=0 D_init | Civilian at work | primary | 0.003490 ± 0.003607 |
| beta=1 D persistent − beta=0 D_init | Public coverage | primary | 0.007408 ± 0.003891 |
| beta=1 D persistent − beta=0 D_init | SEX | primary | -0.012064 ± 0.005334 |
| beta=1 D persistent − beta=0 D_init | SEX | catchup_inclusive | -0.011300 ± 0.006324 |
| beta=1 D persistent − beta=0 D_init | RAC1P | primary | -0.030939 ± 0.010930 |
| beta=1 D persistent − beta=0 D_init | RAC1P | catchup_inclusive | -0.019058 ± 0.014533 |
| beta=1 C persistent − beta=1 C warmup-only | Same residence | primary | 0.001709 ± 0.002063 |
| beta=1 C persistent − beta=1 C warmup-only | Commute >20 min | primary | -0.000411 ± 0.001802 |
| beta=1 C persistent − beta=1 C warmup-only | Income >$50k | primary | 0.002378 ± 0.002802 |
| beta=1 C persistent − beta=1 C warmup-only | Civilian at work | primary | 0.003850 ± 0.001732 |
| beta=1 C persistent − beta=1 C warmup-only | Public coverage | primary | 0.002685 ± 0.004173 |
| beta=1 C persistent − beta=1 C warmup-only | SEX | primary | -0.001501 ± 0.003180 |
| beta=1 C persistent − beta=1 C warmup-only | SEX | catchup_inclusive | -0.001501 ± 0.003180 |
| beta=1 C persistent − beta=1 C warmup-only | RAC1P | primary | -0.008621 ± 0.005822 |
| beta=1 C persistent − beta=1 C warmup-only | RAC1P | catchup_inclusive | -0.008621 ± 0.005822 |
| beta=1 D persistent − beta=1 D warmup-only | Same residence | primary | -0.001574 ± 0.005955 |
| beta=1 D persistent − beta=1 D warmup-only | Commute >20 min | primary | -0.001510 ± 0.003531 |
| beta=1 D persistent − beta=1 D warmup-only | Income >$50k | primary | 0.000979 ± 0.004139 |
| beta=1 D persistent − beta=1 D warmup-only | Civilian at work | primary | 0.000646 ± 0.001228 |
| beta=1 D persistent − beta=1 D warmup-only | Public coverage | primary | 0.002685 ± 0.005029 |
| beta=1 D persistent − beta=1 D warmup-only | SEX | primary | -0.012224 ± 0.006770 |
| beta=1 D persistent − beta=1 D warmup-only | SEX | catchup_inclusive | -0.012224 ± 0.006770 |
| beta=1 D persistent − beta=1 D warmup-only | RAC1P | primary | -0.032481 ± 0.011496 |
| beta=1 D persistent − beta=1 D warmup-only | RAC1P | catchup_inclusive | -0.019005 ± 0.020510 |
| beta=1 D warmup-only − beta=1 C warmup-only | Same residence | primary | 0.003394 ± 0.005396 |
| beta=1 D warmup-only − beta=1 C warmup-only | Commute >20 min | primary | 0.001269 ± 0.001660 |
| beta=1 D warmup-only − beta=1 C warmup-only | Income >$50k | primary | 0.001670 ± 0.001237 |
| beta=1 D warmup-only − beta=1 C warmup-only | Civilian at work | primary | 0.002808 ± 0.001078 |
| beta=1 D warmup-only − beta=1 C warmup-only | Public coverage | primary | -0.000023 ± 0.002705 |
| beta=1 D warmup-only − beta=1 C warmup-only | SEX | primary | 0.011642 ± 0.007194 |
| beta=1 D warmup-only − beta=1 C warmup-only | SEX | catchup_inclusive | 0.011642 ± 0.007194 |
| beta=1 D warmup-only − beta=1 C warmup-only | RAC1P | primary | 0.024217 ± 0.007828 |
| beta=1 D warmup-only − beta=1 C warmup-only | RAC1P | catchup_inclusive | 0.010740 ± 0.018973 |
| beta=1 D persistent − beta=1 C persistent | Same residence | primary | 0.000110 ± 0.000206 |
| beta=1 D persistent − beta=1 C persistent | Commute >20 min | primary | 0.000171 ± 0.000269 |
| beta=1 D persistent − beta=1 C persistent | Income >$50k | primary | 0.000271 ± 0.000213 |
| beta=1 D persistent − beta=1 C persistent | Civilian at work | primary | -0.000396 ± 0.000203 |
| beta=1 D persistent − beta=1 C persistent | Public coverage | primary | -0.000023 ± 0.000303 |
| beta=1 D persistent − beta=1 C persistent | SEX | primary | 0.000920 ± 0.000902 |
| beta=1 D persistent − beta=1 C persistent | SEX | catchup_inclusive | 0.000920 ± 0.000902 |
| beta=1 D persistent − beta=1 C persistent | RAC1P | primary | 0.000356 ± 0.000267 |
| beta=1 D persistent − beta=1 C persistent | RAC1P | catchup_inclusive | 0.000356 ± 0.000267 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Same residence | primary | -0.001991 ± 0.004902 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Commute >20 min | primary | 0.000197 ± 0.005419 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Income >$50k | primary | -0.002609 ± 0.003091 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Civilian at work | primary | -0.001693 ± 0.003764 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Public coverage | primary | 0.000062 ± 0.004675 |
| beta=1 C warmup-only − beta=.1 C warmup-only | SEX | primary | -0.006441 ± 0.001760 |
| beta=1 C warmup-only − beta=.1 C warmup-only | SEX | catchup_inclusive | -0.006441 ± 0.001760 |
| beta=1 C warmup-only − beta=.1 C warmup-only | RAC1P | primary | -0.005325 ± 0.006137 |
| beta=1 C warmup-only − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.005325 ± 0.006137 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Same residence | primary | -0.004543 ± 0.003619 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Commute >20 min | primary | 0.001242 ± 0.007215 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Income >$50k | primary | -0.001355 ± 0.001895 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Civilian at work | primary | 0.000106 ± 0.005716 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Public coverage | primary | 0.000022 ± 0.003711 |
| beta=1 D warmup-only − beta=.1 D warmup-only | SEX | primary | -0.001486 ± 0.007104 |
| beta=1 D warmup-only − beta=.1 D warmup-only | SEX | catchup_inclusive | 0.001617 ± 0.011602 |
| beta=1 D warmup-only − beta=.1 D warmup-only | RAC1P | primary | 0.010024 ± 0.005762 |
| beta=1 D warmup-only − beta=.1 D warmup-only | RAC1P | catchup_inclusive | 0.003654 ± 0.011885 |
| beta=1 C persistent − beta=.1 C persistent | Same residence | primary | 0.000130 ± 0.002454 |
| beta=1 C persistent − beta=.1 C persistent | Commute >20 min | primary | 0.001461 ± 0.002772 |
| beta=1 C persistent − beta=.1 C persistent | Income >$50k | primary | 0.001247 ± 0.003145 |
| beta=1 C persistent − beta=.1 C persistent | Civilian at work | primary | 0.000089 ± 0.003363 |
| beta=1 C persistent − beta=.1 C persistent | Public coverage | primary | 0.003080 ± 0.005567 |
| beta=1 C persistent − beta=.1 C persistent | SEX | primary | -0.002907 ± 0.002222 |
| beta=1 C persistent − beta=.1 C persistent | SEX | catchup_inclusive | -0.002907 ± 0.002222 |
| beta=1 C persistent − beta=.1 C persistent | RAC1P | primary | -0.004900 ± 0.001301 |
| beta=1 C persistent − beta=.1 C persistent | RAC1P | catchup_inclusive | -0.003243 ± 0.002911 |
| beta=1 D persistent − beta=.1 D persistent | Same residence | primary | 0.000150 ± 0.002858 |
| beta=1 D persistent − beta=.1 D persistent | Commute >20 min | primary | 0.001213 ± 0.002312 |
| beta=1 D persistent − beta=.1 D persistent | Income >$50k | primary | 0.000899 ± 0.003102 |
| beta=1 D persistent − beta=.1 D persistent | Civilian at work | primary | -0.000321 ± 0.003782 |
| beta=1 D persistent − beta=.1 D persistent | Public coverage | primary | 0.003123 ± 0.005429 |
| beta=1 D persistent − beta=.1 D persistent | SEX | primary | -0.003475 ± 0.002197 |
| beta=1 D persistent − beta=.1 D persistent | SEX | catchup_inclusive | -0.003475 ± 0.002197 |
| beta=1 D persistent − beta=.1 D persistent | RAC1P | primary | -0.006465 ± 0.000904 |
| beta=1 D persistent − beta=.1 D persistent | RAC1P | catchup_inclusive | -0.004109 ± 0.003719 |

### Validation person weighted

| Left − right | Target | Audit/head selector | Log-loss difference |
| --- | --- | --- | --- |
| W: historical beta=0 − I: shared initialization | Same residence | primary | -0.000246 ± 0.014847 |
| W: historical beta=0 − I: shared initialization | Commute >20 min | primary | -0.001266 ± 0.001776 |
| W: historical beta=0 − I: shared initialization | Income >$50k | primary | -0.015176 ± 0.009201 |
| W: historical beta=0 − I: shared initialization | Civilian at work | primary | -0.010182 ± 0.003007 |
| W: historical beta=0 − I: shared initialization | Public coverage | primary | -0.009833 ± 0.001014 |
| beta=.1 W − I: shared initialization | Same residence | primary | -0.001051 ± 0.005779 |
| beta=.1 W − I: shared initialization | Commute >20 min | primary | -0.003302 ± 0.003689 |
| beta=.1 W − I: shared initialization | Income >$50k | primary | -0.013151 ± 0.007901 |
| beta=.1 W − I: shared initialization | Civilian at work | primary | -0.006954 ± 0.004568 |
| beta=.1 W − I: shared initialization | Public coverage | primary | -0.009694 ± 0.005832 |
| beta=.1 W − W: historical beta=0 | Same residence | primary | -0.000805 ± 0.010519 |
| beta=.1 W − W: historical beta=0 | Commute >20 min | primary | -0.002036 ± 0.001913 |
| beta=.1 W − W: historical beta=0 | Income >$50k | primary | 0.002025 ± 0.003121 |
| beta=.1 W − W: historical beta=0 | Civilian at work | primary | 0.003229 ± 0.002870 |
| beta=.1 W − W: historical beta=0 | Public coverage | primary | 0.000139 ± 0.004913 |
| beta=.1 C warmup-only − beta=.1 W | Same residence | primary | -0.000629 ± 0.004325 |
| beta=.1 C warmup-only − beta=.1 W | Commute >20 min | primary | 0.003017 ± 0.003168 |
| beta=.1 C warmup-only − beta=.1 W | Income >$50k | primary | 0.004547 ± 0.003848 |
| beta=.1 C warmup-only − beta=.1 W | Civilian at work | primary | 0.002491 ± 0.000456 |
| beta=.1 C warmup-only − beta=.1 W | Public coverage | primary | 0.004531 ± 0.004548 |
| beta=.1 C warmup-only − beta=0 C_init | Same residence | primary | -0.007226 ± 0.006494 |
| beta=.1 C warmup-only − beta=0 C_init | Commute >20 min | primary | 0.000537 ± 0.004075 |
| beta=.1 C warmup-only − beta=0 C_init | Income >$50k | primary | 0.005316 ± 0.000300 |
| beta=.1 C warmup-only − beta=0 C_init | Civilian at work | primary | 0.003613 ± 0.001493 |
| beta=.1 C warmup-only − beta=0 C_init | Public coverage | primary | 0.002343 ± 0.002786 |
| beta=.1 C warmup-only − beta=0 C_init | SEX | primary | 0.003316 ± 0.003854 |
| beta=.1 C warmup-only − beta=0 C_init | SEX | catchup_inclusive | 0.003316 ± 0.003854 |
| beta=.1 C warmup-only − beta=0 C_init | RAC1P | primary | 0.005334 ± 0.004515 |
| beta=.1 C warmup-only − beta=0 C_init | RAC1P | catchup_inclusive | 0.002689 ± 0.007210 |
| beta=.1 D warmup-only − beta=.1 W | Same residence | primary | 0.005699 ± 0.009003 |
| beta=.1 D warmup-only − beta=.1 W | Commute >20 min | primary | 0.003579 ± 0.003983 |
| beta=.1 D warmup-only − beta=.1 W | Income >$50k | primary | 0.004610 ± 0.005271 |
| beta=.1 D warmup-only − beta=.1 W | Civilian at work | primary | 0.003278 ± 0.000987 |
| beta=.1 D warmup-only − beta=.1 W | Public coverage | primary | 0.004517 ± 0.005769 |
| beta=.1 D warmup-only − beta=0 D_init | Same residence | primary | -0.001231 ± 0.006666 |
| beta=.1 D warmup-only − beta=0 D_init | Commute >20 min | primary | 0.001752 ± 0.003595 |
| beta=.1 D warmup-only − beta=0 D_init | Income >$50k | primary | 0.005178 ± 0.003467 |
| beta=.1 D warmup-only − beta=0 D_init | Civilian at work | primary | 0.004803 ± 0.001856 |
| beta=.1 D warmup-only − beta=0 D_init | Public coverage | primary | 0.002290 ± 0.004101 |
| beta=.1 D warmup-only − beta=0 D_init | SEX | primary | -0.001528 ± 0.006447 |
| beta=.1 D warmup-only − beta=0 D_init | SEX | catchup_inclusive | -0.003611 ± 0.002843 |
| beta=.1 D warmup-only − beta=0 D_init | RAC1P | primary | -0.011131 ± 0.004095 |
| beta=.1 D warmup-only − beta=0 D_init | RAC1P | catchup_inclusive | -0.002813 ± 0.002330 |
| beta=.1 C persistent − beta=.1 W | Same residence | primary | 0.001200 ± 0.003350 |
| beta=.1 C persistent − beta=.1 W | Commute >20 min | primary | 0.000736 ± 0.000866 |
| beta=.1 C persistent − beta=.1 W | Income >$50k | primary | 0.001763 ± 0.001790 |
| beta=.1 C persistent − beta=.1 W | Civilian at work | primary | 0.003477 ± 0.001136 |
| beta=.1 C persistent − beta=.1 W | Public coverage | primary | 0.005071 ± 0.003212 |
| beta=.1 C persistent − beta=0 C_init | Same residence | primary | -0.005396 ± 0.012782 |
| beta=.1 C persistent − beta=0 C_init | Commute >20 min | primary | -0.001744 ± 0.002130 |
| beta=.1 C persistent − beta=0 C_init | Income >$50k | primary | 0.002532 ± 0.003000 |
| beta=.1 C persistent − beta=0 C_init | Civilian at work | primary | 0.004599 ± 0.001086 |
| beta=.1 C persistent − beta=0 C_init | Public coverage | primary | 0.002883 ± 0.004461 |
| beta=.1 C persistent − beta=0 C_init | SEX | primary | 0.000008 ± 0.004502 |
| beta=.1 C persistent − beta=0 C_init | SEX | catchup_inclusive | 0.000008 ± 0.004502 |
| beta=.1 C persistent − beta=0 C_init | RAC1P | primary | -0.004022 ± 0.003297 |
| beta=.1 C persistent − beta=0 C_init | RAC1P | catchup_inclusive | -0.006157 ± 0.005878 |
| beta=.1 D persistent − beta=.1 W | Same residence | primary | 0.002188 ± 0.002942 |
| beta=.1 D persistent − beta=.1 W | Commute >20 min | primary | 0.001224 ± 0.000684 |
| beta=.1 D persistent − beta=.1 W | Income >$50k | primary | 0.002252 ± 0.001588 |
| beta=.1 D persistent − beta=.1 W | Civilian at work | primary | 0.003478 ± 0.000928 |
| beta=.1 D persistent − beta=.1 W | Public coverage | primary | 0.004929 ± 0.003212 |
| beta=.1 D persistent − beta=0 D_init | Same residence | primary | -0.004741 ± 0.009211 |
| beta=.1 D persistent − beta=0 D_init | Commute >20 min | primary | -0.000603 ± 0.001145 |
| beta=.1 D persistent − beta=0 D_init | Income >$50k | primary | 0.002820 ± 0.002649 |
| beta=.1 D persistent − beta=0 D_init | Civilian at work | primary | 0.005003 ± 0.001183 |
| beta=.1 D persistent − beta=0 D_init | Public coverage | primary | 0.002702 ± 0.001545 |
| beta=.1 D persistent − beta=0 D_init | SEX | primary | -0.010394 ± 0.005941 |
| beta=.1 D persistent − beta=0 D_init | SEX | catchup_inclusive | -0.012715 ± 0.002138 |
| beta=.1 D persistent − beta=0 D_init | RAC1P | primary | -0.024836 ± 0.002541 |
| beta=.1 D persistent − beta=0 D_init | RAC1P | catchup_inclusive | -0.013285 ± 0.006651 |
| beta=.1 C persistent − beta=.1 C warmup-only | Same residence | primary | 0.001829 ± 0.006370 |
| beta=.1 C persistent − beta=.1 C warmup-only | Commute >20 min | primary | -0.002281 ± 0.002768 |
| beta=.1 C persistent − beta=.1 C warmup-only | Income >$50k | primary | -0.002784 ± 0.002739 |
| beta=.1 C persistent − beta=.1 C warmup-only | Civilian at work | primary | 0.000986 ± 0.000682 |
| beta=.1 C persistent − beta=.1 C warmup-only | Public coverage | primary | 0.000540 ± 0.002187 |
| beta=.1 C persistent − beta=.1 C warmup-only | SEX | primary | -0.003308 ± 0.002034 |
| beta=.1 C persistent − beta=.1 C warmup-only | SEX | catchup_inclusive | -0.003308 ± 0.002034 |
| beta=.1 C persistent − beta=.1 C warmup-only | RAC1P | primary | -0.009356 ± 0.007374 |
| beta=.1 C persistent − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.008846 ± 0.007403 |
| beta=.1 D persistent − beta=.1 D warmup-only | Same residence | primary | -0.003511 ± 0.007469 |
| beta=.1 D persistent − beta=.1 D warmup-only | Commute >20 min | primary | -0.002355 ± 0.003378 |
| beta=.1 D persistent − beta=.1 D warmup-only | Income >$50k | primary | -0.002358 ± 0.003684 |
| beta=.1 D persistent − beta=.1 D warmup-only | Civilian at work | primary | 0.000200 ± 0.001468 |
| beta=.1 D persistent − beta=.1 D warmup-only | Public coverage | primary | 0.000413 ± 0.002576 |
| beta=.1 D persistent − beta=.1 D warmup-only | SEX | primary | -0.008866 ± 0.001455 |
| beta=.1 D persistent − beta=.1 D warmup-only | SEX | catchup_inclusive | -0.009104 ± 0.001670 |
| beta=.1 D persistent − beta=.1 D warmup-only | RAC1P | primary | -0.013706 ± 0.001611 |
| beta=.1 D persistent − beta=.1 D warmup-only | RAC1P | catchup_inclusive | -0.010472 ± 0.005109 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Same residence | primary | 0.006328 ± 0.005966 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Commute >20 min | primary | 0.000563 ± 0.001639 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Income >$50k | primary | 0.000064 ± 0.003133 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Civilian at work | primary | 0.000787 ± 0.001290 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | Public coverage | primary | -0.000014 ± 0.001764 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | SEX | primary | 0.007110 ± 0.001654 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | SEX | catchup_inclusive | 0.007348 ± 0.002062 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | RAC1P | primary | 0.005905 ± 0.008027 |
| beta=.1 D warmup-only − beta=.1 C warmup-only | RAC1P | catchup_inclusive | 0.001922 ± 0.002748 |
| beta=.1 D persistent − beta=.1 C persistent | Same residence | primary | 0.000988 ± 0.001514 |
| beta=.1 D persistent − beta=.1 C persistent | Commute >20 min | primary | 0.000489 ± 0.000711 |
| beta=.1 D persistent − beta=.1 C persistent | Income >$50k | primary | 0.000490 ± 0.000261 |
| beta=.1 D persistent − beta=.1 C persistent | Civilian at work | primary | 0.000001 ± 0.000507 |
| beta=.1 D persistent − beta=.1 C persistent | Public coverage | primary | -0.000142 ± 0.000432 |
| beta=.1 D persistent − beta=.1 C persistent | SEX | primary | 0.001551 ± 0.000342 |
| beta=.1 D persistent − beta=.1 C persistent | SEX | catchup_inclusive | 0.001551 ± 0.000342 |
| beta=.1 D persistent − beta=.1 C persistent | RAC1P | primary | 0.001555 ± 0.000172 |
| beta=.1 D persistent − beta=.1 C persistent | RAC1P | catchup_inclusive | 0.000297 ± 0.002277 |
| beta=1 W − I: shared initialization | Same residence | primary | -0.000445 ± 0.000731 |
| beta=1 W − I: shared initialization | Commute >20 min | primary | -0.002829 ± 0.000256 |
| beta=1 W − I: shared initialization | Income >$50k | primary | -0.009477 ± 0.004752 |
| beta=1 W − I: shared initialization | Civilian at work | primary | -0.006081 ± 0.004574 |
| beta=1 W − I: shared initialization | Public coverage | primary | -0.003497 ± 0.002427 |
| beta=1 W − W: historical beta=0 | Same residence | primary | -0.000199 ± 0.014117 |
| beta=1 W − W: historical beta=0 | Commute >20 min | primary | -0.001563 ± 0.001570 |
| beta=1 W − W: historical beta=0 | Income >$50k | primary | 0.005699 ± 0.006689 |
| beta=1 W − W: historical beta=0 | Civilian at work | primary | 0.004101 ± 0.002455 |
| beta=1 W − W: historical beta=0 | Public coverage | primary | 0.006336 ± 0.001564 |
| beta=1 C warmup-only − beta=1 W | Same residence | primary | -0.003339 ± 0.001815 |
| beta=1 C warmup-only − beta=1 W | Commute >20 min | primary | 0.002516 ± 0.000620 |
| beta=1 C warmup-only − beta=1 W | Income >$50k | primary | -0.003203 ± 0.002764 |
| beta=1 C warmup-only − beta=1 W | Civilian at work | primary | -0.000269 ± 0.004721 |
| beta=1 C warmup-only − beta=1 W | Public coverage | primary | -0.000659 ± 0.003986 |
| beta=1 C warmup-only − beta=0 C_init | Same residence | primary | -0.009330 ± 0.013271 |
| beta=1 C warmup-only − beta=0 C_init | Commute >20 min | primary | 0.000509 ± 0.002671 |
| beta=1 C warmup-only − beta=0 C_init | Income >$50k | primary | 0.001240 ± 0.005306 |
| beta=1 C warmup-only − beta=0 C_init | Civilian at work | primary | 0.001725 ± 0.003568 |
| beta=1 C warmup-only − beta=0 C_init | Public coverage | primary | 0.003350 ± 0.003145 |
| beta=1 C warmup-only − beta=0 C_init | SEX | primary | -0.002396 ± 0.005231 |
| beta=1 C warmup-only − beta=0 C_init | SEX | catchup_inclusive | -0.002396 ± 0.005231 |
| beta=1 C warmup-only − beta=0 C_init | RAC1P | primary | 0.000189 ± 0.005517 |
| beta=1 C warmup-only − beta=0 C_init | RAC1P | catchup_inclusive | -0.002456 ± 0.008192 |
| beta=1 D warmup-only − beta=1 W | Same residence | primary | -0.000286 ± 0.006305 |
| beta=1 D warmup-only − beta=1 W | Commute >20 min | primary | 0.003394 ± 0.002308 |
| beta=1 D warmup-only − beta=1 W | Income >$50k | primary | -0.001038 ± 0.004764 |
| beta=1 D warmup-only − beta=1 W | Civilian at work | primary | 0.002201 ± 0.005293 |
| beta=1 D warmup-only − beta=1 W | Public coverage | primary | -0.000559 ± 0.002951 |
| beta=1 D warmup-only − beta=0 D_init | Same residence | primary | -0.006610 ± 0.007335 |
| beta=1 D warmup-only − beta=0 D_init | Commute >20 min | primary | 0.002040 ± 0.005125 |
| beta=1 D warmup-only − beta=0 D_init | Income >$50k | primary | 0.003203 ± 0.003017 |
| beta=1 D warmup-only − beta=0 D_init | Civilian at work | primary | 0.004599 ± 0.004995 |
| beta=1 D warmup-only − beta=0 D_init | Public coverage | primary | 0.003410 ± 0.004236 |
| beta=1 D warmup-only − beta=0 D_init | SEX | primary | -0.001490 ± 0.011323 |
| beta=1 D warmup-only − beta=0 D_init | SEX | catchup_inclusive | -0.003811 ± 0.008059 |
| beta=1 D warmup-only − beta=0 D_init | RAC1P | primary | 0.001403 ± 0.005598 |
| beta=1 D warmup-only − beta=0 D_init | RAC1P | catchup_inclusive | -0.001784 ± 0.010840 |
| beta=1 C persistent − beta=1 W | Same residence | primary | -0.000818 ± 0.000679 |
| beta=1 C persistent − beta=1 W | Commute >20 min | primary | 0.000999 ± 0.000712 |
| beta=1 C persistent − beta=1 W | Income >$50k | primary | -0.000339 ± 0.001200 |
| beta=1 C persistent − beta=1 W | Civilian at work | primary | 0.002835 ± 0.003362 |
| beta=1 C persistent − beta=1 W | Public coverage | primary | 0.001036 ± 0.002752 |
| beta=1 C persistent − beta=0 C_init | Same residence | primary | -0.006809 ± 0.014354 |
| beta=1 C persistent − beta=0 C_init | Commute >20 min | primary | -0.001008 ± 0.002281 |
| beta=1 C persistent − beta=0 C_init | Income >$50k | primary | 0.004104 ± 0.008991 |
| beta=1 C persistent − beta=0 C_init | Civilian at work | primary | 0.004830 ± 0.003385 |
| beta=1 C persistent − beta=0 C_init | Public coverage | primary | 0.005045 ± 0.002633 |
| beta=1 C persistent − beta=0 C_init | SEX | primary | -0.003963 ± 0.003334 |
| beta=1 C persistent − beta=0 C_init | SEX | catchup_inclusive | -0.003963 ± 0.003334 |
| beta=1 C persistent − beta=0 C_init | RAC1P | primary | -0.009342 ± 0.003823 |
| beta=1 C persistent − beta=0 C_init | RAC1P | catchup_inclusive | -0.011986 ± 0.003379 |
| beta=1 D persistent − beta=1 W | Same residence | primary | -0.000730 ± 0.000509 |
| beta=1 D persistent − beta=1 W | Commute >20 min | primary | 0.001105 ± 0.000451 |
| beta=1 D persistent − beta=1 W | Income >$50k | primary | -0.000022 ± 0.001069 |
| beta=1 D persistent − beta=1 W | Civilian at work | primary | 0.002526 ± 0.002817 |
| beta=1 D persistent − beta=1 W | Public coverage | primary | 0.000263 ± 0.001364 |
| beta=1 D persistent − beta=0 D_init | Same residence | primary | -0.007054 ± 0.012522 |
| beta=1 D persistent − beta=0 D_init | Commute >20 min | primary | -0.000249 ± 0.002988 |
| beta=1 D persistent − beta=0 D_init | Income >$50k | primary | 0.004219 ± 0.008631 |
| beta=1 D persistent − beta=0 D_init | Civilian at work | primary | 0.004924 ± 0.003550 |
| beta=1 D persistent − beta=0 D_init | Public coverage | primary | 0.004233 ± 0.004267 |
| beta=1 D persistent − beta=0 D_init | SEX | primary | -0.015313 ± 0.005374 |
| beta=1 D persistent − beta=0 D_init | SEX | catchup_inclusive | -0.017633 ± 0.002390 |
| beta=1 D persistent − beta=0 D_init | RAC1P | primary | -0.031517 ± 0.001753 |
| beta=1 D persistent − beta=0 D_init | RAC1P | catchup_inclusive | -0.019217 ± 0.007085 |
| beta=1 C persistent − beta=1 C warmup-only | Same residence | primary | 0.002521 ± 0.001137 |
| beta=1 C persistent − beta=1 C warmup-only | Commute >20 min | primary | -0.001516 ± 0.000529 |
| beta=1 C persistent − beta=1 C warmup-only | Income >$50k | primary | 0.002864 ± 0.003738 |
| beta=1 C persistent − beta=1 C warmup-only | Civilian at work | primary | 0.003104 ± 0.003499 |
| beta=1 C persistent − beta=1 C warmup-only | Public coverage | primary | 0.001695 ± 0.001424 |
| beta=1 C persistent − beta=1 C warmup-only | SEX | primary | -0.001568 ± 0.002587 |
| beta=1 C persistent − beta=1 C warmup-only | SEX | catchup_inclusive | -0.001568 ± 0.002587 |
| beta=1 C persistent − beta=1 C warmup-only | RAC1P | primary | -0.009530 ± 0.005008 |
| beta=1 C persistent − beta=1 C warmup-only | RAC1P | catchup_inclusive | -0.009530 ± 0.005008 |
| beta=1 D persistent − beta=1 D warmup-only | Same residence | primary | -0.000444 ± 0.005859 |
| beta=1 D persistent − beta=1 D warmup-only | Commute >20 min | primary | -0.002289 ± 0.002137 |
| beta=1 D persistent − beta=1 D warmup-only | Income >$50k | primary | 0.001016 ± 0.005770 |
| beta=1 D persistent − beta=1 D warmup-only | Civilian at work | primary | 0.000325 ± 0.003167 |
| beta=1 D persistent − beta=1 D warmup-only | Public coverage | primary | 0.000823 ± 0.002468 |
| beta=1 D persistent − beta=1 D warmup-only | SEX | primary | -0.013823 ± 0.005984 |
| beta=1 D persistent − beta=1 D warmup-only | SEX | catchup_inclusive | -0.013823 ± 0.005984 |
| beta=1 D persistent − beta=1 D warmup-only | RAC1P | primary | -0.032920 ± 0.007341 |
| beta=1 D persistent − beta=1 D warmup-only | RAC1P | catchup_inclusive | -0.017432 ± 0.016631 |
| beta=1 D warmup-only − beta=1 C warmup-only | Same residence | primary | 0.003053 ± 0.004491 |
| beta=1 D warmup-only − beta=1 C warmup-only | Commute >20 min | primary | 0.000879 ± 0.001720 |
| beta=1 D warmup-only − beta=1 C warmup-only | Income >$50k | primary | 0.002164 ± 0.002015 |
| beta=1 D warmup-only − beta=1 C warmup-only | Civilian at work | primary | 0.002470 ± 0.001314 |
| beta=1 D warmup-only − beta=1 C warmup-only | Public coverage | primary | 0.000099 ± 0.004176 |
| beta=1 D warmup-only − beta=1 C warmup-only | SEX | primary | 0.012859 ± 0.006055 |
| beta=1 D warmup-only − beta=1 C warmup-only | SEX | catchup_inclusive | 0.012859 ± 0.006055 |
| beta=1 D warmup-only − beta=1 C warmup-only | RAC1P | primary | 0.023583 ± 0.002547 |
| beta=1 D warmup-only − beta=1 C warmup-only | RAC1P | catchup_inclusive | 0.008095 ± 0.014779 |
| beta=1 D persistent − beta=1 C persistent | Same residence | primary | 0.000088 ± 0.000330 |
| beta=1 D persistent − beta=1 C persistent | Commute >20 min | primary | 0.000106 ± 0.000261 |
| beta=1 D persistent − beta=1 C persistent | Income >$50k | primary | 0.000317 ± 0.000353 |
| beta=1 D persistent − beta=1 C persistent | Civilian at work | primary | -0.000308 ± 0.000597 |
| beta=1 D persistent − beta=1 C persistent | Public coverage | primary | -0.000773 ± 0.001682 |
| beta=1 D persistent − beta=1 C persistent | SEX | primary | 0.000604 ± 0.000570 |
| beta=1 D persistent − beta=1 C persistent | SEX | catchup_inclusive | 0.000604 ± 0.000570 |
| beta=1 D persistent − beta=1 C persistent | RAC1P | primary | 0.000193 ± 0.000478 |
| beta=1 D persistent − beta=1 C persistent | RAC1P | catchup_inclusive | 0.000193 ± 0.000478 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Same residence | primary | -0.002104 ± 0.006940 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Commute >20 min | primary | -0.000029 ± 0.006561 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Income >$50k | primary | -0.004075 ± 0.005022 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Civilian at work | primary | -0.001888 ± 0.004882 |
| beta=1 C warmup-only − beta=.1 C warmup-only | Public coverage | primary | 0.001007 ± 0.005929 |
| beta=1 C warmup-only − beta=.1 C warmup-only | SEX | primary | -0.005711 ± 0.001878 |
| beta=1 C warmup-only − beta=.1 C warmup-only | SEX | catchup_inclusive | -0.005711 ± 0.001878 |
| beta=1 C warmup-only − beta=.1 C warmup-only | RAC1P | primary | -0.005145 ± 0.009625 |
| beta=1 C warmup-only − beta=.1 C warmup-only | RAC1P | catchup_inclusive | -0.005145 ± 0.009625 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Same residence | primary | -0.005379 ± 0.004604 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Commute >20 min | primary | 0.000288 ± 0.008232 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Income >$50k | primary | -0.001974 ± 0.003859 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Civilian at work | primary | -0.000204 ± 0.006542 |
| beta=1 D warmup-only − beta=.1 D warmup-only | Public coverage | primary | 0.001121 ± 0.003278 |
| beta=1 D warmup-only − beta=.1 D warmup-only | SEX | primary | 0.000037 ± 0.006671 |
| beta=1 D warmup-only − beta=.1 D warmup-only | SEX | catchup_inclusive | -0.000200 ± 0.006469 |
| beta=1 D warmup-only − beta=.1 D warmup-only | RAC1P | primary | 0.012534 ± 0.002219 |
| beta=1 D warmup-only − beta=.1 D warmup-only | RAC1P | catchup_inclusive | 0.001028 ± 0.009418 |
| beta=1 C persistent − beta=.1 C persistent | Same residence | primary | -0.001413 ± 0.001607 |
| beta=1 C persistent − beta=.1 C persistent | Commute >20 min | primary | 0.000736 ± 0.004345 |
| beta=1 C persistent − beta=.1 C persistent | Income >$50k | primary | 0.001572 ± 0.006138 |
| beta=1 C persistent − beta=.1 C persistent | Civilian at work | primary | 0.000231 ± 0.003017 |
| beta=1 C persistent − beta=.1 C persistent | Public coverage | primary | 0.002162 ± 0.006547 |
| beta=1 C persistent − beta=.1 C persistent | SEX | primary | -0.003971 ± 0.001498 |
| beta=1 C persistent − beta=.1 C persistent | SEX | catchup_inclusive | -0.003971 ± 0.001498 |
| beta=1 C persistent − beta=.1 C persistent | RAC1P | primary | -0.005319 ± 0.003999 |
| beta=1 C persistent − beta=.1 C persistent | RAC1P | catchup_inclusive | -0.005829 ± 0.003118 |
| beta=1 D persistent − beta=.1 D persistent | Same residence | primary | -0.002313 ± 0.003317 |
| beta=1 D persistent − beta=.1 D persistent | Commute >20 min | primary | 0.000353 ± 0.004009 |
| beta=1 D persistent − beta=.1 D persistent | Income >$50k | primary | 0.001400 ± 0.006110 |
| beta=1 D persistent − beta=.1 D persistent | Civilian at work | primary | -0.000079 ± 0.002927 |
| beta=1 D persistent − beta=.1 D persistent | Public coverage | primary | 0.001531 ± 0.004470 |
| beta=1 D persistent − beta=.1 D persistent | SEX | primary | -0.004919 ± 0.001441 |
| beta=1 D persistent − beta=.1 D persistent | SEX | catchup_inclusive | -0.004919 ± 0.001441 |
| beta=1 D persistent − beta=.1 D persistent | RAC1P | primary | -0.006681 ± 0.004230 |
| beta=1 D persistent − beta=.1 D persistent | RAC1P | catchup_inclusive | -0.005932 ± 0.005527 |

## Unchanged original-PCA32 references

Each source loss permits +.01 nats. Residence must retain half the positive original-PCA32 advantage over the stronger unprotected rich bank on that seed and split. Nonpositive denominators are undefined. Attribute halving requires positive original-PCA32 gain and complete fitting, validation, evaluation and exposed-control coverage. These margins are descriptive references, not privacy budgets or statistical noninferiority tests. Known failures remain failures when race is undefined.

| Seed | Split | Release | Selector | Source all | Residence half | SEX half | RAC1P half | Joint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | validation | beta=.1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 0 | validation | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 0 | validation | beta=.1 D persistent | primary | pass | pass | fail | undefined | fail |
| 0 | validation | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | validation | beta=1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | validation | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 0 | validation | beta=1 D persistent | primary | pass | fail | fail | undefined | fail |
| 0 | validation | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation | beta=.1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | validation | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation | beta=.1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation | beta=1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation | beta=1 D persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | development_evaluation | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=.1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=.1 D persistent | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=.1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=.1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation_person_weighted | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | validation_person_weighted | beta=.1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=.1 C persistent | primary | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=.1 D persistent | primary | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | validation_person_weighted | beta=1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=1 C persistent | primary | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=1 D persistent | primary | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation_person_weighted | beta=.1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=.1 C persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=.1 D persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | validation_person_weighted | beta=1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=1 C persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | validation_person_weighted | beta=1 D persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=.1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=.1 D persistent | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=.1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=.1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 0 | development_evaluation_person_weighted | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | validation | beta=.1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | validation | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 1 | validation | beta=.1 D persistent | primary | pass | pass | fail | undefined | fail |
| 1 | validation | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | validation | beta=1 D warmup-only | primary | pass | pass | pass | undefined | undefined |
| 1 | validation | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 1 | validation | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 1 | validation | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | beta=.1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | beta=.1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | beta=1 D warmup-only | catchup_inclusive | pass | pass | pass | undefined | undefined |
| 1 | validation | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=.1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=.1 D persistent | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=.1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=.1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation_person_weighted | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | validation_person_weighted | beta=.1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | validation_person_weighted | beta=.1 C persistent | primary | pass | fail | fail | undefined | fail |
| 1 | validation_person_weighted | beta=.1 D persistent | primary | pass | fail | fail | undefined | fail |
| 1 | validation_person_weighted | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | validation_person_weighted | beta=1 D warmup-only | primary | pass | pass | pass | undefined | undefined |
| 1 | validation_person_weighted | beta=1 C persistent | primary | pass | fail | fail | undefined | fail |
| 1 | validation_person_weighted | beta=1 D persistent | primary | pass | fail | fail | undefined | fail |
| 1 | validation_person_weighted | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation_person_weighted | beta=.1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation_person_weighted | beta=.1 C persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | validation_person_weighted | beta=.1 D persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | validation_person_weighted | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | validation_person_weighted | beta=1 D warmup-only | catchup_inclusive | pass | pass | pass | undefined | undefined |
| 1 | validation_person_weighted | beta=1 C persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | validation_person_weighted | beta=1 D persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=.1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=.1 C persistent | primary | pass | fail | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=.1 D persistent | primary | pass | fail | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=.1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=.1 C persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=.1 D persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 1 | development_evaluation_person_weighted | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 2 | validation | beta=.1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 2 | validation | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 2 | validation | beta=.1 D persistent | primary | pass | pass | fail | undefined | fail |
| 2 | validation | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 2 | validation | beta=1 D warmup-only | primary | pass | pass | fail | undefined | fail |
| 2 | validation | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 2 | validation | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 2 | validation | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation | beta=.1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | validation | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation | beta=.1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation | beta=1 D warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | development_evaluation | beta=.1 C warmup-only | primary | pass | fail | fail | undefined | fail |
| 2 | development_evaluation | beta=.1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 2 | development_evaluation | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 2 | development_evaluation | beta=.1 D persistent | primary | pass | fail | fail | undefined | fail |
| 2 | development_evaluation | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 2 | development_evaluation | beta=1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 2 | development_evaluation | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 2 | development_evaluation | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 2 | development_evaluation | beta=.1 C warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development_evaluation | beta=.1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development_evaluation | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | development_evaluation | beta=.1 D persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development_evaluation | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | development_evaluation | beta=1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development_evaluation | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | development_evaluation | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=.1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=.1 D warmup-only | primary | pass | fail | pass | undefined | fail |
| 2 | validation_person_weighted | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=.1 D persistent | primary | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=1 D warmup-only | primary | pass | pass | pass | undefined | undefined |
| 2 | validation_person_weighted | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=.1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=.1 D warmup-only | catchup_inclusive | pass | fail | pass | undefined | fail |
| 2 | validation_person_weighted | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=.1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=1 D warmup-only | catchup_inclusive | pass | pass | pass | undefined | undefined |
| 2 | validation_person_weighted | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | validation_person_weighted | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=.1 C warmup-only | primary | pass | fail | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=.1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=.1 C persistent | primary | pass | pass | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=.1 D persistent | primary | pass | fail | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=1 C warmup-only | primary | pass | pass | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=1 D warmup-only | primary | pass | fail | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=1 C persistent | primary | pass | pass | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=1 D persistent | primary | pass | pass | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=.1 C warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=.1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=.1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=.1 D persistent | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=1 C warmup-only | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=1 D warmup-only | catchup_inclusive | pass | fail | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=1 C persistent | catchup_inclusive | pass | pass | fail | undefined | fail |
| 2 | development_evaluation_person_weighted | beta=1 D persistent | catchup_inclusive | pass | pass | fail | undefined | fail |

## Feature/bank comparisons and support

Every final arm is compared with all four unprotected/erased rich banks. Residence must be at least .01 nats lower and each signed attribute gain at most .005 nats higher. Raw numerical inequalities and coverage-gated results, including failures and undefined cases, are preserved in bank_comparisons.json.

All nine RAC1P codes remain in PER_CLASS.csv. Code 4 (Alaska Native alone) is absent from attacker fitting and validation in every seed, absent from development seeds 0/1, and has one development example in seed 2. Longer fitting cannot resolve the missing support. Original minimum-leaf-20 exposed controls also fail on supported code 5; every control candidate remains published.

[Original full pool and exposed-control evidence](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md). SUPPORT.csv records coverage for every new final selected attack and split; PER_CLASS.csv also retains the original controls and every new candidate. I and W have utility and label-free diagnostics only. Their teacher audit is not relabeled as a new snapshot audit.

## Saved, fresh, and catch-up audit accounting

CATCHUP.csv separates the zero-update saved training adversary, five-candidate fresh selection, direct-coordinate catch-up, and pooled validation selection. Catch-up inherits 20 warm plus 240 continuation representation-fitting passes, then resets Adam for the attacker pool. It is not an equal-lifetime-exposure comparison. FITTING.csv and CURVES.csv preserve exposure, fidelity and all saved validation curves. A validation winner may have worse development loss.

## Coordinate movement and recoverable teacher structure

Direct error is normalized squared distance to immutable raw PCA16 using saved representation-fitting scales. The independent affine decoder predicts standardized teacher coordinates; its intercept and coefficients use representation-fitting examples only. Source-validation/development errors, per-coordinate errors, rank, fixed rank tolerance and fitting-prior errors are retained in PRESERVATION_DIAGNOSTICS.csv and COORDINATE_ERRORS.csv. No outcome chooses a decoder or release.

Low affine error despite direct movement means teacher structure remains linearly recoverable. High affine error does not rule out nonlinear recovery. Neither diagnostic replaces utility heads or attribute audits, and retaining a leaky teacher is not a protection advance.

## Completion and audit extension

6/6 complete core units. Extended budget status: completed. See AUDIT_BUDGET.md for nested trajectory comparisons and runtime.json for measured phase time.
