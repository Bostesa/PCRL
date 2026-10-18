# File-based handoff to the manuscript and claims owners (no manuscript text was changed)

Source: research branch `research/pcrl-income-headroom-successor-v1` (this directory). Public-use U.S. Census Bureau ACS 2019
1-Year PUMS person records for Florida, development roles only; readers estimate the recorded SEX column from the released income
and occupation predictions. Exploratory development evidence on previously used records; not confirmation.

## Statements the record supports (with their qualifiers)

1. A successor income output (`H3_STRONG_LJDIR_PB`: a stronger gradient-boosting income learner plus a budgeted transplant of the
   lineage's stored critic-trained LOCAL / JOINT_TREE income corrections) kept income summed-Brier and NLL margins of 3.8-7.0
   household-bootstrap SEs on DEV_REPORT (the lower end uses the reserve rule's max(successor SE, H_LJ SE)), where the retired
   candidate's binding Brier clause had 0.03-0.04 SE. Qualifier: seed-1 income accuracy margin 0.83 SE; the registered reserve
   (claim R2) is "eroded", not replicated; R3 is INCOME_CONFIDENCE_UNRESOLVED (4 of 12).
2. Against the declared readers and the auxiliary budget only, its linked pair read lower than H_LJ's at the point on both
   seeds (+0.0141 / +0.0166 AUC; both benefit clauses pass at c1 as condition values of claim P, not separate conclusions)
   and lower than NEURAL_RAW's at the point (+0.0128 / +0.0123; the seed-1 clause bound is −0.0015, unresolved). Qualifiers:
   under the enhanced readers the seed-1 income recipient clause against H_LJ is unresolved; with matched household-bootstrap
   reader refits the seed-1 benefit against H_LJ was negative in 3 of 4 refits; benefits are about half the retired
   candidate's.
3. The registered competitive conjunction (claim P) is not met; under the enhanced readers it is also reader-sensitive (the seed-1
   income recipient clause against H_LJ is unresolved under the enhanced bank, resolved under the original bank).
4. Claim M, step 1 (registered wording): the declared readers recover SEX from the successor's linked pair less well than from
   its matched task-only twin's (+0.0285 / +0.0316); the twin differs only by the transplanted stored corrections; the label is
   TASK_ONLY_ATTRIBUTION_ONLY because step 2 (claim P) fails, and the twin itself reads higher than both references, so this is a
   mechanism contrast, not a competitiveness result. Neither the stronger learner's own contribution (claim C) nor any
   family-specific training step is established on DEV_REPORT.
5. The inherited JOINT_TREE occupation constraints still bind (men's within-SEX guard 0.86 / 0.55 SE; occupation NLL 0.40 SE).
   A SEX-balanced, information-preserving recalibration of that output raised the men's guard margin on MODEL_SELECT from 0.32 /
   0.62 to 1.83 / 1.62 SE (below its 2-SE target) without raising occupation recovery, but was not evaluated on DEV_REPORT.
6. Planning arithmetic: claim P's conjunction needs about 129,000 (D1) to 292,000 (D2) households at the plug-in, against about
   4.6 million / 60 million for the retired candidate and about 17,510 households available for a same-population test; the
   requirement is not identified by these data (binding margins 0.40-0.87 SE; 80% power is never reached at zero true slack).

## Statements the record does not support

- That the successor is competitive, superior to all eligible references, or beyond the 0.020 material-improvement target.
- That any mechanism removes SEX information, protects against arbitrary readers, or is novel.
- That seeds 0 and 1 replicate each other, or that DEV_REPORT intervals cover the adaptive search.
- That the occupation recalibration works on DEV_REPORT or makes the occupation constraints non-binding there.
- That a new population would confirm the release (another state or year would test transfer).
