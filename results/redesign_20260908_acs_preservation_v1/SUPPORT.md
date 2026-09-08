# Fixed attribute schemas and support limits

Recorded SEX retains Census codes 1–2; RAC1P retains all nine original codes. Every candidate is scored in the full declared schema. No missing class or undefined balanced-accuracy/macro-AUROC statistic is treated as protection.

RAC1P code 4 (Alaska Native alone) is absent from attacker fitting and attacker validation in all three seeds. It is absent from development seeds 0/1 and has one development observation in seed 2. More epochs cannot recover a class with no fitting examples. The corresponding complete race assessment remains unassessable.

Coverage requires positive fitting, validation and evaluation support for every class, plus positive class recall for the validation-selected exposed-label control. Original minimum-leaf-20 exposed trees also miss supported race code 5. These failures remain visible for every candidate, even when another exposed-control candidate succeeds.

- [Every new final selected-attack coverage flag](SUPPORT.csv).
- [Every primary-budget candidate/class score, including historical priors and exposed controls](PER_CLASS.csv).
- [Every extended-budget candidate/class score and extended exposed control](EXTENDED_PER_CLASS.csv).
- [Exact fixed household-pool supports](beta_0p1/seed_0/support.json), [seed 1](beta_0p1/seed_1/support.json), [seed 2](beta_0p1/seed_2/support.json).
- [Historical pool tables and original exposed-control failures](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md).

All original test fields are DEVELOPMENT EVALUATION. PWGTP sensitivity changes scoring on the same selected predictions; it does not fill a missing class. I/W have no independent attribute audits in this study and cannot inherit a coverage pass from their PCA16 teacher.
