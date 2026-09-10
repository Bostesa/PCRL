# Prior, exposed-target and category-support controls

No prior or exposed-target control was refitted. The original seven-target coalition control objects/records are hash-bound by REUSE_MANIFEST. Their selected120/360 records keep their original exposure and budgets. The original PCA32 contextual audit remains120epochs. Stationary H/E observers are new, distinct exposure controls and do not replace exposed targets.

| Seed | Target | Budget | Selected candidate | Development loss | Development complete class support |
| --- | --- | --- | --- | --- | --- |
| 0 | SEX | 120 | mlp_0 | 0.000026 | True |
| 0 | SEX | 360 | mlp_0 | 0.000002 | True |
| 0 | RAC1P | 120 | mlp_0 | 0.000143 | False |
| 0 | RAC1P | 360 | mlp_0 | 0.000009 | False |
| 0 | income_binary | 120 | mlp_1 | 0.000024 | True |
| 0 | income_binary | 360 | mlp_1 | 0.000002 | True |
| 0 | civilian_at_work | 120 | mlp_1 | 0.000059 | True |
| 0 | civilian_at_work | 360 | mlp_1 | 0.000004 | True |
| 0 | public_coverage | 120 | mlp_0 | 0.000059 | True |
| 0 | public_coverage | 360 | mlp_0 | 0.000004 | True |
| 0 | same_residence | 120 | mlp_1 | 0.000058 | True |
| 0 | same_residence | 360 | mlp_0 | 0.000004 | True |
| 0 | commute_over20 | 120 | mlp_0 | 0.000088 | True |
| 0 | commute_over20 | 360 | mlp_0 | 0.000008 | True |

[Original seed 0 full prior/exposed candidates](../redesign_20260908_acs_coalition_v1/seed_0/controls/metrics.json).

| 1 | SEX | 120 | mlp_0 | 0.000006 | True |
| 1 | SEX | 360 | mlp_0 | 0.000000 | True |
| 1 | RAC1P | 120 | mlp_0 | 0.000113 | False |
| 1 | RAC1P | 360 | mlp_0 | 0.000008 | False |
| 1 | income_binary | 120 | mlp_0 | 0.000055 | True |
| 1 | income_binary | 360 | mlp_0 | 0.000004 | True |
| 1 | civilian_at_work | 120 | mlp_0 | 0.000009 | True |
| 1 | civilian_at_work | 360 | mlp_0 | 0.000001 | True |
| 1 | public_coverage | 120 | mlp_1 | 0.000009 | True |
| 1 | public_coverage | 360 | mlp_1 | 0.000001 | True |
| 1 | same_residence | 120 | mlp_0 | 0.000043 | True |
| 1 | same_residence | 360 | mlp_0 | 0.000003 | True |
| 1 | commute_over20 | 120 | hist_gb_20 | 0.000142 | True |
| 1 | commute_over20 | 360 | mlp_0 | 0.000013 | True |

[Original seed 1 full prior/exposed candidates](../redesign_20260908_acs_coalition_v1/seed_1/controls/metrics.json).

| 2 | SEX | 120 | mlp_0 | 0.000022 | True |
| 2 | SEX | 360 | mlp_0 | 0.000002 | True |
| 2 | RAC1P | 120 | mlp_0 | 0.001747 | True |
| 2 | RAC1P | 360 | mlp_0 | 0.001857 | True |
| 2 | income_binary | 120 | mlp_0 | 0.000009 | True |
| 2 | income_binary | 360 | mlp_0 | 0.000001 | True |
| 2 | civilian_at_work | 120 | mlp_1 | 0.000034 | True |
| 2 | civilian_at_work | 360 | mlp_1 | 0.000002 | True |
| 2 | public_coverage | 120 | mlp_0 | 0.000059 | True |
| 2 | public_coverage | 360 | mlp_0 | 0.000004 | True |
| 2 | same_residence | 120 | mlp_0 | 0.000048 | True |
| 2 | same_residence | 360 | mlp_0 | 0.000003 | True |
| 2 | commute_over20 | 120 | mlp_1 | 0.000072 | True |
| 2 | commute_over20 | 360 | mlp_1 | 0.000007 | True |

[Original seed 2 full prior/exposed candidates](../redesign_20260908_acs_coalition_v1/seed_2/controls/metrics.json).

RAC1P code4 (class index3) is absent from the independent fitting and validation rows. Evaluation support alone cannot train or select that missing category. A selected exposed-target aggregate can look very good while the missing-category assessment fails. All nine class rows are retained in original controls and new candidate score records; no category was merged or removed. New raw-label fitting masks, present counts and hashes are in [EXPOSURE.csv](EXPOSURE.csv).
