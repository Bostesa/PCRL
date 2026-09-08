# Development-only application screen

No real-data representation pilot was admitted. These are fixed policy/output diagnostics, not protection-method results or final-test scores.

| Split | Subjects | Windows | Active/sedentary accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| task_fit | 13 | 4697 | 1.000000 | 1.000000 |
| attacker_fit | 4 | 1361 | 1.000000 | 1.000000 |
| development | 4 | 1294 | 0.999227 | 0.999293 |

| Actual or oracle release | Prior activity accuracy | Activity attack accuracy | Balanced accuracy | Log-loss improvement, nats |
|---|---:|---:|---:|---:|
| oracle_active_bit | 0.198609 | 0.363215 | 0.333333 | 0.685420 |
| learned_hard_active_output | 0.198609 | 0.363215 | 0.333333 | 0.681384 |
| learned_active_probability_20_bins | 0.198609 | 0.363215 | 0.333333 | 0.681398 |

One fixed binary logistic task model, training subjects only. Hard-output and20-bin probability attacks use separate participants; six-way activity targets are recorded source labels. No task/model/bin selection, no final-test evaluation, no PCRL training.

## Retrospective oracle-label associations

| Dataset | Exact task output | Attribute | Prior accuracy | Attack accuracy | Log-loss improvement, nats |
|---|---|---|---:|---:|---:|
| har_official_train | activity | subject | 0.058477 | 0.062557 | -0.008732 |
| har_official_train | activity | activity | 0.194016 | 1.000000 | 1.779940 |
| har_official_train | is_active | subject | 0.058477 | 0.061197 | 0.001536 |
| har_official_train | is_active | activity | 0.194016 | 0.357208 | 0.687253 |
| diabetes_processed_train | primary_diagnosis_category | race | 0.749068 | 0.749068 | 0.005105 |
| diabetes_processed_train | primary_diagnosis_category | gender | 0.522842 | 0.543620 | 0.003356 |
| diabetes_processed_train | primary_diagnosis_category | age_bucket | 0.250333 | 0.250533 | 0.048798 |
| diabetes_processed_train | readmission_outcome | race | 0.749068 | 0.749068 | 0.000270 |
| diabetes_processed_train | readmission_outcome | gender | 0.522842 | 0.522842 | -0.000013 |
| diabetes_processed_train | readmission_outcome | age_bucket | 0.250333 | 0.250333 | 0.001255 |
| diabetes_processed_train | medication_change_outcome | race | 0.749068 | 0.749068 | 0.000442 |
| diabetes_processed_train | medication_change_outcome | gender | 0.522842 | 0.522842 | 0.000216 |
| diabetes_processed_train | medication_change_outcome | age_bucket | 0.250333 | 0.250333 | 0.002384 |
| diabetes_processed_train | joint_three_label_bank | race | 0.749068 | 0.749068 | 0.004475 |
| diabetes_processed_train | joint_three_label_bank | gender | 0.522842 | 0.538159 | 0.003147 |
| diabetes_processed_train | joint_three_label_bank | age_bucket | 0.250333 | 0.250067 | 0.050159 |
