# Focused shared-bank FIT screen

Both prespecified seeds completed the task-only FIT stage and returned
`NO_FEASIBLE_J`. This is a development no-go for the frozen five-mask bank
under the frozen U0, Ucal and full-input F mean-utility caps. It is not a
privacy result: the no-go branch stopped before the FIT SEX read, raw-feature
critics, pair-table construction, final head fits or complete-view attacks.

The same 15,434 FIT people underlie both seeds. The group-disjoint internal
validation sizes were 3,860 (seed 0) and 3,859 (seed 1). No old assessment,
HEAD_VALIDATION, ATTACK_FIT_NEW or INNER_SELECTION rows enter this summary.

## Every frozen mask

Each cell is the smallest of the accuracy, NLL and Brier **cap margins** for
that reference; the binding metric is shown. A negative margin fails its
clause. The three metric scales differ, so the minimum is a compact failure
indicator, not a cross-metric effect size. The complete aggregate margins
remain in the private, hash-bound task tables. Utility required every margin
against **each** of U0, Ucal and F to be nonnegative.

| Seed | Task | Deleted categorical atom | Min vs U0 | Min vs Ucal | Min vs F |
|---:|---|---|---:|---:|---:|
| 0 | Income | Native country | −0.035983 NLL | −0.029869 NLL | +0.005395 Brier |
| 0 | Income | Workclass | −0.039132 NLL | −0.033018 NLL | +0.002698 Brier |
| 0 | Income | Relationship | −0.039399 NLL | −0.033284 NLL | +0.003534 Brier |
| 0 | Income | Marital status | −0.037513 NLL | −0.031399 NLL | +0.005334 Brier |
| 0 | Income | Education fact | −0.071287 NLL | −0.065173 NLL | −0.024283 NLL |
| 0 | Occupation | Native country | −0.045541 NLL | −0.039472 NLL | +0.005739 Brier |
| 0 | Occupation | Workclass | −0.088794 NLL | −0.082726 NLL | −0.033960 NLL |
| 0 | Occupation | Relationship | −0.057981 NLL | −0.051913 NLL | −0.003147 NLL |
| 0 | Occupation | Marital status | −0.044394 NLL | −0.038326 NLL | +0.005403 Brier |
| 0 | Occupation | Education fact | −0.211525 NLL | −0.205457 NLL | −0.156691 NLL |
| 1 | Income | Native country | −0.031630 NLL | −0.023884 NLL | +0.007169 Brier |
| 1 | Income | Workclass | −0.038168 NLL | −0.030423 NLL | +0.002047 Brier |
| 1 | Income | Relationship | −0.034254 NLL | −0.026509 NLL | +0.006133 Brier |
| 1 | Income | Marital status | −0.035885 NLL | −0.028140 NLL | +0.005130 Brier |
| 1 | Income | Education fact | −0.063112 NLL | −0.055366 NLL | −0.018359 NLL |
| 1 | Occupation | Native country | −0.049529 NLL | −0.043356 NLL | +0.005781 Brier |
| 1 | Occupation | Workclass | −0.091722 NLL | −0.085549 NLL | −0.031416 NLL |
| 1 | Occupation | Relationship | −0.062685 NLL | −0.056513 NLL | −0.002380 NLL |
| 1 | Occupation | Marital status | −0.048718 NLL | −0.042546 NLL | +0.005943 Brier |
| 1 | Occupation | Education fact | −0.213912 NLL | −0.207739 NLL | −0.153606 NLL |

All 20 mask-task-seed rows fail U0 and Ucal, with NLL the largest negative
margin in each row. Dropping native country or marital status passes every F
clause for both tasks in both seeds, but still fails the required U0/Ucal
clauses. Thus the screen cannot select a utility-feasible task-only T or J.

## Full-input F reference gap

F uses all 83 permitted columns and the same new head/calibration procedure
as the masks. The figures below are reconstructed algebraically from the
aggregate task-table metrics and margins; no person-level predictions were
read. Each exceeds the permitted accuracy drop 0.010, NLL excess 0.010 nats,
and Brier excess 0.005 **even before feature deletion**.

| Seed | Task | F accuracy drop vs U0 | F NLL excess vs U0 | F Brier excess vs U0 |
|---:|---|---:|---:|---:|
| 0 | Income | 0.030052 | 0.047005 | 0.030091 |
| 0 | Occupation | 0.021762 | 0.054834 | 0.023905 |
| 1 | Income | 0.020731 | 0.044753 | 0.028629 |
| 1 | Occupation | 0.027987 | 0.060306 | 0.026350 |

The main common-criterion shortfall is therefore already present in the
new full-input head relative to U0, with additional mask cost for some
deletions, especially education fact. This screen cannot isolate a privacy
cost of selecting features from a training/reference gap. It also does not
show that the masks would fail against an equally trained F-only criterion:
native-country and marital-status deletions pass that narrower comparison.

## Work and custody

Each seed produced 12 validation head packages (five masks plus F for each
task). Under the fixed two-stage head rule that is 24 training stages and 12
held-out global-temperature calibrations per seed; 24 packages, 48 training
stages and 24 calibrations across the two seeds. The no-go branch produced
zero full-FIT final heads, zero raw SEX critic fits and zero protected pair
releases. Both S3 FIT units exited successfully; their scientific status is
`NO_FEASIBLE_J`, not a competitive method pass. The no-go control plan marks
all ten rule slots invalid because neither task has a feasible mask.

Private aggregate artifact SHA-256 values:

| Seed | TASK_TABLE.json | FIT_RECEIPT.json | CONTROL_PLAN.json |
|---:|---|---|---|
| 0 | `bdbb53c125c8bc4e49892fe54855f6907302d69db1d7386ec8a51805880ac942` | `be387533d68828ebc45d12132ba5f9c4d1a90092a68160d8af45d78434e70534` | `5aa3c14b2164d18b83f54d74f57b7782464b04dcfd5c982584598525b5d2f42d` |
| 1 | `0cd0ca9e5dd936a0d1225b7452e4ad3c31df5e2e853d8a1a6a74b992748709ae` | `0984398e414f9fe3c96e5b988483d67cbb2df56f2762e23b21e7828e807f7ccb` | `bbc8ef1883c57f348eb2c9f6d44abedb798eb82a246ac59ab9a5b1a425a09bd0` |

These are internal FIT validation results from previously exposed Adult
development data. Two model seeds are not independent populations; no
post-selection confirmation or population guarantee follows.
