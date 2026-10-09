# Utility references and role audit

Status: development analysis on previously reused Adult people, 2026-10-09. This
audit covers reference identity and the full-input task gate. It contains no
privacy result and no independent confirmation claim.

## Frozen references and earlier exposure

The frozen U0 task-only release came from the source 83-column, two-purpose
encoder with a 20-epoch warm start and 40-epoch task-only continuation. Its
encoder and deployed logistic heads were fitted on all 15,434 OSF_DEFENSE_FIT
rows; the historical head C choice used HEAD_VALIDATION labels. Ucal applies
one global inverse temperature per task and seed to U0. Its temperatures were
fitted on 2,000 CALIBRATION_HELDOUT rows, while the calibration *family* was
chosen historically using INNER_SELECTION outcomes. This study did not reopen
HEAD_VALIDATION or old assessment person arrays. The source is pinned to
`research/pcrl-heldout-calibration-v1` tip
`1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5`; the exact admitted model,
head and calibrator SHA-256 values are bound by the role and parity receipts.

The previous focused study at
`9e6bfea62e82297d024fd9588a5f4bfb30e0a9de` fitted its validation-stage
full-input F heads on only 11,574 FIT rows in seed 0 and 11,575 in seed 1. It
then scored the excluded 3,860 or 3,859 FIT validation rows, which U0 had
seen during its own fitting. The old F versus U0 gap therefore combines
unequal task-label exposure with learner, head and fit-procedure differences.
The predecessor's `full_f_packages` are empty because its utility screen
stopped before a final F refit. Its immutable validation F packages remain
available for a separately sealed, same-INNER-role observational diagnostic;
that diagnostic does not repair the predecessor's original comparison.

## Data roles and permitted source path

The new study uses role-specific private exports containing only admitted
members. OSF_DEFENSE_FIT has 15,434 rows in 15,428 groups;
CALIBRATION_HELDOUT has 2,000 rows in 2,000 groups; INNER_SELECTION has 2,235
rows in 2,234 groups. The R3 role-identity audit found zero direct row and
group overlap for every FIT/CAL/INNER pair. All two-seed evaluations use the
same people and group identities. The 83-column predictor schema excludes
SEX, task labels and other barred predictor fields. Reference parity read
only role-specific feature and frozen-reference vectors. The new full-input
screen read only FIT features and task labels; calibration was accessed only
after recipe selection. The common-role task gate opened INNER task labels
only after both full-FIT F packages had been frozen in the F artifact lock.
The old mixed-role release archive was excluded from reference restoration.

R1 and the independent R3 implementation restored U0 and Ucal on FIT, CAL
and INNER, for two tasks and seeds 0 and 1. All 48 comparisons were bitwise
equal to the admitted vectors, with maximum absolute difference zero. The
predeclared tolerance was 2e-7 because Torch forward batch geometry can
change low-order float32 results; the observed result was stronger than that
engineering threshold. This checks the frozen reference implementation and
does not create new evidence for U0 or Ucal's privacy.

## FIT-only recipe choice and selected full-input gate

The predeclared recipe screen compared A, the inherited 83-to-64-to-64 task
MLP, with B, a fresh source-style 83-to-64-to-64-to-16 task-only encoder plus
held-out FIT-selected logistic heads. Both were fitted and scored only on
fixed group-disjoint FIT folds; the score was the sum of *uncalibrated* outer
FIT validation NLL for income and occupation. A scored 3.1730484402 across
two paired seeds, versus 3.2056191051 for B, and won by the frozen rule.
Both recipes exceeded their FIT-majority constant-task accuracy by at least
0.010 on both tasks and seeds. This comparison selects a learner recipe; it
does not rank privacy mechanisms.

The selected A heads were independently refitted on all 15,434 FIT rows for
each seed/task, with their algorithmic FIT-internal epoch selection, then
given a fixed global-temperature calibration on the 2,000 CAL rows. Their
model, calibrator and task-package hashes were frozen before INNER scoring.
On INNER, selected F met every *point-estimate* accuracy-drop (0.010), mean
NLL-excess (0.010 nats) and summed multiclass Brier-excess (0.005) cap against
both U0 and Ucal in both tasks and seeds. R3 restored the selected F packages
bitwise, independently recomputed metrics and class recall, and matched the
R1 point costs to at most 6.14e-17. The tightest point margin was occupation
Brier against U0 in seed 1: excess 0.0038394, leaving 0.0011606 below the
cap. Income NLL improved by 0.0211500 versus U0 in seed 1. The complete
eight-row task/reference table, including class counts and recalls, is in
`FULL_INPUT_UTILITY.csv`.

This is a **point utility gate** for conditional method testing. The separately
sealed, 2,048-draw paired-group bootstrap assessed all 24 F-versus-U0/Ucal
accuracy, NLL and Brier costs together: 10 clauses PASS, 14 are UNRESOLVED,
and none shows a VIOLATION. The complete conjunction is therefore
**precision unresolved** despite point feasibility. For example, seed 1
occupation Brier excess versus U0 was 0.0038394, with a simultaneous upper
bound of 0.0099956 against the 0.005 cap. Seed 1 income NLL excess versus U0
was -0.0211500, with upper bound -0.0011534 against the 0.010 cap. R3
independently recomputed the entire group-resampling family and all 24
statuses; its maximum interval-value difference from R1 was 1.33e-15.
The public 24-row table is `FULL_INPUT_UTILITY_INTERVALS.csv`, with hash and
replay summary in `r1/UTILITY_INTERVAL_VERIFICATION.json` and independent
verification in `r3/UTILITY_INTERVAL_NUMERICAL_VERIFICATION.json`.

These one-sided simultaneous bounds are descriptive and conditional on the
selected fitted models. They do not account for portfolio selection or turn
reused Adult and INNER outcomes into independent confirmation. No mask, SEX
reader or coalition result follows from the full-input utility gate.

## Same-role historical F diagnostic

The prior validation-stage F checkpoints and calibrators for both tasks and
seeds have been inventoried and hash-verified without opening the old
assessment release. A separately sealed post-gate diagnostic scored those
unchanged heads on the same 2,235 INNER people as the new F. R3 independently
recomputed both tasks, seeds, reference metrics, class counts and recall from
the aligned frozen predictions. The old heads used 11,574/11,575 task-label
FIT rows by seed; the new heads used all 15,434. The old-versus-new contrast
is observational because fitting exposure and pipeline choices differ.

The tradeoff is mixed. New F minus old F on income NLL was -0.002242 in seed 0
and -0.004173 in seed 1, while accuracy moved -0.003132 and +0.002685.
Occupation NLL moved +0.000485 and -0.000232, with accuracy +0.002685 and
-0.004027. Neither head is uniformly better. The eight-row
`OLD_F_SAME_INNER_DIAGNOSTIC.csv` gives every task, seed and U0/Ucal margin;
`r1/OLD_F_SAME_INNER_VERIFICATION.json` binds the queues, seals, receipts and
R3 replay, and `r3/OLD_F_INNER_NUMERICAL_VERIFICATION.json` reports the
independent PASS. This diagnostic did not enter recipe choice or the
conditional mask gate. It supplies no causal repair or SEX-privacy result.

Evidence: `m1/DATA_ACCESS_MANIFEST.json`, `r3/ROLE_OVERLAP.json`,
`REFERENCE_PARITY.json`, `RECIPE_SELECTION.json`, `F_ARTIFACT_LOCK.json`,
`FULL_INPUT_GATE.json`, `r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json`, and
the pinned private queue receipts. `r1/INTERVAL_ACCESS_NOTE_20261009.md`
records a preseal hash read of two previously admitted INNER task archives and
the correction made before the interval worker ran. Public files contain
aggregates and hashes; person-level labels, predictions and checkpoints
remain private.
