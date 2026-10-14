# Role split and exposure

**Registered exposure statement.** This is exploratory development on repeatedly used Adult people. Source models already trained on all FIT, prior validation and reporting outcomes are exposed, and a new split does not create fresh confirmation.

## Inherited roles (counts verified by the queued role worker)

| Role | Rows | Groups |
| --- | --- | --- |
| FIT (OSF_DEFENSE_FIT) | 15,434 | 15,428 |
| ATT (ATTACK_FIT_NEW) | 4,064 | 4,061 (Stage B only) |
| INNER (INNER_SELECTION) | 2,235 | 2,234 (Stage B reporting lock only) |

FIT folds come from the inherited group hash (`IVR1|FIT-crossfold|20261010|`): fold 0 has 5,185 rows, fold 1 has 5,038, and fold 2 has 5,211. Because each fold is a hash of the group, no duplicate group crosses folds. The worker found 0 crossing groups.

## New continuation split (frozen before any role performance was measured)

- Eligible pool: FIT folds 0/1, 10,223 rows, 10,219 groups.
- Rule: GUARD is the first floor(20% × 10,219) = 2,043 groups in the order given by SHA256(`SPG1|UPDATE-GUARD-split|20261013|` + str(group)). UPDATE holds the remaining groups. The partition is label-independent, and both model seeds use the same one.
- Every copy of a duplicate group stays in one role. Group overlap between every pair of UPDATE, GUARD and VALIDATION is 0.

| Role | Rows | Groups | Occupation class support (0–5) | Use |
| --- | --- | --- | --- | --- |
| UPDATE | 8,180 | 8,176 | 1362 / 2102 / 1394 / 985 / 2335 / 2 | Task minibatches, privacy critics (UPDATE SEX), GLOBAL_UPDATE_GUARD pool |
| GUARD | 2,043 | 2,043 | 361 / 487 / 408 / 244 / 543 / 0 | Whole-pool task term in every arm; SEPARATE_GUARD constraints and acceptance |
| VALIDATION (FIT fold 2) | 5,211 | 5,209 | 879 / 1327 / 936 / 588 / 1479 / 2 | Opened once, after TRAINING_COMPLETE_LOCK |

UPDATE SEX support is 2,624 / 5,556. Class 5 is nearly absent from all FIT roles, as in the source. Membership was not tuned to any utility or privacy score.

## Physical exports and label provenance

The exports are separate private files:
- UPDATE: raw83, occupation, SEX, plus group/row ids.
- GUARD: raw83, occupation, plus group/row ids. GUARD has no SEX file and no income.
- VALIDATION: raw83, income, occupation, and U0/Ucal references for s0/s1.

To make these subsets, the export worker materialized the inherited FIT income, occupation and SEX arrays and the reference arrays. It computed no metric.

Trainer processes receive only UPDATE and GUARD files. The trainer refuses any source path containing a `VALIDATION` component, and its queued source list contains none. There are no in-training selection metrics.

GUARD is fitting data: it supplies task gradients, constraints and acceptance decisions, so it is not held out. VALIDATION is excluded from this continuation's gradients and acceptance decisions. Source training already saw it, so it serves as internal checkpoint selection, not fresh independent data.

No old assessment arrays, HEAD_VALIDATION, all-role loaders, confirmation cohorts or new data are opened. INNER truth stays blocked unless Stage B commits a reporting lock.
