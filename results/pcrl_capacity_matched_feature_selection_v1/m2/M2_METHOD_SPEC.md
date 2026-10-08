# M2 shared-bank joint selector and FIT worker

Development status: prefit implementation. The two seed Adult FIT units require
the effective M0 lock, M5 prefit review and the shared S3 queue. Synthetic
checks do not establish Adult utility or privacy.

## Frozen bank and pair geometry

M1 supplies the exact five masks satisfying four standalone numeric groups,
four of five categorical groups and at most 77 encoded columns. Each mask
deletes one categorical group. Both purposes receive the same complete bank;
the 25 ordered assignments are constructed before task or SEX fitting. A
diagonal assignment discloses eight raw source groups. Every distinct-mask
assignment discloses the full 83-column union. Raw selected values are part
of each recipient's wire; M4's restored deployment adapter includes both
heads, both calibrated probability vectors, decisions and public metadata.

## FIT-only development sequence, per paired model seed

1. After M0's live queue gate, open only OSF_DEFENSE_FIT features, income,
   occupation and seed-specific U0/Ucal references, plus CALIBRATION_HELDOUT
   features and task labels. Verify identical ordered row/group IDs within
   each role. The 83 predictor columns exclude task and SEX labels.
2. Split FIT groups once at seed + 26,000, with 25% validation groups. Move a
   smallest eligible validation group only if a task class is missing from
   the training part, breaking ties by group ID. Neither the split nor this
   repair reads SEX. Validation remains group-disjoint and nonempty.
3. Fit one 64×64 two-stage task head on the fitting part for each of five
   masks and each purpose, and two full-83 F heads. Fit one H-GLOBAL scalar
   per head on the common CAL role. Evaluate all 12 on the held-out FIT
   validation people. Compute U0, Ucal and F task accuracy, NLL and Brier on
   those same people. Apply the original inclusive caps: accuracy drop ≤.010,
   NLL excess ≤.010, Brier excess ≤.005 against all three references.
4. Publish a write-once, fsynced task-only table and M3 T selection receipt,
   binding all five mask IDs/SHAs, split and head IDs. Only after their bytes
   rehash successfully does the worker load FIT SEX. If no task-only mask is
   utility-feasible for either purpose, publish an all-invalid ten-rule
   control plan and a `NO_FEASIBLE_J` FIT receipt, then stop before SEX.
5. Require both SEX classes in the fixed fitting and validation parts. For
   each of five local raw mask views and six unique raw union views, fit
   LogisticRegression(C=1, max_iter=300, solver=lbfgs) after StandardScaler,
   and HistGradientBoosting(max_iter=64, max_leaf_nodes=15,
   learning_rate=.05, max_depth=None, random_state=seed). Each reader uses
   reversible validation AUC max(AUC,1−AUC); the view uses the maximum over
   the two readers. Coalition search cache keys are raw-union SHA only.
   These are fitting-stage search proxies, not complete-view final attacks.
6. Join the 25 full pair rows. J requires both recipients' point utility
   feasibility and local search AUC no more than T's corresponding local AUC
   +.010. Among eligible pairs, minimize coalition search AUC, then summed
   task validation NLL, then ordered mask SHAs. Keep aliases and failures.
   M3 applies matched T/I/S12/S21/shared and equal-union-size controls to
   this same immutable table. Publish the raw J selection and all-ten-rule
   M3 control plan as separate write-once private JSON files, hash-bound by
   the FIT receipt. Control-plan head IDs name validation heads; final wire
   head IDs come from the full FIT package registry and are bound separately.
7. Fit the twelve final task heads on all FIT people with the same two-stage
   architecture and CAL rule. M4 restores only these final packages for the
   actual ATTACK_FIT_NEW and INNER_SELECTION releases and fresh complete-view
   attacks. Clean teacher outputs are never appended.

The FIT worker opens no ATTACK_FIT_NEW or INNER_SELECTION row or label.
Its fixed output is `.private/m2/fit/seed_<seed>/FIT_RECEIPT.json`, with
adjacent `TASK_TABLE.json`, `TASK_ONLY_RECEIPT.json`, `PAIR_TABLE.json`,
`JOINT_SELECTION.json` and `CONTROL_PLAN.json` when eligible.
Checkpoint packages are private and hash-bound to the focused M1 data
manifest, the separately pinned prior role export files, lock, rows/groups,
mask, head spec, calibration and code. An interrupted checkpoint without a
matching immutable receipt fails closed. The per-seed planned work is 24
head packages, 48 neural training subfits, 24 scalar calibration fits and
22 raw search-reader fits. The role-sized synthetic timing receipt will set
the final queue cap before Adult fitting.
