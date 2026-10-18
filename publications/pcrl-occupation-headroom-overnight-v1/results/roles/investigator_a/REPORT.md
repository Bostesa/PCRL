# Investigator A report: stronger categorical boosted learning for the OCCUPATION output

**Saved by the coordinator.** Investigator A (a separate agent) returned this report as the text of its final response, as the
campaign's briefs require (the platform has refused subagent report files before); the coordinator saved it here unchanged in
substance.

Exploratory development on previously used records (public-use ACS 2019 1-Year PUMS, Florida). Task-only family: no fit read or
used SEX; MODEL_SELECT labels were read only through `python3 -m ohn.devrun score`.

## Bottom line

- Yes, conditionally. A plain HistGradientBoosting occupation learner with POBP as a native 106-level category (A0) is clearly
  WORSE than the teacher. Coding POBP as its ordinal place-of-birth code fixes it: the same learner (revision 1, `A1_PO_16`) beats
  T_TREE and TASK_REFIT on NLL, Brier and accuracy on both seeds, and improves most on the condition that binds every task-only
  reference, the men's guard: -0.00385 / -0.00229 (seed 0 / 1) against T_TREE's -0.0010 / +0.0006.
- Promising witnesses: `A1_PO_16` (first), `A1_PO_8` (second); both scored on both seeds by 06:01Z (before the 07:15Z cutoff), both
  RESERVE_MET over both seeds under the frozen rule (binding: H3's seed-1 income accuracy, ratio 2.862; minimum occupation-side ratio
  4.299 / 4.094).
- No revision 2 was fitted; the line is exhausted at 6 of 12 configurations.

## Exact model operation behind the improvement (A1_PO_16; A1_PO_8 identical except 8 leaves)

`p(x) = softmax(raw(x))`, decision = argmax, computed from X alone:
- model: scikit-learn 1.8.0 `HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.025, max_iter=T*, max_leaf_nodes=16,
  min_samples_leaf=100, l2_regularization=3.0, max_bins=255, early_stopping=False, categorical_features=[F,F,T,F,T,F],
  random_state=seed)`, fitted on all 20,000 MODEL_FIT people, occupation labels only; python3 (Homebrew 3.14.3);
- input coding `compact_po(X)` (deterministic from X): AGEP, WKHP as admitted z-scores; COW and MAR as native categories (level
  index); SCHL as its ordinal value (`__OTHER__` = 3); POBP as its numeric place-of-birth code (`__OTHER__` = 999). The code orders
  US states 001-056, territories 072 / 078, Europe 100-164, Asia 207-247, the Americas 301-373 and Africa 440 / 449, so trees can only
  split contiguous code ranges, i.e. pool related countries;
- tree count T* = 168 (seed 0) / 154 (seed 1): the first argmin of the pooled out-of-fold occupation NLL over 5 seed-specific
  household folds of MODEL_FIT (`household_folds(unit, 5, 'OHN_A_cv_S<seed>')`), then one refit on all MODEL_FIT (A1_PO_8: 279 / 240);
- no calibrator (the CALIBRATION temperature was a diagnostic only); no T_TREE / U0 / TASK_REFIT layer, so the cross-fitted check is
  complete.

## Every configuration (MODEL_SELECT, anchor H3; values vs U0, SE in parentheses)

Each fit used 36 HistGradientBoosting fits: n_model_fits = 6 (5 inner-CV models + 1 refit), n_crossfit_fits = 30 (5 registered
OHN1 pipelines x (5 inner + 1 refit)); plus 7 temperature-diagnostic fits. All 12 point clauses and both guards pass for every row.

| Config / seed | Capacity | Unit CPU-s | T* | Occ NLL | Occ Brier | Occ acc loss | Men's guard | Women's guard |
|---|---|---|---|---|---|---|---|---|
| A0_HGB_S / 0 | native cats, lr 0.05, 8 leaves, min leaf 50, l2 1 | 131.6 | 71 | -0.00349 (0.00251) | -0.00220 (0.00108) | -0.00212 (0.00322) | +0.00086 (0.00155) | -0.00533 (0.00151) |
| A0_HGB_L / 0 | native cats, lr 0.05, 31 leaves, min leaf 20, l2 1 | 127.2 | 40 | +0.00010 (0.00278) | -0.00096 (0.00118) | +0.00100 (0.00325) | +0.00251 (0.00170) | -0.00452 (0.00167) |
| A1_PO_8 / 0 | POBP code, lr 0.025, 8 leaves, min leaf 100, l2 3 | 111.6 | 279 | -0.01443 (0.00265) | -0.00628 (0.00121) | -0.00962 (0.00341) | -0.00344 (0.00170) | -0.00917 (0.00171) |
| A1_PO_16 / 0 | as above, 16 leaves | 125.9 | 168 | -0.01614 (0.00271) | -0.00673 (0.00124) | -0.00987 (0.00343) | -0.00385 (0.00174) | -0.00968 (0.00174) |
| A1_PO_8 / 1 | 8 leaves | 111.1 | 240 | -0.01118 (0.00249) | -0.00502 (0.00115) | -0.00975 (0.00342) | -0.00195 (0.00160) | -0.00816 (0.00162) |
| A1_PO_16 / 1 | 16 leaves | 128.7 | 154 | -0.01280 (0.00257) | -0.00546 (0.00118) | -0.00975 (0.00353) | -0.00229 (0.00165) | -0.00870 (0.00166) |

Occupation clauses vs Ucal:

| Config / seed | NLL | Brier | Accuracy loss |
|---|---|---|---|
| A0_HGB_S / 0 | -0.00261 (0.00237) | -0.00187 (0.00103) | -0.00212 |
| A0_HGB_L / 0 | +0.00098 (0.00263) | -0.00063 (0.00113) | +0.00100 |
| A1_PO_8 / 0 | -0.01355 (0.00252) | -0.00594 (0.00116) | -0.00962 |
| A1_PO_16 / 0 | -0.01526 (0.00258) | -0.00640 (0.00119) | -0.00987 |
| A1_PO_8 / 1 | -0.01114 (0.00247) | -0.00501 (0.00114) | -0.00975 |
| A1_PO_16 / 1 | -0.01276 (0.00254) | -0.00546 (0.00117) | -0.00975 |

Income clauses (the frozen H3 anchor, identical for all rows of a seed):

| Seed | Reference | Accuracy loss | NLL | Brier |
|---|---|---|---|---|
| 0 | U0 | +0.00012 (0.00264) | -0.00469 (0.00177) | -0.00279 (0.00132) |
| 0 | Ucal | +0.00012 | -0.00359 (0.00145) | -0.00234 (0.00118) |
| 1 | U0 | +0.00175 (0.00286) | -0.00515 (0.00181) | -0.00330 (0.00138) |
| 1 | Ucal | +0.00175 | -0.00488 (0.00159) | -0.00325 (0.00128) |

Paired differences, candidate minus reference (negative = candidate better):

| Config / seed | NLL vs T_TREE | Brier vs T_TREE | Acc loss vs T_TREE | NLL vs TASK_REFIT | Brier vs TASK_REFIT | Acc loss vs TASK_REFIT |
|---|---|---|---|---|---|---|
| A0_HGB_S / 0 | +0.00864 (0.00196) | +0.00292 (0.00086) | +0.00425 (0.00296) | +0.01030 (0.00201) | +0.00344 (0.00089) | +0.00225 |
| A0_HGB_L / 0 | +0.01223 (0.00220) | +0.00416 (0.00099) | +0.00737 (0.00315) | +0.01389 (0.00224) | +0.00468 (0.00100) | +0.00537 |
| A1_PO_8 / 0 | -0.00231 (0.00166) | -0.00115 (0.00074) | -0.00325 (0.00273) | -0.00064 (0.00166) | -0.00063 (0.00075) | -0.00525 (0.00270) |
| A1_PO_16 / 0 | -0.00402 (0.00171) | -0.00161 (0.00078) | -0.00350 (0.00274) | -0.00235 (0.00171) | -0.00109 (0.00079) | -0.00550 (0.00281) |
| A1_PO_8 / 1 | -0.00293 (0.00161) | -0.00142 (0.00074) | -0.00362 (0.00273) | -0.00086 (0.00164) | -0.00078 (0.00076) | -0.00625 (0.00273) |
| A1_PO_16 / 1 | -0.00456 (0.00168) | -0.00186 (0.00078) | -0.00362 (0.00269) | -0.00248 (0.00170) | -0.00123 (0.00080) | -0.00625 (0.00279) |

Registered references (seed 0 / 1, vs U0): T_TREE NLL -0.0121 / -0.0082, Brier -0.0051 / -0.0036, accuracy -0.0064 / -0.0061, men's
guard -0.0010 / +0.0006, women's -0.0093 / -0.0079; TASK_REFIT NLL -0.0138 / -0.0103, Brier -0.0056 / -0.0042, accuracy -0.0044 /
-0.0035, men's guard -0.0012 / -0.0004, women's -0.0102 / -0.0082.

Guards against T_TREE: the men's guard improves by 0.0028 / 0.0029 (A1_PO_16) and 0.0024 / 0.0025 (A1_PO_8) (differences of point
values, not paired; the scorer reports no paired per-SEX difference against T_TREE); the women's guard is at T_TREE's level.

Reserve: every per-seed condition is met for A0_HGB_S and all A1 rows; over both seeds A1_PO_8 and A1_PO_16 are RESERVE_MET;
A0_HGB_L fails the men's-guard reserve (ratio 1.37); A0_HGB_S's minimum ratio is 2.285 (men's guard).

Where A1_PO_16's gain comes from (occupation NLL contribution vs U0, compared with T_TREE): management (class 0) better by -0.0024 /
-0.0021, service (class 1) by -0.0019 / -0.0023, sales / office (class 2) slightly worse (+0.0007 both seeds); men's NLL vs U0
-0.0081 / -0.0047 (T_TREE -0.0034 / +0.0006); women's -0.0244 / -0.0211 (T_TREE -0.0211 / -0.0173).

## Cross-fitting results (MODEL_FIT out-of-fold; no in-sample layer)

| Config / seed | OOF NLL | OOF Brier | OOF accuracy | Deployed minus fold-average NLL on MODEL_SELECT | Decision agreement |
|---|---|---|---|---|---|
| A0_HGB_S / 0 | 1.22342 | 0.61822 | 0.49365 | -0.00061 (0.00054) | 0.971 |
| A0_HGB_L / 0 | 1.22649 | 0.61907 | 0.49245 | +0.00327 (0.00070) | 0.953 |
| A1_PO_8 / 0 | 1.21253 | 0.61449 | 0.4962 | +0.00021 (0.00050) | 0.969 |
| A1_PO_16 / 0 | 1.21231 | 0.61425 | 0.49805 | -0.00011 (0.00058) | 0.963 |
| A1_PO_8 / 1 | 1.21249 | 0.61448 | 0.4966 | -0.00000 (0.00042) | 0.974 |
| A1_PO_16 / 1 | 1.21215 | 0.61420 | 0.49785 | +0.00020 (0.00049) | 0.967 |

- Only A0_HGB_L shows a real variance gap (the fold-average better by 0.0033).
- Committee minus refit, cross-fitted NLL: A0 -0.00130 / -0.00306; A1_PO_8 -0.00097 / -0.00086; A1_PO_16 -0.00058 / -0.00045 (SEs
  about 0.0003).
- Calibration: CALIBRATION temperatures 0.950-0.964; cross-fitted NLL change -0.00015 to -0.00038 (SE about 0.0006): no calibration
  problem anywhere.
- The scorer's out-of-fold comparisons against U0 / T_TREE on MODEL_FIT favour those references (both in-sample there; +0.030 to
  +0.054 nats).

## What each step taught

1. A0 failure (scored 05:25Z): both settings worse than T_TREE on every clause, all five classes and both SEX groups. The internal
   out-of-fold curve had a sharp early minimum (T* 71 / 40 at lr 0.05, rising fast afterwards); smaller trees did better; the
   committee gained 4-6 SE: variance, not capacity or calibration; the card's replacement branch triggered.
2. Diagnostic D1-D5 (MODEL_FIT only, no release; 25 fits, seed-0 inner folds; pooled minima; A0_HGB_S = 1.22166): D1 (lr 0.025, min
   leaf 100, l2 3) 1.22167; D2 (D1 with 4 leaves) 1.22183; D3 (D1 with 16 leaves, min leaf 200) 1.22349; D4 (D1 with POBP as its
   ordinal code) 1.21073; D5 (admitted one-hot X) 1.22219 (548 CPU-s). The failure was the 106-level native-category POBP split: in
   every node scikit-learn picks the best binary partition of the levels ordered by gradient statistics, which fits noise in level
   means on 20,000 people. Leaf size, L2, learning rate and tree size do not touch it; one level per split (D5) cannot pool related
   countries.
3. Revision 1 (appended to the card and committed cb89a8b51 before its fit) confirmed the diagnosis on MODEL_SELECT (NLL 0.011-0.013
   better than A0_HGB_S; the pooled minimum reproduced D4 exactly).
4. Diagnostics D6-D10 on the new coding (MODEL_FIT only; 25 fits; reference A1_PO_16 = 1.21141): D6 32 leaves 1.21512; D7 lr 0.01
   1.21124; D8 min leaf 200, l2 10 1.21194; D9 COW as numeric code 1.21312; D10 COW and MAR as numeric codes 1.21233. Flat or worse;
   A1_PO_16's committee gain (0.0005-0.0006) is below the card's 0.001 trigger and invisible on MODEL_SELECT; a revision 2 would test
   differences below MODEL_SELECT's resolution (paired NLL SE about 0.0017), so the line stopped.

## Caveats and record-keeping

- The POBP coding was chosen after A0's MODEL_SELECT feedback, by MODEL_FIT-only diagnostics on all of MODEL_FIT, so the pipeline's
  cross-fitted check does not cover that choice (mild optimism: 10 settings); diagnostics recorded in the card and in
  `runs/A/DIAG/S0/*.json`.
- Ranking A1_PO_16 over A1_PO_8 comes from MODEL_SELECT and the difference (about 0.0017 nats) is within noise; against TASK_REFIT
  the A1_PO_16 NLL gain is about 1.4-1.5 SE per seed, not decisive on its own.
- Seed replication: HistGradientBoosting is deterministic, so the seeds differ only through the seed-specific inner folds (T*) and
  each seed's references and anchor (changed before registration so the two seeds would not give bitwise-identical fits).
- Code versions: the seed-0 A1 fits ran on the committed code (3ab1cd74); the seed-1 fits on 8be2baeb, which only adds diagnostic
  settings (fit path unchanged); the final code 57ce5130 (selftest V4 PASS, 30 checks) reproduces all six releases bitwise on all six
  roles, model SHA-256 checked against OCC.json.
- Failures: no queue unit failed (26 done, 0 failed); the scientific failures A0_HGB_S and A0_HGB_L (both worse than T_TREE; A0_HGB_L
  also fails the men's-guard reserve) are scored and stay in the ledger and frontier.
- Refusals met (tool guards; no data refusal; no label read attempted before the gate): "This session is isolated in the worktree
  <repository>/.claude/worktrees/pcrl-occupation-headroom-overnight-v1, but this command is too complex to verify that it
  stays inside the worktree. Refusing to run it — a worktree-isolated session's git operations must target its own worktree. Split it
  into plain, separate commands..." (a heredoc patch command; the Edit tool was used instead); the same refusal worded "...this command
  names git in a form too complex to verify..." twice (a background git-log wait loop; a chained git show / shasum command; split into
  simple commands); "Blocked: sleep 100 followed by ... To wait for a condition, use Monitor with an until-loop ... Do not chain shorter
  sleeps to work around this block." (Monitor and background tasks used instead).

## Resources

Queue: 26 units, 1,583.5 CPU-s (0.44 CPU-h): 6 fits, 6 scores, 4 diagnostic units, 6 verify units, 4 selftests; diagnostic fits
total 75 (the D5 one-hot diagnostic alone 548 CPU-s). Out-of-queue charge: 90 CPU-s (INVESTIGATOR_A_AGENT).

## Files

`ohn/fam_a.py` (commands `selftest`, `fit`, `verify`, `diag`; `predict(seed, config, X, names)`); `roles/investigator_a/CARD.md`;
private run directories `runs/A/<config>/S<s>/` (OCC.npz, OCC.json, CROSSFIT.npz, STATE.json, MODEL.pkl, FOLD_0-4.pkl, EVAL_MS_H3.json,
EVAL_MS_TW.json, VERIFY.json) and diagnostics `runs/A/DIAG/S0/`.

## Additional details from the investigator's second, consistent hand-back (06:14Z)

- Against the inherited JOINT_TREE output, A1_PO_16 occupation NLL -0.01867 / -0.01570 and Brier -0.00652 / -0.00595 (seed 0 / 1).
- Men's-guard reserve ratio of A1_PO_16: 5.10 / 4.41 (H3+T_TREE about 3.33 / 2.60).
- Every configuration's MODEL_SELECT NLL is about 0.015 lower than its MODEL_FIT out-of-fold NLL, for every configuration: a
  difference between the two populations, not overfitting.
- Fold pipelines' T*_k ranges: A0_HGB_S 58-64; A0_HGB_L 37-41; A1_PO_8 208-272 / 214-255; A1_PO_16 131-138 / 139-144 (seed 0 / 1).
- Diagnostic T* values: A0_HGB_S 71; D1 155; D2 273; D3 96; D4 279 (= A1_PO_8); D5 397; A1_PO_16 168; D6 103; D7 440; D8 168; D9 176;
  D10 169. D5's iteration cap was raised from 1,200 to 2,000 after the A0 MODEL_SELECT score was read and before any diagnostic fit
  (recorded in the code and card). Diagnostic units: OHN_A_DIAG_D1D4_S0_V1 (97.7 CPU-s), OHN_A_DIAG_D5_S0_V1 (547.9),
  OHN_A_DIAG_D6D8_S0_V1 (100.1), OHN_A_DIAG_D9D10_S0_V1 (43.6); 50 diagnostic HGB fits in total.
- Counts: 216 HGB fits in configurations (6 x 36) plus 50 diagnostic; 42 temperature fits; no reader fits. Queue CPU 1,583.5 s over
  26 units (fits 735.9, scores 7.9, diagnostics 789.3, selftests 38.8 (V1-V4 PASS: 22 / 25 / 29 / 30 checks), verifications 11.3).
- Release hashes (OCC.npz prefix): A1_PO_16 S0 5681620d893b, S1 2dee6e4bb26a; A1_PO_8 S0 12f23f839d90, S1 0f1cb9bc6349.
- Adaptive choices recorded by the investigator: the POBP coding chosen after A0's MODEL_SELECT feedback (MODEL_FIT-only diagnostics on
  the seed-0 inner folds); seed 1 run because seed 0 met the card's pre-declared trigger; the ranking of A1_PO_16 above A1_PO_8 uses
  MODEL_SELECT.
