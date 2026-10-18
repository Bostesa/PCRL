# Conditional income branch card: POBP-ordinal repair of the STRONG income learner (investigator A, INC role)

Exploratory development on previously used records (PCRL overnight occupation-headroom campaign); not confirmation. Public-use
U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64, civilian employed). The
service predicts an income class (2) and an occupation group (5) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted
145-column encoding `X`). Each recipient receives only its own float64 probability vector and argmax decision. SEX is the
dataset's recorded demographic column; readers are separate evaluation classifiers that estimate SEX from the released vectors.

Written 2026-10-18 06:15-06:25Z by investigator A (the occupation family-A investigator), before any fit of this branch and before
the stage-1 witness selection (07:15Z). Code: `ohn/fam_inc.py`. Releases: `runs/INC/<config>/S<s>/` (RELEASE.npz / RELEASE.json,
both tasks, all six roles). Queue units `OHN_INC_<step>_S<s>_V<n>`. No fit before the coordinator's explicit go, which follows the
frozen trigger evaluation in `WITNESS_SELECTION.json`.

**DEV-guided disclosure.** If the branch opens, it opens through leg B of the frozen trigger (`RULES.json` `income_branch`): the
source anchor's already published DEV_REPORT seed-1 income accuracy clause (point 0.00707 against the 0.010 cap, margin 0.83 x
max(SE_DEV)) fails the registered replication rule. The motivation therefore comes from a DEV_REPORT result of the source campaign.
No new DEV_REPORT read is involved; every fit uses MODEL_FIT / CALIBRATION labels only and every selection quantity is measured on
MODEL_SELECT through the registered scorer.

## 1. Measured failure addressed

- H3's income accuracy is its binding condition: MODEL_SELECT accuracy loss vs U0 +0.000125 / +0.001750 (seed 0 / 1), reserve ratio
  3.511 / 2.862 (the minimum of every H3-anchored release); published DEV_REPORT seed-1 +0.00707 (0.83 SE of margin).
- H3's income base (the source's STRONG learner) carries the same defect that family A measured for occupation. Its Step-1
  HistGradientBoosting learner codes POBP as a NATIVE 106-level category (`ihs.fam_h.CMP_CATEGORICAL`) and its household-CV curves
  have sharp early optima at learning rate 0.05 (chosen: 16 leaves / 94 iterations on seed 0, 8 leaves / 185 iterations on seed 1;
  summed Brier at 600 iterations 0.0026-0.0142 above each setting's optimum; `runs/H/STEP1/S<s>/STEP1.json`), and on CALIBRATION the HGB alone was
  0.0019 / 0.0028 summed Brier WORSE than raw T_TREE (the stack with T_TREE only tied it). For occupation, family A's MODEL_FIT-only
  diagnostic showed that this native-category POBP split alone cost 0.011 nats of held-out NLL (D1-D5), and replacing it with the
  ordinal place-of-birth code made the same booster beat T_TREE and TASK_REFIT on both seeds (`roles/investigator_a/CARD.md`).

## 2. Exact operation (two configurations per seed)

**Step 1 (task learning; MODEL_FIT income labels; no SEX), repaired.** Identical to the source's Step 1 (`ihs.fam_h.step1`) except
the POBP coding: the same 7-setting grid (learning rate 0.05, up to 600 trees, l2 1.0, max_bins 255; leaves 8 / 16 / 32 x min leaf
20 / 80 on the compact coding, plus 16 leaves / min leaf 20 on the admitted one-hot X), the same household 5-fold CV on MODEL_FIT
with the source's folds (`household_folds(unit, 5, 'IHS_H_step1_cv_S<seed>')`, identical function), the same criterion (first
minimum of the pooled out-of-fold summed Brier over settings and tree counts) and a refit on all MODEL_FIT; but the six compact
settings use `compact_po` (AGEP, WKHP as admitted; COW, MAR native categories; SCHL ordinal; **POBP = its numeric place-of-birth
code**, `__OTHER__` = 999; `ohn.fam_a.compact_po`). The one-hot setting is unchanged (it has no native category). Output: s_PO(x),
the repaired HGB income log-odds.

**`INC_TW_PO` (matched TASK-ONLY TWIN, kind `control`).** The source's HC_STRONG with the repaired base: the 3-coefficient
CALIBRATION logistic stack s_B = w0 + w1 s_PO + w2 s_T of s_PO and the frozen T_TREE income log-odds s_T, then the source's
2-parameter CALIBRATION Platt map (offset 0). Income released as `trees.probabilities('income', d)`, decision = argmax.

**`INC_H3_PO` (branch configuration, kind `income_branch`).** The source's H3 step applied UNCHANGED to the repaired base:
d = a * (s_B + D_L + beta * D_J) + b, with D_L / D_J the stored LOCAL_TREE / JOINT_TREE E20 income corrections on the T_TREE
routing, beta chosen by H3's registered precision-aware CALIBRATION budget rule (`ihs.fam_h.CONFIGS['H3_STRONG_LJDIR_PB']`: budget B1
= 0.0045 summed Brier / 0.0080 nats / 0.0035 expected accuracy, cost + 1 household SE for Brier and NLL, grid 0, 0.05, ..., 2.0,
largest value with every smaller one inside, household folds `IHS_H_budget_S<seed>` of CALIBRATION), then the Platt map (a, b)
refitted on CALIBRATION at that beta. No new sweep over beta, bins or mixture weights: the code calls the source's own functions
(`ihs.fam_h.budget_select`, `fit_maps`, `release_logodds`, `offset`, `BUDGETS`, `BETA_GRID`, `CONFIGS`) with labels loaded through
`ohn.common.load_role` only.

**Occupation output of both configurations**: the top SELECTED task-only witness of `WITNESS_SELECTION.json` (expected
`A/A1_PO_16`; set as a constant in `ohn/fam_inc.py` after the selection and before the first fit), recomputed from features by that
family's own `predict` and checked bitwise against its saved release on all six roles. Each configuration is a complete release
saved with `ohn.devrun.save_complete(family='INC', ...)` and scored with `python3 -m ohn.devrun score-complete` (budget class INC8).

## 3. Why it might fix seed-1 accuracy without spending Brier / NLL headroom

- A base learner without the overfitted POBP split should be a better income classifier near the 0.5 threshold: fewer decision
  errors from noise in country means, and better-ranked log-odds for the stack. In the occupation task the repair improved accuracy
  as well as NLL and Brier (accuracy loss vs T_TREE -0.0035 / -0.0036).
- The H3 step is unchanged, so its Brier / NLL spend is again governed by the same CALIBRATION rule; a stronger base starts further
  from the caps, and the source showed that a stronger base converts directly into reserve at a matched step (H3 minus its
  T_TREE-base control: Brier -0.0014 / -0.0011, accuracy +0.0014 / -0.0010).
- Risks recorded now: income accuracy differences are noisy (paired SE about 0.003 on MODEL_SELECT), so an accuracy repair may not be
  resolvable; the repaired base may also change beta (the expected-accuracy term bound H3's beta), and a larger beta spends more of
  the room. Neither is tuned: beta is whatever the unchanged rule returns.

## 4. Matched twin and controls

- Matched task-only twin: `INC_TW_PO` (same repaired base, no transplanted critic-trained correction).
- Fixed-H3 comparisons (no fit): the same witness occupation beside H3's income (`A/A1_PO_16` scored with anchor H3) and beside the
  source twin HC_STRONG (anchor TW); the branch minus H3 and the twin minus HC_STRONG isolate the base repair at a matched step.
- Learner control (internal, no release): the ORIGINAL-coding Step-1 learner recomputed in the same unit, on the same folds, and
  household-cross-fitted on MODEL_FIT exactly like the repaired one, so the HGB layer's held-out difference is attributable to the
  POBP coding alone.

## 5. Primary correctness checks

1. Reproduction: with the original coding, the Step-1 unit reproduces the source's STEP1 grid (selected setting, tree count,
   out-of-fold values) and the source's `HGB.pkl` predictions bitwise on all six roles; with the source HGB, the fit code reproduces
   the source's HC_STRONG and H3 states (stack, beta, Platt) and H3's released income on all six roles (<= 1e-12, identical
   decisions). This shows the H3 step is applied unchanged.
2. Household cross-fitting: for each registered crossfit fold (`ohn.common.crossfit_folds`, OHN1) the entire Step-1 procedure (grid,
   inner CV on the source folds restricted to the training rows, refit) is rerun without the fold and predicts it, for both codings;
   out-of-fold income Brier, NLL and accuracy of the HGB layer with paired household SEs. The stack, budget and Platt layers use
   T_TREE and the stored corrections, which are in-sample on MODEL_FIT, so the complete pipeline cannot be cross-fitted on MODEL_FIT;
   its CALIBRATION-side costs are household-cross-fitted by the budget rule itself (recorded).
3. Deployment: `predict(seed, config, X, names)` recomputes both outputs from features alone and `verify` checks them bitwise on all
   six roles; model files are hash-recorded.
4. Invented-data `selftest`: grid construction (the repaired grid differs from the source grid only in the compact settings' coding),
   model parameters identical to `ihs.fam_h.make_hgb` for the original coding, first-minimum selection, nested cross-fitting equal to
   a brute-force replay, release validity, and `predict` reproducing a saved invented release.

## 6. Plan, triggers and allotment

- After the go: one Step-1 unit per seed (both codings; about 400 HGB income fits per seed including the nested cross-fitting), then
  `INC_TW_PO` and `INC_H3_PO` on seed 0 and seed 1 (4 of the 8 INC configurations), scored with `score-complete`. Both seeds are run
  whatever seed 0 shows, because the motivating failure is the seed-1 accuracy clause.
- Report, not revise: the branch is one bounded repair. A revision is made only if a correctness check fails (a defect), never to
  chase a MODEL_SELECT result. If the repaired base is not better than the original in the cross-fitted MODEL_FIT check and on
  MODEL_SELECT, report "no income repair from the POBP coding"; if `INC_H3_PO` does not meet the reserve or the recipient gate, the
  occupation witness stays a component beside H3 and nothing is relaxed.

## Record (appended as it happens)

### Trigger and witness (06:28Z, before any fit of this branch)

- The frozen income-branch trigger FIRED at stage-1 close (`WITNESS_SELECTION.json`, commit c292353dc): leg B true (published
  DEV_REPORT seed-1 income accuracy of the anchor), leg A false. The branch is DEV-guided; every record says so.
- The top selected witness is `B/B2_SINGLE` (second `A/A1_PO_16`), so `WITNESS = ('B', 'B2_SINGLE')` in `ohn/fam_inc.py` (the only
  code change since commit 5c9c6cef5, plus one selftest check). Its occupation output is recomputed by `ohn.fam_b.predict` and must
  equal `runs/B/B2_SINGLE/S<s>/OCC.npz` bitwise on all six roles. Selftest `OHN_INC_SELFTEST_V2` PASS (19 invented-data checks,
  including the witness's own predict on invented features for both seeds).

### Step 1 (GO 06:29:30Z, commit 721a64c15) and a protocol deviation

- Deviation: units `OHN_INC_STEP1_S0_V1` (and, with the same code, `OHN_INC_STEP1_S1_V1`) raised `TypeError: type dict doesn't define
  __round__ method` in the final console summary print of `step1`, AFTER `STEP1.json` (write-once), `HGB_PO.pkl` and
  `CROSSFIT_INCOME.npz` had been written; the files' SHA-256 match their record. No rerun (it would be an equivalent refit and the
  record is write-once). The print was fixed at 06:41Z (`ohn/fam_inc.py`, summary line only; no computation changed) before any
  INC configuration fit; the queue records the Step-1 units as failed (return code 1) and that stays.
- Seed 0 reproduction of the source Step 1: selection equal (16 leaves, min leaf 20, 94 trees), every grid tree count equal, grid
  out-of-fold Brier difference 0.0, original-coding refit bitwise equal to the source `HGB.pkl` on all six roles.
- Seed 0 repaired selection: POBP-ordinal coding, 16 leaves, min leaf 20, 169 trees (main CV out-of-fold summed Brier 0.34656 vs the
  original 0.34914; NLL 0.51863 vs 0.52268; accuracy 0.7404 vs 0.73785).
- Seed 0 household cross-fitting of the HGB layer (OHN1; entire Step-1 procedure refitted per fold), repaired minus original:
  summed Brier -0.00288 (SE 0.00066), NLL -0.00422 (SE 0.00082), accuracy loss -0.00225 (SE 0.00163).
- Seed 0 CALIBRATION (calibrator side; all layers out of sample there), against raw T_TREE: raw repaired HGB Brier -0.00003 (SE
  0.00186; original +0.00191), cross-fitted repaired stack -0.00020 (SE 0.00111; original +0.00028); repaired minus original stack
  Brier -0.00048 (SE 0.00081), NLL -0.00091 (SE 0.00105), accuracy loss +0.00034 (SE 0.00262).
- Seed 1 (same deviation, same verification): reproduction exact (8 leaves, min leaf 80, 185 trees; tree counts equal; out-of-fold
  Brier difference 0.0; refit bitwise equal to the source `HGB.pkl` on all six roles). Repaired selection: POBP-ordinal coding,
  8 leaves, min leaf 80, 358 trees (main CV summed Brier 0.34657 vs 0.34797). Household cross-fitting, repaired minus original:
  Brier -0.00282 (SE 0.00067), NLL -0.00408 (SE 0.00082), accuracy loss -0.00125 (SE 0.00165). CALIBRATION against raw T_TREE:
  cross-fitted repaired stack Brier -0.00012 (SE 0.00087), NLL -0.00048, accuracy loss -0.00235 (original +0.00033 / +0.00017 /
  +0.00134); repaired minus original stack Brier -0.00046 (SE 0.00065), accuracy loss -0.00369 (SE 0.00230).
- Step-1 cost per seed: 67 main-CV and refit fits + 335 cross-fitting fits (HGB income) + 25 CALIBRATION logistic maps; 407.7 /
  409.1 CPU-s.

### Branch configurations, both seeds (fits 06:42-06:44Z, scores 06:43-06:46Z; witness occupation B/B2_SINGLE)

Reproduction check inside every fit: the same code on the source HGB reproduced the source HC_STRONG / H3 states exactly (beta
equal; stack and Platt coefficient differences 0.0) and the source's released income bitwise on all six roles (max |diff| 0.0,
identical decisions). The witness occupation recomputed by `ohn.fam_b.predict` equalled its saved release bitwise on all six roles.

| Config / seed | beta | stack (w0, w_HGB, w_T) | income acc loss vs U0 | income NLL vs U0 | income Brier vs U0 | income vs Ucal NLL / Brier |
|---|---|---|---|---|---|---|
| INC_TW_PO / 0 | - | 0.043, 0.537, 0.440 | -0.00375 (0.00261) | -0.00963 (0.00183) | -0.00653 (0.00137) | -0.00853 / -0.00609 |
| HC_STRONG + B2_SINGLE / 0 (fixed twin) | - | 0.043, 0.384, 0.615 | -0.00287 (0.00253) | -0.00836 (0.00174) | -0.00572 (0.00129) | -0.00726 / -0.00527 |
| INC_H3_PO / 0 | 0.25 | 0.043, 0.537, 0.440 | +0.00000 (0.00271) | -0.00586 (0.00187) | -0.00354 (0.00141) | -0.00476 / -0.00309 |
| H3 + B2_SINGLE / 0 (fixed anchor) | 0.25 | 0.043, 0.384, 0.615 | +0.00012 (0.00264) | -0.00469 (0.00177) | -0.00279 (0.00132) | -0.00359 / -0.00234 |
| INC_TW_PO / 1 | - | 0.047, 0.410, 0.567 | +0.00075 (0.00294) | -0.00925 (0.00184) | -0.00648 (0.00143) | -0.00897 / -0.00643 |
| HC_STRONG + B2_SINGLE / 1 | - | 0.042, 0.304, 0.688 | +0.00050 (0.00286) | -0.00835 (0.00179) | -0.00583 (0.00138) | -0.00808 / -0.00578 |
| INC_H3_PO / 1 | 0.20 | 0.047, 0.410, 0.567 | +0.00175 (0.00290) | -0.00593 (0.00188) | -0.00389 (0.00145) | -0.00566 / -0.00384 |
| H3 + B2_SINGLE / 1 | 0.20 | 0.042, 0.304, 0.688 | +0.00175 (0.00286) | -0.00515 (0.00181) | -0.00330 (0.00138) | -0.00488 / -0.00325 |

- Occupation clauses and guards are the witness's own (identical in all rows of a seed). All 12 point clauses and both guards pass.
- Whole-release reserve (two seeds, frozen rule): INC_H3_PO RESERVE_MET, min ratio 2.840 (binding s1 income accuracy; H3 + B2_SINGLE
  2.862); INC_TW_PO RESERVE_MET, 3.144 (HC_STRONG + B2_SINGLE 3.296); occupation-side minimum ratio 4.418 for all.
- beta: the unchanged rule returned the source's own values (0.25 / 0.20); the expected-accuracy term bound again (seed 0: 0.00334 at
  0.25, 0.00391 at 0.30; seed 1: 0.00339 at 0.20, 0.00414 at 0.25). CALIBRATION cost of the transplant vs the repaired task-only base:
  Brier +0.00055 / +0.00030, NLL +0.00094 / +0.00086, realized accuracy +0.0034 / +0.0054 (reported, not gated).
- Label-free: the branch changes 1.8-2.3% of income decisions relative to H3 on every role (mean |p difference| 0.013-0.016).

### Conclusion (06:50Z)

- The POBP-ordinal repair is a real income TASK-QUALITY gain: household cross-fitted HGB layer Brier -0.0029 / -0.0028 (4.3 / 4.2
  SE) and NLL -0.0042 / -0.0041 against the original coding; on CALIBRATION the repaired HGB alone is -0.00003 / +0.00108 summed
  Brier against raw T_TREE (original +0.00191 / +0.00279, SEs about 0.0018); on MODEL_SELECT the branch keeps more Brier / NLL room than H3 at the same beta (Brier -0.0008 /
  -0.0006, NLL -0.0012 / -0.0008, point differences of unpaired clause values).
- It does NOT repair the motivating clause: MODEL_SELECT seed-1 income accuracy loss is +0.00175 for both INC_H3_PO and H3 (seed 0:
  +0.00000 vs +0.00012); the reserve ratio is 2.840 vs 2.862. The repaired twin is also not more accurate on seed 1 (+0.00075 vs
  +0.00050). The extra probability quality goes to Brier / NLL, not to decisions near the threshold.
- No revision (card section 6: a revision only for a defect). 4 of 8 INC configurations used.
- Deployment: `python3 -m ohn.fam_inc verify` recomputes all four releases bitwise, both tasks, all six roles, from features alone
  (units `OHN_INC_<config>_S<s>_VERIFY_V1`, PASS).
- Development reader grid (requested by the coordinator at 06:47Z): bank `INC_BANK` (spec `roles/investigator_inc/specs/INC_BANK.json`:
  candidates the two INC releases; comparators the six required references + the matched component B/B2_SINGLE@H3), units
  `OHN_INC_GRID_S<s>_V1`, summary with REF_BANK and WITNESS_BANK.

### Development reader grid (06:48-06:53Z; units OHN_INC_GRID_S0_V1 70.4 / S1_V1 68.7 CPU-s, 256 reader fits and 92 cache hits each; summary OHN_INC_GRID_SUMMARY_V1)

Merged 8 cells (both seeds x 2 repetitions x 2 reader seeds), SCORE AUC differences; recipient = candidate minus comparator (gate:
every cell <= +0.010); pair benefit = comparator minus candidate (positive = candidate reads lower).

| Candidate | Comparator | income view mean / max (cells > 0.010) | income label | occupation view mean | pair benefit mean [min, max] |
|---|---|---|---|---|---|
| INC_H3_PO | B/B2_SINGLE@H3 (matched component: H3 income, same occupation) | +0.0177 / +0.0549 (4) | RECIPIENT_POINT_FAILED | 0 (same output) | -0.0010 [-0.0136, +0.0141] |
| INC_H3_PO | SUCC | +0.0177 / +0.0549 (4) | RECIPIENT_POINT_FAILED | +0.0729 | -0.0411 [-0.0627, -0.0245] |
| INC_H3_PO | H_LJ | +0.0069 / +0.0300 (4) | RECIPIENT_POINT_FAILED | +0.0729 | -0.0236 [-0.0430, -0.0140] |
| INC_H3_PO | RETIRED_B3CTL | +0.0038 / +0.0207 (3) | RECIPIENT_POINT_FAILED | +0.0729 | -0.0480 [-0.0636, -0.0377] |
| INC_H3_PO | NEURAL_RAW | -0.0072 / +0.0061 (0) | RECIPIENT_EDGE | +0.0212 | -0.0252 [-0.0526, -0.0065] |
| INC_H3_PO | B2TWIN | -0.0097 / -0.0028 (0) | RECIPIENT_EDGE | +0.0729 | -0.0351 [-0.0460, -0.0219] |
| INC_H3_PO | TWJ | -0.0163 / +0.0020 (0) | RECIPIENT_EDGE | +0.0729 | -0.0112 [-0.0251, -0.0018] |
| INC_TW_PO | B/B2_SINGLE@H3 / SUCC | +0.0349 / +0.0667 (6) | RECIPIENT_POINT_FAILED | 0 / +0.0729 | -0.0009 / -0.0409 |
| INC_TW_PO | H_LJ, RETIRED_B3CTL, NEURAL_RAW, B2TWIN, TWJ | +0.0241, +0.0210, +0.0100, +0.0075, +0.0009 (means) | RECIPIENT_POINT_FAILED (all) | +0.0729 / +0.0212 | -0.0235, -0.0479, -0.0250, -0.0349, -0.0111 |

- The repaired income output is MORE readable for SEX than H3's income at the same beta (income view +0.0177 mean, +0.055 max
  against the same occupation beside H3's income); the task-only twin more so (+0.0349). A stronger income task model carries more
  of the SEX signal and the unchanged transplant step does not offset it.
- The occupation-view failures and the negative pair benefits against the six required comparators belong to the task-only witness
  occupation (B/B2_SINGLE@H3 has the same occupation-view values and pair means -0.010 to -0.047); against its matched component
  the branch changes the pair by -0.0010 (NOT_POSITIVE).

### Branch outcome (06:55Z)

The income branch FAILS: (1) it does not repair the motivating clause (seed-1 MODEL_SELECT income accuracy loss +0.00175, equal to
H3; reserve ratio 2.840 vs 2.862); (2) the new income view fails the frozen recipient gate against SUCC, H_LJ, RETIRED_B3CTL and the
matched component; (3) no pair benefit. What it does show is a genuine income task-quality gain from the POBP-ordinal repair
(household cross-fitted Brier -0.0029 / -0.0028, NLL -0.0042 / -0.0041; MODEL_SELECT Brier / NLL room +0.0006 to +0.0012 over
H3), which here converts into readability, not into decisions or protection. As pre-declared, the occupation witness stays a
component beside H3; nothing is relaxed; no revision. 4 of 8 INC configurations used.
