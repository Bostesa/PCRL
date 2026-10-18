# Family C card: residual task learning for the occupation output (investigator C)

Exploratory development on previously used records (PCRL overnight occupation-headroom campaign). Not confirmation. Public-use
U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64, civilian employed). The
service predicts an income class (2 classes) and an occupation group (5 official broad civilian groups) from six raw fields
(AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`). Each recipient receives only its own float64 probability
vector and argmax decision. SEX is the dataset's recorded demographic column (sex0 = SEX code 1, men; sex1 = SEX code 2, women).
Readers are separate evaluation classifiers that estimate SEX from the released probability vectors.

Written 2026-10-18 04:59-05:30Z, before any real-data fit of this family, before any label read and before the registration
commit. Code: `ohn/fam_c.py` (written after this card; invented-data self-test before any real-data fit). Unit ids
`OHN_C_<config>_S<s>_V<n>` (fit), `..._SCORE_V<n>` (development score), `OHN_C_SELFTEST_V<n>`. Family id for releases: `C`.
Task-only: no fit of this family reads SEX.

## Measured failure addressed

- **The frozen occupation teacher is budget-truncated.** T_TREE occupation (CatBoost depth-4 oblivious trees, lr 0.05, l2 3)
  chose its tree count as the argmin of a 15% household-disjoint internal validation curve capped at 800 trees, and the argmin
  sits AT the cap: 800 trees (seed 0) and 799 (seed 1) (`STAGE_A_SEED.json`, label-free metadata). The internal loss was still
  falling at the cap (seed 0: 1.2129 at tree 700, 1.2123 at 750, 1.21192 at 800; seed 1: 1.2381 / 1.2379 / 1.2378). The teacher
  was refitted on all MODEL_FIT at that count, so it is an under-converged booster, not a converged one.
- **An existing unconstrained residual learner already recovers part of it.** TASK_REFIT (replication Stage B, privacy terms
  disabled: AdamW on all 800 x 16 x 5 leaf values of the fixed teacher routing, 20 epochs x 40 batches of 256, lr 0.01, weight
  decay 0.01, no held-out stopping or strength check) improves MODEL_SELECT occupation NLL excess vs U0 from -0.0121 / -0.0083
  (T_TREE) to -0.0138 / -0.0103, Brier from -0.0051 / -0.0036 to -0.0056 / -0.0042 (`REFERENCE_MS_TABLE.csv`). It can only
  re-weight cells of the teacher's own partition (no new split, no new feature), has about 64,000 free parameters with implicit
  regularization only, and its strength was never checked on people the teacher did not see.
- **The binding resource is the men's Brier.** T_TREE's men's guard is -0.0010 / +0.0006 (no better than U0) while women's is
  -0.0093 / -0.0079; TASK_REFIT men -0.0012 / -0.0004. The inherited JOINT_TREE output sits at men +0.0044 / +0.0039 (0.32 / 0.62
  SE of margin). A privacy stage spending about 0.004-0.005 of Brier would need the task model's men's excess near -0.003 or
  lower; the task-only references are about 0.002-0.003 short of that.

## Exact operation and deployed output

Base (frozen, seed-specific): z_T(x) = the centred T_TREE occupation raw score (5-vector) from the exact numpy routing of the
admitted teacher (`ohn.common.teacher(seed, 'occupation')`, hash-checked), computed from `X` only. U0 is not used as a base: its
network weights are not admitted (only its stored outputs), so a U0-based correction could not be deployed from features, and
complementary-model stacking is family B's line.

**C1 (capacity-limited residual with a held-out strength).**

    g(x)  = sum_{m=1..M} sum_{c=1..5} eta * f_{m,c}(x) e_c,    f_{m,c} = depth-d regression trees on X (xgboost 3.2.0,
            objective multi:softprob, base_margin = z_T on the training people, so the trees fit the teacher's residual)
    p(x)  = softmax(z_T(x) + alpha * g(x)),   decision = argmax p(x)

- Fixed settings (all configurations): eta 0.05 and reg_lambda 3 (the teacher's learning rate and L2), min_child_weight 1,
  gamma 0, no row or column subsampling, tree_method hist with max_bin 256 (AGEP 40 and WKHP 87 distinct values: exact cut points),
  one thread, M_max = 800 rounds (one more full teacher budget).
- **Round count M**: household 5-fold CV on MODEL_FIT over the registered folds (`ohn.common.crossfit_folds`, tag OHN1_crossfit):
  train on 4 folds with base_margin z_T, record the held-out multinomial NLL (base_margin z_T) at every round; M = first argmin of
  the person-weighted mean held-out curve (1 <= M <= 800; M = 800 is flagged as at the cap). Then g is refitted on all MODEL_FIT
  with M rounds.
- **Strength alpha** (shrinkage of the whole correction): bounded scalar NLL minimization on CALIBRATION over [0, 2] (one
  calibrator fit; at-bound flagged). CALIBRATION people are out of sample for both z_T and g, so alpha measures how much of the
  MODEL_FIT-trained correction transfers to people the frozen teacher did not see. alpha = 0 is the teacher exactly.
- **Deployment**: the fitted trees are saved (xgboost JSON model, sha256) and converted to explicit arrays (split feature, float32
  threshold, children, float32 leaf value; sha256); `predict` evaluates them in numpy (float32 feature < float32 threshold -> yes
  child, float64 accumulation) and adds alpha * g to the float64 teacher margin. The numpy evaluator is checked against xgboost's
  own margin predictions (leaf ids identical, |difference| <= 1e-5) at fit time; the release is the numpy result.

**Capacity settings (the capacity limit is the tree depth: the function class of the correction).**

| Config | Depth d | Function class of g | Role |
|---|---|---|---|
| C1_RB_D1 | 1 | additive main effects (one input column per stump; boosted GAM residual) | capacity setting 1 |
| C1_RB_D2 | 2 | at most pairwise interactions (GA2M-like residual) | capacity setting 2 |

**Matched controls (fitted by this family and counted in its 12 configurations).**

| Config | Operation | What it isolates |
|---|---|---|
| CC_CONT_D4 | identical pipeline (same base margin, eta, lambda, M_max = 800, the same CV rule for M, the same CALIBRATION alpha) with depth 4 = the teacher's own depth (xgboost depthwise trees, at least as flexible as depth-4 oblivious trees): ordinary task continuation of the truncated booster with the same training budget and no capacity limit | the effect of the capacity limit (C1 minus CC_CONT_D4); the alpha = 1 continuation (no strength fit) is reported as a CALIBRATION diagnostic |
| CC_VSCAL | calibration only: vector scaling p = softmax(w * z_T + b), w in R^5 (positive), b in R^5 centred (9 free parameters), NLL on CALIBRATION (L-BFGS-B, start w = 1, b = 0); no MODEL_FIT label | whether the C1 gain is calibration of the frozen teacher; a one-temperature fit on CALIBRATION is recorded as a diagnostic (not released) |

Equally exposed frozen references (no fit; scored by the coordinator): T_TREE, TASK_REFIT (the existing residual learner), U0,
Ucal, and the inherited JOINT_TREE output. "What C1 adds beyond TASK_REFIT" is read from the scorer's paired MODEL_SELECT
comparison with TASK_REFIT and from the paired CALIBRATION comparison below.

## Why it might create occupation headroom

1. The teacher stopped at the cap with its validation loss still falling: continued fitting of its residual should lower
   held-out NLL and Brier for everyone (TASK_REFIT's gain is evidence that part of this residual is real and transfers).
2. A depth-limited correction can add structure outside the teacher's partition (new thresholds of AGEP / WKHP, input levels the
   oblivious trees rarely split on) that TASK_REFIT cannot represent, while having far fewer effective degrees of freedom than a
   depth-4 continuation, so less in-sample noise to learn from residuals computed against an in-sample base.
3. The CALIBRATION strength shrinks whatever part of the correction does not transfer (the in-sample-offset problem below), so
   the release should not be worse than the teacher on held-out people by more than the estimation error of one parameter.

Limits stated now: the ceiling is set by the six inputs; if the teacher and TASK_REFIT are already near it, no residual learner
can create headroom. A correction that fits in-sample teacher residuals can be biased (below). Nothing here targets the men's
Brier specifically (task-only), so a men's-guard gain can only come from a general task gain.

## Exposure of every component

| Component | Labels seen | Status on MODEL_FIT | Status on CALIBRATION / MODEL_SELECT / ATTACK_* / DEV_REPORT |
|---|---|---|---|
| 145-column encoding | none (MODEL_FIT inputs: moments, levels) | label-free | label-free |
| T_TREE occupation teacher z_T (frozen, seed-specific) | all 20,000 MODEL_FIT occupation labels | **in-sample for every person and every household fold**; cannot be refitted per fold (no CatBoost) | out of sample |
| correction g (new) | MODEL_FIT occupation labels | fitted against the in-sample z_T; out of fold only in the cross-fitting check | out of sample |
| round count M (new) | MODEL_FIT occupation labels via internal folds | internal folds use the in-sample z_T as offset on both sides | out of sample |
| alpha / vector scaling (new) | CALIBRATION occupation labels | not fitted there | in-sample on CALIBRATION (1 / 9 parameters); out of sample elsewhere |
| TASK_REFIT (reference; base only in a possible revision) | MODEL_FIT occupation labels | in-sample | out of sample |
| U0, Ucal (references in the scorer only) | MODEL_FIT (+ CALIBRATION temperature for Ucal) | in-sample | Ucal in-sample on CALIBRATION |
| H3 income anchor (frozen, scorer) | per `EXPOSURE_TABLE.md` | in-sample on MODEL_FIT and CALIBRATION | adaptively used on MODEL_SELECT by the source campaign |

**Consequence of the in-sample base (the central exposure issue).** On MODEL_FIT the teacher's margins fit the very labels the
correction is trained on, so the residuals y - softmax(z_T) are smaller and differently shaped than at deployment, where z_T is
out of sample (less accurate and possibly over-confident). A correction trained on MODEL_FIT therefore (a) cannot learn to temper
the teacher's out-of-sample over-confidence, (b) may be mis-scaled, and (c) may learn structure that only exists relative to the
in-sample margins. The CALIBRATION alpha corrects (b) only; CC_VSCAL measures (a); (c) is checked by comparing transfer on
MODEL_FIT out-of-fold households with CALIBRATION.

**The household cross-fitting check is NOT fully out of sample.** For each registered fold k the family refits every
label-using step of its own without fold k: the CV for M_k (4-fold CV over the other folds), the correction g_k (the first M_k
rounds of the model trained without fold k), and alpha_k (CALIBRATION, with g_k). 'MODEL_FIT__oof' = softmax(z_T + alpha_k g_k)
on fold k; 'MODEL_SELECT__foldavg' = the mean of the five fold pipelines' probability vectors. The frozen teacher z_T saw every
MODEL_FIT household, including fold k, so the out-of-fold predictions are out of fold for the correction layer and its selection
steps only; the base margin is in-sample. The absolute OOF utility is therefore optimistic (the teacher's in-sample fit), and the
OOF gain of the correction over z_T measures transfer to new households given an in-sample base, which is not the deployment
situation. A full cross-fitting claim would need the teacher refitted per fold (impossible here; a cross-fitted base learner,
e.g. a routing-fixed out-of-fold re-estimation of the teacher's leaf values, would still have an in-sample routing). For CC_VSCAL,
which uses no MODEL_FIT label, 'MODEL_FIT__oof' is the deployed MODEL_FIT output itself and the fold average equals the deployed
MODEL_SELECT output (recorded as such).

## Checks

**Primary correctness check (mechanism, held-out people only).** For each C1 setting and CC_CONT_D4, on CALIBRATION (people
neither the teacher nor g saw): the occupation NLL, summed Brier and accuracy of (i) z_T alone, (ii) z_T + g (alpha = 1, no
CALIBRATION fit), (iii) z_T + alpha g with alpha cross-fitted over 5 household folds of CALIBRATION (alpha fitted on 4 folds,
applied to the fifth), (iv) TASK_REFIT (stored, frozen), (v) the vector-scaled teacher (household-cross-fitted); paired
differences against z_T and against TASK_REFIT with household-clustered SEs. The mechanism works if (iii) is below (i) in NLL and
Brier and alpha* >= 0.5. Then MODEL_SELECT through the scorer only: the paired NLL / Brier / accuracy differences against T_TREE
and TASK_REFIT, the 6 occupation clauses, both guards, per-class and per-SEX decompositions, the cross-fitting diagnostics.

**Exposure diagnostic.** The teacher's in-sample MODEL_FIT NLL against its CALIBRATION NLL (size of the in-sample optimism);
the correction's out-of-fold gain over z_T on MODEL_FIT against its gain on CALIBRATION at alpha = 1.

**Code checks.** `python3 -m ohn.fam_c selftest` on invented data only (numpy tree evaluator = xgboost margins and leaf ids;
truncated-round prefix = shorter training; a planted residual is recovered and improves held-out NLL; alpha ~ 1 for a real
residual and ~ 0 for a pure-noise residual; alpha = 0 reproduces the base exactly; vector scaling recovers planted scales and
biases; the cross-fitting pipeline never trains on the fold it predicts; release contract; deterministic repeat; parameter hash
refusal). `verify` reproduces each saved release bitwise on all six roles from features with the saved, hash-recorded
parameters.

## Triggers (fixed now; seed 0 first)

Seed 0: C1_RB_D1, C1_RB_D2, CC_CONT_D4, CC_VSCAL (4 configurations), each fitted and scored with anchor H3.

- **T1, transfer (in-sample offset).** If a C1 setting has CALIBRATION alpha* < 0.5, or its CALIBRATION household-cross-fitted
  NLL gain over z_T is <= 0 while its MODEL_FIT out-of-fold gain is > 0: the correction does not transfer from the in-sample base.
  Revise once with the correction trained against a cross-fitted base (routing-fixed out-of-fold re-estimation of the teacher's
  leaf values, deployed on the frozen teacher), same capacity settings; abandon C1 if that also fails T1.
- **T2, capacity.** If CC_CONT_D4's MODEL_SELECT occupation NLL excess vs U0 is lower than the better C1 setting's by more than
  0.001 nats with Brier not worse: the capacity limit costs task quality; it is reported as the finding (no further capacity
  revisions toward depth 3 and up, which would be a disguised sweep).
- **T3, calibration explains.** If CC_VSCAL's MODEL_SELECT occupation NLL and Brier excess vs U0 are within 0.0005 nats and
  0.0003 of the better C1 setting or better: the gain is calibration; CC_VSCAL is reported as the finding and no residual-learning
  component is claimed.
- **T4, beyond TASK_REFIT.** If the better C1 setting does not lower MODEL_SELECT occupation NLL against TASK_REFIT by at least
  0.0005 nats (paired point estimate, scorer) with Brier not worse: C1 is at most a substitute for TASK_REFIT. Revise with the
  same capacity-limited correction (the better setting, or both if they tie within 0.0003) on top of the frozen TASK_REFIT margin
  (z_TR = centred teacher raw score plus the stored TASK_REFIT E20 occupation correction), same CV, alpha and cross-fitting rules,
  to test complementarity rather than substitution.
- **T5, subgroup.** A configuration whose MODEL_SELECT men's guard is worse than T_TREE's by more than 0.0005 (point) is not
  promoted, whatever its NLL (a stronger aggregate NLL cannot compensate for a worse subgroup).
- **Promotion to seed 1.** A design goes to seed 1 if on seed 0 it passes all 6 occupation clauses and both guards at the point,
  lowers NLL against T_TREE (point) and is not flagged by T5; its matched controls (CC_CONT_D4, CC_VSCAL) are run on seed 1 with
  it, budget permitting.
- **Abandon.** If, after at most two revisions, no configuration lowers seed-0 MODEL_SELECT occupation NLL against TASK_REFIT by
  at least 0.0005 nats with Brier not worse and T5 not flagged, the family stops with a negative: the residual signal the
  frozen teacher leaves is exhausted by TASK_REFIT at this data size, and no residual learner of this family creates occupation
  headroom beyond it.

Budget: at most 12 seed-specific configurations including the controls (initial 4 on seed 0, up to 4 on seed 1, at most two
revisions with at most two capacity settings each), about 1 CPU-hour including scoring.

## Registered expectation (written before any fit; bets, not claims)

- C1_RB_D2 lowers MODEL_SELECT occupation NLL against T_TREE by about 0.001-0.002 nats (similar to TASK_REFIT's 0.0017 /
  0.0020) and lands within +-0.0005 of TASK_REFIT; C1_RB_D1 gains less (about 0.0005-0.001).
- CALIBRATION alpha* between 0.6 and 1.2; CALIBRATION gains are within 1 household SE (2,982 people).
- CC_CONT_D4 is no better than C1_RB_D2 (more in-sample noise to fit), possibly worse on CALIBRATION at alpha = 1.
- CC_VSCAL changes NLL by less than 0.0005 against T_TREE and moves the men's guard by less than 0.0005.
- The men's guard moves by at most about 0.001 under any configuration: no configuration of this family is expected to create
  the about 0.003 of men's Brier headroom that a privacy stage would need. My prior is that T4 fires and the family ends near the
  abandon criterion.

## Closest known methods

- Boosting from an initial model / offset (stagewise additive modelling, Friedman 2001; xgboost base_margin; "warm-start"
  continued training of a boosted ensemble); residual learning on a frozen predictor.
- Depth-1 / depth-2 boosting as additive and pairwise-interaction models (GAM / GA2M, Lou et al. 2012-2013; Explainable Boosting
  Machines), here as a correction to a frozen margin.
- Re-estimating the leaf values of a fixed tree ensemble (post-processing / fully corrective refitting, e.g. Friedman & Popescu's
  ISLE post-processing): TASK_REFIT in this lineage.
- Temperature and vector scaling (Guo et al. 2017); stacking with cross-fitted base predictions (Wolpert 1992; Breiman 1996) for
  the exposure discussion.

No novelty or theorem is claimed.

## Revisions (appended with time, reason and result; earlier text is never rewritten)

- **05:06-05:09Z, code checks before any label read (no design change).**
  - `OHN_C_SELFTEST_V1` (invented data only, 23.5 CPU-s): 23 of 23 checks PASS: numpy tree evaluator equals xgboost's margins
    and leaf ids; xgboost training deterministic; a truncated round prefix equals the shorter training bitwise; depth limits
    respected; a planted residual is recovered (held-out NLL 1.4965 -> 1.4745, truth 1.4688; alpha 1.039, M 48); a pure-noise
    residual is shrunk away (M 1, alpha 4.5e-10, held-out NLL unchanged); alpha = 0 is the base exactly; permuting the fold-0
    labels leaves the fold-0 out-of-fold outputs bitwise unchanged (and changes the other folds'); vector scaling, temperature
    and alpha recover planted values; release contract, determinism, row-chunking invariance, clustered-SE sanity, hash
    tamper detection. Timing at the real size: about 100 / 125 / 190 CPU-s per configuration for depth 1 / 2 / 4.
  - `OHN_C_DRYRUN_S0_V1` (REAL features of all six roles, INVENTED labels, nothing released): both bases (T_TREE and
    T_TREE + stored TASK_REFIT correction) recomputed from features match the stored outputs on all six roles (max |dp| <=
    4.4e-16); numpy-vs-xgboost parity on real MODEL_SELECT features 9.0e-8 with identical leaf ids; tree arrays reload bitwise.
  - `OHN_C_DRYRUNFULL_CC_VSCAL_S0_V1`, `OHN_C_DRYRUNFULL_C1_RB_D1_S0_V1` (same invented labels, round cap 60, written only to a
    scratch directory outside the repository and the campaign runs): `fit` -> `save_occupation` -> `verify` end to end, bitwise
    on all six roles.

### Revision 1 (appended 2026-10-18 06:01Z, before its first fit): residual trained against cross-fitted teacher margins

**Seed-0 results of the initial design** (MODEL_SELECT through `ohn.devrun score`, anchor H3; ledger entries 49-56; household-
clustered paired SEs). Every residual configuration selected M = 1 round, with alpha at its upper bound 2 on the full model and in
every cross-fitting fold:

| Config | M | alpha | occ NLL excess vs U0 | NLL vs T_TREE (SE) | NLL vs TASK_REFIT (SE) | Brier vs TASK_REFIT (SE) | men's guard (SE) | women's guard (SE) |
|---|---|---|---|---|---|---|---|---|
| C1_RB_D1 | 1 | 2.000 | -0.012184 | -0.000057 (0.000018) | +0.001605 (0.000400) | +0.000499 (0.000184) | -0.001059 (0.001513) | -0.009322 (0.001536) |
| C1_RB_D2 | 1 | 2.000 | -0.012163 | -0.000036 (0.000024) | +0.001627 (0.000398) | +0.000502 (0.000183) | -0.001047 (0.001512) | -0.009329 (0.001536) |
| CC_CONT_D4 | 1 | 2.000 | -0.012226 | -0.000099 (0.000036) | +0.001564 (0.000397) | +0.000474 (0.000183) | -0.001073 (0.001511) | -0.009359 (0.001535) |
| CC_VSCAL | - | - | -0.011899 | +0.000228 (0.000789) | +0.001891 (0.000849) | +0.000573 (0.000396) | +0.001701 (0.001571) | -0.011995 (0.001566) |

**Diagnosis.**
1. The in-sample CV is blind. At every depth the pooled held-out-fold NLL (teacher margins in-sample on both sides of every split)
   rises from round 1 (D1 1.17257 -> 1.17833 at round 800; D2 1.17258 -> 1.18166; D4 1.17262 -> 1.19105) and is already above the
   uncorrected 1.17255. CALIBRATION asks for more of the same correction (alpha = 2, the bound, everywhere), and the tiny correction
   still lowers CALIBRATION NLL (household-cross-fitted alpha: D2 -0.000086, SE 0.00003; D4 -0.00016, SE 0.00006). A transferable
   out-of-sample residual exists; the in-sample criterion cannot see it.
2. The cause is measured (unit OHN_C_EXPOSURE_S0_V1, MODEL_FIT and CALIBRATION occupation labels, no SEX, nothing released): a
   routing-fixed re-estimation of the teacher's leaf values (Plain boosting, one full-Hessian multiclass Newton step per tree, l2 3,
   lr 0.05, unit weights: the CatBoost options recorded in the admitted model file) reproduces the frozen teacher EXACTLY (max
   |dp| 4.1e-9 on CALIBRATION, identical decisions). Re-estimated without each registered household fold, the teacher's
   out-of-fold MODEL_FIT NLL is 1.20678 against 1.17255 in-sample: the frozen layer's in-sample optimism on MODEL_FIT is 0.034
   nats, far larger than any residual signal (TASK_REFIT's MODEL_SELECT gain over T_TREE is 0.0017).
3. Trigger status. T4 fires (C1 worse than TASK_REFIT by 0.0016). T3 fires trivially (C1 has no gain for calibration to
   explain; CC_VSCAL is within 0.0005 / 0.0003) and T5 flags CC_VSCAL (vector scaling moves Brier from men to women: men's guard
   +0.0027 against T_TREE). T2 does not fire. T1's literal condition did NOT fire (alpha* = 2 >= 0.5; CALIBRATION gain > 0;
   MODEL_FIT out-of-fold gain <= 0): the in-sample offset showed up as the opposite symptom (a blind selection criterion and a
   strength pinned at its upper bound), which I read as the failure T1 was written for. Disclosed as such.
4. T4's listed revision (the same correction stacked on TASK_REFIT's margin) is not run: TASK_REFIT's margins are even more
   in-sample (MODEL_FIT NLL 1.16633 against CALIBRATION 1.18528) and its stored stochastic AdamW correction cannot be refitted per
   fold exactly, so its in-sample CV would collapse the same way. T1's remedy addresses the measured cause and is now exact.

**Revision 1: one change, the margins the correction is trained and selected on.**
- Training margins: each MODEL_FIT person's teacher margin re-estimated exactly (routing fixed) WITHOUT that person's household
  fold (registered folds): out-of-fold leaf values. M by 5-fold CV with validation margins re-estimated without the validation
  fold and training margins re-estimated without {validation fold, own fold}. Everything else unchanged (xgboost multi:softprob,
  eta 0.05, lambda 3, M_max 800, first argmin, numpy deployment).
- Deployed output unchanged in form: p = softmax(z_T + alpha * g) on the FROZEN teacher margin (equal to the full re-estimate
  within 4e-9), alpha by NLL on CALIBRATION in [0, 2]. Runtime inputs: X only.
- Household cross-fitting check, now out of fold for every label-using step except the teacher's split structure: fold k is
  predicted by z^(-k) + alpha_k g_k, z^(-k) the teacher re-estimated without fold k; M_k by a CV over the other folds (validation
  margins without {k, j}, training margins without {k, j, own fold}); g_k trained on margins without {k, own fold}; alpha_k on
  CALIBRATION with z^(-k). The split structure (CatBoost's choice of split features and borders, made on all MODEL_FIT) is NOT
  refitted, so the check is still not fully out of sample. 26 exact re-estimations per configuration.
- Capacity settings: C2_XF_D1 (depth 1) and C2_XF_D2 (depth 2), with the matched continuation control CC_XF_CONT_D4 (depth 4,
  same margins and rules). CC_VSCAL is unchanged (it uses no MODEL_FIT label). Seed 0 first: 3 configurations (7 of 12 used).
- Invented-data checks: OHN_C_SELFTEST_REBOOST_V1 4/4 PASS (centred leaf values, Newton step, out-of-fold independence);
  OHN_C_SELFTEST_XF_V1 7/7 PASS (permuting the fold-0 labels leaves the fold-0 out-of-fold outputs and the fold-0 pipeline's M_k
  and alpha_k bitwise unchanged and changes the other folds').
- Registered expectation: M well above 1 (tens to hundreds of rounds) and alpha in [0.5, 1.5], not at a bound; MODEL_SELECT
  occupation NLL better than T_TREE by 0.001-0.003 and within +-0.001 of TASK_REFIT; the out-of-fold MODEL_FIT gain of the
  correction comparable to its CALIBRATION gain; men's guard moved by less than 0.001.
- Triggers carried over with C2 in place of C1: T2 (CC_XF_CONT_D4 against the better C2), T3, T4 (complementarity is reported, not
  re-revised), T5, promotion and abandonment. Revision 2 only if a C2 setting fails the transfer test under the honest margins
  (CALIBRATION household-cross-fitted gain <= 0 while the out-of-fold MODEL_FIT gain > 0, or alpha* < 0.5).

### Revision 2 (appended 2026-10-18 06:11Z, before its first fit): ordered block codes as inputs of the correction

**Seed-0 results of revision 1** (MODEL_SELECT through the scorer, anchor H3):

| Config | M (fold M_k) | alpha | NLL vs T_TREE (SE) | Brier vs T_TREE (SE) | NLL vs TASK_REFIT (SE) | men's guard (SE) | women's guard (SE) |
|---|---|---|---|---|---|---|---|
| C2_XF_D1 | 7 (1, 1, 38, 1, 16) | 2.000 | -0.000343 (0.000104) | -0.000105 (0.000045) | +0.001319 (0.000405) | -0.001170 (0.001511) | -0.009380 (0.001533) |
| C2_XF_D2 | 1 (4, 1, 19, 1, 6) | 2.000 | -0.000027 (0.000026) | -0.000011 (0.000011) | +0.001636 (0.000398) | -0.001042 (0.001512) | -0.009321 (0.001535) |
| CC_XF_CONT_D4 | 1 (1, 1, 16, 1, 1) | 2.000 | -0.000061 (0.000039) | -0.000027 (0.000017) | +0.001601 (0.000396) | -0.001066 (0.001512) | -0.009329 (0.001535) |

Reading. Honest margins changed the selection (the in-sample twin C1_RB_D1 gained -0.000057; C2_XF_D1 -0.000343, 3.3 SE), but
the CV curves stay nearly flat (D1's minimum at round 7 is within 1e-5 of round 1; D2 and D4 rise from round 1) and alpha stays
at its upper bound. No transfer failure occurred (CALIBRATION gains >= 0). What new trees on the one-hot X can reach on top of
the frozen teacher is at most about a fifth of TASK_REFIT's gain. T2 does not fire; T4 fires (C2 worse than TASK_REFIT by
0.0013-0.0016). C2_XF_D1 meets the seed-1 promotion rule and runs on seed 1 now (no design change).

**Disclosed deviation.** Revision 1's text limited a revision 2 to a transfer failure under honest margins, which did not occur.
Revision 2 is instead motivated by a new fact relayed by the coordinator at about 06:02Z (after revision 1 was written): fresh
learners of families A and B that code POBP as an ordered value reach occupation NLL -0.0128 to -0.0167 against U0 (0.003-0.005
nats beyond the teacher) with men's guard -0.0019 to -0.0039. This is adaptive use of other families' MODEL_SELECT results.

**Why it could work and why it matters.** The teacher sees each block (COW, SCHL, MAR, POBP) only as one-hot columns, so a split
isolates one level; ranges of POBP codes (US states 001-056, then world regions) or of SCHL levels need many splits, which a
budget-truncated booster may not have made for small levels. Revision 1's corrections had the same one-hot inputs. A
correction that can split on ordered codes may reach the coding headroom on top of the FROZEN teacher, and an output that keeps
the teacher's routing stays compatible with the lineage's leaf-score correction machinery used by the privacy stage.

**Revision 2: one change against revision 1, the inputs of the correction trees.** g sees X plus four deterministic, label-free
ordered codes of its one-hot blocks (`ohn.fam_c.compact_codes`: COW, MAR, POBP = index of the active column inside the block,
columns in code order with the block's __OTHER__ level last; SCHL = its ordinal value, __OTHER__ = 3; the coding of
`ihs.fam_h.compact` without the two numeric columns). Everything else as revision 1: cross-fitted training margins, honest CV for
M, alpha on CALIBRATION in [0, 2], deployment on the frozen teacher margin, the nested cross-fitting check (out of fold except
the teacher's split structure). Runtime inputs remain X only (the codes are computed from X).

- Capacity settings: C3_XC_D1 (depth 1: main effects incl. code ranges) and C3_XC_D2 (depth 2). Matched comparisons: C2_XF_D1 /
  C2_XF_D2 (same margins and depth, one-hot inputs only), T_TREE, TASK_REFIT; CC_XF_CONT_D4 as the continuation control.
- Registered expectation (bet): C3 lowers MODEL_SELECT occupation NLL against T_TREE by 0.0005-0.002 and the men's guard by at
  most 0.001; it beats TASK_REFIT by 0.0005 only if the coding headroom is mostly main-effect / pairwise structure reachable
  from the teacher's margin; prior below even odds.
- Stop rule: this is the last revision. If no C3 setting lowers seed-0 NLL against TASK_REFIT by at least 0.0005 with Brier not
  worse and T5 not flagged, the family stops with the negative of the card's abandon rule; the best configuration on both seeds
  is still reported against TASK_REFIT, T_TREE and the controls.

### Closing record (appended 2026-10-18 06:22Z; no design change, no further fit)

- **Revision 2 results** (MODEL_SELECT through the scorer, anchor H3; NLL / Brier differences are candidate minus reference,
  paired household-clustered SEs). C3_XC_D1: seed 0 M = 61, alpha 1.447, NLL vs T_TREE -0.001532 (0.000668), vs TASK_REFIT
  +0.000131 (0.000669), men's guard -0.000759, women's -0.010328; seed 1 M = 70, alpha 1.405, vs T_TREE -0.001615 (0.000683), vs
  TASK_REFIT +0.000463 (0.000690), men's +0.000801, women's -0.008946. C3_XC_D2: seed 0 M = 40, alpha 1.613, vs T_TREE -0.001490
  (0.000824), vs TASK_REFIT +0.000172 (0.000794), men's -0.000623, women's -0.010403; seed 1 M = 37, alpha 1.726, vs T_TREE
  -0.001740 (0.000842), vs TASK_REFIT +0.000338 (0.000815), men's +0.000908, women's -0.009066. C2_XF_D1 seed 1 (promoted): M = 12,
  alpha 2 (bound), vs T_TREE -0.000316 (0.000161), vs TASK_REFIT +0.001762 (0.000453), men's +0.000494.
- With the ordered codes the honest CV finds a transferable residual (M 37-70; alpha 1.4-1.7, inside its bounds; out-of-fold
  MODEL_FIT gains and CALIBRATION gains agree at about 0.0013-0.0024 nats per fold model), so the coding headroom is reachable on
  top of the frozen teacher to about TASK_REFIT's NLL level. It is not beyond it, and it is spent on women: both C3 settings
  leave the men's guard 0.0003-0.0004 worse than T_TREE's (inside T5's 0.0005) and about 0.0012 worse than TASK_REFIT's.
- **Abandon rule applies**: no configuration lowers seed-0 occupation NLL against TASK_REFIT by 0.0005 (best +0.000131). The
  family stops with the negative of the card: no residual learner of this family creates occupation headroom beyond TASK_REFIT.
- **Frozen witness rule** (ohn.rules.witness applied read-only to the stored evaluations): C2_XF_D1, C3_XC_D1, C3_XC_D2 are
  PROMISING_WITNESS with minimum occupation-side ratios 2.656, 2.475, 2.412 (binding: seed-1 men's guard); yardsticks H3+T_TREE
  2.600 and H3+TASK_REFIT 3.186. None is a stronger witness than TASK_REFIT.
- Post-hoc, CALIBRATION only (no fit, no release; unit OHN_C_COMBO_CAL_S{0,1}_V1): TASK_REFIT's margin plus a C3 correction at its
  saved strength lowers CALIBRATION NLL against TASK_REFIT by 0.00085 / 0.00087 (C3_XC_D1, SE about 0.0010) and 0.0015 / 0.0016
  (C3_XC_D2, SE about 0.0013) on seeds 0 / 1; on MODEL_SELECT the two corrections improve different classes (C3: service; TASK_REFIT:
  sales / office). A combined release would be a coordinator decision; it was not made or scored here.
- Configurations used: 12 of 12 (9 on seed 0, 3 on seed 1). Every release reproduces bitwise on all six roles from features
  (`python -m ohn.fam_c verify`).
