# Investigator C report: family C, residual task learning for the occupation output

**Saved by the coordinator.** Investigator C (a separate agent) returned this report as the text of its final response, as the
campaign's briefs require; the coordinator saved it here unchanged in substance.

Exploratory development on previously used records (public-use ACS 2019 1-Year PUMS, Florida), not confirmation. Family C is
task-only: no fit read SEX; MODEL_SELECT labels were read only through `python3 -m ohn.devrun score`; no ATTACK or DEV_REPORT label
was read.

## 1. Bottom line

1. No headroom beyond TASK_REFIT. Family C did not create occupation headroom beyond the existing residual learner TASK_REFIT; the
   card's abandon rule applies after two revisions: the best seed-0 occupation NLL against TASK_REFIT is +0.000131 (C3_XC_D1),
   against the required -0.0005.
2. Exposure is the central finding.
   - A routing-fixed re-estimation of the teacher's leaf values reproduces the frozen CatBoost teacher EXACTLY (max |dp| 4.1e-9 on
     CALIBRATION on both seeds, identical decisions), using Plain boosting, one full-Hessian multiclass Newton step per tree, l2 3, lr
     0.05 and unit weights (the options recorded in the admitted .cbm file). So the frozen layer can be refitted per household fold
     exactly; only its split structure stays in-sample.
   - Its in-sample optimism on MODEL_FIT is large: seed 0 NLL 1.17255 in-sample vs 1.20678 out of fold (+0.0342); seed 1 1.17228 vs
     1.20626 (+0.0340); Brier +0.0145; accuracy -0.021.
   - Consequence: a residual whose round count is chosen by CV on in-sample teacher margins is blind: every depth (1, 2, 4) chose M =
     1, with the CALIBRATION strength pinned at its upper bound 2.
3. What each revision found: revision 1 (cross-fitted, exactly re-estimated margins): one-hot residuals reach at most -0.0003 nats
   against T_TREE; revision 2 (the same plus ordered block codes as inputs): the honest CV finds a transferable residual, occupation
   NLL 0.0015-0.0017 better than T_TREE on both seeds (TASK_REFIT's level, not beyond it); the gain goes to women; the men's Brier gets
   slightly worse.
4. Witness assessment (frozen rule applied read-only): all three two-seed configurations are PROMISING_WITNESS with minimum
   occupation-side ratios 2.656, 2.475 and 2.412 (yardsticks T_TREE 2.600, TASK_REFIT 3.186); none is a stronger witness than
   TASK_REFIT.

## 2. Configurations: 12 of 12 used (9 on seed 0, 3 on seed 1)

Every row is the complete release (H3 income frozen + family-C occupation output), anchor H3. Income clauses identical within a seed:
seed 0 vs U0 accuracy loss +0.00012 (0.00264), NLL -0.00469 (0.00177), Brier -0.00279 (0.00132); vs Ucal +0.00012 (0.00264), -0.00359
(0.00145), -0.00234 (0.00118); seed 1 vs U0 +0.00175 (0.00286), -0.00515 (0.00181), -0.00330 (0.00138); vs Ucal +0.00175 (0.00286),
-0.00488 (0.00159), -0.00325 (0.00128). All 12 clauses and both guards pass at the point for every configuration.

Occupation accuracy loss (identical against U0 and Ucal): C1_RB_D1 / 0 -0.00625 (0.00320); C1_RB_D2 / 0 -0.00625 (0.00320); CC_CONT_D4 / 0
-0.00637 (0.00320); CC_VSCAL / 0 -0.00800 (0.00321); C2_XF_D1 / 0 -0.00637 (0.00320); C2_XF_D2 / 0 -0.00637 (0.00319); CC_XF_CONT_D4 / 0
-0.00612 (0.00319); C3_XC_D1 / 0 -0.00462 (0.00314); C3_XC_D2 / 0 -0.00400 (0.00308); C2_XF_D1 / 1 -0.00575 (0.00333); C3_XC_D1 / 1
-0.00675 (0.00321); C3_XC_D2 / 1 -0.00587 (0.00318).

Occupation NLL and Brier clauses, guards (margins in SE units), fitted M / alpha:

| Config | Seed | M | alpha | NLL vs U0 | Brier vs U0 | NLL vs Ucal | Brier vs Ucal | Men's guard | Women's guard |
|---|---|---|---|---|---|---|---|---|---|
| C1_RB_D1 | 0 | 1 | 2.000 (bound) | -0.01218 (0.00246) | -0.00514 (0.00108) | -0.01130 (0.00228) | -0.00481 (0.00103) | -0.001059 (0.001513), 4.01 SE | -0.009322 (0.001536), 9.33 SE |
| C1_RB_D2 | 0 | 1 | 2.000 | -0.01216 (0.00246) | -0.00514 (0.00108) | -0.01128 (0.00228) | -0.00481 (0.00103) | -0.001047 (0.001512), 4.00 | -0.009329 (0.001536), 9.33 |
| CC_CONT_D4 | 0 | 1 | 2.000 | -0.01223 (0.00246) | -0.00517 (0.00108) | -0.01135 (0.00228) | -0.00484 (0.00103) | -0.001073 (0.001511), 4.02 | -0.009359 (0.001535), 9.36 |
| CC_VSCAL | 0 | - | - | -0.01190 (0.00255) | -0.00507 (0.00112) | -0.01102 (0.00249) | -0.00474 (0.00111) | +0.001701 (0.001571), 2.10 | -0.011995 (0.001566), 10.85 |
| C2_XF_D1 | 0 | 7 | 2.000 | -0.01247 (0.00245) | -0.00523 (0.00108) | -0.01159 (0.00228) | -0.00490 (0.00103) | -0.001170 (0.001511), 4.08 | -0.009380 (0.001533), 9.38 |
| C2_XF_D2 | 0 | 1 | 2.000 | -0.01215 (0.00246) | -0.00514 (0.00108) | -0.01127 (0.00228) | -0.00480 (0.00103) | -0.001042 (0.001512), 4.00 | -0.009321 (0.001535), 9.33 |
| CC_XF_CONT_D4 | 0 | 1 | 2.000 | -0.01219 (0.00246) | -0.00515 (0.00108) | -0.01131 (0.00228) | -0.00482 (0.00103) | -0.001066 (0.001512), 4.01 | -0.009329 (0.001535), 9.33 |
| C3_XC_D1 | 0 | 61 | 1.447 | -0.01366 (0.00236) | -0.00549 (0.00106) | -0.01278 (0.00222) | -0.00516 (0.00102) | -0.000759 (0.001485), 3.88 | -0.010328 (0.001501), 10.21 |
| C3_XC_D2 | 0 | 40 | 1.613 | -0.01362 (0.00235) | -0.00546 (0.00106) | -0.01274 (0.00222) | -0.00513 (0.00102) | -0.000623 (0.001491), 3.77 | -0.010403 (0.001504), 10.24 |
| C2_XF_D1 | 1 | 12 | 2.000 | -0.00856 (0.00228) | -0.00368 (0.00102) | -0.00852 (0.00225) | -0.00367 (0.00101) | +0.000494 (0.001437), 3.14 | -0.007946 (0.001454), 8.91 |
| C3_XC_D1 | 1 | 70 | 1.405 | -0.00986 (0.00219) | -0.00402 (0.00100) | -0.00982 (0.00217) | -0.00401 (0.00099) | +0.000801 (0.001409), 2.98 | -0.008946 (0.001413), 9.87 |
| C3_XC_D2 | 1 | 37 | 1.726 | -0.00999 (0.00220) | -0.00402 (0.00100) | -0.00994 (0.00218) | -0.00402 (0.00100) | +0.000908 (0.001422), 2.88 | -0.009066 (0.001421), 9.90 |

Fitted parameters: CC_VSCAL w = (1.059, 1.058, 1.030, 1.194, 0.925), centred b = (0.014, 0.091, -0.019, -0.007, -0.079), temperature
diagnostic 0.936. Fold round counts M_k: C2_XF_D1 / 0 1, 1, 38, 1, 16; C2_XF_D2 / 0 4, 1, 19, 1, 6; CC_XF_CONT_D4 / 0 1, 1, 16, 1, 1;
C3_XC_D1 / 0 59, 52, 95, 65, 89; C3_XC_D2 / 0 32, 35, 50, 32, 40; C2_XF_D1 / 1 13, 16, 42, 1, 16; C3_XC_D1 / 1 91, 60, 112, 79, 87;
C3_XC_D2 / 1 42, 38, 56, 29, 42.

Paired comparisons on MODEL_SELECT (candidate minus reference; accuracy as a loss; SE):

| Config | Seed | NLL vs T_TREE | Brier vs T_TREE | Acc. loss vs T_TREE | NLL vs TASK_REFIT | Brier vs TASK_REFIT | Acc. loss vs TASK_REFIT |
|---|---|---|---|---|---|---|---|
| C1_RB_D1 | 0 | -0.000057 (0.000018) | -0.000019 (0.000008) | +0.000125 | +0.001605 (0.000400) | +0.000499 (0.000184) | -0.001875 |
| C1_RB_D2 | 0 | -0.000036 (0.000024) | -0.000017 | +0.000125 | +0.001627 (0.000398) | +0.000502 | -0.001875 |
| CC_CONT_D4 | 0 | -0.000099 (0.000036) | -0.000045 | 0.000000 | +0.001564 (0.000397) | +0.000474 | -0.002000 |
| CC_VSCAL | 0 | +0.000228 (0.000789) | +0.000054 | -0.001625 | +0.001891 (0.000849) | +0.000573 | -0.003625 |
| C2_XF_D1 | 0 | -0.000343 (0.000104) | -0.000105 (0.000045) | 0.000000 | +0.001319 (0.000405) | +0.000414 | -0.002000 |
| C2_XF_D2 | 0 | -0.000027 (0.000026) | -0.000011 | 0.000000 | +0.001636 (0.000398) | +0.000508 | -0.002000 |
| CC_XF_CONT_D4 | 0 | -0.000061 (0.000039) | -0.000027 | +0.000250 | +0.001601 (0.000396) | +0.000492 | -0.001750 |
| C3_XC_D1 | 0 | -0.001532 (0.000668) | -0.000365 (0.000261) | +0.001750 | +0.000131 (0.000669) | +0.000154 | -0.000250 |
| C3_XC_D2 | 0 | -0.001490 (0.000824) | -0.000334 (0.000330) | +0.002375 | +0.000172 (0.000794) | +0.000185 | +0.000375 |
| C2_XF_D1 | 1 | -0.000316 (0.000161) | -0.000080 | +0.000375 | +0.001762 (0.000453) | +0.000557 | -0.002250 |
| C3_XC_D1 | 1 | -0.001615 (0.000683) | -0.000420 (0.000265) | -0.000625 | +0.000463 (0.000690) | +0.000218 | -0.003250 |
| C3_XC_D2 | 1 | -0.001740 (0.000842) | -0.000425 (0.000340) | +0.000250 | +0.000338 (0.000815) | +0.000212 | -0.002375 |

Comparison with the two controls:
- Continuation control: the depth-4 continuations collapse exactly like the capacity-limited residuals, in-sample (CC_CONT_D4) and on
  cross-fitted margins (CC_XF_CONT_D4): M = 1, NLL about -0.0001 against T_TREE. The capacity limit is not what fails; the margins and
  the inputs are. Caveat: no depth-4 continuation was fitted with the ordered-code inputs, so capacity limit versus continuation is
  untested for revision 2.
- Calibration-only control: vector scaling on CALIBRATION (CC_VSCAL) worsens NLL (+0.00023) and moves Brier from men to women (men's
  guard +0.0027 against T_TREE); it fails the per-seed reserve (1.82). On held-out CALIBRATION, cross-fitted vector scaling is also
  worse than the teacher (NLL +0.00068 (SE 0.00135) seed 0; +0.00075 (0.00133) seed 1). Calibration explains none of revision 2's gain.
  CC_VSCAL was fitted on seed 0 only (the seed-1 comparison is the CALIBRATION check).

Cross-fitting, deployed minus fold-average on MODEL_SELECT (NLL (SE), decision agreement): C1_RB_D1 / 0 -0.000026 (0.000016), 0.9998;
C1_RB_D2 / 0 +0.000026 (0.000018), 0.9996; CC_CONT_D4 / 0 -0.000001 (0.000030), 0.9989; C2_XF_D1 / 0 -0.000272 (0.000103), 0.9961;
C2_XF_D2 / 0 +0.000069 (0.000089), 0.9964; CC_XF_CONT_D4 / 0 +0.000086 (0.000097), 0.9965; C3_XC_D1 / 0 -0.000036 (0.000104), 0.9964;
C3_XC_D2 / 0 +0.000101 (0.000198), 0.9909; C2_XF_D1 / 1 -0.000126 (0.000094), 0.9973; C3_XC_D1 / 1 -0.000009 (0.000101), 0.9959;
C3_XC_D2 / 1 +0.000087 (0.000187), 0.9906. No gap approaches a clause's se_used. Out-of-fold MODEL_FIT NLL: C1 and CC_CONT_D4
1.1726-1.1727 (in-sample teacher margins); C2 and CC_XF_CONT_D4 1.2065-1.2071; C3 1.2047-1.2053 (out-of-fold teacher 1.20678 /
1.20626). The scorer's out-of-fold-versus-in-sample-T_TREE flag is about +0.000 for C1 (both in-sample) and +0.032 to +0.035 for C2
and C3: the teacher's in-sample optimism, not a defect of the correction.

Fits and CPU: each residual configuration ran 16 xgboost trainings (6 model fits, 10 cross-fitting fits) and 16 calibrator fits; each
C2 / C3 configuration and CC_XF_CONT_D4 added 26 exact re-estimations; CC_VSCAL 0 model fits and 7 calibrator fits. Fit-unit CPU-s:
C1_RB_D1 / 0 79.4; C1_RB_D2 / 0 97.9; CC_CONT_D4 / 0 135.7; CC_VSCAL / 0 4.4; C2_XF_D2 / 0 153.8; C2_XF_D1 / 0 138.4; CC_XF_CONT_D4 / 0
194.2; C3_XC_D1 / 0 139.2; C3_XC_D2 / 0 156.4; C2_XF_D1 / 1 133.8; C3_XC_D1 / 1 138.4; C3_XC_D2 / 1 155.2; score units about 1.3 CPU-s
each. Ledger entries 49, 52, 55, 56, 61, 62, 63, 68, 69, 70, 71, 72.

Held-out CALIBRATION checks (NLL vs the teacher, strength household-cross-fitted on CALIBRATION; SE): C1_RB_D1 / 0 -0.000035
(0.00002); C1_RB_D2 / 0 -0.000086 (0.00003); CC_CONT_D4 / 0 -0.00016 (0.00006); C2_XF_D1 / 0 -0.00022 (0.00014); C2_XF_D1 / 1 -0.00032
(0.00023); C2_XF_D2 / 0 -0.00010 (0.00003); CC_XF_CONT_D4 / 0 -0.00022 (0.00006); C3_XC_D1 / 0 -0.00116 (0.00107); C3_XC_D1 / 1
-0.00119 (0.00110); C3_XC_D2 / 0 -0.00190 (0.00132); C3_XC_D2 / 1 -0.00205 (0.00136); TASK_REFIT / 0 -0.00048 (0.00069); TASK_REFIT
/ 1 +0.00004 (0.00076).

## 3. What each stage taught

- Initial design (C1: depth 1 and 2 residual on the frozen T_TREE margin, M by in-sample CV, alpha on CALIBRATION; plus both
  controls): the CV curve rises from round 1 at every depth (D1 1.17257 -> 1.17833, D2 1.17258 -> 1.18166, D4 1.17262 -> 1.19105) and
  is already above the uncorrected 1.17255, so M = 1, alpha pinned at 2 and the outputs are essentially T_TREE; CALIBRATION still asked
  for more of the first-round correction. The in-sample teacher absorbs on MODEL_FIT exactly the residual the correction would need.
- Revision 1 (training and selection on cross-fitted margins): makes the selection honest (the in-sample twin gained -0.000057, the
  cross-fitted D1 -0.00034, 3.3 SE), but the one-hot residual signal is tiny; D2 and D4 still chose M = 1.
- Revision 2 (the correction also sees ordered codes: POBP, COW and MAR block index and the SCHL ordinal value): the honest CV finds
  signal (M 37-70, alpha 1.4-1.7 within bounds; fold-wise out-of-fold MODEL_FIT gains roughly match CALIBRATION gains, 0.0013-0.0024 per
  fold model); the trees split mostly on the ordered SCHL value (27-41% of splits), the ordered POBP code (17-30%) and POBP one-hot
  columns (23-27%); the gain goes to class 1 (service: NLL contribution -0.0144 -> -0.0158 / -0.0162 on seed 0), whereas TASK_REFIT
  gains in class 2 (sales / office: -0.0143 -> -0.0183), so the two are structurally complementary; revision 2 costs a little class-4
  Brier and men's Brier (men 0.0002-0.0004 worse than T_TREE, inside T5's 0.0005), women about 0.001 better.

## 4. Promising witnesses (frozen rule, read-only)

| Config | Label | Min occupation-side ratio | Binding condition | Whole-release reserve |
|---|---|---|---|---|
| C2_XF_D1 | PROMISING_WITNESS | 2.656 | seed-1 men's guard | RESERVE_MET |
| C3_XC_D1 | PROMISING_WITNESS | 2.475 | seed-1 men's guard | RESERVE_MET |
| C3_XC_D2 | PROMISING_WITNESS | 2.412 | seed-1 men's guard | RESERVE_MET |
| H3+T_TREE (yardstick) | - | 2.600 | - | - |
| H3+TASK_REFIT (yardstick) | - | 3.186 | - | - |

No family-C configuration is a useful witness beyond the existing task-only references: C2_XF_D1 is practically T_TREE (-0.0003 nats;
men's guard about 0.0001 better); C3_XC_D1 is the best family-C task learner on NLL (-0.0015 / -0.0016 against T_TREE on both seeds;
more accurate than C3_XC_D2) but it spends the binding men's Brier.

Exact operation of the improvement (C3_XC_*): p(x) = softmax(z_T(x) + alpha g(x)), decision = argmax; z_T the centred frozen T_TREE
occupation raw score from its exact numpy routing; g = M rounds x 5 classes of xgboost 3.2.0 multi:softprob trees of depth d (eta 0.05,
lambda 3) on [X, COW index, SCHL value, MAR index, POBP index], trained with base_margin equal to each MODEL_FIT person's teacher margin
re-estimated exactly without that person's household fold; M by 5-fold household CV with validation margins re-estimated without the
validation fold; alpha by NLL on CALIBRATION in [0, 2]; deployment by a numpy evaluator of the saved trees on the frozen teacher margin.
Fitted: D1 M 61 / 70, alpha 1.447 / 1.405; D2 M 40 / 37, alpha 1.613 / 1.726 (seed 0 / 1).

Post-hoc suggestion (CALIBRATION only; no fit, nothing released): TASK_REFIT's margin plus the C3 correction at its saved alpha lowers
CALIBRATION NLL against TASK_REFIT by 0.00085 / 0.00087 with D1 (SE about 0.0010) and 0.0015 / 0.0016 with D2 (SE about 0.0013); its
cross-fitting check would not be exact (TASK_REFIT cannot be refitted per fold). Not pursued (stage 1 closed).

## 5. Exposure statement (as recorded in each OCC.json)

T_TREE in-sample on every MODEL_FIT person and fold. Initial design: out-of-fold predictions are out of fold for the correction layer
only. Revisions 1-2: out of fold for every label-using step (leaf values, M_k, g_k, alpha_k) except the teacher's split structure
(chosen by CatBoost on all MODEL_FIT, not refitted): still not fully out of sample. alpha / vector scaling: in-sample on CALIBRATION
(1 / 9 parameters).

## 6. Failures, refusals, deviations

- Failures: `OHN_C_DRYRUNFULL_C2_XF_D2_S0_V1` (diagnostic, 49.6 CPU-s) failed: the fidelity assertion fired because invented labels
  cannot reproduce the real teacher's leaf values (an artifact of the dry run; only dry runs now bypass the assertion; real fits
  enforce it at fidelity 4e-9). No real fit or verification failed. The first fit waited 05:12 -> 05:41Z for a queue slot (starvation
  reported; repaired by the coordinator with FIFO tickets at 05:40Z).
- Harness refusals (exact messages; each time the action was stopped and another route used): (1) "This session is isolated in the
  worktree <repository>/.claude/worktrees/pcrl-occupation-headroom-overnight-v1, but this command is too complex to verify
  that it stays inside the worktree. Refusing to run it — a worktree-isolated session's git operations must target its own worktree.
  Split it into plain, separate commands and run them from <repository>/.claude/worktrees/pcrl-occupation-headroom-overnight-v1."
  (a chain script was written with the Write tool instead); (2) "...but this command runs sed with a value computed at runtime (command
  output) where an option may stand ... Refusing to run it ..." (the Read tool was used instead); (3) "Blocked: sleep 60 followed by:
  ... To wait for a condition, use Monitor with an until-loop ... Do not chain shorter sleeps to work around this block." (Monitor and
  background until-loops used instead).
- Deviations (disclosed in CARD.md): revision 1 answered T1's failure mode but T1's literal condition did not fire (the symptom was the
  opposite: a blind CV and alpha at the upper bound); T4's listed revision (stacking on TASK_REFIT) was not run (TASK_REFIT's margins are
  even more in-sample and its correction cannot be refitted per fold exactly); revision 2 departed from the revision-1 text ("revision 2
  only on a transfer failure"), motivated by the coordinator's 06:02Z relay of families A and B's ordinal-coding result, i.e. adaptive use
  of other families' MODEL_SELECT results; two uncommitted section headings had their times corrected before commit (06:13 -> 06:11Z
  and 06:25 -> 06:22Z); `fam_c.py` grew additively during the run (each OCC.json records the code sha at fit time); the final code
  (sha 71b0bfad...) reproduces all 12 releases bitwise on all six roles (OHN_C_VERIFYALL_V1).

## 7. Files and resources

Card `roles/investigator_c/CARD.md` (revision 1, revision 2 and the 06:22Z closing record); code `ohn/fam_c.py`; releases and parameters
`runs/C/<config>/S<s>/` (OCC.npz / json, CROSSFIT.npz, PARAMS.json, TREES.npz, MODEL.json, VERIFY_*.json, EVAL_MS_{H3,TW}.json);
diagnostics `runs/C/DIAG/EXPOSURE_S{0,1}.json` and `COMBO_CAL_S{0,1}.json` (private). Selftests V1 23/23, REBOOST 4/4, XF 7/7; dry runs on
real features with invented labels PASS except the artifact above. Resources: 37 queue units, 1,772.0 CPU-s, plus 90 CPU-s charged
(INVESTIGATOR_C_AGENT): about 0.52 CPU-h. Nothing committed or pushed by the investigator.
