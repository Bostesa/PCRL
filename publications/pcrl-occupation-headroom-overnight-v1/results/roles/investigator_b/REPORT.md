# Investigator B (family B) report: complementary occupation learners and a household-cross-fitted stack

**Saved by the coordinator.** Investigator B (a separate agent) returned this report as the text of its final response, as the
campaign's briefs require; the coordinator saved it here unchanged in substance.

Exploratory development on previously used ACS 2019 Florida PUMS records, not confirmation. Family B is task-only: no SEX enters any
fit, input or output; no reader was run. All MODEL_SELECT numbers come from `python3 -m ohn.devrun score` (anchor H3) with
household-clustered SEs.

## 1. Bottom line

- The stack adds task value beyond the best equally tuned single learner, on both seeds, in both designs; the pre-registered rule is
  met for B1 and B2. On fully cross-fitted MODEL_FIT (20,000 people) STACK minus SINGLE NLL: B1 -0.00206 (SE 0.00047) / -0.00210
  (0.00046); B2 -0.00253 (0.00047) / -0.00237 (0.00046). On MODEL_SELECT the stack's NLL vs U0 is lower than its single learner's on
  every seed (by -0.0013 to -0.0020).
- New occupation headroom comes mostly from repairing the tree learner, plus the stack: B2_STACK beats TASK_REFIT on both seeds (NLL
  -0.00294 (SE 0.00147) / -0.00306 (0.00148); Brier -0.00149 (0.00067) / -0.00168 (0.00069)) and T_TREE by -0.0046 / -0.0051 NLL
  (about 3 SE); its men's guard is -0.0035 / -0.0021 (T_TREE -0.0010 / +0.0006; inherited JOINT_TREE +0.0044 / +0.0039).
- Recommended witness: B2_STACK (both seeds) with matched control B2_SINGLE; fully cross-fitted, no frozen layer, verified bitwise.
- B3 (CALIBRATION layer adding the frozen TASK_REFIT) does not meet its rule. 10 of 12 configurations used; all 10 scored on both
  seeds by 06:21Z.

## 2. Designs and exact operations (`ohn/fam_b.py`, sha256 a3a605f1...9549, committed 5c9c6cef5)

- TREE_V1 (B1): scikit-learn 1.8.0 HistGradientBoosting, multinomial, on the compact coding (AGEP z, WKHP z, COW code, SCHL ordinal,
  MAR code, POBP code; COW / MAR / POBP as native categories); learning rate 0.1, up to 600 iterations, l2 1; grid leaves {15, 31,
  63} x min leaf {20, 100}; tree count read off the inner-CV curve (checkpoints of 10).
- ADD_V1 (B1): L2 multinomial logistic regression (lbfgs, tol 1e-6) on a label-free additive basis: cubic B-splines of AGEP and WKHP
  (6 uniform knots); indicators for the 13 WKHP values held by >= 200 MODEL_FIT people; the one-hot blocks of X; capacity S (332
  columns: EDU6xCOW, EDU6xAGEP-spline, COWxWKHP, EDU6xWKHP, AGEP-splinexWKHP, MARxAGEP, POB6xEDU6); capacity L (840 columns) adds
  SCHLxCOW, EDU6/COWxWKHP-spline, AGEP x WKHP splines, POB6xCOW/AGEP/WKHP, MARxEDU6/COW; grid {S, L} x C {0.03, 0.15, 0.75}.
- Equal tuning effort: 6 settings each, the same seed-specific household 5-fold inner CV, the same criterion (out-of-fold NLL), 31
  fits per training set.
- Final layers on inner out-of-fold logits only: STACK = softmax(a_T z_T + a_A z_A), a >= 0; SINGLE = the learner with the lower
  temperature-scaled OOF NLL, softmax(tau z).
- Full nested cross-fitting: for each OHN1_crossfit fold the entire pipeline is rerun without that fold (tuning with its own inner
  folds, refit, learner choice, layer); every outer fit asserts household separation.
- B2 (revision 1): TREE_V2 with POBP as its ordinal level index (COW, MAR native categories), learning rate 0.05, up to 500
  iterations, leaves {4, 8, 15} x min leaf {50, 200}; ADD_V2 C {0.003, 0.01, 0.03} x {S, L}; layers unchanged.
- B3 (revision 2, last): B3_CALTR = softmax(c1 s_B2 + c2 z_TASK_REFIT), c >= 0, the 2 weights fitted on CALIBRATION (2,982 people);
  TASK_REFIT scores recomputed from features (routing plus stored E20 leaf corrections) match the frozen bank to within 3e-16.
- Deployed parameters (seed 0 / seed 1): B1_STACK pool (TREE_V1, ADD_V1) (0.287390, 0.739153) / (0.277586, 0.735046); B1_SINGLE
  (ADD_V1) tau 1.007016 / 1.006925; B2_STACK pool (TREE_V2, ADD_V2) (0.592491, 0.422059) / (0.633499, 0.381509); B2_SINGLE (TREE_V2)
  tau 1.003430 / 1.005370; B3_CALTR c (pool, TASK_REFIT) (0.879428, 0.159105) / (0.917066, 0.122973).
- Chosen base learners: TREE_V1 15 leaves / min leaf 100 / 20 iterations, inner-CV NLL 1.22586 (seed 0); 15 / 20 / 30, 1.22579
  (seed 1); ADD (V1 = V2) L, C 0.03, 1.21470 / 1.21420; TREE_V2 8 leaves / min leaf 50 / 130 iterations, 1.21224 / 1.21065.

## 3. MODEL_SELECT results

Income clauses (H3 anchor; identical for every family-B configuration of a seed): seed 0 vs U0 accuracy +0.000125 (SE 0.002637), NLL
-0.004692 (0.001773), Brier -0.002790 (0.001319); vs Ucal accuracy +0.000125, NLL -0.003592 (0.001452), Brier -0.002342 (0.001179);
seed 1 vs U0 accuracy +0.001750 (0.002861), NLL -0.005154 (0.001812), Brier -0.003303 (0.001384); vs Ucal accuracy +0.001750, NLL
-0.004879 (0.001595), Brier -0.003254 (0.001284).

Occupation clauses (vs U0 / vs Ucal; accuracy loss identical against both), guards and comparisons:

| Config | Seed | Ledger | Acc. loss | NLL vs U0 | Brier vs U0 | NLL vs Ucal | Brier vs Ucal | Men's guard | Women's guard | NLL vs T_TREE | NLL vs TASK_REFIT | Brier vs TASK_REFIT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_STACK | 0 | 50 | -0.004999 (0.003127) | -0.013102 (0.002377) | -0.005516 (0.001055) | -0.012222 (0.002254) | -0.005185 (0.001017) | -0.000873 (0.00149) | -0.010264 (0.001485) | -0.000975 (0.001556) | +0.000688 (0.001581) | +0.000127 (0.000723) |
| B1_SINGLE | 0 | 51 | -0.002250 (0.003235) | -0.011811 (0.002561) | -0.004823 (0.001166) | -0.010930 (0.002433) | -0.004492 (0.00112) | -0.000450 (0.001628) | -0.009295 (0.001663) | +0.000316 (0.001758) | +0.001979 (0.00177) | +0.000820 (0.00082) |
| B1_STACK | 1 | 66 | -0.005124 (0.003203) | -0.010314 (0.00228) | -0.004445 (0.001022) | -0.010266 (0.00226) | -0.004440 (0.001017) | +0.000023 (0.001445) | -0.009013 (0.001438) | -0.002065 (0.001551) | +0.000013 (0.001579) | -0.000209 (0.000733) |
| B1_SINGLE | 1 | 67 | -0.002250 (0.00334) | -0.008457 (0.00248) | -0.003575 (0.001129) | -0.008409 (0.00246) | -0.003571 (0.001123) | +0.001078 (0.001587) | -0.008333 (0.001602) | -0.000209 (0.001726) | +0.001870 (0.001744) | +0.000661 (0.000815) |
| B2_STACK | 0 | 59 | -0.007124 (0.003322) | -0.016730 (0.002461) | -0.007134 (0.001122) | -0.015849 (0.002335) | -0.006803 (0.001077) | -0.003494 (0.001575) | -0.010857 (0.001578) | -0.004603 (0.001467) | -0.002940 (0.00147) | -0.001491 (0.000673) |
| B2_SINGLE | 0 | 60 | -0.008749 (0.003436) | -0.014695 (0.002663) | -0.006545 (0.001219) | -0.013814 (0.002531) | -0.006213 (0.001168) | -0.004069 (0.001709) | -0.009077 (0.001711) | -0.002567 (0.001668) | -0.000905 (0.001678) | -0.000901 (0.000767) |
| B2_STACK | 1 | 64 | -0.007374 (0.003274) | -0.013391 (0.00236) | -0.005914 (0.001081) | -0.013343 (0.002341) | -0.005909 (0.001075) | -0.002055 (0.001513) | -0.009859 (0.001527) | -0.005142 (0.001456) | -0.003064 (0.001477) | -0.001678 (0.000689) |
| B2_SINGLE | 1 | 65 | -0.008749 (0.003421) | -0.011365 (0.002532) | -0.005304 (0.001164) | -0.011317 (0.002511) | -0.005300 (0.001156) | -0.002495 (0.00162) | -0.008176 (0.001648) | -0.003116 (0.001667) | -0.001038 (0.001687) | -0.001068 (0.000783) |
| B3_CALTR | 0 | 73 | -0.007749 (0.003215) | -0.017430 (0.002397) | -0.007314 (0.001095) | -0.016550 (0.002329) | -0.006983 (0.001074) | -0.002532 (0.001544) | -0.012204 (0.001531) | -0.005303 (0.001358) | -0.003640 (0.001307) | -0.001671 (0.000593) |
| B3_CALTR | 1 | 74 | -0.007249 (0.0032) | -0.014022 (0.002345) | -0.006074 (0.001076) | -0.013973 (0.002338) | -0.006070 (0.001075) | -0.001165 (0.001514) | -0.011093 (0.001513) | -0.005773 (0.001397) | -0.003695 (0.001398) | -0.001838 (0.000649) |

- All 10 pass all 12 point clauses and both guards. Per-seed evaluations report RESERVE_PENDING_SECOND_SEED (both seeds now scored);
  B2_STACK minimum ratio 3.511 / 2.862, bound by the H3 income accuracy clause.
- Brier vs T_TREE, B2_STACK: -0.002009 (SE 0.000665) / -0.002315 (0.000666).
- Per SEX, B2_STACK vs U0: men NLL -0.0078 (0.0037) / -0.0043 (0.0035), accuracy +0.0017 / +0.0027; women NLL -0.0259 (0.0032) /
  -0.0227 (0.0030). The gain is still concentrated in women, as for T_TREE.
- Per class: the decomposition vs U0 depends on the seed-specific U0 (seed 0: classes 1 and 2 gain, 0 and 4 lose; seed 1: 0 and 2 gain).

## 4. Cross-fitting

Fully cross-fitted MODEL_FIT out-of-fold NLL / Brier / accuracy (seed 0; seed 1): B1_STACK 1.212792 / 0.614397 / 0.49575; 1.212769 /
0.614400 / 0.49540. B1_SINGLE 1.214855 / 0.615447 / 0.49555; 1.214864 / 0.615449 / 0.49555. B2_STACK 1.210337 / 0.613611 / 0.49745;
1.210064 / 0.613511 / 0.49715. B2_SINGLE 1.212861 / 0.614603 / 0.49675; 1.212435 / 0.614444 / 0.49695. B3_CALTR 1.1943; 1.1946 (B3's
out-of-fold values are optimistic because TASK_REFIT is in-sample on MODEL_FIT; flagged in OCC.json).
- Cross-fitted diagnostics vs SINGLE (NLL): TREE_V1 raw +0.0104 (0.0016); for B2: ADD +0.0020 / +0.0024 (0.0012), linear pool
  -0.00239 / -0.00227, multinomial-logistic stacker -0.00231 / -0.00188 (the logistic stacker beat the pool in B1 (-0.0028) but is
  worse than the pool in B2); Brier gain of the B2 pool -0.00099 (0.00022) / -0.00093 (0.00022).
- Deployed minus fold-average on MODEL_SELECT: B2_STACK NLL -0.00023 (0.00035) / +0.00013 (0.00037), decision agreement 0.977 /
  0.976; all configurations within 2 SE (largest B2_SINGLE seed 1 +0.00090 (0.00057)); agreement 0.963-0.990.
- B3 cross-fitted CALIBRATION check (CALPOOL minus the unrefitted B2 pool, NLL): +0.00044 (0.00068) / +0.00004 (0.00059); against
  the pool with a CALIBRATION temperature +0.00039 / -0.00005.

## 5. Failures and diagnosis

- TREE_V1 overfits catastrophically: every setting's inner-CV NLL is minimized at 20-30 iterations and rises to 1.35-1.70 by 600
  (family A independently saw T* about 40). A MODEL_FIT-only diagnosis (unit OHN_B_DIAG_TREE_ENCODING_S0_V1, 15 fits, 213.5 CPU-s)
  found the cause: native category-subset splits on the 106 POBP codes cost about 0.012 nats; at TREE_V1's chosen setting the
  inner-CV NLL was TREE_V1 encoding 1.22586, POBP ordinal 1.21382, no native categories 1.21466, one-hot X 1.22146 (14x the CPU).
  So B1's "best single learner" was ADD.
- ADD chose the grid edge twice (C = 0.03 is the low edge in V1 and the high edge in V2); bracketed by the union of the grids (0.01
  -> 1.2190, 0.03 -> 1.2147, 0.15 -> 1.2204).
- Selftest V4 failed (test-harness defect: it iterated over the new B2 registry entries without building their invented bases;
  KeyError); V5 and V6 PASS with 23 and 25 checks (ledger entry 75).
- No progress lines for the first base units (stdout block-buffered); fixed with flushing; no effect on results.
- Ledger typo: entry 80's text says 06:27Z; correction entry 81 gives the actual 06:21:39Z.

## 6. What each revision taught

- B1: the convex pool adds about 0.002 nats even with a weak tree base, but the tree base itself was defective.
- B2: repairing POBP (ordinal), smaller trees and lr 0.05 make the tree the best single learner (cross-fitted 1.2129 vs ADD's
  1.2149); pooling two learners of comparable strength gives the largest stack gain; B2_STACK clears TASK_REFIT by about 2 SE.
- B3: the frozen teacher gets small weights (0.16 / 0.12); its MODEL_SELECT point gain over B2_STACK (-0.0007 / -0.0006 NLL) does not
  show on cross-fitted CALIBRATION; it comes from women and costs men's margin (+0.0010 / +0.0009 on the men's guard). Rule not met.

## 7. Caveats

- The second seed is not an independent refit of B2's bases: the deployed TREE_V2, ADD_V2 and ADD_V1 models are bitwise identical
  across seeds (the inner CV picked the same setting; the fits are deterministic; HGB random_state is unused); ADD_V2 equals ADD_V1 as
  deployed. B2's and B3's seeds differ only in the layer weights (fitted on seed-specific inner out-of-fold logits) and in the
  cross-fitting pipelines; their seed-to-seed differences mostly reflect the seed-specific references.
- Adaptive choices: B2 and B3 were chosen after logged MODEL_SELECT feedback (disclosed in the card).
- B3 exposure: CALIBRATION occupation labels read by B3 only; B3's MODEL_FIT out-of-fold predictions are not fully out of sample.
- No privacy claim: better occupation utility is not evidence that SEX recovery falls; no reader was run.
- Card timing: the initial card is in the registration commit b1fd591d0; B2 committed 2f785ec18 (05:45:46Z) before its first fit; B3
  committed 5c9c6cef5 (06:19:50Z) before its fit; the B3 results, family close and seed caveat were appended afterwards (committed
  c51f8035e). The diagnostic unit ran on an intermediate code version whose hash was not recorded.

## 8. Fits, CPU, files

- Base units: 8 units of 186 fits each = 1,488 model fits (248 main-pipeline fits including inner CV, 1,240 cross-fitting fits),
  shared within each design and seed.
- Per configuration (OCC.json): n_model_fits 62; STACK n_calibrator_fits 1, n_crossfit_fits 315; SINGLE 2 and 320; B3 2 and 320 plus
  10 CALIBRATION-check fits; each configuration also has 45 small cross-fitted diagnostic layer fits (not released).
- CPU (queue): bases TREE_V1 669 + 637 s, ADD_V1 152 + 144 s, TREE_V2 262 + 263 s, ADD_V2 48 + 48 s; diagnostic 214 s; configuration
  fits 3-11 s each; scores about 1.3 s each; selftests (6 runs incl. the failed V4) 64 s; timing probe 4 s; verifyall 12.6 s. Totals
  2,579.9 queue CPU-s over 37 units (1 failed) plus a 190 s agent charge (INVESTIGATOR_B_AGENT): about 0.77 CPU-h.
- Deployment: `python3 -m ohn.fam_b verifyall` (unit OHN_B_VERIFYALL_V1) PASS: `predict` reproduces all 10 releases bitwise on all
  six roles (needs Homebrew python3 3.14 with scikit-learn 1.8.0; pickled models, sha256 recorded).
- Ledger rows 50, 51, 59, 60, 64-67, 73, 74 (scores); 75-81 (failure, diagnosis, decisions, stop, correction).
- Files: `ohn/fam_b.py`; `roles/investigator_b/CARD.md`; releases `runs/B/{B1_STACK,B1_SINGLE,B2_STACK,B2_SINGLE,B3_CALTR}/S{0,1}/`;
  bases `runs/B/BASES/{TREE_V1,ADD_V1,TREE_V2,ADD_V2}/S{0,1}/`; diagnostic `runs/B/DIAG/` (private).

## 9. Refusals (tool permissions only; no data-access or label refusal)

1. Three times: "This session is isolated in the worktree <repository>/.claude/worktrees/pcrl-occupation-headroom-overnight-v1,
   but this command is too complex to verify that it stays inside the worktree. Refusing to run it — a worktree-isolated session's git
   operations must target its own worktree. Split it into plain, separate commands and run them from
   <repository>/.claude/worktrees/pcrl-occupation-headroom-overnight-v1."
2. A Monitor loop containing git: the same message, with "this command names git in a form too complex to verify".
3. A shell loop over unit names: "...this command runs python with a program computed at runtime (the variable u) inside a construct
   too complex to verify, so what it runs cannot be shown not to be git..."
4. A plain sleep: "Blocked: sleep 45 followed by: ... To wait for a condition, use Monitor with an until-loop ... Do not chain shorter
   sleeps to work around this block."
Each time the command was split or a background wait loop used. Nothing was committed or pushed by the investigator.

Coordinator's note: under the frozen witness rule (ranking by the occupation-side leximin reserve vector, i.e. led by the men's
guard), the selected witnesses are B2_SINGLE (this family's matched single-learner control) and A1_PO_16; the investigator's
recommended B2_STACK ranks third (WITNESS_SELECTION.json).
