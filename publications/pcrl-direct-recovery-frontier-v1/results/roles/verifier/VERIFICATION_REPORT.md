# Verification report (campaign `pcrl_direct_recovery_frontier_v1`, DRF1; independent verifier)

Written from 2026-10-25T04:04Z (`date -u`). Exploratory development on reused ACS 2019 Florida MODEL_FIT / MODEL_SELECT records; a
verification of recorded computations, not a confirmation of any finding. Every check below was computed by separate verifier code
(`roles/verifier/code/`: `vlib.py` and the check scripts; numpy and the standard library only for every metric, objective,
selection and decision rule). Data access used only `drf.common.load(..., 'verify', ...)` inside queue units, `drf.common.features`
(label-free), the saved-array loaders (`drf.release`, `frc.release`, `ohn.devrun.reference_release`, `ohn.common.frozen_bank` for
row order, `tsz.evaluate` for W), `nrm.fa.prc._proba` / `ohn.fam_a.compact_po` to apply frozen teachers, and the admitted reader
recipes only for the refits. Machine-readable results: `roles/verifier/CHECKS.json` (append-only) and `roles/verifier/out/*.json`.
No check was relaxed after a failure. The only failed record is a defect of my own integrity script (section 5); it is kept.

## Summary

| Check | Unit / mode | Items | Result |
|---|---|---|---|
| 1 finite-law objective, v1 + R1, both versions | DRF_VER_C1_V1 (queue, MODEL_FIT labels) | 32,188 | PASS |
| 2 decisive task metrics (28 conditions, G1, paired costs, plan records) | DRF_VER_C2_V1 (queue, MODEL_SELECT labels) | 5,936 | PASS |
| 3 reader comparisons and D-labels, R1 reading (before and against DECISIONS.json) | label-free | 365 + 388 | PASS |
| 3 stopped v1 reading against DECISIONS.json | label-free | 21 | PASS |
| 3b optimistic diagnostic numbers, both readings | label-free | 138 | PASS |
| 3c grid validity of DRF1_R1 by my reading of RULES.md 9 | label-free | 37 | PASS |
| 3d D1 confidence conjunction and two-SE reserve (cand, OBJ_MI, TWIN, START), R1 reading | label-free | 32 | PASS |
| 4 reader refits and bootstrap SEs, grid DRF1_R1 | DRF_VER_C4_V1 (queue, MODEL_SELECT SEX; locked interpreter) | 998 | PASS |
| 5 integrity, first run (defect in my revision parser) | label-free | 189 | FAIL (2, verifier-code defect; kept) |
| 5 integrity, corrected run (04:07Z) | label-free | 189 | PASS |

## 1. Finite-law objective (check 1; `check1_finite_law.py`)

Inputs: `dr/FIT_V<v>.json`, `dr/FIT_R1_V<v>.json`, `dr/BUDGETS.json`, the pinned frozen inputs (SHA-256 against `drf/dr.py`
SOURCE_PINS, parsed as text), MODEL_FIT labels, every saved `runs/DR/<config>/V<v>/MODEL.pkl` / `RELEASE.json` / `RELEASE.npz`, the
FRC1 twin release and MODEL_SELECT / MODEL_FIT features.
- Sources: all pins, row sets and the twin identity agree; the 16 budgets recompute from the pinned FRC1 ledger to 1e-15.
- Routing by OUT-OF-FOLD statistics with my own nearest-codeword, quartile and fine-cell code: my identity B1 tables equal the
  twin's stored tables within 1e-12 (no empty atom; smallest best-to-runner-up cross-entropy gap 6.4e-7, so no near tie); my
  quartile cuts equal the recorded cuts within 1e-15 and give identical bins.
- Every saved configuration (DR_AUC, DR_MI, DR_CTRL_LO, DR_CTRL_HI and the R1 four): my B1 tables equal the saved tables within
  1e-12; on the emitted law (identical vectors merged) and on atom ids, under the estimated (pi) and the empirical (true SEX) law,
  AUC*_pair / _inc / _occ (closed form, checked against the tie-grouped ranking form to 1e-12 and, for the empirical law, against
  exact rational arithmetic), the plug-in MI terms, J_auc, J_mi, the 8 spends (v1: realized labels; R1: expected costs under the
  frozen teachers, with the realized spends reported beside), spend / budget ratios, feasibility, the binding constraint, atom and
  distinct-wire counts all agree with the fit records and the RELEASE.json records within 1e-9 (most to 1e-15); the traces' best
  objectives agree.
- The SEX-blind coarsening path, recomputed independently (v1: realized NLL; R1: expected NLL), reproduces every step's total-NLL
  spend and all in-sample values of `path_full` (71 steps per fit; smallest gap between the best and the runner-up merge 0.0022),
  and the saved CTRL points are my path's partitions at their recorded bracket steps.
- Deployment: the release recomputed from MODEL_SELECT (and MODEL_FIT) features alone equals RELEASE.npz bitwise; its rows are the
  saved table rows of their atoms; the payload teachers equal the pinned teachers; my routing of the twin reproduces FRC1's stored
  TWB_C64 MODEL_SELECT wires bitwise. The deployed MODEL_FIT fine cells differ from the out-of-fold cells for 7,950 / 7,038 (v0) and
  8,248 / 6,901 (v1) of 20,000 people (income / occupation), which confirms that the deployed MODEL_FIT arrays are not the fitted law.
- CTRL bracket selection recomputed from the nested records (check 2's request): candidate spends 0.0389 / 0.0398 (v1) and 0.0206 /
  0.0213 (R1); lower / upper steps 66 / 69 and 67 / 69 (v1), 59 / 62 and 60 / 63 (R1) under both readings of "largest / smallest"
  (step index or spend); my fineness reading is FINE in all four, equal to the plans' records; the nested path spend is not monotone
  (a few decreasing steps), which does not affect the chosen points.
- Observations (MODEL_FIT, label-dependent but no MODEL_SELECT): the v1 DR_AUC's EXPECTED-cost spends under the frozen teachers are
  2.3 / 2.5 x the occupation-NLL budget and 2.6 / 3.4 x the men's-guard budget while its realized spends sit at the budgets, an
  independent confirmation of R1's MODEL_FIT trigger (selection on label noise). R1's realized in-sample spends exceed the budgets by
  up to 1.12 x (expected-cost constraints satisfied). v1's version-1 optimizer shortfall (J_auc 0.2319 for DR_AUC above DR_MI's
  0.2167) is reproduced; in R1 DR_AUC_R1 is below DR_MI_R1 in both versions (0.2415 vs 0.2491, 0.2514 vs 0.2621).

## 2. Decisive task metrics (check 2; `check2_task_metrics.py`)

Inputs: MODEL_SELECT labels, the saved MODEL_SELECT arrays of all eight DR configurations, the twin, SUCC and every frozen reference,
U0 / Ucal through `ohn.devrun.reference_release` (bitwise equal to the frozen bank; row order checked), the evaluation records.
- The 14 conditions per version (my metric code): every value and household-clustered SE agrees with the evaluation records within
  1e-10, every point pass agrees; the twin's clause values also agree with FRC1's published ledger (6 decimals). Evaluated arrays:
  file hashes, release records and my wire hashes all agree.
- G1 (all 28): DR_AUC_R1 and DR_MI_R1 PASS (largest value / cap 0.56 / 0.69 and 0.72 / 0.68); the v1 DR_AUC and DR_MI FAIL (occupation
  NLL excess, occupation Brier excess in some versions, men's guard); all four CTRL points FAIL (men's guard, and occupation NLL
  except DR_CTRL_LO_R1 v0); the twin, SUCC and all nine references PASS. Equal to the coordinator's reading.
- Paired costs: every recorded paired cost of every DR evaluation record (total NLL and the ten per-task metrics; value, SE, se_mc
  = 0) recomputed within 1e-10, including DR_AUC_R1 against each CTRL point, DR_MI_R1, FRC:FA/TWB_C64 and REF:SUCC.
- Reading plans: the distinct-input counts (three views) and the exact-alias relations of both plans recomputed from the wires.

## 3. Reader comparisons and decisions (checks 3, 3b, 3c, 3d; label-free)

Inputs: GRID.json (DRF1_R1, both versions), my check-2 costs and G1, the committed plans, then DECISIONS.json (compared only).
- All 360 recorded cell values of DR_AUC_R1's comparisons reproduce exactly from the per-cell SCORE AUCs; cell SEs as recorded (and
  recomputed independently in check 4).
- My aggregate screen (block means, se_hh = RMS of cell SEs, se_rs, se_agg = max(sqrt(se_hh^2 + se_rs^2), 0.001)) for DR_AUC_R1:
  vs DR_CTRL_HI_R1 (Q*) +0.0340 (se_agg 0.0096; version means 0.0375 / 0.0306) AGG_POSITIVE, but income Brier (+0.0038, SE 0.0013)
  and income NLL (+0.0061, SE 0.0018) are resolved higher -> SEPARATED_ON_PRIMARY_COST_ONLY; vs DR_MI_R1 +0.0076 (0.0117; -0.0004 /
  +0.0156) point only, spend difference +0.0011 (SE 0.0021) OBJ_MI_SPENDS_LESS_UNRESOLVED; vs TWB_C64 +0.0410 (0.0090) AGG_POSITIVE;
  vs SUCC +0.0008 point only with both recipient gates failing; vs NEURAL_RAW +0.0232 (0.0110) AGG_POSITIVE with both gates, 0.020
  target point only; vs H_LJ +0.0135 point only, occupation gate failing (v0 +0.0166). W, TR100, TT100 fail the income gate (v1
  +0.0129); H_LJ, TWJ, B2TWIN fail the occupation gate; RETIRED_B3CTL is not positive. Q* = DR_CTRL_HI_R1 from the pooled paired
  total-NLL costs (the only path point at least as expensive as DR_AUC_R1: pooled d = -0.0022); no cheaper point dominates.
- My labels, computed before the decision record existed and then compared: D1 TASK_FEASIBLE, D2 NOT_SEPARATED, FC
  FAMILY_CHANGE_NOT_RESOLVED, TWIN BELOW_TWIN_RESOLVED, D3 NOT_SUPERIOR (passed NEURAL_RAW only; with the recorded reader-budget
  flags, all READER_BUDGET_ASYMMETRIC, NEURAL_RAW stays credited), D4 NOT_MET, D5 NOT_APPLICABLE_NOT_JOINT, D6 NOT_CLAIMED, optimistic
  USEFUL_OPTIMISTIC_POINT, stage STAGE_3_KEEPS_UTILITY_TIES_CONTROLS (L3), class FEASIBLE_NO_RELIABLE_GAIN: all equal to
  `claims/DRF_R1/DECISIONS.json`, as are the bracket and the D3 passed / failing lists. Stopped v1 reading: D1 CAND_INFEASIBLE, D2-D5,
  FC, TWIN NOT_ASSESSED_TASK_INFEASIBLE, STAGE_2_OPTIMISTIC_SUCCEEDS_LEGAL_MAPPING_FAILS, class TASK_INFEASIBLE: all equal to
  `claims/DRF_V1/DECISIONS.json`.
- Optimistic diagnostic (3b): recomputed by my reading of RULES.md 5.1 from the recorded OPT_L / OPT_S values and the CTRL path values
  verified in check 1: OPT_L useful in-sample in both versions (pair gains 0.0243 / 0.0226 against path steps 61 / 63), OPT_S not
  (0.0056 / 0.0105), neither in the nested mode; all 138 numbers equal the records. Limitation: the OPT mappings are not saved, so
  the OPT values themselves were not recomputed. Observation: OPT_S, whose search space contains OPT_L's, ended at a worse in-sample
  objective (J_auc 0.2338 vs 0.2316, v0), an optimizer limitation of the oracle search (the label does not depend on it).
- Grid validity (3c): my reading of RULES.md 9 over the 3,072 k0 / k1 fingerprint checks finds 82 seed-invariant equalities, all
  explained (66 on the minimum set by extra-trees support, 16 off it by NRM1 (iii)), the same set and explanations as the record;
  READER_SEEDS_EFFECTIVE; the 32 distinct-input counts of the record equal mine.
- D1 beside (3d): my confidence conjunction (value +/- 2 own SE against the cap, 28 conditions) and two-SE reserve (SE floored at
  H_LJ's SE per condition; margin >= 2 SE) reproduce the record: candidate CONFIDENCE_UNRESOLVED (20 of 28 pass; unresolved: income
  accuracy v0, occupation NLL both versions, men's guard both versions) and reserve not met (minimum ratio 1.11 / 0.81, binding the
  men's guard); OBJ_MI unresolved and not met; TWIN all pass and met; START unresolved (men's guard) and not met. With D2's labels
  this supports the class flags POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED, POINT_FEASIBLE_NOT_RESERVE_QUALIFIED,
  OBJ_MI_SPENDS_LESS_UNRESOLVED and TWIN_ONLY_ATTRIBUTION_INSUFFICIENT.

## 4. Reader refits and bootstrap SEs (check 4; `check4_refits.py`, locked interpreter numpy 2.4.2 / scipy 1.17.1 / sklearn 1.8.0)

- Splits R1-R4 recomputed (SHA-256 household order, round robin into tenths) equal READER_SPLITS.json (people, households, set
  hashes); reader seeds equal 2026101205 + 1100 + 10 v + k and avoid the 160 forbidden seeds; member wires equal the grid's.
- For the eight decisive members x 3 views x 8 registered cells: my SCORE AUCs from the cached reader scores equal the records
  within 1e-12; my 400 household-bootstrap draws equal BOOT.npz within 1e-12; the 168 paired cell SEs of DR_AUC_R1 against the
  seven decisive comparators equal the recorded SEs within 1e-12; labels recomputed with my SEs are unchanged.
- Refits (sample fixed by rule before fitting): DR_AUC_R1 pair S0R1k0 (HGB15), DR_AUC_R1 pair S1R2k1 (CATI01), DR_CTRL_HI_R1 pair
  S0R2k0 (LOG_LR), DR_MI_R1 pair S1R1k1 (CATI01), NEURAL_RAW pair S0R2k1 (ET25), DR_AUC_R1 occupation S1R1k0 (MLP6432): SCORE and
  VALIDATION AUCs reproduced exactly, orientation equal, refit scores bitwise equal to the cached scores (6 / 6).

## 5. Integrity (check 5; `check5_integrity.py`, label-free, over the logs as of 04:07Z)

- Registration receipt dae5914f4 verified 03:15:46Z (remote SHA = receipt commit; present in local history) before the first label
  read 03:16:17Z.
- Cards before their fits: DR_DRF_v1 committed 03:14:30Z (reads 03:16:17Z, 03:16:19Z and the OPT reads), DR_DRF_R1 03:27:33Z (reads
  03:27:50Z), DR_DRF_R2 03:59:40Z (reads 03:59:56Z / 03:59:57Z); each read's card commit, SHA-256 and revision match; every saved
  release names its governing card with the committed SHA-256 and its file hash equals its record. dr/BUDGETS.json committed before
  the first fit and unchanged.
- Plans: PLAN_DRF_V1 and PLAN_DRF_R1 committed 03:34:13Z (13d9dbdb9), before the first MODEL_SELECT read 03:34:32Z, unchanged since.
  Readings DRF_V1 and DRF_R1 are registered (cards before any MODEL_SELECT outcome), as recorded; R2's card postdates the first
  MODEL_SELECT read, so any R2 reading is adaptive (its card says so).
- Specifications: 20 distinct, no duplicate, consecutive numbering (12 under DR_DRF_v1 including the 4 optimistic diagnostics, 8
  under DR_DRF_R1); the R2 fits were still running.
- Label access (23 reads): every read's role and fields are allowed by its operation; no closed-role or CALIBRATION label; no
  MODEL_SELECT read with drf.dr in the module chain; every read inside a queue unit that had started. My three reads: MODEL_FIT
  income / occupation / sex (DRF_VER_C1_V1), MODEL_SELECT income / occupation / sex (DRF_VER_C2_V1), MODEL_SELECT sex (DRF_VER_C4_V1).
- The first run of this check failed 2 items: my revision parser (written before R2 existed) mapped every non-R1 revision to v1 and
  therefore expected card DR_DRF_v1 for the R2 fit reads. Corrected to read `--revision` from the unit's argv and require card
  DR_DRF_<revision> (stricter, not relaxed); the failed run is kept (`out/..._check5_integrity_revision_parser_defect.json`).
- Not yet checkable: deployment packages (none built yet).

## Disclosures and limitations

- My code was uncommitted when DRF_VER_C1_V1, DRF_VER_C2_V1 and DRF_VER_C4_V1 ran (the queue recorded the paths and hashes). In
  DRF_VER_C1_V1, `vlib.py` changed between registration and launch and during the run; each change inserted new functions that
  check 1 does not call (the drift of DRF_VER_C2_V1 is a new file it does not import; DRF_VER_C4_V1 had none). A clean re-run of
  check 1 on committed code is pending: DRF_VER_C1_V2 was registered at 04:04Z (CODE_AT_START.json only) and withdrawn with SIGINT at
  04:08:45Z while still waiting for a slot (no unit_start / unit_end, no label read); the re-run will be DRF_VER_C1_V3 (v1, R1, R2).
- My uncommitted verifier files appear in the coordinator's units' `uncommitted_code_paths` / `code_changed_during_run` records
  because the queue hashes the verifier folder; those units do not import verifier code.
- CPU: queue units 29.5 + 1.3 + 4.1 CPU-s; out-of-queue label-free work charged VERIFIER_OUT_OF_QUEUE_1 (60 CPU-s, 04:03:18Z) and
  VERIFIER_OUT_OF_QUEUE_2 (15 CPU-s, 04:08:59Z), both upper estimates. Smoke tests of checks 1, 2 and 4 ran out of queue with
  invented labels only (scratchpad outputs, not records).
- Not recomputed: the nested MODEL_FIT estimator's held-out values (they need the optimizer refitted per fold), the OPT mappings'
  values (not saved), the reader-budget flags (taken as recorded). The `drf/devreaders.py` module docstring still names the source
  grid's seed offset and bootstrap base (1000, 2026102400); the code constants and READER_SPLITS.json use 1100 and 2026102500, as the
  rules require.

## 6. Update 2026-10-25T04:32Z: clean re-run of check 1, R2 check 1 and 2, D2R, R2 labels before the record

- **Check 1 clean re-run, DRF_VER_C1_V3** (registered 04:21:03Z at ffb524dcc with every verifier file committed; `vlib.py` and
  `check1_finite_law.py` unchanged through the run; the only drift was `check3_readers.py`, not imported): v1, R1 and R2, both
  versions, 50,969 items PASS. It reproduces DRF_VER_C1_V1 exactly (check 1r: identical item names, verifier values, results and
  per-configuration values for v1 and R1), so the first run's code drift had no effect. R2 (check 1, group G): R2's per-seed best
  objectives for R1's four seeds equal R1's exactly (deterministic optimizer) and R2's best objective is not above R1's, for DR_AUC_R2
  and DR_MI_R2 in both versions (in-sample estimated-law pair AUC* 0.6212 / 0.6205 for DR_AUC_R2).
- **Check 2 with R2, DRF_VER_C2_V2**: 8,118 items PASS (every configuration of v1, R1 and R2, the twin, SUCC, all references, every
  recorded paired cost and all three plans). G1: DR_AUC_R2 and DR_MI_R2 pass all 28 (largest value / cap 0.81 and 0.90); DR_CTRL_LO_R2
  fails the men's guard in both versions, DR_CTRL_HI_R2 the men's guard and occupation NLL.
- **D2R of the R1 reading (check 3e)**, computed from the registered grid DRF1_R1 and the robustness grid DRF1_R1_ROB before
  `claims/DRF_R1/D2R.json` existed, then compared (434 / 434 PASS): robustness cells = the 24 non-registered cells per comparator (R1 /
  R2 x k2 / k3 and R3 / R4 x k0-k3; the record uses the same set). ROBUSTNESS_NOT_CONFIRMED: against the CTRL bracket DR_CTRL_HI_R1
  confirmed (registered +0.0340, robustness +0.0308, AGG_POSITIVE on the robustness cells and on all cells, robustness recipient
  gates RESOLVED); against DR_MI_R1 not confirmed (robustness +0.0054, point only). Beside: TWB_C64 confirmed (+0.0388); NEURAL_RAW
  (+0.0197) and H_LJ (+0.0123) point only; SUCC changes sign (+0.0008 -> -0.0002) and fails the robustness income gate;
  RETIRED_B3CTL not positive and fails the robustness income gate. Refit REFIT_GENUINELY_EFFECTIVE: R3 and R4 run, assignment hashes
  distinct and equal to READER_SPLITS.json, SCORE overlaps recorded and in [0, 1), 6,144 seed checks with 234 equalities all explained.
  The registered cells of the robustness grid equal the registered grid's (AUCs and SEs). Every per-comparator mean and SE agrees
  within 1e-12. Record note: the record's `beside` list names SUCC twice, so `sign_changes`, `recipient_failures` and the headline
  caveat repeat SUCC ("against SUCC, SUCC"); the sets equal mine (cosmetic).
- **R2 reading, labels before the record** (check 3 with `--no-record`: `claims/DRF_R2/DECISIONS.json` was never opened): registration
  ADAPTIVE_R2 (trigger MODEL_SELECT; card committed 03:59:40Z after the first MODEL_SELECT read 03:34:32Z), so every label carries the
  prefix `ADAPTIVE_R2__`. D1 TASK_FEASIBLE; D2 NOT_SEPARATED (CTRL frontier SEPARATED_ON_PRIMARY_COST_ONLY with bracket DR_CTRL_HI_R2;
  OBJ spend OBJ_MI_SPENDS_AT_LEAST_AS_MUCH); FC FAMILY_CHANGE_NOT_RESOLVED; TWIN BELOW_TWIN_RESOLVED; D3 NOT_SUPERIOR (passed before
  the reader-budget credit rule: NEURAL_RAW, TWJ); D4 NOT_MET; optimistic USEFUL_OPTIMISTIC_POINT; STAGE_3; class
  FEASIBLE_NO_RELIABLE_GAIN. The comparison with the record, R2's grid validity, confidence / reserve and the R2 refits are next.
- Disclosures: the first start of check 3e stopped at `import drf` (the repository root was not on the path) before writing anything;
  fixed and re-run. `check3_readers.py` gained the registration tag (prefixed labels) and `--no-record`; `check4_refits.py` reads the
  reading's members from its plan (the R2 refit sample is the R1 rule with the reading's own ids and the reference refit at S1R2k0);
  `check2_task_metrics.py` lists the R2 configurations; new `check3e_d2r.py`, `check6_deploy.py`, `check1r_reproduce.py`.
  VERIFIER_OUT_OF_QUEUE_3 charged (25 CPU-s, 04:32Z). Queue units: DRF_VER_C1_V3 44.1 CPU-s, DRF_VER_C2_V2 1.5 CPU-s.

## 7. Update 2026-10-25T04:50Z: the R2 reading, R2 robustness, deployment packages

- **R2 decisions (check 3 against `claims/DRF_R2/DECISIONS.json`, after my pre-record labels; 400 / 400 PASS)**: every label equals
  the record both unprefixed and with the prefix `ADAPTIVE_R2__` (registered false, adaptive tag ADAPTIVE_R2; the optimistic label is
  unprefixed, as recorded): D1 TASK_FEASIBLE; D2 NOT_SEPARATED (vs DR_CTRL_HI_R2 +0.0322, se_agg 0.0120, AGG_POSITIVE but income Brier
  and NLL resolved higher -> SEPARATED_ON_PRIMARY_COST_ONLY; vs DR_MI_R2 +0.0039, se 0.0082, point only, OBJ_MI spends at least as
  much, d = -0.0008); FC FAMILY_CHANGE_NOT_RESOLVED (vs SUCC -0.0017); TWIN BELOW_TWIN_RESOLVED (+0.0385); D3 NOT_SUPERIOR (passed
  NEURAL_RAW +0.0207 and TWJ +0.0238, both credited under the reader-budget rule; the others fail); D4 NOT_MET; STAGE_3; class
  FEASIBLE_NO_RELIABLE_GAIN. Grid validity of DRF1_R2 (3c): 3,072 checks, 82 equalities, all explained as recorded (37 / 37);
  confidence conjunction and reserve (3d) as recorded (32 / 32).
- **R2 refits and bootstrap SEs (DRF_VER_C4_V2; 998 / 998 PASS)**: SCORE AUCs from the cache, all bootstrap draws and the 168 paired
  cell SEs within 1e-12; refits DR_AUC_R2 pair S0R1k0 and S1R2k1 (OUT_MLP), DR_CTRL_HI_R2 pair S0R2k0 (LOG_LR; the same wire as
  DR_CTRL_HI_R1 in version 0, so the same reader), DR_MI_R2 pair S1R1k1 (CATI01), NEURAL_RAW pair S1R2k0 (OUT_MLP), DR_AUC_R2
  occupation S1R1k0 (OUT_HGB): exact, refit scores bitwise equal to the cache.
- **D2R of the R2 reading (check 3e)**, computed at 04:38:25Z before `claims/DRF_R2/D2R.json` (written 04:38:37Z), then compared
  (434 / 434 PASS): ROBUSTNESS_NOT_CONFIRMED; CTRL DR_CTRL_HI_R2 confirmed (robustness +0.0302); OBJ DR_MI_R2 not (+0.0040, point
  only); no sign change; robustness recipient failure against SUCC (income gate); refit effective (6,144 checks, 232 equalities
  explained). The record again names SUCC twice in `recipient_failures` and the headline caveat (cosmetic; the sets agree).
- **Deployment packages (check 6, queue units DRF_VER_C6_V1 / _V2, label-free; 244 / 244 PASS each)**: for DR_AUC_R1, DR_MI_R1,
  DR_AUC_R2 and DR_MI_R2, both versions: all 23 dependency hashes; release hash = RELEASE.json = the evaluation record; the stored
  arrays equal RELEASE.npz bitwise on all six roles; my own emission from MODEL.pkl equals the stored MODEL_FIT and MODEL_SELECT arrays
  bitwise; the package used as a black box on the six raw fields of all 20,000 MODEL_FIT rows and on MODEL_SELECT features (whole
  batch; requests of 333 rows over all rows, of 7 rows over the first 700 and of 1 row over the first 100; the coordinator's report
  covers every request size) equals my emission bitwise, decisions are the argmax, only probability and decision are exported;
  the SEX-field, hidden-output and reordered-field probes are refused; DEPLOYMENT_CHECK.json and DEPLOYMENT_CHECK_R2.json report
  all_bitwise, all_attacked_full_batch_bitwise, all_decisions_equal_every_request_size and all_refused true.
- CPU: DRF_VER_C4_V2 4.5, DRF_VER_C6_V1 25.8, DRF_VER_C6_V2 25.6 CPU-s; VERIFIER_OUT_OF_QUEUE_4 charged (30 CPU-s, 04:49Z).
  CHECKS.json holds 167 records; the only non-PASS records are the kept check-5 parser-defect run.

## 8. Final checks 2026-10-25T06:00Z (queue closed 05:53:36Z; label-free, out of queue)

- **Final check 5** (`out/OUTOFQUEUE_20261025T0555Z_check5_integrity_final.json`; 274 / 274 PASS): 38 label reads, all after the
  receipt (03:15:46Z), each inside its queue unit, role and fields allowed by its operation; no ATTACK / DEV_REPORT / CALIBRATION
  label; no MODEL_SELECT read with drf.dr in the chain; cards DR_DRF_v1 / R1 / R2 committed before their fit reads; PLAN_DRF_V1 and
  PLAN_DRF_R1 before the first MODEL_SELECT read; readings DRF_V1, DRF_R1 registered and DRF_R2 ADAPTIVE_R2, as recorded; 28
  specifications (12 / 8 / 8) of 32, no duplicate.
- **Closeout cross-checks 5b** (three recorded runs): (a) PLAN_DRF_R2 committed 04:19:12Z before its first MODEL_SELECT read
  04:27:07Z; bank DRF1_R1_ROB 04:00:11Z before 04:12:43Z; DRF1_R2_ROB 04:30:30Z before 04:30:40Z. (b) ACCESS_RECEIPT.json equals my
  recount (38 reads, 0 before the receipt, 0 closed-role, 0 CALIBRATION, 0 MODEL_SELECT outside benchmark / readers / verify, 0 late
  cards). (c) RESOURCE_ACCOUNTING.json equals my recount at its own time (58 units done, 0 failed, none left without an end, 3.994
  CPU-h, 3.596 h elapsed, 28 configurations; no unit started after the close). (d)-(f) custody: the backup and the drive copy each
  hold TIP_COMMIT 5a791d7f1 and four manifest files of 568,210,103 bytes whose SHA-256 re-hash equals MANIFEST.sha256; their
  RESTORE_CHECK.json report all_ok; the summaries' tip, manifest, bundle, time, read-back and all_ok fields equal the files.
  The first run compared three restore keys at the wrong nesting level (6 FAIL), the second compared the whole nested dictionaries
  (2 FAIL: the summaries omit some per-pass detail fields); the third shows every field the summaries carry equals the file (PASS).
  All three runs are kept; the summaries are abridged, not inconsistent. The drive was only read.
- **Documents**: RESEARCH_DECISION.md's recovery table (60 numbers) equals the grid records within rounding (check 7, 70 / 70);
  ten numbers quoted in START_HERE.md equal my own outputs (check 7b, 10 / 10).
- VERIFIER_OUT_OF_QUEUE_5 charged (20 CPU-s). CHECKS.json: 196 records; the non-PASS records are the kept check-5 parser-defect run
  and the two check-5b layout runs described above.

## Pending (when resumed)

- Done since the 04:10Z report: the commit of the verifier files (coordinator, ffb524dcc), check 1 for R2 and the clean re-run
  (DRF_VER_C1_V3), check 2 with R2 (DRF_VER_C2_V2), D2R (check 3e), the R2 labels before the record.
- Done since 04:32Z (section 7): the R2 reading against its record, R2 grid validity, confidence / reserve and refits, R2 D2R,
  the eight deployment packages.
- Nothing pending from the verifier. New or changed files since the last commit: code/check5b_closeout.py, code/check7_document_numbers.py,
  code/check7b_start_here.py, this report, CHECKS.json and the new files in out/.

Progress reports: 04:10Z (checks 1-5 for v1 and R1); 04:32Z (section 6: clean re-run, R2 checks 1-2, D2R, R2 labels before the
record); 04:50Z (section 7: R2 reading, R2 robustness, deployment).
