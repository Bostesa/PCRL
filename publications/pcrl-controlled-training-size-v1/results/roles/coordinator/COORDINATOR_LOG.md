# Coordinator log (times are `date -u` readings taken before each entry was written)

- 23:45:15Z: campaign start (first action). Read the pinned correction-transfer records (START_HERE, ADVISOR_BRIEF,
  RESEARCH_DECISION, TRAINING_SIZE_BET, TRAINING_SIZE_CURVE) and the source code paths (cts.common, cts.trackf, cts.nested,
  cts.evaluate, cts.devreaders, cts.deploy, ohn.fam_a, ohn.rules, ohn.protect, ohn.evaluate, acs_replication.stage_b).
  No AGENTS.md exists in the repository (checked the worktree to depth 4 and the main checkout). No worktree or branch matches this
  campaign, so nothing is resumed.
- 23:53:57Z: created branch research/pcrl-controlled-training-size-v1 from 507024649 in the worktree directory this session is bound
  to (.claude/worktrees/pcrl-correction-transfer-stability-v1); the previous research branch is unchanged. Label-free counts:
  MODEL_FIT 20,000 people / 13,337 households; CALIBRATION 2,982 / 1,974 (no SEX admitted); MODEL_SELECT 8,001 / 5,306.
- about 23:58Z: DESIGN_BRIEF.md written. Assignments started (wave 1): statistical reviewer (rules, trigger, selection rule, reader
  rules; tsz/rules.py), task-refit investigator (tsz/sizefit.py, tsz/control.py), conditional-correction investigator (tsz/correct.py,
  tsz/nested.py, tsz/select.py, compute benchmark). None may read a label before the registration receipt.
- 00:00-00:08Z: coordinator code: tsz/common.py (access layer, subsets), tsz/subsets.py, tsz/jobqueue.py, tsz/lane.py,
  tsz/receipt.py (+ tests pass), tsz/publish_build.py, tsz/admit.py (SOURCE_ADMISSION.json ok: 342 + 36 published files equal to the
  local copy, 2,618 private source copies bitwise identical), tsz/devreaders.py (fresh grid; selftest all_ok; 6 of 16 recipes are
  seed-sensitive), READER_SPLITS.json, DEVGRID_CODE_LOCK.json (00:07:48Z), ACCESS_POLICY.json, EXPOSURE_MAP.md, PROTOCOL.md (draft).
- 00:10-00:18Z: RULES.md / RULES.json drafted by the statistical reviewer (00:06-00:13Z). Coordinator requests sent (nested CONT
  optional; a separate point-only utility label beside C2; T3 record path). Independent verifier assignment started (wave 2; its
  deployment record DEPLOY_W100.json gates the trigger's T3) and the engineering / claims reviewer (wave 1 code review before any
  outcome). Six live agents including the coordinator (the limit). The correction investigator ran label-free compute benchmarks
  through the shared queue (stage 'benchmark', random labels). tsz/evaluate.py trigger / size commands written and structurally
  tested on invented labels (58 fits; no file written to the campaign results); the Stage B short-batch adaptation for S50 is
  registered as PROTOCOL_DEVIATIONS item 3 before any fit. Lineage DEV_REPORT log snapshot at 00:12:45Z: c7829e99... (unchanged
  since the correction-transfer campaign).
- 00:18Z: statistical reviewer's rules final (RULES.json 7bb2d251...; 16 tests pass). 00:19Z registration commit b3a5343f4;
  publication/pcrl-controlled-training-size-v1 at bc7f95d4a pushed; receipt verified with ls-remote at 00:20:07Z (commit 0e44696dc).
- 00:20:13Z / 00:20:27Z: two selftest label reads after the receipt (deviation 4). 00:21:51Z sizefit.py frozen (de5d9cb4), committed
  77a388930.
- 00:22:39Z: first real fits (S100, both versions, 25 CPU-s each). Both reproduce the source A1_PO_16 S0 / S1 packages bitwise
  (T* 168 / 154, identical model files, every role). Coordinator deployment check failed twice before any comparison (a shadowed
  function name in tsz.deploy's CLI; then the packages' pinned code hash after that fix); packages rebuilt as W100_V*_B2 with
  narrower code pins (commits c2b0a4226, 7d12a75a2); the check then passed (bitwise, every refusal).
- 00:23:13Z control.py frozen (fd778afb), committed b30c1c691 with the size lane plan; size lanes launched 00:25Z.
- 00:26:20Z verifier's DEPLOY_W100.json (both versions bitwise, 24 of 24 refusals). 00:27Z trigger evaluated once (unit
  TSZ_TRIGGER_V1): TRIGGER_FIRES, minimum occupation-side ratio 4.88 (v0) / 4.30 (v1), commit 8da1ad661. A1 adjudicated (00:28Z).
- 00:29Z verifier V2 replay agrees with the trigger (values within 1.7e-16). 00:30Z correction code frozen by its investigator
  (correct 46cfb5e5, nested a63fb59e, select 1ffd66cf); committed 368c9ce93 with the coordinator's reader-bank, seed-check,
  conditional-evaluation and claims modules and access amendment 1. 00:33Z both correction lanes launched (13 units per version,
  the last one the descriptive nested CONT); size lanes 40 of about 96 units done. ERRATA E1 (RULES.md T0 wording).
- 00:35-00:38Z: both correction bases (125 CPU-s each) reproduce W100_v bitwise on all roles and the source A1_PO_16 cross-fitted
  out-of-fold base bitwise (max difference 0.0); full-pool directions CONT / LOC / PAIR fitted; releases written (V0 00:38Z, V1
  00:40Z).
- 00:39-00:41Z: engineering review wave 1 received (1 BLOCKER conditional, 1 MAJOR, 9 MINOR); ENG1-01 and ENG1-05 applied before the
  size analysis and any selection; adjudication A2; deviations 5 and 6; commit 937257ecc.
- 00:42-01:18Z: nested 10-fold estimator per version (witness 2 x ~590 CPU-s, routing ~77 s, LOC / PAIR ~250 s each); verifier V3 /
  V4 interim checks pass; deviations 7 and 8 (verifier provenance FAIL on pre-registration benchmarks; queue record defect).
- 01:14Z / 01:17Z: selections (frozen rule, fitting information only): LOC 0.3 and PAIR 0.3 in both versions (first failing
  strength 0.5, binding men's Brier / NLL); committed 17cc919d9 at 01:18Z before any MODEL_SELECT read of a corrected release (only
  two MODEL_SELECT reads before it: the trigger and the verifier's trigger replay). Verifier V5 replay agrees; one nested fold
  recomputed end to end bitwise by the verifier.
- 01:19-01:31Z: reader bank COND committed (66a8cd9ec); fresh readers V0 (1,664 recipe fits, 535 CPU-s) and V1; conditional MODEL_SELECT
  utility: LOC / PAIR / CONT at 0.3 keep the 14 point conditions and the occupation-side reserve in both versions (min ratios
  3.83 / 3.64, 4.08 / 3.78, 4.95 / 4.52); size analysis: SIZE_GAIN_RESOLVED, FULL_ONLY (70% retains), SUPPORTED_ON_THIS_POOL, control
  explains the size gain (NLL, Brier, men's Brier), REPRODUCED_BITWISE; final deployment: 11 packages bitwise, every refusal.
- 01:32-01:33Z: reader-seed check NOT_EFFECTIVE (16 ExtraTrees checks on the income view of the two 6-level reference income wires);
  adjudication requested from the statistical reviewer before any stability label (deviations 9, 10).
- 01:34-01:38Z: the verifier disclosed its own early reader labels (deviations 11, 12); the statistical reviewer ruled A-SC1 at
  01:36:00Z (an amendment; it read no reader outcome); coordinator applied it (A3, deviation 13, commit 02e824972).
- 01:38-01:42Z: wire-cardinality record and claims: grid EFFECTIVE under A-SC1 (literal NOT_EFFECTIVE kept); CORRECTION_RECIPE_CLOSED
  (LOC 0.3 POSITIVE_NOT_STABLE vs W and CONT; PAIR SIGN_CHANGES; PAIR vs LOC NOT_POSITIVE; utility C2 / C2p both arms); the closure is
  the same under the literal reading. Identity checks; decision documents drafted.
- 01:43-01:48Z: custody: live restore dry run V1 failed on the superseded packages (moved to deploy/superseded, deviation 14), V2 and
  the same-device backup B1 restore check passed (all_ok); no external drive attached (off-device custody PENDING); figure SIZE_CURVE.
- 01:52-01:58Z: claims review applied (A5); verifier final report (121 checks; changes adjudicated A1, A4, A6); ERRATA E2; closeout
  receipts (3.60 CPU-h charged including out-of-queue agent work; 152 label reads; no process left). Five agent assignments used,
  at most six live including the coordinator; all agents finished.
