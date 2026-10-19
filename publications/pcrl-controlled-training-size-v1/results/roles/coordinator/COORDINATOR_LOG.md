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
