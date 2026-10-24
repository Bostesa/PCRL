# Coordinator log (campaign `pcrl_strong_anchor_release_v1`)

Times are UTC, each copied from a `date -u` read at or before the event.

- **04:02:04Z** Campaign start (first `date -u`). Host Nathans-MacBook-Pro.local (macOS 15.6, arm64), the same host as the storage
  relocation. Internal free space about 125 GiB. `/Volumes` held no YOTUO (BackgroundSyncService Setup, Google Chrome, Macintosh HD).
  The brief (a new campaign, after the storage cleanup) read; memory notes, the relocation index (`~/storage-relocation-20261022/`:
  manifest 156 jobs, 84 relocated / 72 archived and kept; RESTORE.md; verify_restored.py; dot-archive-first order) and the
  input-and-pair records (START_HERE, RESEARCH_DECISION, TABLES, cards) read.
- **04:05-04:10Z** Dependency trace: the cohort exports, U0 / Ucal, T_TREE, the Stage B leaf corrections (JOINT / LOCAL / TASK_REFIT,
  every checkpoint) and the NEURAL_RAW checkpoints live in the protected-tree replication tree (archived and kept, present); the
  references load from the admitted mirrors in the source worktree's `.private/`. Relocated items the code mentions (occupation-
  headroom, income-headroom and autonomous-method-development worktree `.private` trees; frozen-hybrids `.private`) are read through
  those mirrors. No archive restore needed.
- **04:11:05Z** New isolated worktree `.claude/worktrees/pcrl-strong-anchor-release-v1` created with `git worktree add` on the new
  branch `research/pcrl-strong-anchor-release-v1` from the input-and-pair tip 9c9f95385; the session switched into it.
- **04:11:26-04:11:31Z** Working cache: the nine inherited `.private/<campaign>` trees cloned (APFS `cp -cRp`) from the source
  worktree (6.2 GB logical, about 0.01 GB allocated). Reference loading smoke-tested label-free.
- **04:15-04:20Z** `sar.anchors table`: the label-free anchor table from the saved IPR1_MAIN grid (GRID.json, BOOT.npz) and the
  MODEL_SELECT utility records. Anchor selection frozen (A: SUCC; B: NEURAL_RAW; `anchors/ANCHOR_SELECTION.md`). `sar/` derived from
  `ipr/` by a mechanical rename. Design brief written. Commit 9dc59946b (before any fit or label read).
- **04:23Z** Investigators A and B launched (independent agents, label-free, 600 CPU-s cap each, brief `DESIGN_BRIEF.md`).
- **04:25:08Z** YOTUO present (not mounted by this session; `/Volumes` was modified at about 04:03Z): `/dev/disk6s1`, ExFAT (FSKit),
  USB, volume UUID 3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D (the manifest's), 1.0 TB, 582,460,375,040 B free, writable.
- **04:25-04:26Z** `sar.admit run` `ok` on the second run (the first listed two input-and-pair files changed after its publication:
  deviation 13 and the publication log lines appended by the tip commit 9c9f95385, 8 and 11 lines added, none removed; recorded as
  explained). Working cache verified file by file against the source trees (every file's size and SHA-256 equal). Access policy
  written.
- **04:26:42Z** The drive's relocation manifest is byte-identical to the local copy (`cmp`; SHA-256 f904c871...4efc). Internal free
  space 126.1 GiB.
- **04:27Z** Statistical reviewer launched (SAR1 rules; brief `REVIEWER_BRIEF.md`; 900 CPU-s cap).
- **04:28Z** `sar.devreaders selftest` all_ok (invented data); `manifest` (READER_SPLITS.json, IPR1 overlap added) and `lock`
  (DEVGRID_CODE_LOCK.json, torch interpreter) written.
- **04:29:19Z** Queue opened (`sar.jobqueue init`).
- **04:30-04:40Z** PROTOCOL.md, EXPOSURE_MAP.md, FAILURE_MAP.md and the card template written; absolute personal paths removed from
  the gated files so that the sanitized receipt equals the research copy.
- **04:39:43Z** Storage admission record written (`sar.storage admit`): drive identity ok, external index equal to the local copy, 135.4 GB
  internal free. SAR1_REFS bank spec written (label-free; its output to be read only after the cards are committed).
- **04:48-04:52Z** Investigator B's proposal FB_PFD and hand-back received (saved verbatim); 04:52-04:57Z investigator A's FA_TFR.
  Adjudication: both PROCEED with required changes (FA: correct the CALIBRATION exposure statement, two-point CALIBRATION CTRL
  bracket, CONT declared; FB: add the brief's untargeted CTRL = DQ of NEURAL_RAW, RAW-J as the ALT path, CONT declared).
- **04:56-05:07Z** Statistical reviewer: SAR1 rules frozen (23 then 25 tests pass; mutation checks 24 / 24 then 33 / 33); its report was
  returned as text (harness) and saved verbatim. The coordinator answered its four questions (UM point-margin clause adopted; targeting
  gate kept; budget 8 / 12; few-valued bound 320) and asked for the ALT path and declared selection data (revision 2, 05:07Z).
- **05:00-05:08Z** Admission extended: the protected-tree replication inputs FB reads (U0_MODEL.pt and all 15 NEURAL_RAW checkpoints per
  version equal their seed records) and SUCC's 14 frozen component files FA reads (equal to the mirror's admission record); FA's
  label-free preflight passes for both versions. The `calibrate` operation widened (CALIBRATION task labels for a card's selection and
  matching; never SEX). Claims builder adapted to the SAR1 record (START / CONT / CTRL / ALT).
- **05:05-05:09Z** Both investigators' revisions received (saved verbatim as HANDBACK_2.md). Cards FA_TFR_v1 and FB_PFD_v1 written from
  their drafts with a coordinator declaration block (cont / ctrl origin / alt / selection data); protocol completed with the reviewer's
  findings 1-22 and the adjudication. Selftests rerun: FA 24 / 24, FB all ok.
