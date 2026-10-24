# Coordinator log (campaign `pcrl_strong_anchor_release_v1`)

Times are UTC, each copied from a `date -u` read at or before the event.

- **04:02:04Z** Campaign start (first `date -u`). Host <owner host> (macOS 15.6, arm64), the same host as the storage
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
- **05:10:24Z** Protocol completed. **05:10-05:11Z** Registration committed (890ae7a67, with both cards); receipt ede0755fb (a root commit
  of 89 sanitized files; no gated file changed by the sanitizer) pushed to `publication/pcrl-strong-anchor-release-v1` at 05:11:33Z and
  verified on the remote at 05:11:39Z (`sar.receipt`); `sar.common.registered()` true. No label read before this point.
- **05:12-05:18Z** Registered fits through the queue (two lanes): SAR_FIT_FB_V0 (74 CPU-s), SAR_FIT_FB_V1 (54), SAR_FIT_FA_V0 (196),
  SAR_FIT_FA_V1 (195); reference utility (SAR_REF_EVAL_V1) and DDR_PTC_R1 utility in the free slot; the SAR1_REFS grid queued.
  - FB_PFD_v1: version 0 k* = E1 (CAND spend over the cross-fitted RAW-J E2 +0.00016, se 0.0004; E2 fails income Brier 0.0053 and
    income NLL 0.0066); version 1 NO_FEASIBLE_STRENGTH (E1 fails the income components). Income share of CAND's NLL spend about 1.0
    against 0.40-0.53 for the matched controls: the card's contradiction 1 holds. Bitwise checks hold (RAW-J E2 = NEURAL_RAW, U0 recipe
    = U0). Five version-0 configurations were saved (FB_PFD, FB_PFD_TASK, FB_PFD_DQ_LO, FB_PFD_DQ_HI, FB_PFD_RAWJ_HI).
  - FA_TFR_v1: NO_FEASIBLE_STRENGTH in both versions at rho = 0.3 (the smallest grid point): (a) CALIBRATION income Brier +2 SE bound
    0.0043 / 0.0056 against 0.0030 (occupation NLL -0.016 / -0.014 over SUCC); (b) MODEL_FIT out-of-fold men's / women's occupation Brier
    over the lambda = 0 LO point +0.0073 / +0.0144 and +0.0139 / +0.0110 against 0.0005 / 0.0050. Nothing saved.
- **05:20Z** Both registered readings recorded (`claims/FA_REG`, `claims/FB_REG`): D1 NO_FEASIBLE_STRENGTH, class TASK_INFEASIBLE (plans
  committed first, c302b2bb5). Both investigators asked for one substantive revision each, designed from the MODEL_FIT / CALIBRATION
  records only (REVISED_, no MODEL_SELECT outcome of either family has been read).
- **05:32:13Z** Card FB_PFO_R1 committed (8f18b979a; occupation-only coalition objective, income frozen at START; 10-step checkpoints;
  occupation-only DQ CTRL and the RAW-J ALT, upper points only). Fits SAR_FIT_FB_R1_V0 / _V1 (66 CPU-s each): k* = S50 in both versions,
  CAND spend over START +0.0020 (se 0.0003) / +0.0019 (0.0004), income bitwise START; CTRL DQ_HI and ALT RAW-J HI saved (S160 / S130).
- **05:37:58Z** Card FA_TFRO_R1 committed (a21aab93b; occupation-only residual on SUCC, income SUCC's own; budgets SUCC's margins to the
  caps; CALIBRATION-average plus MODEL_FIT-contrast guard estimate; rho 0.3-10 scanned upward). Its fits started.
- **05:38Z** FB_R1 plan (committed a21aab93b) and bank SAR1_FB_R1 (21b25d188) committed before any MODEL_SELECT computation of the
  reading; its evaluation, grid and claims lane started. Both revision cards being committed, the SAR1_REFS outputs were opened for the
  first time (references only; registered cells): RETIRED_B3CTL 0.6086, SUCC 0.6166, B2TWIN 0.6219, DDR_PTC_R1 0.6257, H_LJ 0.6308,
  NEURAL_RAW 0.6342, TWJ 0.6413, U0 0.6506; every level about 0.01 above IPR1_MAIN's (new splits and seeds), the order unchanged.
- **05:39Z** FB_R1 MODEL_SELECT utility: FB_PFO point-feasible in both versions (max 0.601 / 0.785 of cap; men's guard +0.0030 / +0.0039,
  reserve 1.10 / 0.63); paired total NLL over NEURAL_RAW +0.0039 / +0.0038 (MODEL_FIT out-of-fold +0.0020 / +0.0019: the spend about
  doubled from MODEL_FIT to MODEL_SELECT); against the CTRL upper point CAND spends more on MODEL_SELECT (+0.0011 / +0.0018), so the CTRL
  frontier has no bracket; against the ALT upper point CAND spends less (-0.0009 / -0.0006).
- **05:45Z** FA_R1 MODEL_SELECT utility: FA_TFRO_R1 point-feasible in version 0 (max 0.141 of cap; men's guard +0.0007 against SUCC's
  +0.0044; reserve 2.16; occupation NLL -0.0036 over SUCC) but FAILS G1 in version 1 (men's guard +0.0099 against the 0.005 cap; men
  +0.0060 and women -0.0085 summed Brier over SUCC), although its MODEL_FIT / CALIBRATION guard estimate at rho 0.6 read men -0.0039
  (se 0.0019): the critic-driven residual shifted the occupation cost onto men on held-out people in a way the estimate did not see.
- **05:45:38Z** FB_R1 decisions (`claims/FB_R1`, grid effective): D1 TASK_FEASIBLE; UM MARGIN_NOT_RESOLVED; D2 PARTIAL_START+CONT_ONLY
  (vs NEURAL_RAW +0.0125, se 0.0056, recipient gates pass; vs CONT +0.0143, se 0.0055; CTRL frontier NO_BRACKET: CAND spends 0.0014 nats
  more than the DQ upper point on MODEL_SELECT; ALT frontier +0.0012, se 0.0074, not separated from NEURAL_RAW's own RAW-J recipe
  continued to matched spend); D3 NOT_SUPERIOR (vs H_LJ +0.0091, se 0.0096; SUCC -0.0050; RETIRED_B3CTL -0.0130; B2TWIN +0.0003); D4
  NOT_MET; class ADAPTIVE_R1__FEASIBLE_NO_RELIABLE_GAIN.
- **05:50:08Z** FA_R1 decisions (`claims/FA_R1`, grid effective): D1 CAND_INFEASIBLE (version 1 fails the men's occupation guard on
  MODEL_SELECT); D2-D5 NOT_ASSESSED_TASK_INFEASIBLE; class ADAPTIVE_R1__TASK_INFEASIBLE. Descriptive only (no claim): registered-cell
  pair AUC by version FA_TFRO_R1 0.6027 / 0.6117 against SUCC 0.6178 / 0.6155, RETIRED_B3CTL 0.6083 / 0.6089, H_LJ 0.6315 / 0.6301 and
  NEURAL_RAW 0.6303 / 0.6380; occupation view 0.5845 / 0.5795 (SUCC 0.5743 / 0.5792). The feasible version 0 (rho 1.0; men's guard
  +0.0007, reserve 2.16) is the lowest pair of the campaign so far; version 1 (rho 0.6) shifted occupation cost onto men.
- **05:51-05:53Z** Investigator A asked for FA's second revision (ADAPTIVE_R2: control the men's guard robustly without MODEL_SELECT
  tuning; at most 6 new specifications). FB stops after R1 (its reading equals the anchor's own operation run further); investigator B
  asked to design the one replacement family FS (START SUCC: the income output designed for the pair with JOINT occupation frozen;
  adaptive as a whole; at most 8 new specifications). FB_R1's D2R robustness lane started (bank SAR1_FB_R1_ROB). Independent verifier
  launched on the readings so far (brief VERIFIER_BRIEF.md). Specifications used: 19 of 36; CPU 0.94 h; elapsed 1.84 h.
- **06:08:15Z** Card FA_TFRB_R2 committed (d3b453d1d; ADAPTIVE_R2: the deployed occupation residual is the cross-fit bag on which the
  guard contrast is measured; the contrast's SE includes the between-fold SD). Fits: version 0 rho* = 0.6 (CTRL LO s 0.4, HI s 0.1),
  version 1 rho* = 1.0 (CTRL LO s 0.25, HI = the origin); 5 new specifications (24 of 36 used).
- **06:12:18Z** Card FS_CPI_v1 committed (3a8ee012c; the replacement family: an income-only residual on SUCC against the whitened pair
  best response with JOINT occupation frozen and a KL anchor to H3; CALIBRATION income spends against the cross-fitted H3; CTRL = DQ of
  SUCC's income, one hedged upper point; ALT = SUCC's own LJDIR transplant continued, two points). Its fits started.
- **06:14Z** FA_R2 plan and bank SAR1_FA_R2 committed (a85bd035a) before any MODEL_SELECT computation of FA_R2; its reading lane started.
- **06:13-06:14Z** FS_CPI fits: version 0 NO_FEASIBLE_STRENGTH (lambda 0.025 already fails the CALIBRATION income accuracy-loss bound,
  +0.0027, se 0.0013, against 0.0040; NLL +0.0006 and Brier +0.0004 were inside), version 1 lambda* 0.05 (4 configurations saved). The
  reading has no candidate in both versions: `claims/FS_CPI` ADAPTIVE_FS__NO_FEASIBLE_STRENGTH / TASK_INFEASIBLE. 28 of 36 specifications
  used.
- **06:14-06:17Z** FA_R2 utility: FA_TFRB_R2 point-feasible in both versions (max 0.012 / 0.698 of cap; men's guard -0.0008 / +0.0035
  against SUCC's +0.0044 / +0.0039; occupation NLL -0.0111 / -0.0013 over SUCC). The evaluation unit stopped on the reused CONT's
  write-once record after writing the candidate's record (deviation 8); the lane resumed.
- **06:19:11Z** FB_R1 D2R (`claims/FB_R1_D2R`; R3 / R4 splits and k2 / k3 seeds; refit genuinely effective): ROBUSTNESS_NOT_CONFIRMED.
  CONT confirmed (+0.0140, se 0.0064); NEURAL_RAW +0.0119 (se 0.0067), point only on the robustness cells; ALT changes sign (+0.0012
  registered, -0.0012 robustness): carried into the headline; H_LJ +0.0093, SUCC -0.0030, RETIRED_B3CTL -0.0123 (robustness means).
- **06:21:28Z** FA_R2 decisions (`claims/FA_R2`, grid effective, plan_sha256 recorded): D1 TASK_FEASIBLE (FA_TFRB_R2 meets all 28
  conditions at the point); UM MARGIN_GAIN_PROFILE_NOT_RETAINED (the binding men's occupation Brier improves on START, pooled -0.0029,
  se 0.0013, by version -0.0053 / -0.0004, read on adaptively reused MODEL_SELECT people; the occupation view worsens in version 0,
  +0.0252, and improves in version 1, -0.0141; pair CAND - START +0.0047 / -0.0179); D2 PARTIAL_CONT_ONLY (CONT +0.0432, se 0.0183,
  resolved; the CTRL frontier +0.0066, se 0.0140, NOT_SEPARATED_FROM_TASK_SPENDING; START not resolved); D3 NOT_SUPERIOR (H_LJ +0.0208,
  se 0.0145; NEURAL_RAW +0.0241, se 0.0191; SUCC +0.0066; B2TWIN +0.0119; RETIRED_B3CTL -0.0014); D4 MATERIAL_0020_POINT_ONLY (both
  material references above 0.020 at the point, neither resolved); class ADAPTIVE_R2__FEASIBLE_NO_RELIABLE_GAIN. FA's revisions are used.
- **06:22:29Z** D2R robustness bank SAR1_FA_R2_ROB written for FA_R2 (repetitions R1-R4, reader seeds k0-k3; the registered decisive
  comparators CONT and FA_R2_CTRL_HI, START SUCC and the references NEURAL_RAW, H_LJ, RETIRED_B3CTL); no new specification.
- **06:27Z** FA_R2 D2R lane started (`lanes/D2R_FA_R2.json`, after commit e58c76a74). **06:30Z** Verifier wave 2 requested (FA_R2;
  FS_CPI selections; FB_R1 D2R; FA_R2 D2R once its unit ends; the plan-hash sidecars). Investigator B asked for an FS revision card
  (diagnosis: CALIBRATION income accuracy-loss noise from decision flips binds at the first strength; suggested change: a
  sign-preserving income residual) or a stop memo, with predictions from fitting data only.
- **06:35Z** Investigator B's stop memo (`roles/investigator_b/HANDBACK_FS_STOP.md`, verbatim): predicted MODEL_SELECT pair benefit
  of a decision-preserving FS revision over SUCC about 0.005 (80% interval 0.000-0.011) at its budget-feasible strength (Brier binds
  one grid step later at most); on MODEL_FIT, SUCC's own beta transplant (ALT) reproduces all but 0.001-0.004 of CAND's pair drop at
  matched spend; predicted D2 probability about 0.03. Coordinator decision (06:37Z): **FS is closed** (no revision card; no
  specification spent). With FA's two revisions used, FB stopped after R1 (its reading equals NEURAL_RAW's own operation run further)
  and FS closed, no further release is fitted: 28 of 36 specifications used (8 candidates, 3 task twins, 17 matched controls); the
  remaining 8 are not spent. The rest of the window goes to follow-through (FA_R2 D2R), verification, packaging, custody and closeout.
- **06:38:00Z** FA_R2 D2R (`claims/FA_R2_D2R`; R3 / R4 splits and k2 / k3 seeds; refit genuinely effective; grid valid):
  ROBUSTNESS_NOT_CONFIRMED. CONT confirmed (robustness mean +0.0374, se 0.0156); START SUCC +0.0034 (0.0139) and CTRL_HI +0.0040
  (0.0134) point only; beside: NEURAL_RAW +0.0183 (0.0177) and H_LJ +0.0158 (0.0135), both below the 0.020 material level at the
  point on the robustness cells (registered +0.0241 / +0.0208); RETIRED_B3CTL -0.0059. No sign change. The verifier edited its code
  during SAR_GRID_SAR1_FA_R2_ROB_V1 (recorded drift; deviation 11's class).
- **06:42-06:44Z** Reader custody (5 readers of SAR1_FA_R2 S1R1k0 and SAR1_FB_R1 S0R2k1 rescore bitwise and equal the grid cache) and
  13 deployment packages (the point-eligible releases FA_TFRB_R2 and FB_PFO in both versions and their informative controls; each
  binds every repository module its deployment function loads): bitwise with the evaluated wires on the full DEV_REPORT batch through
  the API, CSV and encoded entry points; every refusal probe refused; probability validity (worst row-sum error 4.4e-16).
- **06:46-06:48Z** Custody scope from a label-free open-file trace; FB's fit-time replication inputs snapshotted (hash-checked);
  rehearsal backup (796 MB) restored from the backup alone, both interpreter passes clean. Campaign summary (RULES.md 7) and tables
  (the first tables unit failed on the no-candidate readings and was rerun). Verifier wave 2: 124 checks, 124 PASS.
- **07:00-07:09Z** Draft closeout documents; claims review by the statistical reviewer (12 required corrections, 11 recommended) and
  verifier wave 3 (deployment and publication scan: 109 checks, 102 PASS, 7 FAIL kept: six FB packages differ at the ULP level on
  50-row requests with identical decisions, and the owner's name appears in the relocation folder name in the public registration
  receipt).
- **07:13-07:30Z** Deployment batch-dependence sweep (13 packages, batch sizes 1 to 5,943: worst 4.4e-16, no decision change; a first,
  unbounded attempt was stopped after about 10 minutes); deviation 14 states the 6e-16 tolerance. All public branches searched (read
  only): only this campaign's receipt among the publication branches carries owner identifiers (deviation 13); the publication
  builder now redacts runtime-derived owner identifiers and refuses any remainder (dry build 248 files, no problem). Every claims-review
  correction applied to RESEARCH_DECISION, START_HERE, HANDOFF, QUICKSTART and ENDPOINTS.
- **07:34-07:41Z** Final same-device backup (tip 4ae4809bf, 803 MB) restored from the backup alone; its copy on YOTUO (identity re-checked) checked WARM; resource accounting and access receipt; sanitized publication bca6e5b28 (260 files, child of the receipt; owner identifiers redacted; gated files equal to the receipt except PROTOCOL.md's one redacted line) pushed at 07:40:32Z and verified on the remote at 07:40:40Z; queue closed at 07:41:25Z; final accounting 3.66 h, 2.28 CPU-h.
