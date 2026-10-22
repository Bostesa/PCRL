# Coordinator log (campaign `pcrl_input_and_pair_release_v1`)

Times are UTC, each copied from a `date -u` read at or before the event.

- **01:19:06Z** Campaign start (first `date -u`). The task brief repeats the previous campaign's brief; the previous campaign
  (`pcrl_dual_design_release_v1`) was complete and published (e30cf1c6b). This is a new campaign: memory notes, the finite-solver-
  adequacy and finite-routing records, the inherited failure map and the dual-design records read again. YOTUO not attached.
- **01:27:15Z** Branch `research/pcrl-input-and-pair-release-v1` created from c577424c0 (the dual-design research tip).
- **01:30:24Z** Design brief written (`DESIGN_BRIEF.md`): Designer A's space = before the release (inputs, training rows, targets,
  objective), Designer B's space = the joint release map for the linked pair; the dual-design idea family (STD, IPW-STD, PTC, ZTILT)
  added to the closed list; the release-only explanation stated as an alternative every design must answer. Both designers launched
  just after (independent agents, label-free, 400 CPU-s cap each).
- **01:33-01:37Z** `ipr/` derived from `ddr/` by a mechanical rename (scratchpad script; family modules not copied; new seed offset
  900, bootstrap base 2026102300, split tags IPR1); `ipr.admit run` `ok` (dual-design publication e30cf1c6b pinned, parent = its
  receipt, two post-publication result differences explained; ddr private saved releases hash-checked); access policy; queue opened
  01:35:28Z; release, famkit, ctlkit, devreaders and evaluate selftests pass; READER_SPLITS.json written (DDR1 overlaps added).
- **01:38:33Z** Protocol draft written (three gating control paths SBC / SPEND / RO; D5 planned for a jointly designed family;
  planned resolution about 0.02-0.025). Statistical reviewer launched just after (pre-registration review and the IPR1 rules).
- **01:39-01:41Z** The generic release-only path RO (the dual-design ZTILT law on a 1.1-ratio grid) and a 1.1-ratio DQ grid added to
  `ipr/ctlkit.py`; selftest passes. A correction note added to the design brief: TWJ's income is HC_STRONG, not DA_TWIN's.
- **01:42-01:50Z** `ipr/ctl/twin.py` (the shared twin CTL_TWIN, card CTL_TWIN_v1 drafted), `ipr/plan.py` and `ipr/claims.py`
  generalized to three control paths and joint-design controls; `ipr/paths.py` (shared DQ / RO bracket points with a fineness
  record); read-only support for the dual-design releases as beside references (`ipr.evaluate source_ddr`, bank prefix `DDR:`);
  reference bank IPR1_REFS and lane REF_EVAL pre-written. Infrastructure commits 091d0c88f (01:47:39Z) and a6f3bb1dc.
- **01:53Z** Designer A's proposal (SFG, shared SEX-aware field generalization) and hand-back received; the hand-back is saved
  verbatim (`roles/designer_a/HANDBACK.md`). The designer predicts no separation from its matched controls (release-only tilt about as
  good at equal spend) and recommends a lean run as the input-side closure test.
- **02:02Z** Designer B's proposal (CSF-T, coordinated cell-targeted selective flattening) and hand-back received and saved verbatim.
  The designer predicts no separation and no joint-design advantage, and recommends a reduced allocation. A note was sent to the
  statistical reviewer asking that the rules give a defined label for a SEX-blind release-only operation and for D5 controls that
  collapse by construction.
- **02:03Z** The designers' label-free CPU charged (160 + 150 CPU-s). Challenger launched on both proposals (brief
  `CHALLENGER_BRIEF.md`).
- **02:03-02:07Z** Draft family modules written with invented-data selftests passing: `ipr/ia/sfg.py` (menu checked label-free on
  the admitted MODEL_FIT features; full stage selftest on invented labels) and `ipr/ib/csf.py` (operation algebra). Not yet cards.
- **02:08-02:13Z** The statistical reviewer's hand-back arrived (RULES.md completed 02:08:15Z by its own `date -u`; 28 of 28 rule
  tests; rule-file hashes checked). The harness refused the agent's own write of its report, so the report and the hand-back were saved
  verbatim (`REPORT_PREREGISTRATION.md`, `HANDBACK_PREREGISTRATION.md`). All REQUIRED findings 1-10 and advisories 12-17 adopted into
  PROTOCOL.md in the reviewer's wording (section 13); grid code lock written with the locked torch interpreter (02:12:50Z); protocol
  completed 02:13:01Z.
- **02:13:46Z** Registration committed (adfe3e75a). **02:14:11Z** receipt e909fb8c0 (a root commit of 77 sanitized files; no file
  changed by the sanitizer) pushed to `publication/pcrl-input-and-pair-release-v1`; **02:14:15Z** verified on the remote
  (`ipr.receipt`); `ipr.common.registered()` true. No label read before this point.
- **02:15Z** CTL_TWIN fitted through the queue (2 configurations; no label read): its 12 released arrays per version are bitwise the
  dual-design DA_TWIN's (exact alias of the beside reference DDR_TWIN). The challenger's review was still running; no MODEL_SELECT read
  until both mechanism cards are committed.
- **02:25Z** The challenger's hand-back arrived (review `roles/challenger/REVIEW_DESIGNS.md`, 21 findings, 13 REQUIRED; hand-back saved
  verbatim). Verdicts: IA PROCEED_WITH_REQUIRED_CHANGES; IB REDIRECT to the uniform SEX-blind CSF; the cross-recipient keyed swap
  named as the one replacement family worth the reserve.
- **02:26-02:36Z** Adjudication (`PROTOCOL_AMENDMENT_1.md`, from 02:36:13Z): IA at S3 with rung 1, intermediate menu levels, the
  permuted-SEX SBC, BLIND as the ablation, RO = GENERIC_ZTILT as the declared fallback with the substituted ladder computed
  descriptively, no PROJ / ATTR; IB redirected to CSF (operation_sex_blind) at S3 with CSF-T and IND as ablations and D5 declared
  collapsed (both orders = JOINT); the DQ tag made family-independent and the shared SPEND / RO points saved once under CTL (card
  CTL_PATHS_v1). Code rewritten accordingly; invented-label stage selftests of both families pass. At 02:36Z the redirected Designer B
  was asked to specify the replacement family FS_SWAP.
- **02:38:59Z** Cards IA_SFG_v1, IB_CSF_v1 and CTL_PATHS_v1 committed with their code (9b70c5d77). **02:39:08Z** both versions' fit
  lanes started (IA then IB in each version's lane); the first label reads (MODEL_FIT, operation fit, card IA_SFG_v1).
- **02:41-02:43Z** A new module `ipr/roles.py` was written into the package while the fit units ran and moved out again about two
  minutes later (to keep the package unchanged during the units); the units did not import it. Its final version was restored after
  the fits (below).
- **02:44Z** Designer B's FS_SWAP hand-back (saved verbatim): in the plug-in world the swap is no better than occupation-only
  flattening or CSF at equal spend; decision `roles/coordinator/FS_SWAP_DECISION.md`: not run, the replacement slot kept.
- **02:46-02:47Z** Fits done (IA 436 / 462 CPU-s, IB 20 / 20 CPU-s; 23 configurations saved in all). IA: the SEX-ordered ladder
  starts MAR2 (V0) / AGE5, MAR2 (V1); k* = 1 (MAR2) in V0 and 2 (AGE5 + MAR2) in V1, MODEL_FIT out-of-fold spend -0.0001 / -0.0002
  nats (both below the twin); the next rung (MAR suppressed) fails S3 on income NLL (1.07 / 1.29 of the budget) and income accuracy
  (1.95 / 1.69). SBC (permuted SEX): V0 brackets at rungs 1 / 2, V1 at TWIN / rung 1; SPEND and RO: TWIN and the first grid point
  (CAND spends less than the twin, so the fineness check does not apply). The descriptive substituted ladder equals SFG's rung k* in
  V1 but not in V0. IB: pi* = 0.075 in both versions (spend 0.0108 / 0.0097; the men's occupation guard and income NLL bind at 0.10);
  SPEND brackets inside the 25% bands. MODEL_FIT out-of-fold readers (descriptive) show pair changes of -0.001 to -0.005 against the
  twin.
- **02:47Z** Revision triggers checked on MODEL_FIT records only: IA R2's trigger holds in both versions (the next rung overshoots its
  binding component by 95% / 69%) but its binding step, MAR suppression, has no predeclared half beyond MAR2, which the registered
  menu already contains, so R2 cannot change the release; IA R1's trigger fails in version 0 (the women's occupation guard keeps 45% of
  its slack, below the 50% required); IB's R1 (men's guard binds, not income alone) and R2 (pi* > 0.025) triggers do not hold. No
  revision card is written before the registered readings.
- **02:48-02:50Z** Pending code applied (`ipr/roles.py` role members for the shared CTL points named per version; `ipr/plan.py` from
  the fit records; `ipr/claims.py` with the card declarations and the fineness record; evaluation of role members; report and rich
  readers for role members); selftests pass. `ipr.plan registered --bank IPR1_MAIN` wrote SHARED_ROLES.json (02:49:09Z), the bank
  (25 members), PLAN_IA_REG / PLAN_IB_REG and the rich spec; evaluation, grid, rich and claims lanes written. All before any
  MODEL_SELECT read.
- **02:50:32Z** First MODEL_SELECT read (IPR_EVAL_TWIN_V1). The registered grids IPR_GRID_MAIN_V1 / V0 started at 02:50:32Z /
  02:50:34Z. **About 02:51Z** the verifier was started with VERIFIER_BRIEF.md (independent recomputation of the registered readings).
  **02:51:45Z** robustness bank IPR1_ROB committed (4773d7126), before any registered-grid output.
- **02:54-02:55Z** The robustness lanes were withdrawn before start (PROTOCOL_DEVIATIONS item 4).
- **03:09:53Z / 03:11:28Z** Registered grids done (1155 / 1248 CPU-s). Both record the verifier's code files as changed during the
  run; the files were written 02:58-03:07Z into the hashed `roles/verifier/code/` (item 6). The utility evaluations ran from 03:09:54
  to 03:10:17Z (twin, IA, IB, role members, references, dual-design sources).
- **03:11:35-03:11:48Z** Registered claims IA_REG / IB_REG and the data-rich reader scores.
  - Both readings: TASK_FEASIBLE, D2 NOT_SEPARATED, D3 NOT_SUPERIOR, D4 NOT_MET.
  - **03:11:54Z** robustness grids IPR_GRID_ROB2_V0 / V1 started.
  - **03:12:32-03:12:39Z** strongest-release record (IA_SFG, IB_CSF), beside comparisons, D2R plans and campaign summary (0 of 2 D2
    looks met).
  - **03:13:22Z** commit 92312acdb.
  - **03:14:39Z** deployment lane written and queued behind the robustness grids.
- **03:24Z** The verifier's report arrived; it is saved verbatim as `roles/verifier/VERIFIER_REPORT.md`. Every check passes and every
  decisive number replays exactly; it raised five observations (O1-O5), none of which changes a label.
- **03:25-03:35Z** Diagnosis of the registered readings (MODEL_SELECT grid per view; MODEL_FIT ladder statistics; Designer A's
  label-free plug-in).
  - **The leakage sits in the occupation view.** The twin's pair recovery (0.6480) is carried by its occupation view (0.6443; income
    0.5498).
  - **IA:** its candidate moved occupation by 0.0035, while its refitted income model became more recoverable (0.5600). The shared
    ladder was stopped by income at MAR1 (dT_inc 0.0048 / 0.0043 against dT_occ 0.0022 / 0.0028).
  - **IB:** it moved occupation by 0.008, as SEX-blind quantization does at matched spend.
  - **Decision:** one adaptive IA revision, occupation-side generalization (IA_OSG, card IA_OSG_R1), addressing the three observed
    failures. Its gating controls are restricted to the occupation output so that they spend where it spends.
  - **No IB revision.** Its declared triggers did not hold. Its targeting (CSF-T) and coordination (IND) ablations showed no gain,
    and its binding constraint (the men's occupation guard at pi = 0.10) would bind any deeper collapse.
  - The replacement slot stays unused (FS_SWAP decision).
- **03:35-03:48Z** Card IA_OSG_R1 written, with registered predictions (P(D2 met) = 0.03). Its code (`ipr/ia/osg.py`, the `osg`
  payload, occupation-restricted path points, revision readers, `ipr.plan revision`) is held outside the package until the queue is
  idle, so that no running unit sees a code change; on a scratch copy of the package its invented-data selftests pass. Lanes
  FIT_R1_V0 / V1 and D2R_REG written.
- **04:04:45Z / 04:07:29Z** Robustness grids IPR_GRID_ROB2_V1 / V0 done (3153 / 3318 CPU-s). Both record `v6_seeds.py` (verifier code
  written at 03:20:24Z) as changed during the run (item 6).
  - **04:04:46-04:05:41Z** Deployment lane: 10 packages stored, built and checked, all bitwise and all refusals.
  - **04:04:50-04:04:59Z** The D2R lane ran early on the slot freed by the version-1 grid, while version 0 still ran. Its two
    records read D2R NOT_RUN. They were moved to `claims/superseded/` at 04:07:58Z, and the readings were rerun (IPR_D2R2_*,
    04:07:58-04:08:03Z): both ROBUSTNESS_NOT_CONFIRMED, refits genuinely effective (item 9).
- **04:08-04:11Z** With the queue idle, the revision code and the closeout fixes were applied to the package. The selftests passed:
  - invented-data selftests;
  - the revision's end-to-end stage test on invented labels (16 CPU-s);
  - SFG's stage test (64 CPU-s);
  - 28 / 28 rule tests.
  - **04:11:44Z** card IA_OSG_R1 and its code committed (221a91f21).
- **04:11:55-04:21:00Z** Revision fits (508 / 540 CPU-s).
  - Recomputed ladder statistics equal IA_SFG_v1's in both versions.
  - k* = 2 / 4: marital status suppressed for the occupation recipient (version 0); ten-year age bands, three hours bands and marital
    status suppressed (version 1).
  - 16 configurations, 39 of 48 in all.
  - **04:21:14Z** `ipr.plan revision` wrote PLAN_IA_R1, bank IPR1_R1 and the rich spec, before any grid output; committed at 04:21:32Z
    (b7dc5c698).
- **04:21:44-04:21:48Z** Revision utility (4 CPU-s).
  - IA_OSG is point-feasible in both versions; reserve 4.02 / 1.39.
  - IA_OSG_RO_HI fails G1 in version 1 (men's occupation Brier guard at 1.005 of the cap).
- **04:21:48-04:28:32Z** Revision grids (392 / 400 CPU-s).
- **04:28:50-04:28:55Z** Revision claims and rich readers (ADAPTIVE_R1): TASK_FEASIBLE, D2 NOT_SEPARATED, D3 NOT_SUPERIOR, D4 NOT_MET.
  - vs the twin +0.0017 (0.0062): the occupation view fell to 0.6351, but the pair hardly moved.
  - SPEND: no bracket. RO: -0.0072 (0.0064).
  - Strongest release, beside records and the summary followed: IA_OSG is IA's strongest by 0.0001 over IA_SFG; 0 of 3 D2 looks met.
  - **04:30:28Z** robustness bank IPR1_ROB_R1 and its D2R plan committed (c00c56910).
- **04:30:39-04:50:40Z** Refit grids of IA_OSG (1184 / 1194 CPU-s, versions 1 / 0). **04:50:49-04:50:52Z** its D2R: ROBUSTNESS_NOT_CONFIRMED, refits genuinely
  effective.
- **04:51:04Z** The first reader-custody unit failed on a composite winner recipe (1.2 CPU-s). It was rerun in base-recipe cells
  (04:51:42-04:51:47Z): all_ok (item 10).
- **04:52:24Z** D2R, custody, tables and first document drafts committed (508a7c3c2); drafts updated at 04:53:48Z (5c497a958).
- **04:53:48Z** The verifier was resumed with VERIFIER_ADDENDUM_BRIEF.md, and the statistical reviewer with CLOSING_REVIEW_BRIEF.md.
  The YOTUO drive was not attached at 04:37:17Z (nothing mounted).
- **05:08Z** The verifier's addendum arrived (checks A1-A8): all pass and every label reproduces exactly. It is saved verbatim as
  `roles/verifier/VERIFIER_ADDENDUM_REPORT.md`. It found one more launch-drift record (IPR_DEPLOY_STORE_CTL_TWIN_V0_V1, `v6_seeds.py`),
  now added to item 6.
- **05:09Z** The statistical reviewer's closing statement arrived (labels all follow; corrections C1-C19; the required headline). It is
  saved verbatim as `roles/statistical_reviewer/CLOSING_STATEMENT.md`.
  - The headline and corrections were applied by 05:17Z.
  - C2 withdraws the reading that IA_SFG's leakage moved to the income view: a one-repetition reader-selection sensitivity. With it
    goes IA_OSG's card failure F2.
- **05:30Z** The claims review arrived (corrections R1-R23, advisories A1-A18). It is saved verbatim as
  `roles/engineering_review/REVIEW_CLAIMS.md`.
  - **R3:** two refit-cell numbers had no record. They are now written by the committed script `roles/coordinator/osg_spend_refit.py`
    to `claims/IA_R1_SPEND_REFIT.json` (05:35:40Z, label-free) and cited.
  - **A10:** IA_OSG had no deployment package. Lane DEPLOY_R1 stored, built and checked IA_OSG and IA_OSG_RO_HI in both versions
    (05:36:14-05:36:33Z): all bitwise, all refusals (`deploy/DEPLOYMENT_CHECK_R1.json`).
  - The other corrections and the advisories were applied to the documents by 05:40Z.
- **Corrections to earlier entries of this log** (entries are never rewritten):
  - (C2) The 03:25-03:35Z entry's "its refitted income model became more recoverable (0.5600)" is not supported: it is a reader-selection
    sensitivity in one repetition (RESEARCH_DECISION question 2).
  - (C4) The same entry's "Its targeting (CSF-T) and coordination (IND) ablations showed no gain" means no resolved gain.
  - (C12) Its "binding constraint (the men's occupation guard at pi = 0.10)": at pi = 0.10 income NLL (1.22 / 1.001 of S3) and the men's
    occupation guard (1.26 / 1.17) both fail, and in version 0 the women's guard too (1.18).
  - (A17) The 02:14:11Z entry's "a root commit of 77 sanitized files" counts this campaign's files only. The receipt's tree holds 149
    files: 77 under `publications/pcrl-input-and-pair-release-v1/` and 72 under `publications/pcrl-dual-design-release-v1/`, carried
    over from the dual-design receipt.
- **05:17:58Z / 05:47:18Z** Out-of-queue CPU charged:
  - the challenger 5 s;
  - the statistical reviewer 10 + 5 s;
  - the verifier 25 s;
  - the claims reviewer 10 s;
  - the coordinator 600 s (an upper estimate for the whole campaign).
- **05:43-05:46Z** Disk budget for custody.
  - The volume had about 4.4 GiB free, and the inherited backup scope plus its restore extraction (about 5 GB) would not fit.
  - The private archive is now compressed, and the occupation-headroom run tree (785 MB, with its own backup) is left out (deviation
    11).
  - Committed with the closing reviews and the corrections (0f4d7ccc7, 05:46:54Z).
- **05:47:03-05:48:04Z** First custody attempt (IPR_CUSTODY_V1): the manifest and bundle are ok, but the restore check failed.
  - **Cause:** the ten packages of 04:05Z pin code that commit 221a91f21 changed later, so `ipr.deploy` refused them from the restored
    copy.
  - **Repair (deviation 12):**
    - the attempt was recorded (`custody/BACKUP_ATTEMPT1_FAILED.json`) and its 1.14 GB folder removed;
    - the ten stale packages were moved to the private `deploy_stale_v1/`;
    - they were rebuilt (lanes/DEPLOY_V2.json, 05:51Z) and all fourteen checked (05:51:20-05:51:31Z): bitwise, all refusals;
    - committed (0f1806880, 05:51:56Z).
- **05:52:06-05:53:23Z** Second custody attempt (IPR_CUSTODY_V2): all ok from the backup alone (`custody/BACKUP_VERIFICATION.json`).
  - 14 packaged models re-run bitwise, and every package re-deploys with all refusals.
  - 5 custody readers reproduce their scores bitwise.
  - 39 releases, 201 admitted copies and 2,368 cache entries re-hash.
  - The lowest free space during the check was 1.8 GiB.
  - **05:53:32Z** the YOTUO drive was not attached, so off-device custody is pending (nothing mounted).
- **05:53:53Z** Queue closed (no unit running or waiting). **05:53:54Z** ledgers, RESOURCE_ACCOUNTING.json and ACCESS_RECEIPT.json
  written:
  - 90 units done and 1 failed; 4.282 CPU-h; 39 configurations; 47 label reads, none before the receipt;
  - the lineage log is unchanged, and no process was left running.
