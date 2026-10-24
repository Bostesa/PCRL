# Protocol deviations and disclosures (campaign `pcrl_strong_anchor_release_v1`; append-only)

Each item: what happened, when (UTC from `date -u`), why, and what it changes. Registered files are never edited; corrections are
appended here or in dated amendments.

1. **Shared code edited while fit units ran (05:13-05:15Z).** The coordinator extended `sar/roles.py` and `sar/evaluate.py` (role
   members that resolve to the frozen START reference in a version, for bracket points that coincide with the origin) while units
   SAR_FIT_FB_V1 and SAR_FIT_FA_V0 were running; the queue recorded `code_changed_during_run: [sar/evaluate.py, sar/roles.py]` for
   both. Neither fit imports either module (`sar.fb.pfd` and `sar.fa.tfr` import `sar.common`, `sar.famkit`, `sar.famrun`, `sar.release`
   only; checked with grep at 05:16Z), so no released array depends on the edit. Shared modules are not edited during running units
   from here on.
2. **Coordinator-only modules edited during reader / fit units (05:18-05:24Z).** `sar/plan.py` and `sar/claims.py` (the plan and
   claims writers) were written while SAR_FIT_FA_V1 and SAR_GRID_REFS_V0 ran (`code_changed_during_run` records them). Neither unit
   imports them (`sar.devreaders` and `sar.fa.tfr` do not). Item 1's statement is narrowed accordingly: modules that a running unit
   imports are not edited while it runs.
3. **Investigator B ran one read-only git command (05:15Z).** While listing files it ran `git -C <worktree> log --oneline -3` against
   the coordinator's no-git instruction (self-reported in `roles/investigator_b/HANDBACK_R1.md`); nothing was changed.
4. **The FB package entry `sar/fb/__init__.py` changed after the FB_PFD_v1 card's commit (05:30Z).** It now routes FB_PFO* configurations
   to `sar.fb.pfo.predict`; FB_PFD* configurations still reach the unchanged `sar.fb.pfd.predict` (investigator B's python3 deployment
   check, `roles/investigator_b/check_deploy_python3.json`). Committed with the FB_PFO_R1 card.
5. **Adaptive tag of the revision readings.** The reference utility unit (05:13Z) and the reference reader grid (from 05:14Z) read
   MODEL_SELECT before the revision cards were committed. No one opened their outputs before the cards were committed, and no FA / FB
   MODEL_SELECT outcome existed; the revisions were designed from MODEL_FIT / CALIBRATION records only. The mechanical rule (first
   MODEL_SELECT read before the card) still tags every revision reading ADAPTIVE_, and that conservative tag is kept.
6. **The FA package entry `sar/fa/__init__.py` changed after the FA_TFR_v1 card's commit (05:36Z).** It now routes FA_TFRO_R1* ids to
   `sar.fa.tfro.predict`; every other FA id still reaches the unchanged `sar.fa.tfr.predict` (FA_TFR_v1 saved no configuration).
   Committed with the FA_TFRO_R1 card.
7. **FB_R1 saves only the upper bracket points (specification budget).** The plan's fineness record reads SAVED_POINT_RULE_NOT_FOLLOWED
   for CTRL and ALT in both versions because the origin serves as the lower point while the full grid holds cheaper points with spend
   <= CAND's (they were computed and recorded, flag LO_NOT_SAVED_SPEC_BUDGET, but not saved). Reported, never gating (RULES.md 4.4); the
   frontier's (c3) check covers only the origin on that side.
8. **FA_R2 evaluation unit failed on a write-once record (06:14:18Z).** SAR_EVAL_FA_R2_V1 evaluated the candidate FA_TFRB_R2 (record
   written, with paired costs against SUCC, the reused CONT FA_TFRO_R1_CONT and both CTRL role points) and then stopped when it tried to
   re-evaluate the reused CONT, whose MODEL_SELECT record already existed from FA_R1 (write-once guard). The failed unit is recorded and
   charged; the lane was resumed without that step (`lanes/READING_FA_R2b.json`, 06:16Z); CONT's existing record serves both readings.
9. **Plan hash missing from the first decision records (verifier wave 1, checks S6_FB_R1 / S6_FA_R1).** PROTOCOL 6 requires each
   reading's plan SHA-256 in its decision record; `claims/<reading>/DECISIONS.json` of FA_REG, FB_REG, FB_R1, FA_R1 and FS_CPI carried
   only the plan id. The records are write-once and are not edited: a sidecar `claims/<reading>/PLAN_SHA256.json` now holds each plan's
   hash (equal to the verifier's: FB_R1 349a6f6e..., FA_R1 5a78be99...) and the decision record's hash; `sar.claims` records
   `plan_sha256` in every later decision record. Every plan was committed before its reading's first MODEL_SELECT computation (verifier
   check 6), so the ordering requirement holds.
10. **FB_PFO's income output is not bitwise NEURAL_RAW's (verifier check S1_FB_R1_income_bitwise_START).** Card FB_PFO_R1 states that
    the candidate's income equals START's bitwise; on MODEL_SELECT it differs from the frozen NEURAL_RAW reference wire by at most
    6.5e-8 (v0) / 6.8e-8 (v1) because the family recomputes NEURAL_RAW's income network with its own float64 numpy forward pass (the
    stored reference wire came from the torch pass). Decisions are identical, the income utility differs by less than 1e-9 and every
    income-view grid cell difference is exactly 0 (the rules read the income view as ALIAS_ZERO). The card's statement is corrected
    here to "equal to START's income up to 7e-8 (float64 recomputation); identical decisions"; no label depends on it.
11. **Verifier code edits recorded as code drift in coordinator units (recorded 06:31Z).** The queue's code manifest includes
    `roles/verifier/code/` by design (verify units carry its provenance). The verifier edited its own scripts while coordinator units
    ran, so SAR_GRID_SAR1_FB_R1_ROB_V0 / _V1, SAR_FIT_FA_R2_V1, SAR_FIT_FS_CPI_V1 and SAR_GRID_SAR1_FA_R2_V0 list verifier files in
    `code_changed_during_run` (06:04-06:21Z). No campaign module imports the verifier's code (checked by search: `sar/` references the
    folder only as a path in `jobqueue.CODE_PACKAGES`, `closeout` and the access policy), so no unit's computation could depend on
    those edits. The verifier was asked at 06:30Z to edit its code only while no unit is running.
12. **Two further units recorded verifier-file drift after deviation 11 (recorded 07:03Z).** The verifier created new scripts of its
    own while coordinator units ran: SAR_GRID_SAR1_FA_R2_ROB_V1 lists `v2_ms.py` and `v2_refit.py` (06:36-06:37Z) and
    SAR_CUSTODY_REHEARSAL_V1 lists `v2_assemble.py` (06:47Z). As in deviation 11, no campaign module imports the verifier folder, so
    neither unit could depend on those files; every verifier unit itself records an empty `code_changed_during_run` (verifier wave 2).
13. **Owner identifiers in the public registration receipt (found by verifier wave 3 at 07:09Z; recorded 07:29Z).** The receipt commit
    ede0755fb (public on the GitHub publication branch since 05:11Z) carries the owner's name inside the drive relocation folder name
    (PROTOCOL.md line 59; STORAGE_ADMISSION.json lines 6 and 47) and the machine's host name (roles/coordinator/COORDINATOR_LOG.md line
    5). The shared sanitizer has no pattern for either and the leak scan did not flag them. The research files are not edited
    (PROTOCOL.md is registered and immutable). From the final publication on, `sar.publish_build` derives the owner's identifiers at run
    time (full name, its tokens, home folder name, host names, e-mail local part; none is written into the code), redacts them in every
    published text file after the shared sanitizer, and refuses the build if any remains; a registered file may differ from its
    registered bytes only by that declared redaction (`gated_files_owner_redacted`, PUBLICATION_RECEIPT.json). The receipt commit, the
    parent of the final publication, still holds the original bytes, and the public history is not rewritten: removing it would need a
    force push to a public branch and would break the timestamped registration evidence, which is the owner's decision.
14. **Deployment equivalence restated as a tolerance (found by verifier wave 3; recorded 07:29Z).** `deploy/DEPLOYMENT_CHECK.json`
    reported bitwise parity for the full DEV_REPORT request (true). The deployed probabilities depend on the request's batch size at
    floating-point level (the batched matrix products' summation order): across batch sizes 1, 2, 7, 50, 333, 1,000, 2,971 and 5,943,
    the largest absolute difference from the evaluated output is 4.4e-16 in this campaign's sweep (`deploy/DEPLOYMENT_TOLERANCE.json`)
    and 5.6e-16 in the verifier's 50-row sample; every decision is identical. PROTOCOL 11 allows "bitwise, or the stated tolerance": the
    stated tolerance is 6e-16 absolute per probability, with identical decisions; bitwise holds for the full evaluated batch.
