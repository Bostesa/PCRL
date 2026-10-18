# Coordinator log (append-only; times UTC, 2026-10-18 unless stated)

- 18:32:40 First action (`date -u`); campaign clock starts. Read the source records (START_HERE, RESEARCH_DECISION, MECHANISM_CARDS,
  EXPOSURE_TABLE, PROTOCOL of the occupation-headroom campaign) and the source code that will be reused (ohn.common, ohn.receipt,
  ohn.jobqueue, ohn.protect, ohn.fam_a / fam_b headers, ohn.devrun, ohn.evaluate, ohn.rules, acs_replication.trees / stage_a / stage_b,
  ihs.fam_h header).
- 18:38 CatBoost absent from both interpreters; branch `research/pcrl-correction-transfer-stability-v1` created in its own worktree from
  the source research tip ca8725e72; fetched the pinned publication commits 1435c395d and 89fd2f6e6 from GitHub.
- 18:40 Installed CatBoost 1.2.8 (the replication's pinned version) with numpy 2.4.2 and scikit-learn 1.8.0 into a campaign-private
  virtual environment (PyPI; software, not data), for routing fits only.
- 18:43 Copied the source private trees (occupation-headroom runs and logs; the H3 mirror) into this worktree for admission.
- 18:45-18:50 Design brief (`roles/coordinator/DESIGN_BRIEF.md`); statistical reviewer launched (rules, uncertainty rule, claim ladder,
  screen comparison; invented-data tests only).
- 18:50-19:00 Wrote `cts/common.py` (single logged access layer by role x field x operation x purpose with authorized entry points;
  label-free partitions), `cts/jobqueue.py` (job-start code hashes), `cts/receipt.py` + `cts/test_receipt.py` (the write-once-on-failure
  defect repaired: 8 / 8 data-free checks pass), `cts/admit.py`, `cts/partitions.py`, `cts/trackf.py`, `cts/preflight.py`,
  `cts/evaluate.py`, `cts/nested.py`, `cts/publish_build.py`, `cts/lane.py`. Refit investigator (exposure map, then Track R) and reader
  investigator (grid adapter, then the screen study) launched.
- 18:53 Label-free admission: 2,618 private copies bitwise identical to the source worktree; the 201 H3-mirror copies match the
  published SOURCE_ADMISSION.json; 234 of 242 published results files identical (184 directly, 50 after the source sanitizer); the 8
  others are the source's post-publication closeout re-run (4b9ccf3e9) and its dated custody amendment (ca8725e72).
- 18:56 Label-free preflight of representative source outputs: A1_PO_16, B2_SINGLE, P2_PAIR_B80, P2A_PAIR_B80 occupation and H3 income,
  both seeds, recomputed from MODEL_SELECT features: all bitwise equal to the stored releases. Run out of the queue before it opened
  (python3, default BLAS threads; about 24 CPU-s; charged at 19:01 with the other pre-queue label-free work, 60 CPU-s upper estimate).
- 19:01 Queue opened. 19:02 Exposure map delivered by the refit investigator (code-verified; 9 corrections to the predecessor table).
- 19:07 Refit investigator found a recipe difference before any fit: the REFIT routing used seed_for(...) % 2**31 instead of the
  replication's unreduced CatBoost seed, and a one-task support repair; both fixed in cts/trackf.py and cts/nested.py before any fit.
- 19:12 Statistical reviewer finished RULES.md / RULES.json / cts/rules.py (26 / 26 invented-data tests). phi_prior of AGG1 uses the
  gross construction for all three views after the reader investigator found the formula text and the published income value
  disagreed (net vs gross).
- 19:12:42 Registration commit 39a27cc1e; sanitized package 2ad31d2af pushed as publication/pcrl-correction-transfer-stability-v1;
  receipt verified with ls-remote at 19:13:08Z (gated files byte-identical). Label reads open from here.
- 19:13 Support check passed (no tag suffix). Source replay: the P2_PAIR / P2A_PAIR B80 and B50 projection scales reproduce exactly
  on both seeds (max difference 0.0); realized MODEL_SELECT minus OOF men's-guard cost +0.0020 / +0.0030 (seed 0, P2 / P2A B80),
  -0.0012 / -0.0009 (seed 1), as published. Routing reproduction: CatBoost 1.2.8 refits the admitted seed-0 occupation T_TREE exactly
  (identical teacher SHA-256, 800 trees).
- 19:13 Lanes A (D0 then R0) and B (D1 then R1) started. 19:14 Pre-outcome record from the reader investigator (frozen source grid,
  label-free): H3's income view fails the old per-cell gate (vs H_LJ, TWJ) and AGG1 (vs H_LJ, RETIRED_B3CTL), so no Track F candidate
  that keeps H3's income can pass either recipient screen, whatever its occupation output.
- 19:15 Witness refits (A1_PO_16 recipe on 8,419 / 8,390 FIT people; T* 113 / ...): MODEL_SELECT occupation NLL excess vs U0 -0.0058
  (D0) and -0.0005 (D1) against -0.0161 / -0.0128 for the source witness fitted on all 20,000; men's guard +0.0007 / +0.0022 (source
  -0.0039 / -0.0023). Reserve slack of the binding men's guard: +0.0005 (D0) and -0.0008 (D1). Recorded as a finding: most of the
  source witness margin depends on its larger training set.
- 19:26 Lineage DEV_REPORT access log (replication private tree, read-only) snapshot: SHA-256 c7829e99e6af303cdf070f698b6088e024f3c2b806974a6b1f711785bb50c567,
  identical to the source campaign's admission and closeout snapshot; to be compared again at closeout (DEV_REPORT stays closed).

**Catch-up entries for 19:26-20:58.** They were written at 21:00 from the commit history, the queue ledger and the records listed;
they are not contemporaneous notes. Times are commit or record times.
- 19:13-19:35 Code for question A, Track R, deployment, sensitivity witness, screen study, compression, figures, tables, ledger,
  custody and claims-review brief committed (019241108 ... c2bc040e1). Deviations 1-4 recorded (0f6267004).
- 19:21 Independent verifier launched (separate code, 'verify' operation). 19:29 Engineering reviewer launched. 19:09 Correction
  investigator launched for the B2_SINGLE sensitivity witness (D0B).
- 19:46 Engineering review (no blocker; 5 major, 15 minor) and its 11 patches applied before any ladder, selection or realized
  outcome (4a17c64a3); deviations 6-7 (79edfb034).
- 19:53-19:55 Screen operating-characteristic study results and the reader investigator's findings and report saved with
  attribution (12d692291, 6834ef240, 1b21c42ae); statistical reviewer's SCREEN_REVIEW.md (f314f8a75).
- 19:57 F4. 20:08 Refit investigator's report saved (601ef6844). 20:10 Pending amendment P70 (a279f9f8f), diagnosis tables and
  recipe gates (c06a8e2aa), F5. 20:13:25 Amendment 9 committed after the written review (5002fed58).
- 20:17 D-stage frontier readers and F6 (113612ac7). 20:22 Training-size diagnostic results and F7 (b17107a1d). 20:23 F8.
- 20:23:59 REPAIR_FREEZE committed (c269256d2): NO_RECIPE, no redesign, comparison plan frozen; public receipt d3f28497b verified
  with ls-remote at 20:24:44Z (a12f2887f). Repair panels opened only after that.
- 20:25-20:26 R-stage banks: R_FRONTIER (c76ee2d11), R0-only compression twins (8150f4259), Track R (b9096af57).
- 20:28 Adjudication A1 accepted; the question-A bootstrap issue went to the statistical reviewer as A2 (a45fb7a9e). 20:29
  Diagnosis tables now include the R partitions.
- 20:34:05 Recorded correction of the question-A bootstrap (eece7dbdb). 20:34:41-20:35:44 The 34 units ran under the registered
  scheme. 20:37:36 Primary labels, ERRATA E1, A3 and deviation 11 committed (5b8ea7a86).
- 20:38-20:52 R_FRONTIER S0 (790 CPU-s) and S1 (706 CPU-s), R_COMP0 S0 and R_TRACKR S0 / S1. 20:39-20:50 Deployment lane: 10
  packages stored, built and checked (all bitwise, every refusal). 20:42-20:50 Bank evaluations.
- 20:47 cts.claims (2f581043e). 20:53 Repair-stage results, figures and deployment checks (4290cc2a1). 20:55 Frontier drivers and
  the P70 task-witness package (5db2acace).
- 20:56 Launch audit of units that started with uncommitted code (deviation 14). Deviations 12-15 recorded. RESEARCH_DECISION,
  START_HERE and F9-F12 drafted.

**Entries for 21:00-21:35 (written 21:35 from commits and message timestamps).**
- 21:00:44 Decision documents drafted (f4e559843). Independent claims reviewer launched (assignment 7).
- 21:00-21:01 Verifier's final report received: 263 checks; genuine FAILs F3 V2 (post-freeze code change) and C2.1 (P70A record).
  Saved with attribution at 21:02:44, committed with A4 and deviations 16-17 at 21:04:27 (f3d183f3c).
- 21:02:56 Section 8 conflict referred to the statistical reviewer. 21:04:46 Its ruling, ADJUDICATION_SECTION8.md: (b), with
  conditions. 21:05:10 Conditions 1 and 4 requested from the verifier. 21:06:36 A5, deviation 18 and the SCHEME_DEPENDENT flags
  committed (ce4c6079b).
- 21:07:12 Restore check made to find the packages in the deploy folder (fd8565463). 21:09:18 Quickstart (30809faaf). 21:09:49
  Deviation 19 on time stamps written ahead of the clock (2b49277ec).
- 21:10-21:12 Verifier follow-up: the trigger used D-stage inputs only (P4 V2 PASS), plus the R1 variance detail. A5 resolved as (b)
  and A6 recorded (6956507a1, 21:12:07). 21:12:36 Closeout process filter fixed (97de83f0f).
- 21:22 Claims review returned: 130 statements, 14 required edits. Saved with attribution; all edits applied, ERRATA E2 and A7
  recorded (d1a829736, 21:34:56).
