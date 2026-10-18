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
