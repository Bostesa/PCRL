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
