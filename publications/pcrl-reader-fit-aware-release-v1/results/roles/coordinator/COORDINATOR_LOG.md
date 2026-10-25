# RFA1 coordinator log (append-only; times from `date -u`)

- 2026-10-25T20:58:41Z: start. 118 GiB free; YOTUO not mounted at that moment.
- 21:00-21:05Z: read the EDG1 coordinator log and receipt, SAR1's protocol, anchor table and FB records (`sar/fb/pfd.py`,
  `sar/fb/pfo.py`), NEURAL_RAW's training code (`acs_replication/neural_raw.py`, `acs_replication/leafcorr.py`), DRF1's START_HERE.
  No label, private array or model was opened.
- about 21:05Z: isolated worktree `.claude/worktrees/pcrl-reader-fit-aware-release-v1`, branch `research/pcrl-reader-fit-aware-release-v1`
  from the DRF1 research tip 913eb5c91; the session switched into it (EnterWorktree).
- about 21:06Z: mechanism investigator and independent challenger spawned (label-free, synthetic only, 40-minute design window).
- 21:06Z: MEASUREMENT_CONTRACT.md drafted; statistical reviewer spawned at about 21:07Z (contract review, then the frozen rules).
- 21:08:39Z: `rfa/` derived from `drf/` (+ `famkit`, `famrun`, `ctlkit` from `sar/`) by `scripts/rfa1_rename.py`.
- 21:09:58-21:10:03Z: working cache cloned (nine trees, APFS clone). 21:11:29Z: `rfa.admit run` ok (working cache, publications
  DRF1 c4896479e and EDG1 d987e3685 on the remote, U0 / NEURAL_RAW hashes).
- 21:11Z: YOTUO found attached (identity checked by volume UUID). 21:12:04Z: the owed EDG1 off-device copy written
  (`PCRL_private_backups/pcrl_estimator_diagnostic_v1/pcrl_estimator_diagnostic_v1_20261025T180632Z`, via a `.partial` name).
  21:12:18Z: EDG1's restore check on the drive copy FAILED at its analysis step with the Homebrew interpreter (no matplotlib; record
  `EDG1_OFFDEVICE_RESTORE_ATTEMPT1_WRONG_INTERPRETER.json`); 21:12:38Z: rerun with the PCRL venv interpreter: all_pass (warm).
- 21:13-21:16Z: reader-grid constants (seed offset 1200, bootstrap 2026102520, DRF1 added to the overlap list), READER_SPLITS.json
  written label-free, DEVGRID_CODE_LOCK.json written with the torch interpreter.
- 21:16-21:21Z: `rfa/fam/net.py` (deployment), `rfa/fam/fit.py` (training loops and fit stage, adapted from `sar.fb.pfo`),
  EXPOSURE_MAP.md, FAILURE_MAP.md, ACCESS_POLICY.json; STORAGE_ADMISSION.json (relocation inventory hashes of the used replication
  files: equal; nothing restored).
- 21:21Z: PROTOCOL.md draft. 21:22:53Z: queue opened (`rfa.jobqueue init`: 8 CPU-h, 6 h elapsed, research window 5.5 h).
- 21:23Z: two time stamps written ahead of the clock were corrected before any commit (FAILURE_MAP 21:23Z -> 21:20Z, PROTOCOL 21:24Z ->
  21:21Z).
- about 21:21-21:30Z: verifier spawned (wave 1: independent derivative and graph check, label-free preparation). Fit-stage plumbing
  checked with a stub objective on invented labels (scratchpad only).
- 21:27Z: challenger hand-back (PROCEED; C1-C7), saved verbatim (`roles/challenger/HANDBACK_MESSAGE.md`; its own HANDBACK_1.md too).
  C1-C3 implemented in `rfa/fam/fit.py` (F0 gate, per-epoch mechanism statistics, ALTCRIT lag in the record); C5: descriptive
  per-recipe SCORE AUCs added to the grid entries.
- about 21:35Z: investigator hand-back (19 of 19 selftests; rfit.py 04b22c664caa8c55...), saved by the investigator itself.
- about 21:36Z: a coordinator response was stopped by the platform's safety classifier while writing the first card draft; the
  partial card file it left was deleted unread and a new card was written (RF_RFIT_v1.md). Nothing had been committed.
- about 21:38Z: statistical reviewer wave-1 hand-back saved verbatim; all contract findings adopted (MEASUREMENT_CONTRACT.md section 9).
- 21:40-21:42Z: rfit and rules selftests re-run (pass); fit-stage selftest with the real objective on invented labels: all_ok,
  including the bracket branch (rho 200 run).
- 21:43Z: reviewer asked to freeze the rules for the card (answers to q1-q4); PROTOCOL section 15 written; reader-grid selftest passed
  and DEVGRID_CODE_LOCK.json rewritten (the earlier lock kept as `DEVGRID_CODE_LOCK_superseded_before_registration.json`).
- 21:44-21:47Z: `rfa/rfclaims.py` (plan, decision and D2R assembly for the RFA1 record format).
- 21:49Z: the verifier's wave-1 wiring probe (in progress) flagged two record defects in `rfa/fam/fit.py`: the fit-path share helper read the wrong info key, and the objective counters were not recorded. Both fixed before registration (the output-space share now read from `output_grad_norms.fit_share`; inner reader fit counts recorded); fit-stage selftest re-run at 21:50Z: all_ok.
