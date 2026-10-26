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
- 21:51:34Z: reviewer's wave-1b rules final (21:49:38Z; hashes 76baa04a / b00482c4 / 4e03270c / 1548b6de; 24 of 24 tests); saved
  verbatim (`roles/statistical_reviewer/HANDBACK_1b.md`). Its reading that a NEAR_ALIAS ALTCRIT still gates is adopted.
- 21:51:39Z: PROTOCOL.md completed. 21:51:57Z: registration commit 43830ae19.
- 21:52-21:53Z: sanitized receipt package built (79 files, no problems; gated files unchanged by the sanitizer); root commit
  34dc42b51 on `publication/pcrl-reader-fit-aware-release-v1`. Private counts-only scan (`scripts/rfa1_private_scan.py`): owner 0,
  e-mail 0 in tree, message and history; 3 absolute-path hits, all mount-point paths in inherited code (`/Volumes/...` in
  `rfa/admit.py` and `rfa/storage.py`), as in DRF1. Pushed as a new branch at 21:53:16Z (no force); remote tip equals the commit.
- 21:53:27Z: `rfa.receipt verify --kind registration` verified; the label gate opened (`registered()` True). No label read before.
- 21:53:40Z: fit units RFA_FIT_RF_V0 and RFA_FIT_RF_V1 launched in two lanes (two heavy jobs, one thread each).
- 21:54-21:56Z: `rfa/rfclaims.py` adapted to the reviewer's final record keys (config ids as grid member names, f0, lo_not_saved,
  ALTCRIT lag, falsifier records F1 / F3 / F5 / F6).
- 21:56Z: correction of the line above: the queue ledger records both fit units' unit_start at 21:53:37Z (not 21:53:40Z).
- about 21:58Z: verifier wave-1 hand-back saved verbatim (`roles/verifier/HANDBACK_1.md`): 26 of 26 derivative checks, 19 of 19 wiring
  checks; defects D1 (ALTCRIT inner-fit count understated), D2 (a reader refusal aborts the unit without a FIT record; the queue
  records the failure), D3 (a log-sampling label) are recorded as deviations, not fixed during the running fits.
- 21:57:31Z / 21:57:32Z: both fit units ended (233 / 234 CPU-s). F0 MECHANISM_ACTIVE in both versions (orthogonal share 0.102 / 0.112,
  cos 0.9951 / 0.9940). k* = S20 (v0; women's occupation Brier binds at S30) and S70 (v1; occupation NLL / Brier bind at S80). Brackets:
  STOPFIT S10 / S20 and S60 / S70, ALTCRIT HI S20 / S70, RAW-J HI S120 / S200. 12 configurations saved. ALTCRIT median lag 0.11 / 0.21
  (not a near alias).
- 21:59Z: PLAN_RFA_v1 written from the fit records; bank specs RFA1_MAIN and RFA1_LC; lanes; commit ff741a72c before any MODEL_SELECT
  read. 21:59:42Z: lanes A (reference and release utilities, main grid V0, learning curve V0) and B (main grid V1, learning curve V1)
  launched.
- 22:00Z: correction: the ledger records both fit units' unit_end at 21:57:33Z (the line above said 21:57:31Z / 21:57:32Z). RFA_EVAL_REFS ended 21:59:55Z and RFA_EVAL_RFA_v1 21:59:59Z (both done).
- 22:00-22:14Z: utilities (RF_FULL G1 at the point in both versions; v1 binding the men's occupation guard at 0.885 of cap; RF_RAWJ_HI
  fails G1 in v1: income Brier vs Ucal 1.059 of cap) and the main grid RFA1_MAIN (740 / 792 CPU-s); learning curve RFA1_LC after it.
- 22:16Z: investigator resumed to implement the revision's reader family (label-free, new module `rfa/fam/rfit_r1.py`), from the
  descriptive main-grid pattern (ties with ALTCRIT and STOPFIT; winning readers CATI1 / ET25 / MLP64; LOG_LR fell most).
- 22:16-22:18Z: `rfa/fam/fit.py` given an optional objective module, a revision label and an alias hook for the revision (v1 runs
  unaffected: they had ended).
- 22:17:42Z: claims RFA_v1: D1 TASK_FEASIBLE; D2 PARTIAL_START_ONLY (START +0.0127, se 0.0059; CONT +0.0086 (0.0092); STOPFIT at
  RF_STOP_HI +0.0028 (0.0045); ALTCRIT +0.0005 (0.0010 floor); RAWJ -0.0016 (0.0084)); D3 NOT_SUPERIOR; D4 NOT_MET; LC
  LEARNABILITY_LIMITATION; falsifiers F2, F3, F4 triggered; outcome BOUNDED_NEGATIVE_EXPERIMENT. First fitted comparison reached at
  1 h 19 min.
- 22:18:25Z: second-split bank RFA1_ROB written from the decision record (commit 586ed2560); ROB lanes V0 / V1 launched.
- 22:19-22:22Z: `rfa/fam/fit_r1.py` (the R1 entry point; no alias hook, 6 configurations per version at most); R1 card drafted.
- about 22:22Z: investigator hand-back 2 (saved by the investigator, `roles/investigator/HANDBACK_2.md`): `rfa/fam/rfit_r1.py`
  (24c2bd99...), 12 of 12 selftests, rfit.py unchanged; frozen softbin values B 8, width 0.3, tau 0.25 (standardized), ridge 1e-3.
- 22:22-22:23Z: R1 card completed with the frozen values; R1 fit-stage plumbing check on invented labels (scratchpad): all_ok.
- 22:23:27Z: R1 card and code committed (4c746de74) before the R1 fit; R1 fit lanes queued at about 22:24Z behind the RFA1_ROB grids
  and two verifier units (FIFO).
- about 22:24-22:26Z: `rfa/custody.py` (base 913eb5c91, drf and the SAR1 / DRF1 results in the code scope) and `rfa/restore_check.py`
  (family FAM models re-hashed; no DR source pins) adapted for closeout; EDG1 memory notes updated (off-device copy made).
- about 22:27Z: verifier interim message (saved verbatim, `roles/verifier/INTERIM_R1_DERIVATIVE.md`): the R1 objective passes 11 of 11
  derivative checks with the verifier's own code (FULL vs finite differences of the re-solved objective 6.3e-10); a first harness run
  failed on a float32 test input of the verifier's own and is kept beside the passing run.
- about 22:28Z: statistical reviewer's R1 card check and claims-review wave 1 saved verbatim (`roles/statistical_reviewer/HANDBACK_2.md`):
  R1 reads unchanged as ADAPTIVE_R1 (addendum RULES_ADDENDUM_1_R1.md: a beside family-change reading FC_R1; R1 ids kept even if
  bitwise equal to v1 points); every v1 label follows; two REQUIRED record fixes (half-budget grid validity; falsifier detail).
- 22:29:52Z: v1 record rebuilt with both fixes (`claims/RFA_v1_rebuilt/`; deviation 6): half-budget validity effective; labels
  unchanged. Verifier interim message 2 saved (`roles/verifier/INTERIM_2.md`).
- 22:39:55Z: second-split grids RFA1_ROB ended (1275 CPU-s each). 22:40Z: the verifier's two short units ran (grid recomputation,
  12-reader refit); R1 fit units started 22:40:05Z / 22:40:07Z.
- 22:41:29Z: D2R of RFA_v1 (label-free, outside the queue: `claims/RFA_v1_D2R/`): ROBUSTNESS_NOT_CONFIRMED (refit genuinely
  effective). Robustness-cell pair benefit: NEURAL_RAW +0.0120 (0.0075), RF_TASK +0.0102 (0.0074), RF_STOP_HI -0.0009 (0.0036) and
  RF_ALTC_HI -0.0023 (0.0049) (both sign changes), RF_RAWJ_HI -0.0044 (0.0090); SUCC sign change; recipient failures on the
  robustness cells against RF_RAWJ_HI, H_LJ, RETIRED_B3CTL, SUCC.
- 22:45:30Z: R1 fits ended (303 / 312 CPU-s): F0 active (orthogonal share 0.098 / 0.101, not larger than v1's); k* S30 / S60;
  brackets STOPFIT S20 / S30 and S50 / S60, ALTCRIT HI S30 / S60, RAW-J HI S140 / S180; 12 configurations saved (24 of 24 used).
- 22:45:52Z: PLAN_RFA_R1 (ADAPTIVE_R1; beside member RF_FULL; no bitwise identity with v1 points), banks RFA1_R1_MAIN / RFA1_R1_LC,
  lanes; commit 2b830dc5e before any R1 MODEL_SELECT read. R1 lanes launched 22:46Z; deployment lane queued behind them.
- 22:47Z: R1 utilities: RF1_FULL G1 at the point in both versions (v1 men's guard 0.00457, 0.915 of cap); RF1_RAWJ_HI fails G1 in v1
  (income Brier vs Ucal 1.137 of cap).
- about 22:51Z: verifier wave-2 hand-back saved verbatim (`roles/verifier/HANDBACK_2.md`): selection, usefulness, grid (bitwise),
  12 reader refits (bitwise) and every v1 label reproduced; F1 (income wire 6.8e-8 from the frozen START wire) and D4-new recorded as
  deviations 7 and 8.
- 22:54:16Z: R1 claims (label-free, outside the queue; `claims/RFA_R1/`): ADAPTIVE_R1__BOUNDED_NEGATIVE_EXPERIMENT; D2 NOT_SEPARATED
  (START +0.0098 (0.0054) point only; CONT +0.0090 (0.0051); STOPFIT -0.0014 (0.0041); ALTCRIT -0.0030 (0.0045); RAWJ -0.0046
  (0.0052)); D3 NOT_SUPERIOR; D4 NOT_MET; LC LEARNABILITY_LIMITATION (point, vs H_LJ); half-budget validity effective. FC_R1 NOT_POSITIVE
  (-0.0029, 0.0046) after the field fix of deviation 9.
- 22:55Z: deployment lane done: 6 packages (RF_FULL, RF_STOP_HI, RF1_FULL x 2 versions) stored bitwise, built and checked (MODEL_FIT
  raw-input parity bitwise; every refusal; attacked MODEL_SELECT full batch bitwise; request sizes 1 / 7 / 50 / 333 within 6.7e-16,
  decisions equal at every size). R1 second-split grids launched 22:55:50Z.
- 22:56-22:58Z: verifier wave 3 and reviewer claims-review wave 2 requested; RESEARCH_DECISION.md and START_HERE.md drafted.
- 23:08:08Z: R1 second split assembled label-free (`claims/RFA_R1_D2R/`): ROBUSTNESS_NOT_CONFIRMED; sign changes vs RF1_ALTC_HI and
  SUCC; robustness recipient failures vs RF1_RAWJ_HI, H_LJ, RETIRED_B3CTL, SUCC; refit genuinely effective.
- 23:08:39-23:08:42Z: governing records rebuilt with the second splits attached (`claims/RFA_v1_final/`, `claims/RFA_R1_final/`;
  labels unchanged; deviation 10). 23:09:42Z: `claims/CAMPAIGN_SUMMARY.json` (rules' campaign summary; binding MODEL_FIT budget
  components).
- about 23:12Z: reviewer claims-review wave 2 (`roles/statistical_reviewer/HANDBACK_3.md`) and verifier wave 3
  (`roles/verifier/HANDBACK_3.md`, every task-1-3 check passes; disclosure D-W3) saved verbatim.
- 23:15:15Z: `claims/PER_RECIPE_PAIR_MEANS.json` (label-free; the R1 card's per-recipe prediction judged against START: not met).
- 23:15-23:25Z: reviewer REQUIRED items R1-R7 and the advisories applied to RESEARCH_DECISION.md, START_HERE.md, HANDOFF.md and
  QUICKSTART.md; TABLES.md / ENDPOINTS.json regenerated from the final records (FC_R1, reference usefulness, second-split table,
  same-checkpoint spends). Outside-queue CPU charged: agents' later waves 35 s, claims finalization 30 s.
- 23:25-23:26Z: research commits ef4d945b0 (records and documents) and 42ef29832 (final custody lane); verifier wave 4 requested
  (document conformance, per-recipe means and R1 D2R recomputed independently).
- 23:27:18-23:29:29Z: final backup (unit RFA_CUSTODY_FINAL_V1; tip 42ef29832; 846,815,076 bytes) and restore from the backup alone:
  all_ok in both interpreters (24 models and 24 releases re-hashed, the 6 packaged models re-run bitwise, 140 grid member wires,
  1,480 reader-cache entries, representative readers bitwise, all 6 deployment packages bitwise with every refusal; no label read).
  `custody/BACKUP_VERIFICATION.json`.
- about 23:28Z: YOTUO disappeared from /Volumes without any write to it; it was back by 23:30:36Z (remounted by macOS; nothing was
  mounted or forced by this campaign). Identity re-verified read-only (`diskutil info`: volume UUID 3D74932B-..., exFAT, 578 GB free)
  before the off-device copy (unit RFA_CUSTODY_OFFDEVICE_V1, started about 23:31Z).
- 23:33:14Z: restore check from the drive copy passed (WARM; manifest equal to the source; 106.8 CPU-s in the queue); identity
  re-verified after the copy. `custody/OFFDEVICE_COPY.json`.
- 23:34Z: queue closed; `python3 -m rfa.closeout` wrote RESOURCE_ACCOUNTING.json and ACCESS_RECEIPT.json (2.594 h, 2.497 CPU-h,
  46 units done, 2 failed: RFA_VER_USEFULNESS_v1 by design and the verifier's RFA_VER_W4_R1ROB_v1, a KeyError in its own merge, rerun
  as _v2; 40 label reads, none before the receipt, no closed role, no process left). Research commit bf4ca1b40.
- 23:36-00:06Z: verifier wave 4 (hand-back 4, saved verbatim with one local path replaced): 187 of 188 numeric lines matched; one
  factual error (same-checkpoint spend: its paired SEs are recorded and the compared points sit at that checkpoint), one rounding
  slip (+0.0101 for +0.0100), three wording defects, an unprefixed R1 D2R label and seven nits; per-recipe falls from rounded means;
  R1 D2R reproduced. All corrected in the documents before publication (deviation 11; corrected per-recipe record written beside the
  first).
- 00:10-00:12Z: research commit c4bc191e1; publication tree built from it (211 files, 0 problems, 29 sanitized) and committed
  locally as a9e535345 (child of 34dc42b51, not pushed). Private counts-only scan: owner 0, e-mail 0 in tree, message and history;
  4 absolute-path hits, all non-personal mount points.
- 00:13-00:36Z: verifier wave 5 (hand-back 5, verbatim): independent publication scan clean (no owner identifier, user, host or
  e-mail in tree or messages; 35 absolute paths, none personal; no private content; gated files byte-identical); one unrecorded
  CSV line-ending normalization; observation that commit identity lines carry the configured git e-mail (pre-existing, as on every
  pushed commit; anonymity is not claimed); closeout cross-checks 33/33 (first run with its own check-design failure kept); one slip
  in the SE range (3.1-6.2 for 3.1-6.1), fixed in the documents (deviation 12). The drive was absent at 00:25Z and 00:33Z.
- about 00:39Z: late outside-queue charges (verifier 18 s, coordinator 45 s); ledger total 9,052 CPU-s (2.514 CPU-h).
- 00:40-00:42Z: research commit 60acaf352; the unpushed local publication ref was moved back from a9e535345 to the receipt 34dc42b51
  (a9e535345 was never pushed). The rebuild refused on one leak-scan hit: a bare session-scratch folder prefix in the verifier's
  v5_pubscan.py, line 93. It was reviewed as harmless and added to REVIEWED_SCAN_HITS (deviation 13).
- about 00:43Z: the build of 7637ef7e8 refused on three more hits, all in my own descriptions of that review, which quoted the prefix
  literally (the publish_build comment, deviation 13 and this log). They were reworded without the literal string. The scan was not
  relaxed for them.
