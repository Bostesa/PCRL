# Coordinator log (campaign `pcrl_finite_solver_adequacy_v1`)

All times UTC on 2026-10-21, from `date -u`. Append-only.

- **02:22:58Z** Campaign start (first `date -u`). No AGENTS.md at the worktree root or below (checked to depth 3) or at the main
  checkout. Branch `research/pcrl-finite-solver-adequacy-v1` created from the finite-routing research tip 278cb1ed6.
- **02:23-02:25Z** Evidence read and checked: every named file of 04ac9161f (START_HERE, RESEARCH_DECISION, ADVISOR_BRIEF, TABLES,
  cards, strength-probe records, coordinator log, ER1, verification) is blob-identical to the local research copy; `nrm/fa/prc.py`
  is blob-identical to 682068d52's published copy; frc/common.py differs only by the publication sanitizer (one path). Card FA_PRC_v1
  section 1 documents the shrunk routing tables, cell-mean codeword refresh, seven-tilt first-improvement search and exact-objective
  acceptance as the specification; its R2 remedy was "a stronger search under the same objective".
- **02:26:55Z** Queue opened (`fsa.jobqueue`, 12 CPU-h, 8.5 h). Package `fsa/` copied from `frc/` with this campaign's constants
  (commit ac0626aa4).
- **02:28-02:30Z** Agents launched: statistical and challenger reviewer (FSA1 rules, advancement condition, reader validity, pilot
  rules) and exact-reference investigator (laws committed before experiments, narrow and broader references). Live: coordinator + 2.
- **02:30Z** Source admission ok (frc 04ac9161f and nrm 682068d52 on the remote; 31/31 frc code files identical, results identical
  except the post-publication PUBLICATION_RECEIPT.json; every frc private artefact re-hashes).
- **02:31-02:40Z** Solver code: `fsa.fa.solver` (SV1 exact tilt line search with best improvement; SV1_first; traced original;
  selftest 9/9 on invented data: cost terms equal `route`, interval partitions equal `route` at midpoints, incremental objective
  equals `objective`, traced original equals `lloyd` and `protected_path`, SV1 monotone and rule-consistent, HQ books handled) and
  `fsa.fa.solve` (stages replay / audit / source / nested). Card FA_FSA_v1 (no release configuration). Reader splits (+800 seeds,
  FSA1 tags; FIT ~4,000 / VALIDATION ~1,590 / SCORE ~2,400 people per repetition). Committed 250a78950 at 02:40:24Z.
- **02:40-02:42Z** PROTOCOL.md, EXPOSURE_MAP.md, FAILURE_MAP.md drafted (committed 0f03bc975 at 02:42:35Z; the first version of this line and of the two maps carried times ahead of the clock, corrected at 02:43Z). Registration waits for the statistical reviewer's RULES.
- **02:42:16Z** The investigator committed LAWS.md / laws.json (c27a91477) before any numerical experiment on them.
- **02:44-02:47Z** Invented-data timing and trace (not a registered law; 6,000-20,000 random rows): both solvers stalled after the
  first strengths; every partition on the refreshed book's tilt line was worse than the current state. The SV1 proposal class was
  extended to the current book's tilt line (card FA_FSA_v2 supersedes FA_FSA_v1, which governed no run; commit 91ba9a27e at
  02:46:39Z), before any real-data or registered-law run.
- **02:44Z** Engineering reviewer launched; **02:50Z** independent verifier launched. Live: coordinator + 4.
- **02:50Z** The investigator's first result files (narrow reference, enumeration) exist. The coordinator does not read them and
  relays nothing about them to the statistical reviewer until the reviewer's rules, including the advancement condition, are
  committed.
- **02:59Z** Investigator's STEPs 1-4 hand-back (no implementation defect; the old tilt grid misses feasible lower-recovery
  partitions in laws D and E). Statistical reviewer's rules committed 71f8de33b at 02:59:46Z (15/15 tests); the coordinator committed
  `fsa/rules.py` / `fsa/test_rules.py` unchanged (6e8594bf0, hashes as reported) with the stage changes the rules require (replay
  record in the SELECTION structure with estimator identity; saved books, effort and stop records; expressivity record; route S
  input builder `fsa.fa.advance`).
- **03:02Z** Investigator resumed for STEP 5 (original / SV1 / SV1_first on every registered law; route E input).
- **03:02:25Z** Registration commit f301965a7 (protocol section 13 answers the reviewer's challenge and discloses the order of events).
  **03:02:57Z** Receipt a771587ca (root of `publication/pcrl-finite-solver-adequacy-v1`, 57 files, 0 leak-scan problems) pushed.
  **03:03:08Z** Receipt verified on the remote (`fsa.receipt`; the eight registration files byte-identical). Label reads open.
- **03:03Z** Solver lanes SOLVER_V0 / SOLVER_V1 launched (replay, audit, source, nested).
- **03:07Z** Investigator's STEP 5 hand-back: on the 60 registered law cells, SV1 is never worse than the original (32 improve, 28
  equal); it is useful by exact recovery in all 12 cells of law D (pair AUC 0.6882 to 0.5803, every task loss < +0.0005); SV1_first
  equals SV1 everywhere. Provisional route E label (placeholders for the source fields) E_ADVANCES. **Coordinator's reading of RULES
  3.1** (not a rule change): the laws (02:42:16Z) and the revised solver's code (02:46:39Z) were committed before ANY solver ran on a
  registered law (first unit 02:49:12Z), so `laws_registered_before_run` and `solver_registered_before_run` are true. The clause
  "any law already run by a solver is excluded" concerns laws run before that registration (such as the 02:44-02:47Z invented trace),
  not the original solver's runs that the narrow reference requires after it. Route S governs the decision anyway (RULES 5.2).
- **03:08:20Z** Engineering review returned (26f9fea8e): no open blocker at HEAD. B1 (a TypeError in the source stage at 91ba9a27e)
  was already fixed in 6e8594bf0 before the lanes started (they launched from 9d62b4c3e). Dispositions: S1 (near-equal breakpoints
  can make knife-edge tilt intervals) is not changed in the solver after registration; instead the decision builder re-routes every
  saved source book at tilt x (1 +- 1e-9) and reports the people moved. S3 / S4: the decision builder hashes the private files and
  compares every per-person array of the nested original entries with the replay (not three keys). S2 / S5 / S7 / S8 / S9 / S6's
  comment: small fixes after the lanes finish (they must not change a running unit's code); PROTOCOL section 6's claim that the
  selftest covers empty cells and ties is inaccurate and is recorded as a deviation. N1 / N4: SV1_first is equal effort per step,
  not per chain; per-chain CPU and steps are reported from the source stage.
- **03:23-03:24Z** Source checks done (FSA_SOURCE_V0/V1_V1, about 1,030 CPU-s each, mostly the SV1_first control). PRC: SV1's
  fitted objective is below the original's at every registered strength in both versions (largest gain at strength 8: -0.0056 /
  -0.0105), every SV1 state rule-consistent and distinct from the original's and the twin's partition (S-a holds). The fitted pi
  pair MI falls by at most 0.0002 nats while re-routing rises (0.236 / 0.212 against 0.156 / 0.138) and the emitted tables' training
  cross-entropy rises by about 0.0008. HQ: SV1 is worse than the original at strength 8 in version 0 (+1.4e-4). SV1_first creeps
  along the current-book line (30 accepted steps per strength) and ends worse than both.
- **03:24Z, prediction registered before any nested result existed (committed 7600d6ae2 at 03:24:24Z; first written as 03:25Z in error):** route S fails at S-b (`S_PROXY_ONLY`), so the decision is
  `OPTIMIZER_REPAIR_CLOSED` (route E is superseded by S under RULES 5.2). Reason: a fitted information gain of 0.0002 nats is too
  small to move the held-out pair LOOKUP AUC by the pooled 2-SE margin (about 0.004 to 0.005). Expected nested pair AUC difference
  against the bracket: within +-0.003. Probability that route S advances: about 0.15.
- **03:29-03:30Z** Nested comparisons done (FSA_NESTED_V0/V1_V1, about 352 CPU-s each). **03:30:35Z** Route S input built
  (FSA_DECIDE_ROUTE_S_V1): the nested original-solver entries equal the P1 replay in all 117 per-person arrays (max difference 0.0);
  no saved source book moves a person when its tilt is perturbed by 1e-9 (relative); card committed before the runs; MODEL_SELECT not
  opened. SV1's S2 selection is strength 8 in both versions (nested total-NLL spend 0.00153 / 0.00121 against the original S2 point's
  0.00214 / 0.00194); against the measured-spend bracket (PRC@2.0 in both versions) the held-out pair LOOKUP AUC differs by -0.0018
  (se 0.0020) / -0.0038 (se 0.0017) with spend -0.0006 / +0.0003; pooled -0.0028 + 2 x 0.0019 > 0, so the recovery rationale fails;
  the task rationale fails (spend -0.0006 / -0.0007 against the -0.001 requirement). `S_PROXY_ONLY`.
- **03:31:39Z** Advancement (FSA_DECIDE_ADVANCE_V1, frozen `fsa.rules.advancement`): **OPTIMIZER_REPAIR_CLOSED**. D:
  `D_NO_MATERIAL_DEFECT` (the four documented approximations examined; no contradiction of the specification found). E: `E_ADVANCES`
  on the invented laws (law D), superseded by S (`E_SUPERSEDED_BY_S`, RULES 5.2). S: `S_PROXY_ONLY`. The registered prediction
  (`S_PROXY_ONLY`, probability of advancing about 0.15) held; the predicted range +-0.003 for the pair AUC difference held in version
  0 and missed in version 1 (-0.0038). The conditional pilot is therefore not authorized and does not run.
- **03:33Z** Equal-effort attribution (descriptive): on the real statistics the finer candidate set alone never helped (0 of 12 cells,
  PRC and HQ) and best-improvement selection produced every SV1 gain (12 of 12); on the exact laws SV1_first equalled SV1 everywhere.
- **03:35-03:38Z** Custody. FSA_CUSTODY_B1_V1 (…T033518Z) failed its restore check at 03:35:35Z: the stage records were not yet
  committed, so the backup lacked `replay/P1_REPLAY_V0.json`. I committed them (237c953a8, 03:37:09Z). FSA_CUSTODY_B1_V2
  (…T033709Z) verified from the backup alone at 03:37:32Z. The failed attempt is kept. Off-device custody is pending: no drive
  is attached.
- **03:38-03:45Z** Tables generated by `fsa.report` (e6994dcce). Drafts of RESEARCH_DECISION (5ad143f54), the narratives (94c00e1ef)
  and the other documents (54b97d623), all written before the closing reviews.
- **03:45:34Z** The verifier's final report, saved verbatim as `roles/verifier/REPORT_FINAL_0345.md`.
  - 57 checks; no numerical FAIL; `P1_REPLAY_VERIFIED` from its own recomputation; route S and the advancement reproduce.
  - It raised V2.ESTIMATOR_TEXT: the replay renames three local variables in the re-implemented selection loop. I asked the
    statistical reviewer for a ruling at 03:50Z.
- **03:42-03:52Z** The statistical reviewer's closing statement (177e4eeb1) and addendum 1 (af57badc8).
  - Every label was re-derived from the saved inputs.
  - It gave a required headline and corrections C1-C17.
  - It adopted my RULES 3.1 reading as the intended meaning, with two disclosures.
  - It ruled that `estimator_unchanged = True` stands (strict text: `P1_ESTIMATOR_CHANGED`), and recorded the RULES 8 pin as its
    own erratum.
- **03:55Z** The engineering reviewer's claims review (2312b801f). The tables reproduce exactly from the records. It gave corrections
  R1-R17 and further wording on usefulness, the cap and the prediction timing.
- **03:56-04:09Z** I applied both reviews.
  - RESEARCH_DECISION.md was rewritten around the required headline. OBJECTIVE_AUDIT, EXACT_REFERENCE_CASES, SOLVER_COMPARISONS,
    ADVISOR_BRIEF, START_HERE, HANDOFF and QUICKSTART were corrected.
  - New: VERIFICATION.md and PROTOCOL_DEVIATIONS entries 2-6.
  - My own checks while doing so:
    - The three exact-reference claims the reviewer could not trace are in RESULTS_V1.json: law B 0.5795 (`auc_feasible_hq`);
      identical information and AUC optima among feasible partitions in every law; cell-id information equal to released-value
      information in all 343 states; law E's original HQ path at pair AUC 0.5979.
    - SV1 stays above the same-class reference in 20 of 60 cells (10 distinct) and below it in 8 (law E 0.5-4).
    - The ledger lists uncommitted code paths for 17 unit starts, and launch digests differ from registration digests for 3 of 32
      (ENUM_V1, unused; two failed verifier units). No stage imported an uncommitted file.
  - **Corrections to earlier entries.**
    - At 03:23Z I wrote "SV1_first creeps along the current-book line (30 accepted steps per strength)". No record says which line
      its accepted steps came from, and HQ version 0 at strength 8 stopped after 20 steps. The documents no longer say it.
    - The 03:33Z attribution is confounded by the 30-step cap (deviation 6).
- **About 03:53Z (logged 04:09Z)** Stopped a leftover wait loop from the previous campaign. It watched FRC1's RULES.md for an ER1 completion record, but
  ER1 was recorded in RULES_ER1.md, so the loop never ended. It was not a queue unit and did no computation.
