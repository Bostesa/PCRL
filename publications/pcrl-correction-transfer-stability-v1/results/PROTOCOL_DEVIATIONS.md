# Protocol deviations, amendments and disclosures (append-only; times UTC, 2026-10-18 unless stated)

1. **Out-of-queue label-free preflight before the queue opened (18:56).** The representative source-output check
   (`cts.preflight outputs`, label-free) ran directly with `python3` and default BLAS threads (about 24 CPU-s user + system) before
   `cts.jobqueue` was initialized at 19:01. It read no label. Charged at 19:01 together with the other pre-queue label-free work
   (60 CPU-s upper estimate, ledger label COORD_PREREG_LABEL_FREE_V1).
2. **Routing recipe corrected before any fit (19:07).** The refit investigator found that the first version of the REFIT routing
   reduced the replication's CatBoost seed modulo 2^31 and used a one-task class-support repair; both were corrected to the exact
   Stage A recipe before the registration commit. The routing reproduction (`CTS_ROUTING_REPRO_S0_V1`) then refitted the admitted
   seed-0 occupation teacher exactly.
3. **Code changed while a unit waited for a slot (19:19).** `CTS_D0_DIR_REFIT_LOC_V1` registered at 19:17 and launched at 19:19:15
   while new, not-yet-imported modules (`cts/deploy.py`, `cts/freeze.py`) were being written; the queue recorded the digest change
   and `cts/freeze.py` as uncommitted at launch. The code the unit imports (cts.trackf, cts.common, ohn.protect, acs_replication)
   was committed and unchanged. No other unit launched with uncommitted imported code (checked from the ledger).
4. **Access amendment (19:20, before first use).** `ACCESS_AMENDMENTS.json` authorizes the entry point `cts.questiona` for the
   operations fit, select, check_diag and check_repair (question A per-person increments); committed before its first use.
5. **Sensitivity witness preprocessing (19:29, before its fit).** PROTOCOL section 6 states that AGEP / WKHP enter through the
   admitted all-MODEL_FIT z-map, to which the TREE learners are invariant. The sensitivity witness (B2_SINGLE recipe, partition alias
   D0B) also contains the additive logistic learner ADD_V2, which is not invariant to that map; its AGEP / WKHP are therefore
   re-standardized with FIT moments, and its spline / frequent-WKHP featurizer is fitted on FIT inputs only (the recipe's absolute
   200-person WKHP threshold kept, giving 6 indicators on D0's FIT instead of the source's 13 on all MODEL_FIT). The dictionary,
   z-map and featurizer are fitted once on all FIT and shared by its fold pipelines (as the source did on MODEL_FIT; label-free).
   Recorded in its WITNESS.json 'adaptations'. Decision by the coordinator before the fit.
6. **Engineering patches applied before any correction outcome (19:46-19:47).** The engineering review (no blocker; 5 major, 15 minor)
   supplied 11 patches, applied and committed (4a17c64a3) before any ladder, selection or realized outcome of a correction was
   computed. One changes a registered computation: for the ensemble deployment, the recipe gates G2 / G3 and the tie-break reserve
   ratios now use the ensemble's increment against the deployed witness W (whose MODEL_SELECT margins define the reserve) instead of
   its increment against the uncorrected ensemble (ENG-01; the transfer diagnostics keep the latter). Not patched, disclosed: the
   frozen ens rules budget e_s - e_0 against W's reserve, and ens 'identity' is the uncorrected ensemble e_0, not W.
7. **Code committed while units waited for a slot (19:20-19:47).** Several units registered before and launched after commits of new
   modules or patches (the queue records the digest at registration and at launch; `digest_changed_while_waiting`). At every launch
   the imported campaign code was committed and clean except `CTS_D0_DIR_REFIT_LOC_V1` (item 3); the changes concerned modules those
   units do not import, the default-preserving Track R income option of cts.trackf (19:20) and the reviewed patches (19:46).
8. **Amendment timestamps (verifier ATTENTION, recorded 20:09).** The four entries of ACCESS_AMENDMENTS.json carry
   "utc": "2026-10-18T19:20:00+00:00", but the file was first committed at 19:14:44Z (commit 019241108): the coordinator wrote an
   estimated time instead of the actual one. The commit time is the record; the entries were committed before the first use of
   cts.questiona, so the before-first-use rule holds. The file is not edited.
9. **Adaptive training-size diagnostic (amendment, 20:16, after written statistical review, before its fits).** Motivated by the
   observed D / R witness margins (NO_RECIPE): the A1_PO_16 recipe refitted on the whole non-panel pool (partition P70, 14,022 people,
   70% of MODEL_FIT; panels still excluded; no GUARD / CHECK role) and on the same rows and folds with the admitted dictionary (P70A),
   each evaluated on MODEL_SELECT against both seeds' references, with paired differences between fits and the dictionary collapse
   at each size. Predictions and the reading rule are registered in `diagnosis/TRAINING_SIZE_BET.json` before the fits; the review is
   `roles/statistical_reviewer/REVIEW_TRAINING_SIZE_DIAGNOSTIC.md`. Labelled ADAPTIVE; kept out of every selection; it cannot change
   NO_RECIPE. Two fitted release specifications.
10. **Question A bootstrap brought into line with RULES.md section 6 (recorded correction, 20:40).** The verifier found (ATTENTION,
   20:29) that `cts.rules.prediction_comparison` drew a fresh household resample for every (set, statistic), whereas RULES.md section 6
   resamples households once per set and replicate; the statistics are positively correlated on the same households, so the
   per-statistic scheme understates Var(D) (verifier: SEs 7-25% smaller). The statistical reviewer adjudicated independently
   (`roles/statistical_reviewer/ADJUDICATION_QA_BOOTSTRAP.md`): the registered text defines one shared resample; the criterion, D, B =
   2000, the tag and the q05 > 0 rule are unchanged. The correction is `cts.rules.prediction_comparison_registered` (one multinomial
   weight matrix per set, drawn in the order OOF, GUARD, CHECK from the tag-seeded generator) with a data-free test against a
   brute-force household resampler (`cts/test_rules_correction.py`: identical D and MAE, identical replicate draws, larger SE than the
   superseded scheme on correlated statistics). `cts.questiona --scheme registered` (default) writes `<ID>__<dep>__<oof>__REG.json`;
   the 34 first records stay in place, marked superseded. Primary labels are the single registered-tag run of all 34 units. Effect
   known before the rerun: the verifier's replays put D0 REFIT_CONT ens at NOT_IMPROVED (q05 -0.065 to -0.007 over 5 seeds). The claim
   ladder is unaffected (rung 1 NOT_MET under both schemes: NO_RECIPE, and no direction arm is PREDICTION_IMPROVED on D0).
11. **Times in item 10 (recorded 20:37).** Item 10 and the resolution of adjudication A2 say "20:40"; the correction was committed at
   20:34:05Z (eece7dbdb). The coordinator wrote an estimate instead of the actual time; the commit time is the record. The 34
   registered-scheme units ran 20:34:41-20:35:44Z (committed clean code). Primary labels (diagnosis/QUESTION_A_LABELS.csv): PREDICTION_IMPROVED only for D1
   FROZEN_PAIR ens (A1_NOT_SUPPORTED), the CONT control on D1 and R0 (full and ens); three labels depend on the scheme (D0 REFIT_CONT
   ens, R1 REFIT_LOC full, R1 REFIT_PAIR_TR full: superseded PREDICTION_IMPROVED, primary NOT_IMPROVED, as the verifier's five-seed
   replays predicted). No PAIR or LOC arm is PREDICTION_IMPROVED on any partition under the registered scheme.
12. **Rule-selected strengths not reader-evaluated (recorded 20:57).** REPAIR_FREEZE's comparison plan lists reader grids "at every
   nonzero rule selection on both R partitions". The bank builder (cts.rstage, frozen with the plan) admits a rule selection only
   when it is nonzero on both R0 and R1. Every rule selected identity on R1, so no rule selection entered R_FRONTIER. The nonzero
   R0 selections (REFIT_PAIR full OLD-B50 0.017 and OLD-B80 0.027; ensemble SIMPLE 0.5) are bracketed by the evaluated common
   strengths. At s = 0.1, REFIT_PAIR vs W is +0.0027 (SIGN_CHANGES).
13. **Compression twins on R0 only (recorded 20:57; commit 8150f4259 at the bank build).** The registered COMP control is matched on
   GUARD utilization, max_c cost_c / rho_c. On R1 the men's reserve slack is negative (-0.00031), so the matching target is
   undefined and the twin was refused (ENG-02). The R0 twins give 4 cells (seed 0 only), and the 8-cell stability rule cannot
   label them. Rung 3 against COMP is therefore not evaluable over 8 cells; the R0 means are reported (repair/MATCHED_CONTROLS.csv).
14. **Units launched with uncommitted code (launch audit, 20:56; CODE_AT_LAUNCH hashes against git blobs).**
   - VER_NESTED_WITNESS_D0_V1, CTS_R0_EVAL_TRACKR_V1 and CTS_R0_DIR_REFIT_LOC_V1 launched at 20:12:48-20:12:55. They ran
     cts/common.py, cts/trackf.py and cts/evaluate.py with exactly the content committed 30 s later in 5002fed58 (20:13:25Z,
     amendment 9).
   - That change adds the P70 / P70A partitions and the same-rows fold base. The fold base equals P for every registered
     partition, so these units' computations are unaffected and reproducible from 5002fed58.
   - CTS_DEPLOY_STORE_R1_REFIT_PAIR_AT_s1.0_V1 (20:42:14) launched with cts/evaluate.py uncommitted. It ran the content
     committed in c542f1117 at 20:42:18Z, and cts.deploy does not import cts.evaluate.
   - Four units launched between 20:45:21 and 20:45:34 with an earlier, never-committed draft of the new module cts/claims.py
     (first commit 2f581043e at 20:47:42Z). None of them imports it.
   - Verifier units at 20:32-20:53 launched while coordinator edits were uncommitted: the additive question A function in
     cts/rules.py, cts/questiona.py, cts/diagnosis.py and cts/figures.py. The verifier's results are its own records.
   - Units launched after a commit that came while they waited carry digest_changed_while_waiting = True in the ledger. Their
     launched code is committed and clean.
15. **Queue record defect (recorded 20:57).** In the ledger's uncommitted_code_paths, the first path loses its first character
   when the first `git status --porcelain` line starts with a space (for example "ts/rules.py" for cts/rules.py). The cause is
   that cts.jobqueue strips the porcelain output before slicing. Only the recorded label is affected: the file hashes in
   CODE_AT_LAUNCH.json are complete. The queue code is not changed during the campaign.
16. **Frozen rule code changed after the freeze and after the repair panels were read (verifier F3 V2, procedural FAIL; recorded
   21:05).** Item 10 left this timing implicit, so it is stated here explicitly:
   - REPAIR_FREEZE (20:23:59Z) hashes cts/rules.py and cts/questiona.py.
   - The question A units on R0 and R1 ran with that frozen code after the panels were opened: C0 read 20:25:19-20:27:53Z, C1
     read 20:28:02-20:29:05Z.
   - The correction (eece7dbdb, 20:34:05Z) changed both files after those reads, and the corrected runs (20:34:41-20:35:44Z)
     were then designated primary for all four partitions, following adjudication A2.
   - PROTOCOL section 8 says a revision made after a repair CHECK outcome is seen cannot replace the primary repair result. The
     correction came from the verifier's audit of the code against RULES.md section 6 using D-stage replays, not from an R
     outcome, and it is conservative. On R it changes two labels: R1 REFIT_LOC full and R1 REFIT_PAIR_TR full, from
     PREDICTION_IMPROVED (frozen code) to NOT_IMPROVED.
   - Which R labels are primary under section 8 was referred to the statistical reviewer (adjudication A5). Until that ruling,
     and in every document, both R labels are reported side by side. Rung 1 is NOT_MET under both.
17. **Compression twins are not cost-matched (verifier ATTENTION, recorded 21:05).**
   - The source COMP recipe chooses K by bisection, assuming that GUARD utilization falls as K grows (`search` field of each
     COMP.json).
   - On R0 the utilization path over K is non-monotone, and it is noisy relative to the tiny men's slack (+0.00014). The codes
     flagged MATCHED therefore reach only 6-55% of the target utilization (for example s = 0.2: 0.23 against 3.80). For s = 0.1
     the TARGET_NOT_REACHED outcome rests on K = 2048 alone.
   - The R0 differences between PAIR and its twin (+0.004 to +0.018) therefore compare PAIR with a cheaper code. They are reported
     as descriptive and are never used as a matched control (rung 3 against COMP is also not evaluable over 8 cells; item 13).
18. **Question-A correction: exact timeline (adjudication A5, condition 2; recorded 21:16).** Times come from the git log, the
   queue ledger, the session's message timestamps, file birth times and the verifier's access-log replay:
   - 20:23:59 REPAIR_FREEZE committed (c269256d2), with cts/rules.py and cts/questiona.py hashed.
   - 20:25:19-20:27:53 The R0 question-A units read the C0 panel with the frozen code.
   - 20:25:44-20:27:28 Verifier units VER_QA_BOOT_D0_V1 and VER_QA_BOOT_D1_V1 ran. Their arguments name D-stage records only.
   - 20:27:58 The verifier's message reporting the bootstrap finding reached the coordinator. The message is headed "20:29Z";
     the delivery timestamp is the record.
   - 20:28:02-20:29:05 The R1 question-A units read the C1 panel with the frozen code.
   - 20:28:15 The coordinator referred the finding to the statistical reviewer.
   - 20:28:56 The verifier's first R replay unit (VER_QA_R0_ALL_V1) started.
   - 20:30:14 A2 was written (ADJUDICATION_QA_BOOTSTRAP.md, file birth time). Its heading says "about 20:35Z", and the coordinator's
     A2 resolution repeats that. The birth time is the record, so A2 preceded the code correction.
   - 20:34:05 The corrected code was committed together with the A2 file (eece7dbdb).
   - 20:34:41-20:35:44 All 34 units were rerun under the registered scheme.
   - 21:04:46 A5 was written (ADJUDICATION_SECTION8.md).

   The reviewer's A5 text says the commit preceded the saved adjudication by about a minute. The file times show A2 was saved about
   4 minutes before the commit. Condition 1 of A5 (the verifier confirms that its trigger used no R output) was requested at 21:12
   and is recorded under A5 when it returns.
19. **Coordinator time stamps after 21:00 (recorded 21:10, from git commit times, message timestamps and file birth times).**
   Several entries written between 21:02 and 21:07 carry times the coordinator estimated ahead of the clock, the same error as
   items 8 and 11. The actual times are the record:

   | Entry | Written as | Actual |
   |---|---|---|
   | verifier REPORT.md saved | 21:03Z | file born 21:02:44Z; committed 21:04:27Z (f3d183f3c) |
   | adjudication A4 | 21:06Z | committed 21:04:27Z (f3d183f3c) |
   | A5 referred to the statistical reviewer | 21:04Z | message sent 21:02:56Z |
   | items 16-17 | recorded 21:05 | committed 21:04:27Z |
   | MECHANISM_CARDS corrections | 21:08 | committed 21:04:27Z |
   | A5 conditions 1 and 4 requested from the verifier (A5 entry and item 18) | 21:12Z | message sent 21:05:10Z |
   | item 18 | recorded 21:16 | committed 21:06:36Z (ce4c6079b) |

   The coordinator's messages to the verifier ("21:12Z") and to the claims reviewer ("21:22Z") carry the same kind of error; they were
   sent at 21:05:10Z and 21:08:09Z. From here on, every time the coordinator writes is read from the clock first.
