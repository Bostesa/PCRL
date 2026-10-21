# Protocol deviations (campaign `pcrl_finite_solver_adequacy_v1`)

Departures from the registered PROTOCOL.md, cards and rules, in the order they were found. Each entry gives what happened, when
(`date -u`), why, and the effect on results. Pre-registration changes are in the coordinator log, not here.

(None at registration, 03:03:08Z.)

1. **The protocol overstates the solver selftest's coverage (engineering review S6; found 03:08Z).** PROTOCOL section 6 says SV1 "is
   checked for objective arithmetic, finite termination, empty cells, tie handling and rule consistency (`fsa.fa.solver.selftest`)".
   The selftest covers objective arithmetic, rule consistency and monotone acceptance on invented data, but it has no empty-cell,
   tie, duplicate-person or current-book-line case. The engineering reviewer's adversarial cases (12 invented configurations, 4,039
   intervals, an 801-point tilt grid) and the exact-reference investigator's laws exercise these paths, with no mismatch against
   `nrm.fa.prc.route` / `objective`. Effect: none on any value. The claim is corrected here, and the registered file is not edited.

Entries 2-6 were written at 2026-10-21T04:08Z (`date -u`), after the closing reviews.

2. **The first same-device backup failed its restore check (found 03:35:35Z).**
   - Unit FSA_CUSTODY_B1_V1 (`…_20261021T033518Z`) packed the committed code. At that moment the stage records (`replay/`,
     `audit/`, `source/`, `nested/`) were not yet committed, so the restore check from the backup did not find
     `replay/P1_REPLAY_V0.json` and reported `all_ok` false.
   - The records were committed (237c953a8), and FSA_CUSTODY_B1_V2 (`…_20261021T033709Z`) verified from the backup alone at
     03:37:32Z.
   - The failed attempt is kept and listed in `custody/BACKUP_VERIFICATION.json`. Effect: none on any value.
3. **Units launched while files under hashed code paths were uncommitted (verifier report 03:45:34Z; full ledger scan about
   04:00Z).**
   - The queue records `uncommitted_code_paths` at each launch. It lists them for 17 units:
     - five exact-reference units (NARROW_V1/V2, ENUM_V1/V2, SUMMARY_V1; 02:49-02:55Z), while the statistical reviewer's
       `fsa/rules.py` and `fsa/test_rules.py` were uncommitted;
     - the replay, audit, source and STEP 5 units (03:03-03:06Z), and the two nested units and two verifier units (03:23Z), while
       files under `roles/verifier/code/` were uncommitted;
     - FSA_VER_EXACT_V2 (03:34:53Z), while `fsa/restore_check.py` was uncommitted.
   - No stage imported an uncommitted file.
     - `fsa_exact.py` imports `fsa.rules` only in `run_step5`, and STEP 5 started at 03:05:44Z, after the rules commit 6e8594bf0
       (03:01:29Z).
     - No stage module imports the verifier's code or `fsa.restore_check`.
   - The verifier separately notes files outside the import closure that changed during the replay run (for example
     `fsa/fa/advance.py`).
   - The code digest at launch equals the digest at registration for 29 of 32 unit starts. The three exceptions:
     - FSA_EXACT_ENUM_V1 (02:49:13Z): the digest changed in the second between queueing and launch, while the reviewer was writing
       `fsa/rules.py`. Its end digest equals its launch digest, nothing changed during the run, and the enumeration does not import
       `fsa.rules`. No result uses ENUM_V1: `RESULTS_V1.json` is built from NARROW_V2 and ENUM_V2.
     - FSA_VER_P1_V0_V1 and FSA_VER_P1_V1_V1 (03:23Z): the verifier's units that failed on an import-path bug before any label
       read. They were rerun as FSA_VER_P1_V*_V2.
   - Effect: none on any value.
4. **The RULES 8 estimator pin and the "textually identical" clause (verifier ATTENTION items, 03:45:34Z; ruled 03:51:44Z).**
   - RULES 8 pins `frc/fa/route.py` 933d0d6f, but P1 ran under the b538fa7a4 version. The only change on the P1 path is that
     `emit` dispatches through `route_any`. That is behaviour-preserving for every book that is not a UNT book, and P1 has none.
   - The replay also re-implements the `stage_paths` selection loop with three renamed local variables. The verifier found it
     AST-identical after renaming.
   - The statistical reviewer ruled that `estimator_unchanged = True` stands, giving `P1_REPLAY_VERIFIED`. This is disclosed as an
     interpretation made after the result. The strict text gives `P1_ESTIMATOR_CHANGED`. The reviewer records the pin as an erratum
     to its frozen RULES 8, which is not edited (CLOSING_STATEMENT addendum 1).
   - Effect: none on any number. No decision reads this label.
5. **The advancement record omits the SV1_first control records (statistical reviewer C13, 03:49Z).**
   - RULES 9 lists "the SV1_first control records" among the advancement record's contents. `decisions/ADVANCEMENT.json` carries
     `equal_effort_attribution: NOT_RECORDED` and no controls.
   - The attribution was computed separately, at 03:33Z (`decisions/EQUAL_EFFORT_ATTRIBUTION.json`). SV1_first's route E input
     (`roles/exact_reference/route_e_input_SV1_first.json`) was not passed to the decision builder.
   - Effect: no label changes, because controls never advance (RULES 5.3).
6. **SV1_first's effort was comparable per step, not per chain (source records 03:23Z; both closing reviews).**
   - PROTOCOL section 6 says SV1_first "gives the old selection rule comparable objective-evaluation effort". On the source it
     reached the 30-step path cap, still improving, in 23 of 24 strength chains. It used 13-20 times SV1's CPU and 13-18 times its
     Lloyd steps.
   - The resolution-versus-selection attribution on the source is therefore confounded by the cap, and is reported as such. It was
     descriptive and never a decision input.
   - Effect: no label changes.
