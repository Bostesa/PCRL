# Protocol deviations (campaign `pcrl_dual_design_release_v1`)

Every departure from the registered text (PROTOCOL.md, RULES.md, the cards) or from the planned run order, with its time (`date -u`) and
its consequence. Amendments made before any fit are in `PROTOCOL_AMENDMENT_1.md` and are not repeated here.

1. **17:23-17:26Z, the statistical reviewer's completion time.** RULES.md first stated a completion time ahead of the clock (17:30:53Z
   when the clock read 17:23:55Z). Corrected by the reviewer to 17:26:03Z with a disclosure line, before registration. No rule changed.
2. **17:41-17:42Z, the protocol's revision time.** The coordinator first wrote "revised 17:28-17:50Z" at 17:41:46Z (an end time ahead
   of the clock); corrected to 17:28-17:41Z within the same minute, before the registration commit.
3. **18:15:3xZ, two waiting robustness units withdrawn.** The lanes for `DDR_GRID_ROB_V0` / `_V1` had been queued at about 18:10Z and
   were still waiting for a slot (no `unit_start` in the ledger; only their CODE_AT_START.json was written). They were stopped so that
   DB's short R1 fits (triggered on MODEL_FIT; card committed at 18:15:30Z) would run first. The same bank (`DDR1_ROB`, unchanged) was
   re-queued as `DDR_GRID_ROB2_V0` / `_V1`. Nothing was computed or read by the withdrawn units; their unit directories are kept.
4. **18:33:1xZ, the waiting robustness unit `DDR_GRID_ROB2_V1` withdrawn before start** (no `unit_start`; only its unit directory was
   written), so that the verifier's units, the registered claims, the data-rich scoring and DB R1's short evaluation / grid units do not
   wait about 30 minutes behind it. The unchanged bank `DDR1_ROB` (version 1) is re-queued as `DDR_GRID_ROB3_V1`. Nothing was computed
   or read by the withdrawn unit.
5. **18:41Z, DB_ZTILT's MODEL_SELECT utility record was missing.** The evaluation lane (`lanes/EVAL_ALL.json`) had been written before
   ZTILT was added to DB's controls (PROTOCOL_AMENDMENT_1), so `DDR_EVAL_DB_V1` did not evaluate it. Found by the table generator; the
   same evaluation is run as `DDR_EVAL_DB_ZTILT_V1` (utility and paired costs against the twin and DB_PTC). ZTILT is a control reported
   beside D2; no decision input changes.
6. **18:17-18:49Z, code committed while reader-grid units were running.** The queue ledger records `code_changed_during_run` for the
   grid units. The changed files were outside the grid's code lock (`DEVGRID_CODE_LOCK.json`): `ddr/db/ptc.py`, `ddr/db/ptc_r1.py`,
   `ddr/plan.py`, `ddr/report.py`, `ddr/restore_check.py`, `ddr/da/std.py`, `ddr/da/ipw.py`, `ddr/richreaders.py` and the
   verifier's code. A running unit had already imported its modules, and `ddr.devreaders` imports none of these files. The 15
   locked grid files were re-hashed at 18:55Z and are unchanged. `ddr/common.py`, which the grid also imports, has not changed
   since registration. No grid result depends on the changed code.
7. **19:13Z, DA revision R1 run although its declared trigger (b) holds only under a purpose reading.** DA_STD_v1 section 6 (b)
   requires "the registered comparison is NOT_SEPARATED" and a MODEL_FIT condition. In the registered rules `NOT_SEPARATED` means
   that no D2 comparison passes. DA_REG is `PARTIAL_SBC_FRONTIER_ONLY`: the SBC frontier passed, the SPEND frontier is
   `NOT_SEPARATED_FROM_TASK_SPENDING`, and TWIN is point-only. Strictly, (b)'s first condition fails. The MODEL_FIT condition holds
   in both versions (income R^2 ratio 0.666 / 0.661 > 0.5; `da/R1_TRIGGER_B2.json`). Trigger (a) does not hold. The revision is run
   under the task's standing authority for justified revisions that address an observed failure: D2 not met, and a measured
   residual propensity dependence in the income release. Card DA_STD_R1 states this. Its reading is ADAPTIVE_R1, reported beside
   the registered reading and never as the family result. It counts as one of DA's two revisions and as one D2 look.
8. **19:17-19:22Z, the strongest-release rule needed a scalar strength.** `ddr.rules.strongest_nonzero` (frozen) reads a number or a
   per-version number; DB_PTC_R1's strength is a recipient pair (alpha_I, alpha_O) per version, which the frozen helper read as
   `NOT_NONZERO`. The coordinator's script (`roles/coordinator/strongest.py`) passes alpha_I + alpha_O, the quantity DB_PTC_R1's own
   selection maximized; the rules are unchanged. With it, DB's strongest feasible nonzero release is DB_PTC_R1 (x = +0.0239 against
   DB_PTC's +0.0164), DA's stays DA_STD (DA_IPW +0.0072).
9. **19:21-19:27Z, two configurations added for DB's strongest release, each under its own card committed before its fit.** Card
   DB_PTC_R1 had said that ablations, ZTILT and SHUF are not refitted. RULES.md 8 and the task require a component ablation and
   matched refits for the strongest feasible comparison, and the task requires the simplest alternative explanation. So:
   - card DB_PTC_R1_ABL (7def05c9c) adds the occupation-only ablation DB_PTC_R1_OCC;
   - card DB_ZTILT_R1 (93da8ca06) adds a spend-matched ZTILT along DB_PTC_R1's split law.
   Both are part of the ADAPTIVE_R1 reading, descriptive and beside D2. They bring the configuration count to 48 of 48. Their refit
   banks (DDR1_ROB_DB_R1, DDR1_R1_ZTILT) were written after the configurations were saved and before any of their grid output.
10. **19:33:44Z, one reader-custody unit failed** (`DDR_READER_CUSTODY_R1_V1`). The pair winner of DB_PTC_R1 in cell S0R1k0 is the
    composite recipe `IGNORE_income__ET5`, which the inherited custody refitter (`ffv.readers.fit_new`) does not build. The same
    items were re-run in cell S0R2k0, whose winners are MLP64 (`DDR_READER_CUSTODY_R1_V2`, done). The failed attempt is kept in the
    ledger and counted.
11. **19:28Z, table generator merge.** `ddr.claims.merge_grids` replaces a whole cell when two banks share one. That is harmless for
    its own use: each D2R reading's minimum set lies in its robustness bank. The tables, however, merge several banks.
    `ddr/report.py` now merges cells member by member. Descriptive tables only; no decision input changes.
12. **19:47Z, record gaps found by the verifier's addendum** (`roles/verifier/VERIFIER_ADDENDUM_REPORT.md`, discrepancies 1-4). None
    changes a label.
    - (a) The beside records' single-view fields averaged all cells of a bank, and the pair fields only the 8 registered cells.
      `roles/coordinator/beside.py` now uses the registered cells for every field, and the three BESIDE records were rewritten at
      19:47Z.
    - (b) DA_IPW's lambda* differs by version (1.0 / 0.75), so PROTOCOL 4's flag `CAND_STRENGTH_DIFFERS_BY_VERSION` applies to
      DA_R1. The revision-plan writer did not record it (`PLAN_DA_R1.json` has no `strength_differs_by_version` field). It is
      disclosed here and in the documents. DB_R1's pair (0.35, 0.65) is the same in both versions.
    - (c) The `d2r.release` field of the three D2R records is null, because the plans' strongest-release stub has no `selected`
      key. The releases are DA_STD (DA_REG_D2R), DB_PTC (DB_REG_D2R) and DB_PTC_R1 (DB_R1_D2R).
    - (d) Under RULES.md 8 the family's D2R is that of its strongest feasible nonzero release (`claims/STRONGEST.json`): DA_REG_D2R
      for DA and **DB_R1_D2R for DB**. DB_REG_D2R, of the registered DB_PTC, was computed before DB_PTC_R1 existed. It is reported
      as an extra, descriptive reading.
    - (e) (closing statement C19; added after 20:01Z) DB_R1's D1 record lists DB_PTC_R1 under `ignored_zero_strength_entries`, with
      `fitted_strengths` empty. The frozen helpers do not read its strength pair [0.35, 0.65] as a nonzero number, the same cause as
      deviation 8. The list is used only when CAND fails G1, and DB_PTC_R1 passes G1, so no label changes. STRONGEST.json's
      "strength 1.0" for DB_PTC_R1 is alpha_I + alpha_O, not alpha = 1.
13. **Wording: DB_ZTILT_R1 is not "spend-matched".** Card DB_ZTILT_R1 and deviation 9 call it the spend-matched ZTILT. That was the
    intent: the registered first-reaching rule on the registered scale grid. In fact it spent 2.08x / 1.73x the release's MODEL_FIT
    spend and 1.7x its MODEL_SELECT cost (0.0169 vs 0.0099 nats), because the grid jumps from k = 1 (0.90x / 0.75x) to k = 1.5. The
    documents call it the release-only control DB_ZTILT_R1, intended to match the release's spend, and state the overspend beside
    its declared reading, which stands. The card is unchanged.
