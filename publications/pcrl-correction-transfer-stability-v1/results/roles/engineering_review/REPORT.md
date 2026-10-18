# Engineering / deployment review (ENG_REVIEW_V1)

Saved by the coordinator at 19:47Z from the engineering reviewer's returned text (the platform refused the subagent's own write of
this file: "subagents return findings as text"). CHECKS.json, PATCHES/, and tests/ were written by the reviewer. Reviewed tip:
fde40f058 (D0 / D1 witnesses, routing, directions, EVAL and NESTED existed; no LADDER, SELECT or REALIZED file yet).

**Summary.** No BLOCKER and no exclusion leak. 5 MAJOR and 15 MINOR findings over 46 checks. All 11 patches apply with
`git apply --check` and compose; the coordinator applied all 11 at 19:46Z, before any ladder, selection or realized outcome of a
correction was computed (commit recorded in COORDINATOR_LOG.md).

## MAJOR findings

- **ENG-01: ensemble outcomes judged against the full witness's margins** (`PATCHES/1.diff`; cts/select.py, cts/dstage.py). For
  `RULE|ens`, `realized` measured e_s - e_0 and compared it with the reserve and margin of the deployed full witness W; the released
  ensemble spends (e_s - e_0) + (e_0 - W). The gates G2 / G3 and the tie-break reserve ratios were therefore not computed for the
  release that would be deployed. Patch: records `outcome_complete` (increment against W) and the gates use it for ens; transfer
  diagnostics keep e_s - e_0. For the statistical reviewer (rules frozen, not patched): the ens rules budget e_s - e_0 against W's
  reserve; ens "identity" is e_0, not the witness.
- **ENG-02: a compression twin with an infinite target labelled MATCHED** (`PATCHES/2.diff`; cts/compression.py). With a
  nonpositive reserve slack (e.g. D1's men's guard) the candidate's utilization is infinite and the inherited bisection returned K = 2
  labelled MATCHED. Patch: refuses to build the twin and records the bisection's monotonicity assumption.
- **ENG-03: the R stage would drop a FROZEN or ensemble recipe** (`PATCHES/3.diff`; cts/rstage.py). The R bank and lane hard-coded
  REFIT routing and full-deployment rule keys. Patch: reads RECIPE_SELECTION.json and adds the recipe, its matched CONT and FROZEN
  directions. Outside rstage: for a FROZEN recipe its R0 / R1 directions, GUARD ladders and selections must exist before the freeze.
- **PROV-01: a timed-out unit's CPU was never charged** (`PATCHES/20.diff`; cts/jobqueue.py). On timeout the queue killed without
  waiting, so RUSAGE_CHILDREN charged 0; an interrupted runner freed its slot without a unit_end. Patch: reaps the child, records
  status 'interrupted', records the code digest at unit end.
- **PROV-02: closeout compared local-offset git times with UTC log times** (`PATCHES/23.diff`; cts/closeout.py). The access-receipt
  certification of closed panels could miss a read up to 4 h before the freeze (load() still enforced the lock, so no result was
  affected).

## MINOR findings

- ENG-04 (`PATCHES/4.diff`): freeze content guards (missing RECIPE_SELECTION, R partitions without directions or selections;
  `repair_frozen()` now requires a freeze written by cts.freeze; more code hashes in the freeze).
- ENG-05 (`PATCHES/5.diff`): the ladder's `mean_abs_logit_change` was 5x the RULES section 5 definition; the registered per-class key
  is added beside it (D0's NESTED_PAIR.json already used the old key).
- ENG-06 (`PATCHES/6.diff`): D0B now follows a D0 tag suffix (latent); partition / prole on non-MODEL_FIT roles is refused.
- ENG-07 (no patch; narrowing the registered policy needs an amendment): `source_replay` is authorized for every function in
  cts.trackf; only routing_repro uses it. The verifier should assert every source_replay log line's entry point.
- ENG-08 (no patch): `diagnosis.factors` z-scores treat same-people pairs (NESTED vs OOF on FIT; CHECK vs CHECK_ENS) as independent.
- ENG-11: COMP_REALIZED.json is written once per partition, so the lane must run it after every COMP fit.
- PROV-03 (in PATCHES/20.diff): code hash after launch. PROV-04 (`PATCHES/21.diff`): NaN or negative charges. PROV-05
  (`PATCHES/22.diff`): stale interpreter-version cache.
- DEP-05, DEP-07, DEP-08 (`PATCHES/25.diff`): the W package's wire is now tied to the evaluated witness; RELEASE_CODE includes
  ohn.deploy; the deployment-check report is write-once.
- No patch: DEP-09 (PACKAGE.json is an unpinned root of trust), DEP-10 (`_TR` directions could be packaged with H3 income), PROV-11
  (H3R + HCR counted as one specification; the 72 cap is not enforced before a fit).

## Checks that pass

Access layer (registration gate, role x field x operation policy, row restriction, panel lock; all 26 label reads in coordinator
code go through `load`); exclusion (all fits on FIT only; GUARD only by select; CHECK only by the check operations; MODEL_SELECT only
for the registered design inputs and outcomes); estimators (OLD uses OOF; SIMPLE / UNCERTAINTY use GUARD or GUARD_ENS; bases match on
the same people; RULE|dep keys agree across modules; selections never read CHECK or MS outcomes - D ladders compute them earlier but
nothing reads them; G3 exact for the full deployment; the nested pipeline truly nested); deployment (missing, extra, reordered, target
and SEX fields refused at both entry points; hidden outputs refused; hashes checked before any pickle is loaded); receipt (8 / 8);
queue (append-only ledger, window limits, slots).
