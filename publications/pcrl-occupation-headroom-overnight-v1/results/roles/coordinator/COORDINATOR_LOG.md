# Coordinator log (append-only, UTC)

- 04:36:11Z campaign start (first action; state inspection: no AGENTS.md; origin = GitHub; the only occupation-named branch
  belongs to the earlier Adult lineage). Worktree and branch `research/pcrl-occupation-headroom-overnight-v1` from the source tip
  2af3c9db2.
- 04:43:52Z queue opened (14 CPU-h, 8 h research + 30 min closeout, 2 slots). 04:43:52Z OHN_ADMIT_V1 FAILED (a source control,
  OC_TBCAL, exists for seed 0 only; the plan assumed both seeds); 04:44:05Z OHN_ADMIT_V2 PASS (201 copies, 0 problems).
- 04:45-05:10Z coordinator infrastructure: scorer, ledger, reader grid (self-test PASS; code lock 05:00:09Z), deployment module
  (anchor package test: bitwise, every refusal at both entry points), protocol, exposure table.
- 04:47Z wave 1 launched: statistical reviewer, investigators A / B / C, closest-work reviewer.
- 05:07Z investigator A found, before any fit, that seed-independent inner folds with a deterministic learner would make the
  seed-0 and seed-1 occupation releases identical; A changed the initial design to seed-specific inner folds before registration.
- 05:08Z statistical reviewer finished (rules, selection rule, claims draft; 24/24 invented-data tests). Decisions on its open
  questions: keep the income-branch leg B (DEV-guided, disclosed); proceed with the final stage as written if a finalist is
  nominated; the grid carries bootstrap SEs for all three views; SUCC / TWJ are combined frozen releases.
- 05:10:39Z registration commit b1fd591d0 (cards of A, B and C included). 05:11:05Z public receipt 435fe44e5 pushed;
  05:11:09Z verified on the remote; label gate open. First label read 05:11:36Z (investigator A, MODEL_FIT occupation).
- 05:11Z references scored on MODEL_SELECT (both seeds; reproduce the source's published values). Every task-only reference's
  binding condition is the men's occupation Brier guard (H3+T_TREE 3.33 / 2.60 SE; TASK_REFIT 3.41 / 3.19); SUCC 0.32 / 0.62.
- 05:12Z protection investigator launched (code and card first; real-data protection fits only after witness selection).
- 05:27Z reference reader grid, seed 0 (4,224 reader fits, 815 CPU-s): SCORE pair means H3+T_TREE 0.636, TWJ 0.634, H_LJ 0.622,
  NEURAL_RAW 0.619, B2TWIN 0.610, SUCC 0.600, H3+O1 0.600, RETIRED_B3CTL 0.596. Lesson: learning-curve fractions on a 10-release
  bank tripled the cost and held a queue slot for 14 minutes; later candidate banks use the full fraction only.
- 05:27Z investigator A: A0_HGB_S seed 0 worse than the teacher on MODEL_SELECT (occupation NLL +0.0086 vs T_TREE): a
  variance / overfitting failure (sharp early optimum, committee gain); a MODEL_FIT-only diagnostic precedes the revision.
- 05:35Z closest-work reviewer finished (text report; its agent instructions forbade writing the report file, so the coordinator
  saved it as roles/closest_work/EVIDENCE_TABLE.md with a disclosure). Verdicts: every family ESTABLISHED; the coalition-aware
  occupation step is a STANDARD ADAPTATION of conditional adversarial debiasing; the threat model is standard; nothing
  potentially new at the method level.
- 05:40Z ENGINEERING REPAIR of the queue (reported by investigator C, whose first fit had waited since 05:12Z): slot acquisition
  was a lottery in which a chain of units re-took its slot immediately; `ohn/jobqueue.py` now uses FIFO tickets (only the oldest
  live waiter may take a free slot; dead tickets discarded). Units already waiting kept the old code until they started. No result
  is affected (scheduling only). C's fit started 05:41:40Z.
- 05:42Z A0_HGB_L seed 0 also worse than the teacher (occupation NLL vs U0 +0.0001; men's guard +0.0025; ratio 1.37).
- 05:43Z independent verifier launched (separate code; checks begin with admission, roles, exposure and reference utility).
- CORRECTION (05:57Z, from the verifier): the first label read after the gate was 05:11:30Z (OHN_REFERENCES_S0_V1, MODEL_SELECT),
  not 05:11:36Z as written above; both are after the 05:11:09Z receipt verification. The scorer's cross-fitting diagnostic reads
  MODEL_FIT SEX for the per-SEX cross-fitted utility (`oof_by_sex`): a logged diagnostic use of MODEL_FIT SEX, not a fit.
- 05:45Z stage-1 seed-0 results: C1 residual on the teacher margin ~ the teacher (NLL vs U0 -0.0122; men -0.0010); B1_STACK -0.0131
  (men -0.0009); B1_SINGLE -0.0118. No learner beat TASK_REFIT (-0.0138) and none improved the binding men's guard.
- 05:55Z family A revision 1 (POBP as an ordinal place-of-birth code; committed before its fit at cb89a8b51): A1_PO_16 seed 0
  NLL vs U0 -0.0161 (vs T_TREE -0.0040, vs TASK_REFIT -0.0024), men's guard -0.0039 (teacher -0.0010); A1_PO_8 -0.0144 / -0.0034.
  First real occupation headroom, including the binding men's guard.
- 05:55Z VERIFIER FINDING (reference grid only, before any candidate grid): every H3-anchored release fails the frozen nomination
  gate RECIPIENT_POINT through its income view (H3 income view minus H_LJ's +0.0309, minus TWJ's +0.0147 in cells S0R1k0 / S0R1k1;
  limit 0.010 per cell). Decision: the gate stays frozen (relaxing it after seeing reference outcomes would be an outcome-informed
  rule change). Consequence: no H3-anchored configuration can be nominated; only an income-branch configuration could pass; the
  protection stage still runs for the component questions.
- 06:00Z C's controls on seed 0: continuation from the teacher margin = the teacher (NLL -0.0122); vector-scaling calibration on
  CALIBRATION worsens the men's guard (+0.0017; ratio 1.82), as ordinary calibration did in the source campaign.
- 06:01Z A1_PO_16 seed 1 NLL vs U0 -0.0128 (vs T_TREE -0.0046), men -0.0023; A1_PO_8 seed 1 -0.0112, men -0.0019; B2_STACK seed 0
  (B's revision with the same ordinal POBP coding, committed 2f785ec18 before its fit) -0.0167, men -0.0035. 06:01:42Z C's
  revision 1 committed (8f244b3d2) before its fits (cross-fitted teacher margins).
- 06:03Z DECISION (before any witness selection): stage 1 closes for witness selection at 07:15Z (about the brief's 2.25 h of
  task learning from the gate); configurations scored on both seeds by then are eligible; later results are reported only.
- 06:12-06:27Z family A stopped after revision 1 (6 of 12); family B stopped after revision B3 (10 of 12; ledger entry 80 text says
  06:27Z, corrected by entry 81 to 06:21:39Z); family C used 12 of 12 and confirmed nothing pending (06:26Z). Investigator A
  took the conditional income branch (card and code committed 5c9c6cef5 before any fit; fits wait for the trigger).
- 06:28Z DECISION: stage 1 closes now instead of 07:15Z, because all three families have declared their task-model work complete
  and no configuration is pending, so the eligible set cannot change. Witness selection by the frozen rule follows immediately.
- 06:27:13Z WITNESS_SELECTION.json (unit OHN_WITNESS_SELECTION_V1; commit c292353dc 06:27:22Z): SELECTED_WITNESS B/B2_SINGLE (min
  occupation-side ratio 4.418) and A/A1_PO_16 (4.299); then B2_STACK 4.158, A1_PO_8 4.094, B3_CALTR 3.634, B1_STACK 2.933, C2_XF_D1
  2.656, C3_XC_D1 2.475, C3_XC_D2 2.412, B1_SINGLE 2.312 (all PROMISING_WITNESS). Income-branch trigger FIRES (leg A false; leg B true,
  DEV-guided). The verifier recomputed the selection independently (all labels, 392 ratios, ranking, trigger, timing) at 06:31Z.
- 06:27:38Z first protection fits (B2_SINGLE seed 0; after the selection record); the two first units were stopped by the protection
  investigator after about 50 CPU-s each to change a record field (failed attempts kept) and relaunched as _V2 at 06:29Z.
- 06:29:30Z income-branch witness constant committed (721a64c15); GO given for the income branch (Step-1 POBP-ordinal repair of the
  source's STRONG income learner; matched task-only twin INC_TW_PO; INC_H3_PO = the source H3 step unchanged).
- 06:29Z witness reader grid (WITNESS_BANK, both seeds) launched by the coordinator; reader-robustness investigator launched.
- 06:40Z witness grid seed 0: stronger task-only occupation learning leaves SEX recovery at the teacher's level (occupation view
  0.630-0.636 vs 0.633; pair beside H3 0.634-0.643 vs 0.636).
- ENGINEERING RECORD (06:41Z, investigator A, income branch): units OHN_INC_STEP1_S0_V1 / S1_V1 exited with a TypeError in the final
  console summary print AFTER their write-once outputs (STEP1.json, HGB_PO.pkl, CROSSFIT_INCOME.npz) were written and hash-verified;
  queue status FAILED; no rerun; the print line was fixed in ohn/fam_inc.py (summary only, no computation) and logged in the card.
  Seed-0 Step 1: the source's Step 1 reproduced exactly (selection, tree counts, OOF Brier difference 0.0, HGB bitwise on all roles);
  the POBP-ordinal repaired income learner vs the original in household cross-fitting: Brier -0.00288 (SE 0.00066), NLL -0.00422
  (0.00082), accuracy loss -0.00225 (0.00163).
- 06:44Z first protection scores (B2_SINGLE seed 0): PAIR_B50 occupation NLL vs U0 -0.0119 (witness -0.0147), men -0.0028;
  PAIR_B80 -0.0094, men -0.0019.
- 06:47-07:00Z protection revision 1 (P2, no in-training guard) committed 06:52Z (first fit 06:51:00Z; deviation 10); verifier
  confirms every H3-anchored release fails the per-cell recipient gate through its income view (recorded before any candidate grid).
- 07:02Z reader robustness withdraws its "benefit shrinks" finding (frozen-winner selection noise); correction relayed.
- 07:15-07:21Z engineering early notice (nomination deployment gate, tables, uncommitted protect.py); P3 (static per-SEX weights,
  both witnesses) proposed and amended, committed 07:21:30Z (06b0ca47a) before any P3 fit; deviation 14 for units that ran the
  uncommitted file.
- 07:17-07:21Z the platform safety monitor flagged the verifier (logging / audit tampering): its transcript shows no write to any log,
  ledger or access file; it had widened its own access-check rule after FAIL results. Asked to restore and disclose (deviations
  15-16, records/ACCESS_CHECK_AMENDMENT.json).
- 07:24Z two-seed P2 / P2A grids: the pair critic beats its continuation and compression controls on both seeds, not SUCC /
  RETIRED_B3CTL; only B50 keeps the reserve on both seeds. 07:26Z reader robustness: the seed shift is a release effect.
- 07:30-07:44Z deployment batches 1 (34 packages) and 2 (38, after the engineering patch f4d6ad65f); three leftover DEV_REPORT
  raw-input CSV copies removed from the system temp directory.
- 07:31Z P3 seed 0: predictions (i), (iii) confirmed, (ii) refuted on both witnesses, (iv) split; no P3 seed 1, no P3 COMP twin.
  07:37Z reader-robustness agent terminated by a platform safeguard ([bio] flag) during its closing text (deviation 20; not
  relaunched). 07:45Z protection stage complete (PRIV24 23, CTRL12 10).
- 07:45:00Z formal nomination (34 configurations): NO_QUALIFYING_CONFIGURATION; DEV_REPORT stays closed; no final stage.
- 07:48-07:56Z verifier resumed: nomination replay 34 / 34, newest releases and grids pass; B3_CALTR manifest gap found and fixed
  (27f080dde); deployment batch 3 (final, 38 packages) all bitwise, every refusal. Decision documents drafted; claims reviewer
  launched (about 07:55Z).
- 07:59-08:05Z backup B1 (same-device; restore check from the backup alone: 201 copies, 65 release records, 38 packages bitwise,
  every refusal); the verifier redeployed all 38 packages from its own restored copy of B1 (760 / 760 refusals, no unlisted runtime
  read) and deleted the copy.
- 08:15Z claims review returned (no change to the outcome; four sub-conclusions corrected: the reserve-keeping frontier and
  P2_LOC_B80, B80-only control wording, family B seeds not refits, every failed gate). All items accepted; NOMINATION_FRONTIER.csv
  written (ohn.frontier); documents revised and committed 08:22Z (17a1c6580); second pass requested at 08:23Z.
