# Independent verifier, first hand-back (saved verbatim with attribution)

Author: the independent verifier of campaign `pcrl_finite_routing_comparison_v1` (separate code under `roles/verifier/code/`).
Returned at about 00:30Z on 2026-10-21 (its own header: written 00:30:03Z; work from 23:44:46Z). Saved verbatim by the coordinator
(indentation removed). The coordinator's dispositions of its non-PASS checks are deviations 7 and 8 in PROTOCOL_DEVIATIONS.md.

---

**Independent verifier, first hand-back (written 00:30:03Z on 2026-10-21; work from 23:44:46Z)**

**Bottom line.** Everything decisive up to the qualification decision, and the protected-stage tables, reproduces with my own code. There is no numerical FAIL.

CHECKS.json holds 92 append-only checks:

| | PASS | FAIL | ATTENTION | NOT_APPLICABLE | NOT_RUN |
|---|---|---|---|---|---|
| Numerical | 80 | 0 | 0 | 11 | 1 |
| Procedural | 57 | 2 | 3 | 30 | 0 |

All non-PASS checks are listed below. Code: `results/pcrl_finite_routing_comparison_v1/roles/verifier/code/` (vlib.py, v1_access.py, v1_admission.py, v1b_card_amendments.py, v2_routing.py, v3_nested.py, v4_utility.py, v5_selection.py, v5b_protected_tables.py, v5c_exact_aliases.py, v5d_unt_selection.py). Every script has an invented-data selftest, all passing. No frc/nrm numerical function is imported.

**V1: access, admission, receipt**
- **Receipt (PASS).** My own ls-remote of publication/pcrl-finite-routing-comparison-v1 equals the receipt commit 7fed2ca44. The registration files are committed, unchanged, and byte-equal to the research commit 98ede167e and to the published copies.
- **Source admission (PASS).**
  - The NRM pin 682068d52 is the remote tip.
  - TASK.pkl, MODEL.pkl and RELEASE.npz of PRC_TWIN and PRC_TWIN_C128 (V0/V1) were re-hashed against their own records and equal SOURCE_ADMISSION.json.
  - The GATE records equal their published blobs.
  - No source file was modified after 22:49:23Z.
  - All 26 reference MODEL_SELECT wires and the library versions match the admission record.
- **Access log (audit V3, 27 reads, every one after the receipt).** No closed role was read. Entry points and module chains are correct, no fit-stage unit read MODEL_SELECT, row sets equal the admitted roles, and every read is inside its unit's interval.

**V2: fixed-routing constructions (T2), all PASS**
- My own out-of-fold routing and B1 tables equal the TWB_C64 / TWB_C128 stored tables bitwise (difference 0.0) in both versions.
- The A decoders recomputed from MODEL_FIT labels equal the source decoders bitwise. So my partition is the one the source counted, and A and B1 count the same people.
- There are no unoccupied cells. The TWB book equals the source book and the teachers' pickled bytes are equal.
- On all six roles:
  - the TWB and A releases induce the same partition (income, occupation, pair);
  - each TWB vector equals B1[cell] bitwise;
  - distinct vectors equal occupied cells;
  - nobody is routed into an out-of-fold-empty cell.
- My own replay of the deployment function on all six roles reproduces both the A and the B1 releases bitwise: the saved teachers' decision functions, then my own softmax/clip, then my own routing.
- The minimum best/second-best routing gap is 8e-8 or more, so arithmetic order cannot flip an assignment.
- Full-pool max |B1−A| is 0.090–0.145 on occupation; |B1−B0| is 0.0019–0.0062.
- Joint out-of-fold vs deployed cell agreement on MODEL_FIT is 0.52 (16×64) and 0.39–0.40 (24×128). These are both cells together; the protocol's 56–76% is presumably per task.

**V3: nested comparison (T1), all PASS**
- Every derivable number of the four NEST records reproduces from the per-person files, with a largest difference of 4.3e-19. That covers means, spends over CONT with my sandwich SEs, gate labels, paired differences, my own outer folds, fold-level SEs and SE_used, shares, and per-fold entries.
- The GATE reproduction (20 numbers per record) has a largest difference of 4.3e-19, with the FAIL labels matching.
- A label-free consistency check passes: within each fold and cell, A, B0 and B1 emit cell-constant vectors on the same stored cells.
- Nested B1−A occupation NLL (SE_used):

| Capacity | V0 | V1 |
|---|---|---|
| 16×64 | −0.00388 (0.00083) | −0.00486 (0.00101) |
| 24×128 | −0.00848 (0.00131) | −0.00900 (0.00113) |

- Q-PE is PE_ERROR_SUPPORTED at both capacities and the construction is B1 at both, as recorded.

**V4: MODEL_SELECT utility and the decision (run tag V3), all PASS**
- Coverage: every evaluation record (TWB ×2, source ×2, P2 ×7, UNT_BR, ablations) plus 11 references.
- Agreement: clause/guard values within 1.7e-16, SEs within 9e-19, paired costs within 4.3e-19. Reserve, reserve_rules, absolute metrics, distinct counts and wires all agree.
- MODEL_SELECT paired occupation NLL B1−A: −0.00374 / −0.00546 at 16×64 and −0.00722 / −0.00618 at 24×128.
- QUAL at 16×64:
  - B1 is QUALIFIED; its V1 men's guard is 0.000185, leaving 0.00181 of slack after the S2 increment.
  - A is NOT_QUALIFIED with Q2_FAILED_V1: guard 0.002552 against a limit of 0.002.
- QUAL at 24×128: both constructions qualify.
- From my own numbers: PROTECT at 16×64 with B1, and D1 = REPAIRED_TASK_ONLY_RELEASE, matching QUAL_DECISION.json.

**V5, done as far as the saved records allow**
- **V5a (P1 selection, PASS).** From the recorded audit, I reproduce the selection rule in registered path order, the aliases (S3 = S2 in both versions), the bracket target, spends and brackets, and the F4 decision. Audit estimates equal the nested-mean differences within 2.8e-16.
  - F4 passes: diff 0.00715 (SE 0.00225) in V0 and 0.01113 (SE 0.00231) in V1.
- **V5b (PASS on all 18 configuration-versions).** I reimplemented the protected, HQ, SHUF, ORD and UNT routing myself and recomputed the B1 tables on each configuration's own out-of-fold partition. They equal the stored tables bitwise, and every role's vectors are rows of those tables. Minimum gap is 2.5e-9 or more.
- **Not done.**
  - **P1 numerical replay (NOT_RUN).** I skipped F4's bootstrap SE, the nested spend SEs and the nested LOOKUP AUCs, as you allowed. P1 saved no per-person outputs, and I did not refit.
  - **V5d (UNT bracket replay).** Written and selftested, but not run: both slots are held by FRC_GRID_ROB. I stopped its waiting process at 00:30:03Z, before any unit_start.
  - **Grid replay and D labels.** Waiting for claims/*_ER1.

**Every non-PASS check and its cause**
- **V1.ACCESS_LOG__V2 / __V3, procedural FAIL.** Rule R6 (a card unchanged after use) fires on the 4 T1 fitting reads.
  - Card FA_FRC_v1.md gained Addendum A1 in b538fa7a4 at 23:56:09Z, after the T1 reads (23:48–23:49Z).
  - V1B (ATTENTION) shows the change is append-only: the logged bytes (98ede167e, sha 49056b46…) are an exact prefix of the current card and are recoverable from git.
  - The P1 reads at 23:56:21Z log the amended card.
  - I have not widened the rule; the adjudication is yours.
- **V1.UNIT_CODE_AT_LAUNCH__V3, ATTENTION (new rule R11).** No unit's entry package differs from its launch HEAD.
  - 23 units launched while other files were uncommitted. Among them are the P2 fitting units FRC_CFG_V0/V1 at 00:01:16Z, with frc/claims.py and frc/evaluate.py uncommitted.
  - Neither file is imported by the fitting path.
- **V5C.EXACT_ALIASES_V1, ATTENTION.** PRC_S1 (PRC@2.0) and PRC_S2 (PRC@8.0) in V1 are bitwise-identical on all six roles: same CI/CO, tilt 0.0625, only lam differs.
  - Both were saved because the registered alias rule compares selection names.
  - This matches your deviation 3. No configuration is an exact alias of its twin.
- **V5A.P1_PERSON_LEVEL_REPLAY, NOT_RUN.** No person-level P1 output exists (see above).

**Descriptive observations from the recorded P1 numbers (not correctness findings)**
- The PRC chain's nested means plateau: λ 0.25/0.5/1 are identical, and 2/4/8 are identical in V0 (4/8 in V1).
- PRC_S2 is the top of the grid, PRC@8.0, in both versions, so the S2 budget does not bind within the grid.
- SHUF is identical at all six λ: about 6% of people re-routed, total spend −0.00025, never reaching the target. This matches the stated failure in the R1 card.

**My disclosures**
- **Failed unit FRC_VER_UTILITY_V2 (00:10:30Z).** My V4 did not handle a comparator recorded as missing (FA/HQ_BR at V0). It read MODEL_SELECT labels once (logged), then crashed before writing anything. Fixed, and rerun as FRC_VER_UTILITY_V3.
- **Three selftest defects in my own invented cases, all fixed.** v3 and v4 failed their first selftest runs; both were fixed before any real-data run. Commit 3b7bfa8a1 carried a failing v5d selftest (the exit status came from `tail` in a pipe); the fix is the next commit. No acceptance rule was changed.
- **Two ad hoc out-of-queue, label-free looks.** The launch-code scan and the PRC_S1/PRC_S2 comparison were done by hand first, then repeated as queued checks (V1.UNIT_CODE_AT_LAUNCH__V3 and V5C).
- **Unit FRC_VER_UNT_V1 left unused.** Its directory holds only CODE_AT_START.json; no ledger event exists.
- **Admitted loaders used.** frc.common.features/load, ohn.fam_a.compact_po, ohn.devrun.reference_release, ohn.common.frozen_bank, tsz.evaluate source_a1/h3_income, and the saved teachers' decision functions.
- **Label reads (4, all inside queue units, after the receipt).** All use operation `verify`: one MODEL_FIT income/occupation read and three MODEL_SELECT income/occupation/SEX reads.
- All times come from `date -u` or the ledger; none was rounded ahead.

**CPU.** 14 verifier units, one of them failed, used 28.1 CPU-s; I charged a further 10 CPU-s in three charges. Total 38.1 CPU-s. The whole campaign stood at 0.795 CPU-h at 00:27:30Z.

**Files.** Commits are plain messages touching only roles/verifier/; the last is the V5d fix. The CHECKS.json and records were last committed in 1ffcd622e. Records: V2_ROUTING__V1/V2, V3_NESTED__V1, V4_UTILITY__V1/V3, V5A_SELECTION__V1, V5B_TABLES__V1, V5C_ALIASES__V1, plus history/.

**Next, when you tell me**
- V5d as unit FRC_VER_UNT_V2.
- The grid aggregate screens, frontier comparisons and D1–D6 from GRID.json / BOOT.npz once claims/*_ER1 exist.
- Classifier refits and deployment from the backup.
