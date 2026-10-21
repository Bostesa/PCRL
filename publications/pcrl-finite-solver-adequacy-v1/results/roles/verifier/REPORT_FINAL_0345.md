# Verifier's final report (saved verbatim)

The coordinator saved this file at 2026-10-21T04:06Z (`date -u`). The verifier did not write it. The text is the verifier's
hand-back message, received at 03:46:14Z (the report says it was written at 03:45:34Z), de-indented. One change was made: the
absolute worktree path in the file list is shortened to `<worktree>/`. The verifier's machine-readable record is `CHECKS.json`.
The coordinator's summary, with the dispositions of the non-PASS items, is `../../VERIFICATION.md`.

---

**Independent verifier, campaign pcrl_finite_solver_adequacy_v1. Full report V1-V6, written 03:45:34Z on 2026-10-21 (`date -u`). Work ran from 02:50Z.**

**Bottom line.** Every decisive number I recomputed with my own code agrees with the records. There is no numerical FAIL.
- The P1 replay reproduces FRC1's record bitwise. Under RULES 8 it reads P1_REPLAY_VERIFIED in both versions: I applied fsa.rules.p1_replay to my recomputation and got that label.
- I rebuilt the route S inputs from the nested npz. Applied to them, the frozen rules give the recorded labels: route S is S_PROXY_ONLY, and the advancement is OPTIMIZER_REPAIR_CLOSED.
- The one procedural FAIL is my own path-mapping bug, superseded by a passing rerun.

**Totals** (CHECKS.json holds 57 append-only, hash-chained checks; the chain verifies)

| | PASS | FAIL | ATTENTION | NOT_APPLICABLE | NOT_RUN |
|---|---|---|---|---|---|
| Numerical | 34 | 0 | 0 | 1 | 2 |
| Procedural | 8 | 1 | 9 | 0 | 2 |

Check ids end in `__V<version>__V<run>` for per-version checks and `__V<run>` otherwise.

**V1. Receipt, admission, access log**
- **Receipt.** My own `git ls-remote` of publication/pcrl-finite-solver-adequacy-v1 returns a771587ca, which is the receipt commit (verified 03:03:08Z).
  - All 8 registration files are byte-equal across the published copy, research commit f301965a7, HEAD and the worktree, and match the receipt hashes.
  - No later commit touches them, and RECEIPTS.json carries the same receipt.
- **Admission.** All 72 pinned artefact files re-hash to their pins.
  - The FRC1 (04ac9161f) and NRM (682068d52) publications on the remote equal their pins.
  - The estimator pins (prc.py 1192f34d, route.py 933d0d6f) equal the working files and the published copies.
  - frc/, nrm/ and their results are unchanged since 278cb1ed6.
- **Access log: 6 reads.** They are FSA_REPLAY_V0/V1, FSA_NESTED_V0/V1 and my two.
  - All are after the receipt and on MODEL_FIT only: no closed role and no MODEL_SELECT read.
  - Every entry point is authorized.
  - Card FA_FSA_v2 was committed at 02:46:39Z, before every fit read, and is unchanged.
  - Every read falls inside its unit's interval, and every row set is MODEL_FIT's (20,000 people, 13,337 households).
  - The audit and source units read no label.

**V2. P1 replay (decisive), both versions**
- My code agrees with both the replay record and SELECTION_V<v>.json:
  - means: 275 values per version, max |diff| 0.0;
  - spends and clustered SEs: 288 leaves, max 2.2e-19;
  - selection, brackets, nested_lookup_pair (73 leaves) and F4: bitwise equal;
  - no margin lies within 1e-9 of a threshold, so there are no boundary ties.
- The replay record matches SELECTION over 809 leaves (max 0.0; every discrete output equal).
- The npz households and labels equal the admitted MODEL_FIT data (my two label reads), and the per-person identities hold.
- **F4.** V0: 0.007152675309294199, SE 0.0022452309757188993. V1: 0.011134730134079729, SE 0.0023130609015072773. Entry PRC@8.0, passes in both.
- **Selections.** PRC S1 is None in V0 and PRC@2.0 in V1. In both versions: PRC S2/S3 = PRC@8.0, HQ = HQ@8.0, SHUF = SHUF@8.0, ORD = None / 12x48 / 8x32.
- **Brackets.** V0: HQ and SHUF did not reach the target, ORD@8x32. V1: HQ@1.0, SHUF did not reach the target, ORD@8x32.
- My recomputation in the P1 structure is in P1_VERIFIER_V0.json and P1_VERIFIER_V1.json. They carry audit_spend_over_twin, selection, F4, nested_lookup_pair and brackets. I left out nested_means because six of its keys need training-side partitions, and p1_replay would flag them as MISSING.

**V3. Exact laws**
- **All five laws pass.** 253 release records and 740 objective values (including every narrow trace step and the card/wide best partitions) are recomputed from laws.json. Max objective difference is 5.8e-14.
- **Own enumeration of D_witness.** 1,296 assignments, 36 feasible; the optima equal the investigator's.
- **Route E.** 60 cells, 240 reported partitions and 60 unrestricted objectives (max 4.4e-16). The usefulness deltas match ADVANCEMENT.json to 2.2e-16.

**V4. Objective audit and source check**
- **72 saved states** (both versions × PRC/HQ × original/SV1/SV1_first × 6 strengths) checked with my own objective:
  - my objective is within 8.9e-16 of the recorded one;
  - the partition hash, B1 and prototype cross-entropies, information, guards and re-routing all equal SOURCE_CHECK.
- **Rule consistency.** My own routing of each saved book reproduces its partition in 72 of 72 states, so every SV1 state is rule-consistent.
  - The smallest gap between a person's best and second-best routing cost is 4.7e-9 (V0) and 2.5e-9 (V1). That is far above arithmetic-order noise, but small.
- **SV1 against the original.** SV1 lowers the fitted objective at every strength: V0 by 1.2e-4 to 5.6e-3, V1 by 3e-6 to 1.05e-2. This concerns the surrogate objective only.
- **Audit.** My routing of the saved PRC_S2 / HQ_S2 / TWB_C64 releases reproduces the audit's numbers and its chain objectives.

**V5. Nested solver comparison**
- In both versions, means and AUCs differ by 0.0 and spends with SEs by at most 1.1e-19.
- The 13 original-solver entries equal the replay exactly: 117 arrays per version, max 0.0.

**V6. Route S**
- My nested inputs equal ROUTE_S_INPUT.json with 0 mismatches. In both versions the revised point is PRCsv1@8.0, the original S2 point is PRC@8.0, and the bracket is PRC@2.0.
- **Paired differences** (revised minus original):

| Version | dAUC pair vs bracket | d total NLL vs bracket | d total NLL vs S2 |
|---|---|---|---|
| V0 | −0.00181 (SE 0.00201) | −0.00061 (SE 0.00060) | −0.00061 (same point) |
| V1 | −0.00382 (SE 0.00170) | +0.00034 | −0.00072 |

- **Pooled condition.** Pooled dAUC is −0.00281 with pooled SE 0.00185, so estimate + 2 SE = +0.0009 is not below 0 and the recovery rationale fails. The task rationale fails because the S2 spend gains miss the −0.001 requirement in both versions.
- I checked the validity flags from git, the ledger and the access log myself.
- **Labels.** route_s on my input gives S_PROXY_ONLY; on the recorded input it also gives S_PROXY_ONLY. advancement with my S input and the recorded D and E inputs gives OPTIMIZER_REPAIR_CLOSED. **The label reproduces.**

**Every non-PASS check and its cause**
- **Procedural FAIL: V1.ADMISSION__V1.** My own bug: the key `results/designer_a/` points to `roles/designer_a/`. The four GATE files match their pins. Superseded by __V2.
- **Procedural ATTENTION ×9:**
  - **V1.ACCESS_LOG__V1 and __V2 (rule A9).** The four coordinator label-reading units launched while verifier files were uncommitted, outside their import closures. For the replay units that was my code folder; for the nested units, v3_exact.py.
  - **V1.ADMISSION__V2.** FRC1's previous verifier left an untracked empty lock file; it was already disclosed in FRC1.
  - **V2.ORIGINAL_CODE ×2.** Since the original P1 run (route.py 7ed1003f), `emit` now calls `route_any`. That is behaviour-preserving for the P1 books, and the replay is bitwise equal.
  - **V2.PROVENANCE ×2.** Uncommitted verifier code at launch, and files outside the import closure that changed during the run (fsa/fa/advance.py and others).
  - **V2.ESTIMATOR_TEXT ×2.** RULES 8 asks for the selection loop to be "textually identical". The replay's copy differs only in the local names sel2, k2 and v2, and its AST is equal after renaming them. If the reviewer reads the rule strictly, P1_ESTIMATOR_CHANGED would apply, although nothing changed numerically. This needs a ruling.
- **Numerical NOT_APPLICABLE ×1.** The nested_means training-side keys cannot be recomputed from held-out outputs.
- **Numerical NOT_RUN ×2:**
  - re-enumeration of the tilt-line breakpoints, and full enumeration of the m = 6 and m = 8 laws;
  - the audit's tilt-line Spearman and interval values, runtime in-sample vs out-of-fold agreement, expressivity, and refresh deltas.
- **Procedural NOT_RUN ×2.** The advancement's D and E inputs were taken as recorded (the route E objectives were not recomputed), and EQUAL_EFFORT_ATTRIBUTION.json was not checked.

**Label reads.** 2, both operation `verify` on MODEL_FIT (income, occupation, SEX), inside FSA_VER_P1_V0_V2 (03:29:35Z) and FSA_VER_P1_V1_V2 (03:29:54Z). Both are after the receipt. V5 compared SEX with the replay file instead of reading the labels again.

**CPU.** 14 queue units (4 failed) used 59.0 CPU-s. I charged another 182 CPU-s for selftests, dry runs and the rule application, about 0.067 CPU-h in all. The campaign stood at about 0.99 CPU-h at 03:43Z.

**Disclosures**
- **Failed units, all my bugs:**
  - FSA_VER_P1_V0_V1 and FSA_VER_P1_V1_V1: an import-path bug, before any label read;
  - FSA_VER_ACCESS_V1: exited 3 on the mapping-bug FAIL after recording its checks; scripts now exit 0 once checks are recorded;
  - FSA_VER_EXACT_V1: crashed on law E's derived definition before recording anything.
  - Each was rerun under a new id.
- **Out of queue:** invented-data selftests; label-free git/AST dry runs; one scratch V3 dry run on D_witness, not recorded and later repeated in the queue; p1_replay applied to JSON. All of it is charged.
- **Imports:** no fsa, frc or nrm numerical function was imported for any verified quantity. I read their code only for definitions. I used the admitted fsa.common features/load, the saved pickles (books, tables, cached predictions) and npz files, and called fsa.rules.route_s, advancement and p1_replay only on my own inputs, as you asked.
- **RULES:** before registration I read the uncommitted RULES.json draft to plan. Every real run took its tolerances from the committed RULES.
- **Git:** I committed only under roles/verifier/, with plain messages; the last commit is 3a4ffde37. records/.checks.lock is intentionally left untracked, and I did not push.

**Files** (all under <worktree>/results/pcrl_finite_solver_adequacy_v1/roles/verifier/)
- CHECKS.json
- records/
- P1_VERIFIER_V0.json, P1_VERIFIER_V1.json
- ROUTE_S_FITTED_V0.json, ROUTE_S_FITTED_V1.json
- ROUTE_S_VERIFIER_INPUT.json
- code/: vcore.py, v1_access.py, v2_replay.py, v3_exact.py, v4_objective.py, v5_nested.py, vnotes.py
