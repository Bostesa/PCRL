# Closing statement of the statistical reviewer (campaign `pcrl_finite_solver_adequacy_v1`)

Author: the statistical and challenger reviewer who wrote the frozen FSA1 rules (`RULES.md` / `RULES.json`, commit 71f8de33b;
`fsa/rules.py` bd8a4c5b..., `fsa/test_rules.py` a0233274..., committed unchanged by the coordinator in 6e8594bf0 and unchanged
since). Review started 2026-10-21T03:42:05Z (`date -u`).

**Inputs.** Committed files and saved JSON records only:
- the draft `RESEARCH_DECISION.md` (5ad143f54);
- `decisions/ADVANCEMENT.json`, `ROUTE_S_INPUT.json` and `EQUAL_EFFORT_ATTRIBUTION.json`;
- `source/`, `nested/` (through the route S input and the tables), `replay/` and `audit/` (through the tables);
- `roles/verifier/P1_VERIFIER_V<v>.json`;
- `roles/exact_reference/{RESULTS_V1, NARROW_V1, NARROW_V2}.json`, and the route E input as recorded in ADVANCEMENT.json;
- `tables/*.md`;
- `roles/coordinator/COORDINATOR_LOG.md`;
- FRC1 `fa/SELECTION_V<v>.json`, as the recorded side of the replay rule.

I read no label, opened nothing under `.private/` and ran no real-data computation. I re-derived the labels by calling the frozen
functions (`fsa.rules.advancement`, `route_d`, `route_e`, `route_s`, `p1_replay`) on the saved inputs.

## Required headline

The coordinator must use this wording (numbers may only be re-rounded, not changed):

> Development evidence on adaptively used MODEL_FIT records only; MODEL_SELECT was not opened. Under the frozen FSA1 rules the
> advancement decision is `OPTIMIZER_REPAIR_CLOSED`, and the conditional development-data pilot did not run, so no pilot outcome
> class (method candidate, component, engineering result or bounded negative result) was computed. No implementation defect was
> found: the four documented approximations are not defects under the rules (route D). The revised solver SV1 lowers PRC's stated
> objective at every registered strength in both versions, by 3e-6 to 0.0105 nats. At its S2 point, the MODEL_FIT nested held-out
> pair LOOKUP AUC against the spend-matched original point changed by -0.0018 (se 0.0020) and -0.0038 (se 0.0017), pooled -0.0028
> (mean se 0.0019; 2-SE interval -0.0065 to +0.0009). That is not resolved, and the task alternative also failed (spend -0.0006 /
> -0.0007 against the required -0.001). Route S therefore ended `S_PROXY_ONLY` and supersedes route E's advance, which rests on a
> single registered invented witness law built to contain a feasible gain. The previous campaign's P1 evidence was reproduced
> exactly and independently verified (`P1_REPLAY_VERIFIED`). Better optimization is shown for the surrogate objective only. Whether
> it improves usefulness or recovery for the MODEL_SELECT readers, and whether propensity adds anything beyond the equally optimized
> HQ control, remain untested.

## Label check (re-derived with the frozen code from the saved inputs)

| Item | Record | Draft | Check |
|---|---|---|---|
| Advancement | `OPTIMIZER_REPAIR_CLOSED`, flag `E_SUPERSEDED_BY_S_SV1` (ADVANCEMENT.json, 03:31:39Z) | line 9 | OK, re-derived |
| D | `D_NO_MATERIAL_DEFECT`; all four claims `NOT_A_DEFECT_DOCUMENTED_APPROXIMATION` | line 14 | OK |
| E | own label `E_ADVANCES`; useful cells: the 12 law-D cells only; 32 of 60 cells improve; never worse | line 15 | OK, but see C4 |
| S | `S_PROXY_ONLY`. S-a holds at all six strengths. Recovery point conditions hold in both versions (-0.0018 / -0.0038 ≤ -0.001; spend -0.00061 / +0.00034 < 0.0005; single views within 0.010). The pooled resolution fails: -0.002813 + 2 x 0.001854 = +0.00090 > 0. Task: -0.00061 / -0.00072 > -0.001 | lines 16, 62-66, 88 | OK |
| P1 replay | `P1_REPLAY_VERIFIED` in both versions under the frozen `p1_replay`: reproduction max difference 0.0, verifier max 2.2e-19, no boundary ties | line 111 | label missing from the draft; see C14 |
| SV1 fitted gains (PRC) | -1.2e-4 … -5.58e-3 (version 0) and -2.7e-6 … -1.049e-2 (version 1) | line 39 "3e-6 to 0.0105" | OK as rounding; say PRC (C15) |
| Spends at S2 | SV1 0.00153 / 0.00121; original 0.00214 / 0.00194 | lines 62-63 | OK |
| Version-1 tilt line | 2.573211 at tilt 0.1875 against 2.590162; version 0 line best 2.622865 > 2.607569 | lines 35-38 | OK |
| Law windows | D (0.0027169, 0.0028661); E (0.28668, 0.46622) | lines 32-33 | OK |
| Law C / E | C, strength ≥ 2: occupation accuracy loss +0.033, infeasible. E, strengths 0.5 / 1: pair AUC 0.6182 to 0.5979, occupation accuracy loss +0.0060, feasible | lines 70-73 | OK; wording, see C10 |
| HQ | SV1 worse at version 0, strength 8 (+1.37e-4); nested values as quoted | lines 78-81 | OK; framing, see C11 |
| Expressivity | 17,744 states; 1.03% / 1.34% of people share scores | lines 83, 112 | OK; wording, see C11 |
| Rule consistency | 72 of 72 saved states rule-consistent; 0 people moved at 1e-9 relative tilt | lines 55-56 | OK |
| B1 vs prototypes | 0.0095 / 0.0066 (PRC_S2), 0.0316 / 0.0544 (HQ_S2) | line 50 | OK; wording, see C7 |

## Required corrections

C1. **Outcome wording** (line 18, question 6 lines 93-95). "A bounded negative result for the optimizer-repair hypothesis, with
engineering results" uses the names of two pilot outcome classes that were not computed. `ENGINEERING_RESULT` (RULES 7.6) needs C1,
C2 and a measured gain over the original solver on MODEL_SELECT, and none was assessed. Required in both places: "Rule outcome
`OPTIMIZER_REPAIR_CLOSED`; no pilot outcome class was computed. Descriptively, the optimizer-repair hypothesis closes on MODEL_FIT,
with the bound stated (pooled pair LOOKUP change -0.0028, 2-SE interval -0.0065 to +0.0009, the objective's own LOOKUP reader
only). The campaign also produced verified engineering findings, which are not the rules' `ENGINEERING_RESULT` class." Rename the
bullet "Engineering results" to "Verified engineering findings (descriptive)".

C2. **"SV1 is a better optimizer of the stated objective on the source"** (line 21). It holds for PRC in 12 of 12 strength-version
cells. For HQ it holds in 11 of 12 (worse at version 0, strength 8, by 1.4e-4). Required: "SV1 lowers PRC's fitted objective at
every registered strength in both versions; it is not uniformly better (HQ, version 0, strength 8) and not optimal (C3)".

C3. **Exact references** (line 22 and the "What was done" row, "SV1 finds them and is never worse"). SV1 stays above the best
same-class reference in 20 of the 60 cells (10 distinct):
- law C at 0.25 and 1;
- law D at all six strengths, by 0.00006-0.00015;
- law E at 0.25 and 8.

The largest gaps are 0.0101 (C, strength 1) and 0.0102 (E, strength 8). Add: "SV1 finds the missed windows but does not reach the
best same-class reference in 10 of 30 distinct cells (largest 0.0102)".

C4. **Duplicate seeds** (line 15; also log 03:07Z). Seed 1 equals seed 0 in every cell (EXACT_REFERENCE_TABLES header), so the
registered set has 30 distinct cells and law D has 6. Duplicates count once. Required: "On 60 registered cells (30 distinct: seed 1
equals seed 0) SV1 was never worse, and it gave an exact recovery gain with no task loss up by 0.0005 at all six strengths of law D
(12 cells). Law D is the witness law built to contain a feasible gain, and no other law gave a useful cell."

C5. **"None changed a decision"** (question 2, line 42). This is not supported. Route D classifies the documented approximations as
non-defects by definition and does not measure their effect on decisions. The seven-tilt search demonstrably changes the solutions
(questions 1 and 3). Required: "Under the frozen rule they are documented approximations, not defects (`D_NO_MATERIAL_DEFECT`).
Route D does not measure whether they change decisions. The seven-tilt search changes the solutions (questions 1 and 3); the other
three are quantified below."

C6. **"Away from small tilts the proposal direction is a poor guide to the objective"** (line 46). The Spearman is over all
82,052 / 91,490 intervals of the line, and no record breaks it down by tilt. Required: "Along the final refreshed book's tilt line at
strength 8, across its 82,052 / 91,490 distinct partitions, the routing's shrunk dependence score and the exact dependence term are
anti-ranked (Spearman -0.995 / -0.996, PRC)." Keep the acceptance sentence. Delete the "small tilts" clause unless a record supports
it.

C7. **B1 against prototypes** (lines 51-52).
- Write "lower or equal": income cross-entropy is equal to six decimals for PRC_S2 version 0.
- Replace "but it is not what selects or emits it", which is wrong as written: the objective accepts every partition. Use: "The
  objective's task term uses the prototypes, so it slightly overstates the emitted release's training task loss. Partitions are
  accepted by the objective; strengths are selected on the emitted release's nested held-out spend."

C8. **In-sample against out-of-fold routing** (line 57). "A known train / deploy shift of the teachers, not a reassignment"
asserts more than the records show. Required: "Routed with the deployed full-pool teachers, 48-53% of the training rows land in the
same joint cell as their out-of-fold routing. About half move, because full-pool predictions on training rows differ from
out-of-fold ones. The B1 tables use the out-of-fold cells, which is what new people resemble. This was not examined further and is
not a defect under the rules."

C9. **The spend-matched point** (question 3, line 64).
- In version 1 the bracket is PRC@2.0 (spend 0.00088). It spends 0.00034 less than SV1's point, inside the 0.0005 no-worse band.
- In version 0 the bracket PRC@2.0 behaves as the same partition as PRC@8.0: identical nested spend and identical paired
  differences.
- Add the bound: pooled -0.0028, 2-SE interval -0.0065 to +0.0009.
- Add, descriptively: in version 1 the occupation-view LOOKUP AUC is 0.0023 (se 0.0022) higher than at the original S2 point,
  within the 0.010 allowance.
- Add "the point conditions held in both versions; only the pooled 2-SE condition failed", so that "not resolved" is not read as
  "no change".

C10. **"Below the original limits at high strength"** (line 74). This is reversed. Required: "beyond the original limits (infeasible)
at strength 2 and above in laws C and E".

C11. **The descriptive nested PRC-versus-HQ sentence** (lines 79-85). Under the original solver, the strength-8 nested gap between
PRC and HQ was larger: 0.6227 vs 0.6262 and 0.6206 vs 0.6276, that is 0.0035 / 0.0070. In FRC1 that larger nested LOOKUP gap did not
carry to the MODEL_SELECT reader grid (PRC_S2 against HQ_S2, benefit +0.0003, se 0.0051). SV1 narrows the gap to 0.0016 / 0.0048,
because it improves HQ more. Required replacement:

> "Descriptively (no SEs, no registered comparison, MODEL_FIT nested LOOKUP only), at strength 8 revised PRC's pair LOOKUP AUC is
> 0.0016 / 0.0048 below revised HQ's, at lower spend (0.0015 / 0.0012 against 0.0029 / 0.0063). Under the original solver the same
> gap was 0.0035 / 0.0070, and in the previous campaign that larger nested gap did not carry to the MODEL_SELECT reader grid (benefit
> +0.0003, se 0.0051). LOOKUP is the reader type the objective targets, so by card FA_PRC_v1 an advantage seen only by it is
> reader-targeted. Question C, whether propensity adds anything beyond HQ, is unanswered."

Also change "an unrestricted task-score mapping could reproduce any PRC partition" to "could reproduce any PRC partition except on
the 1.0-1.3% of people whose task scores are shared with another support state (MODEL_FIT support)".

C12. **The prediction** (line 90). It was committed at 03:24:24Z, before any nested result. That was after the source-stage
results, which already showed S-a and the fitted pair-information change of at most 0.0002 nats the prediction cites, and after
the nested units had started (03:23:37Z / 03:23:49Z). Its predicted range of ±0.003 held in version 0 and missed in version 1
(-0.0038). State all three facts.

C13. **Equal-effort attribution** (the "What was done" row, line 114).
- SV1_first stopped on the 30-step path cap at every strength for PRC in both versions, and at 11 of 12 for HQ. SV1 stopped by "no
  improving proposal" everywhere.
- SV1_first used 437-511 CPU-s per chain against 25-33 for SV1.
- So it is equal candidate set per step, not equal effort per chain, and both "SV1_first ends worse than the original" and
  "best-improvement selection produced every SV1 gain" are confounded by the cap. Say so.
- Disclose that ADVANCEMENT.json carries `equal_effort_attribution: NOT_RECORDED` and no controls. The attribution was computed
  separately (EQUAL_EFFORT_ATTRIBUTION.json), and SV1_first's route E input exists but was not passed as a control. No label
  changes, because controls never advance.

C14. **P1 replay and missing files** (lines 111, 118, 132).
- State the label `P1_REPLAY_VERIFIED`, re-derived with the frozen `p1_replay` from FRC1 SELECTION_V<v>.json,
  `replay/P1_REPLAY_V<v>.json` and `roles/verifier/P1_VERIFIER_V<v>.json`.
- `VERIFICATION.md`, `ADVISOR_BRIEF.md`, `START_HERE.md`, `OBJECTIVE_AUDIT.md`, `EXACT_REFERENCE_CASES.md` and
  `SOLVER_COMPARISONS.md` do not exist in the campaign folder at 5ad143f54. Write them or remove the references.
- The verifier recomputed spends, SEs, brackets, LOOKUP AUCs, F4 and the selection. It did not recompute all 413 nested means:
  `nested_means` is not among its fields.

C15. **Question 1, line 39.** Write "SV1 lowers PRC's fitted objective …"; the HQ exception is in C2.

C16. **Rules identity.** ADVANCEMENT.json records no rules hash. State in the decision: RULES.json ec4af5e6..., `fsa/rules.py`
bd8a4c5b....

## Ruling on the coordinator's reading of RULES 3.1

**Adopted: the reading is the intended meaning, and it is not a rule change.**
- "Any law already run by a solver is excluded" refers to laws that a solver ran on before the law's registration commit. That is
  where the risk lies: a law chosen after seeing solver behaviour on it. Any solver's runs after registration are the route-E
  evaluation itself, and excluding them would make route E impossible. This covers the original, SV1, SV1_first and the references.
- Checked from the records: the laws were committed at 02:42:16Z (c27a91477), the SV1 card at 02:46:39Z (91ba9a27e), and the first
  registered-law run started at 02:49:12Z (NARROW_V1 `utc_start`).
- `RESULTS_V1`, `ENUM_V1` and `STEP5_V1` carry no run times, so their order rests on the coordinator log and the queue ledger,
  which I did not open.
- The 02:44-02:47Z timing trace was never registered and is correctly outside route E.

**Two disclosures are required with the ruling:**
1. The laws were committed before my rules were frozen (02:59:51Z). The rules did not shape the laws, and I did not see them.
2. Law D was constructed as a witness containing a feasible gain the seven-tilt grid misses. Route E's advance therefore shows that
   SV1 can find such a gain when one exists. It does not show that such gains are common. Route S, the source evidence, governs
   anyway (RULES 5.2).

## Additions required

- In the limits section:
  - E's dependence on the witness law (C4);
  - the SV1_first cap (C13);
  - HQ not uniformly improved (C2);
  - SV1 short of the same-class reference (C3);
  - the decision resting on the pooled 2-SE condition, with the point conditions met (C9);
  - the nested evidence is the objective's own LOOKUP reader on label-derived MODEL_FIT statistics.
- In "What remains unknown":
  - the reader-validity and pilot rules (sections 6-7) were never exercised; no grid was built and MODEL_SELECT was not read, so
    nothing is known about the MODEL_SELECT readers;
  - question C stays unanswered.
- A sentence that FRC1's labels are unchanged: the replay rule never relabels them, and route D found no defect affecting them.

## Completion record

- Started 2026-10-21T03:42:05Z (`date -u`). The completion time is the `date -u` reading in the hand-back, taken after this file was
  committed and the CPU charge recorded.
- Computation: `fsa.rules` functions on saved records, JSON reads and `git log`; no label, no real data, no `.private/` file.
- Nothing else was edited.

## Addendum 1 (2026-10-21T03:51:44Z `date -u`): estimator identity of the P1 replay

Appended on the coordinator's 03:50Z request; nothing above is changed. The verifier flags `V2.ESTIMATOR_TEXT` as ATTENTION, with
no numerical failure. Two facts are at issue. I checked the second myself from git.

1. **The selection loop.** The replay (`fsa/fa/solve.py` `stage_replay`) calls `frc.fa.route.nested_c`, `select_path`,
   `brackets_for` and `nested_lookup` unchanged. It re-implements the selection loop of `stage_paths` with renamed local variables
   (`sel2` for `sel`, and `k2` / `v2` in comprehensions). The verifier reports that its AST equals `stage_paths`' loop after the
   renaming.
2. **The pinned file.** My RULES 8 pin, `frc/fa/route.py` 933d0d6f (commit 01a86ce20, revision R1), is not the file that ran P1.
   The P1 records were written under the b538fa7a4 version (the coordinator's 7ed1003f). `git diff b538fa7a4 01a86ce20 --
   frc/fa/route.py` shows exactly one change on the P1 path: `emit` calls `route_any(E, book)` instead of
   `P.route(entry_embedding(E, book), book)`. `route_any` returns that same call for every book without `frc_route == 'unt'`, and
   only `unt_book` sets that key. No P1 book is a UNT book, so the computation is identical for P1. Every other P1 function is
   untouched by the diff. Pinning the then-current file without checking which version produced the record was **my error in
   writing RULES 8**. It changes no number.

**Ruling.** `estimator_unchanged = True` stands, and the replay label is **`P1_REPLAY_VERIFIED`**. This is an interpretation of the
identity clause made after the result, so it is disclosed. The strict-text reading is reported beside it: the re-implemented loop
is not "textually identical", which literally gives `P1_ESTIMATOR_CHANGED`.

The purposive reading governs, for these reasons:
- The clause exists to guarantee that the replay computes the same function. A consistent, collision-free renaming of local
  variables cannot change a Python 3 function, and comprehension variables are local in Python 3. The verifier established AST
  identity under that renaming mechanically, not by inspection.
- The loop only applies the unchanged `select_path` / `brackets_for` / `nested_lookup` to the unchanged `nested_c` outputs. The
  per-person outputs do not pass through it.
- All 809 recorded leaves reproduce bitwise (largest difference 0.0). The verifier's own recomputation from the per-person
  outputs agrees within 2.2e-19, with no boundary ties.
- No decision depends on the P1 label. `advancement` does not read it, and route S's condition (nested original entries equal the
  replay, 117 arrays, 0.0) holds on its own.
- The strict label would describe "an estimator or saved outputs that differ from the registered P1 estimator", which is false
  here.

**Consequences.**
- *Headline:* replace the sentence "The previous campaign's P1 evidence was reproduced exactly and independently verified
  (`P1_REPLAY_VERIFIED`)." with:

  > The previous campaign's P1 evidence was reproduced bitwise and independently recomputed (`P1_REPLAY_VERIFIED` under the
  > reviewer's estimator-identity ruling; the strict text of RULES 8 gives `P1_ESTIMATOR_CHANGED`, because the replay
  > re-implements the selection loop with renamed local variables).

- *Correction C17:* in RESEARCH_DECISION.md's "What was done" P1 row, "estimator files byte-identical" is inaccurate. Replace it
  with:

  > `nrm/fa/prc.py` byte-identical; the `frc/fa/route.py` functions imported unchanged from the pinned 933d0d6f, which differs from
  > the file that ran P1 (b538fa7a4) only in `emit`'s dispatch through `route_any`, a change that is behaviour-preserving for P1
  > books; the selection loop re-implemented with three renamed local variables (AST-identical after renaming, verifier); all 809
  > recorded values equal; `P1_REPLAY_VERIFIED` under the reviewer's ruling (strict text: `P1_ESTIMATOR_CHANGED`).

- *Erratum to RULES 8 (frozen, not edited):* the pin should have named the file version that produced the record, or required
  behavioural identity on the P1 books. Both hold here.
- The FRC1 labels are unchanged under either reading.
