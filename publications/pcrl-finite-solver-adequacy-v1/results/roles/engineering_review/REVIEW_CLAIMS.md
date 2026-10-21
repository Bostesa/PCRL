# Claims review of the closing documents (engineering and claims reviewer)

Campaign `pcrl_finite_solver_adequacy_v1`. Request received 03:46Z; review started 03:45:40Z (`date -u`); numbers checked until
03:55Z; written after that. Documents at commit `54b97d623`.

**Method.**
- I checked every number and factual claim in the seven documents (RESEARCH_DECISION.md only where a phrase repeats a flagged item)
  against:
  - the generated tables (`tables/*.md`), which I regenerated myself (see below);
  - the records: `replay/`, `audit/`, `source/`, `nested/`, `decisions/ROUTE_S_INPUT.json` and `ADVANCEMENT.json`,
    `roles/exact_reference/{RESULTS_V1,ENUM_V2}.json`, `custody/BACKUP_VERIFICATION.json`, the coordinator log and the queue ledger.
- No label and no real data were read. The records are aggregate or label-derived statistics.

**Commands run, all label-free.** The scratch copy came from `git archive HEAD` in my scratchpad; I imported it with `PYTHONPATH`
and `python3 -P`, so nothing in the worktree was touched.

| Command | Result |
|---|---|
| `python3 -m fsa.report` | regenerated all three tables. `diff -r` against the committed `tables/` is empty: the tables reproduce exactly from the records. |
| `python3 -m fsa.fa.solver` | 9/9 |
| `python3 -m fsa.test_rules` | 15/15. With only this campaign's results in the copy it gave 11/15 (FileNotFoundError): it needs `results/pcrl_new_release_mechanisms_v1/` and `results/pcrl_occupation_headroom_overnight_v1/` `roles/statistical_reviewer/RULES.json` (see Q3). |

CPU charged: 30 CPU-s (upper estimate; about 6 s measured).

## 1. Required corrections

Each entry gives the file, the current text, and the correct text or value with its record.

| # | File | Current text | Correct value (record) |
|---|---|---|---|
| R1 | `EXACT_REFERENCE_CASES.md` §Findings 1 | "Objective within 2.4e-14" | "within 1.2e-13". The record is `RESULTS_V1.json` checks.max_abs_objective_diff_code_vs_mine = 1.208e-13, as OBJECTIVE_AUDIT.md states correctly. |
| R2 | `EXACT_REFERENCE_CASES.md` §Findings 3 | "(46,656-1,679,616 partitions)" | "(1,296-1,679,616 partitions)". D_witness has 1,296 (`tables/EXACT_REFERENCE_TABLES.md` A). |
| R3 | `EXACT_REFERENCE_CASES.md` §Findings 3 | "(pair AUC 0.5214 / 0.5338, occupation accuracy loss up by 0.033)" | "occupation accuracy loss up by 0.033 / 0.036". The 0.5338 optimum (C at 0.5) has occ_acc_loss 0.036 (`ENUM_V2.json` C_distinct obj_0.5). The 0.5214 partition has 0.033 (ADVANCEMENT E cells). E's infeasible points at 2-8 are +0.036 to +0.039. |
| R4 | `ADVISOR_BRIEF.md` Q3 | "lower than the spend-matched original point, at slightly lower task spend" | Lower spend only in version 0. Against the bracket, SV1's spend difference is -0.00061 (se 0.00060) in version 0 and **+0.00034** (se 0.00052) in version 1, i.e. higher (`tables/SOLVER_COMPARISON_TABLES.md` D; ADVANCEMENT routes.S per_version d_spend_vs_bracket). Suggested text: "at lower spend in version 0 and slightly higher spend in version 1 (-0.0006 / +0.0003)". |
| R5 | `ADVISOR_BRIEF.md` Strongest evidence, favourable 2 | "On the source SV1's selected point is at lower held-out pair recovery, in point estimate, at lower task spend in both versions." | The sentence holds against neither reference in both versions. Against the bracket: AUC -0.0018 / -0.0038, spend -0.0006 / **+0.0003**. Against the original S2 point: spend -0.0006 / -0.0007, AUC only -0.0018 / **-0.0003**. State one comparison with its own numbers. |
| R6 | `ADVISOR_BRIEF.md` Q1; `START_HERE.md` l.16; `HANDOFF.md` l.10 | "It lowers the stated objective at every registered strength in both versions" / "lowers the objective at every registered strength" / "lowers that objective at every registered strength" | True for PRC only (12/12). For HQ, SV1 is lower in 11 of 12 cells and **higher** in version 0 at strength 8 (+0.000137, tables A). Add "for PRC (HQ: 11 of 12)". SOLVER_COMPARISONS.md and the brief's Q4 already say this. |
| R7 | `SOLVER_COMPARISONS.md` Results 2 | "SV1_first creeps along the current book's line, 30 accepted steps per strength" | 30 steps (the path_steps cap) at 23 of 24 family-version-strength chains. HQ version 0 at strength 8 stopped after 20 steps, with no improving proposal (`source/SOURCE_CHECK_V0.json` runs HQ\|SV1_first, lloyd_steps [30,30,30,30,30,20], stopping_reason). |
| R8 | `SOLVER_COMPARISONS.md` Results 2 ("Selection, not resolution, does it on the source … so the finer candidate set alone never helped. Best improvement produced every SV1 gain."); `ADVISOR_BRIEF.md` adverse 4 ("The finer candidate set with the original first-improvement rule made things worse on the source.") | causal attribution | The attribution is confounded by the step cap. SV1_first ended at "path_steps reached", still accepting improving steps, in 23 of 24 chains, and used 436-511 CPU-s against 25-33 for SV1 and 4-5 for the original (tables B). State: "under the registered 30-step cap, which first improvement on the finer set reached at 23 of 24 strengths while still improving, it ended above the original at every strength". Do not claim that resolution cannot help. EQUAL_EFFORT_ATTRIBUTION is already marked descriptive. |
| R9 | `SOLVER_COMPARISONS.md` solver table; `START_HERE.md` l.29; `EXACT_REFERENCE_CASES.md` Findings 4 | "equal-effort control" | Equal effort per step only (same candidate set, same incremental evaluation). On the source it took 13-18 times SV1's Lloyd steps (170-180 against 10-13) and 13-20 times its CPU (436-511 against 25-33 CPU-s; tables B). Write "equal-effort per step". |
| R10 | `SOLVER_COMPARISONS.md` Results 4; `ADVISOR_BRIEF.md` Q3 | "the spend-matched original point (the bracket; PRC@2.0 in both versions)" | Give the rule: the cheapest original point whose spend is >= SV1's spend - 0.0005 (RULES.md l.144). In version 1 the bracket spends less than SV1's point (0.0009 against 0.0012), so it is matched only within that 0.0005 tolerance. |
| R11 | `SOLVER_COMPARISONS.md` Results 5 | "Task scores identify 98.7-99.0% of the support states" | The record measures **people**: 1.03% / 1.34% of people share their task scores with another state (`tables/OBJECTIVE_AUDIT_TABLES.md` D). Write "for 98.7-99.0% of people", as RESEARCH_DECISION.md l.83 does. As a share of states: 17,670 / 17,652 distinct vectors for 17,744 states (99.6% / 99.5%). The "about 99% of the support states" in `ADVISOR_BRIEF.md` Q4 and `HANDOFF.md` l.26 is approximately right in either unit. Prefer "of people". |
| R12 | `START_HERE.md` l.31; `HANDOFF.md` l.37; `SOLVER_COMPARISONS.md` l.30; `RESEARCH_DECISION.md` l.111, l.132 | "`VERIFICATION.md`: the independent verifier's replay" / "the verifier's recomputation in `VERIFICATION.md`" | `VERIFICATION.md` does not exist at `54b97d623`; only `roles/verifier/` (CHECKS.json and others) exists. Write it before publication, or point to `roles/verifier/records/CHECKS.json`. |
| R13 | `HANDOFF.md` l.35; `START_HERE.md` l.43 | "the reviewer's `CLOSING_STATEMENT.md`" / "(rules, closing statement)" | `roles/statistical_reviewer/CLOSING_STATEMENT.md` does not exist at `54b97d623`; the commit says the documents are drafts made before it. Mark it pending or add it. |
| R14 | `START_HERE.md` l.45 | "Ledgers, access, resources, deviations: `EXPERIMENT_LEDGER.csv`, `CONFIG_LEDGER.csv`, `ACCESS_RECEIPT.json`, `RESOURCE_ACCOUNTING.json` …" | These four files exist but are **untracked** (`git status`, 03:45Z). Commit them before the documents cite them as evidence. |
| R15 | `QUICKSTART.md` §2 | "`python3 -m fsa.lane …/lanes/SOLVER_V0.json   # replay, audit, source, nested (version 0)`" and "a rerun into an existing output path is refused, so use a new output name" | As written, the lanes rerun nothing. Their unit ids (FSA_REPLAY_V0_V1, …) are `done` in the ledger, so `fsa.lane` skips them as `already_done` (`fsa/lane.py:24-25`). The four stages have fixed output paths and no output-name option (`fsa/fa/solve.py`): only `fsa.fa.advance` takes `--out`. Say the lane files record what ran, and that a rerun needs a new lane file with new unit ids plus new output paths (a code change). |
| R16 | `OBJECTIVE_AUDIT.md` §Findings 2; `ADVISOR_BRIEF.md` Q2 | "This is why large tilts are always rejected" / "This is why large moves are always rejected" | Overclaim of mechanism. The anti-ranking (Spearman -0.995 to -1.000) was measured on one line per family-version, at strength 8, and no causal test was run. Write "consistent with the rejection of large tilts". "Always" is not established for the other strengths. |
| R17 | `OBJECTIVE_AUDIT.md` §Findings 3 | "Where nothing is accepted, every partition on the refreshed book's line is worse than the current state (PRC, version 0) …" | The refreshed line was enumerated only at strength 8 (audit B). Write "at strength 8, every partition on the refreshed book's line is worse than the current state (PRC, version 0: best 2.622865 against 2.607569)". |

## 2. Overclaims about usefulness or protection

- The documents avoid claiming usefulness or protection from the better objective. START_HERE "What this campaign does not claim",
  the HANDOFF "What must not be said" list, QUICKSTART §1 ("not a usefulness or protection guarantee") and the brief's Q3 are all
  consistent with `S_PROXY_ONLY` and `OPTIMIZER_REPAIR_CLOSED`.
- The remaining leaks toward a benefit reading are R4, R5, R8 and R10.
  - R4 and R5 present SV1's nested point as cheaper and less recoverable in both versions. In version 1 it is more expensive than
    its bracket, and against the original S2 point its recovery difference is -0.0003.
  - R8 turns a cap-bound control into a causal statement.
- `ADVISOR_BRIEF.md` favourable 1 says "SV1 finds them" (the deployable improvements). On the source, SV1 does not reach the best
  partition the audit exhibited. In version 1 at strength 8, one refreshed-line partition of the original's final book has J = 2.5732
  (`audit/OBJECTIVE_AUDIT_V1.json` PRC_tilt_line_at_8 best_interval, t in [0.187547, 0.187552]). SV1's chain ends at 2.5797
  (tables A). SV1 finds improvements, not the ones the audit showed.
  - Required: "SV1 finds deployable improvements (on the source smaller than the best partition the audit exhibited at version 1,
    strength 8: 2.5797 against 2.5732)".
  - The improving window near t = 0.19 is 4.8e-6 wide. Say so where "near tilt 0.19" appears.
- `ADVISOR_BRIEF.md` Q5: "The coordinator's prediction, registered before the nested results". It is true:
  - the prediction was committed at 03:24:24Z (`7600d6ae2`), and the nested records were written at 03:29:34Z and 03:29:49Z;
  - but the nested units had already started at 03:23:37Z and 03:23:49Z (ledger). Disclose that in one clause.
- `ADVISOR_BRIEF.md` adverse 1: "21-24% of people are re-routed". Add "relative to the twin (the original re-routes 14-16%)", so the
  re-routing is not read as caused by SV1 alone (tables A: 0.236 / 0.212 against 0.156 / 0.138).

## 3. Verified as stated

**ADVISOR_BRIEF.md**
- Q1: law D 0.5803 against 0.6882; version 1 at strength 8 near tilt 0.19, between 0.125 and 0.25 (audit 0.18755).
- Q2: Spearman -0.995 / -0.996 for PRC; refresh changes -9.1e-7 / -7.1e-7; B1 against prototypes 0.0095 for PRC, with emitted
  cross-entropy not above the prototypes' (equal for income in version 0: "lower" should read "not higher").
- Q3: -0.0018 / -0.0038; pooled -0.0028, se 0.00185 (rounds to 0.0019); C and E traded utility and were infeasible at 2-8.
- Q4: HQ 11 of 12; HQ 0.6225 / 0.6251 against 0.6262 / 0.6276 at higher spend; PRC 0.6209 / 0.6203 at 0.0015 / 0.0012.
- Q5: S_PROXY_ONLY, E_SUPERSEDED_BY_S, D_NO_MATERIAL_DEFECT.
- Q6: P1 replay difference 0.0 in both versions, 413 means; F4 equal to the record (0.0071527 / 0.0111347).
- Q7: 0.018-0.023 (frc RESEARCH_DECISION: 0.0177 / 0.0233).
- Adverse 1: pair information -0.0002 (0.03322 to 0.03300; 0.03273 to 0.03252).

**OBJECTIVE_AUDIT.md**
- 1.2e-13, 0 mismatches and 1e-14 (record 5.1e-15 / 8.5e-15).
- Traced chains reproduce the saved partitions.
- 68,921-91,490 intervals; HQ Spearman -1.000 / -0.999.
- Guard within 3.2e-4.
- 7/1, 9/3 and 8/2 steps; acceptance at 0.25 for PRC and one later strength for HQ (re-routing shares).
- 72 books re-route bitwise (source rule_consistent all True; custody); 0 knife-edge moves at 1e-9 for all 72 (ROUTE_S_INPUT).
- In-sample agreement 0.48-0.53; 17,744 states, 17,670 / 17,652 vectors, 1.0% / 1.3%.

**EXACT_REFERENCE_CASES.md**
- Commit and run times: 02:42:16Z, 02:46:39Z, 02:49:12Z (ledger FSA_EXACT_NARROW_V1).
- 60 path states not worse than the twin (60 true, 0 false).
- Law-D window and λ/64 >= 0.0039; law-E window at 0.5 and 1.
- STEP 5 table: D task losses <= +0.00038; E +0.006 occupation accuracy loss.
- 0 of 60 worse; SV1_first = SV1.
- Not traced to a record in the time available: the B task-score-only 0.5795; "lowest information = lowest AUC in every law"; "HQ
  ahead of PRC in E is a search artifact".

**SOLVER_COMPARISONS.md**
- 413 means, F4 0.00715 (0.00225) / 0.01113 (0.00231).
- 117 per-person arrays equal to the replay, difference 0.0 (ROUTE_S_INPUT original_entries_vs_replay_all_keys).
- 3e-5 to 0.0056 and 3e-6 to 0.0105.
- Distinct and rule-consistent PRC states; HQ +1.4e-4; effort numbers.
- 23.6% / 21.2%; +0.0008 emitted cross-entropy; -0.0018 pair-term share.
- Spends 0.00153 / 0.00121 against 0.00214 / 0.00194; -0.0018 / -0.0003 and -0.0006 / -0.0007 against the original S2 point.

**START_HERE.md and HANDOFF.md**
- Outcome labels, registration time 03:03:08Z, card supersession.
- Custody: T033518Z FAILED_RESTORE_CHECK, T033709Z VERIFIED_FROM_BACKUP_ALONE, off-device PENDING_DRIVE_NOT_ATTACHED, the backup
  path as recorded.

**QUICKSTART.md**
- §1 commands run as described, subject to the test_rules dependency below.
- §1 code snippet: the `S.chain` and `P.twin_book` signatures match; `route_params('PRC_S2')` and `('HQ_S2')` need no version. HQ
  needs qO0 / qO1 with 5 occupation classes.
- `fsa.fa.advance route_s --out` and `decide --e --s --out` exist.
- `fsa.custody backup` / `offdevice` exist.
- `fsa.report` regenerates the committed tables exactly. Note that it overwrites `tables/` in place.
- **Add:** `fsa.test_rules` needs the two earlier campaigns' RULES.json files. A publication copy holding only `code/fsa` fails 4 of
  15 tests.

## 4. Earlier code findings (REVIEW_PREREGISTRATION_CODE.md) now that no pilot runs

| Finding | Still matters? |
|---|---|
| S2 write-once checks after the computation | **No** for this campaign's evidence. Each stage ran once per version (ledger: one unit each), and no rerun is planned. It matters only if the stages are reused; see also R15. |
| S3 source private outputs not write-once, not hashed in the record | **Mitigated.** The SHA-256 of SOURCE_PARTITIONS_V0/V1.npz and SOURCE_BOOKS_V0/V1.pkl is pinned in `decisions/ROUTE_S_INPUT.json` (private_file_sha256), and the restore check re-routes every saved book, so an overwrite would be detected. A code fix is still advisable before reuse. |
| S4 nested-vs-replay equality optional and partial | **Resolved in practice.** ROUTE_S_INPUT compares all 117 per-person arrays per version (max difference 0.0, same units and labels). The replay files existed when the nested units ran. |
| S5 cached-statistics stages not tied to a queue unit | **No** for the evidence. The audit and source records were written by queue units (FSA_AUDIT_V*_V1 and FSA_SOURCE_V*_V1 in the ledger), and the documents disclose the statistics as label-derived. Code hygiene only. |
| S8 `frc:` prefix denotes this campaign in plan/claims | **No.** No plan or claims bank was built and MODEL_SELECT was not opened, so `fsa.plan` and `fsa.claims` were never used. It matters only on reuse. |

S1 (ulp-narrow intervals) is addressed for the real data by the knife-edge check. The audit and ROUTE_S_INPUT show that a 1e-9
relative tilt change moves no one in any of the 72 saved books, so no accepted state sits within rounding of a breakpoint.

## 5. Not checked

- RESEARCH_DECISION.md beyond the phrases listed in R6, R10, R11 and R12 (the statistical reviewer reviews its wording).
- The verifier's records (CHECKS.json), except as a substitute citation for R12.
- The three EXACT_REFERENCE_CASES.md items marked "not traced" in section 3.
- The publication tree, which does not exist yet.
