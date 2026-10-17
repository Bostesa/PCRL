# Claims review: SCM utility-matched follow-up (deployment, reporting and closeout)

**Reviewer.** Separate claims-reviewer agent, 2026-10-17. Worked only in this worktree. Ran no numerical job, read nothing under `.private/`, edited no other file and made no commit.

**What was reviewed.** Every sentence of `START_HERE.md`, `RESEARCH_DECISION.md`, `MANUSCRIPT_CLAIMS_HANDOFF.md`, `QUICKSTART.md` and `SELECTION_AND_MATCHING.md`, as committed at `b9299804f`. The branch tip `095e123c2` adds only `BACKUP_VERIFICATION.json`.

**Records checked against.** `report/DEV_REPORT_RESULTS.json` (claims and condition vectors, `research_decision_qualifiers`, `separate_status_fields`, all 58 + 24 endpoints, `comparison_differences`, `diagnostics`, `per_seed`, `registered_bets`), `PRIMARY_ENDPOINTS.csv` (identical to `report/PRIMARY_ENDPOINTS.csv`), `report/SECONDARY_ENDPOINTS.csv`, `SELECTION_LOCK.json`, `TRAJECTORY_SUMMARY.json`, `SOURCE_ADMISSION.json`, `READERS_SUMMARY.json`, `DEPLOYMENT_RESULTS.json`, `BACKUP_VERIFICATION.json`, `RECOVERY_TABLE.csv`, `UTILITY_TABLE.csv`, `PROTOCOL.md`, the `FROZEN DESIGN` block, `PREFIT_REVIEW_DISPOSITION.md`, the source `HANDOFF.json` and `scm_um/deploy.py` / `scm_um/common.py`. Numbers were read with small read-only Python snippets.

**Overall.** The decision label, the four qualifiers in `RESEARCH_DECISION.md`, every claim label and almost every number match the frozen records. The problems are in wording:
- "matched" is used for MODEL_SELECT matches that did not carry over;
- failure categories are merged;
- the failed match is partly repaired by extrapolation and by attribution to PUB_IBSI or to SEX guidance;
- the common-cell check is described as covering releases it did not cover;
- one sign is reversed and one statistic is misnamed;
- the mandatory qualifiers and scope caveats are missing from `START_HERE.md`;
- a verifier replay is described as done before any verifier record exists.

Severity rule used here:
- **MUST-FIX:** a sentence the frozen record contradicts, a registered wording rule that is broken, or a commissioning requirement that is not met.
- **SHOULD-FIX:** a missing qualifier or a precision problem that could mislead.
- **NOTES:** verified items, rounding, and things to check once the pending records exist.

---

## MUST-FIX

### MF-1. `START_HERE.md` prints the decision without its four mandatory qualifiers, without the scope caveats, and without the recovery qualifiers

**Evidence.**
- PROTOCOL §11: "Every decision is printed with four mandatory qualifiers". `research_decision_qualifiers` gives:
  - `DEV_utility_confidence` = `CONFIDENCE_UNRESOLVED`;
  - claim C `SUBGROUP_UNRESOLVED`;
  - ORD_NLLM DEV match: seed 0 `POINT_MISMATCH` / `MISMATCH_ESTABLISHED`, seed 1 `POINT_MISMATCH` / `EQUIVALENCE_UNRESOLVED`, both `SCM_SPENT_MORE`;
  - claim B `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.
- `START_HERE.md` never states `CONFIDENCE_UNRESOLVED`.
- It never says that:
  - pairs of numbers are seed 0 / seed 1;
  - the seeds share people;
  - intervals are conditional and exclude the development and selection history;
  - the service is a controlled benchmark with explicit assumptions.
- Its recovery numbers (items 2–5) carry no reader-budget, redundancy or lookup-reader qualifier. PROTOCOL §8 says the budget bias "is reported beside every recovery comparison".

**Current** (line 12):
> **The fixed comparison closed as inconclusive at the registered tolerance: `COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE`.**

**Replace with:**
> **The fixed comparison closed as inconclusive at the registered tolerance: `COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE`** (from claim M = `MATCH_DID_NOT_CARRY_OVER`). Its four mandatory qualifiers:
> - DEV utility confidence: `CONFIDENCE_UNRESOLVED` (3 PASS, 21 UNRESOLVED, 0 violations).
> - Claim C: `SUBGROUP_UNRESOLVED`.
> - ORD_NLLM match on DEV_REPORT:
>   - seed 0: `POINT_MISMATCH` and `MISMATCH_ESTABLISHED`;
>   - seed 1: `POINT_MISMATCH` and `EQUIVALENCE_UNRESOLVED`;
>   - direction `SCM_SPENT_MORE` on both seeds.
> - Claim B: `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.
>
> No release is frozen as an engineering reference.
>
> **Scope of every number below.**
> - Pairs are seed 0 / seed 1 on DEV_REPORT unless marked MODEL_SELECT.
> - The two seeds evaluate the same 5,943 people in 3,955 households. They are not separate populations.
> - Intervals are conditional on the fitted releases and readers. They exclude the development and selection history on these records, and they are not ACS survey-design variance.
> - The service is a controlled benchmark. Its readers rely on three explicit assumptions:
>   - SEX-labelled auxiliary records (ATTACK_FIT, 8,001 people);
>   - accurate linkage of the two outputs;
>   - distributional similarity between the auxiliary and the released people.

**Insert** a new item after item 6 (after line 25):
> 7. **Reading the recovery numbers.**
>    - Every AUC above comes from a fixed 8,001-person reader budget.
>    - SCM's finer partition can be under-read, which favours SCM. Its pair vectors hold a median of 16 / 17 ATTACK_FIT people, against 53 / 64 for the MODEL_SELECT-matched ordinary control.
>    - PUB_IBSI's partition is finer still (median 13).
>    - For every grouped release, the occupation output identifies the shared cell and income is a function of it. The linked pair therefore carries no information beyond the occupation output for these people, and differences between pair readers and occupation-only readers are differences between fitted readers.
>    - The lookup reader is not population-optimal, and no AUC here is a privacy guarantee.

### MF-2. "Matched" is used for MODEL_SELECT matches that did not carry over

**Evidence.**
- PROTOCOL §11 and FROZEN DESIGN MF-2: "'Matched' in any label means the DEV_REPORT point match only".
- The comparison-only status is "never called matched".
- No comparator point-matched on DEV_REPORT.

Replace each occurrence:

| File, line | Current | Replace with |
|---|---|---|
| START_HERE 15 | "2. **Matched ordinary control.** On MODEL_SELECT a strongest-proxy ordinary control was matched within 0.0002 nats on both seeds." | "2. **MODEL_SELECT-matched ordinary control (ORD_NLLM; the match did not carry over).** On MODEL_SELECT, a strongest-proxy ordinary control was within 0.0002 nats of SCM's held-out occupation NLL on both seeds (SCM − control: −0.00003 / +0.00015)." |
| START_HERE 16 | "On DEV_REPORT, SCM's linked-pair AUC read 0.034 / 0.025 below that control. The clause bounds are above 0, and neither recipient worsened." | "On DEV_REPORT, SCM's linked-pair AUC read 0.034 / 0.025 below that control (clause bounds 0.0225 / 0.0144 > 0), and all four recipient non-worsening clauses pass. This is the comparison-only status `BENEFIT_AND_NON_WORSENING_CLAUSES_PASS (not cost-matched on DEV_REPORT)`, not a matched result." |
| START_HERE 18 | "SCM also spent more income NLL, so the advantage is again not separated from extra loss." | "SCM also spent more income NLL: +0.0064 / +0.0035, ABOVE_ZERO on seed 0 and UNRESOLVED on seed 1. The lower pair AUC is therefore not separated from extra loss." |
| START_HERE 24 | "The matched ordinary control's held-out spend went from 0.0083 on MODEL_SELECT to 0.0043 / 0.0054 on DEV_REPORT." | "The MODEL_SELECT-matched ordinary control's held-out spend went from 0.0083 / 0.0083 on MODEL_SELECT to 0.0043 / 0.0054 on DEV_REPORT. Over the same step, SCM's went from 0.0082 / 0.0084 to 0.0113 / 0.0072." |
| START_HERE 25 | "No matched comparator (ordinary, PUB_IBSI or shuffled) carried over." | "No MODEL_SELECT-matched comparator (ordinary, PUB_IBSI or shuffled) carried over to DEV_REPORT." |
| RESEARCH_DECISION 49 | "…so the matched comparison is inconclusive at the registered tolerance." | "…so the comparison is inconclusive at the registered tolerance." |
| RESEARCH_DECISION 70 | "On DEV_REPORT, C0 minus the matched ordinary control is +0.0049 / +0.0047 (unresolved), and the same-count ordinary control minus the matched one is +0.0024 / +0.0023." | "On DEV_REPORT, C0 minus the MODEL_SELECT-matched ordinary control is +0.0049 / +0.0047 (UNRESOLVED), and the same-count ordinary control minus the MODEL_SELECT-matched one is +0.0024 / +0.0023 (UNRESOLVED)." |
| RESEARCH_DECISION 107 | "**No matched comparator carried over to DEV_REPORT.**" | "**No MODEL_SELECT-matched comparator carried over to DEV_REPORT.**" |
| RESEARCH_DECISION 119 (table row) | "Matched ordinary control" | "MODEL_SELECT-matched ordinary control (ORD_NLLM)" |
| RESEARCH_DECISION 173–175 (§9 rows) | "Ordinary matched", "PUB_IBSI matched", "Shuffled matched" | "Ordinary (MODEL_SELECT match)", "PUB_IBSI (MODEL_SELECT match)", "Shuffled (MODEL_SELECT match)" |
| MANUSCRIPT 18 | "C0 minus the matched ordinary control: +0.005 / +0.005, unresolved;" | "C0 minus the MODEL_SELECT-matched ordinary control on DEV_REPORT: +0.005 / +0.005, UNRESOLVED;" |
| MANUSCRIPT 23 | "The matched ordinary control's held-out expenditure fell from 0.0083 (MODEL_SELECT) to 0.0043 / 0.0054 (DEV_REPORT)." | "The MODEL_SELECT-matched ordinary control's held-out expenditure fell from 0.0083 (MODEL_SELECT) to 0.0043 / 0.0054 (DEV_REPORT), while SCM's moved from 0.0082 / 0.0084 to 0.0113 / 0.0072." |

RESEARCH_DECISION lines 129 and 150 and MANUSCRIPT line 46 are handled in MF-3 and MF-8. The claim-table row "M, matched ordinary" is the registered claim name; MF-3 adds "(MODEL_SELECT match)".

### MF-3. Failure categories are merged, and the label conjunctions are incomplete

**Evidence.** PROTOCOL §11 says: "Text never merges these categories." `START_HERE.md` line 27 promises that `RESEARCH_DECISION.md` gives "every label as an explicit conjunction and every failure category". The B condition vector contradicts the current text:

| B condition | Category in the record | Current text |
|---|---|---|
| income worsening vs NEURAL_RAW, seed 0 | POINT_EXCESS | "UNRESOLVED" |
| income worsening vs NEURAL_RAW, seed 1 | POINT_EXCESS | "UNRESOLVED" |
| income worsening vs H_LJ, seed 1 | POINT_EXCESS | "UNRESOLVED" |
| income worsening vs H_LJ, seed 0 | UNRESOLVED | UNRESOLVED |
| occupation worsening vs H_LJ, seed 1 | POINT_EXCESS | no category given |
| occupation worsening vs NEURAL_RAW, seed 1 | UNRESOLVED | not mentioned |
| nested 0.020 benefit, all four | POINT_SHORTFALL | not mentioned |

Three further gaps:
- M's vector also contains `DEV_SUBGROUP_POINT_GUARDS_4` = false (POINT_FAILURE). It is never mentioned for M.
- §7 lists SCM's 24 DEV **point** clauses under "Established (family intervals)", but only 3 of 24 are PASS on the family interval (`CONFIDENCE_UNRESOLVED`).
- §7 has no point-failure category. It also omits `NOT_TESTED_*`, which PROTOCOL §11 assigns to UNAVAILABLE. Its "Established" list includes favourable established results but leaves out unfavourable ones:
  - SCM spent more occupation NLL than ORD_CNT, ABOVE_ZERO on both seeds;
  - SCM spent more income NLL than ORD_NLLM, ABOVE_ZERO on seed 0.

**(a) Current** (RESEARCH_DECISION lines 17–25, the claim table): the table with columns "Claim | Label | First failing condition (category)".

**Replace with:**

> | Claim (α/4 each) | Label | First failing condition (category) | Other false conditions in the conjunction (category) | Comparison-only status |
> |---|---|---|---|---|
> | M, matched ordinary (primary; MODEL_SELECT match) | `MATCH_DID_NOT_CARRY_OVER` | DEV occupation-NLL point match vs ORD_NLLM, both seeds (POINT_FAILURE: seed 0 `MISMATCH_ESTABLISHED`, seed 1 `EQUIVALENCE_UNRESOLVED`; both `SCM_SPENT_MORE`) | four DEV subgroup point guards (POINT_FAILURE, men, both seeds); nested 0.020 `NOT_ESTABLISHED` (seed 1 clause bound 0.0144, UNRESOLVED) | `BENEFIT_AND_NON_WORSENING_CLAUSES_PASS (not cost-matched on DEV_REPORT)` |
> | M components (IBSI → SHUF) | `NOT_TESTED_M_NOT_ESTABLISHED` | M not established (UNAVAILABLE) | descriptive status `MATCH_DID_NOT_CARRY_OVER` for both | — |
> | K, same capacity | `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD` | men's occupation Brier point guard (POINT_FAILURE) | income worsening vs ORD_CNT, both seeds (UNRESOLVED; clause bounds 0.0136 / 0.0174); nested 0.020 `NOT_ESTABLISHED` (seed 1 clause bound 0.0169, UNRESOLVED); IBSI_CNT `NOT_TESTED_K_NOT_ESTABLISHED` (UNAVAILABLE) | `COMPARISON_CLAUSES_NOT_ALL_PASS` |
> | B, competitive | `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD` | men's occupation Brier point guard (POINT_FAILURE) | benefit vs NEURAL_RAW and vs H_LJ, seed 1 (UNRESOLVED); income worsening vs NEURAL_RAW both seeds and vs H_LJ seed 1 (POINT_EXCESS), vs H_LJ seed 0 (UNRESOLVED); occupation worsening vs H_LJ seed 0 (ESTABLISHED_VIOLATION), seed 1 (POINT_EXCESS), vs NEURAL_RAW seed 1 (UNRESOLVED); nested 0.020: all four POINT_SHORTFALL | `COMPARISON_CLAUSES_NOT_ALL_PASS` |
> | C, subgroup | `SUBGROUP_UNRESOLVED` (point status `SUBGROUP_POINT_GUARD_FAILED`) | men's guard clause bounds 0.0104 / 0.0113 > 0.005 (UNRESOLVED) | — (women's guards PASS) | — |
> | D, joint component | `NOT_CLAIMED` | | | |
> | E, novelty | `NOT_CLAIMED (existing-method engineering comparison on previously used development records)` | | | |

**(b) Current** (RESEARCH_DECISION lines 97–102):
> - Against NEURAL_RAW: benefit +0.017 / +0.005, unresolved on both seeds.
> - Against H_LJ: +0.018 (family-established, seed 0) / +0.010 (unresolved, seed 1).
> - **Per recipient: SCM reads higher.**
>   - Income recipient: +0.021 / +0.023 against NEURAL_RAW and +0.009 / +0.025 against H_LJ. The clause fails at the 0.010 margin; the statuses are UNRESOLVED.
>   - Occupation recipient against H_LJ: +0.048, an established violation on seed 0; +0.034 on seed 1.

**Replace with:**
> - Against NEURAL_RAW, benefit +0.017 / +0.005:
>   - seed 0: clause bound 0.0024 > 0, so the condition holds, but the family interval [−0.002, 0.036] is UNRESOLVED;
>   - seed 1: the clause fails (bound −0.0079; UNRESOLVED).
> - Against H_LJ:
>   - seed 0: +0.018, clause bound 0.0046 > 0, family-ESTABLISHED (lower bound 0.0001);
>   - seed 1: +0.010, clause bound −0.0030 (UNRESOLVED).
> - All four reference benefits fall below the original 0.020 target at the point (POINT_SHORTFALL; MATERIAL_SUPERIORITY not established).
> - **Per recipient: SCM reads higher.**
>   - Income recipient: +0.021 / +0.023 against NEURAL_RAW and +0.009 / +0.025 against H_LJ. All four non-worsening clauses fail at the 0.010 margin, in two categories:
>     - POINT_EXCESS against NEURAL_RAW on both seeds and against H_LJ on seed 1 (the point is above 0.010);
>     - UNRESOLVED against H_LJ on seed 0 (point 0.009, clause bound 0.022).
>
>     None is an established violation: every family interval contains 0.010.
>   - Occupation recipient:
>     - against H_LJ, +0.048 on seed 0 (ESTABLISHED_VIOLATION; family interval [0.021, 0.074]) and +0.034 on seed 1 (POINT_EXCESS; family interval [0.0085, 0.059] contains 0.010);
>     - against NEURAL_RAW, −0.014 on seed 0 (PASS) and −0.005 on seed 1 (UNRESOLVED; clause bound 0.0105 > 0.010).

**(c) Current** (RESEARCH_DECISION §7, lines 126–145): the "Established", "Unresolved" and "Unavailable" lists.

**Replace with:**
> ## 7. What is established, a point result, unresolved, unavailable or diagnostic
>
> No claim is established, so everything in this section is descriptive under the labels above.
>
> **Established on the family interval:**
> - Pair benefit (lower SCM pair AUC) against ORD_NLLM (MODEL_SELECT match), ORD_CNT (same count) and SHUF_NLLM (MODEL_SELECT match), on both seeds. None of these is cost-matched on DEV_REPORT, because SCM spent more:
>
>   | Comparator | Extra occupation NLL | Extra income NLL |
>   |---|---|---|
>   | ORD_NLLM | +0.0070 / +0.0017 | +0.0064 / +0.0035 |
>   | ORD_CNT | +0.0102 / +0.0055 | +0.0072 / +0.0063 |
>   | SHUF_NLLM | +0.0042 / +0.0017 | +0.0041 / +0.0026 |
> - Pair benefit against H_LJ, seed 0 (lower bound 0.0001).
> - SCM spent more occupation NLL than ORD_CNT on both seeds, and more income NLL than ORD_NLLM on seed 0 (ABOVE_ZERO).
> - The ORD_NLLM match mismatch on seed 0 (`MISMATCH_ESTABLISHED`; SCM spent more).
> - C0 minus SCM pair AUC (+0.039 / +0.030) and the 7-reader-slate ORD_NLLM minus SCM (+0.034 / +0.025), above zero.
> - SCM's pair reader above its occupation-only reader (+0.015 / +0.015). This is a difference between fitted readers; see Diagnostic.
> - Recipient non-worsening (PASS):
>   - income and occupation against ORD_NLLM;
>   - occupation against ORD_CNT;
>   - occupation against NEURAL_RAW, seed 0.
> - One occupation-recipient violation against H_LJ, seed 0 (`ESTABLISHED_VIOLATION`).
> - 3 of the 24 utility clauses (PASS), and the women's subgroup guard on both seeds (PASS).
> - Controls.
>
> **Point results (neither established nor refuted by an interval):**
> - SCM's 24 DEV utility clauses all pass at the point (`POINT_FEASIBLE_24`). The family-interval status is `CONFIDENCE_UNRESOLVED`.
> - POINT_FAILURE:
>   - the DEV occupation-NLL match against ORD_NLLM, both seeds;
>   - the men's subgroup point guard, both seeds;
>   - the IBSI_NLLM and SHUF_NLLM matches are also point mismatches (descriptive; the components were not tested).
> - POINT_EXCESS:
>   - income recipient against NEURAL_RAW (both seeds) and H_LJ (seed 1);
>   - occupation recipient against H_LJ (seed 1).
> - POINT_SHORTFALL: all four reference pair benefits are below the 0.020 target.
>
> **Unresolved:**
> - Every equivalence. All five other match endpoints are `EQUIVALENCE_UNRESOLVED`.
> - The PUB_IBSI pair benefits (MODEL_SELECT-matched and same-count) and the NEURAL_RAW pair benefits; the seed-1 benefit against H_LJ.
> - The men's guard. Its family intervals contain 0.005.
> - 21 of the 24 utility confidence statuses.
> - Recipient clauses: income against ORD_CNT (both seeds) and H_LJ (seed 0); occupation against NEURAL_RAW (seed 1).
> - The secondary differences:
>   - pair AUC: C0 − ORD_NLLM, ORD_CNT − ORD_NLLM, HIST_SCM − SCM;
>   - income NLL against ORD_NLLM, seed 1;
>   - occupation NLL against IBSI_CNT.
>
> **Unavailable (UNAVAILABLE category):**
> - Not tested, so evidence neither way:
>   - the IBSI and SHUF matched components (`NOT_TESTED_M_NOT_ESTABLISHED`);
>   - IBSI_CNT inside K (`NOT_TESTED_K_NOT_ESTABLISHED`).
> - No final-bank role is missing: all ten roles were filled on both seeds, and every endpoint is AVAILABLE.
> - Deployment: see SF-14. The missing seed-1 NEURAL_RAW package is a coverage gap. It is neither a demographic-inference result nor a method win.

### MF-4. The MODEL_SELECT mismatch is quoted with the wrong sign

**Evidence.**
- FROZEN DESIGN: `nll_diff` = SCM − comparator, and "Positive means SCM spends more".
- The report field `MODEL_SELECT_achieved_mismatch_SCM_minus_comparator` is −0.0000276 / +0.0001543.
- `SELECTION_LOCK` gives `SCM_minus_role_MODEL_SELECT.occupation_nll` = −0.00003 / +0.00015.
- The text uses `SELECTION_LOCK.delta`, which is comparator − SCM.
- Section 6 of the same file uses SCM − comparator.

**Current** (RESEARCH_DECISION line 52):
> - The strongest ordinary control within ±0.001 nats was found on both seeds: 129 / 105 cells, mismatch +0.00003 / −0.00015.

**Replace with:**
> - The strongest ordinary control within ±0.001 nats was found on both seeds: 129 / 105 cells, MODEL_SELECT mismatch (SCM − control) −0.00003 / +0.00015 nats.

See SF-11 for the unsigned Δ columns of `SELECTION_AND_MATCHING.md`.

### MF-5. The failed match is repaired by extrapolation

**Evidence.**
- Commissioning rule: a MODEL_SELECT match that does not carry over "must be stated as such and never repaired".
- On DEV_REPORT, the most any ordinary release in the bank spent was 0.0043 / 0.0054 nats over C0 (ORD_NLLM; `RECOVERY_TABLE.csv`). That is well below SCM's 0.0113 / 0.0072, so no fresh-reader evidence exists at SCM's DEV spend.
- The "flat" ordinary curve is a MODEL_SELECT cross-fitted proxy. It has the disclosed max-over-readers and LOOKUP-under-reading biases.
- On seed 0 that proxy spans 0.6448–0.6488, not 0.645–0.648. Seed 1 spans 0.6430–0.6483 up to 0.0125 nats.

**Current** (RESEARCH_DECISION line 71):
> - An ordinary control that spent as much as SCM would therefore probably not read much lower. This record does not establish that, because the DEV_REPORT match failed.

**Replace with:**
> - No ordinary release in the final bank spent as much DEV_REPORT occupation NLL as SCM. The highest was 0.0043 / 0.0054 nats over C0, against SCM's 0.0113 / 0.0072. How an ordinary control at SCM's DEV_REPORT spend would read is therefore not measured, and the failed match is not repaired by extrapolating from the MODEL_SELECT proxies.

Three related sentences overstate the range:

| File, line | Current | Replace with |
|---|---|---|
| RESEARCH_DECISION 68 | "Ordinary merging barely moves recovery at any loss level measured." | "Ordinary merging barely moved recovery at the loss levels measured." |
| RESEARCH_DECISION 69 | "On MODEL_SELECT, the ordinary trajectory's proxy stays at 0.645–0.648 from 0 to 0.0145 nats (seed 0)." | "On MODEL_SELECT (cross-fitted proxies, not fresh readers), the ordinary trajectory's proxy stays within 0.645–0.649 from 0 to 0.0145 nats on seed 0, and within 0.643–0.648 from 0 to 0.0125 nats on seed 1." |
| RESEARCH_DECISION 86 | "**Ordinary merging** reduces recovery very little at any expenditure (§2)." | "**Ordinary merging** reduced recovery very little at the expenditures measured (§2)." |
| START_HERE 20 (first sentence of item 4) | "Ordinary grouping barely changes recovery at any loss measured." | "Ordinary grouping barely changed recovery at the losses measured (MODEL_SELECT proxies up to 0.0145 / 0.0125 nats; fresh DEV_REPORT readers only up to 0.0043 / 0.0054 nats, below SCM's DEV spend)." |
| MANUSCRIPT 17 | "**Ordinary merging barely moves recovery** at any loss level measured (descriptive):" | "**Ordinary merging barely moved recovery at the loss levels measured** (descriptive; on DEV_REPORT no ordinary release reached SCM's spend):" |
| MANUSCRIPT 19 | "on MODEL_SELECT, the ordinary proxy stays flat at 0.645–0.648 across 0 to 0.0145 nats." | "on MODEL_SELECT (cross-fitted proxies), the ordinary proxy stays within 0.645–0.649 from 0 to 0.0145 nats on seed 0 and within 0.643–0.648 from 0 to 0.0125 nats on seed 1." |

### MF-6. The PUB_IBSI and SEX-guidance attributions are not separated from extra loss, reader budget or a failed comparator clause

**Evidence.** The IBSI component is `NOT_TESTED_M_NOT_ESTABLISHED`, and its descriptive status is `MATCH_DID_NOT_CARRY_OVER`. The PUB_IBSI comparison has five further problems:
- **Unresolved, not equivalent.** SCM's pair benefit over IBSI_NLLM is +0.0063 [−0.0077, 0.0203] / −0.0101 [−0.0248, 0.0046], UNRESOLVED on both seeds. A wide interval is not equivalence.
- **The match drifted.** DEV drift +0.0011 / −0.0039. On seed 1, PUB_IBSI spent more.
- **A comparator clause failed.** `comparison_differences["IBSI_NLLM|s1"].comparator_DEV_failed_point_clauses` lists `income|U0|accuracy_loss`, `income|Ucal|accuracy_loss` and `occupation|Ucal|nll_excess` (`DEV_point_feasible_12` false). The registered S4 qualifier ("a comparator failing a DEV point clause is named as a qualifier beside each comparison"; disposition: "Applied") is nowhere in the text.
- **Income not matched.** SCM spent more income NLL than PUB_IBSI (+0.0026 / +0.0011; `INCOME_EXPENDITURE_NOT_MATCHED`).
- **Reader budget favours PUB_IBSI.** Its partition is finer than SCM's: 471 / 471 ATTACK_FIT pair vectors, median 13 people, against SCM's 297 / 299 and 16 / 17. So in this comparison the reader-budget bias favours PUB_IBSI, and it is not reported beside the comparison.

"SEX-guided merging lowers recovery" and "points to the SEX-rate signal" rest on comparisons in which the SEX-guided release spent more DEV loss:
- SCM against ORD_NLLM, ORD_CNT and SHUF_NLLM: more occupation NLL and more income NLL on every comparison and seed.
- PUB_IBSI against ordinary and shuffled: PUB_IBSI's DEV spend was 0.0101 / 0.0111 nats over C0, against at most 0.0070 for the ordinary and shuffled controls.

"SCM minus IBSI pair benefit is +0.006" also reads as SCM AUC − IBSI AUC. The benefit is IBSI AUC − SCM AUC, so the text inverts the meaning on seed 0.

**Current** (START_HERE line 20):
> 4. **What accounts for the earlier difference.** Ordinary grouping barely changes recovery at any loss measured. The published IBSI objective, at matched MODEL_SELECT loss, read about the same as SCM (+0.006 / −0.010, unresolved). SEX-guided merging lowers recovery; SCM's specific ordering showed no measured edge over the published objective. This is descriptive.

**Replace with:**
> 4. **The earlier difference.** This item is descriptive: the components are `NOT_TESTED_M_NOT_ESTABLISHED`.
>    - Ordinary grouping barely changed recovery at the losses measured (MODEL_SELECT proxies up to 0.0145 / 0.0125 nats; fresh DEV_REPORT readers only up to 0.0043 / 0.0054 nats, below SCM's DEV spend).
>    - The published IBSI objective (PUB_IBSI, γ = 2.0), selected at a MODEL_SELECT match, read within about 0.01 AUC of SCM on DEV_REPORT. SCM's pair benefit was +0.006 / −0.010 (positive means SCM read lower), UNRESOLVED on both seeds. This is not equivalence.
>    - Its match also drifted (+0.0011 / −0.0039 nats). On seed 1 it fails three DEV utility point clauses.
>    - Its finer partition (median 13 ATTACK_FIT people per vector) can be under-read even more than SCM's.
>    - SCM's specific ordering showed no measured edge over the published objective.
>    - The SEX-guided releases (SCM, PUB_IBSI) read lower than the ordinary and shuffled controls, but they spent more DEV occupation and income loss, so the lower recovery is not separated from that extra loss.

**Current** (RESEARCH_DECISION line 85):
> **Ordinary grouping did not; the published IBSI objective largely did** (descriptive: the components were not tested because M was not established).

**Replace with:**
> **Ordinary grouping did not. The published IBSI objective read about as low as SCM, but neither comparison is cost-matched on DEV_REPORT** (descriptive: the components are `NOT_TESTED_M_NOT_ESTABLISHED`, with descriptive status `MATCH_DID_NOT_CARRY_OVER`).

**Current** (RESEARCH_DECISION lines 88 and 91):
> - SCM minus IBSI pair benefit is +0.006 / −0.010, unresolved on both seeds. On seed 1, PUB_IBSI read lower than SCM.
>
> - SEX-guided merging lowers measured recovery. SCM's specific ordering shows no measured advantage over the published objective at similar loss.

**Replace with:**
> - SCM's pair benefit over PUB_IBSI (PUB_IBSI AUC − SCM AUC) is +0.006 / −0.010, UNRESOLVED on both seeds; the family intervals are [−0.008, 0.020] and [−0.025, 0.005]. This is not equivalence.
> - On seed 1, PUB_IBSI read lower than SCM while spending 0.0039 nats more occupation NLL. That seed-1 PUB_IBSI release fails three DEV utility point clauses: income accuracy loss against U0 and Ucal, and occupation NLL excess against Ucal. This is a comparator qualifier; it does not change the release's MODEL_SELECT eligibility.
> - SCM spent more income NLL than PUB_IBSI on both seeds (+0.0026 / +0.0011; `INCOME_EXPENDITURE_NOT_MATCHED`).
> - Reader budget: PUB_IBSI keeps 471 / 471 ATTACK_FIT pair vectors (median 13 people), against SCM's 297 / 299 (median 16 / 17). In this comparison the fixed budget can under-read PUB_IBSI more than SCM.
>
> - The SEX-guided releases (SCM, PUB_IBSI) read lower than the ordinary and shuffled controls. In each of those comparisons the SEX-guided side spent more DEV_REPORT occupation and income NLL, so the lower recovery is not separated from that extra loss. SCM's specific ordering shows no measured advantage over the published objective.

**Current** (RESEARCH_DECISION line 92):
> - **Shuffled control (r2)** at a MODEL_SELECT match: SCM benefit 0.020 / 0.020, family-established. This points to the SEX-rate signal in the merge order rather than merging per se. The match drifted here too (+0.0042 / +0.0017), and shuffled controls are component comparisons, not permutation tests.

**Replace with:**
> - **Shuffled control (r2)** at a MODEL_SELECT match: SCM benefit 0.020 / 0.020, family-ESTABLISHED. It is not cost-matched on DEV_REPORT:
>   - the match drifted, with SCM spending +0.0042 / +0.0017 more occupation NLL;
>   - SCM also spent +0.0041 / +0.0026 more income NLL.
>
>   The result is consistent with a role for the SEX-rate signal in the merge order, but it is not separated from SCM's extra loss. Shuffled controls are component comparisons, not permutation tests.

**Current** (MANUSCRIPT line 20):
> - **The published IBSI objective accounts for most of SCM's advantage over ordinary merging.** At matched MODEL_SELECT loss it read about the same as SCM (+0.006 / −0.010, unresolved). This is descriptive; the component sequence was not tested.

**Replace with:**
> - **No measured edge for SCM's ordering over the published IBSI objective (descriptive).**
>   - Selected at a MODEL_SELECT occupation-NLL match, the published IBSI objective read within about 0.01 AUC of SCM on DEV_REPORT.
>   - SCM's pair benefit was +0.006 / −0.010, unresolved on both seeds. This is not equivalence.
>   - Its match also did not carry over (+0.0011 / −0.0039 nats). On seed 1 it fails three DEV utility point clauses.
>   - Its finer partition can be under-read more than SCM's.
>   - The component sequence was not tested, because the matched claim was not established.

Add to "Must not be stated" in `MANUSCRIPT_CLAIMS_HANDOFF.md`:
> - That PUB_IBSI and SCM are equivalent, that PUB_IBSI "accounts for" SCM's difference from ordinary merging, or that the SEX-guided releases' lower recovery is separated from the extra occupation and income loss they spent.

### MF-7. The common-cell (redundancy) check is described as covering every release

**Evidence.** `diagnostics.per_role.{S0,S1}` has a `common_cell_check` only for the eight grouped roles: SCM, ORD_NLLM, ORD_CNT, IBSI_NLLM, IBSI_CNT, SHUF_NLLM, C0 and HIST_SCM. Each covers ATTACK_FIT, ATTACK_VALIDATION, MODEL_SELECT and DEV_REPORT. NEURAL_RAW and H_LJ have no such check (keys: `DEV_REPORT_occupation_by_sex`, `DEV_REPORT_vs_C0`, `release`, `service`).

**Current** (RESEARCH_DECISION line 149):
> For every release on every role, the occupation output identifies the shared cell and income is a function of it (distinct occupation vectors = occupied cells = distinct pair vectors). So a Bayes-optimal reader gains nothing from the linked pair beyond the occupation output for these people.

**Replace with:**
> For every grouped release in the bank (SCM, ORD_NLLM, ORD_CNT, IBSI_NLLM, IBSI_CNT, SHUF_NLLM, C0 and HIST_SCM, both seeds), on every role checked (ATTACK_FIT, ATTACK_VALIDATION, MODEL_SELECT and DEV_REPORT), the occupation output identifies the shared cell and income is a function of it (distinct occupation vectors = occupied cells = distinct pair vectors). The check was not computed for NEURAL_RAW or H_LJ. For these grouped releases, a Bayes-optimal reader therefore gains nothing from the linked pair beyond the occupation output for these people.

**Current** (MANUSCRIPT line 47):
> - **Redundancy.** For every release, the occupation output identifies the shared cell and the income output is a function of it.

**Replace with:**
> - **Redundancy.** For every grouped release in the bank (the check was not run for NEURAL_RAW or H_LJ), the occupation output identifies the shared cell and the income output is a function of it.

### MF-8. Reader budget: "average" is a median, "about three times" and "At matched loss" are wrong, and the caveat is not placed beside the comparisons

**Evidence.**
- `per_seed.*.coverage_attack_roles` gives `median_ATTACK_FIT_people_per_vector` = 16 / 17 for SCM and 53 / 64 for ORD_NLLM. Those are medians. The means are about 26.9 / 26.8 against 62.0 / 76.2.
- Occupied ATTACK_FIT pair vectors are 297 / 299 against 129 / 105. That is 2.3 / 2.8 times, not about three times.
- The share of ATTACK_FIT people in vectors with at least 15 people is 0.867 / 0.872 against 0.991 / 0.995.
- Loss was not matched on DEV_REPORT.
- PROTOCOL §8 and disposition item S4 require the bias "beside every recovery comparison". It currently appears only in RESEARCH_DECISION §7.

**Current** (RESEARCH_DECISION line 150):
> - **Reader budget.** With 8,001 ATTACK_FIT people, SCM's pair vectors average 16 / 17 people, against 53 / 64 for the matched ordinary control. The fixed budget can under-read SCM's finer partition, which favours SCM (disclosed before fitting).

**Replace with:**
> - **Reader budget.**
>   - With 8,001 ATTACK_FIT people, SCM's 297 / 299 occupied pair vectors hold a median of 16 / 17 people each. The 129 / 105 vectors of the MODEL_SELECT-matched ordinary control hold a median of 53 / 64.
>   - 87 % of ATTACK_FIT people sit in SCM vectors with at least 15 people, against 99 % for that control.
>   - Other roles: ORD_CNT median 21, SHUF_NLLM 33 / 30, PUB_IBSI (IBSI_NLLM) 13.
>   - The fixed budget can under-read finer partitions. Against the ordinary and shuffled controls this favours SCM; against PUB_IBSI it favours PUB_IBSI. The bias was disclosed before fitting.

**Current** (MANUSCRIPT line 46):
> - **Reader budget.** At matched loss SCM keeps about three times as many pair vectors as the ordinary control: a median of 16 / 17 against 53 / 64 people per vector. A fixed reader budget can under-read finer partitions, which favours SCM.

**Replace with:**
> - **Reader budget.** Against the MODEL_SELECT-matched ordinary control, SCM keeps 2.3 / 2.8 times as many occupied pair vectors on ATTACK_FIT (297 / 299 against 129 / 105), with a median of 16 / 17 against 53 / 64 people per vector. A fixed reader budget can under-read finer partitions. That favours SCM against ordinary merging and favours PUB_IBSI (median 13) against SCM.

**Insert** one line beside each recovery comparison in RESEARCH_DECISION:

| Where | Insert |
|---|---|
| §2, after line 60 | "- Reader budget: SCM's pair vectors hold a median of 16 / 17 ATTACK_FIT people against 53 / 64 for this control. The fixed budget can under-read SCM's finer partition, which favours SCM. For these grouped releases the pair is redundant with the occupation output (§7), and the lookup reader is not population-optimal." |
| §3, after line 77 | "- Reader budget: ORD_CNT occupies 298 / 300 ATTACK_FIT vectors, against SCM's 297 / 299, but its people are spread more evenly (median 21 against 16 / 17). The budget bias is therefore smaller here, but it still favours SCM." |
| §5, after line 99 | "- Reader budget: no occupancy diagnostic exists for NEURAL_RAW or H_LJ. The fixed-budget and lookup-reader caveats apply to every AUC here." |

§4 is covered by MF-6.

### MF-9. A verifier replay is described as done, but no verifier record exists

**Evidence.**
- At review time `INDEPENDENT_VERIFICATION.json` and `roles/verifier/VERIFICATION.md` did not exist.
- `roles/verifier/` holds only `check1_admission.*` and `vlib.py`.
- The DEV_REPORT replay is Check 5 of the verifier brief.

**Current** (START_HERE line 70):
> - One separately logged verifier replay.

**Replace with**, until the verifier record exists:
> - One separately logged verifier replay is registered (Check 5 of the verifier brief, event `DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER`). Its result is pending and will be in `INDEPENDENT_VERIFICATION.json` and `roles/verifier/VERIFICATION.md`.

**Replace with**, once the verifier record exists:
> - One separately logged verifier replay (`DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER`, <UTC from the access log>): <per-check PASS/FAIL and discrepancy count, quoted from `INDEPENDENT_VERIFICATION.json`>.

Also check every number the verifier changes or flags against this record before closeout.

### MF-10. "Class 0 more often" is false on seed 1

**Evidence.** `per_seed.S0/S1.services.{SCM,C0}.diagnostics.occupation_predicted_class_frequency[0]`:

| Seed | SCM | C0 |
|---|---|---|
| 0 | 0.5945 | 0.5894 |
| 1 | 0.5872 | 0.5889 |

**Current** (RESEARCH_DECISION line 151):
> It predicts occupation class 3 less often (0.026 / 0.020 against C0's 0.036 / 0.028; true 0.090) and class 0 more often.

**Replace with:**
> It predicts occupation class 3 less often (0.026 / 0.020 against C0's 0.036 / 0.028; true 0.090). It predicts class 0 slightly more often on seed 0 (0.594 against 0.589) and slightly less often on seed 1 (0.587 against 0.589).

---

## SHOULD-FIX

**SF-1. The direction sub-status says only who spent more.** (Brief item 2.)
- START_HERE line 17, "That is the confounding direction, and seed 0's mismatch is established.", becomes:
  > The direction sub-status is `SCM_SPENT_MORE` on both seeds; it states only who spent more. Seed 0's mismatch is established (`MISMATCH_ESTABLISHED`), and seed 1's is `EQUIVALENCE_UNRESOLVED`. Because SCM spent more, its lower pair AUC is not separated from the extra loss.
- RESEARCH_DECISION line 64, "- It is in the direction that recreates the original confound: SCM spent more prediction loss.", becomes:
  > - The direction sub-status is `SCM_SPENT_MORE` on both seeds. It states only who spent more; because SCM did, the lower pair AUC is not separated from the extra occupation loss.

**SF-2. Print the nested 0.020 status and the comparison-only status with the decision** (MF-6 of the statistical review). In RESEARCH_DECISION, after line 13, add:
> - nested 0.020 target (claim M): `NOT_ESTABLISHED` (seed 0 clause bound 0.0225 > 0.020; seed 1 0.0144, UNRESOLVED). This is comparison-only and not cost-matched on DEV_REPORT;
> - comparison-only status: `BENEFIT_AND_NON_WORSENING_CLAUSES_PASS (not cost-matched on DEV_REPORT)`.

At line 60, "…holds on seed 0 (0.0225) but not on seed 1 (0.0144)." becomes:
> …holds on seed 0 (0.0225) but not on seed 1 (0.0144, UNRESOLVED; nested status `NOT_ESTABLISHED`). It is not cost-matched on DEV_REPORT either way.

**SF-3. Give the income-NLL statuses.**
- RESEARCH_DECISION line 65: append "(secondary family: ABOVE_ZERO on seed 0, [0.0015, 0.0114]; UNRESOLVED on seed 1, [−0.0020, 0.0090])".
- MANUSCRIPT line 15: "and also more income NLL" becomes "and also more income NLL (+0.0064 / +0.0035; above zero on seed 0, unresolved on seed 1)".

**SF-4. "Which then regresses" is an interpretation, and SCM's own drift is omitted.**
- RESEARCH_DECISION line 115, "Matching a comparator on its MODEL_SELECT estimate therefore selects partly on noise, which then regresses.", becomes:
  > Matching a comparator on its MODEL_SELECT estimate therefore selects partly on noise. The drift seen on DEV_REPORT is consistent with that: the control's spend fell by 0.0040 / 0.0028, while SCM's rose by 0.0030 on seed 0 and fell by 0.0012 on seed 1. This record does not separate noise from a real difference between the selection and evaluation people.
- START_HERE line 23, "…so a MODEL_SELECT match selects partly on noise.", becomes:
  > …so a MODEL_SELECT match selects partly on noise. Both sides moved: see the next line.

**SF-5. Finish the incomplete bullet.** RESEARCH_DECISION line 124, "- **Recipient-level reads against the references.**", becomes:
> - **Recipient-level reads against the references.** SCM's income recipient read higher than both references at the point (POINT_EXCESS against NEURAL_RAW on both seeds and against H_LJ on seed 1; UNRESOLVED against H_LJ on seed 0). Its occupation recipient read higher than H_LJ's (ESTABLISHED_VIOLATION on seed 0, POINT_EXCESS on seed 1).

**SF-6. Quote claim B's label and categories wherever "not competitive" appears.**
- RESEARCH_DECISION line 96, "**Not competitive.** The subgroup gate fails first, and the recipient conditions fail as well.", becomes:
  > **Not competitive: claim B is `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.** The first failing condition is the men's occupation Brier point guard (POINT_FAILURE), and the comparison-only status is `COMPARISON_CLAUSES_NOT_ALL_PASS`. The recipient conditions fail as well (categories below).
- START_HERE line 21 becomes:
  > 5. **References.** Not competitive (claim B `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`). Pair benefits against NEURAL_RAW (+0.017 / +0.005) and H_LJ (+0.018 / +0.010) are UNRESOLVED on the family interval except H_LJ seed 0 (ESTABLISHED), and all four are below the 0.020 target at the point (POINT_SHORTFALL). SCM's income recipient reads higher than both references (POINT_EXCESS on three of four clauses, UNRESOLVED on H_LJ seed 0). Its occupation recipient reads higher than H_LJ's (ESTABLISHED_VIOLATION on seed 0, POINT_EXCESS on seed 1).
- MANUSCRIPT lines 24–27 become:
  > - **Not competitive with the frozen references** (claim B `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`).
  >   - Pair benefits against NEURAL_RAW and H_LJ are unresolved except H_LJ seed 0, and all four are below 0.020 at the point.
  >   - SCM's income recipient reads higher than both references at the point.
  >   - Its occupation recipient reads higher than H_LJ's, an established violation on seed 0.

**SF-7. Quote claim K's label in START_HERE.** Line 19 becomes:
> 3. **Same capacity.** Claim K is `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD` (men's guard, POINT_FAILURE). The benefit over the same-count ordinary control is 0.036 / 0.028 (clause bounds 0.024 / 0.017), but SCM spent +0.010 / +0.006 nats more occupation NLL (above zero on both seeds) and +0.007 / +0.006 more income NLL. The income non-worsening clauses are UNRESOLVED (comparison-only status `COMPARISON_CLAUSES_NOT_ALL_PASS`).

**SF-8. Qualify bet P5.** RESEARCH_DECISION line 161, "| P5 (0.55) | held |", becomes:
> | P5 (0.55) | held (scored on ORD_NLLM's MODEL_SELECT role status MATCHED; the DEV_REPORT match did not carry over, so the comparison it scores is not cost-matched) |

**SF-9. Describe ordinary merging's SEX access accurately** (PROTOCOL §2, "SEX access, described accurately").
- RESEARCH_DECISION line 3: after "ordinary task-only merging" add
  > (a task-only merge objective that still uses SEX through the shared within-SEX guard; its comparator was selected with the SEX-labelled MODEL_SELECT reader proxy)
- START_HERE line 7: add the same parenthetical after "ordinary merging".

**SF-10. QUICKSTART.md.**
- **Refusals (line 55).** Current: "**Refusals** (both entry points): extra, missing or reordered columns or keys; any forbidden column (SEX, for example); any request for anything but probability and decision (cell or group ids, teacher probabilities, propensity)." The problem: `DEPLOYMENT_RESULTS.json` has no `api_missing_key` test, and the `hidden_*` requests were exercised only through the API. Replace with:
  > **Refusals.** Both entry points share one input check (exactly the six fields, in this order, and no forbidden field) and one output check (probability and decision only). `DEPLOYMENT_RESULTS.json` exercised:
  > - CSV path: an extra, missing or reordered column; a SEX column; and SEX replacing a field;
  > - API path: an extra key, reordered keys and a SEX key;
  > - through the API: requests for cell id, fine cell, group id, income group, teacher probability or propensity.
  >
  > A missing API key, and hidden-output requests through the CSV path, are refused by the same shared checks but were not exercised separately.
- **Step 9 (line 33).** Append:
  > It has been used. `scm_um.common.dev_report_truth` refuses a second coordinator re-read, so a reproduction starts from the committed `report/DEV_REPORT_RESULTS.json`. The verifier's single replay is a separate registered event.
- **Custody (lines 68–70).** Replace with:
  > `scm_um/custody.sh backup <dest parent>` writes a branch bundle, a source archive and the private tree. It restores them into a fresh directory and, from the backup alone, re-runs two checks:
  > - the SCM, ORD_NLLM and IBSI_NLLM packages (both seeds), compared bitwise with the restored evaluated wires;
  > - the synthetic tests.
  >
  > The frozen NEURAL_RAW and H_LJ source packages are checked in `DEPLOYMENT_RESULTS.json`, not in the restore check. `scm_um/custody.sh offdevice <backup dir> <dest>` repeats the check on the external drive. Record whether that read-back is warm (the drive stayed mounted, so the file cache may have served it) or cold (after an ordinary remount). `BACKUP_VERIFICATION.json` holds the same-device result, the warm off-device result, the cold re-check when it is done, and the final refresh.

**SF-11. SELECTION_AND_MATCHING.md.** This file is generated by `scm_um.tables`, so fix the generator and regenerate.
- Sign the Δ columns: "Δ expenditure vs SCM" becomes "Δ expenditure (comparator − SCM)", and "Δ expenditure" becomes "Δ expenditure (candidate − SCM)". The report and RESEARCH_DECISION §6 use SCM − comparator.
- Add under the title:
  > "MATCHED" below is the MODEL_SELECT role status. None of these matches carried over to DEV_REPORT (`RESEARCH_DECISION.md` §6).
- Print the matching-outcome lines without Python dict syntax (`{'0': True, '1': True}`).

**SF-12. MANUSCRIPT_CLAIMS_HANDOFF.md.**
- **Add a decision bullet** at the top of "May be stated":
  > - **Decision.** `COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE` (claim M `MATCH_DID_NOT_CARRY_OVER`). Qualifiers:
  >   - DEV utility confidence `CONFIDENCE_UNRESOLVED`;
  >   - claim C `SUBGROUP_UNRESOLVED`;
  >   - ORD_NLLM DEV match `MISMATCH_ESTABLISHED` (seed 0) and `EQUIVALENCE_UNRESOLVED` (seed 1), with SCM spending more on both;
  >   - claim B `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.
  >
  >   Claim K is also `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.
- **Source labels (line 37).** Current: "They stand: `RECIPE_EFFECT_NOT_ESTABLISHED`, `NOMINEE_FAILS_DEV_UTILITY_GATE`, `SUBGROUP_UNRESOLVED`." Replace with:
  > They stand (source `HANDOFF.json`):
  > - A (recipe vs twin): `RECIPE_EFFECT_NOT_ESTABLISHED`;
  > - attribution vs PUB_IBSI and SHUF: `NOT_TESTED_RECIPE_NOT_ESTABLISHED`;
  > - B: `NOMINEE_FAILS_DEV_UTILITY_GATE`;
  > - C: `SUBGROUP_UNRESOLVED`;
  > - D and E: `NOT_CLAIMED`.
- **Suggested sentence (line 51).** Replace with:
  > "In an exploratory follow-up on previously used development records, a conservative shared-cell merging endpoint met all 24 utility clauses at the point (simultaneous confidence unresolved). It exceeded the men's within-sex occupation Brier guard at the point (unresolved, not established harm). Its linked-pair sex recovery read lower than an ordinary-merging control selected for matching held-out occupation log-loss in the selection role. That match did not carry over to the evaluation role, where the merged release spent more occupation and income log-loss, so the comparison is inconclusive at the registered 0.001-nat tolerance, a tolerance these sample sizes cannot certify. The fixed reader budget may under-read the finer merged partition, and the release was not competitive with the frozen neural and tree references."
- **Add to "Must not be stated":**
  > - That the missing seed-1 NEURAL_RAW package, or any other missing coverage, is a demographic-inference result or a method win.

**SF-13. Small wording fixes in RESEARCH_DECISION.**
- **§1, line 45.** "The cost of SCM's merges falls on men, as in the source." becomes "The cost of SCM's merges falls mostly on men, as in the source." Women's occupation NLL also rose, by 0.0076 / 0.0049.
- **§3, line 81.** "the benefit of 0.010 / 0.003 is unresolved and was not tested in sequence" becomes:
  > the benefit of 0.010 / 0.003 is UNRESOLVED on the family interval and was not tested (`NOT_TESTED_K_NOT_ESTABLISHED`). Seed 0's clause bound (0.0022 > 0) is therefore not a test.

**SF-14. Say which roles are packaged** (brief item 7). In RESEARCH_DECISION line 145, "- Only deployment has a gap: no seed-1 NEURAL_RAW package was ever built by the source." becomes:
> - Deployment: packages exist for SCM, ORD_NLLM and IBSI_NLLM (both seeds), the frozen NEURAL_RAW (seed 0) and H_LJ (both seeds). Each reproduces its evaluated DEV_REPORT wire bitwise through the API and CSV paths, and all tested refusals hold. No seed-1 NEURAL_RAW package exists (the source never built one); that is a coverage gap, not a result. ORD_CNT, IBSI_CNT, SHUF_NLLM, C0 and HIST_SCM were not registered for packaging.

---

## NOTES

### N-1. Rounding

| Location | Text | Record | Fix |
|---|---|---|---|
| RESEARCH_DECISION line 45, women's NLL rise over C0, seed 1 | 0.0050 | 0.00492 | 0.0049 |
| RESEARCH_DECISION §6 table, SCM MODEL_SELECT, seed 0 | 0.0083 | 0.008248 | 0.0082 (§1's "0.00825" is right) |
| Seed-0 ordinary proxy maximum | 0.648 | 0.6488 | 0.649 (MF-5) |

### N-2. Verified correct (no change needed)

- **Decision and claims.** The decision and its four qualifiers in RESEARCH_DECISION; claim labels M, K, B, C, D and E; separate status fields (`POINT_FEASIBLE_24`, `ORIGINAL_POINT_FEASIBLE_24`, `NOMINATION_ELIGIBLE`, `CONFIDENCE_UNRESOLVED` 3 / 21 / 0).
- **Endpoint numbers.** Every M, K, B, C and utility point, clause bound and family interval quoted (apart from the categories in MF-3); the match SEs (0.00155–0.00201); the §6 drift table; IBSI and shuffled drifts.
- **Utility and subgroup.** Men's and women's Brier excesses; NEURAL_RAW and H_LJ occupation-NLL excesses and men's burdens (0.0000 / 0.0021, 0.0013 / 0.0022, 0.0032 / 0.0040); fitted, MODEL_SELECT and DEV expenditures.
- **Selection.** The nomination grid; the historical nominee (261 / 266 cells; MODEL_SELECT seed-1 men's excess 0.00595; DEV Ucal NLL 0.0106 fails); selection effort; bets and mean Brier 0.10375.
- **Controls and readers.** Controls (planted 1.0; shuffled 0.487–0.517; computed on SCM's wires); 246 reader fits and 32 / 32 bitwise replay.
- **Admission.** 12 manifests, 51 copies, 31 / 31 arms bitwise on both seeds.
- **Trajectories.** 10 per seed; ORD terminal at 60 / 58 cells; IBSI at 32; 272 / 261 effective releases; gate ALL_PASS.
- **PUB_IBSI curve.** On MODEL_SELECT, PUB_IBSI γ = 2 traces about the same or a lower proxy curve than SCM. It is never more than 0.002 above (+0.0018 at seed 1's matched point), so "the same or lower" is approximately right.
- **Deployment.** The `DEPLOYMENT_RESULTS.json` packages and the NEURAL_RAW_S1 gap. Inputs are `AGEP,WKHP,COW,SCHL,MAR,POBP`. The environment versions match `SOURCE_ADMISSION.environment`. The QUICKSTART commands and arguments exist in `scm_um/`, and the test modules exist.

### N-3. Receipts and branches

- `738dcbbbe` was committed at 05:19:18Z and `3c7d69317` at 05:30:02Z. Both are on `publication/pcrl-scm-utility-matched-v1`, and `738dcbbbe` is an ancestor of its tip `3c7d69317`.
- START_HERE's "pushed 05:19:22Z / 05:30:06Z" are push times, which git objects cannot confirm. Cite the push log in `HANDOFF.json`.
- The report's `utc` (05:30:14Z) is after the reporting receipt.
- The manuscript branch (`9e6d01f28`) and claims branch (`33124f965`) tips predate this follow-up, so "Neither was modified" holds.
- `results/pcrl_scm_engineering_benchmark_v1/` is unchanged between `2738cc30b` and HEAD, and the source labels match the source `HANDOFF.json`.

### N-4. The "52-check" gate

The gate has 52 keys, including the summary key `ALL_PASS`. Either say "51 checks plus ALL_PASS", or confirm that 52 is the intended count.

### N-5. Records that did not exist at review time

`INDEPENDENT_VERIFICATION.json`, `roles/verifier/VERIFICATION.md`, `RESOURCE_ACCOUNTING.json` and `HANDOFF.json` were missing; START_HERE read-order items 8–9 list them. `BACKUP_VERIFICATION.json` has since appeared. When the rest land, check the following:

- **Verifier:**
  - Check 5 (DEV replay) agrees with the report on every label and endpoint, within its stated tolerance;
  - Check 6 covers the refusals that SF-10 says were not exercised, or the wording stays as in SF-10;
  - any discrepancy D-n that touches a number quoted in these five files is corrected here.
- **Custody (brief item 8).** At review time `BACKUP_VERIFICATION.json` shows:
  - same-device: PASS at `b9299804f`;
  - off-device: PASS, but **WARM** (the cold re-check after an ordinary remount is pending);
  - `final_refresh`: pending.

  The closeout text must keep three things distinct: same-device, warm off-device read-back, and cold read after remounting. It must not call the off-device check cold unless that check was run. The final refresh must cover the last content commit, including the fixes from this review.
- **Resources (brief item 9).** `RESOURCE_ACCOUNTING.json` must charge every unit, including:
  - verifier work, reader refits and agent analysis (with upper estimates);
  - failed attempts, such as any repeated custody, deploy or table run.

  Any figure quoted in START_HERE or HANDOFF must equal the ledger. PROTOCOL §15 estimates about 4.8 CPU-h against a 10 CPU-h, 6-hour cap starting 04:38:00Z.
- **Handoff (brief item 10).** `HANDOFF.json` should restate the decision with its four qualifiers, exactly as `research_decision_qualifiers` does.
- **Publication.** The publication branch currently holds only the two receipts. The handoff tells maintainers to cite `publication/pcrl-scm-utility-matched-v1`, so the final sanitised package must be pushed there before anyone cites it.

### N-6. Commit message

The message of `b9299804f` says "PUB_IBSI accounts for most of the earlier difference (descriptive)". History is not rewritten, so the corrected files (MF-6) supersede it. A later commit message can say so.

### N-7. Wording that is already right (keep)

- RESEARCH_DECISION line 3: the controlled-benchmark assumptions, conditioning and seeds sentence.
- §2 line 66: "It is not a matched result."
- §7 Diagnostic: the fitted-reader versus information distinction and "The lookup reader is not population-optimal".
- MANUSCRIPT "Must not be stated": the novelty, equivalence, whole-utility-vector, lookup-optimality, sealed-assessment and generalisation items.
- START_HERE line 8: no novelty, new-algorithm or joint-design claim.
- START_HERE line 6: the sealed assessment stayed closed.
