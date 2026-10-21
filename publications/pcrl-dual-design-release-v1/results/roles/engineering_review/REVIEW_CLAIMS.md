# Claims review (engineering and claims reviewer, campaign `pcrl_dual_design_release_v1`)

Written 2026-10-21, 19:45:53Z to 20:08Z (`date -u`), following `roles/coordinator/CLAIMS_REVIEW_BRIEF.md`. The author is the
engineering and claims reviewer, an agent. Line numbers refer to the current files at commit **ae6626c44** (20:04:58Z), the documents
rewritten to apply the closing statement.

**Inputs.**
- The documents at 0678392a3 (first pass) and at ae6626c44 (re-based):
  - `START_HERE.md` (SH)
  - `RESEARCH_DECISION.md` (RD)
  - `ADVISOR_BRIEF.md` (AB)
  - `HANDOFF.md` (HO)
  - `QUICKSTART.md` (QS)
  - `PROTOCOL_DEVIATIONS.md` (DEV, deviations 1-13)
  - `TABLES.md` / `TABLES.json`
- The records named in the brief; `git log` of `research/pcrl-dual-design-release-v1`; the branch
  `publication/pcrl-dual-design-release-v1`.
- Both verifier reports and their records.
- `roles/statistical_reviewer/CLOSING_STATEMENT.md` (CS).
- The aggregate private `.private/.../EVAL_MS.json`.

**Method.** The checks were split among the reviewer and three forked sub-agents: numbers and labels; paths, commands and commits;
utility, fits and triggers. Every finding below was re-checked against the record named.

**Constraints observed.**
- No label was read. No real-data computation and no queue unit was run.
- The only computations were the label-free table regeneration (section 5) and `python3 -m ddr.test_rules` on invented data (22 / 22
  pass).
- No campaign file was changed except this one. `TABLES.md` / `TABLES.json` were restored byte for byte, and `git status` shows them
  clean.

## 0. Summary

- **Labels and numbers.** Every D-label, outcome class, D2 / D2R / D3 / D4 mean and se, beside comparison, utility value, reserve
  ratio, budget count, fit parameter, card order, path and command checked matches its record.
- **Tables.** They regenerate identically, apart from `utc`, against both the 19:43Z and the HEAD (19:47Z) versions.
- **Earlier findings.** The ae6626c44 rewrite fixed most of the first pass's 42 findings, in nearly every case through a CS C-item.
  Section 1.1 gives the status of each.
- **What is still open.** Section 1.2. Each item is short, but three are factual:
  - R37 and R36: per-version utility counts and clause-vs-guard. R37's error was introduced in six places by applying C29's
    wording.
  - R43: C15's "every release had lower linked-pair recovery than the twin" is false for three saved controls.
  - R15: "did worse ... not resolved" contradicts itself; the reader check quotes one version, with the opposite sign convention.
- **Strength.** No claim in the current files is stronger than the rules allow, except the residues in R15 and R31, and HO 49
  against RULES.md 6 (R12b).

## 1. Required corrections

### 1.1 Status of the first-pass findings (drafts at 0678392a3) in the current files

**Fixed or covered by a CS C-item:**

| Finding | Topic | Covered by (current lines) |
|---|---|---|
| R1 | One idea, not two | C1, C18 (RD 50-51, SH 21-22, AB 17, AB 31-32, HO 12-13) |
| R2 | Full campaign-level statement | C21 (RD 55, SH 26, AB 45, HO 17) |
| R3 | "Strongest result" | C3, C4 |
| R4, R5 | "Both lowered" / "gains" | C15 (AB 84, HO 57, RD 291-293) |
| R6 | Recipient gate allowance | C11 (RD 158) |
| R7 | DA_COMP "essential" | C12 (RD 251; HO 64) |
| R8, R9 | Ablation shares, "gained" | C13 (RD 305) |
| R10 | "Is the twin's" | C10 (RD 185) |
| R11 | "Spend-matched" | C6, deviation 13 |
| R13 | Point-only adverse statements | C16, C24 (RD 87, 359; SH 48) |
| R14 | "Consistent" | C25 (RD 342) |
| R16 | Verifier's reader check | C24 (RD 283-284) |
| R17 | "All 32 cells" | Sentence removed |
| R18 | AB scope | AB 10 |
| R22 | Deviation 12(b) flag | RD 111 |
| R23 | Extra D2R reading | C22 (RD 224, 230; AB 66) |
| R24 | Deviation 8 | C14 (RD 241) |
| R26 | Untested section | RD 382-392; C9 (no finer SBC grid) |
| R28 | Data-rich scope | C23 (RD 205-208) |
| R29 | "At or below 0.6217" | RD 261 |
| R30 | "About +0.017" | AB rewritten |
| R38 | Data roles | AB 5-6 now "roles' labels"; RD 398 discloses the label-free DEV_REPORT features |
| R41 | DA R2 reason | AB untested list rewritten; RD 372, HO 90-91 |
| R42 | "Misses zero" | C8 (RD 177) |
| A2 | Prefixed labels | C20 (RD 94) |
| A6 | STD and SEX | C17 (AB 24-27). Checked: `da/R1_TRIGGER_B2.json` R^2 0.0665 vs 0.0999 (V0) |
| A10 | Record quirk | C19, deviation 12(e) |

**Partly fixed:** R12, R15, R31 and A1. **Still open:** R19, R20, R21, R25, R27, R32, R34, R35, R36, R37, R39 and R40. The open items
are below, with new items R43 and R44.

### 1.2 Open corrections in the current files

**R37. Utility conditions are counted per release but stated per configuration.** This error was introduced by applying C29's wording.
- Quoted text:
  - RD 59: "Every saved configuration met all 28 original utility conditions at the point (G1)."
  - RD 134-135: "Every saved configuration (48 of the 48-configuration budget) met all 28 original utility conditions at the point
    (G1): the 24 clauses plus the four occupation subgroup guards."
  - SH 23: "Every saved configuration met all 28 original utility conditions at the point (G1)."
  - AB 13: "Every saved configuration met all 28 original utility conditions at the point."
  - AB 38: "All 48 saved configurations meet all 28 original utility conditions at the point".
  - HO 14: "Every saved configuration met all 28 original utility conditions at the point."
- Record:
  - A configuration is one version (PROTOCOL 8: "seed- / version-specific release configurations").
  - Each version's utility record has 12 clauses and 2 guards (`ddr/evaluate.py` `point_status`, key `point_pass_12_plus_2`; every
    `EVAL_MS.json` has n = 12 + 2).
  - G1's 24 + 4 = 28 conditions span the two versions of a release (`ddr/rules.py` d1).
  - All 48 rows of `dev/UTILITY_LEDGER.csv` pass, so the substance holds.
- Corrected wording (RD 59, SH 23, AB 13, HO 14): "Every release met all 28 original utility conditions at the point (G1: 12 clauses
  and 2 guards in each of its two version-specific configurations; all 48 configurations pass)." Apply the same change to RD 134-135
  and AB 38.
- Same error in the tables: TABLES.md section 1 says "Per version: original point status (24 clauses + 4 guards)". This text comes
  from `ddr/report.py` line 111. Change it to "(12 clauses + 2 guards)" and regenerate. Only that line changes.

**R36. "Worst clause" is a maximum over clauses and guards, and one value is mis-rounded.**
- Quoted text:
  - RD 140: "The twin passes with a wide margin (worst clause -0.43 / -0.10 of its cap)."
  - RD 143 header: "| Release | Worst clause / cap (V0 / V1) | ...".
  - RD 280: "(worst clause 0.79-0.88 of its cap)".
- Record:
  - `ddr/evaluate.py` `point_status` takes `max_value_over_cap` over the 12 clauses and the 2 guards.
  - In `EVAL_MS.json`, the maximum for DA_STD (-0.089 / +0.099) and for DB_PTC_R1 (+0.011 / +0.325) is the men's occupation Brier
    guard (`guard|sex0`; SEX 0 = men, PROTOCOL 0). Their worst clauses are -0.387 / -0.050 and -0.362 / +0.012.
  - DA_TWIN's version-0 value is -0.424947, which rounds to -0.42.
- Corrected wording:
  - RD 143: "Worst clause or guard, value / cap (V0 / V1)", with a note: "for DA_STD and DB_PTC_R1 the binding condition is the men's
    occupation Brier guard".
  - RD 140: "(worst value -0.42 / -0.10 of its cap)".
  - RD 280: "worst clause or guard".

**R43. C15's wording is too broad: "every release".**
- Quoted text: RD 291: "At the point, every release had lower linked-pair recovery than the twin."
- Record: TABLES.json `releases` pair AUC. The twin is 0.6506. DA_COMP (0.6635), DA_SBC_HI (0.6604), DA_SBC_LO (0.6603), DB_SBC_HI
  (0.6515), DB_SPD_HI (0.6512), DB_ZTILT (0.6512) and others are above it. Only the four candidate releases (DA_STD 0.6331, DB_PTC
  0.6342, DB_PTC_R1 0.6267, DA_IPW 0.6434) are below.
- Corrected wording: "At the point, each of the four candidate releases (DA_STD, DB_PTC, DB_PTC_R1, DA_IPW) had lower linked-pair
  recovery than the twin."

**R15. DA's revision.**
- Quoted text: RD 357-358: "DA's revision did worse than the registered STD (-0.0103, se 0.0107, not resolved). Its MODEL_FIT reader
  check (pair LOG_LR -0.022 vs the twin) did not carry over to MODEL_SELECT (+0.0072)."
- Records:
  - `claims/BESIDE_DDR1_DA_R1_BESIDE.json`: DA_IPW vs DA_STD -0.0103 ± 0.0107, `AGG_NOT_POSITIVE`. "Did worse" asserts what "not
    resolved" denies.
  - `da/FIT_R1_V0.json` / `da/FIT_R1_V1.json` `mf_reader_check`: DA_IPW minus TWIN pair LOG_LR AUC is -0.0216 / -0.0160. This is
    candidate-minus-twin, the opposite sign to the "+0.0072" benefit in the same sentence.
  - `da/FIT_V*.json`: DA_STD's own check was -0.0226 / -0.0235, so IPW was not better than STD on MODEL_FIT either.
- Corrected wording: "DA's revision was not better than the registered STD (point -0.0103, se 0.0107, not resolved). Its descriptive
  MODEL_FIT reader check (pair LOG_LR benefit over the twin +0.022 / +0.016 by version; DA_STD's was +0.023 / +0.024) did not carry
  over to MODEL_SELECT (+0.0072, point only)."

**R19. AB overstates what the challenger required.**
- Quoted text: AB 33-35: "It required stronger controls before any fit: a SEX-independent-target (SBC) path, a SEX-blind spending
  (SPEND) path, ZTILT (the same tilt driven only by what can be estimated from the twin's release) and data-rich readers."
- Records: `roles/challenger/REVIEW_DESIGNS.md` and the `PROTOCOL_AMENDMENT_1.md` table.
  - Required: ZTILT (finding 2, "ADOPTED (REQUIRED)"); data-rich readers (findings 3 and 18, required for DB); a fairer SBC fallback
    attribute (findings 6, 8 and 15).
  - The SPEND change (finding 5) was "ADOPTED in a stronger form" from an advisory finding.
  - The SBC and SPEND paths themselves were already PROTOCOL 8 controls.
- Corrected wording: "Before any fit it required ZTILT (...) as a saved DB control with a reading declared in advance, data-rich
  readers, and a fairer SEX-independent fallback target for both families' SBC paths. On its advice, adopted in a stronger form, the
  SPEND path became SEX-blind dithered quantization."

**R27. The joint-design controls are said to be "not built" without a reason.**
- Quoted text:
  - RD 299: "They were not built, and no joint-design advantage is claimed."
  - AB 87-88: "Separately designed recipients and both sequential orders were not built, and no joint-design advantage is claimed."
- Record: PROTOCOL 6.6 registers D5 as `NOT_ASSESSED_MISSING_CONTROLS` unless those controls exist. The 48-configuration allocation
  (PROTOCOL 8: DA 16, DB 16, revisions and replacement 16) has no place for them, and no card planned them.
- Corrected wording: append "They were not planned: the protocol registered D5 as not assessed without them, and the
  48-configuration budget (PROTOCOL 8) allocated none."

**R40. DA R2's failing condition is called a guard, but it is a MODEL_FIT budget clause.**
- Quoted text:
  - RD 374: "the condition failing at 1.0 was the women's occupation Brier guard, not an accuracy clause."
  - HO 91: "in R1 the binding condition was a guard, not an accuracy clause."
- Record: `da/FIT_R1_V1.json`. At lambda 1.0, the women's occupation Brier estimate + 2 SE is 0.00567, above the 0.0045 MODEL_FIT
  budget; the other five clauses pass. That is a MODEL_FIT budget clause, not the MODEL_SELECT guard. The substance holds.
- Corrected wording: "the women's occupation Brier budget clause (MODEL_FIT), not an accuracy clause".

**R39. The 66% figure is not precise (minor).**
- Quoted text: RD 370: "on MODEL_FIT the income release kept 66% of the twin's linear propensity dependence."
- Record: `da/R1_TRIGGER_B2.json`: ratio 0.6657 / 0.6612 of out-of-fold linear R^2 of the logit SEX propensity (0.0665 / 0.0648 vs
  0.0999 / 0.0980), threshold 0.5.
- Corrected wording: "about two thirds (R^2 ratio 0.666 / 0.661, threshold 0.5)". This is AB 26's wording.

**R31 (residue).**
- Quoted text: AB 51: "resolved but weak, because those controls are more recoverable than the twin."
- Record: VERIFIER_REPORT observation 2 gives version means only, with no resolution. RD 175 already says "at the point".
- Corrected wording: add "at the point".

**R12b. RULES.md 6 is stricter than HO 49 and CS 3.0.**
- Quoted text: HO 49: "Not "replicated", "confirmed" in the confirmation sense, "independent evidence", or "robust" standing alone."
- Records:
  - `claims/DB_R1/DECISIONS.json` outcome-class flag: `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`.
  - RULES.md 6 says that flag is "(near-limit; never called robust)".
  - The D2R label ROBUSTNESS_CONFIRMED is legitimate, but it concerns the recovery comparisons under classifier refitting, not the
    release.
- Corrected wording: "or "robust" for DB_PTC_R1 in any form other than the D2R label (robustness of its recovery comparisons to
  classifier refitting, descriptive): the release carries POINT_FEASIBLE_NOT_RESERVE_QUALIFIED, which RULES.md 6 says is never called
  robust."
- This wording comes from CS 3.0, so the statistical reviewer should agree to the change.

**R20. Publication stated as done.** This is the coordinator's publication lane; it is listed so the sentence is not missed.
- Quoted text: SH 84: "Code: `ddr/`, published under `publications/pcrl-dual-design-release-v1/code/ddr/`."
- Record: the branch `publication/pcrl-dual-design-release-v1` holds only the registration receipt c5a84c19d. Its `code/ddr/` lacks
  `da/ipw.py`, `db/ptc_r1.py`, `ctlkit.py`, `claims.py`, `report.py`, `richreaders.py` and other later modules.
- Corrected wording: "[PENDING: published at the publication commit under ...; the registration receipt c5a84c19d holds only the
  registration-time code]".
- SH 82 lists `EXPERIMENT_LEDGER.csv`, `CONFIG_LEDGER.csv`, `ACCESS_RECEIPT.json` and `RESOURCE_ACCOUNTING.json`. None of them is
  committed yet; they are in the same pending lane.

**R21. A forward reference.**
- Quoted text: SH 64-65: "`roles/statistical_reviewer/CLOSING_STATEMENT.md` and the claims review: the closing reviews, whose
  corrections the documents apply."
- Record: CS's corrections are applied (ae6626c44). This review's are not yet.
- Corrected wording: keep the sentence only once this review's corrections are applied, and name `roles/engineering_review/REVIEW_CLAIMS.md`
  (as HO 74 already does).

**R25. Deviation 9 is not cited (minor).**
- Quoted text:
  - RD 128: "- DB_PTC_R1_OCC: the occupation component of DB_PTC_R1."
  - The budget lines RD 379-380: "- 2 for DB_PTC_R1's ablation;" and "- 2 for DB_ZTILT_R1."
- Record: DEV 9 says card DB_PTC_R1 had said ablations, ZTILT and SHUF would not be refitted. Both additions were made under their own
  cards, ADAPTIVE_R1 and descriptive.
- Corrected wording: add "(added after card DB_PTC_R1 said they would not be refitted; deviation 9)".

**R32. The evidence index is incomplete (minor).**
- SH 77: the bank list omits `DDR1_REFS` (`grid_specs/DDR1_REFS.json`).
- SH 78: the Decisions row omits `claims/ZTILT_R1_READING.json`, which RD 190 cites.
- SH 77, "(written before their outputs)":
  - `DDR1_R1_ZTILT` was committed at 19:37:01Z, 27 s after its first grid unit started (19:36:34Z).
  - `DDR1_ROB_DB_R1` was committed in the same second as its first unit (19:22:17Z).
  - For these two, "written before" rests on file-write times (deviation 9; commit 87de6b6d1).
- Corrected wording: add both files and "(file-write times; two banks were committed at or just after their first grid unit
  started, deviation 9)".

**R34. QUICKSTART's run order is incomplete (minor).**
- QS step 8 (lines 47-55) omits `roles/coordinator/ztilt_r1_reading.py`, which writes `claims/ZTILT_R1_READING.json` (the declared
  reading RD cites). It also omits the beside run for `DDR1_R1_ZTILT`.
- QS step 4 (RICH_MAIN): `ddr/richreaders.py` changed at 1c7064bd6 (18:48Z), after DDR_RICH_MAIN_V1 ran at 18:34:22Z (VERIFIER_REPORT
  observation 4).
- Corrected wording: add both commands to step 8. Add to step 4: "(code changed after this run, commit 1c7064bd6; reproduction not
  checked)".

**R35. "Decisive controls" is imprecise (minor).**
- Quoted text: SH 80 and QS 78-79: "the decisive controls DA_SPD_HI, DB_SBC_R1_HI".
- Record: `deploy/DEPLOYMENT_CHECK.json` packages only these two. The other bracketing controls are decisive comparators too, and
  were not packaged.
- Corrected wording: "the bracketing control each release is not separated from (DA_SPD_HI, DB_SBC_R1_HI)".

**R44 (new; attribution). "Injective on its support" is stated as fact.**
- Quoted text:
  - AB 29-30: "The release writes the propensity into the vector and is injective on its support."
  - RD 322: "- PTC writes the propensity into the released vector and is injective on its support."
- Record: `cards/DB_PTC_v1.md` line 19 states it as a design property, conceded in `roles/designer_b/PROPOSAL.md` line 206 and in the
  challenger review. No computation in this campaign tested it. RD 108 attributes it correctly ("disclosed in the card before any
  outcome").
- Corrected wording: add "(design property, card DB_PTC_v1; not tested here)" at AB 30 and RD 322.

## 2. Claims stronger than the rules allow (current files)

| Watch item | Status in ae6626c44 |
|---|---|
| A partial D2 outcome presented as a result | Clean. The registered readings come first (C3). DB_PTC_R1's D2 is shown prefixed, the SPEND pass with its +0.0001 margin, and the SBC miss as a non-separation (RD 177). |
| An adaptive reading presented as registered | Clean. Prefixed labels (RD 94), ADAPTIVE_R1 rows, the extra DB_REG_D2R marked (RD 224, 230). |
| Point-only or unresolved differences called reductions | Residues only: R15 ("did worse ... not resolved") and R31 (AB 51). RD 335, "resolved 0.0239 reduction", is resolved and scoped ("registered cells, declared readers"). |
| "Protection" beyond the declared readers | Clean. Scope sentences in RD 12, RD 17, SH 99, AB 10 and HO 39. |
| Novelty | Clean. D6 `NOT_CLAIMED`; one idea family stated (C1). HO 59 "Both devices have published antecedents" rests on the designers' own citations; no separate prior-art review was run (RD 310 says so). Prefer RD 310's wording. |
| Joint design | Clean. "No contribution is resolved" (RD 305). R27: reason for not building the controls. |
| The reserve | Mostly clean. "Never called robust" at RD 156 and an untagged D2R row fixed. HO 49 residue (R12b). Advisory: DB_SBC_R1_HI (reserve 1.37 / 1.59) and DB_ZTILT_R1 (0.86 / 0.09; worst value 0.965 of its cap at the men's guard) are near-limit controls. C29 asks to add that DB_ZTILT_R1 is not reserve-qualified; DB_SBC_R1_HI is worth a word too. |

## 3. User-required report items (current files)

| Item | Present and separate? | Open |
|---|---|---|
| Task feasibility | Yes (RD "Task feasibility", AB Q1) | R36, R37 |
| Improvement beyond matched controls | Yes (RD D2, AB Q2) | none |
| Robustness to classifier refitting | Yes (RD D2R with the ZTILT_R1 refit and margins, AB Q3) | R12b |
| Competitiveness against strong references | Yes (RD D3, AB Q4) | none |
| The material criterion (0.020, apart from smaller gains) | Yes (RD D4, AB Q5, HO 57) | R43 (RD 291's range sentence) |
| Joint-design evidence | Yes (RD D5, AB Q6) | R27 |
| Novelty | Yes (RD D6, AB Q7) | R44 |
| Favourable and adverse evidence | Yes (RD) | R15 |
| Actual resources | [PENDING] (coordinator) | none |
| Untested questions | Yes (RD 382, AB 96, HO 84) | none |
| The public commit | [PENDING] (coordinator) | R20 |

## 4. What was not run, and why (current files)

Stated with a reason:
- DB R2 and DA R2 (not triggered; R40 corrects the wording of the DA R2 reason).
- The replacement family (48 of 48).
- Confirmation (not authorized; a fifth adaptive look on these people).
- A finer ZTILT grid (needed for a spend-matched point; the SBC grid would not help).
- Data-rich readers on DB_ZTILT_R1 (RD 207).
- An independent-reader check of DB_PTC_R1 (RD 284).
- Off-device custody (drive not attached; HO).

Still without a reason:
- **Joint-design controls (R27).**

Not mentioned (advisory):
- The failed reader-custody unit (deviation 10).
- The other deviations without a document line (1-4, 6, 11). One line under "Limits" would cover them: "Thirteen deviations are
  recorded in PROTOCOL_DEVIATIONS.md; none changes a decision label."
- The protocol's registered expectations (PROTOCOL 10), and card DA_STD_R1's failed prediction "lam* = 1.0 in both versions (p 0.6)".
  Only ZTILT_R1's predictions are reported (RD 353-354).

## 5. Table regeneration

The brief's command was run twice, label-free. Each time the tables were backed up to scratch with their SHA-256, regenerated,
compared, and restored, and `shasum -c` and `git status` were checked afterwards.

| Run | Against | Result |
|---|---|---|
| 19:46:31Z | Tables of 0678392a3 (utc 19:43:15Z) | `TABLES.md` identical; `TABLES.json` 0 differences apart from `utc` |
| 19:57:38Z | HEAD tables (utc 19:47:20Z, after deviation 12(a) rewrote the BESIDE records) | `TABLES.md` identical; `TABLES.json` 0 differences apart from `utc` |

- Both runs: 35 releases, 226 comparisons, 4 readings.
- The coordinator's regeneration at 19:47:20Z came after the first restore, at about 19:46:40Z, so nothing was overwritten.
- Only error found: the generator's "24 clauses + 4 guards" per-version text (R37).

## 6. Checks that passed

- **Decisions:** every D1-D6 label, outcome class and ADAPTIVE_R1 prefix; the campaign count and statement (`CAMPAIGN_SUMMARY.json`).
- **D2 table:** all 12 means and se values, version means, Q* names and labels; the 0.0008-nat bracket overspend; the 0.0001 SPEND
  margin; the 2-SE bound -0.0013.
- **Power:** PROTOCOL 6.2 and RULES.md 3; realized se_agg 0.0087-0.0178.
- **Beside comparisons:** every value, including ZTILT_R1 (+0.0128 ± 0.0081, refit +0.0066 ± 0.0095, 0.0169 vs 0.0099 nats, MODEL_FIT
  chosen spend 2.08x / 1.73x) and SHUF.
- **Data-rich readers:** the stated ranges.
- **D2R:** every value and label, `REFIT_GENUINELY_EFFECTIVE`, and the refit margins +0.0013 / +0.0007.
- **Ablations:** every mean and se.
- **D3 / D4:** fails 9 / 9 / 8 / 9 references; the reference pair AUCs; the 0.5600 shared occupation AUC; worst values 0.788-0.883;
  +0.0110 / -0.0058.
- **Utility:**
  - RD 145-149 table: every value matches TABLES.json, `EVAL_MS.json` and the ledger.
  - All 48 configurations pass. 32 + 6 + 6 + 2 + 2 = 48.
  - `RESERVE_ONE_SEED_ONLY` and `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED` are in the records.
- **Fit parameters:** lambda* 1.0 / 1.0, alpha 0.35, (0.35, 0.65), IPW 1.0 / 0.75.
- **Triggers:** the R1 trigger records; DB R2 not triggered; deviation 7.
- **Data roles:** 63 label reads, MODEL_FIT and MODEL_SELECT only; no CALIBRATION.
- **Order and timing:**
  - All six cards were committed before their fits.
  - Receipt c5a84c19d at 17:42:42Z, before the first label read at 18:06:27Z.
  - Amendment 1 at 18:06:14Z.
  - Start 16:37:33Z.
  - MODEL_SELECT is 8,001 people (PROTOCOL 1).
- **Code:** the rules unchanged since registration; 22 / 22 tests pass; all 15 `DEVGRID_CODE_LOCK.json` hashes match.
- **QUICKSTART:** every module, subcommand, flag and lane named exists.
- **Deployment:** 10 packages, all bitwise; every probe refused.
- **Novelty citations:** all traceable.
- **Custody:** same-device backup verified; off-device pending.

Not checked:
- QS's claim that the stage tests write nothing (not run).
- HO 3, "no message was sent".
- The CS headline beyond the numbers listed in section 7.

## 7. Consistency with the closing statement

- **Headline numbers.** Every number in CS section 2 that this review checked matches its record: D2 values, the SPEND margin, refit
  values, ZTILT_R1, D4, the per-look statement and 8,001 people.
- **Headline clarity.** For its author: "(8,001 people)" follows "cohort", but the 8,001 are the MODEL_SELECT people.
- **Withdrawn first-pass items.** Two first-pass items conflicted with CS and were withdrawn:
  - equality language per view, which CS 3.0 item 3 and C7 permit with the AUCs shown;
  - a finer SBC grid, which CS C9 shows cannot help.
- **CS wording that needs a fix.** Applying it introduced or kept three errors, all listed above: C29 (R37), C15 (R43), and the CS 3.0
  "standing alone" (R12b against RULES.md 6).
- **Agreed.** This review agrees with every other C-item applied in ae6626c44.

## 8. Resources

- **Wall time:** 19:45:53Z to 20:08Z.
- **CPU:**
  - The two table regenerations used 0.33 CPU-s (`getrusage`).
  - The reviewer's other label-free record reads used about 7 CPU-s.
  - The three forked sub-agents reported about 3, 5 and 5 CPU-s.
  - Estimated total: about 20 CPU-s (under 0.01 CPU-h).
- **Agents:** the reviewer plus three forks.
- **Not used:** no queue unit, no label read, no real-data computation, no paid compute, and no campaign file changed except this
  report.
