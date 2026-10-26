# Statistical reviewer, claims review wave 2 (verbatim; delivered to the coordinator at about 23:08Z; harness indentation removed)

# RFA1 statistical reviewer — claims review, wave 2

Written 2026-10-25 22:59–23:12Z. CPU about 2 CPU-s this round, about 41 CPU-s across all waves.

**What I read** (records and code only):
- `claims/RFA_v1_rebuilt/` and `claims/RFA_R1/`, each with `DECISIONS.json` and `FALSIFIER_DETAIL.json`;
- `FAMILY_CHANGE_vs_RF_FULL_corrected.json`;
- `claims/RFA_v1_D2R/D2R.json`;
- `PROTOCOL.md` registered expectations (lines 159–162);
- the drafts `RESEARCH_DECISION.md`, `START_HERE.md`, `TABLES.md` and `ENDPOINTS.json`.

I read nothing under `.private/`, no grid file and no label.

**Verified labels.** Every label in your list follows from the records under the frozen rules (hash 4e03270c…).
- **v1 rebuilt:**
  - the learning-curve half-budget grid validity is now its own: `READER_SEEDS_EFFECTIVE`, 1,536 checks, strict reading effective;
  - the falsifier detail is saved;
  - the labels are unchanged from wave 1.
- **R1:**
  - ADAPTIVE_R1: card committed 22:23:27Z, after the 21:59:53Z read; learning-curve validity effective (1,728 checks).
  - D1 TASK_FEASIBLE, with 27 of 28 conditions confidence-resolved; only the version-1 men's guard is unresolved (0.00457 of 0.005).
  - D2 NOT_SEPARATED: START +0.0098 (se 0.0054) and CONT +0.0090 (0.0051), both point only; STOPFIT −0.0014 (0.0041); ALTCRIT −0.0030 (0.0045); RAWJ −0.0046 (0.0052).
  - D3 NOT_SUPERIOR: passes W, TR100, TT100. D4 NOT_MET.
  - LC limitation only against H_LJ.
  - F2, F3 and F4 triggered.
  - Outcome BOUNDED_NEGATIVE_EXPERIMENT, partial NOT_SEPARATED.
- **FC_R1:** `ADAPTIVE_R1__FAMILY_CHANGE_NOT_POSITIVE`, −0.0029 (se 0.0046; version means −0.0016 / −0.0042). Spend state `R1_SPENDS_MORE_UNRESOLVED` (+0.00019 ± 0.00013). Men's occupation Brier is resolved higher (+0.00041 ± 0.00008). The income view is an exact alias; occupation reads RECIPIENT_AGG_EDGE.
- **v1 D2R:** ROBUSTNESS_NOT_CONFIRMED. The refit is genuinely effective (6,912 checks).

**Recipient-gate view: your statement is confirmed from the records.** I checked it from the `TABLES.md` version means.
- CAND's occupation view is 0.5971 / 0.5969. B2TWIN, H_LJ, RETIRED_B3CTL, SUCC and TWJ share an occupation output at 0.5665 / 0.5682. So r = +0.0306 / +0.0287 > 0.010, and the gate fails on **occupation**.
- On the registered cells income passes (r ≤ 0.010). The largest income r is RETIRED_B3CTL in version 0, +0.0077.
- On the robustness cells, RETIRED_B3CTL's income gate also fails (D2R record). Say so where D2R is discussed.

## REQUIRED

**R1. The second-split result must be reported in full (both documents).**
- D2R confirmed nothing for v1, including the only registered pass. Against NEURAL_RAW the robustness cells read +0.0120 (se 0.0075), which is not resolved; all cells together are point only. Say explicitly: "the registered benefit over NEURAL_RAW was not confirmed on the second split".
- Carry the whole D2R caveat:
  - sign changes against RF_STOP_HI (+0.0028 → −0.0009), RF_ALTC_HI (+0.0005 → −0.0023) **and SUCC** (+0.0037 → −0.0036);
  - robustness recipient failures against RF_RAWJ_HI (income), H_LJ (occupation), RETIRED_B3CTL (both views) and SUCC (occupation).
- The drafts omit the SUCC sign change, all four failures and the non-confirmation against START.
- `START_HERE` says "the small point edges reversed on the second reader split". Replace it with "the point edges over STOPFIT and ALTCRIT changed sign, and the benefit over NEURAL_RAW was not confirmed".

**R2. Bring the decision records in line with D2R.**
- `RFA_v1_rebuilt` still reads D2R NOT_RUN, and the outcome flags carried into `ENDPOINTS` lack `D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE` and `D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE`.
- Rebuild with `rec['d2r']` set to the D2R record (descriptive; the class is unchanged), or add an explicit note that the D2R record supersedes that field.
- Do the same for R1 when `claims/RFA_R1_D2R/` arrives. Until then, both documents mark R1's second split as pending.

**R3. Decision paragraph (RESEARCH_DECISION).**
- "It did not lower recovery … beyond any mechanism-specific control at comparable usefulness" must become "its recovery by the declared readers was not separated from any mechanism-specific control (the paths bracketed at matched MODEL_SELECT total-NLL spend)". "Not separated" is not "no effect".
- "At comparable usefulness" is inaccurate:
  - the RAW-J bracket points fail G1 in version 1 (RF_RAWJ_HI max value/cap 1.059; RF1_RAWJ_HI 1.137, `TABLES.md`);
  - CONT is not spend-matched.
- Disclose that the ALT comparison is against a control that is not itself usefulness-feasible in version 1.

**R4. The R1 learning-curve sentence is inaccurate.** "Only the benefit against H_LJ shrank at the point" is wrong. Write instead:
- "R1's learning-curve limitation is against H_LJ only (−0.0067, se 0.0116, point only)";
- "the point estimates against RF1_STOP_HI (−0.0014) and RF1_ALTC_HI (−0.0021) also fell, but from non-positive full-budget benefits, so they are not limitations by the rule".

**R5. Registered expectations: report all of them, judged by their own definitions.**
- **PROTOCOL's "the fit path changes the release measurably"** is registered as "CAND and STOPFIT outputs differ beyond rounding". Judge it on that: the occupation view is 0.5971 / 0.5969 against RF_STOP_HI's 0.5986 / 0.5989, and the wires differ. Do not judge it on F0.
- **Add every prediction in card RF_RFIT_v1 §6:**

  | prediction | probability | outcome |
  |---|---|---|
  | F0 active | 0.8 | yes |
  | k* between S20 and S200 | 0.7 | yes (S20 / S70; S20 is the boundary) |
  | CAND G1-feasible | 0.7 | yes |
  | point benefit over START in [+0.003, +0.020] | 0.6 | yes (+0.0127) |
  | resolved over START | 0.3 | yes on the registered cells, not confirmed on the second split |
  | tie with STOPFIT | 0.75 | yes |
  | resolved over STOPFIT | 0.10 | no |
  | resolved over ALT | 0.10 | no |
  | D3 | 0.02 | no |
  | D4 | 0.01 | no |
  | F4 given a positive benefit | 0.5 | yes (point rule) |

- **Add all nine R1 predictions.** The draft lists three. The missing six:

  | prediction | probability | outcome |
  |---|---|---|
  | RF1_FULL G1-feasible | 0.7 | yes |
  | FULL_R1 resolved over STOPFIT_R1 | 0.10 | no |
  | D2 met | 0.04 | no |
  | D3 | 0.03 | no |
  | D4 | 0.01 | no |
  | the learning-curve limitation again | 0.6 | yes (H_LJ, point) |

- Reporting only some registered predictions is not acceptable.

**R6. START_HERE answer 8 overstates the evidence.**
- "The fit path does not help" must become "this campaign gives no evidence that the fit path helps: the candidate was not separated from the stopped-fit control (development bound about 0.012 on this grid) or from a lagging critic of the same family".
- "Reproduces the gain" asserts equality; replace it with "was not separated from".
- Keep "further learning-aware penalties … are not supported" only as "are not supported by this evidence".

**R7. F3 wording.** F3 triggered on tiny descriptive margins with no uncertainty:
- v1: LOG_LR gain over RF_STOP_HI +0.0007 against a tree/table mean of −0.0002;
- R1: +0.0003 against −0.0005.

LOOKUP's gain is a structural 0.0: on continuous outputs every SCORE wire is unseen by the table, and that zero enters the tree/table mean. Say both wherever F3 is cited. The R1 card's per-recipe falls are measured against START, which is a different contrast from F3's.

## ADVISORY

- **Neutral wording.**
  - "Not even resolved against NEURAL_RAW" → "not resolved".
  - "Changed the release in a small but systematic way" → "F0 found the fit-path term active on MODEL_FIT (orthogonal share 0.10–0.11)".
  - "The fit path is real but small" → the same.
- **v1 D3 passes.** "Resolved below NEURAL_RAW, W, TR100 and TT100 with both recipient gates; resolved below TWJ, but its occupation gate fails."
- **START_HERE answer 3.** Add that v1's version-1 occupation NLL vs Ucal is also unresolved (0.00737 ± 0.00133 of 0.010), not only the men's guard.
- **Version 0 barely moved.** k* = S20; MODEL_FIT out-of-fold spend 0.0008 nats; the STOPFIT and ALTCRIT HI points sit at the same S20. Per-version benefits over START are +0.0101 / +0.0154.
- **"Usefulness binds early … 20–70 steps"** spans both versions (S20 / S70). Name the MODEL_FIT budget component that stopped k* in each version, from the fit records.
- **The successor sentence** ("a question about the output representation, not about the critic") is an interpretation. Label it as one.
- **F6.** It was only weakly assessable in version 0 (2–3 logged steps, 1 monitor, because k* came early).
- **F5.** The coordinator code implements only the "mostly negative orientation" disjunct. The other disjunct is not met: the START comparison is resolved.
- **D2R coverage.** B2TWIN, TWJ, W, TR100 and TT100 were not in the robustness bank (they read NOT_RUN). Disclose it.
- **ENDPOINTS.**
  - Add FC_R1 from the corrected file (deviation 9): its label, mean, se, version means, spend state and the men's-Brier result.
  - Add R1's D2R when it arrives.
- **Campaign summary.** Quote the rules' `campaign_summary` statement verbatim, with K = 2 and the per-look false-pass sentence.
- **Sources for the remaining numbers.** Cite where these come from:
  - "per-recipe means within 0.0003 of v1's";
  - "STOPFIT and ALTCRIT spent 0.0002–0.0006 nats more total NLL at the same checkpoint";
  - the references' men's guard "0.883 / 0.788 of cap".
