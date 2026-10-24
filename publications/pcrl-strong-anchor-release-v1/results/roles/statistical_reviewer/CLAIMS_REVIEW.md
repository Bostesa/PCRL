# Claims review of the SAR1 closeout documents (statistical reviewer; 2026-10-24, from 07:00:45Z, written 07:08:11Z, `date -u`)

Label-free: I read only saved records and documents (no queue unit, no git, no edit of any document or record). Reviewed at
commit 6cab898ec (as stated by the coordinator): `RESEARCH_DECISION.md` (fully), `START_HERE.md` (fully), `HANDOFF.md` (fully),
`QUICKSTART.md` (fully), `ENDPOINTS.json` (spot checks), against `claims/*/DECISIONS.json`, `claims/*_D2R/D2R.json`,
`claims/CAMPAIGN_SUMMARY.json`, `fa/`, `fb/`, `fs/` fit records, the cards, `PROTOCOL.md` 9, `PROTOCOL_DEVIATIONS.md` (now 12 items),
the verifier's two hand-backs, `roles/investigator_b/HANDBACK_FS_STOP.md` and the coordinator log, and my RULES.md / reports.
CPU used: about 4 CPU-s (small read-only extraction scripts).

## Required corrections

1. **Registration status of the D2 looks is missing** (RULES.md 7; PROTOCOL 13: a campaign-level statement needs the number of D2
   looks and their registration status).
   - RESEARCH_DECISION.md, Decision: "Across six readings D2 was met 0 times in 2 looks, D3 0 times and D4 0 times".
   - START_HERE.md, Result section: the same information is missing altogether.
   - Record: `claims/CAMPAIGN_SUMMARY.json`: "D2 was met in 0 of 0 registered family readings and in 0 of 2 non-registered readings
     (2 D2 looks in total)". FA_REG and FB_REG are NO_FEASIBLE_STRENGTH.
   - Fix (both files): "Both D2 looks were non-registered (adaptive) revisions, FB_R1 and FA_R2. The registered readings FA_REG and
     FB_REG found no feasible strength, so no registered reading reached D2. No multiplicity correction is applied."
   - Also quote the non-registered readings' labels with their tag at first mention, or mark them "(adaptive)": the three-outcome
     bullets and the readings table's Class column. Example: `ADAPTIVE_R2__PARTIAL_CONT_ONLY`, `ADAPTIVE_R1__FEASIBLE_NO_RELIABLE_GAIN`.
2. **The FB_R1 ALT sign change is not in the headline** (RULES.md 8; your required statement).
   - RESEARCH_DECISION.md, Decision section: no mention. It currently appears only in the D2R section, in "What prevents" 3 and in
     plain answer 7.
   - Record: `claims/FB_R1_D2R/D2R.json`: sign_changes ["FB_R1_ALT_HI"]; headline_caveat "the pair benefit changes sign on the
     robustness cells against FB_R1_ALT_HI: carried into the headline".
   - Fix: add to the Decision paragraph: "FB_PFO's edge over NEURAL_RAW's own recipe continued to matched spend (ALT) changes sign on
     the robustness split (+0.0012 registered, −0.0012 robustness, se 0.0070; D2R ROBUSTNESS_NOT_CONFIRMED)." HANDOFF.md State
     should say the same in one clause.
3. **Equality wording drawn from NOT_SEPARATED results** (RULES.md 4.1: NOT_SEPARATED is the default reading, never "equal"; the
   spend is matched by the bracket rule, not equal).
   - Quoted text:
     - RESEARCH_DECISION.md: "NEURAL_RAW's own joint recipe, continued to matched spend (ALT), reproduces it".
     - RESEARCH_DECISION.md: "Its effect equals NEURAL_RAW's own joint recipe run further".
     - RESEARCH_DECISION.md: "NEURAL_RAW's own continued recipe matches it at equal spend".
     - RESEARCH_DECISION.md, plain answer 1: "FB_PFO turned out to be reproduced by NEURAL_RAW's own recipe run further".
     - START_HERE.md: "NEURAL_RAW's own recipe, continued to equal spend, does nearly as well".
     - HANDOFF.md: "because its reading equals NEURAL_RAW's own operation run further".
   - Record (`claims/FB_R1/DECISIONS.json`, D2.comparisons.ALT_FRONTIER):
     - NOT_SEPARATED_FROM_TASK_SPENDING, benefit +0.0012 (se 0.0074) at Q* = FB_R1_ALT_HI.
     - residual_spend_mismatch.at_bracket = −0.00075 (se 0.00055): the bracket spends about 0.0008 nats more pooled total NLL than
       FB_PFO.
     - recipient_vs_bracket.income = RECIPIENT_AGG_POINT_FAILED: the recipient gate against ALT also fails.
   - Fix: "is not separated from NEURAL_RAW's own recipe continued to matched spend. Against the ALT bracket the benefit is +0.0012
     (se 0.0074); the bracket spends about 0.0008 nats more, and FB_PFO's income view also fails the recipient gate there."
4. **FB's stop is described as rule-like.** In HANDOFF.md and RESEARCH_DECISION.md ("so FB stopped after revision 1") it is a
   discretionary decision, not a PROTOCOL 9 stop rule.
   - PROTOCOL 9 stops a family when its revisions are used, when it cannot change the output, or when the budget forbids its
     controls. None applied: FB had one revision left.
   - Record: PROTOCOL.md 9; the coordinator log ("FB stops after R1 ...").
   - Fix: "FB was stopped after revision 1 by decision, with one revision unused, because its gain was not separated from ALT."
5. **The CONT passes are not called weak-twin attributions for FB, nor in D2R** (your required statement; PROTOCOL 8).
   - Quoted text:
     - RESEARCH_DECISION.md: "Against NEURAL_RAW +0.0125 (0.0056) and its continuation +0.0143 (0.0055), both resolved."
     - RESEARCH_DECISION.md, D2R: "CONT confirmed, +0.0374 (0.0156)." and "CONT confirmed, +0.0140 (0.0064)."
     - RESEARCH_DECISION.md, plain answer 7: "only the comparisons against the zero-protection continuations were confirmed".
   - "its continuation" is also ambiguous with ALT.
   - Record: FB_R1's CONT is FB_PFO_TASK (AGG_POSITIVE). D2R: FB_PFO_TASK and FA_TFRO_R1_CONT are ROBUSTNESS_CONFIRMED.
   - Fix: name it "the equally exposed zero-protection continuation FB_PFO_TASK (CONT)". Add "(an attribution against the weak
     zero-protection twin, not a component gain)" in the FB bullet, in both D2R bullets and in plain answer 7.
6. **Wrong statement about FA's income output.** RESEARCH_DECISION.md: "All three keep SUCC's income output bitwise in revisions 1
   and 2."
   - Record: card FA_TFR_v1. Its operation sets income = softmax(−(d_S + s·r_I)/2, ...), and it says "Both outputs move under one
     joint objective". Verifier wave 2: "the candidate's income is SUCC's bitwise" (FA_R2).
   - Fix: "FA_TFR (revision 0) moved both outputs. Revisions 1 and 2 (FA_TFRO_R1, FA_TFRB_R2) keep SUCC's income output bitwise."
7. **FA_R2's CTRL bracket in version 1 is SUCC itself, which is not disclosed.**
   - Quoted text:
     - RESEARCH_DECISION.md: "against the untargeted control at its bracket (FA_R2_CTRL_HI) +0.0066 (0.0140)".
     - RESEARCH_DECISION.md, plain answer 4: "+0.0066 (0.0140) over the untargeted control".
     - START_HERE.md: "over the untargeted control +0.0066 (0.0140)".
   - Record:
     - Coordinator log: "version 1 rho* = 1.0 (CTRL LO s 0.25, HI = the origin)".
     - Verifier wave 2: "The v1 CTRL_HI is the frozen SUCC bitwise".
     - `claims/FA_R2/DECISIONS.json` D2.targeting: the version-1 means of FA_R2_CTRL_HI and SUCC are identical (0.017914).
   - So in version 1 the CTRL comparison repeats the START comparison (my report, advisory 13).
   - Fix: add "(in version 1 the bracketing CTRL point is SUCC itself, so there the CTRL comparison repeats the START comparison;
     only version 0 is compared with a distinct untargeted control)".
8. **Wrong integrity note.** RESEARCH_DECISION.md: "Every card, limit, comparison definition and access boundary was committed before
   the first label computation".
   - Record (`claims/*/DECISIONS.json` registration_status.card_first_commit): four cards came after the first label read (05:11:51Z
     per HANDOFF) and after the first MODEL_SELECT read (05:13:09Z):

     | Card | First commit |
     |---|---|
     | FB_PFO_R1 | 05:32:12Z |
     | FA_TFRO_R1 | 05:37:57Z |
     | FA_TFRB_R2 | 06:08:14Z |
     | FS_CPI_v1 | 06:12:17Z |

   - Fix: "The registered cards FA_TFR_v1 and FB_PFD_v1 (05:10:48Z), the limits, the comparison definitions and the access boundary
     were committed before the first label computation. Each later card was committed before its own first fit, after earlier
     MODEL_SELECT outcomes, hence ADAPTIVE."
9. **Deviation count is stale.**
   - Quoted text: RESEARCH_DECISION.md "Eleven protocol deviations"; HANDOFF.md "Deviations. Eleven"; START_HERE.md "(11 items,
     append-only)".
   - Record: `PROTOCOL_DEVIATIONS.md` now has 12 items. Item 12, verifier-file drift, was recorded at 07:03Z, after the documents
     were written.
   - Fix: "twelve" / "12 items", with item 12 noted (no label effect).
10. **The FS stop memo is overstated.** RESEARCH_DECISION.md: "with SUCC's own beta transplant reproducing most of it at matched
    spend on MODEL_FIT".
    - Record: `roles/investigator_b/HANDBACK_FS_STOP.md`: a predicted 0.005 (80% interval 0.000–0.011), and "the part of it that
      SUCC's own operation (ALT) does not reproduce is about 0.001–0.004", so "most" holds only at the low end. P(≥ 0.010) is about
      0.2.
    - Fix: quote the memo: "the part not reproduced by SUCC's own operation (ALT) was predicted at about 0.001–0.004 on the fitting
      data, so D2 could not be met".
11. **ENDPOINTS.json is internally inconsistent.** readings.FB_R1 and readings.FA_R2 `three_outcomes_statement` say "(D2R NOT_RUN)",
    but their `D2R` fields say ROBUSTNESS_NOT_CONFIRMED, and FB_R1 also has the sign change.
    - Record: the write-once DECISIONS.json statements predate the D2R records (FB_R1 05:45:38Z vs D2R 06:19:11Z; FA_R2 06:21:28Z vs
      06:38:00Z).
    - Fix: leave DECISIONS.json unchanged. In the ENDPOINTS script, append the D2R label and headline_caveat to the statement. In
      RESEARCH_DECISION, say that the D2R labels come from `claims/*_D2R/D2R.json`.
12. **START_HERE.md does not separate point eligibility from the confidence conjunction and the reserve** (RULES.md 2; your required
    statement).
    - Quoted text: "It met all 28 conditions at the point" (FA) and "met all 28 conditions at the point" (FB).
    - Record: for both FA_R2 and FB_R1, D1.confidence.cand = CONFIDENCE_UNRESOLVED (27 of 28; v1|guard|sex0), and the reserve is not
      qualified.
    - Fix: add "(point eligibility only: 27 of 28 conditions pass the confidence conjunction, with version 1's men's guard
      unresolved; not reserve-qualified)".

## Recommended changes

1. "Feasible" means G1 at the point only.
   - In the readings table, "Feasible; ..." → "Point-feasible (G1); ...".
   - "in both feasible readings" → "in both point-eligible readings".
   - HANDOFF and QUICKSTART: "surviving releases" → "point-eligible releases".
2. Plain answer 3: "Both are confidence-resolved on 27 of 28 conditions" → "27 of 28 conditions pass the confidence conjunction
   (release label CONFIDENCE_UNRESOLVED)". RULES.md 2 reserves "confidence-resolved" for CONFIDENCE_ALL_PASS.
3. D3 detail:
   - FA_TFRB_R2's recipient gate also fails against H_LJ, B2TWIN and SUCC. Among the point-only comparisons, only NEURAL_RAW's gate
     passes.
   - FB_PFO is AGG_POSITIVE against TR100, TT100, TWJ and W but fails their recipient gates, so "passes only NEURAL_RAW" is partly a
     gate result.
4. FB_R1 controls: FB_R1 saved only the upper CTRL and ALT points (deviation 7; flags CONTROL_POINT_RULE_NOT_FOLLOWED_CTRL / _ALT),
   so (c3) checks only the origin. Also add the ALT income-gate failure from correction 3.
5. UM: the resolved improvement is marginal. The upper bound is −0.0002, and version 1's difference is −0.0004 against version 0's
   −0.0053. Say "just resolved".
6. "with the income output frozen" (FB_PFO): add "(equal to NEURAL_RAW's within 7e-8, not bitwise; deviation 10)".
7. Plain answer 7: "Every label and number checked by the verifier's own code reproduced" → add "(253 of 256 checks PASS; the 3
   FAILs concern two missing plan hashes and one card's bitwise claim; the CALIBRATION spends were re-applied, not recomputed)".
8. HANDOFF: "The best evidence is FA_TFRB_R2's version 1" → call it a post hoc single-version observation. Version 0 is worse than
   SUCC, the pooled benefit is unresolved, and version 1's men's guard is unresolved in the confidence conjunction.
9. FS wording:
   - Use the card's name: "conditional pair-increment residual on SUCC's income, occupation frozen, trained against FB_PFD's
     closed-form whitened quadratic best response".
   - START_HERE "It had no feasible strength" → "no feasible strength in version 0 (version 1 had λ 0.05), so no candidate".
   - Plain answer 3 per version: FB_PFD v0 had k* = E1 and FS_CPI v1 had λ 0.05; a reading needs both versions.
10. FA_TFR critics: the card names `acs_replication.leafcorr.Critics` (linear / MLP32 / MLP64 on the income, occupation and pair
    wires). Prefer "online SEX critics (acs_replication.leafcorr)" unless the card itself calls them leaf-correlation critics.
11. ENDPOINTS.json `campaign_summary`: include the full statement from CAMPAIGN_SUMMARY.json (registration status and multiplicity).

## Verified against the records (no change needed)

- **FA_TFRB_R2 (FA_R2) D2:**
  - START +0.0066 (0.0141), point only, occupation gate failing.
  - CTRL_HI +0.0066 (0.0140), not separated.
  - CONT +0.0432 (0.0183).
- **FA_TFRB_R2 D3 and D4:**
  - D3 passes W, TR100 and TT100.
  - NEURAL_RAW +0.0241 (0.0191), H_LJ +0.0208 (0.0145), B2TWIN +0.0119 (0.0143), RETIRED_B3CTL −0.0014 (0.0158), TWJ +0.0313 (0.0144).
  - D4 MATERIAL_0020_POINT_ONLY; version-0 means +0.0078 / +0.0090.
- **FA_TFRB_R2 UM and utility:**
  - UM pooled −0.0029 (0.0013); occupation view +0.0252 / −0.0141.
  - Worst value over cap 0.012 / 0.698.
  - Men's guard −0.0008 / +0.0035; SUCC +0.0044 / +0.0039.
  - Confidence 27 of 28; SUCC 26 of 28.
- **FB_PFO (FB_R1) D2:**
  - START +0.0125 (0.0056); CONT +0.0143 (0.0055).
  - ALT +0.0012 (0.0074).
  - CTRL NO_BRACKET.
- **FB_PFO D3 and utility:**
  - D3 passes NEURAL_RAW only.
  - H_LJ +0.0091 (0.0096), SUCC −0.0050 (0.0093), RETIRED_B3CTL −0.0130 (0.0106).
  - D4 NOT_MET.
  - Worst value over cap 0.601 / 0.785; confidence 27 of 28.
- **D2R numbers (both banks):**
  - FA_R2: SUCC +0.0034 (0.0139), CTRL_HI +0.0040 (0.0134), CONT +0.0374 (0.0156), NEURAL_RAW +0.0183 (0.0177), H_LJ +0.0158 (0.0135).
  - FB_R1: NEURAL_RAW +0.0119 (0.0067), CONT +0.0140 (0.0064), ALT +0.0012 / −0.0012 (0.0070).
  - Refits genuinely effective.
- **Readings table and recovery:**
  - The recovery table equals ENDPOINTS.json.
  - Reading strengths: FB S50 / S50; FA_R1 rho 1.0 / 0.6; FA_R2 rho 0.6 / 1.0.
  - FA_R1's guard +0.0099 (0.0020); in the confidence conjunction it is a VIOLATION.
  - FA_REG rho 0.3 failures; FB_REG bound 0.00383 > 0.0025; FS_CPI bound 0.0052 > 0.0040.
- **FA_TFRB_R2 selection:** between-fold SE 0.0034 / 0.0025 vs household 0.0014 / 0.0013.
- **Verifier and log facts:** 256 / 253 / 3 checks; refits 6 / 6 / 2; receipt verified at 05:11:39Z.
- **Conventions:** the sign convention (comparator − candidate) is applied consistently in every quoted benefit, and the
  adaptive-search and reused-household caveats are present in RESEARCH_DECISION, START_HERE and ENDPOINTS.

## Not verified (outside what I could check label-free and without git in the time)

- Plan commit times against each reading's first MODEL_SELECT computation.
- CPU-hours, backup size and custody counts in HANDOFF.
- The specification count (28 = 8 + 3 + 17), which matches the coordinator log only.
- FA_TFRO_R1's guard formula.
- TABLES.md beyond the quoted rows.
