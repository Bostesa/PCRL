# Statistical reviewer hand-back: claims review (saved verbatim by the coordinator at 07:13Z on 2026-10-24, `date -u`)

The claims review is done: the documents need 12 required corrections and 11 recommended changes. The underlying numbers check out against the records. The required fixes concern missing required statements, wording that overstates what the records show, and three stale or wrong facts.

The review is written to `results/pcrl_strong_anchor_release_v1/roles/statistical_reviewer/CLAIMS_REVIEW.md` (SHA-256 c3ae2acc6ff7842bfc43f7bc2de3c37b5538dc6eb260cda70fd30733afaf6711), started 07:00:45Z and finished 07:09:17Z. I worked label-free: saved records only, no queue units, no git, no edits to any document or record. **CPU used: about 4 CPU-s** (read-only extraction scripts). I read RESEARCH_DECISION.md, START_HERE.md, HANDOFF.md and QUICKSTART.md in full and spot-checked ENDPOINTS.json.

## Required corrections
1. **Registration status of the D2 looks is missing** (RULES.md 7, PROTOCOL 13).
   - Text: RESEARCH_DECISION Decision, "Across six readings D2 was met 0 times in 2 looks, D3 0 times and D4 0 times". START_HERE's Result section has nothing on this at all.
   - Record: `claims/CAMPAIGN_SUMMARY.json` says "D2 was met in 0 of 0 registered family readings and in 0 of 2 non-registered readings"; FA_REG and FB_REG are NO_FEASIBLE_STRENGTH.
   - Fix (both files): "Both D2 looks were non-registered (adaptive) revisions, FB_R1 and FA_R2; the registered readings FA_REG and FB_REG found no feasible strength; no multiplicity correction is applied." Also quote adaptive readings' labels with their tag, or mark them "(adaptive)", in the outcome bullets and the table's Class column.
2. **The FB_R1 ALT sign change is not in the headline** (RULES.md 8).
   - Text: the RESEARCH_DECISION Decision section does not mention it.
   - Record: `claims/FB_R1_D2R/D2R.json`, sign_changes ["FB_R1_ALT_HI"] with a headline_caveat.
   - Fix: add "FB_PFO's edge over NEURAL_RAW's own recipe continued to matched spend (ALT) changes sign on the robustness split (+0.0012 registered, −0.0012 robustness, se 0.0070; D2R ROBUSTNESS_NOT_CONFIRMED)". Add one clause to HANDOFF State too.
3. **Equality wording drawn from a NOT_SEPARATED result** (RULES.md 4.1).
   - Text, RESEARCH_DECISION:
     - "(ALT), reproduces it"
     - "Its effect equals NEURAL_RAW's own joint recipe run further"
     - "matches it at equal spend"
     - plain answer 1: "FB_PFO turned out to be reproduced by NEURAL_RAW's own recipe"
   - Text, START_HERE: "continued to equal spend, does nearly as well".
   - Text, HANDOFF: "because its reading equals NEURAL_RAW's own operation run further".
   - Record: `claims/FB_R1/DECISIONS.json`, ALT_FRONTIER:
     - NOT_SEPARATED_FROM_TASK_SPENDING, +0.0012 (0.0074) at FB_R1_ALT_HI.
     - Residual spend mismatch −0.00075 (se 0.00055): the bracket spends about 0.0008 nats more.
     - The recipient gate fails at the bracket (income view, RECIPIENT_AGG_POINT_FAILED).
   - Fix: "is not separated from NEURAL_RAW's own recipe continued to matched spend (+0.0012, se 0.0074; the bracket spends about 0.0008 nats more; FB_PFO's income view also fails the recipient gate there)".
4. **FB's stop is described as rule-like.**
   - Text: HANDOFF, and RESEARCH_DECISION "so FB stopped after revision 1".
   - Problem: none of PROTOCOL 9's stop rules applied (revisions used, cannot change the output, budget forbids controls); FB had one revision left.
   - Record: PROTOCOL.md 9 and the coordinator log.
   - Fix: "stopped after revision 1 by decision, one revision unused, because its gain was not separated from ALT".
5. **The CONT passes are not called weak-twin attributions for FB or in D2R.**
   - Text, RESEARCH_DECISION: "and its continuation +0.0143 (0.0055), both resolved"; D2R "CONT confirmed, +0.0374 (0.0156)" and "CONT confirmed, +0.0140 (0.0064)"; plain answer 7 "only the comparisons against the zero-protection continuations were confirmed". "Its continuation" can also be confused with ALT.
   - Fix: name it "the equally exposed zero-protection continuation FB_PFO_TASK (CONT)" and add "(an attribution against the weak zero-protection twin, not a component gain)" in all four places.
6. **Wrong statement about FA's income output.**
   - Text, RESEARCH_DECISION: "All three keep SUCC's income output bitwise in revisions 1 and 2."
   - Record: card FA_TFR_v1 moves income, "income = softmax(−(d_S + s·r_I)/2, …)", and says "Both outputs move under one joint objective".
   - Fix: "FA_TFR (revision 0) moved both outputs; revisions 1 and 2 keep SUCC's income output bitwise."
7. **FA_R2's CTRL bracket in version 1 is SUCC itself, and this is not disclosed.**
   - Text: RESEARCH_DECISION "(FA_R2_CTRL_HI) +0.0066 (0.0140)", plain answer 4, and START_HERE "over the untargeted control +0.0066 (0.0140)".
   - Record:
     - The coordinator log says "version 1 rho* = 1.0 (CTRL LO s 0.25, HI = the origin)".
     - Verifier wave 2 says "The v1 CTRL_HI is the frozen SUCC bitwise".
     - In the FA_R2 targeting records, the version-1 means of CTRL_HI and SUCC are identical (0.017914).
   - Fix: add "in version 1 the bracketing CTRL point is SUCC itself, so there the CTRL comparison repeats the START comparison; only version 0 is compared with a distinct untargeted control".
8. **Wrong integrity note.**
   - Text, RESEARCH_DECISION: "Every card, limit, comparison definition and access boundary was committed before the first label computation".
   - Record: from DECISIONS.json card_first_commit, four cards came after the first label read (05:11:51Z) and the first MODEL_SELECT read (05:13:09Z): FB_PFO_R1 05:32:12Z, FA_TFRO_R1 05:37:57Z, FA_TFRB_R2 06:08:14Z, FS_CPI_v1 06:12:17Z.
   - Fix: "the registered cards (05:10:48Z), limits, definitions and access boundary came before the first label computation; each later card came before its own first fit, after earlier MODEL_SELECT outcomes (hence ADAPTIVE)".
9. **The deviation count is stale.**
   - Text: "Eleven" in RESEARCH_DECISION and HANDOFF; "(11 items)" in START_HERE.
   - Record: PROTOCOL_DEVIATIONS.md now has 12 items; item 12 was recorded at 07:03Z.
   - Fix: change to twelve and note item 12 (no label effect).
10. **The FS stop memo is overstated.**
    - Text, RESEARCH_DECISION: "with SUCC's own beta transplant reproducing most of it".
    - Record: the memo (`roles/investigator_b/HANDBACK_FS_STOP.md`) predicts 0.005 (80% interval 0.000–0.011) and says "the part of it that SUCC's own operation (ALT) does not reproduce is about 0.001–0.004", so "most" holds only at the low end.
    - Fix: quote the memo's statement.
11. **ENDPOINTS.json contradicts itself.**
    - Problem: the FB_R1 and FA_R2 `three_outcomes_statement` fields say "(D2R NOT_RUN)", but their D2R fields say ROBUSTNESS_NOT_CONFIRMED, and FB_R1 has the sign change.
    - Record: the write-once DECISIONS.json files predate the D2R records (05:45:38Z vs 06:19:11Z; 06:21:28Z vs 06:38:00Z).
    - Fix: have the script append the D2R label and caveat, leaving DECISIONS.json unchanged. RESEARCH_DECISION should say that the D2R labels come from the D2R records.
12. **START_HERE does not separate point eligibility from the confidence conjunction and the reserve.**
    - Text: "met all 28 conditions at the point", for both FA and FB.
    - Record: both are CONFIDENCE_UNRESOLVED (27 of 28; version 1's men's guard), and neither is reserve-qualified.
    - Fix: add "(point eligibility only: 27 of 28 conditions pass the confidence conjunction, with version 1's men's guard unresolved; not reserve-qualified)".

## Recommended changes
1. Write "Point-feasible (G1)" instead of "Feasible" in the readings table and "both feasible readings". In HANDOFF and QUICKSTART, write "point-eligible releases" instead of "surviving releases".
2. "confidence-resolved on 27 of 28" → "27 of 28 conditions pass the confidence conjunction (release label CONFIDENCE_UNRESOLVED)".
3. D3 detail:
   - FA_TFRB_R2 also fails the recipient gate against H_LJ, B2TWIN and SUCC.
   - FB_PFO is AGG_POSITIVE against TR100, TT100, TWJ and W but fails their recipient gates.
4. FB_R1 saved only the upper CTRL and ALT points (deviation 7, CONTROL_POINT_RULE_NOT_FOLLOWED flags); say so.
5. UM is only just resolved: the upper bound is −0.0002 and version 1's difference is −0.0004.
6. FB_PFO "income frozen": add "equal to NEURAL_RAW's within 7e-8, not bitwise (deviation 10)".
7. Plain answer 7: add "253 of 256 checks PASS; the 3 FAILs concern two missing plan hashes and one card's bitwise claim; the CALIBRATION spends were re-applied, not recomputed".
8. HANDOFF "The best evidence is FA_TFRB_R2's version 1" → call it a post hoc single-version observation.
9. FS wording:
   - Use the card's name: "conditional pair-increment residual … against FB_PFD's closed-form whitened quadratic best response".
   - Say "no feasible strength in version 0 (version 1 had λ 0.05)".
   - Give plain answer 3 per version: FB_PFD version 0 had E1.
10. "leaf-correlation SEX critics" → "online SEX critics (acs_replication.leafcorr)", unless the card itself calls them leaf-correlation critics.
11. ENDPOINTS `campaign_summary`: include CAMPAIGN_SUMMARY.json's full statement.

## Checked and correct
- Every other number in the four markdown files matches its record:
  - All D2, D3 and D4 benefits and SEs.
  - The version means, including +0.0078 / +0.0090 and −0.0047 / +0.0179.
  - UM and the guards.
  - Worst value over cap: 0.012 / 0.698 and 0.601 / 0.785.
  - Confidence counts: 27, 27 and 26 of 28.
  - The recovery table (equal to ENDPOINTS.json).
  - D2R figures.
  - Strengths: S50; rho 1.0 / 0.6 and 0.6 / 1.0.
  - Failure bounds: 0.00383, 0.0052, and the FA_REG rho 0.3 failures.
  - The between-fold SE 0.0034 / 0.0025 vs household 0.0014 / 0.0013.
  - Verifier counts and the 05:11:39Z receipt.
- The sign convention (comparator − candidate) is applied consistently.
- The adaptive-search and reused-household caveats are present.

## Not verified
- Plan commit times (needs git).
- CPU-hours, backup and custody figures.
- The specification count, beyond its agreement with the coordinator log.
- FA_TFRO_R1's guard formula.
- TABLES.md beyond the quoted rows.
