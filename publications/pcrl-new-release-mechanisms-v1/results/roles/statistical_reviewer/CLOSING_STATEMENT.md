# Statistical reviewer, closing statement (saved verbatim with attribution)

Author: the statistical reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 16:34-16:35Z on
2026-10-20, in answer to the coordinator's 16:33Z request. Read-only; no sub-agents. Saved verbatim; nothing below the line was edited.

---

Statistical reviewer: closing statement (16:34Z; read-only; no sub-agents)

**(1) Application of the rules.**
- I checked each record against NRM1-rules-v1 (RULES.json SHA-256 84f18776…, the hash recorded in every decisive file), C1-C7 and RULING_FB_V3 with the 16:16Z clarification. I compared the summaries and decisive records with the rules. I did not recompute any aggregate; recomputation rests on the verifier's MAIN_V2 replay (0 mismatches).
- **claims/MAIN_V2 and claims/FINAL conform to the frozen rules:**
  - FB2_SEX_S3 and FB2_SEX_S5: G1 TASK_FEASIBLE; G2 NOT_MET; G3 NOT_SEPARATED_FROM_TASK_SPENDING on the MINF path and, in FINAL, on the CTL path (C7); G4 NOT_SUPERIOR; G5 NOT_MET; G6 NOT_ASSESSED_MISSING_CONTROLS; G7 NOT_CLAIMED.
  - FB2_MINF_*: NO_MATCHED_CONTROLS (C4).
  - FB_*: TASK_INFEASIBLE, so not selectable.
  - CTL_REM: G6 set mechanically.
  - Selection: NO_DEVELOPMENT_CANDIDATE.
- **The decisive records conform:**
  - FB2P: NOT_RUN under 2(a). Both strengths are only AGG_POSITIVE_POINT_ONLY against same-strength MINF (+0.0027 and +0.0019, each with SE 0.0089). The 16:16Z clarification was therefore never exercised.
  - F1_HOLDS in both MAIN_V2 and FINAL: neither strength is DOMINATES.
  - MEAN: NOT_FITTED under C5.
- **No departure found.** One consequence of the frozen rule, not a departure: PRC_TWIN's G2-G6 are NOT_COMPUTED_GRID_INVALID.
  - Cause: A-SC1 clause (ii) gives no input-cardinality explanation on the minimum set, and finite-cell wires make ET5/ET25 seed-invariant.
  - Effect: it touches only family A's twin and no decision.
  - A future campaign should allow that explanation for finite-cell minimum sets. This campaign stays as frozen.

**(2) Deviation 10.**
- I agree: no rule changed. `nrm/rules.py`, RULES.json and RULES.md have had no commit since 6dc7375f0.
- The one-line fix restored the distinct-input counts that A-SC1 clause (iii)(c) requires.
- The superseded MAIN output stays on record. The deviation entry must keep saying that the fix was made after that output was read (it does).

**(3) Decision A4.**
- No objection. Every reason is a measured result of the registered grids. The decision is recorded as a coordinator judgement, not a rule outcome.
- One wording condition: "Invertible reshaping removes nothing" must read as a measured result for the declared readers ("CTL_REM_0p50 changed pair recovery by about 0.001"). It must not read as a general principle, since the injectivity lesson was withdrawn (deviation 12).
- Documents must not present A4 as evidence that no mechanism can close the gap.

**(4) Required headline wording.**

> "No new release mechanism met this campaign's frozen development criteria, and no development candidate was selected. Family A (propensity-routed finite cells) closed at its internal capacity stage; only its task-only twin was evaluated. Family B's channel-trained learner failed the original utility limits. Its revision, a SEX-targeted Gaussian logit channel applied post hoc to the strongest admitted task-only base, kept both tasks within the original limits at two strengths, but did not lower linked-pair SEX recovery beyond its same-form, SEX-blind control at the same strength (benefit +0.0027 and +0.0019 AUC, each with SE 0.0089, not resolved). It was not separated from task spending on either SEX-blind control path, so its registered falsifier F1 holds and the family closed. The recovery reduction it did achieve against its base (about 0.011 pair AUC at the higher strength) was matched by untargeted noise of the same cost. No superiority over the nine admitted references, no 0.020 material improvement, no useful-component result, no joint-design contribution and no novelty is claimed. All results are development evidence on adaptively used MODEL_SELECT records, for the declared readers only; lower recovery for these readers is not protection against every reader."

**Words that must not appear:**
- Not "targeting helps", "promising" or "useful component": G2 was not met anywhere.
- Not "post-hoc noise cannot work" or "the gap is unclosable": no general claim follows.
- Not "confirmed" or "replicated".
