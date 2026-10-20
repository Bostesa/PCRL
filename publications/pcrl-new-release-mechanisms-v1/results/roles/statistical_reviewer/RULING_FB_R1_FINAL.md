# Statistical reviewer: FINAL ruling on family B revision 1 (FB2) (saved verbatim with attribution)

Author: the statistical reviewer of campaign pcrl_new_release_mechanisms_v1, replying to the coordinator's resent request. Returned at
about 15:38Z on 2026-10-20 (the reviewer's clock read 15:36:04Z). It supersedes RULING_FB_R1.md where they differ. In particular, F1 is
read as in C6 below, not per strength as in the earlier A6. Saved verbatim; nothing below was edited. Before any FB2 fit or FB2 read.

---

Statistical reviewer, ruling on family B revision 1 (FB2). Clock read at 15:36:04Z. Read-only; no sub-agents.

**RULING: APPROVE WITH CHANGES.** Conditions C1-C7 below apply before any FB2 fit, and C3 also before any FB2 read.

## Why I approve

- **R1 applies.** P1 failed: FB_TWIN fails v1 on the men's guard (0.0085), and every FB_* configuration fails v1.
- **Dropping channel training is consistent with the card.** F3 holds in screening: POSTHOC 0.6292 / 0.6383 against SEX_S2 0.6325 / 0.6391.
- **The card's stop rules are not triggered in screening.**
  - F2 does not hold: MEAN (0.6410 / 0.6457) and ID (0.6403 / 0.6454) recover more than SEX_S2.
  - F1 does not hold: SEX is below MINF at S2 and at S3.
- **The SEX-below-MINF signal is not a spending artifact at the point level.**
  - Realized total-NLL excess at S2: SEX 0.0081 / 0.0103 against MINF 0.0093 / 0.0111.
  - At S3: 0.0122 / 0.0153 against 0.0128 / 0.0151.
  - These are screening point values with no uncertainty.
- **The fix fits the failure.** A frozen task-only base with 0.013-0.026 nats of slack is the right remedy for a twin with none.
- **Disclosure.** I read these family-B aggregates in dev/SCREEN_LEDGER.csv and dev/UTILITY_LEDGER.csv after registration. I read no label.

## Conditions

**C1. Record the deviation.** Replacing the card's registered R1 (a wider MLP, otherwise UTILITY_INFEASIBLE and close) after MODEL_SELECT outcomes is a protocol deviation. It is not yet in PROTOCOL_DEVIATIONS.md, whose items 1-7 do not cover it. Add it. Attach this sentence to every FB2 label: "FB2 was designed after family-B screening on MODEL_SELECT (targeting signal, F3); it is evaluated on the same people; development evidence only."

**C2. Commit an FB2 card before the fit.** It must contain:
- the G2, G3 and F1 declarations of DECLARATIONS_V1, as amended by C4-C7;
- strengths {0.12, 0.21} plus the contingent MINF_S6 at 0.26, with no other strength;
- R = 8 and the instance map: R1 → 0, R2 → 1, R3 → 6, R4 → 7, with k2 and k3 using their block's instance;
- MC replicates at +2 and +4 for FB2_SEX_S5 and FB2_MINF_S5, run through the full 16-recipe procedure;
- G6 = NOT_ASSESSED_MISSING_CONTROLS, with `outputs_changed` set mechanically;
- G7 = NOT_CLAIMED;
- the scope sentence "one released instance per person; repeated-query protection not claimed";
- an updated closest-prior-attempt section. Without channel training, FB2 is post-hoc noise on a frozen predictor, so the closed stochastic-channel and replacement-channel studies are now its closest prior work.

**C3. Gate on the base first.** FB2_BASE must pass all 24 clauses and 4 guards in both versions, through `nrm.evaluate` on hash-checked wires, before any protected FB2 release is read on MODEL_SELECT. If it fails, stop before reading any protected release.
- From published values it should pass: its conditions are TWJ's income conditions plus W's occupation-side conditions.
- Record two alias relations: its income view is an exact alias of TWJ's, and its occupation view of W's.

**C4. FB2_MINF_S3 has no valid G2 mapping as written.** "Against the twin only" is NO_MATCHED_CONTROLS under the frozen G2, which requires the twin AND a declared alternative. Either declare an alternative now, or record FB2_MINF_* as NO_MATCHED_CONTROLS and not selectable. I recommend recording them as not selectable.

**C5. Fix the MEAN strength now (decisive phase, +2).** MEAN is fitted at the highest FB2_SEX strength that meets two conditions: it is TASK_FEASIBLE in both versions, and it is DOMINATES_CONTROL_FRONTIER against both declared paths. If no strength qualifies, MEAN is not fitted.

**C6. F1 reading accepted as declared, with one guard.** F1 holds unless the result is DOMINATES at both S3 and S5. An INCOMPLETE frontier record (missing paired costs or grid entries) must be completed. It is never read as F1 and never as a pass.

**C7. Declare and fund the CTL_REM path on FB2_BASE now.** G3 requires DOMINATES against each declared SEX-blind path, so an unbracketed strength fails G3 mechanically.
- Bracketing both FB2_SEX_S3 and FB2_SEX_S5 takes up to 2 levels × 2 versions = 4 configurations, chosen by measured spend only.
- If only one level is funded, the CTL path must be declared for that strength only.

## Budget

FB2 needs 10, plus 2 for MINF_S6, plus 2 for MEAN, plus 2-4 for CTL_REM. That total must fit the remaining budget. Resolve the verifier's duplicate configuration number from concurrent saves first, because the budget counter is not atomic.

## Treatment of the already-fitted FB_* releases

- **They stay.** All 18 stay in the ledger and count toward the 60. Every result is reported from the actual outputs, failed checks included.
- **G1 = TASK_INFEASIBLE for every FB_* configuration.** Every one fails v1; only TWIN, ID and MEAN pass v0.
- **What that implies under the frozen rules:**
  - A protection label that would pass is MET_WITHOUT_TASK_FEASIBILITY. One that fails keeps its failing label.
  - None carries the useful-component label, none has a tier, and none is selectable.
- **They are reported as family-B v1 outcomes, labelled as screening:** F3 holds; F2 and F1 were not triggered.
- **They are kept apart from FB2.** FB_* are never pooled with FB2 and never act as FB2 matched controls, because the base differs. A formal-grid read of FB_* is optional and descriptive only.
