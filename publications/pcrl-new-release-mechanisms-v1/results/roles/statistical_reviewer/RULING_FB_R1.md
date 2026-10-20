# Statistical reviewer: ruling on family B revision 1 and card check (saved verbatim with attribution)

Author: the statistical reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:37Z on 2026-10-20
(the reviewer's `date -u` read 15:33:28Z). Saved verbatim; nothing below was edited. Before any FB2 fit or FB2 read.

---

Statistical reviewer, reply of 15:35Z (`date -u` read at 15:33:28Z). It covers the card check and the FB revision 1 ruling.

**No change to NRM1 is needed.** Everything below is a declaration or an application of the frozen rules, and it must be recorded before the outcomes it governs are read.

To check the proposal I read, after registration, the public aggregates for family B in `dev/SCREEN_LEDGER.csv` and `dev/UTILITY_LEDGER.csv`. No label was read.

## A. FB revision 1 (FB2): APPROVE WITH CHANGES

**Reasons to approve:**
- P1 failed, so the card's R1 applies. FB_TWIN fails v1 (men's guard 0.0085); every FB configuration fails v1.
- F3 holds in screening: POSTHOC 0.6292 / 0.6383 against SEX_S2 0.6325 / 0.6391. Dropping channel training is therefore consistent with the card, which says to report that joint adaptation adds nothing.
- The card's stop rules are not triggered:
  - F2 does not hold. MEAN (0.6410 / 0.6457) and ID (0.6403 / 0.6454) recover more than SEX_S2.
  - F1 does not hold in screening. SEX is below MINF at S2 and S3.
- The SEX-below-MINF signal is not a spending artifact at the point level:
  - realized total NLL excess at S2: SEX 0.0081 / 0.0103 against MINF 0.0093 / 0.0111;
  - at S3: SEX 0.0122 / 0.0153 against MINF 0.0128 / 0.0151.
  - This is screening only, with no uncertainty attached.
- A frozen base with 0.013-0.026 nats of slack is the right fix for a twin that has none.

**Required changes (all before any FB2 fit):**
1. **Protocol deviation.** The registered R1 (a wider MLP, else close) is replaced after MODEL_SELECT outcomes. Record this, and attach to every FB2 label: "FB2 was designed after family-B screening on MODEL_SELECT (targeting signal, F3); it is evaluated on the same people; development evidence only".
2. **An FB2 card committed before the fit,** declaring:
   - the twin: FB2_BASE;
   - the G2 alternative per candidate: FB2_MINF_S3 for FB2_SEX_S3 and FB2_MINF_S5 for FB2_SEX_S5;
   - the G3 path: {FB2_BASE, FB2_MINF_S3, FB2_MINF_S5};
   - the fixed strengths {0.12, 0.21} for SEX and MINF, with no others;
   - R = 8, the instance map and which configurations get MC replicates (I suggest the S3 pair);
   - G6 = NOT_ASSESSED_MISSING_CONTROLS (a coupled channel on both outputs, with no separate design or sequential orders);
   - G7 = NOT_CLAIMED;
   - the scope sentence "one released instance per person; repeated-query protection not claimed";
   - an updated closest-prior-attempt section. With training dropped, FB2 is post-hoc noise on a frozen predictor, and the closed stochastic-channel and replacement-channel studies are now the closest prior work.
3. **A contingent MINF bracketing level, declared now:** +2 configurations, chosen by measured MODEL_SELECT spend only. It is fitted only if `nrm.rules.frontier_comparison` returns CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT for an FB2_SEX point. Without it, an unbracketed S5 can only be NOT_SEPARATED. No extension is allowed after the fact.
4. **CTL_REM moves to FB2_BASE.** It must not run on FB_TWIN. Its level is chosen by spend only to bracket the FB2_SEX spend named in the CTL v2 card, which is committed before the CTL fit. It is a second SEX-blind G3 path (see B2).
5. **FB2_BASE is evaluated first.** It must pass 24 + 4 in both versions through `nrm.evaluate`, on hash-checked wires, before any protected FB2 release is read on MODEL_SELECT.
   - From published values it should pass (TWJ's income conditions plus W's occupation-side conditions).
   - Its income view is an exact alias of TWJ's and its occupation view of W's. These are recorded once.
6. **F1 must be written unambiguously.** Recommended reading: evaluated per strength with `frontier_comparison` against the MINF path; F1 holds, and the family closes, if either strength is NOT_SEPARATED_FROM_TASK_SPENDING.
7. **Recommended, not required:** estimate δ and Σ_w from MODEL_FIT out-of-fold logits where an admitted out-of-fold bank exists (A1_PO_16 has one). In-sample HGB logits are sharper than deployed ones. If in-sample logits are used, disclose it.

**Budget:** 10 + 2 contingent MINF + 2 decisive MEAN = 14, within the remainder.

**Treatment of the already-fitted FB_* releases:**
- All 18 stay in the ledger and count toward the 60. Every result, failed checks included, is reported from the actual outputs.
- **G1 = TASK_INFEASIBLE for every FB_* configuration.** All fail v1; only FB_TWIN, ID and MEAN pass v0.
- Consequences under the frozen rules:
  - a protection label that would pass becomes MET_WITHOUT_TASK_FEASIBILITY; one that fails keeps its failing label;
  - no useful-component label (it requires G1);
  - no tier, so none is selectable (BELOW_MINIMUM_TIER).
- The v1 falsifier outcomes are reported as family-B v1 results, with screening labelled as screening: F3 holds, F2 and F1 do not.
- FB_* are never pooled with FB2 and never serve as FB2 matched controls; the base differs. Reading them on the formal grid is optional and descriptive only.

## B. Card check (FA_PRC_v1, FB_FSC_v1, CTL_CARD_v1, CARD_DECISIONS_V1)

What already conforms:
- Twins are declared and fitted first.
- SEX-blind G3 paths exist (ORD; MINF, now in FB2).
- R = 8 with the MC SE kept separate; MEAN is never substituted.
- Verdicts use spend measured on MODEL_SELECT.
- Aliases are never saved twice.
- No joint-design claims.

Declarations needed before outcomes:

1. **G2 mapping for every configuration.** `claims()` takes one alternative, and the cards list several. Choosing among them after outcomes would be forking. Declare, and compute as a conjunction (G2 is run once per listed alternative and must hold for each):
   - PRC_S1 / S2 / S3: twin PRC_TWIN; alternative ORD at the same budget. PRC_S2 also needs SHUF_S2.
   - HQ_S2 (FA) and ID / MEAN / POSTHOC (FB) are mechanism-attribution comparisons. They are reported with the same screens beside G2 and are never part of it.
   - Controls may win only if a G2 mapping is declared for them now. A control left without one is NO_MATCHED_CONTROLS, so not selectable. Say so now.
2. **G3 paths, listed by configuration id:**
   - FA: {PRC_TWIN, ORD_S1, ORD_S2, ORD_S3} plus the contingent spend-only ORD entry (declared).
   - FB2: as in A2.
   - G3 should require DOMINATES_CONTROL_FRONTIER against each declared SEX-blind path that brackets the candidate (the family path and any CTL path). That is conservative: a SEX-blind control that explains the benefit defeats G3.
3. **The CTL verdict rule versus NRM1.** The CTL card's credit rule (CANDIDATE_AHEAD in both versions, on the max-clause-usage axis, tol 0.10 / 0.001 / 0.0005) is a second rule for the same question.
   - Declare now: NRM1 `frontier_comparison` decides G3. The `nrm.ctl.frontier` verdicts are the challenger's diagnostic, reported beside, with any disagreement disclosed. Making them an additional necessary condition is also acceptable, but only if declared now.
   - "Margin m from nrm/rules.py": NRM1 has no constant margin. Use m = 2·se_agg of that specific comparison from `nrm.rules.aggregate` (registered cells; `per_version[v]['se_agg']` for per-version verdicts).
   - Recovery means the grid's pair view. Ignore-one readers already compete there through VALIDATION selection. A SCORE-side max is not used for any NRM1 label.
4. **FA falsifier 1 needs fixed reader sets.** It credits PRC-vs-ORD only "for smooth readers and for the cross-fitted LOOKUP" and labels LOOKUP-only advantages "reader-targeted". Fix the exact recipe lists now, for example:
   - smooth = {OUT_LR, LOG_LR, OUT_MLP, LOG_MLP, MLP64, MLP6432};
   - LOOKUP-type = {LOOKUP, CATI01, CATI1}.
   Also fix the decision rule (for example, the per-recipe SCORE AUC difference favours PRC in every registered cell's mean). It is a family qualifier, not an NRM1 label.
5. **Selection inputs:**
   - every selectable configuration, controls included, needs registered-cell grid entries against all 9 references; otherwise it is NOT_SELECTABLE_INCOMPLETE_REFERENCE_GRID. Decide now whether controls get those reads;
   - `entry` = the row index of the version-0 evaluation in runs/EVAL_LEDGER.jsonl;
   - `n_fitted_components` = model + calibrator + critic fits from the release records, summed over both versions;
   - `grid_valid` from `nrm.rules.grid_validity`, with the minimum set = the candidate plus its declared matched controls;
   - the wires of all 9 references.
6. **G6 and G7:**
   - `outputs_changed` is determined mechanically, by wire hash against the family twin.
   - FA (pair-cell routing) and FB / FB2 (coupled channel) are joint without separate designs, so G6 = NOT_ASSESSED_MISSING_CONTROLS.
   - FA's card states "not novel (privacy funnel / IBSI; FARE)", so record G7 = PRIOR_ART_FOUND for FA. FB2 is NOT_CLAIMED.

## C. How to deliver randomized-release grid cells (mc_cell inputs)

- **Decision value.** Every configuration uses the registered instance: r = repetition index, with k0 and k1 sharing it.
  - Each block uses an independent instance, because keys differ per version. The block spread se_rs therefore already contains the instance-to-instance MC variance.
  - D = 1 configurations are valid; their MC_SE_NOT_ESTIMABLE flag is informational.
- **MC replicate members (offsets +2 and +4).** They must be evaluated by the full grid procedure: 16 recipes, VALIDATION winners, same split, same reader seed, same bootstrap weights.
  - Deliver per (candidate, comparator, view) and cell: `{'values': [x_r, x_r+2, x_r+4], 'ses': [their paired bootstrap SEs]}`.
  - I apply `nrm.rules.mc_cell` to get the cell value (the mean), the cell SE (RMS) and `cell_se_mc`.
  - Pair draws by offset when both releases have replicates. Otherwise pair each candidate draw with the comparator's cell.
- **If only one recipe is refit on the replicates** (as FB card §8 says), those are not evaluated draws:
  - the cell value stays the registered winner value;
  - the replicates give a reported MC diagnostic only (flag MC_SE_FROM_SINGLE_RECIPE), never `cell_se_mc`.
- **Robustness repetitions on randomized releases.** Use instances 6 (R3) and 7 (R4). Setting instance = repetition index would reuse the R1 / R2 replicate instances 2-5 and correlate blocks.
