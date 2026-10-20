# Statistical reviewer, card check against NRM1 (saved verbatim with attribution)

Author: the statistical reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:32Z on
2026-10-20. The reviewer states it read no outcome. Saved verbatim; nothing below was edited.

---

**Statistical reviewer: card check against NRM1 (RULES.md §12 and the claim definitions).** 15:31Z (`date -u`). Read-only; no sub-agents; no outcome read.

**Bottom line.**
- Nothing in the cards forces a change to NRM1 or to `nrm/rules.py`.
- The cards conform on the points they cover:
  - twins are declared before the fit, and FA fits its twin first;
  - each family has a SEX-blind G3 path (FB: MINF; FA: ORD);
  - FB uses R = 8 and reports the MC SE separately;
  - MEAN is never substituted for the randomized release;
  - verdicts use MODEL_SELECT spend measured by `nrm.evaluate`;
  - aliases are never saved twice;
  - no joint-design claim is made.
- Several items still have to be **declared before any MODEL_SELECT outcome is read**. If any of them is chosen after outcomes, the choice becomes a forking path.

## Declare now

**D1. G2 mapping for EVERY configuration, controls included.**
- `claims()` takes one twin and one alternative. Both cards list several "simplest alternatives". G2 is therefore computed once per declared alternative, and holds only if it holds for all of them.
- Proposed mapping:

| Configuration | Twin | Declared alternative(s) |
|---|---|---|
| FB_SEX_S2 | FB_TWIN | FB_MINF_S2 and FB_ISO_S2 (SEX-blind channels at matched second-order cost) |
| FB_SEX_S3 | FB_TWIN | FB_MINF_S3 |
| PRC_S2 | PRC_TWIN | SHUF_S2 and ORD_S2 |
| PRC_S1 | PRC_TWIN | ORD_S1 (same spend budget) |
| PRC_S3 | PRC_TWIN | ORD_S3 (same spend budget) |

- FB ID / MEAN / POSTHOC (F2, F3) and FA HQ_S2 (falsifier 3) are mechanism-attribution comparisons. They are reported beside G2 with the same aggregate screen and never enter G2.
- Controls may win, but only with a declared mapping. Proposed: FB_MINF_S2 ↔ FB_ISO_S2, and ORD_S2 ↔ SHUF_S2.
- Any configuration left without a declared alternative is `NO_MATCHED_CONTROLS` and cannot be selected. That must be stated now, not discovered later.

**D2. G3 paths, listed by point id, each including its twin as the s = 0 anchor.**
- FB path: FB_TWIN, FB_MINF_S2, FB_MINF_S3. ISO is not on the path; it is reported beside.
- FA path: PRC_TWIN, ORD_S1-S3, plus the contingent ORD bracketing entry (spend-only trigger, as declared).
- Proposed: G3 needs `DOMINATES_CONTROL_FRONTIER` against EACH declared SEX-blind path. For FB that is the MINF path and the CTL_REM path (FB_TWIN plus its levels), plus CTL_BLIND_OCC if fitted.
- **Risk to note now.** FB's own card says the 1.5 cap binds mostly for MINF, so MINF's realized cost may fall below SEX's at the same s. If FB_SEX_S3 spends more than every MINF point, the registered result is `NOT_SEPARATED_FROM_TASK_SPENDING` (reason: `CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT`).
  - If a MINF extension is wanted, declare it now, with a spend-only trigger as FA did for ORD.
  - Otherwise no MINF strength may be added after outcomes.

**D3. One frontier rule decides G3.**
- G3 is `nrm.rules.frontier_comparison`: bracket on paired total NLL, AGG_POSITIVE against Q*, all 8 secondary metrics checked.
- The CTL card's `nrm.ctl.frontier` verdicts are the challenger's declared diagnostic: clause-usage axis, tol 0.10 / 0.001 / 0.0005, and "CANDIDATE_AHEAD in both versions". They are reported beside G3, and any disagreement is disclosed next to the G3 label.
- If the coordinator prefers the CTL verdict as an extra NECESSARY condition, that is acceptable, but only if declared now.
- **"Margin m from nrm/rules.py":** NRM1 has no constant margin. For the diagnostic, use comparison-specific values from `nrm.rules.aggregate(entry)` on the registered cells:
  - m = 2 × `se_agg`;
  - for per-version verdicts, m = 2 × `per_version[v]['se_agg']`.
- **Linked recovery:** NRM1 uses the grid's pair view, where the ignore-one readers already compete through VALIDATION winner selection. A maximum taken on SCORE is selection on the scoring set. It never enters an NRM1 label; if the CTL diagnostic uses it, label it as such.

**D4. Falsifier readings.**
- FB F1, "not below the MINF path at S2 and S3": proposed reading is that F1 holds unless `frontier_comparison` gives DOMINATES at BOTH S2 and S3. Please confirm.
- FA: two reader classes must be enumerated before outcomes, with the decision rule. Proposal: per-recipe paired SCORE differences, each favouring PRC at the point.
  - "smooth readers", which falsifier 1 relies on. Proposal: OUT_LR, LOG_LR, OUT_MLP, LOG_MLP, MLP64, MLP6432.
  - "LOOKUP-type readers", which drive the reader-targeted label. Proposal: LOOKUP, CATI01, CATI1.
- These family rules qualify, but never replace, the NRM1 labels.

**D5. G6 and G7.**
- `outputs_changed` is determined mechanically: a wire hash differs from the family twin's in either version. Both families describe joint designs (FB: coupled channel with joint training; FA: pair-cell routing) with neither the separate design nor the sequential orders, so G6 = `NOT_ASSESSED_MISSING_CONTROLS`. A one-output result gives `NOT_APPLICABLE_ONE_OUTPUT_CHANGED`.
- FA's card states "Not novel (privacy funnel / IBSI; FARE)". Record G7(FA) = `PRIOR_ART_FOUND`.
- FB stays `NOT_CLAIMED` unless a review is done.
- FB's claims carry the scope sentence "one released instance per person; repeated-query protection not claimed".

**D6. Selection inputs (§12).**
- Every configuration that may be selected, controls included, needs registered-cell grid entries against all 9 references. Otherwise it is `NOT_SELECTABLE_INCOMPLETE_REFERENCE_GRID`. Decide now whether controls are read against all 9.
- `entry` = the row index of the version-0 evaluation in `runs/EVAL_LEDGER.jsonl`.
- `n_fitted_components` = model + calibrator + critic fits from the release records, summed over both versions.
- `grid_valid` = `nrm.rules.grid_validity`, with minimum set = the configuration plus its declared matched controls.
- The wires of all 9 references are passed to `select_candidate`.
- G3 needs `nrm.evaluate` paired costs, including `total|nll` and the 8 secondary metrics, plus pair-view grid entries of each candidate against every path point, its twin included.

## Randomized grid cells: how `mc_cell` inputs should be delivered

**Decision cell.** Every configuration uses its registered instance in each cell: r = repetition index, shared by k0 and k1. The two versions use different keys, so every block has its own independent instance. The block spread `se_rs` therefore already contains the instance-to-instance MC variance.
- FB_SEX_S3, FB_MINF_S3 and FB_ISO_S2 run with D = 1. They carry `MC_SE_NOT_ESTIMABLE` as an informational flag; their decision SE remains valid.

**FB_SEX_S2 and FB_MINF_S2.**
- The replicate members (offsets +2, +4) must be evaluated by the FULL grid procedure: 16 recipes, VALIDATION winner, the same split, reader seed and bootstrap weights, and the same comparators.
- Then, per cell and comparison, deliver `{'values': [x_r, x_{r+2}, x_{r+4}], 'ses': [s_r, s_{r+2}, s_{r+4}]}`. `R.mc_cell` returns the value (mean of the 3 draws), the household SE (RMS of the draw SEs) and `se_mc` (SD/√3). Build the screen entry as `{'cells': value, 'cell_se': se, 'cell_se_mc': se_mc}`.
- **If only ONE recipe is refit on the replicate instances (as FB card §8 says),** those are not evaluated draws under NRM1 §9:
  - the cell value stays the registered single-instance winner value;
  - the replicates serve only to estimate the MC SD, from that recipe's three values with the main instance included;
  - that MC SD enters as `cell_se_mc` = SD (not SD/√3, because the value is one draw), flagged `MC_SE_FROM_SINGLE_RECIPE`.
  - Full-bank replicates are preferred.

**Pairing of draws.**
- Two members that both have replicates (FB_SEX_S2 vs FB_MINF_S2) pair draw offset d with offset d.
- A member with replicates against one without pairs each draw with the comparator's single cell value.

**Instance independence.**
- With instance = repetition index (R1 = 0, R2 = 1), the sets are R1 {0, 2, 4} and R2 {1, 3, 5}, which are disjoint.
- If robustness repetitions R3 and R4 are run on randomized FB members, they must use instances 6 and 7, not 2 and 3, which are R1's and R2's replicate instances. Declare this now.
- k2 and k3 use their block's instance.

No NRM1 rule, threshold or code changes. Every item above is a declaration or an application of the frozen rules. Each must be recorded before the first MODEL_SELECT read of a protected or control release.
