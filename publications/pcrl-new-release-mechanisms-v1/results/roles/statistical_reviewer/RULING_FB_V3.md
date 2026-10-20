# Statistical reviewer: ruling on card FB_FSC_v3 (saved verbatim with attribution)

Author: the statistical reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 16:00Z on 2026-10-20
(the reviewer's clock read 15:59:33Z; its header says 16:03Z, a time ahead of that clock read; recorded as received). Given before any
output of the registered grid NRM1_MAIN was read. Saved verbatim; nothing below was edited.

---

Statistical reviewer: ruling on the draft card FB_FSC_v3 (Designer B), given at 16:03Z (clock read at 15:59:33Z). Read-only, no sub-agents, and no output of NRM1_MAIN read.

**RULING: APPROVE WITH CHANGES.** Part 1 is approved as written, with an exact F2 definition added. Part 2 needs four changes. Part 3 is defined exactly below. Designer B may commit v3 once these are written into the card.

## 1. MEAN (C5): APPROVE

The strength rule matches C5 exactly: the highest FB2_SEX strength that is TASK_FEASIBLE in both versions and DOMINATES_CONTROL_FRONTIER against both declared SEX-blind paths (MINF and CTL_REM on FB2_BASE). If no strength qualifies, MEAN is not fitted.

Add to the card:
- **F2, exact definition.** F2 HOLDS iff `nrm.rules.benefit_screen` on the pair view is NOT `AGG_POSITIVE`.
  - Candidate FB2_SEX_S<j>, comparator FB2_SEX_S<j>_MEAN, registered cells.
  - In words: randomization counts as the active ingredient only if the randomized release is resolved as less recoverable than its own mean vector.
  - Report the aggregate, both version means, MEAN's own G1, and the single-view recipient differences beside it.
- **MEAN is not selectable.** It has no declared G2 alternative, so it is `NO_MATCHED_CONTROLS` (DECLARATIONS_V1, D1).

## 2. FB2P training perturbation: APPROVE WITH CHANGES

It is a genuine perturbation of everything the mechanism itself fits:
- the SEX geometry, from a household bootstrap of MODEL_FIT;
- the temperatures, from a household bootstrap of CALIBRATION, the same resample for SEX and MINF;
- the release key, which is fresh.

The frozen base is an admitted input, and the stated limitation covers it.

**Change (a): strength rule.** The brief says "repeat a PROMISING comparison", and a utility-only rule can repeat a comparison that is not promising. Replace the rule with this one, declared now before NRM1_MAIN is read:
- FB2P uses the highest FB2_SEX strength that meets both conditions:
  - TASK_FEASIBLE in both versions;
  - its registered-cell pair benefit against its same-strength FB2_MINF is `AGG_POSITIVE` on NRM1_MAIN.
- If no strength qualifies, FB2P is not fitted (label NOT_RUN), and its 4 configurations are not spent.
- FB2_MINF_S5's v1 utility failure does not block S5. The control is still perturbed at the same strength, and its own G1 is reported.

**Change (b): only the equal pair is read.**
- FB2_BASE and the CTL_REM points are not refitted. If they carry fitted parameters (temperatures, remainder shares), they are not equally perturbed.
- So the consistency reading uses only FB2P_SEX_S<j> against FB2P_MINF_S<j>.
- Comparisons against unperturbed points are descriptive.
- No frontier label is computed on a path that mixes perturbed and unperturbed points.

**Change (c): paired read.**
- Read FB2P on the registered cells with the declared instance map (R1 → instance 0, R2 → instance 1, k0 and k1 sharing it).
- Put it in a bank that also contains FB2_SEX_S<j> and FB2_MINF_S<j>. Their readers are cache hits, so nothing is refit, and the perturbed-minus-original contrast shares the bootstrap weights.
- Monte Carlo replicates are not required.

**Change (d): report utility and spend.**
- Report the G1 of both FB2P releases.
- Report the `nrm.evaluate` paired costs of FB2P_SEX against FB2P_MINF: `total|nll` and the 8 secondary metrics.

## 3. Perturbation reading: defined exactly (descriptive only)

All screens use NRM1 functions on the registered cells.

**Inputs.**
- B0 = `benefit_screen` (pair view) of FB2_SEX_S<j> against FB2_MINF_S<j>, with aggregate x0 and standard error s0.
- B1 = the same for FB2P_SEX_S<j> against FB2P_MINF_S<j>, with x1 and s1.
- Δ = x1 − x0.
- se_Δ = `se_agg` of the per-cell difference entry, with cells = b1_c − b0_c and cell_se = the SD of the paired draw differences from the shared bank (`nrm.rules.aggregate`). Without a shared bank, use the bound se_Δ = s0 + s1.
- c1 = pooled paired total-NLL cost of FB2P_SEX against FB2P_MINF (`nrm.rules._primary`), with value and SE.

**Labels.**
- `NOT_RUN`: FB2P was not fitted (rule 2a) or not read on the registered cells.
- `NOT_APPLICABLE_ORIGINAL_NOT_POSITIVE`: B0 is not AGG_POSITIVE. The numbers are still reported. Under rule 2a this cannot occur for a fitted FB2P.
- `PERTURBATION_CONSISTENT`: all three of the following hold:
  - B1 is AGG_POSITIVE;
  - |Δ| ≤ 2·se_Δ;
  - SEX does not spend resolved-more than MINF after the perturbation, that is, NOT (c1.value − 2·c1.se > 0).
- `PERTURBATION_NOT_CONSISTENT`: otherwise. The failing parts are named: "perturbed benefit not resolved", "benefit changed by more than 2·se_Δ (direction)", or "SEX now spends more than MINF".

**Status.** The label is a qualifier reported beside FB2's "SEX beyond MINF" statement.
- It never changes a decision, a tier, a selection or a falsifier.
- The FB2P releases are kind 'perturbation'. They are never pooled with FB2, never matched controls, and never selectable (no declared G2 alternative).
- The limitation sentence "the frozen base models are not refitted; the perturbation covers the channel's fitted parameters and the release key only" travels with every FB2P result.

## Budget

At most 6 more configurations (MEAN 2, conditional on C5; FB2P 4, conditional on rule 2a), so family B holds at most 34.

No change to NRM1 is needed.
