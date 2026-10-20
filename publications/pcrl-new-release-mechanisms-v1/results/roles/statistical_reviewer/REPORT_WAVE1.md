# Statistical reviewer, wave 1 report (saved verbatim with attribution)

Author: the statistical reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:12Z on 2026-10-20
as its final message. Saved verbatim by the coordinator; nothing below was edited. The governing texts are RULES.md and RULES.json in
this folder (commit 6dc7375f0).

---

**Statistical reviewer, campaign `pcrl_new_release_mechanisms_v1`: the generic claims and selection rules are frozen (schema NRM1-rules-v1).** They are committed in 6dc7375f0 on `research/pcrl-new-release-mechanisms-v1`, before any candidate outcome existed. `python3 -m nrm.test_rules` passes 18 of 18 tests, all on invented data. No label was read and no real-data computation was run. Writing started at 2026-10-20T14:45:00Z and finished at 15:11:09Z (`date -u`). RULES.md is in the role folder, so it is summarised below rather than reproduced.

## Files committed (only mine; commit message has no trailer)

| File | SHA-256 |
|---|---|
| `results/pcrl_new_release_mechanisms_v1/roles/statistical_reviewer/RULES.md` | (human-readable rules; its completion record holds the other hashes) |
| `results/pcrl_new_release_mechanisms_v1/roles/statistical_reviewer/RULES.json` | 84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652 |
| `nrm/rules.py` | d5df868678d483a3452e251d8f24575932d23f0e5d22e3ef1ca3627a35030df1 |
| `nrm/test_rules.py` | d03bbed2f2a4b94c85f48b24926274567849a80e65ff816ee8217c4b668f60d5 |
| `results/pcrl_new_release_mechanisms_v1/roles/statistical_reviewer/SCREEN_CHECKS.json` | fc37fc64d6f750f9c69594600aed708347e82120c585ebb27343200d35713d81 |

- `SCREEN_CHECKS.json` is deterministic (rng seed 20261020).
- `load_rules()` refuses a RULES.json whose constants disagree with the module.
- CPU was charged through `nrm.jobqueue charge`: 90 s upper estimate, which includes the review subagent's probes.

## Original limits (unchanged)

- 24 clauses: accuracy loss ≤ 0.010, NLL excess ≤ 0.010 nats, summed Brier excess ≤ 0.005. These cover 2 tasks × U0_S<v>/Ucal_S<v> × 2 versions, inclusive.
- 4 guards: men's and women's occupation summed-Brier excess vs U0 ≤ 0.005.
- Recipient allowance, as defined in OHN1 and TSZ1: each single output's SEX-recovery AUC minus the comparator's must be ≤ 0.010, per comparator, view and version.
- Material target: 0.020 AUC of pair benefit vs NEURAL_RAW and H_LJ, reported separately from smaller positive improvements.
- Reserve: `ohn.rules.reserve` unchanged (k = 2, H_LJ SE floor). It is applied identically to candidates, controls and references. For randomized releases the own SE is sqrt(se² + se_mc²).

## Generic claims (kept distinct; none implies another)

- **G1, task feasibility (`TASK_FEASIBLE`):** the actual emitted outputs pass 24 + 4 at the point. Randomized releases use draw-averaged values. The reserve is reported beside G1, never as part of it.
- **G2, incremental reduction (`INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS`):** G1 holds, and against both the task-only twin and the declared simplest alternative explanation the pair benefit is `AGG_POSITIVE` and the recipient gate passes.
- **G3, stable frontier improvement (`STABLE_FRONTIER_IMPROVEMENT`):** G1 holds and the frontier label is `DOMINATES_CONTROL_FRONTIER` against the family's SEX-blind path.
- **G4, original-benchmark superiority (`SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES`):** G1 holds, and against every one of the 9 point-eligible references the pair benefit is AGG_POSITIVE and the recipient gate passes. This is intersection-union, so no multiplicity adjustment is needed. The old every-cell profile is shown beside it.
- **G5, the 0.020 criterion (`MATERIAL_0020_RESOLVED`):** G1 holds, `TARGET_AGG_RESOLVED` vs NEURAL_RAW and vs H_LJ, and the recipient gate passes against both.
  - Other labels: `TARGET_RESOLVED_RECIPIENT_GATE_FAILED`, `MATERIAL_0020_POINT_ONLY`, `POSITIVE_BELOW_MATERIAL_TARGET`, `NOT_MET`.
  - G5 is never nested in G4.
- **G6, joint-design contribution:** checked in this order.
  1. Fewer than two outputs changed: `NOT_APPLICABLE_ONE_OUTPUT_CHANGED`, with the sentence "only the X output changes".
  2. Both changed but not declared joint: `NOT_APPLICABLE_NOT_JOINT`.
  3. The separate design or either sequential order is missing: `NOT_ASSESSED_MISSING_CONTROLS`.
  4. Otherwise `JOINT_CONTRIBUTION_SUPPORTED` (AGG_POSITIVE, recipient gate, and primary cost not resolved higher vs all three) or `NOT_SUPPORTED`.
- **G7, novelty:** copied from a separate prior-art review; `NOT_CLAIMED` by default. It is never inferred from performance.
- **Useful component:** `USEFUL_COMPONENT_NOT_COMPETITIVE` iff G2 holds and G4 does not. Always shown.
- **Without G1**, any protection label is `MET_WITHOUT_TASK_FEASIBILITY` and is never claimable.
- **Invalid reader grid:** `grid_valid` must be recorded and true. If it is missing, G2-G6 are `NOT_COMPUTED_GRID_VALIDITY_UNKNOWN`; if false, `NOT_COMPUTED_GRID_INVALID`.

## Reference set (frozen, with published labels)

| Reference | Point label | Whole-release reserve | Min ratio v0 / v1 |
|---|---|---|---|
| NEURAL_RAW | POINT_ELIGIBLE | NOT_MET | 1.947 / 1.558 |
| H_LJ | POINT_ELIGIBLE | NOT_MET | 0.321 / 0.624 |
| B2TWIN | POINT_ELIGIBLE | NOT_MET | 0.321 / 0.624 |
| RETIRED_B3CTL | POINT_ELIGIBLE | NOT_MET | 0.321 / 0.349 |
| SUCC | POINT_ELIGIBLE | NOT_MET | 0.321 / 0.624 |
| TWJ | POINT_ELIGIBLE | NOT_MET | 0.321 / 0.624 |
| W (A1_PO_16 + H3) | POINT_ELIGIBLE | MET | 3.511 / 2.862 |
| TR100 (TASK_REFIT + H3) | POINT_ELIGIBLE | MET | 3.412 / 2.862 |
| TT100 (T_TREE + H3) | POINT_ELIGIBLE | MET | 3.326 / 2.600 |

- All nine form the original benchmark, whatever their reserve status. The published label governs; a re-evaluation is reported beside it and flagged `REFERENCE_POINT_STATUS_DISAGREES` if it differs.
- Diagnostic only: PUB_IBSI (`DIAGNOSTIC_POINT_INELIGIBLE`, men's guard 0.0050142); U0 and Ucal (`UTILITY_REFERENCE_ONLY`).
- Known alias groups, verified by wire hash and counted once:
  - occupation view: H_LJ, B2TWIN, SUCC and TWJ (JOINT_TREE);
  - income view: SUCC, W, TR100 and TT100 (H3).
- A candidate carrying H3 income is `ALIAS_ZERO` on the income view: it passes non-worsening and can never show a benefit.

## Reserve-qualified frontier

- Qualified means `RESERVE_MET` on both versions, under the same rule for every method.
- The Pareto set uses pair AUC and minimum ratio.
- `RESERVE_FRONTIER_IMPROVEMENT` requires AGG_POSITIVE against every qualified control and reference.
- Failing references stay listed, and stay in the original benchmark.
- A reserve-frontier result is not superiority over every historical method.

## Aggregate paired screen (prospective)

**Cells.**
- Decisions use the 8 registered cells S<v>R<r>k<k>.
- R3, R4, k2 and k3 give only a `ROBUSTNESS_CONFIRMED` / `ROBUSTNESS_NOT_CONFIRMED` / `NOT_RUN` label. They never change a decision or a ranking.
- Learning-curve '@' cells are never used.

**Estimator.**
- Each block (version, repetition) has a mean m_b over its reader-seed cells. The aggregate x is the mean of the block means.

**Uncertainty:** se_agg = max(sqrt(se_hh² + se_rs² + se_mc²), **0.001**).
- se_hh is the cell-weighted RMS of the paired bootstrap SEs. It is an upper bound on the household-sampling SE under any correlation between cells, so it covers households shared across versions, seeds and repetitions.
- se_rs = sqrt(Σ_b(m_b − x)²/n_blocks). For the crossed 2-version design it is unbiased for the version and split components and conservative for the interaction and within-block components.
- se_mc is the same RMS bound over MC SEs.
- The 0.001 floor binds only for near aliases.
- An exact alias (every value and SE equal to 0) is `ALIAS_ZERO`. It is never positive and is counted once.

**Benefit, z = 2.** `AGG_POSITIVE` iff x − 2·se_agg > 0 **and** both version means > 0. Otherwise `AGG_POSITIVE_POINT_ONLY` or `AGG_NOT_POSITIVE`.

**Target.** The same rule at 0.020: `TARGET_AGG_RESOLVED` / `TARGET_AGG_POINT_ONLY` / `TARGET_AGG_NOT_MET`.

**Recipient view.** Evaluated per version, because the original allowance is per version.
- The gate passes iff r_v ≤ 0.010 for both versions, inclusive, at the aggregate point. One noisy cell cannot veto.
- Labels: `RECIPIENT_AGG_RESOLVED` (r_v + 2se_v ≤ 0.010), `_EDGE`, `_POINT_FAILED`, `_RESOLVED_WORSE`.

**Reported for every comparison.** Sign reversals by cell, block and version; n_eff over distinct cells; `k_differs`; the reader-seed qualifier.

**Old every-cell rules.** `tsz.rules.stability_label`, `target_label` and `recipient_gate` are kept beside as diagnostics. Any disagreement with the new decision is flagged `EVERY_CELL_RULE_DIVERGES` and disclosed. Earlier campaigns are never relabelled.

## Known-law behaviour (SCREEN_CHECKS.json; invented laws, not a universal guarantee)

The simulation used 20,000 replicates per setting. The cells were correlated: shared-household errors with correlation 0.8 within a block, 0.5 across versions on the same split, and 0.15 across splits, plus version, split, interaction and reader-seed effects. The scalar code agreed with a vectorized copy on 8,800 checks.

- **Benefit false-pass rate at a true mean of 0:**
  - new screen 0.0007-0.0197 under six of the eight laws, old rule 0.0003-0.0070;
  - about 0.08 when one version or split component dominates (SD 0.008, new) against 0.054-0.064 for the old rule. This is a disclosed limitation: those components rest on 1 df each.
- **Benefit pass rate, new screen / old rule** (five moderate laws):
  - true mean 0.005: 0.05-0.16 / 0.01-0.08;
  - true mean 0.010: 0.45-0.51 / 0.21-0.37;
  - true mean 0.020: 0.96-0.996 / 0.87-0.98.
  - Under the large-noise law the new screen passes 0.45 at 0.020.
- **Target screen** at a true 0.020: false pass ≤ 0.020 under the six moderate and large-noise laws (extreme laws in the JSON).
- **Recipient gate** (eight laws):
  - truly non-worsening outputs pass 0.79-0.997, against 0.38-0.91 for the old every-cell gate, which vetoed 9-62% of them;
  - at the allowance (0.010), 0.25-0.44 pass;
  - at 0.015, 0.02-0.16 pass.
  - `RESOLVED` (upper bound within the allowance) is reported beside the gate.
- **Exact aliases** never pass.

## Reader stability and grid validity

- **Reported per comparison:** sign changes over distinct cells; cells beyond the threshold; winner changes; seed-dependent winner cells; deterministic cells; near ties (< 0.010); `k_differs`, n_eff and the qualifier. An identical view is reported as `EXACT_ALIAS_COUNTED_ONCE`.
- **Grid validity:** the A-SC1 criterion, applied prospectively.
  - The four registered reader seeds must be recorded and distinct, and none may reuse offsets +300 or +500 (this campaign uses +600).
  - No seed-dependent equality on the minimum set.
  - Every other equality must be explained by recorded differing checks plus a recorded distinct-input count.

## Frontier against the SEX-blind control path (no exact cost match)

**Primary cost.** The paired difference in total NLL (income + occupation) with household SEs. Versions are pooled with SE equal to the mean of the two SEs.

**Rule.**
- The bracket Q* is the cheapest control whose pooled cost is ≥ the candidate's.
- `DOMINATES_CONTROL_FRONTIER` iff the benefit vs Q* is AGG_POSITIVE and none of the 8 secondary metrics shows resolved extra spending (d − 2se > 0). All 8 are required; otherwise the result is `INCOMPLETE`.
- If only the secondary check fails: `SEPARATED_ON_PRIMARY_COST_ONLY`.
- `NOT_SEPARATED_FROM_TASK_SPENDING` has one of three reasons: the candidate spends more than every control point; no resolved benefit at Q*; or the control path is missing.

**Interpolation.** Reported, never gating.

## Selection (at most one development candidate)

**Universe.** The complete ledger, controls included. A control can win and carries `CONTROL_DESIGNATED_CANDIDATE`.

**Tiers** (minimum T2):
- T5 = G1-G4 plus G5 resolved;
- T4 = G1-G4;
- T3 = G1-G3;
- T2 = G1 and G2.

**Choice.**
- Score = minimum over the 9 references of the aggregate x. Near-tie width 0.005.
- Tie-breaks, in order:
  1. reserve-qualified;
  2. leximin ratio vector;
  3. fewer fitted components;
  4. earliest ledger entry;
  5. config_id.

**Not selectable.**
- Output hashes from the evaluation and the grid differ (no substitution).
- `ALIAS_OF_<reference>`, detected by wire hash or by an ALIAS_ZERO pair comparison. Reference wires are a required input.
- An alias of an earlier configuration.
- An incomplete reference grid.
- Below T2.

If nothing qualifies, the outcome is `NO_DEVELOPMENT_CANDIDATE`.

## Randomized releases

- **Utility:** each metric on each emitted draw, averaged over R one-draw repetitions. Every draw's own point status is reported.
- **Grid cell:** the mean over draws, with the household SE as the RMS of the draws' SEs and the MC SE as SD/√D. D = 1 is flagged `MC_SE_NOT_ESTIMABLE`.
- MC uncertainty enters decisions as its own component and is always reported separately from household and reader variation.

## Changes before freezing (disclosed in RULES.md §13)

**Independent code review.** I launched a review subagent, which found seven correctness defects. All were reproduced, fixed and given tests:
1. G3 could pass with secondary spending metrics missing or NaN.
2. The fitted-components tie-break turned 0 into infinity and crashed on per-version records.
3. Grid validity counted unchecked recipes as "differs" and accepted a single seed.
4. An unrecorded or numpy-False `grid_valid` did not block claims.
5. Reference-alias detection was off by default.
6. G6 gave the wrong label and a false sentence when both outputs changed without a joint design.
7. G5 gave positive levels without G1, and mislabelled "target resolved, gate failed".

No threshold, z, floor, cell set, reference or tier changed. The screen was not tuned to the simulated rates.

**Process note.** One shell call at about 14:50Z contained an empty heredoc. It executed nothing, but the brief forbids heredocs.

## Open points for the coordinator (RULES.md §12)

1. Per configuration and version, records must carry:
   - `wire_sha256_eval` and `wire_sha256_grid` (every draw for randomized releases);
   - `n_fitted_components`, `entry`, `kind` and `grid_valid`.
2. Pass the wires of all 9 references to `select_candidate`.
3. Each mechanism card must declare its twin and its simplest alternative before the fit. Joint designs must also declare the separate design and both sequential orders. G3 needs a SEX-blind control path.
4. Grid entries in the `paired()` format are needed against the matched controls, the 9 references and the bracketing control points, with `cell_se_mc` for randomized releases.
5. `nrm.evaluate` should provide a `total|nll` paired cost. Without it, the conservative sum-of-SEs bound applies.
