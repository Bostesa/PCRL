# Statistical reviewer brief (coordinator; 2026-10-24, 04:27Z)

Campaign `pcrl_strong_anchor_release_v1` (package `sar/`, a mechanical rename of `ipr/`). Read `roles/coordinator/DESIGN_BRIEF.md`,
`anchors/ANCHOR_SELECTION.md`, `anchors/ANCHOR_TABLE.md`, and the inherited rules you adapt:
`results/pcrl_input_and_pair_release_v1/roles/statistical_reviewer/RULES.md` / `RULES.json`, `ipr/rules.py` (now copied as
`sar/rules.py` with names changed only) and `nrm/rules.py` (the aggregate, screens, frontier comparison it builds on).

You write, before any new recovery result exists, the frozen rules of this campaign (schema `SAR1-rules-v1`):
`results/pcrl_strong_anchor_release_v1/roles/statistical_reviewer/RULES.md` and `RULES.json`, the implementation `sar/rules.py`
(adapt it; keep `nrm.rules` imported unchanged) and `sar/test_rules.py` (invented-data tests, all passing). Then a short report and
hand-back. The rules must specify the sign conventions, household resampling, uncertainty reporting, comparison hierarchy and
adaptive-search limitations.

## What changed from IPR1 (the comparison structure)

The recent campaigns compared a candidate with an equally trained task-only twin plus SBC / SPEND / RO control paths. This campaign
starts from strong frozen releases instead. For each serious release (a family's candidate CAND, chosen by its committed card's
MODEL_FIT-only rule; revisions and a replacement family are read under the same rules with the ADAPTIVE_ / REVISED_ prefix):

| Comparator | What | Gating role |
|---|---|---|
| START | the family's unchanged starting release (FA: SUCC; FB: NEURAL_RAW), frozen, both versions | the component gain is measured against it |
| CONT | an equally exposed task-only or zero-protection continuation (same learners, data, folds, seeds, iterations and selection effort, protection term off), when the card declares it needed for attribution | gating when declared |
| CTRL path | the simplest untargeted operation matched to the mechanism, on a strength grid, with two saved points chosen on MODEL_FIT before any MODEL_SELECT read so that they bracket CAND's MODEL_FIT paired total-NLL spend over START (the largest grid point with spend <= CAND's and the smallest with spend >= CAND's + 2 SE); read on MODEL_SELECT with the inherited spend-bracketing frontier rule (`nrm.rules.frontier_comparison`); a missing bracket reads NOT_SEPARATED with that reason; residual mismatch reported | gating |
| References | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (frozen; re-read on this campaign's grid with the same reader fitting and selection opportunity); U0, Ucal as utility diagnostics; DDR_PTC_R1 beside | D3 / D4 |

Decisions to define (keep them separate; the brief requires three separately reported outcomes: (1) any reliable component gain,
(2) performance against the strong references under the original conditions, (3) whether the original material target is
established):
- **D1 task feasibility**: G1 = all 28 original conditions at the point (12 task clauses + 2 within-SEX occupation guards per
  version; caps accuracy loss 0.010, NLL excess 0.010, summed Brier excess 0.005 vs U0 and Ucal, guards 0.005 vs U0). Report point
  eligibility separately from the confidence conjunction (each condition PASS / UNRESOLVED / VIOLATION from household-clustered SEs;
  an unresolved condition is neither a pass nor a violation), and from the inherited two-SE reserve heuristic (reported, never gating,
  never redefining eligibility).
- **UM usefulness margin** (meaningful for FA, defined for every reading): the binding condition of START (FA: the men's occupation
  Brier guard) and the whole-release reserve, as paired CAND - START differences with household-clustered SEs; MARGIN_RECOVERED needs
  a resolved improvement of the binding condition with G1 kept and the recovery profile retained (pair and both single views not
  worse than START beyond the original 0.010 recipient allowance, per version at the aggregate point, and no resolved pair worsening).
  A margin gain alone is preparatory evidence, never a protection or competitive claim.
- **D2 component gain**: against START, CONT (if declared) and the CTRL frontier, each with the pair benefit (comparator AUC minus
  candidate AUC, positive favours the candidate) resolved by the inherited aggregate screen (mean - 2 se_agg > 0 and both version
  means > 0) and the recipient gate (each single view not more recoverable than the comparator by more than 0.010 per version at the
  aggregate point). Name partial outcomes. State what a CTRL comparison without a bracket means.
- **D2R robustness**: at least one genuinely different household reader split (R3, R4: new household assignments of the same
  MODEL_SELECT people) plus new reader seeds for promising comparisons; ROBUSTNESS_CONFIRMED only when confirmed on the robustness
  cells and on all cells with genuinely effective refits; a sign change must be carried into the headline; no retuning on these cells.
- **D3 strong references**: per reference, resolved pair benefit with the recipient gates; SUPERIOR_TO_ALL only if every one passes;
  list the ones passed and failed.
- **D4 material target**: a resolved 0.020 pair benefit against NEURAL_RAW and against H_LJ, with the recipient gates.
- **D5 joint design**: only if a card claims joint design (local twin and both sequential orders built); otherwise
  NOT_CLAIMED / NOT_APPLICABLE.
- **D6 novelty**: NOT_CLAIMED unless separately reviewed.
- An overall outcome class per reading that cannot rename a partial result as the desired method.

## Grid and uncertainty (keep the inherited machinery)

SAR1 reader grid inside MODEL_SELECT: repetitions R1, R2 registered and R3, R4 robustness, households ordered by
SHA-256('SAR1_devgrid_<R>|<household>') into FIT 0-4 / VALIDATION 5-6 / SCORE 7-9; reader seeds MODEL_SEED_BASE + 1000 + 10 v + k
(k0, k1 registered; k2, k3 robustness); every earlier grid's seed offsets (300, 400, 500, 600, 700, 800, 900) forbidden; household
bootstrap B = 400 with seed 2026102400 + 100 v + repetition index; the admitted 16-recipe bank fitted on FIT, oriented and selected
on VALIDATION, scored on SCORE; the pair winner among pair and ignore-one-recipient readers; grid extras (non-table winner, TABLE_CV,
enhanced winner) reported beside; at most one bounded mechanism-specific reader per family, reported beside, never gating. The
inherited aggregate (block means over version x repetition; se_agg = sqrt(se_hh^2 + se_rs^2 + se_mc^2), floor 0.001). Grid validity
as IPR1 section 9 with this grid's seeds. Versions, repetitions and reader seeds are not independent replications; MODEL_SELECT is
adaptively reused by the whole lineage; intervals do not correct the adaptive search and are not confirmation. Few-valued outputs
may make reader seeds ineffective: count actual diversity (n_eff) rather than invalidating the comparison.

## Budget facts for any power or calibration statement

Up to 36 seed-specific fitted release specifications (each version counts), at least 12 for matched controls and follow-through;
two families, at most two revisions each and one replacement family; realized se_agg in the input-and-pair grid was 0.0053-0.0081
against a close twin and 0.0074-0.0139 between strong references (ANCHOR_TABLE.md section 2).

## Practicalities

You work label-free (no label file, nothing new from MODEL_SELECT). Write only under `roles/statistical_reviewer/`, `sar/rules.py`
and `sar/test_rules.py`. Use `date -u` for every time you write. Cap: 900 CPU-s. No agents, no git, no processes killed. Aim to hand
back within about 45 minutes with: the rule file paths and SHA-256, the test result, required changes to PROTOCOL.md (numbered,
REQUIRED vs ADVISORY), and what the plan file for each reading must contain (members, paths, aliases, strengths).
