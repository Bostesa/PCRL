# Coordinator declarations V1: G2/G3 mappings, falsifier readings, selection inputs, randomized grid cells

Written and committed at about 15:34Z on 2026-10-20 (time from `date -u` at commit).

## Timing disclosure (read first)

- **The order of events.**
  1. The statistical reviewer drafted D1-D6 at 15:31Z without reading any outcome (roles/statistical_reviewer/CARD_CHECK_1531.md).
     It asked that they be recorded before the first MODEL_SELECT read of a protected or control release.
  2. The coordinator had already read family B's MODEL_SELECT utility and screening results (FB_*: nrm.evaluate 15:27Z, nrm.screen
     15:27-15:28Z) before that request arrived.
- **What was done about it.**
  - The reviewer's D1-D6 are adopted **verbatim**, with no change to any item the reviewer proposed. Where the reviewer offered a
    choice, the coordinator takes the reviewer's proposed option.
  - The additions below (FB2, robustness instances, extension triggers) concern only configurations not yet fitted or read.
  - The FB_* results stay as read. They are adaptive development evidence of a family under revision. No FB_* configuration is
    claimable: its twin fails the original limits, so G1 fails.
  - PROTOCOL_DEVIATIONS item 6 records this.

## Adopted from the statistical reviewer (D1-D6, verbatim choices)

- **D1 (G2 mapping; G2 holds only if it holds against the twin and every declared alternative).**
  - FB_SEX_S2: twin FB_TWIN; alternatives FB_MINF_S2 and FB_ISO_S2.
  - FB_SEX_S3: twin FB_TWIN; alternative FB_MINF_S3.
  - PRC_S2: twin PRC_TWIN; alternatives SHUF_S2 and ORD_S2.
  - PRC_S1: twin PRC_TWIN; alternative ORD_S1. PRC_S3: twin PRC_TWIN; alternative ORD_S3.
  - Controls that may win: FB_MINF_S2 against FB_ISO_S2, and ORD_S2 against SHUF_S2.
  - Mechanism-attribution comparisons, reported beside G2 and never entering it: FB ID, MEAN, POSTHOC; FA HQ_S2.
  - Any configuration without a declared alternative is NO_MATCHED_CONTROLS and not selectable.
- **D2 (G3 paths, twin as the s = 0 anchor).**
  - FB: FB_TWIN, FB_MINF_S2, FB_MINF_S3; ISO reported beside.
  - FA: PRC_TWIN, ORD_S1-S3, plus the contingent ORD bracketing entry (spend-only trigger).
  - G3 needs DOMINATES_CONTROL_FRONTIER against EACH declared SEX-blind path. For FB that is the MINF path and the CTL_REM path
    (plus CTL_BLIND_OCC if fitted).
  - For FB (v1), no MINF extension is declared, so no MINF strength may be added to FB.
- **D3.**
  - G3 is decided by `nrm.rules.frontier_comparison` alone.
  - The challenger's `nrm.ctl.frontier` verdicts are a declared diagnostic, reported beside G3 with any disagreement disclosed. They
    are NOT an extra necessary condition.
  - The diagnostic margin m = 2 x `se_agg` from `nrm.rules.aggregate` (per version: 2 x `per_version[v]['se_agg']`). A
    SCORE-selected maximum is labelled as such and never enters an NRM1 label.
- **D4 (falsifier readings).**
  - FB F1 holds unless `frontier_comparison` gives DOMINATES_CONTROL_FRONTIER at BOTH strengths of the path.
  - FA reader classes: "smooth" = OUT_LR, LOG_LR, OUT_MLP, LOG_MLP, MLP64, MLP6432; "LOOKUP-type" = LOOKUP, CATI01, CATI1. Each is
    read as per-recipe paired SCORE differences favouring PRC at the point.
  - These qualify, never replace, the NRM1 labels.
- **D5.**
  - G6 = NOT_ASSESSED_MISSING_CONTROLS for joint designs (no separate design or sequential orders), or
    NOT_APPLICABLE_ONE_OUTPUT_CHANGED. `outputs_changed` is set mechanically from wire hashes against the family twin.
  - G7: FA = PRIOR_ART_FOUND (privacy funnel / IBSI; FARE); FB = NOT_CLAIMED.
  - FB claims carry "one released instance per person; repeated-query protection not claimed".
- **D6 (selection inputs).**
  - Controls are read against all 9 references: every family member is a grid member with all 9 references as comparators.
  - `entry` = the row index of the version-0 evaluation in runs/EVAL_LEDGER.jsonl.
  - `n_fitted_components` = model + calibrator + critic fits from the release records, summed over both versions.
  - `grid_valid` = `nrm.rules.grid_validity` with minimum set = the configuration plus its declared matched controls.
  - The wires of all 9 references are passed to `select_candidate`.
  - G3 uses `nrm.evaluate` paired costs, including `total|nll` and the 8 secondary metrics, plus pair-view grid entries of each
    candidate against every path point, its twin included.

## Randomized grid cells (the reviewer's rule)

- Each cell reads its registered instance: R1 instance 0, R2 instance 1, shared by k0 and k1.
- Full-bank Monte Carlo replicate members (draw offsets +2 and +4) are evaluated for the S2 SEX and MINF members (and for FB2 at the
  strengths named below). They run through the full 16-recipe procedure, with the same splits, seeds and bootstrap weights and the same
  comparators.
- Cell inputs to `R.mc_cell` are {values: [x_r, x_{r+2}, x_{r+4}], ses: [...]}. Paired draws match offset with offset; a member
  without replicates pairs with its single value.
- **Robustness instances (declared now).** R3 reads instance 6 and R4 reads instance 7, so they never reuse R1's or R2's replicate
  instances 2-5. k2 and k3 read their block's instance.

## Family B revision FB2 (pending the reviewer's approval; declared before any FB2 fit or read)

- **G2.**
  - FB2_SEX_S3: twin FB2_BASE; alternative FB2_MINF_S3.
  - FB2_SEX_S5: twin FB2_BASE; alternative FB2_MINF_S5.
  - Controls that may win: FB2_MINF_S3 against FB2_SEX_S3's twin only (NO_MATCHED_CONTROLS otherwise).
- **G3 paths.**
  - MINF path: FB2_BASE, FB2_MINF_S3, FB2_MINF_S5.
  - CTL_REM path: FB2_BASE plus the challenger's remainder-flattening levels on FB2_BASE, chosen by spend only to bracket the
    realized spend of FB2_SEX_S3 and FB2_SEX_S5.
- **MINF extension trigger (spend only).** If FB2_SEX_S5's realized pooled total-NLL cost exceeds that of every MINF point, add
  FB2_MINF_S6 (s = 0.26) as a logged adaptive step (+2 configurations). No other MINF strength may be added.
- **F1** holds unless DOMINATES_CONTROL_FRONTIER at both S3 and S5.
- **Monte Carlo replicates** (offsets +2, +4): FB2_SEX_S5 and FB2_MINF_S5.

## Family A capacity decision

See roles/coordinator/FA_CAPACITY_DECISION.md.
