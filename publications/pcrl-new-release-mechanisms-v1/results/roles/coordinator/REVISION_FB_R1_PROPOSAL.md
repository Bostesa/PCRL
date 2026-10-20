# Family B, revision 1 (proposed by the coordinator at 15:31Z; needs the statistical reviewer's approval before any fit)

## Evidence (all on MODEL_SELECT through nrm.evaluate / nrm.screen; adaptive development evidence)

1. **The task-only learner fails the original limits (P1 false).**
   - FB_TWIN v1 fails the men's occupation Brier guard: 0.00854 against 0.005. Its occupation NLL excess vs U0 is +0.0067.
   - FB_TWIN v0 passes only at the edge: men's guard 0.00460, maximum value/cap 0.92.
   - Every protected configuration inherits the failure.
   - The card's rule applies: "if the task-only architecture fails utility, improve or replace it before drawing conclusions about
     protection".
2. **Joint adaptation adds nothing (F3).**
   - Screening pair AUC (max of 4 recipes, screening halves): FB_POSTHOC_S2 0.6292 / 0.6383 vs FB_SEX_S2 0.6325 / 0.6391.
   - The channel on the frozen twin learner is at least as protective as the channel-trained learner, at similar utility cost.
3. **A targeting signal at matched second-order cost.**
   - Screening pair AUC: FB_SEX_S2 0.6325 / 0.6391; FB_MINF_S2 0.6389 / 0.6413; FB_ISO_S2 0.6391 / 0.6413; FB_TWIN 0.6397 / 0.6460.
   - Screening pair AUC: FB_SEX_S3 0.6284 / 0.6361; FB_MINF_S3 0.6391 / 0.6398.
   - The identity channel of the S2 learner (FB_SEX_S2_ID) reads 0.6403 / 0.6454, the same as the twin.

## Proposed revision FB2: the post-hoc SEX-targeted coupled channel on the strongest admitted task-only base

- **Base (the twin), TWA_v.** HC_STRONG S<v> income (the admitted task-only strong-income twin, `ohn.devrun.anchor_income('TW', v, role)`)
  plus A1_PO_16 S<v> occupation (`ohn.devrun.load_occupation('A', 'A1_PO_16', v)`), both frozen. Measured utility:
  - income NLL excess vs U0: −0.0084 / −0.0084;
  - occupation NLL excess vs U0: −0.0161 / −0.0128;
  - men's guard: −0.0039 / −0.0023.

  This is far more slack than any learner of this family. TW+T_TREE and TW+TASK_REFIT pass all clauses, which establishes
  HC_STRONG's income.
- **Operation.** The family's frozen channel law, applied post hoc to the base's logits:
  - Helmert coordinates, per-person Fisher, the SEX direction d_i = (F_i + s²Σ_w⁻¹)⁻¹Σ_w⁻¹δ, unit Fisher norm, the 1.5 cap;
  - one shared ε per person per instance, R = 8, the same key scheme;
  - per-task CALIBRATION temperatures fitted on emitted draws;
  - geometry δ, Σ_w from the base's MODEL_FIT logits. These are in-sample for both base models, and SEX-labelled exposure, disclosed.

  The channel is not trained through; F3 retired that ingredient. The changes are the base and the strength range.
- **Configurations** (each in both versions; 10 in total):

  | Configuration | Content |
  |---|---|
  | FB2_BASE | the twin (identity channel); deterministic |
  | FB2_SEX_S3 | s = 0.12 |
  | FB2_SEX_S5 | s = 0.21, second-order linked cost about 0.022 nats, within the base's slack |
  | FB2_MINF_S3 | SEX-blind control, same form |
  | FB2_MINF_S5 | SEX-blind control, same form |

  Decisive-phase additions, only if SEX beats MINF: FB2_SEX_S?_MEAN, the deterministic mean-vector construction at the winning
  strength (+2).
- **Falsification (unchanged in form).**
  - F1: at matched realized MODEL_SELECT cost, SEX's linked recovery is not below MINF's at S3 and S5. The family then closes.
  - F2, decisive phase: the MEAN construction reaches SEX's recovery.
  - Utility: a FB2 configuration that fails the original limits is reported and never claimed.
- **Budget after this revision.**
  - Family B: 18 used + 10 = 28.
  - Family A: 18 + 4 contingent.
  - CTL_REM: 2.
  - Remainder for decisive ablations and training perturbations: about 8 of 60.
- **No claims of joint training, coupling or joint design.**

This revision replaces the card's predeclared R1 (a wider MLP). The evidence favours a stronger frozen base: an MLP matched to U0 has
about zero margin, while the TWA base has 0.013-0.026 nats.
