# Stage 5 admission note

Written 2026-10-03, after lock amendment L1 and before any stage-5 fit.

**Synthetic checks** (official FARE 89cb1b66 build in its isolated environment):
- A 6-class task with 4 sensitive groups fits.
- The portable encoder equals the official encoder.
- The certificate path runs.

**γ ceiling.** The official FairGini criterion keeps the binary constant 0.5:
FairGini = (1−γ)·Gini_y + γ·(0.5 − Gini_s). When Gini_s exceeds 0.5, which happens with more than 2 groups, the root
impurity is ≤ 0 for γ ≥ Gini_y / (Gini_y + Gini_s − 0.5), and the official builder returns a single leaf.

Ceilings, computed from defense_fit **label frequencies only**:
- Adult employment/age_group: **0.864**;
- HMDA pricing/race: 0.989;
- HMDA fair_lending/race: 0.982;
- every other candidate cell: no ceiling (1.0).

**Consequence.** If the screen selects employment/age_group, grid settings 4–6 (γ 0.9, 0.95, 0.95) produce 1-cell trees
by construction. These are valid constant controls, but they fail the ≥ 80 % gain-retention rule, so they are
infeasible under the registered nominee rule. The grid is **not** changed: the method is applied as registered, and
this property is reported, not patched.

**Not an admission failure.** The method's objective is defined and runs for these class counts. Its behaviour at high
γ is a documented property of the published criterion.
