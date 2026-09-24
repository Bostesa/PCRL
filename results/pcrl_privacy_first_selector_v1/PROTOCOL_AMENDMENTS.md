# Protocol amendments

Each entry gives the UTC time, the outcome-access status, the trigger and the change. Margins, endpoints, families, arms and decision rules are unchanged by every entry here.

## A1 — 2026-09-24 ~23:55Z (after the first fixed-bank solve; before any audit, inner_check score or outer access)

- **Outcome access.** Only the fixed-bank solve on coefficient_split: t, task and assignments for each arm, computed from the frozen basis blocks. No attacker, probe or audit was fitted, no inner_check or outer row was scored, and no label was read by this study.
- **Trigger.** PROTOCOL §5 gave D4 and D1 an explicit tie rule: among assignments within 1e-12 of the maximum t, take the lower 0.5U + 0.5W task. It gave the LP arms (R4, R1, NM4PF) none. HiGHS then returns an arbitrary optimal vertex.
  - On anchor 1 the maximum t is 0 for every arm.
  - R4 returned a mixed law at t = 0 whose task was about +.001 above D17.
  - NM4PF did the same.
  - That law buys no bank-measured privacy at all, so the D-vs-R comparison was asymmetric for a reason unrelated to randomization.
- **Change.** The LP arms get the same tie rule as D, applied as a lexicographic second stage on the same constraints:
  - minimize 0.5(task_U + task_W) subject to t ≥ t*, where t* is the stage-1 optimum. The only tolerance is HiGHS's primal feasibility of 1e-9. A 1e-9 relaxation was tried first on synthetic data, and it let t fall below 0 at t* = 0, so it was rejected;
  - NM4PF is re-solved with this study's own LP on the full nested parameterisation (B, A, η), and its stage-1 t* must equal the SC `channel.solve_privacy_first` τ within 1e-7.
- **Unchanged.** Every constraint, the objective t, the arms and the controls. D4, D1, TASK_SEL4 and DET_SEL4 are deterministic enumerations and are unaffected.
- **Record keeping.** The first solve outputs are kept, not deleted:
  - the host moves them to `private/superseded_solve_A1/`;
  - `SOLVE_REPORTS_SUPERSEDED_A1.json` is committed next to the new `SOLVE_REPORTS.json`.
- **Direction.** The stage-1 t is unchanged by construction. The second stage can only lower task, so it removes a disadvantage of the LP arms and favours none of D, D17 or the controls.

## A2 — 2026-09-24 ~23:58Z (Texas power-table planning values; before any audit result, inner_check score or outer access)

- **Outcome access.** As in A1. The inner audits were running, but no audit output had been read.
- **Purpose.** PROTOCOL §10 required "registered planning values" for the conditional Texas 2018 table but did not state them. They are fixed here, before any number they could depend on exists.
- **Family.** The D4-vs-D17 conjunction only: task and the four recovery roles in both weightings, so M = 10 endpoints and K = 10 conjunctive components.
- **Error rates.** α = .05 and β = .20, giving z = z_{1−α/M} + z_{1−β/K} = z_.995 + z_.98 ≈ 2.576 + 2.054 = 4.630. The transport SD inflation is κ = 1.25.
- **Per-endpoint SD.** s_e = SE_boot(e) · √H_2018.
  - SE_boot(e) is this study's outer bootstrap SE for D4 vs D17.
  - H_2018 = 2,968: the union of outer-assessment households across the three anchors (`DATA_ROLE_COUNTS.json` `global_households.outer_assessment`), which is the bootstrap's resampling unit.
- **Planning effects and formulas** (CONFIRMATION_PLAN §6):
  - AB/SEX target: Δ* = −.0035 (the size of the prior lead). The margin is −.002, so formula (c) applies: H = ⌈κ² z² s² / (.0015)²⌉.
  - Task and the three guards: Δ* = 0 against the +.001 non-inferiority margin, formula (d): H = ⌈κ² z² s² / (.001)²⌉.
  - Required final-pool households = the maximum over the 10 endpoints. The admitted total = that / 0.30.
- **Sensitivity rows** (descriptive, labelled as chosen after the outer result):
  - Δ*_ABSEX equal to D4's outer point estimate;
  - κ = 1.0.
- **Scope.** No Texas file is read. The table only restates what the 2018 development SEs imply.
