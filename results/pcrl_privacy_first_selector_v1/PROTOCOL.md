# Privacy-first per-context selector v1: protocol (2018 development)

Registered 2026-09-24, before any release in this study was solved and before any new outcome was observed.
Later changes go into `PROTOCOL_AMENDMENTS.md`, each with its UTC time and outcome-access status.

## 0. Registered status statement (verbatim)

"These results may meet the new predeclared numerical criteria, but remain exploratory development evidence because the assessment data informed this study's design. They cannot provide independent confirmation."

This applies to every number this study produces. It applies most of all to the outer-role numbers.

- **What motivated the study.** A post-hoc comparison (`results/pcrl_shared_context_release_v1/POSTHOC_EXPLORATORY.json`) of the outer role that the predecessor had already opened. That role is re-used here.
- **The lead.** DET_SEL4 was chosen on task, not privacy. Its AB/SEX recovery against D17 was −.0036 unweighted and −.0034 PWGTP, with task change about 0. Its intervals allow no improvement.
- **Consequence.** This study cannot confirm that lead. It can only ask whether a privacy-first choice, fixed before this study scores anything, meets new numerical criteria on the same data.

## 1. Question

Given the existing policies:

1. Can selecting one policy per context *for privacy* reduce coalition SEX recovery (AB/SEX) while keeping task loss near D17?
2. Does randomizing over the same policies add anything?

## 2. Setting (unchanged)

- **Service.** A published, immutable prediction service. It sends H_A (4 probabilities) to recipient A and H_B (2) to recipient B.
- **Extra token.** One extra token Z from a 17-token alphabet is appended for A only. It is one persistent keyed token per record.
- **Permitted task.** A may predict `same_residence`.
- **Protected attributes.** SEX and RAC1P, protected from A alone and from the A+B coalition.
- **Audited views.** A = (H_A, Z) and AB = (H_A, H_B, Z). Every attacker uses the continuous H.
- **Data.** California ACS 2018, the predecessor's household-split roles, three anchors (0, 1, 2), paired household bootstrap.
- **Status of the data.** Everything on 2018 is development evidence on repeatedly used data.

## 3. Reused objects (exact, pinned by hash; no new policies)

Everything below comes from branch `research/pcrl-shared-context-release-v1`:

- final commit `537e74a44bef44ea86882a19769f3c4eb53d410c`;
- lock commit `e5d555a1438e14def094b1cdc1b1f5d2c2d52394`;
- archives in `s3://pcrl-ux-archive-ed9d21fd/pcrl_shared_context_release_v1/units/`, version- and SHA-256-pinned in that branch's `MODEL_MANIFEST.json`.

| Object | Source (per anchor a) | Archive SHA-256 pin |
|---|---|---|
| Policy bank (M=5: D17, task_only, local_priced, coalition_priced, all_priced_x2), context rules (K=1, K=4), round-0 decoder/bank | `a{a}_bank` | MODEL_MANIFEST `unit_archives.a{a}_bank.archive_sha256` |
| **Frozen decoder** (task blocks) and **frozen attacker bank** (calibrated cut blocks with rho and floors) | `a{a}_NM4_U/closing/{TASK_BLOCKS.npz, CALIBRATED_BLOCKS.npz, CALIBRATED_BANK.json}`. This is the basis on which DET_SEL4 (the lead) was chosen. | MODEL_MANIFEST `unit_archives.a{a}_NM4_U.archive_sha256` |
| Previous task-selected deterministic control DET_SEL4 | `a{a}_DET_SEL4` | MODEL_MANIFEST `unit_archives.a{a}_DET_SEL4.archive_sha256` |
| D17 reference witness | the pinned index's historical map `D17` (`laws.historical_spec`) | index `results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json` |
| Data roles | AR roles, SHA256(`pcrl_adaptive_release_v1\|household_id`); sanitized 2018 anchors via the SC `INPUT_STAGE.json` (version `frEADbCBRCcJowSwg_znRdN8ivLnAi2R`, SHA-256 `9ddb13a3…`) | as in the predecessor |
| Attacker/audit slate, route rules, seeds (26000 + 1000·anchor + role index) | `experiments/pcrl_shared_context_release_v1/audit_panel.py`, used unchanged | code at 537e74a |
| Positive controls (AB/SEX, AB/RAC1P × 3 anchors, 6/6 detected) | the predecessor's POS receipts, pinned in its `SELECTION_LOCK.json` | lock SHA-256 `d9954841…` |

Handling rules:

- After restoring the archives, the SHA-256 of every basis file actually used is recorded in `BASIS_MANIFEST.json` before any solve.
- A restored archive whose SHA-256 differs from its pin aborts the study.

## 4. Objective (identical for every arm)

Fixed quantities, per anchor, on coefficient_split rows:

- the frozen decoder's task blocks C_v, for v ∈ {U, W};
- the frozen bank's cut blocks L_c;
- rho_c: the minimum over the bank of that role and weighting's loss at D17. It is the same rho the predecessor used.

Every arm solves:

    maximize t
    subject to  L_c(q) ≥ rho_c + t          for every AB/SEX cut c (both weightings)
                L_c(q) ≥ rho_c − 0.001      for every A/SEX, A/RAC1P, AB/RAC1P cut c (both weightings)
                C_v(q) ≤ C_v(D17) + 0.001   for v ∈ {U, W}

- **What t measures.** t is the worst bank-measured coalition-SEX improvement over D17 across both weightings and every AB/SEX attacker in the bank.
- **The D17 witness.** All-D17 is always feasible with t = 0.
- **Feasibility tolerance.** The inherited `PRIMAL_TOL`.

## 5. Arms and controls

| ID | Role | Definition |
|---|---|---|
| **D4** | primary arm D | One policy per K=4 context: B = 0, η = 1, each row of A one-hot. Solved **exactly by enumerating all 5^4 = 625 assignments** (gap 0). Ties within 1e-12 in t are broken by: lower 0.5U + 0.5W task, then fewer non-D17 contexts, then lexicographic index. |
| **R4** | primary arm R | Probabilities over the same policies per K=4 context: B = 0, η = 1, A ≥ 0 with rows summing to 1. A linear program (HiGHS), with the solver status and dual bound on t reported. |
| NM4PF | secondary | The full nested mixture (B, A, η free; K=4) under the same objective: `channel.solve_privacy_first` on this basis, one solve, no alternation |
| D17 | control | The exact reference |
| TASK_SEL4 | control ("task-only richer selector") | One policy per K=4 context minimizing 0.5U + 0.5W task over the 625 assignments, with **no** privacy constraint. Ties go to fewer non-D17 contexts, then lexicographic index. |
| DET_SEL4 | control (previous task-selected deterministic control) | The archived predecessor release: the lowest task among bank-feasible assignments on this same basis. It is re-derived here as a check, and an assignment mismatch aborts the study. |
| D1, R1 | K=1 controls | D4 and R4 with K=1 (one global context). The K=1 blocks are the K=4 A blocks summed over contexts, which is exact, so D1 ⊂ D4 and R1 ⊂ R4 on one fixed basis. D1 enumerates 5 assignments. |

Structural consequences on the fixed bank:

- t(R4) ≥ t(D4) ≥ t(D1) ≥ 0, and t(R4) ≥ t(R1) ≥ t(D1).
- t(R4) − t(D4) is the value of randomization *on the fixed bank*. Only fresh attackers can say whether any of it is real.

## 6. Selection and audit (inner roles only)

- **Solving.** Every arm is solved only on the coefficient_split role. No outer row, inner_check label or outer label enters any solve.
- **Audit.** It is the predecessor's slate, unchanged (`audit_panel.run_panel`):
  - fresh attackers and a utility probe for each release, fitted on audit_fit;
  - routes (own release and ignore-channel H-only ancestors) chosen on inner_selection;
  - scores on inner_check;
  - common seeds;
  - identical laws audited once and recorded in the alias ledger;
  - the inherited hist_gb min-leaf rule, identical for every release. Stochastic laws get a coarser exact-tree leaf, which favours R (disclosed).
- **Inner report.** Before the lock the study reports:
  - three-anchor inner_check contrasts versus D17;
  - fixed-bank t, task and slacks;
  - within-T32 decision variation.
- **These inner numbers gate nothing.** Every registered arm and control is scored on the outer role regardless.
- **Decision-variation report,** for every release on coefficient_split, inner_selection and inner_check (and on outer rows after the unlock):
  - weighted and unweighted mean TV to D17;
  - weighted and unweighted within-T32 V (TV to the person's T32-state mean law);
  - maximum pairwise within-state TV;
  - the number of states with variation;
  - affected people and affected unique households, for both TV to D17 and within-state variation.

## 7. Comparison family and endpoints (generated by code, counted and hashed into the lock before scoring)

**Roles** (5): utility:A/same_residence, attack:A/SEX, attack:A/RAC1P, attack:AB/SEX, attack:AB/RAC1P. **Weightings** (2): unweighted (U) and PWGTP.

Signs:

- task = CE_Y(candidate) − CE_Y(comparator);
- recovery = CE_S(comparator) − CE_S(candidate).

Negative is better for both.

Clauses use the P-route thresholds on the simultaneous upper bound:

- task ≤ +.001;
- AB/SEX ≤ −.002 (target);
- A/SEX, A/RAC1P, AB/RAC1P ≤ +.001 (guards).

**Primary family (30 endpoints):**

| Slot | Candidate | Comparator | Endpoints |
|---|---|---|---|
| D | D4 | D17 | 10 |
| R | R4 | D17 | 10 |
| R_vs_D | R4 | D4 | 10 |

**Secondary family** (descriptive, separately corrected, never part of a pass), up to 110 endpoints:

| Candidate | Comparators |
|---|---|
| NM4PF | D17, D4, R4 |
| D4 | DET_SEL4, TASK_SEL4, D1 |
| R4 | R1 |
| D1, R1, DET_SEL4, TASK_SEL4 | D17 |

**Capability family:** D4 and R4 versus H on task, both weightings; the threshold is CE_cand − CE_H ≤ −.01. That is 4 endpoints.

**Generation rules.** The predecessor generator is reused:

- Comparators that are the same law on all three anchors are merged within a slot.
- Endpoints that are identical on every anchor (the same law, or the same H-only route) are labelled `exact_zero_same_route`.
- The generated counts set Bonferroni: z = norm.isf(.05 / (2m)), per family.

**Uncertainty:**

- the predecessor's procedure: paired household bootstrap, 10,000 draws, one multinomial over the union of household IDs across the three anchors;
- the estimator is the equal mean of per-anchor weighted ratios;
- two-sided normal-approximation Bonferroni bounds within each family;
- **bootstrap seed 20260926** (new, registered here);
- the intervals condition on the fitted, selected objects. They do not cover the adaptive research history or the ACS survey design.

## 8. Decision rules (fixed now)

| Label | Rule |
|---|---|
| ARM_D_MEETS_CRITERIA | All 10 D-vs-D17 primary clauses pass |
| ARM_R_MEETS_CRITERIA | All 10 R-vs-D17 primary clauses pass |
| RANDOMIZATION_ADDS | All 10 R-vs-D primary clauses pass |
| LEAD_REPRODUCED (point estimates; triggers §10) | For D4 vs D17, all of: AB/SEX ≤ −.002 in both weightings; task ≤ +.001 in both weightings; each of the other six recovery contrasts ≤ +.001 |

If any MEETS_CRITERIA label is earned, it is reported together with the §0 statement, verbatim, and nowhere without it. Otherwise the status is `NOT_ESTABLISHED`.

## 9. Registered predictions (subjective probabilities, before any solve)

These record calibration. Verdicts use results, not these.

| # | Prediction | P |
|---|---|---|
| 1 | **D beats D17 on AB/SEX by at least .002 within the cap.** Outer point estimates: AB/SEX recovery contrast ≤ −.002 in both weightings, and task contrast ≤ +.001 in both weightings. | .40 |
| 1b | ARM_D_MEETS_CRITERIA (all 10 simultaneous clauses) | .04 |
| 2 | **R beats D.** Outer R-vs-D AB/SEX point estimate ≤ −.001 in both weightings, and R-vs-D17 task point estimate ≤ +.001 in both weightings. | .15 |
| 2b | RANDOMIZATION_ADDS (all 10 R-vs-D simultaneous clauses) | .01 |
| 3 | **D's within-32-state decisions differ from D17.** D4's coefficient_split law has within-T32 variation (≥ 1 state with variation, V_W > 0) on at least one anchor. | .90 |
| 3b | The same holds on all three anchors | .55 |
| 4 | D4 differs from DET_SEL4 on at least one anchor | .85 |
| 5 | R4 is one-hot (deterministic) on every anchor | .15 |
| 6 | Fixed-bank t(R4) − t(D4) > 1e-9 on at least one anchor | .85 |
| 7 | LEAD_REPRODUCED | .30 |
| 8 | NM4PF's outer AB/SEX point estimate beats D4's by ≥ .001 in both weightings | .15 |

## 10. Conditional follow-up (label-free)

Condition: LEAD_REPRODUCED holds.

Action: finish the Texas 2018 sample-size (power) table of the predecessor's `CONFIRMATION_PLAN.md` in this study's results directory.

- It covers both weightings and every guard.
- It uses this study's outer bootstrap SEs and registered planning values.
- No Texas file is downloaded or opened, and no Texas label is read.
- Acquiring Texas data remains a separate, later decision.

## 11. Outer access, verification, resources

**Outer access.**

- The outer role is opened exactly once, for this study's whole panel, including the controls.
- Order of steps:
  1. `SELECTION_LOCK.json` is committed and pushed, and the remote SHA is verified.
  2. A write-once unlock receipt is written.
  3. The predecessor's pinned original objects are restored.
- Scoring reuses the frozen inner routes, with no refit and no reselection.

**Independent verification.** A separate verifier writes separate code that:

- re-solves D (enumeration) and R (LP, a different formulation) from the basis files;
- recomputes every reported point estimate and decision from the outer contribution archives, with its own bootstrap;
- re-checks the custody chain (lock commit before the outer unlock, before any outer score).

**Resources.**

- One c7i.8xlarge in us-east-1, tagged `Project=pcrl`, `Study=pcrl_privacy_first_selector_v1`. It has a watchdog and zero-ingress SSM access.
- Ceilings: US$50 or 20 instance-hours, whichever comes first, and a calendar ceiling of 2026-09-25T19:30Z.
- Only resources carrying this tag are terminated. Nothing else is touched.

**Scope.** The manuscript branch and submission metadata are not modified.
