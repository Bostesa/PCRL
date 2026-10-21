# FA_FSA_v1: solver adequacy of the propensity-routed finite-cell objective (audit, replay, source check, nested comparison)

Campaign `pcrl_finite_solver_adequacy_v1`. Written from 2026-10-21T02:39Z (`date -u`), before any real-data computation of this
campaign and before the registration receipt. Governs units `FSA_REPLAY_*`, `FSA_AUDIT_*`, `FSA_SOURCE_*` and `FSA_NESTED_*`.

**Configurations.** None. This card fits no release configuration (0 of the 32). A pilot, if the advancement condition of
`roles/statistical_reviewer/RULES.md` is met, needs its own card committed before its fits.

## 1. What is held fixed (imported unchanged)

The following come from `nrm/fa/prc.py` (publication 682068d52), which `frc/fa/route.py` (publication 04ac9161f) imports and which
card FA_PRC_v1 section 1 specifies:
- the objective (`objective`);
- the routing family (`route`: joint argmin of CE + t·[dependence on the shrunk tables + κ·guard], tilt t ≤ λ, on the
  restricted embedding for HQ);
- the codeword and table refresh (`update_book`) and the twin (`twin_book`);
- the strength continuation over the registered grid {0.25, 0.5, 1, 2, 4, 8}, including the twin reset and the 30 path steps;
- the B1 emitted tables (out-of-fold teacher means of each configuration's own partition, `frc.fa.route.tables`);
- the teachers, the capacity (16 x 64), the outer folds (`NRM_FA_ncv_S<v>`, 5) and T_fixed (the P1 records).

## 2. Solver variants (Lloyd step only; at most two distinct variants in this campaign)

- **original.** `nrm.fa.prc.lloyd`: tilts λ, λ/2, …, λ/64, accepting the first that lowers the exact objective.
  `fsa.fa.solver.lloyd_original_traced` is a statement-for-statement copy that records each step; its selftest shows identity with
  `lloyd`, and the audit shows it reproduces the saved frc partitions.
- **SV1 (variant 1): exact tilt line search with best improvement.**
  - For the refreshed book, the route cost of every person and joint cell is A + t·D, linear in t. The deployable partitions
    route(E, book, t), t ∈ [0, λ], are therefore piecewise constant.
  - Every person's breakpoints are enumerated (lower envelope), and the exact objective is evaluated incrementally on every interval.
    The interval with the lowest objective is chosen (ties: smallest tilt).
  - The step is accepted only if `nrm.fa.prc.objective`, on the partition that `nrm.fa.prc.route` itself returns at the interval's
    midpoint, is lower than the current objective by more than 1e-12.
  - Proposal resolution is decoupled from λ. Zero, the previous tilt and every boundary of the declared class (t ≤ λ, the card's own
    bound) are candidates.
  - Every accepted state is rule-consistent and the accepted exact objective decreases strictly. This is not an optimality
    certificate: the line search is exact for one proposal step (fixed refreshed book), not for the alternating optimizer.
- **SV1_first (equal-effort control, not a variant).** The SV1 candidate set with the original selection rule: intervals scanned from
  the largest tilt down, accepting the first improving one. It isolates selection (best vs first) from resolution (seven tilts vs
  every breakpoint) at equal candidate evaluations. CPU time and route evaluations are recorded for every chain.
- **A second variant (SV2)** is not defined here. If the audit justifies one (for example a guard-consistent prototype refresh), it
  gets a dated amendment card with its own control, committed before its first fit.

PRC and HQ get the same variant, the same initialization (the B1 twin TWB_C64) and the same effort rules. SHUF is not rerun: the
previous campaign replaced it by UNT as the untargeted control.

## 3. Stages, data and exposure

| Stage | Inputs | Labels | Output |
|---|---|---|---|
| replay | MODEL_FIT features; income, occupation, SEX via `fsa.common.load` (operation `fit`, this card) | as in the original P1: SEX as a training label of the propensity and SEX-conditional occupation teachers and as the held-out LOOKUP evaluation label | `replay/P1_REPLAY_V<v>.json`; per-person held-out outputs (private) |
| audit | MODEL_FIT features (label-free); the cached out-of-fold statistics of NRM1 (task teachers) and P1 (SEX-side teachers), the saved frc models; all pinned | none read here; the cached statistics are derived from MODEL_FIT labels | `audit/OBJECTIVE_AUDIT_V<v>.json` |
| source | as audit | none read here (label-derived statistics) | `source/SOURCE_CHECK_V<v>.json`; partitions (private) |
| nested | as replay | as replay | `nested/NESTED_SOLVERS_V<v>.json`; per-person held-out outputs (private) |

- MODEL_SELECT, CALIBRATION and every closed role are not read by any stage of this card.
- The audit and source stages refuse to run before the verified registration receipt and without this committed card.
- The replay calls `frc.fa.route.nested_c` unchanged, with the recorded plan, construction, capacity, T_fixed and folds.
  - The nested stage runs the original and SV1 chains of PRC and HQ in the SAME outer folds and teacher fits. Its original-solver
    entries must equal the replay's per-person outputs.

## 4. What the stages measure (questions A and B on MODEL_FIT; question C by PRC against HQ under the same solver)

- **Audit.**
  - The original chain's accepted objective trace (strict decrease) and the effect of the refresh step on the unchanged partition
    (the cell-mean codewords ignore the guard).
  - The routing proposal score against the acceptance objective along the tilt line of the final refreshed book (shrunk
    linearized dependence against unshrunk plug-in information plus guard).
  - The emitted B1 tables against the search prototypes (task cross-entropy, guard, largest difference).
  - In-sample against out-of-fold routing of MODEL_FIT rows (train/deploy consistency).
- **Source.** Original, SV1 and SV1_first objectives at every registered strength, together with:
  - partitions and re-routing;
  - fitted π-weighted information;
  - the training cross-entropy of the emitted B1 tables;
  - effort.
- **Nested.** For both solvers, at every strength: held-out task spend over the twin (household-clustered SE) and held-out LOOKUP AUCs
  (pair, income, occupation).

The statistical reviewer's advancement condition (`roles/statistical_reviewer/RULES.md`) decides whether a pilot follows. Nothing in
this card is a decision rule.
