# Exact-reference laws and plan (campaign `pcrl_finite_solver_adequacy_v1`, role exact_reference)

Written from 02:41Z on 2026-10-21 (`date -u`). Committed BEFORE any numerical experiment on these laws. Machine-readable twin:
`laws.json` (same folder). Everything here is an invented finite law: no real data, no labels.

## 1. Laws

Each law is a joint law P(X, S, Y_inc, Y_occ) with m support states. P(x) = n/N with integer n, so each law maps exactly to N
replicated rows. P(S=1|x) = pi is a multiple of 0.01; P(Y|x,S) entries are multiples of 0.01. Derived statistics are exact rationals
(denominator 10^4): qI = (1 - pi) P(Y_inc|x,0) + pi P(Y_inc|x,1), qO likewise, qO0 / qO1 = P(Y_occ|x,S=0/1). The code under test
receives E = {qI, qO, pi, qO0, qO1} on the N replicated rows (floats of these rationals), so its pi-weighted plug-in objective is the
population objective.

Occupation prototypes used below (first two classes vary, others 0.10 unless shown): A = (.46,.24), A2 = (.44,.26), B = (.24,.46),
M = (.36,.34), C = (.10,.10,.50,.20,.10).

| Law | m | N | What | Witness / comparator |
|---|---|---|---|---|
| A_null | 6 | 2000 | (a) S independent of X (pi = 0.40 everywhere); occupation depends on S (guard active) | none can lower recovery; zero-recovery comparator = task-natural grouping, AUC 0.5 exactly |
| B_collide | 8 | 2000 | (b) four collision pairs (identical qI, qO, qO0, qO1; different pi); the boundary pair b5 (pi .10) / b6 (pi .90) has posterior M between A and B | split: b5 to the A cell, b6 to the B cell (not h(q)-expressible) |
| C_distinct | 8 | 2000 | (c) all (qI, qO) distinct; pi correlated with task scores; c7, c8 S-dependent occupation | none hand-built; STEP 4 constrained optimum |
| D_witness | 4 | 2000 | (d) two cheap competing merges: A1+A2 (task-optimal, similar pi .60/.62) vs C1+C2 (pi .20/.70) | merge d3+d4, keep d1, d2 apart |
| E_distinct_N200 | 8 | 200 | the population law of C on 200 rows (n / 10): routing-table shrinkage 10 times stronger | none |

Full per-state values are in `laws.json`. Capacity everywhere: KI = 2 income cells, KO = 3 occupation cells.

### Hand-computed predictions for the witnesses (STEP 2 recomputes them exactly; any mismatch will be reported, not repaired)

Utility is relative to the full-posterior release (the U0 analogue). Exact Bayes pair AUC with ties as 1/2.

- **B_collide witness (split):** occupation NLL excess about 0.0042, summed Brier excess about 0.0028, accuracy loss 0.0015 (b6 decided
  as class 1 against its Bayes class 0), per-SEX occupation Brier excess about 0.0031 (S=1) and 0.0025 (S=0); income exact (no loss).
  Pair cell SEX rates 0.500, 0.500, 0.486; pair AUC about 0.5065, against about 0.5825 for the task-natural grouping {A, M} | {B} | {C}
  and about 0.5915 for the full-posterior release. FEASIBLE by these hand numbers (margins: NLL 0.006, Brier 0.0022).
- **D_witness witness:** occupation NLL excess about 0.0004, Brier about 0.0002, no accuracy loss; income NLL about 0.0003. Pair AUC
  about 0.5828, against about 0.6882 for the task-optimal merge (A1 + A2) and about 0.6907 for the full release; the income view alone
  gives about 0.5803, so 0.5803 is the floor for every partition that keeps the income cells {d1, d2} | {d3, d4}.
- **A_null:** every release has AUC exactly 0.5 in every view; no benefit is possible and none may be required.

## 2. Fixed settings (code defaults unless stated)

- Strength path lambda grid {0.25, 0.5, 1, 2, 4, 8} (`LAMBDA_GRID`); kappa = 1; shrinkage pseudo-count m_shrink = 20 (against N
  replicated rows); w_pair = w_loc = 1; path_steps 30; init_steps 50 (`route_params('PRC_S2')`, `route_params('PRC_TWIN')`).
- Twin: `twin_book(E, 2, 3, seed, rp)` for seeds 0 and 1 (the two "versions"); the protected path is `protected_path` from each twin,
  recording every grid lambda.
- Emitted tables for every reported release: the B1 analogue, exact posterior means of the final cells. The search-time codewords are
  reported separately where they differ.

## 3. Utility limits (relative to the full-posterior release, per task)

accuracy loss <= 0.010; NLL excess <= 0.010 nats; summed (over classes) Brier excess <= 0.005; per-SEX occupation summed Brier excess
<= 0.005 for S = 0 (men) and for S = 1 (women). The per-SEX excess can be negative (qO is not the per-SEX predictor). The statistical
reviewer may tighten these; tightening is applied as a re-read of the stored metrics, never a new law.

## 4. Reference classes

- R_FULL: full-posterior release (qI, qO), identical vectors merged.
- R_DEC: decision-only release: cells = argmax decisions per task, tables = posterior means within them (outside the capacity class
  when more than 3 occupation decisions occur).
- R_TWIN: the code's `twin_book` partition (seeds 0, 1) with B1 tables.
- R_WIT: the declared witness / zero-recovery comparator (B_collide, D_witness, A_null) with B1 tables.
- N_card(lambda), N_wide(lambda): at each proposal step of the old solver, the refreshed book nb = update_book(current partition) held
  fixed; every partition route(E, nb with tilt t) for t in (0, lambda] (card range) or (0, 4 lambda] (wide, outside the card). Obtained
  by enumerating every pairwise crossing of the per-state linear costs A_x(c) + t B_x(c) and evaluating each open interval; each
  partition is re-derived by running the code's `route` at an interior tilt, which must reproduce it. Exact for that step only, not for
  the alternating optimizer.
- ENUM: every labelled assignment of the m states to the KI x KO cells (2^m x 3^m), codewords = cell means (the declared update rule),
  tables = B1. A RELAXATION unless a partition is shown realizable by the runtime rule (certificate: the partition is route(E, its own
  rule book at some tilt t), or the solver reached it). Also a codeword-optimised objective lower bound (per-cell convex minimisation
  of CE + lambda kappa guard over the simplex), a relaxation lower bound for any solver state. For B_collide the h(q) sub-class
  (collision groups kept together) is reported separately.

## 5. Comparison rules

- Objective: my exact objective must equal `nrm.fa.prc.objective` on the replicated rows within 1e-9 absolute; otherwise a
  discrepancy, reported with both numbers.
- Narrow step gap = J(first-improvement proposal) - J(best in N_card); > 1e-9 means the first-improvement choice was suboptimal at
  that step. A step where the solver stopped while some partition in N_card is below J_current - 1e-9 is a missed improvement.
- Exact Bayes AUC differences below 0.001 are "no material change"; task-metric changes below 0.0005 are "no worse".
- Surrogate (pi-weighted plug-in MI on cell ids) and actual exact Bayes AUC on the released vectors (identical vectors merged) are
  reported separately.
- Never require improvement below a feasible zero-recovery comparator (pair AUC within 0.001 of 0.5). In A_null no benefit is
  possible.
- Where task scores identify the support states (C, D, E), no claim that propensity adds structural expressivity over unrestricted
  task-score mappings is made. HQ routes on twin-cell means of pi, qO0, qO1, a further restriction.

## 6. Budgets and units

- Queue units (stage `exact`, `--rss-gib 1`): FSA_EXACT_NARROW_V1 (STEPs 2-3, all laws, both seeds), FSA_EXACT_ENUM_V1 (STEP 4). A rerun
  gets a new suffix.
- CPU: at most 300 CPU-s per law for STEPs 2-3 and 300 CPU-s per law for STEP 4; role total at most 1 CPU-h including selftests.
- Selftests of my reference code run out of queue on separate invented inputs (not these laws) and are charged.
- Secondary, only if time allows: the HQ_S2 path's final states (exact metrics only, no narrow reference). STEP 5 waits for the
  coordinator.
