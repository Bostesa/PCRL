# Registered Branch D execution, 2026-09-22

This instantiates the bounded numerical recovery already permitted by `PROTOCOL.md` and `CONFIGS.json`. It changes neither a scientific margin nor a fitted optimization problem. The primary validation comparisons have already been examined; the current-run evaluation partition remains sealed. The trigger is the retained solver-failure status, not a utility or disclosure outcome.

Exactly four primary slots returned an independently feasible coarse witness after both registered solvers failed their acceptance checks:

| Anchor | Configuration |
| --- | --- |
| 0 | Ttask_C_0.0005_a17 |
| 0 | Ttask_C_0.01_a17 |
| 0 | Ttask_L_0.002_a17 |
| 2 | Trisk_C_0.01_a17 |

The original two attempts, valid fallback, fitted artifacts and complete audits remain evidence. `NUMERICAL_DIAGNOSTIC_PRIMARY.json` records their errors. In particular, the local Ttask middle-budget fallback has a worse fitted task objective than the corresponding coalition solution; this precludes an optimum-ordering claim for those returned solutions.

One additional deterministic CLARABEL call per slot will use normalized entropy-cone arguments. For positive `m=p(s,c)`, positive homogeneity gives `rel_entr(a,b)=m*rel_entr(a/m,b/m)`. Both normalized arguments are conditional output probabilities. Exactly zero-mass cells contribute zero and may be omitted. The role-key set, original laws, cost, support equations, row simplex, nonnegativity, privacy budgets and embedded-witness objective bound remain identical. The solver settings and independent acceptance tolerances remain those in `finite.py`. No data are smoothed, no positive-mass cell is dropped, and no failed candidate is accepted by increasing a tolerance. The implementation and proof are in `numerical_recovery.py`.

Before any new solve, `NUMERICAL_RECOVERY_REGISTRATION.json` will pin all four original maps, audits, prepared encoders, coarse witnesses and retry sources. A persistent attempt directory permits at most one solve per slot. Retry outputs are staged separately. A successful numerical retry deterministically replaces its slot before final selection, irrespective of subsequent utility or attack performance, and receives the complete common audit again. An unsuccessful retry leaves the valid original witness in place with its optimization limitation. The original directories are preserved intact. There is no choice between old and new releases based on held-out scores and no additional method configuration in the search.

Installation waits for scientific writers to close, takes task-owned locks, and is prohibited after final selection freezes. All changed receipts and original versions enter the private archive; public results report nominal slots separately from fitted numerical versions and repeated audits. An accepted `optimal_inaccurate` status remains inaccurate. A solver-reported `optimal` status is not an independent dual certificate.

The new solver module passed 16 focused synthetic tests alongside 20 existing finite-channel tests and independent source review before any live retry. The full study suite at source `f119c2ca7f201f7d953ea43ac17fda3ed14746d1` passed 457 tests. These checks justify the repair attempt, not a claim that the four ACS optimizations have already been repaired.
