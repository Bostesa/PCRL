> Saved verbatim by the coordinator (2026-10-25, after 03:09:53Z by `date -u`): investigator A's first hand-back message (Tasks 1
> and 2), unchanged below.

## Investigator A hand-back (DRF1): Tasks 1 and 2 complete

Both tasks are done: `drf/finite_law.py` passes its selftest (14 of 14 checks), and `drf/tiny.py` runs and writes TINY_EXACT.json. I found no implementation defect in `drf/solver.py`, but the solver often stops at local optima, so a shortfall is not evidence that the objective failed. Started 02:38:33Z, last `date -u` 03:00:16Z. Synthetic data only; nothing under `.private/` was read, no label or real record was touched, `solver.py` is unchanged, and I ran no state-changing git command.

### Files (`shasum -a 256`)
| File | SHA-256 |
|---|---|
| `drf/finite_law.py` | 946c46c654668dfdddbd33b4e308fbb2fa40d93ddef8c072bc5f17b49bf0aba2 |
| `drf/tiny.py` | 04eda9838eec112d3ab683c1bc997f0b3edec6a75dab05e42f7b68a51c6b5352 |
| `results/pcrl_direct_recovery_frontier_v1/roles/investigator_a/DERIVATION.md` | fe7316e65abf2984c2987a2fefab79c94b4da13800c9d44f1ea670031da3173e |
| `results/pcrl_direct_recovery_frontier_v1/roles/investigator_a/TINY_EXACT.json` | fe568b05b87e6f604c63844b55c43111c4f2040f4381879e423c229d10c2e5ba |
| `drf/solver.py` (the version tested; also recorded inside TINY_EXACT.json) | 6eacc900f2541b6f02d328490b2a2212faa6d53c0e0dc52f3fbf6b5b521faa9f |

TINY_EXACT.json contains no timestamps; two runs produced the same hash.

### Task 1: `drf/finite_law.py`
**What it provides:**
- `merge_wires(wires, w1, w0=None, return_inverse=False)` returns `(unique, m1, m0[, inverse])`.
  - Identical wires merge on exact float64 equality; -0.0 and +0.0 count as equal, and non-finite wires are refused.
  - A tuple of arrays is a linked pair, merged only when both components match.
  - `w0` defaults to `1 - w1`.
- Float path: `auc_star(m1, m0)` (sorted, O(M log M)), `auc_star_people(ids_or_vectors, w1, w0=None)`, `pair_ids`, `masses_from_ids`, `lr_scores`, `auc_star_closed_form` (blocked O(M²) check), `auc_of_score` (ties count 1/2).
- `plugin_mi(m1, m0, clip=None)` is the PRC formula sum_w P(w) KL(r_w ‖ r̄) in nats. `clip=1e-9` reproduces the clipping used by PRC and the solver.
- Exact path in Fractions: `auc_star_exact`, `auc_star_closed_form_exact`, `auc_of_score_exact`, `weak_orders`, and `best_auc_by_enumeration` (all ordered set partitions, k ≤ 7, i.e. 47,293 orders, in exact integer arithmetic).

**Selftest** (`python3 -m drf.finite_law selftest`): exit 0, 14/14 pass, 0.56 CPU-s.
- The weak-order generator gives exactly the Fubini counts with no duplicates.
- On 70 random laws with ties, pure wires and unequal class totals, the closed form, the sorted method and the enumeration agree exactly; float error is at most 1.1e-16.
- Pure-wire cases pass: a separable law gives exactly 1, and one pure wire of mass ε gives 1/2 + ε/2.
- A one-wire law gives 1/2, and zero-mass wires are ignored.
- AUC* is unchanged by relabelling or permuting wires, by scaling m1 and m0 separately, and by swapping the classes.
- Merging identical vectors works (including -0.0 = +0.0), and 150 random coarsenings never raised AUC* (data processing).
- AUC*_pair ≥ max of the two views. A pair that is a function of one view equals that view exactly; this is defined on the merged income wire, not the internal cell.
- Results match `sklearn.metrics.roc_auc_score` to within 2.2e-16, both for the posterior score and for arbitrary scores with ties.
- AUC* beats every fixed score exactly (480 scores tested).
- `plugin_mi` matches an independent joint-table formula; the PRC clip changes it by at most 2e-8.
- Both counterexamples below hold.
- On an estimated (pi-weighted) law with 1,500 wires, the sorted method and the closed form agree to 5.6e-16.
- Floats agree with their exact binary values, and invalid inputs are refused.

**Derivation (short):**
- Write Δ(a,b) = P1(a)P0(b) − P1(b)P0(a). AUC(f) = ½ Σ_a P1(a)P0(a) + Σ over unordered pairs a<b of c_f(a,b).
- Each pair term is at most max(P1(a)P0(b), P1(b)P0(a)) = their mean + |Δ|/2. Summing gives AUC* = ½ + ¼ Σ_{a≠b} |Δ(a,b)|.
- Equality holds exactly when f orders every pair with Δ ≠ 0 the way the likelihood ratio m1/m0 does. Pairs with equal ratios (Δ = 0) contribute the same either way, so tie handling does not matter.
- Pure wires are exact: m0 = 0 has ratio +∞, m1 = 0 has ratio 0.
- Sorted form: AUC* = Σ_g P1_g (P0 below g + P0_g/2).

**Scope** (in the docstring and DERIVATION.md): these values are exact for the specified finite law only and are not a population privacy guarantee for fitted census models.

Counterexamples, verified exactly in the selftest. Law A: m1 = (3,2), m0 = (2,3). Law B: m1 = (1,9), m0 = (0,10). Same prior.

| Measure | Law A | Law B |
|---|---|---|
| Plug-in MI (nats) | 0.0201 | 0.0360 |
| Cov(S, P(S=1\|W)) | 1/100 | 1/76 |
| Optimal Brier critic loss | 6/25 | 9/38 (lower) |
| Optimal log-loss critic loss | 0.673 | 0.657 (lower) |
| AUC* | **3/5** | **11/20** |

Law A has lower MI, covariance and critic leakage but the higher AUC*. For linear separability: an XOR law has AUC exactly 1/2 under every linear score yet AUC* = 1.

### Task 2: `drf/tiny.py` (`python3 -m drf.tiny run`, about 23 CPU-s)
**Method:**
- 7 named instances: six random ones (40–300 people, I = 2–3 income cells, J = 3–6 occupation cells, 2 income atoms, 2–3 occupation atoms, both pi-weighted and label laws) plus one built with tied tables.
- A random scan of 120 more instances at three budget levels.
- Every assignment is enumerated (up to 5,832). Tables, all 8 spends, and every AUC and MI are recomputed per person through `finite_law`, both on atom ids and on merged emitted vectors.
- Budget levels: unconstrained; per-metric 60% and 25% quantiles of the positive spends ("q60", "q25"); zero.
- In the tables below, "default" means `solve` with seed 0 and no restarts; "restarts" means seed 0 with 3 restarts.

**Solver correctness:**
- The solver's J_auc, J_mi and spends match the independent recomputation to within 8.9e-16 on every assignment.
- Every solver answer is feasible, and its reported J matches the recomputed J.
- Every answer is a true single-move local optimum: no feasible improving single move remains. **No implementation defect found.**

**Optimizer shortfalls** (the solver stays above the exhaustive optimum of its own objective), across the 348 scan cases where improvement was possible:

| | AUC objective | MI objective |
|---|---|---|
| Default call | 35 | 36 |
| Any of 4 runs | 46 | 47 |
| Restarts | 11 | 11 |
| Best of 4 runs | 9 | 10 |
| Default shortfalls closed by one joint two-cell move | 34 of 35 | 36 of 36 |
| Median relative gap (default) | 10.5% | 4.8% |

- One case had a relative gap of 1.0: the default call made zero moves although a feasible improvement existed.
- In 27 of the 35 AUC default shortfalls, the optimum was reachable from the identity by improving single moves. So the greedy best-move order is usually what goes wrong, not the neighbourhood itself.

**Named instance A** (240 people, I=3, J=6), at q25: gap 0.0211 in J_auc, and 0.0013 in J_mi, on every run including restarts. One two-cell move closes it exactly.

**Minimal reproducers.** All rebuilt and confirmed by calling `solver.solve` / `solver.evaluate` directly; full data is in the JSON under `random_scan.smallest_*`.

1. **Stuck at two balanced atoms.** `tiny.random_instance(seed=1101, N=43, I=3, MI=2, J=4, MO=2, law='sex')`, unconstrained budgets (`{m: 1e9 for m in solver.METRICS}`), AUC objective. The solver converges at A_inc [1,1,1], A_occ [0,0,1,1] with J_auc 0.002222. Merging everything gives 0.0, but either single occupation move raises J to about 0.22. Seeds 0–5 without restarts all stick; 10 restarts fix it.
2. **No move although improvable.** Seed 1114, N=118, q25, AUC. The default call returns the identity (J 0.17565), but a feasible optimum (A_inc [0,1,0], A_occ [0,1,2,2]) reaches 0.15867. A "no move" result therefore does not prove the identity is optimal.
3. **Default call, MI objective.** Seed 1117, N=42, q60. Other seeds or 3 restarts reach the optimum.

**Objective result** (the exhaustive optimum does not improve on the identity): for the six random named instances this happens only at zero budgets. It also happens in 12 of 360 scan cases (instance × budget level, per objective).

**Objective misalignment.** Under the same constraints, the MI-optimal assignment can carry a higher J_auc than the AUC-optimal one:
- instance A at q25: 0.2442 vs 0.2157;
- instance G at q25: excess 0.0430;
- in the scan, 12 of 360 cases, maximum excess 0.0305.

**Tied-tables caveat.** In instance E (24 people), two atoms emit bit-identical vectors.
- The solver scores atom ids, so it measures leakage that no reader can see: at the identity, J_auc is 0.2917 on atoms vs 0.1667 on the observable wires.
- Its default answer is already observably optimal (observable J 0), but it reports 0.1667.
- With continuous teacher outputs this is unlikely in practice, but the final release must be re-scored on merged emitted vectors.

**Suggestions for `drf/solver.py`:**
- add a joint two-cell (pair or swap) move, at least as a final polishing pass;
- run several seeds plus restarts and keep the best;
- never read "no move" as "the identity is optimal";
- check that emitted vectors are distinct, or re-score with `finite_law`.

### CPU
About 130 CPU-seconds in total: five full tiny runs of about 23 s each, plus about 15 s for selftests, smoke tests and reproducer checks. All single-threaded.
